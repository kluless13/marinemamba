"""
Experiment 11: Multi-Head Curriculum with 6-mer Tokenization

Same curriculum approach as script 09 but with 6-mer tokenization instead of
character-level. 6-mer captures codon structure — expected to significantly
improve over character-level since k-NN 6-mer already beats character-level
neural models on everything.

Usage:
    python3 scripts/11_curriculum_6mer.py --data-dir data/processed --output-dir results
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from itertools import product

# ── 6-mer Tokenizer ──────────────────────────────────────────────────────────

def build_6mer_vocab():
    """Build 6-mer vocabulary: 4096 possible 6-mers + special tokens."""
    bases = "ACGT"
    all_6mers = ["".join(p) for p in product(bases, repeat=6)]
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i, kmer in enumerate(all_6mers):
        vocab[kmer] = i + 2
    return vocab

KMER_VOCAB = build_6mer_vocab()
KMER_VOCAB_SIZE = len(KMER_VOCAB)  # 4098 (4096 + PAD + UNK)
KMER_K = 6
MAX_KMER_TOKENS = 655  # 660bp - 6 + 1 = 655 possible 6-mers


def tokenize_6mer(seq, max_tokens=MAX_KMER_TOKENS):
    """Tokenize sequence into overlapping 6-mers."""
    seq = seq.upper()
    tokens = []
    for i in range(len(seq) - KMER_K + 1):
        kmer = seq[i:i + KMER_K]
        tokens.append(KMER_VOCAB.get(kmer, KMER_VOCAB["[UNK]"]))

    # Pad or truncate
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    elif len(tokens) < max_tokens:
        tokens = [KMER_VOCAB["[PAD]"]] * (max_tokens - len(tokens)) + tokens

    return tokens


# ── Dataset ──────────────────────────────────────────────────────────────────

class HierarchicalDataset6mer(Dataset):
    def __init__(self, df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx):
        self.seqs = df["nucleotides"].tolist()
        self.order_labels = [order_to_idx.get(str(row.get("order_name", "")), 0)
                             for _, row in df.iterrows()]
        self.family_labels = [family_to_idx.get(str(row.get("family_name", "")), 0)
                              for _, row in df.iterrows()]
        self.genus_labels = [genus_to_idx.get(str(row.get("genus_name", "")), 0)
                             for _, row in df.iterrows()]
        self.species_labels = [species_to_idx.get(str(row.get("species_name", "")), 0)
                               for _, row in df.iterrows()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(tokenize_6mer(self.seqs[idx]), dtype=torch.long)
        return (x,
                torch.tensor(self.order_labels[idx], dtype=torch.long),
                torch.tensor(self.family_labels[idx], dtype=torch.long),
                torch.tensor(self.genus_labels[idx], dtype=torch.long),
                torch.tensor(self.species_labels[idx], dtype=torch.long))


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadMamba6mer(nn.Module):
    """BarcodeMamba with 6-mer vocab + 4 classification heads."""
    def __init__(self, d_model=384, n_orders=60, n_families=427,
                 n_genera=14216, n_species=23964):
        super().__init__()

        if not os.path.exists("BarcodeMamba"):
            os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")
        if "BarcodeMamba" not in sys.path:
            sys.path.insert(0, "BarcodeMamba")

        from utils.barcode_mamba import BarcodeMamba

        self.backbone = BarcodeMamba(
            d_model=d_model, n_layer=2, d_inner=d_model * 4,
            vocab_size=KMER_VOCAB_SIZE, resid_dropout=0.0, embed_dropout=0.1,
            residual_in_fp32=True, pad_vocab_size_multiple=8,
            mamba_ver="mamba2", n_classes=8, use_head="pretrain",
            layer={"d_model": d_model, "d_state": 64, "d_conv": 4,
                   "expand": 2, "headdim": 48},
        )

        self.d_model = d_model
        self.shared_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1))
        self.order_head = nn.Linear(d_model, n_orders)
        self.family_head = nn.Linear(d_model, n_families)
        self.genus_head = nn.Linear(d_model, n_genera)
        self.species_head = nn.Linear(d_model, n_species)

        total = sum(p.numel() for p in self.parameters())
        print(f"  6-mer MultiHead: {total:,} params (vocab={KMER_VOCAB_SIZE})")

    def get_features(self, x):
        h = self.backbone.get_hidden_states(x)
        h = h.mean(dim=1)
        return self.shared_proj(h)

    def forward(self, x):
        features = self.get_features(x)
        return {
            "order": self.order_head(features),
            "family": self.family_head(features),
            "genus": self.genus_head(features),
            "species": self.species_head(features),
        }


# ── Curriculum Training ──────────────────────────────────────────────────────

def get_curriculum_weights(epoch, total_epochs):
    progress = epoch / total_epochs
    if progress < 0.20:
        return {"order": 1.0, "family": 0.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.40:
        return {"order": 0.5, "family": 1.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.65:
        return {"order": 0.3, "family": 0.5, "genus": 1.0, "species": 0.0}
    else:
        return {"order": 0.2, "family": 0.3, "genus": 0.5, "species": 1.0}


def train_multihead(model, train_dl, val_dl, epochs=40, lr=8e-4, device="cuda"):
    min_epochs = int(epochs * 0.85)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        weights = get_curriculum_weights(epoch, epochs)
        active = [k for k, v in weights.items() if v > 0]

        model.train()
        total_loss, n_batches = 0, 0

        for batch in tqdm(train_dl, desc=f"  Epoch {epoch+1}/{epochs} [{'+'.join(active)}]", leave=False):
            x, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
            logits = model(x)
            loss = torch.tensor(0.0, device=device)
            labels = {"order": y_o, "family": y_f, "genus": y_g, "species": y_s}
            for level, w in weights.items():
                if w > 0:
                    loss = loss + w * F.cross_entropy(logits[level], labels[level])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
        val_total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_dl:
                x, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
                logits = model(x)
                labels = {"order": y_o, "family": y_f, "genus": y_g, "species": y_s}
                for level in ["order", "family", "genus", "species"]:
                    val_correct[level] += (logits[level].argmax(1) == labels[level]).sum().item()
                    val_loss_sum += F.cross_entropy(logits[level], labels[level]).item() * len(x)
                val_total += len(x)

        val_accs = {k: v / val_total for k, v in val_correct.items()}
        avg_val_loss = val_loss_sum / (val_total * 4)

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"O={val_accs['order']:.3f} F={val_accs['family']:.3f} "
              f"G={val_accs['genus']:.3f} S={val_accs['species']:.3f} "
              f"[{'+'.join(active)}]")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/curriculum_6mer_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"  Early stopping at epoch {epoch+1}")
                break
            elif patience_counter >= patience:
                print(f"  Patience hit at epoch {epoch+1} but min_epochs={min_epochs}, continuing...")
                patience_counter = 0

    model.load_state_dict(torch.load("results/curriculum_6mer_best.pt", weights_only=True))
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)

    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    unseen_df = pd.read_csv(data_dir / "unseen.csv")

    print("=" * 60)
    print("6-MER CURRICULUM LEARNING")
    print("=" * 60)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Unseen: {len(unseen_df)}")
    print(f"Tokenization: 6-mer (vocab={KMER_VOCAB_SIZE})")

    all_df = pd.concat([train_df, val_df, test_df])
    orders = sorted(all_df["order_name"].dropna().astype(str).unique())
    families = sorted(all_df["family_name"].dropna().astype(str).unique())
    genera = sorted(all_df["genus_name"].dropna().astype(str).unique())
    species = sorted(all_df["species_name"].dropna().astype(str).unique())

    order_to_idx = {o: i for i, o in enumerate(orders)}
    family_to_idx = {f: i for i, f in enumerate(families)}
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    species_to_idx = {s: i for i, s in enumerate(species)}

    print(f"Orders: {len(orders)} | Families: {len(families)} | Genera: {len(genera)} | Species: {len(species)}")

    train_ds = HierarchicalDataset6mer(train_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    val_ds = HierarchicalDataset6mer(val_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("\nBuilding 6-mer model...")
    model = MultiHeadMamba6mer(
        d_model=384,
        n_orders=len(orders), n_families=len(families),
        n_genera=len(genera), n_species=len(species),
    )

    print("\nTraining with curriculum...")
    model = train_multihead(model, train_dl, val_dl, epochs=args.epochs, lr=args.lr, device=device)

    # ── Evaluation ──
    print("\n=== EVALUATION ===")
    model.eval()
    model.to(device)

    # Test set
    print("\n--- Test Set (Seen Species) ---")
    test_ds = HierarchicalDataset6mer(test_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
    test_total = 0
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="  Testing"):
            x, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
            logits = model(x)
            for level, y in [("order", y_o), ("family", y_f), ("genus", y_g), ("species", y_s)]:
                test_correct[level] += (logits[level].argmax(1) == y).sum().item()
            test_total += len(x)

    test_accs = {k: v / test_total for k, v in test_correct.items()}
    for level, acc in test_accs.items():
        print(f"  {level.title()}: {acc:.4f}")

    # Eval A: Unseen genera
    print("\n--- Eval A: Unseen Genera ---")

    class SeqDataset6mer(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return torch.tensor(tokenize_6mer(self.seqs[idx]), dtype=torch.long)

    def extract_features(seqs):
        dl = DataLoader(SeqDataset6mer(seqs), batch_size=128, shuffle=False, num_workers=4)
        embeds = []
        with torch.no_grad():
            for x in dl:
                embeds.append(model.get_features(x.to(device)).cpu().numpy())
        return np.vstack(embeds)

    X_train = extract_features(train_df["nucleotides"].tolist())
    X_unseen = extract_features(unseen_df["nucleotides"].tolist())

    genus_tax_map = {}
    for _, row in pd.concat([train_df, unseen_df]).iterrows():
        g = str(row.get("genus_name", ""))
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {"family": str(row.get("family_name", "")),
                                "order": str(row.get("order_name", ""))}

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train, train_df["genus_name"].astype(str).tolist())
    genus_preds = knn.predict(X_unseen).tolist()
    true_genera = unseen_df["genus_name"].astype(str).tolist()

    eval_a = {}
    for level in ["genus", "family", "order"]:
        if level == "genus":
            valid = [(t, p) for t, p in zip(true_genera, genus_preds) if t and p]
        else:
            true_l = [genus_tax_map.get(g, {}).get(level, "") for g in true_genera]
            pred_l = [genus_tax_map.get(g, {}).get(level, "") for g in genus_preds]
            valid = [(t, p) for t, p in zip(true_l, pred_l) if t and p]
        if valid:
            t, p = zip(*valid)
            acc = accuracy_score(t, p)
            n_correct = sum(a == b for a, b in valid)
            eval_a[level] = {"accuracy": float(acc), "n_correct": n_correct, "n_total": len(valid)}
            print(f"  {level.title()}: {acc:.4f} ({n_correct}/{len(valid)})")

    # Eval B: Seen genera
    print("\n--- Eval B: Seen Genera, Unseen Sequences ---")
    X_test_emb = extract_features(test_df["nucleotides"].tolist())
    knn_b = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_b.fit(X_train, train_df["genus_name"].astype(str).tolist())
    genus_preds_test = knn_b.predict(X_test_emb)
    eval_b_genus = float(accuracy_score(test_df["genus_name"].astype(str), genus_preds_test))
    print(f"  Genus: {eval_b_genus:.4f}")

    # Save
    results = {
        "experiment": "curriculum_6mer",
        "tokenization": "6-mer",
        "vocab_size": KMER_VOCAB_SIZE,
        "epochs": args.epochs,
        "test_accuracies": {k: float(v) for k, v in test_accs.items()},
        "eval_a_unseen_genera": eval_a,
        "eval_b_genus_seen_genera": eval_b_genus,
    }

    output_path = os.path.join(args.output_dir, "curriculum_6mer_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("6-MER CURRICULUM RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
