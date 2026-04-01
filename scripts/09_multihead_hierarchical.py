"""
Experiment: Multi-Head Hierarchical Classification with Curriculum Learning

Hypothesis: If we give the model separate classification heads for each taxonomic
level (order, family, genus, species) and train coarse-to-fine, the backbone will
learn hierarchical features that generalize better to unseen genera at family/order level.

Method:
  1. BarcodeMamba backbone with 4 classification heads
  2. Curriculum learning: train order head first, progressively add family → genus → species
  3. Hierarchical label smoothing: species loss spreads probability to same-family species
  4. Evaluate at all taxonomic levels including unseen genera

Usage:
    python3 scripts/09_multihead_hierarchical.py --data-dir data/processed --output-dir results
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

sys.path.insert(0, "BarcodeMamba")

# ── Tokenizer ────────────────────────────────────────────────────────────────

CHAR_VOCAB = {"[MASK]": 0, "[SEP]": 1, "[UNK]": 2, "A": 3, "C": 4, "G": 5, "T": 6, "N": 7}
MAX_SEQ_LEN = 660


def tokenize(seq):
    tokens = [CHAR_VOCAB.get(ch, CHAR_VOCAB["N"]) for ch in seq.upper()]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    return [CHAR_VOCAB["N"]] * (MAX_SEQ_LEN - len(tokens)) + tokens


# ── Dataset ──────────────────────────────────────────────────────────────────

class HierarchicalDataset(Dataset):
    """Dataset returning sequence + labels at all taxonomic levels."""
    def __init__(self, df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx):
        self.seqs = df["nucleotides"].tolist()
        self.order_labels = [order_to_idx.get(row.get("order_name", ""), 0)
                             for _, row in df.iterrows()]
        self.family_labels = [family_to_idx.get(row.get("family_name", ""), 0)
                              for _, row in df.iterrows()]
        self.genus_labels = [genus_to_idx.get(row.get("genus_name", ""), 0)
                             for _, row in df.iterrows()]
        self.species_labels = [species_to_idx.get(row.get("species_name", ""), 0)
                               for _, row in df.iterrows()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)
        return (x,
                torch.tensor(self.order_labels[idx], dtype=torch.long),
                torch.tensor(self.family_labels[idx], dtype=torch.long),
                torch.tensor(self.genus_labels[idx], dtype=torch.long),
                torch.tensor(self.species_labels[idx], dtype=torch.long))


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadMamba(nn.Module):
    """BarcodeMamba backbone with 4 classification heads."""
    def __init__(self, backbone, d_model=384, n_orders=60, n_families=427,
                 n_genera=14216, n_species=23964):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        # Shared feature projection
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )

        # Separate heads for each taxonomic level
        self.order_head = nn.Linear(d_model, n_orders)
        self.family_head = nn.Linear(d_model, n_families)
        self.genus_head = nn.Linear(d_model, n_genera)
        self.species_head = nn.Linear(d_model, n_species)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total:,} | Trainable: {trainable:,}")

    def get_features(self, x):
        """Extract shared features from backbone."""
        h = self.backbone.get_hidden_states(x)  # (B, seq, d_model)
        h = h.mean(dim=1)  # (B, d_model)
        return self.shared_proj(h)

    def forward(self, x):
        """Return logits for all 4 levels."""
        features = self.get_features(x)
        return {
            "order": self.order_head(features),
            "family": self.family_head(features),
            "genus": self.genus_head(features),
            "species": self.species_head(features),
        }


# ── Curriculum Training ──────────────────────────────────────────────────────

def get_curriculum_weights(epoch, total_epochs):
    """Progressive curriculum: start with order, end with all levels.

    Phase 1 (0-25%):   order=1.0, family=0.0, genus=0.0, species=0.0
    Phase 2 (25-50%):  order=0.5, family=1.0, genus=0.0, species=0.0
    Phase 3 (50-75%):  order=0.3, family=0.5, genus=1.0, species=0.0
    Phase 4 (75-100%): order=0.2, family=0.3, genus=0.5, species=1.0
    """
    progress = epoch / total_epochs

    if progress < 0.25:
        return {"order": 1.0, "family": 0.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.50:
        return {"order": 0.5, "family": 1.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.75:
        return {"order": 0.3, "family": 0.5, "genus": 1.0, "species": 0.0}
    else:
        return {"order": 0.2, "family": 0.3, "genus": 0.5, "species": 1.0}


def train_multihead(model, train_dl, val_dl, epochs=40, lr=8e-4, device="cuda"):
    """Train with curriculum learning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        weights = get_curriculum_weights(epoch, epochs)
        active_levels = [k for k, v in weights.items() if v > 0]

        model.train()
        total_loss, n_batches = 0, 0

        for batch in tqdm(train_dl, desc=f"  Epoch {epoch+1}/{epochs} [{'+'.join(active_levels)}]", leave=False):
            x, y_order, y_family, y_genus, y_species = [b.to(device) for b in batch]
            logits = model(x)

            loss = torch.tensor(0.0, device=device)
            labels = {"order": y_order, "family": y_family, "genus": y_genus, "species": y_species}

            for level, w in weights.items():
                if w > 0:
                    level_loss = F.cross_entropy(logits[level], labels[level])
                    loss = loss + w * level_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate (all levels)
        model.eval()
        val_losses = {"order": 0, "family": 0, "genus": 0, "species": 0}
        val_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
        val_total = 0

        with torch.no_grad():
            for batch in val_dl:
                x, y_order, y_family, y_genus, y_species = [b.to(device) for b in batch]
                logits = model(x)
                labels = {"order": y_order, "family": y_family, "genus": y_genus, "species": y_species}
                for level in ["order", "family", "genus", "species"]:
                    val_losses[level] += F.cross_entropy(logits[level], labels[level]).item() * len(x)
                    val_correct[level] += (logits[level].argmax(1) == labels[level]).sum().item()
                val_total += len(x)

        val_accs = {k: val_correct[k] / val_total for k in val_correct}
        avg_val_loss = sum(val_losses.values()) / (val_total * 4)

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"O={val_accs['order']:.3f} F={val_accs['family']:.3f} "
              f"G={val_accs['genus']:.3f} S={val_accs['species']:.3f} "
              f"[{'+'.join(active_levels)}]")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/multihead_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("results/multihead_best.pt", weights_only=True))
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-ckpt", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    unseen_df = pd.read_csv(data_dir / "unseen.csv")

    print("=" * 60)
    print("MULTI-HEAD HIERARCHICAL + CURRICULUM LEARNING")
    print("=" * 60)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Unseen: {len(unseen_df)}")

    # Build label encodings
    all_df = pd.concat([train_df, val_df, test_df])
    orders = sorted(all_df["order_name"].dropna().unique())
    families = sorted(all_df["family_name"].dropna().unique())
    genera = sorted(all_df["genus_name"].dropna().unique())
    species = sorted(all_df["species_name"].dropna().unique())

    order_to_idx = {o: i for i, o in enumerate(orders)}
    family_to_idx = {f: i for i, f in enumerate(families)}
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    species_to_idx = {s: i for i, s in enumerate(species)}

    print(f"Orders: {len(orders)} | Families: {len(families)} | Genera: {len(genera)} | Species: {len(species)}")

    # Build datasets
    train_ds = HierarchicalDataset(train_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    val_ds = HierarchicalDataset(val_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    print("\nBuilding model...")
    ckpt = args.pretrain_ckpt
    if not ckpt:
        for candidate in [
            "checkpoints/model_d/lightning_logs/version_0/checkpoints/last.ckpt",
            "checkpoints/model_e/lightning_logs/version_0/checkpoints/last.ckpt",
        ]:
            if os.path.exists(candidate):
                ckpt = candidate
                break

    if not os.path.exists("BarcodeMamba"):
        os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")

    from utils.barcode_mamba import BarcodeMamba

    backbone_config = {
        "d_model": 384, "n_layer": 2, "d_inner": 384 * 4,
        "vocab_size": 8, "resid_dropout": 0.0, "embed_dropout": 0.1,
        "residual_in_fp32": True, "pad_vocab_size_multiple": 8,
        "mamba_ver": "mamba2", "n_classes": 8, "use_head": "pretrain",
        "layer": {"d_model": 384, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
    }

    backbone = BarcodeMamba(**backbone_config)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        cleaned = {k.replace("model.", ""): v for k, v in state.items()}
        backbone.load_state_dict(cleaned, strict=False)
        print(f"  Loaded backbone from {ckpt}")

    model = MultiHeadMamba(
        backbone, d_model=384,
        n_orders=len(orders), n_families=len(families),
        n_genera=len(genera), n_species=len(species),
    )

    # Train
    print("\nTraining with curriculum learning...")
    model = train_multihead(model, train_dl, val_dl, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate
    print("\n=== EVALUATION ===")
    model.eval()
    model.to(device)

    # Test set: direct classification at each level
    print("\n--- Test Set (Seen Species) ---")
    test_ds = HierarchicalDataset(test_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="  Testing"):
            x, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
            logits = model(x)
            test_correct["order"] += (logits["order"].argmax(1) == y_o).sum().item()
            test_correct["family"] += (logits["family"].argmax(1) == y_f).sum().item()
            test_correct["genus"] += (logits["genus"].argmax(1) == y_g).sum().item()
            test_correct["species"] += (logits["species"].argmax(1) == y_s).sum().item()
            test_total += len(x)

    test_accs = {k: v / test_total for k, v in test_correct.items()}
    for level, acc in test_accs.items():
        print(f"  {level.title()}: {acc:.4f}")

    # Unseen genera: extract embeddings, kNN
    print("\n--- Unseen Genera (kNN on backbone embeddings) ---")

    class SeqOnlyDataset(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)

    def extract_features(seqs):
        dl = DataLoader(SeqOnlyDataset(seqs), batch_size=128, shuffle=False, num_workers=4)
        embeds = []
        with torch.no_grad():
            for x in dl:
                h = model.get_features(x.to(device))
                embeds.append(h.cpu().numpy())
        return np.vstack(embeds)

    X_train = extract_features(train_df["nucleotides"].tolist())
    X_unseen = extract_features(unseen_df["nucleotides"].tolist())

    # Build genus → family/order map
    genus_tax_map = {}
    for _, row in pd.concat([train_df, unseen_df]).iterrows():
        g = row.get("genus_name", "")
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {
                "family": row.get("family_name", ""),
                "order": row.get("order_name", ""),
            }

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train, train_df["genus_name"].tolist())
    genus_preds = knn.predict(X_unseen).tolist()
    true_genera = unseen_df["genus_name"].tolist()

    hierarchical = {}
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
            hierarchical[level] = {"accuracy": float(acc), "n_correct": n_correct, "n_total": len(valid)}
            print(f"  {level.title()}: {acc:.4f} ({n_correct}/{len(valid)})")

    # Save
    results = {
        "experiment": "multihead_hierarchical_curriculum",
        "epochs": args.epochs,
        "pretrain_ckpt": ckpt or "none",
        "test_accuracies": {k: float(v) for k, v in test_accs.items()},
        "hierarchical_unseen": hierarchical,
    }

    output_path = os.path.join(args.output_dir, "multihead_hierarchical_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MULTI-HEAD HIERARCHICAL RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
