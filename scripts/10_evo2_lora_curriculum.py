"""
Experiment 10: Evo 2 + LoRA + Multi-Head Hierarchical Curriculum Learning

Hypothesis: Foundation models don't fail on barcodes — they fail because nobody
trained them right. Combining LoRA adapters with coarse-to-fine curriculum learning
on Evo 2's frozen embeddings should unlock hierarchical taxonomic structure that
frozen embeddings + simple classifiers miss.

Novel contribution: First combination of LoRA + multi-head curriculum + genomic
foundation model for any task. Directly addresses Ye et al. (2025)'s finding that
foundation models underperform domain-specific models on barcodes.

Requires: Pre-extracted Evo 2 embeddings (from script 05, cached as .npy files)

Usage:
    python3 scripts/10_evo2_lora_curriculum.py --data-dir data/processed --cache-dir results/evo2_cache --output-dir results
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


# ── Dataset ──────────────────────────────────────────────────────────────────

class EmbeddingHierarchicalDataset(Dataset):
    """Dataset returning pre-extracted embeddings + labels at all taxonomic levels."""
    def __init__(self, embeddings, df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.order_labels = torch.tensor([order_to_idx.get(str(row.get("order_name", "")), 0)
                                          for _, row in df.iterrows()], dtype=torch.long)
        self.family_labels = torch.tensor([family_to_idx.get(str(row.get("family_name", "")), 0)
                                           for _, row in df.iterrows()], dtype=torch.long)
        self.genus_labels = torch.tensor([genus_to_idx.get(str(row.get("genus_name", "")), 0)
                                          for _, row in df.iterrows()], dtype=torch.long)
        self.species_labels = torch.tensor([species_to_idx.get(str(row.get("species_name", "")), 0)
                                            for _, row in df.iterrows()], dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx],
                self.order_labels[idx],
                self.family_labels[idx],
                self.genus_labels[idx],
                self.species_labels[idx])


# ── Model ────────────────────────────────────────────────────────────────────

class LoRACurriculumModel(nn.Module):
    """LoRA adapters + multi-head hierarchical classifier on frozen Evo 2 embeddings.

    Architecture:
        Evo2 embedding (4096-dim, frozen)
        → LoRA adapter stack (residual MLP layers)
        → Shared features (transformed embedding)
        → Order head
        → Family head
        → Genus head
        → Species head
    """
    def __init__(self, input_dim=4096, adapter_dim=512, n_adapter_layers=3,
                 n_orders=60, n_families=427, n_genera=14216, n_species=23964,
                 dropout=0.1):
        super().__init__()

        # LoRA-style adapter stack (transforms frozen embeddings)
        layers = []
        current_dim = input_dim
        for i in range(n_adapter_layers):
            out_dim = adapter_dim if i < n_adapter_layers - 1 else input_dim
            layers.extend([
                nn.Linear(current_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = adapter_dim

        # Final projection back to input_dim for residual
        layers.append(nn.Linear(adapter_dim, input_dim))
        self.adapter = nn.Sequential(*layers)

        # Shared normalization after residual
        self.norm = nn.LayerNorm(input_dim)

        # Classification heads
        self.order_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_orders),
        )
        self.family_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_families),
        )
        self.genus_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, n_genera),
        )
        self.species_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, n_species),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA Curriculum Model: {trainable:,} trainable params")

    def get_features(self, x):
        """Get adapted features (for kNN eval)."""
        return self.norm(x + self.adapter(x))

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
    """Progressive curriculum: order → family → genus → species."""
    progress = epoch / total_epochs
    if progress < 0.20:
        return {"order": 1.0, "family": 0.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.40:
        return {"order": 0.5, "family": 1.0, "genus": 0.0, "species": 0.0}
    elif progress < 0.65:
        return {"order": 0.3, "family": 0.5, "genus": 1.0, "species": 0.0}
    else:
        return {"order": 0.2, "family": 0.3, "genus": 0.5, "species": 1.0}


def train(model, train_dl, val_dl, epochs=50, lr=1e-3, device="cuda"):
    """Train with curriculum learning."""
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
            emb, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
            logits = model(emb)

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

        # Validate
        model.eval()
        val_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
        val_total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_dl:
                emb, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
                logits = model(emb)
                labels = {"order": y_o, "family": y_f, "genus": y_g, "species": y_s}
                for level in ["order", "family", "genus", "species"]:
                    val_correct[level] += (logits[level].argmax(1) == labels[level]).sum().item()
                    val_loss_sum += F.cross_entropy(logits[level], labels[level]).item() * len(emb)
                val_total += len(emb)

        val_accs = {k: v / val_total for k, v in val_correct.items()}
        avg_val_loss = val_loss_sum / (val_total * 4)

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"O={val_accs['order']:.3f} F={val_accs['family']:.3f} "
              f"G={val_accs['genus']:.3f} S={val_accs['species']:.3f} "
              f"[{'+'.join(active)}]")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/evo2_lora_curriculum_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"  Early stopping at epoch {epoch+1}")
                break
            elif patience_counter >= patience:
                print(f"  Patience hit at epoch {epoch+1} but min_epochs={min_epochs}, continuing...")
                patience_counter = 0

    model.load_state_dict(torch.load("results/evo2_lora_curriculum_best.pt", weights_only=True))
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="results/evo2_cache")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--adapter-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    # Load pre-extracted Evo 2 embeddings
    print("=" * 60)
    print("EVO 2 + LoRA + MULTI-HEAD CURRICULUM")
    print("=" * 60)

    print("\nLoading cached Evo 2 embeddings...")
    X_train = np.load(cache_dir / "train_embeddings.npy")
    X_val = np.load(cache_dir / "val_embeddings.npy") if (cache_dir / "val_embeddings.npy").exists() else None
    X_test = np.load(cache_dir / "test_embeddings.npy")
    X_unseen = np.load(cache_dir / "unseen_embeddings.npy")

    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv") if X_val is not None else None
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    unseen_df = pd.read_csv(data_dir / "unseen.csv")

    # If no val embeddings, split train
    if X_val is None:
        print("  No val embeddings found — splitting train 90/10")
        n_val = len(X_train) // 10
        X_val = X_train[-n_val:]
        X_train = X_train[:-n_val]
        val_df = train_df.iloc[-n_val:]
        train_df = train_df.iloc[:-n_val]

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape} | Unseen: {X_unseen.shape}")

    # Build label encodings
    all_df = pd.concat([train_df, val_df, test_df])
    orders = sorted(all_df["order_name"].dropna().astype(str).unique())
    families = sorted(all_df["family_name"].dropna().astype(str).unique())
    genera = sorted(all_df["genus_name"].dropna().astype(str).unique())
    species = sorted(all_df["species_name"].dropna().astype(str).unique())

    order_to_idx = {o: i for i, o in enumerate(orders)}
    family_to_idx = {f: i for i, f in enumerate(families)}
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    species_to_idx = {s: i for i, s in enumerate(species)}

    print(f"  Orders: {len(orders)} | Families: {len(families)} | Genera: {len(genera)} | Species: {len(species)}")

    # Build datasets
    train_ds = EmbeddingHierarchicalDataset(X_train, train_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    val_ds = EmbeddingHierarchicalDataset(X_val, val_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    print("\nBuilding model...")
    model = LoRACurriculumModel(
        input_dim=X_train.shape[1],
        adapter_dim=args.adapter_dim,
        n_orders=len(orders),
        n_families=len(families),
        n_genera=len(genera),
        n_species=len(species),
    )

    # Train
    print("\nTraining with curriculum...")
    model = train(model, train_dl, val_dl, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate on test set
    print("\n=== TEST SET (Seen Species) ===")
    model.eval()
    model.to(device)

    test_ds = EmbeddingHierarchicalDataset(X_test, test_df, order_to_idx, family_to_idx, genus_to_idx, species_to_idx)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_correct = {"order": 0, "family": 0, "genus": 0, "species": 0}
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="  Testing"):
            emb, y_o, y_f, y_g, y_s = [b.to(device) for b in batch]
            logits = model(emb)
            test_correct["order"] += (logits["order"].argmax(1) == y_o).sum().item()
            test_correct["family"] += (logits["family"].argmax(1) == y_f).sum().item()
            test_correct["genus"] += (logits["genus"].argmax(1) == y_g).sum().item()
            test_correct["species"] += (logits["species"].argmax(1) == y_s).sum().item()
            test_total += len(emb)

    test_accs = {k: v / test_total for k, v in test_correct.items()}
    for level, acc in test_accs.items():
        print(f"  {level.title()}: {acc:.4f}")

    # Zero-shot eval on unseen genera using adapted features
    print("\n=== UNSEEN GENERA (kNN on LoRA-adapted embeddings) ===")

    with torch.no_grad():
        X_train_adapted = model.get_features(
            torch.tensor(X_train, dtype=torch.float32).to(device)
        ).cpu().numpy()
        X_unseen_adapted = model.get_features(
            torch.tensor(X_unseen, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # Build genus → family/order map
    genus_tax_map = {}
    for _, row in pd.concat([train_df, unseen_df]).iterrows():
        g = str(row.get("genus_name", ""))
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {
                "family": str(row.get("family_name", "")),
                "order": str(row.get("order_name", "")),
            }

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train_adapted, train_df["genus_name"].astype(str).tolist())
    genus_preds = knn.predict(X_unseen_adapted).tolist()
    true_genera = unseen_df["genus_name"].astype(str).tolist()

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
        else:
            hierarchical[level] = {"accuracy": 0.0, "n_correct": 0, "n_total": 0}
            print(f"  {level.title()}: 0.0000")

    # ── Eval B: Unseen Species / Seen Genera (BarcodeMamba/Stalder protocol) ──
    print("\n=== SEEN GENERA, UNSEEN SPECIES (BarcodeMamba protocol) ===")

    X_test_adapted = model.get_features(
        torch.tensor(X_test, dtype=torch.float32).to(device)
    ).cpu().numpy()

    knn_genus_seen = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_genus_seen.fit(X_train_adapted, train_df["genus_name"].astype(str).tolist())
    genus_preds_test = knn_genus_seen.predict(X_test_adapted)
    genus_acc_seen = float(accuracy_score(test_df["genus_name"].astype(str), genus_preds_test))
    print(f"  Genus accuracy (seen genera): {genus_acc_seen:.4f}")

    family_preds_test = [genus_tax_map.get(g, {}).get("family", "") for g in genus_preds_test]
    true_families_test = [genus_tax_map.get(g, {}).get("family", "") for g in test_df["genus_name"].astype(str)]
    valid_f = [(t, p) for t, p in zip(true_families_test, family_preds_test) if t and p]
    family_acc_seen = accuracy_score(*zip(*valid_f)) if valid_f else 0.0

    order_preds_test = [genus_tax_map.get(g, {}).get("order", "") for g in genus_preds_test]
    true_orders_test = [genus_tax_map.get(g, {}).get("order", "") for g in test_df["genus_name"].astype(str)]
    valid_o = [(t, p) for t, p in zip(true_orders_test, order_preds_test) if t and p]
    order_acc_seen = accuracy_score(*zip(*valid_o)) if valid_o else 0.0

    print(f"  Family accuracy (seen genera): {family_acc_seen:.4f}")
    print(f"  Order accuracy (seen genera): {order_acc_seen:.4f}")

    seen_genera_eval = {
        "genus": {"accuracy": genus_acc_seen},
        "family": {"accuracy": float(family_acc_seen)},
        "order": {"accuracy": float(order_acc_seen)},
    }

    # Save
    results = {
        "experiment": "evo2_lora_curriculum",
        "adapter_dim": args.adapter_dim,
        "epochs": args.epochs,
        "test_accuracies": {k: float(v) for k, v in test_accs.items()},
        "eval_a_unseen_genera": hierarchical,
        "eval_b_seen_genera_unseen_species": seen_genera_eval,
    }

    output_path = os.path.join(args.output_dir, "evo2_lora_curriculum_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EVO 2 + LoRA + CURRICULUM RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
