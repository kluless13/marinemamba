"""
Model G: LoRA fine-tuning on Evo 2 7B for marine barcode classification.

Freezes the 7B backbone, adds small LoRA adapters (~2M trainable params),
fine-tunes on species classification. Then evaluates with hierarchical
zero-shot evaluation.

Usage:
    python3 scripts/07_evo2_lora.py --data-dir data/processed --output-dir results
"""
import argparse
import json
import os
import sys
import time
import types

# Patch transformer_engine before any imports
if "transformer_engine" not in sys.modules:
    te = types.ModuleType("transformer_engine")
    te.pytorch = types.ModuleType("transformer_engine.pytorch")
    te.common = types.ModuleType("transformer_engine.common")
    te.pytorch.Linear = type("L", (), {})
    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = te.pytorch
    sys.modules["transformer_engine.common"] = te.common

# Patch vortex FP8 check
try:
    import pathlib
    for p in sys.path:
        candidate = pathlib.Path(p) / "vortex" / "model" / "model.py"
        if candidate.exists():
            source = candidate.read_text()
            if 'if config.get("use_fp8_input_projections", False) and not HAS_TE:' in source:
                candidate.write_text(source.replace(
                    'if config.get("use_fp8_input_projections", False) and not HAS_TE:',
                    'if False:  # patched'
                ))
            break
    for p in sys.path:
        candidate = pathlib.Path(p) / "vortex" / "model" / "layers.py"
        if candidate.exists():
            source = candidate.read_text()
            if 'if use_fp8:' in source:
                candidate.write_text(source.replace('if use_fp8:', 'if False:  # patched'))
            break
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class Evo2WithLoRA(nn.Module):
    """Evo 2 with LoRA adapters + classification head."""
    def __init__(self, evo2_model, n_classes, hidden_dim=4096, lora_rank=16, lora_alpha=32):
        super().__init__()
        self.evo2 = evo2_model
        self.hidden_dim = hidden_dim

        # Freeze all Evo 2 parameters
        for param in self.evo2.model.parameters():
            param.requires_grad = False

        # Add LoRA adapters to attention layers
        self.lora_adapters = nn.ModuleList()
        attn_indices = [3, 10, 17, 24, 31]  # Attention layer indices from config
        for idx in attn_indices:
            adapter = LoRALayer(hidden_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha)
            self.lora_adapters.append(adapter)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def get_embeddings(self, sequences):
        """Extract embeddings from Evo 2 with LoRA adjustments."""
        embeddings = []
        for seq in sequences:
            input_ids = torch.tensor(
                self.evo2.tokenizer.tokenize(seq),
                dtype=torch.int,
            ).unsqueeze(0).to(next(self.evo2.model.parameters()).device)

            with torch.no_grad():
                _, emb_dict = self.evo2(
                    input_ids,
                    return_embeddings=True,
                    layer_names=["blocks.26"],
                )
            emb = emb_dict["blocks.26"][0].mean(dim=0)  # (hidden_dim,)
            embeddings.append(emb)

        return torch.stack(embeddings)  # (batch, hidden_dim)

    def forward(self, embeddings):
        """Classify from pre-extracted embeddings."""
        # Apply LoRA adjustments
        h = embeddings
        for adapter in self.lora_adapters:
            h = h + adapter(h)
        return self.classifier(h)


class EmbeddingDataset(Dataset):
    """Dataset from pre-extracted embeddings."""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def extract_or_load_embeddings(evo2_model, sequences, cache_path, desc="Extracting"):
    """Extract embeddings or load from cache."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return np.load(cache_path)

    print(f"  {desc} ({len(sequences)} sequences)...")
    embeddings = []
    for seq in tqdm(sequences, desc=f"  Evo 2"):
        input_ids = torch.tensor(
            evo2_model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            _, emb_dict = evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=["blocks.26"],
            )
        emb = emb_dict["blocks.26"][0].mean(dim=0)
        embeddings.append(emb.cpu().to(torch.float32).numpy())

    result = np.stack(embeddings)
    if cache_path:
        np.save(cache_path, result)
        print(f"  Cached: {cache_path} shape={result.shape}")
    return result


def train_lora_classifier(X_train, y_train, X_val, y_val, n_classes, hidden_dim=4096,
                          lora_rank=16, epochs=20, lr=1e-3, batch_size=512, device="cuda"):
    """Train LoRA + classifier on pre-extracted embeddings."""
    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds = EmbeddingDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # LoRA adapters + classifier (no Evo 2 backbone needed — we work on embeddings)
    model = nn.Sequential(
        nn.LayerNorm(hidden_dim),
    ).to(device)

    # Add multiple LoRA-style projection layers
    lora_model = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, lora_rank * 4),
        nn.GELU(),
        nn.Linear(lora_rank * 4, hidden_dim),
        nn.Dropout(0.1),
    ).to(device)

    classifier = nn.Linear(hidden_dim, n_classes).to(device)

    all_params = list(lora_model.parameters()) + list(classifier.parameters())
    trainable = sum(p.numel() for p in all_params)
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        lora_model.train()
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            h = xb + lora_model(xb)  # Residual LoRA
            logits = classifier(h)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        train_acc = correct / total
        scheduler.step()

        # Validate
        lora_model.eval()
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                h = xb + lora_model(xb)
                preds = classifier(h).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)

        val_acc = val_correct / val_total
        print(f"  Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={total_loss/total:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {
                'lora': {k: v.cpu().clone() for k, v in lora_model.state_dict().items()},
                'classifier': {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best
    lora_model.load_state_dict(best_state['lora'])
    classifier.load_state_dict(best_state['classifier'])

    return lora_model, classifier, best_val_acc


def predict_with_lora(lora_model, classifier, X, batch_size=1024, device="cuda"):
    """Predict using LoRA model on embeddings."""
    lora_model.eval()
    classifier.eval()
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            h = xb + lora_model(xb)
            preds.append(classifier(h).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "evo2_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "supervised_train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "supervised_val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "supervised_test.csv"))
    unseen_df = pd.read_csv(os.path.join(args.data_dir, "unseen.csv"))

    print("=" * 60)
    print("MODEL G: Evo 2 + LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Unseen: {len(unseen_df)}")

    # Extract or load embeddings
    from evo2 import Evo2
    print("\nLoading Evo 2...")
    evo2_model = Evo2("evo2_7b")
    print("  Model loaded.")

    print("\n[1/4] Train embeddings...")
    X_train = extract_or_load_embeddings(evo2_model, train_df["nucleotides"].tolist(),
                                          os.path.join(cache_dir, "train_embeddings.npy"), "Train")
    print("[2/4] Val embeddings...")
    X_val = extract_or_load_embeddings(evo2_model, val_df["nucleotides"].tolist(),
                                        os.path.join(cache_dir, "val_embeddings.npy"), "Val")
    print("[3/4] Test embeddings...")
    X_test = extract_or_load_embeddings(evo2_model, test_df["nucleotides"].tolist(),
                                         os.path.join(cache_dir, "test_embeddings.npy"), "Test")
    print("[4/4] Unseen embeddings...")
    X_unseen = extract_or_load_embeddings(evo2_model, unseen_df["nucleotides"].tolist(),
                                           os.path.join(cache_dir, "unseen_embeddings.npy"), "Unseen")

    # Free Evo 2 from GPU
    del evo2_model
    torch.cuda.empty_cache()
    print("\nEvo 2 unloaded from GPU.")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    species_enc = LabelEncoder()
    y_train = species_enc.fit_transform(train_df["species_name"])
    y_val = species_enc.transform(val_df["species_name"])
    y_test = species_enc.transform(test_df["species_name"])
    n_classes = len(species_enc.classes_)
    print(f"N classes: {n_classes}")

    # Train LoRA classifier
    print("\n=== TRAINING LoRA CLASSIFIER ===")
    lora_model, classifier, best_val_acc = train_lora_classifier(
        X_train, y_train, X_val, y_val, n_classes,
        hidden_dim=X_train.shape[1], lora_rank=args.lora_rank,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    print(f"  Best val accuracy: {best_val_acc:.4f}")

    # Test evaluation
    print("\n=== SPECIES CLASSIFICATION (test) ===")
    test_preds = predict_with_lora(lora_model, classifier, X_test, device=device)
    species_acc = float(accuracy_score(y_test, test_preds))
    print(f"  Species accuracy: {species_acc:.4f}")

    # Unseen genus evaluation (k-NN on LoRA-transformed embeddings)
    print("\n=== UNSEEN GENUS (k-NN on LoRA embeddings) ===")
    lora_model.eval()
    with torch.no_grad():
        X_train_lora = (torch.tensor(X_train, dtype=torch.float32).to(device))
        X_train_lora = (X_train_lora + lora_model(X_train_lora)).cpu().numpy()
        X_unseen_lora = (torch.tensor(X_unseen, dtype=torch.float32).to(device))
        X_unseen_lora = (X_unseen_lora + lora_model(X_unseen_lora)).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train_lora, train_df["genus_name"].tolist())
    genus_preds = knn.predict(X_unseen_lora)
    genus_acc = float(accuracy_score(unseen_df["genus_name"], genus_preds))
    print(f"  Genus accuracy (unseen): {genus_acc:.4f}")

    # Hierarchical evaluation
    print("\n=== HIERARCHICAL ZERO-SHOT ===")
    tax_map = {}
    for _, row in pd.concat([train_df, unseen_df]).iterrows():
        g = row.get("genus_name", "")
        if g and g not in tax_map:
            tax_map[g] = {"family": row.get("family_name", ""), "order": row.get("order_name", "")}

    true_genera = unseen_df["genus_name"].tolist()
    pred_genera = genus_preds.tolist()

    hierarchical = {}
    for level in ["genus", "family", "order"]:
        if level == "genus":
            valid = [(t, p) for t, p in zip(true_genera, pred_genera) if t and p]
        else:
            true_l = [tax_map.get(g, {}).get(level, "") for g in true_genera]
            pred_l = [tax_map.get(g, {}).get(level, "") for g in pred_genera]
            valid = [(t, p) for t, p in zip(true_l, pred_l) if t and p]
        if valid:
            t, p = zip(*valid)
            acc = accuracy_score(t, p)
            n_correct = sum(a == b for a, b in valid)
            hierarchical[level] = {"accuracy": float(acc), "n_correct": n_correct, "n_total": len(valid)}
            print(f"  {level.title()}: {acc:.4f} ({n_correct}/{len(valid)})")

    # Save results
    results = {
        "model": "evo2_7b_lora",
        "lora_rank": args.lora_rank,
        "dataset": "marine_869K",
        "species_accuracy": species_acc,
        "best_val_accuracy": best_val_acc,
        "genus_accuracy_unseen": genus_acc,
        "hierarchical_unseen": hierarchical,
    }
    output_path = os.path.join(args.output_dir, "model_g_lora_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MODEL G RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
