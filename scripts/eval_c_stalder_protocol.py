"""
Eval C: Stalder Protocol — Unseen Species, Seen Genera

Matches Stalder et al. (2025) evaluation exactly:
  - Hold out species that the model never saw during training
  - Their genera ARE in training (other species from same genus)
  - Evaluate: can the model assign the correct genus to a new species?

This runs AFTER script 09 finishes — loads the saved model checkpoint,
creates a species-holdout split, and evaluates.

Stalder's numbers to beat:
  - Genus: 50.7% (DNA only)
  - Family: 80.5% (DNA only)
  - Order: 86.7% (DNA only)

Usage:
    python3 scripts/eval_c_stalder_protocol.py --data-dir data/processed --output-dir results
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, "BarcodeMamba")
if not os.path.exists("BarcodeMamba"):
    os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")

CHAR_VOCAB = {"[MASK]": 0, "[SEP]": 1, "[UNK]": 2, "A": 3, "C": 4, "G": 5, "T": 6, "N": 7}
MAX_SEQ_LEN = 660


def tokenize(seq):
    tokens = [CHAR_VOCAB.get(ch, CHAR_VOCAB["N"]) for ch in seq.upper()]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    return [CHAR_VOCAB["N"]] * (MAX_SEQ_LEN - len(tokens)) + tokens


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)


def build_multihead_model(n_orders, n_families, n_genera, n_species):
    """Rebuild the MultiHeadMamba architecture to load checkpoint."""
    from utils.barcode_mamba import BarcodeMamba

    backbone_config = {
        "d_model": 384, "n_layer": 2, "d_inner": 384 * 4,
        "vocab_size": 8, "resid_dropout": 0.0, "embed_dropout": 0.1,
        "residual_in_fp32": True, "pad_vocab_size_multiple": 8,
        "mamba_ver": "mamba2", "n_classes": 8, "use_head": "pretrain",
        "layer": {"d_model": 384, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
    }
    backbone = BarcodeMamba(**backbone_config)

    # Rebuild MultiHeadMamba (must match script 09's architecture)
    class MultiHeadMamba(nn.Module):
        def __init__(self, backbone, d_model=384, n_orders=60, n_families=427,
                     n_genera=14216, n_species=23964):
            super().__init__()
            self.backbone = backbone
            self.d_model = d_model
            self.shared_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1))
            self.order_head = nn.Linear(d_model, n_orders)
            self.family_head = nn.Linear(d_model, n_families)
            self.genus_head = nn.Linear(d_model, n_genera)
            self.species_head = nn.Linear(d_model, n_species)

        def get_features(self, x):
            h = self.backbone.get_hidden_states(x)
            h = h.mean(dim=1)
            return self.shared_proj(h)

    model = MultiHeadMamba(backbone, d_model=384,
                           n_orders=n_orders, n_families=n_families,
                           n_genera=n_genera, n_species=n_species)
    return model


def extract_features(model, seqs, batch_size=128, device="cuda"):
    dl = DataLoader(SeqDataset(seqs), batch_size=batch_size, shuffle=False, num_workers=4)
    embeds = []
    with torch.no_grad():
        for x in tqdm(dl, desc="  Extracting"):
            h = model.get_features(x.to(device))
            embeds.append(h.cpu().numpy())
    return np.vstack(embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint", default="results/multihead_best.pt")
    parser.add_argument("--holdout-fraction", type=float, default=0.2,
                        help="Fraction of species to hold out per genus")
    parser.add_argument("--min-species-per-genus", type=int, default=3,
                        help="Only hold out from genera with at least this many species")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("EVAL C: STALDER PROTOCOL")
    print("Unseen species, seen genera")
    print("=" * 60)

    # Load all training data
    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    print(f"Original train: {len(train_df)} sequences")

    # Create species-holdout split
    # For each genus with enough species, hold out some species entirely
    print(f"\nCreating species-holdout split (holdout={args.holdout_fraction}, min_species={args.min_species_per_genus})...")

    genus_species = train_df.groupby("genus_name")["species_name"].nunique()
    eligible_genera = genus_species[genus_species >= args.min_species_per_genus].index
    print(f"  Eligible genera (>={args.min_species_per_genus} species): {len(eligible_genera)}")

    # For each eligible genus, hold out some species
    holdout_species = set()
    for genus in eligible_genera:
        genus_spp = train_df[train_df["genus_name"] == genus]["species_name"].unique()
        n_holdout = max(1, int(len(genus_spp) * args.holdout_fraction))
        np.random.seed(42)
        held = np.random.choice(genus_spp, size=n_holdout, replace=False)
        holdout_species.update(held)

    print(f"  Held-out species: {len(holdout_species)}")

    # Split into reference (training) and query (held-out species)
    reference_mask = ~train_df["species_name"].isin(holdout_species)
    query_mask = train_df["species_name"].isin(holdout_species)
    reference_df = train_df[reference_mask].copy()
    query_df = train_df[query_mask].copy()

    print(f"  Reference (seen species): {len(reference_df)} sequences, {reference_df['species_name'].nunique()} species")
    print(f"  Query (unseen species): {len(query_df)} sequences, {query_df['species_name'].nunique()} species")
    print(f"  Genera in reference: {reference_df['genus_name'].nunique()}")
    print(f"  Genera in query: {query_df['genus_name'].nunique()}")

    # Verify: all query genera exist in reference
    ref_genera = set(reference_df["genus_name"].unique())
    query_genera = set(query_df["genus_name"].unique())
    overlap = ref_genera & query_genera
    print(f"  Genera overlap: {len(overlap)}/{len(query_genera)} ({100*len(overlap)/len(query_genera):.1f}%)")

    # Load model checkpoint
    print(f"\nLoading model from {args.checkpoint}...")

    # Get label counts from the original training data
    all_df = pd.concat([train_df, pd.read_csv(data_dir / "supervised_val.csv"),
                        pd.read_csv(data_dir / "supervised_test.csv")])
    n_orders = all_df["order_name"].dropna().nunique()
    n_families = all_df["family_name"].dropna().nunique()
    n_genera = all_df["genus_name"].dropna().nunique()
    n_species = all_df["species_name"].dropna().nunique()

    model = build_multihead_model(n_orders, n_families, n_genera, n_species)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)
    print("  Model loaded.")

    # Extract embeddings
    print("\nExtracting embeddings...")
    X_ref = extract_features(model, reference_df["nucleotides"].tolist(), device=device)
    X_query = extract_features(model, query_df["nucleotides"].tolist(), device=device)

    # Build taxonomy map
    genus_tax_map = {}
    for _, row in train_df.iterrows():
        g = row.get("genus_name", "")
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {
                "family": row.get("family_name", ""),
                "order": row.get("order_name", ""),
            }

    # Evaluate at each level
    print("\n=== EVAL C RESULTS (Stalder Protocol) ===")

    # Species accuracy
    knn_sp = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_sp.fit(X_ref, reference_df["species_name"].tolist())
    species_preds = knn_sp.predict(X_query)
    species_acc = float(accuracy_score(query_df["species_name"], species_preds))
    print(f"  Species: {species_acc:.4f}")

    # Genus accuracy
    knn_g = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_g.fit(X_ref, reference_df["genus_name"].tolist())
    genus_preds = knn_g.predict(X_query)
    genus_acc = float(accuracy_score(query_df["genus_name"], genus_preds))
    print(f"  Genus: {genus_acc:.4f}")

    # Family accuracy
    pred_families = [genus_tax_map.get(g, {}).get("family", "") for g in genus_preds]
    true_families = [genus_tax_map.get(g, {}).get("family", "") for g in query_df["genus_name"]]
    valid_f = [(t, p) for t, p in zip(true_families, pred_families) if t and p]
    family_acc = accuracy_score(*zip(*valid_f)) if valid_f else 0.0
    print(f"  Family: {family_acc:.4f}")

    # Order accuracy
    pred_orders = [genus_tax_map.get(g, {}).get("order", "") for g in genus_preds]
    true_orders = [genus_tax_map.get(g, {}).get("order", "") for g in query_df["genus_name"]]
    valid_o = [(t, p) for t, p in zip(true_orders, pred_orders) if t and p]
    order_acc = accuracy_score(*zip(*valid_o)) if valid_o else 0.0
    print(f"  Order: {order_acc:.4f}")

    print(f"\n  Stalder comparison:")
    print(f"    {'Metric':<10} {'Stalder (DNA only)':>20} {'Ours':>10}")
    print(f"    {'Species':<10} {'0.7%':>20} {species_acc*100:>9.1f}%")
    print(f"    {'Genus':<10} {'50.7%':>20} {genus_acc*100:>9.1f}%")
    print(f"    {'Family':<10} {'80.5%':>20} {family_acc*100:>9.1f}%")
    print(f"    {'Order':<10} {'86.7%':>20} {order_acc*100:>9.1f}%")

    # Save
    results = {
        "experiment": "eval_c_stalder_protocol",
        "holdout_fraction": args.holdout_fraction,
        "min_species_per_genus": args.min_species_per_genus,
        "n_reference_sequences": len(reference_df),
        "n_query_sequences": len(query_df),
        "n_holdout_species": len(holdout_species),
        "results": {
            "species": species_acc,
            "genus": genus_acc,
            "family": float(family_acc),
            "order": float(order_acc),
        },
        "stalder_comparison": {
            "species": 0.007,
            "genus": 0.507,
            "family": 0.805,
            "order": 0.867,
        },
    }

    output_path = os.path.join(args.output_dir, "eval_c_stalder_protocol.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
