"""
Tree Recovery Test: Does the curriculum model's embedding space match the Tree of Life?

Extracts embeddings for species that exist in both our dataset and the Fish Tree of Life,
computes pairwise cosine distances in embedding space, compares with pairwise evolutionary
distances from the tree, and reports Pearson correlation.

If r > 0.7: the model learned evolutionary relationships from classification alone.
If r > 0.9: extraordinary — the model recovered the tree of life from 660bp barcodes.

Usage:
    python3 scripts/tree_recovery_test.py --data-dir data/processed --checkpoint results/multihead_best.pt
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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from pathlib import Path
import dendropy

sys.path.insert(0, "BarcodeMamba")
if not os.path.exists("BarcodeMamba"):
    os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")

CHAR_VOCAB = {"[MASK]": 0, "[SEP]": 1, "[UNK]": 2, "A": 3, "C": 4, "G": 5, "T": 6, "N": 7}
MAX_SEQ_LEN = 660
TREE_PATH = "data/phylo/actinopt_12k_treePL.tre"


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


def build_model(n_orders, n_families, n_genera, n_species):
    from utils.barcode_mamba import BarcodeMamba

    backbone = BarcodeMamba(
        d_model=384, n_layer=2, d_inner=384 * 4,
        vocab_size=8, resid_dropout=0.0, embed_dropout=0.1,
        residual_in_fp32=True, pad_vocab_size_multiple=8,
        mamba_ver="mamba2", n_classes=8, use_head="pretrain",
        layer={"d_model": 384, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
    )

    class MultiHeadMamba(nn.Module):
        def __init__(self, backbone, d_model=384):
            super().__init__()
            self.backbone = backbone
            self.shared_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1))
            self.order_head = nn.Linear(d_model, n_orders)
            self.family_head = nn.Linear(d_model, n_families)
            self.genus_head = nn.Linear(d_model, n_genera)
            self.species_head = nn.Linear(d_model, n_species)

        def get_features(self, x):
            h = self.backbone.get_hidden_states(x)
            return self.shared_proj(h.mean(dim=1))

    return MultiHeadMamba(backbone)


def extract_species_embeddings(model, df, species_list, batch_size=128, device="cuda"):
    """Extract one embedding per species (average of all sequences for that species)."""
    model.eval()
    model.to(device)

    species_embeddings = {}
    for species in tqdm(species_list, desc="  Species embeddings"):
        species_seqs = df[df["species_name"] == species]["nucleotides"].tolist()
        if not species_seqs:
            continue

        # Take up to 10 sequences per species to keep it fast
        species_seqs = species_seqs[:10]
        dl = DataLoader(SeqDataset(species_seqs), batch_size=batch_size, shuffle=False)
        embeds = []
        with torch.no_grad():
            for x in dl:
                h = model.get_features(x.to(device))
                embeds.append(h.cpu().numpy())
        species_embeddings[species] = np.vstack(embeds).mean(axis=0)

    return species_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="results/multihead_best.pt")
    parser.add_argument("--tree-path", default=TREE_PATH)
    parser.add_argument("--max-pairs", type=int, default=50000,
                        help="Max species pairs to sample for correlation (full matrix too large)")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TREE RECOVERY TEST")
    print("Does the curriculum model recover the Tree of Life?")
    print("=" * 60)

    # Load tree
    print(f"\nLoading Fish Tree of Life from {args.tree_path}...")
    tree = dendropy.Tree.get(path=args.tree_path, schema="newick")
    tree_species = {t.label.replace("_", " "): t for t in tree.taxon_namespace}
    print(f"  Tree species: {len(tree_species)}")

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "supervised_train.csv"))
    bold_species = set(train_df["species_name"].unique())
    print(f"  BOLD species: {len(bold_species)}")

    # Find overlap
    matched_species = sorted(bold_species & set(tree_species.keys()))
    print(f"  Matched: {len(matched_species)}")

    if len(matched_species) < 100:
        print("  Too few matched species for meaningful correlation. Exiting.")
        return

    # Build model and load checkpoint
    print(f"\nLoading model from {args.checkpoint}...")
    all_df = pd.concat([train_df,
                        pd.read_csv(os.path.join(args.data_dir, "supervised_val.csv")),
                        pd.read_csv(os.path.join(args.data_dir, "supervised_test.csv"))])
    n_orders = all_df["order_name"].dropna().nunique()
    n_families = all_df["family_name"].dropna().nunique()
    n_genera = all_df["genus_name"].dropna().nunique()
    n_species = all_df["species_name"].dropna().nunique()

    model = build_model(n_orders, n_families, n_genera, n_species)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    print("  Model loaded.")

    # Extract per-species embeddings
    print(f"\nExtracting embeddings for {len(matched_species)} species...")
    species_embs = extract_species_embeddings(model, train_df, matched_species, device=device)
    matched_species = [s for s in matched_species if s in species_embs]
    print(f"  Got embeddings for {len(matched_species)} species")

    # Compute pairwise distances
    print(f"\nComputing pairwise distances...")
    n = len(matched_species)

    # Sample pairs if too many
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > args.max_pairs:
        np.random.seed(42)
        pair_indices = np.random.choice(len(all_pairs), size=args.max_pairs, replace=False)
        sampled_pairs = [all_pairs[idx] for idx in pair_indices]
    else:
        sampled_pairs = all_pairs
    print(f"  Evaluating {len(sampled_pairs)} pairs")

    # Compute tree distances (phylogenetic distance matrix)
    print("  Computing tree distances...")
    pdm = tree.phylogenetic_distance_matrix()

    embedding_dists = []
    tree_dists = []
    failed = 0

    for i, j in tqdm(sampled_pairs, desc="  Pairs"):
        sp_i = matched_species[i]
        sp_j = matched_species[j]

        # Embedding cosine distance
        emb_i = species_embs[sp_i]
        emb_j = species_embs[sp_j]
        cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
        cos_dist = 1 - cos_sim

        # Tree distance
        try:
            taxon_i = tree_species[sp_i]
            taxon_j = tree_species[sp_j]
            t_dist = pdm(taxon_i, taxon_j)
            embedding_dists.append(cos_dist)
            tree_dists.append(t_dist)
        except Exception:
            failed += 1

    embedding_dists = np.array(embedding_dists)
    tree_dists = np.array(tree_dists)

    if failed > 0:
        print(f"  {failed} pairs failed (missing from tree)")

    # Correlate
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"  Pairs evaluated: {len(embedding_dists)}")
    print(f"  Tree distance range: {tree_dists.min():.1f} - {tree_dists.max():.1f} MYA")
    print(f"  Embedding distance range: {embedding_dists.min():.4f} - {embedding_dists.max():.4f}")

    pearson_r, pearson_p = pearsonr(embedding_dists, tree_dists)
    spearman_r, spearman_p = spearmanr(embedding_dists, tree_dists)

    print(f"\n  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

    if pearson_r > 0.7:
        print(f"\n  *** STRONG CORRELATION — model learned evolutionary relationships! ***")
    elif pearson_r > 0.4:
        print(f"\n  ** Moderate correlation — model captures broad evolutionary structure **")
    else:
        print(f"\n  Weak correlation — model creates classification clusters, not evolutionary space")

    # Save
    results = {
        "experiment": "tree_recovery",
        "n_matched_species": len(matched_species),
        "n_pairs": len(embedding_dists),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "tree_dist_range": [float(tree_dists.min()), float(tree_dists.max())],
        "embedding_dist_range": [float(embedding_dists.min()), float(embedding_dists.max())],
    }

    output_path = os.path.join(args.output_dir, "tree_recovery_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
