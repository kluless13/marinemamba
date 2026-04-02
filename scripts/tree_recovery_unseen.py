"""
Tree Recovery on UNSEEN species — tests if evolutionary distance generalises.

Uses the already-trained PhyloMamba model but evaluates ONLY on species from
held-out genera (never seen during training). If Pearson r stays high,
the model genuinely learned evolutionary relationships, not memorised them.

Loads the saved checkpoint, extracts embeddings for unseen species,
computes pairwise cosine distances, compares with Fish Tree of Life distances.

Usage:
    python3 scripts/tree_recovery_unseen.py --checkpoint results/phylo_fish_best.pt --embed-dim 64
    python3 scripts/tree_recovery_unseen.py --checkpoint results/phylo_fish_best.pt --embed-dim 128
    python3 scripts/tree_recovery_unseen.py --checkpoint results/phylo_fish_best.pt --embed-dim 384
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="results/phylo_fish_best.pt")
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"TREE RECOVERY — UNSEEN SPECIES (dim={args.embed_dim})")
    print("Tests if evolutionary distance generalises to species never seen during training")
    print("=" * 60)

    # Load tree
    print("\nLoading Fish Tree of Life...")
    tree = dendropy.Tree.get(path=TREE_PATH, schema="newick")
    pdm = tree.phylogenetic_distance_matrix()
    tree_species = {t.label.replace("_", " "): t for t in tree.taxon_namespace}

    # Load unseen data (held-out genera)
    unseen_df = pd.read_csv(os.path.join(args.data_dir, "unseen.csv"))
    unseen_fish = unseen_df[unseen_df["species_name"].isin(tree_species.keys())].copy()
    unseen_species = sorted(unseen_fish["species_name"].unique())
    print(f"  Unseen fish species (in tree): {len(unseen_species)}")
    print(f"  Unseen fish sequences: {len(unseen_fish)}")

    if len(unseen_species) < 20:
        print("  Too few unseen species in tree for meaningful test.")
        return

    # Load trained data for comparison
    train_df = pd.read_csv(os.path.join(args.data_dir, "supervised_train.csv"))
    train_fish = train_df[train_df["species_name"].isin(tree_species.keys())].copy()
    train_species = sorted(train_fish["species_name"].unique())
    print(f"  Train fish species (in tree): {len(train_species)}")

    # Build model
    print(f"\nLoading model (dim={args.embed_dim})...")
    from utils.barcode_mamba import BarcodeMamba

    backbone = BarcodeMamba(
        d_model=384, n_layer=2, d_inner=384*4, vocab_size=8,
        resid_dropout=0.0, embed_dropout=0.1, residual_in_fp32=True,
        pad_vocab_size_multiple=8, mamba_ver="mamba2",
        n_classes=8, use_head="pretrain",
        layer={"d_model": 384, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
    )

    class PhyloMamba(nn.Module):
        def __init__(self, backbone, d_model=384, embed_dim=128):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, embed_dim))
        def forward(self, x):
            h = self.backbone.get_hidden_states(x).mean(dim=1)
            return self.proj(h)
        def get_embeddings(self, x):
            return F.normalize(self.forward(x), dim=1)

    model = PhyloMamba(backbone, d_model=384, embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    model.to(device)
    print("  Model loaded.")

    # Extract per-species embeddings for UNSEEN species
    print(f"\nExtracting embeddings for {len(unseen_species)} UNSEEN species...")
    unseen_embs = {}
    for sp in tqdm(unseen_species, desc="  Unseen"):
        seqs = unseen_fish[unseen_fish["species_name"] == sp]["nucleotides"].tolist()[:10]
        if not seqs:
            continue
        dl = DataLoader(SeqDataset(seqs), batch_size=64, shuffle=False)
        embs = []
        with torch.no_grad():
            for x in dl:
                embs.append(model.get_embeddings(x.to(device)).cpu().numpy())
        unseen_embs[sp] = np.vstack(embs).mean(axis=0)

    # Also extract for a sample of TRAIN species (for cross-comparison)
    print(f"  Extracting embeddings for 500 TRAIN species (for comparison)...")
    np.random.seed(42)
    sample_train = np.random.choice(train_species, size=min(500, len(train_species)), replace=False)
    train_embs = {}
    for sp in tqdm(sample_train, desc="  Train sample"):
        seqs = train_fish[train_fish["species_name"] == sp]["nucleotides"].tolist()[:10]
        if not seqs:
            continue
        dl = DataLoader(SeqDataset(seqs), batch_size=64, shuffle=False)
        embs = []
        with torch.no_grad():
            for x in dl:
                embs.append(model.get_embeddings(x.to(device)).cpu().numpy())
        train_embs[sp] = np.vstack(embs).mean(axis=0)

    # Test 1: Unseen-Unseen pairs (pure generalisation)
    print(f"\n=== TEST 1: UNSEEN-UNSEEN pairs (pure generalisation) ===")
    unseen_list = [sp for sp in unseen_species if sp in unseen_embs]
    pairs = [(i, j) for i in range(len(unseen_list)) for j in range(i+1, len(unseen_list))]
    np.random.seed(42)
    if len(pairs) > 30000:
        pairs = [pairs[i] for i in np.random.choice(len(pairs), 30000, replace=False)]

    emb_dists, tree_dists = [], []
    for i, j in pairs:
        sp_i, sp_j = unseen_list[i], unseen_list[j]
        ei, ej = unseen_embs[sp_i], unseen_embs[sp_j]
        cos = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej) + 1e-8)
        try:
            td = pdm(tree_species[sp_i], tree_species[sp_j])
            emb_dists.append(1 - cos)
            tree_dists.append(td)
        except:
            pass

    if len(emb_dists) > 10:
        emb_dists = np.array(emb_dists)
        tree_dists = np.array(tree_dists)
        pr, pp = pearsonr(emb_dists, tree_dists)
        sr, sp_val = spearmanr(emb_dists, tree_dists)
        print(f"  Pairs: {len(emb_dists)}")
        print(f"  Pearson r:  {pr:.4f} (p={pp:.2e})")
        print(f"  Spearman r: {sr:.4f} (p={sp_val:.2e})")
        unseen_unseen = {"pearson_r": float(pr), "spearman_r": float(sr), "n_pairs": len(emb_dists)}
    else:
        print("  Too few valid pairs")
        unseen_unseen = {"pearson_r": 0, "n_pairs": 0}

    # Test 2: Unseen-Train pairs (cross generalisation)
    print(f"\n=== TEST 2: UNSEEN-TRAIN pairs (cross generalisation) ===")
    train_list = [sp for sp in sample_train if sp in train_embs]
    cross_pairs = []
    for i in range(len(unseen_list)):
        for j in range(len(train_list)):
            cross_pairs.append((i, j))
    np.random.seed(42)
    if len(cross_pairs) > 30000:
        cross_pairs = [cross_pairs[i] for i in np.random.choice(len(cross_pairs), 30000, replace=False)]

    emb_dists2, tree_dists2 = [], []
    for i, j in cross_pairs:
        sp_i = unseen_list[i]
        sp_j = train_list[j]
        ei, ej = unseen_embs[sp_i], train_embs[sp_j]
        cos = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej) + 1e-8)
        try:
            td = pdm(tree_species[sp_i], tree_species[sp_j])
            emb_dists2.append(1 - cos)
            tree_dists2.append(td)
        except:
            pass

    if len(emb_dists2) > 10:
        emb_dists2 = np.array(emb_dists2)
        tree_dists2 = np.array(tree_dists2)
        pr2, pp2 = pearsonr(emb_dists2, tree_dists2)
        sr2, sp2 = spearmanr(emb_dists2, tree_dists2)
        print(f"  Pairs: {len(emb_dists2)}")
        print(f"  Pearson r:  {pr2:.4f} (p={pp2:.2e})")
        print(f"  Spearman r: {sr2:.4f} (p={sp2:.2e})")
        unseen_train = {"pearson_r": float(pr2), "spearman_r": float(sr2), "n_pairs": len(emb_dists2)}
    else:
        print("  Too few valid pairs")
        unseen_train = {"pearson_r": 0, "n_pairs": 0}

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (dim={args.embed_dim})")
    print(f"{'=' * 60}")
    print(f"  Train-Train (from script 12):      r = 0.978/0.969/0.946 (reported earlier)")
    print(f"  Unseen-Unseen (generalisation):     r = {unseen_unseen.get('pearson_r', 0):.4f}")
    print(f"  Unseen-Train (cross generalisation): r = {unseen_train.get('pearson_r', 0):.4f}")

    if unseen_unseen.get("pearson_r", 0) > 0.7:
        print(f"\n  *** STRONG GENERALISATION — model learned evolutionary structure, not memorised! ***")
    elif unseen_unseen.get("pearson_r", 0) > 0.4:
        print(f"\n  ** Moderate generalisation — some evolutionary understanding transfers **")
    else:
        print(f"\n  Weak generalisation — model mostly memorised training species distances")

    # Save
    results = {
        "experiment": "tree_recovery_unseen",
        "embed_dim": args.embed_dim,
        "n_unseen_species": len(unseen_list),
        "n_train_sample": len(train_list),
        "unseen_unseen": unseen_unseen,
        "unseen_train": unseen_train,
    }
    output_path = os.path.join(args.output_dir, f"tree_recovery_unseen_dim{args.embed_dim}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
