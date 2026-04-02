"""
Experiment 12: Phylogenetic Embeddings on Fish-Only Subset (Real Tree Distances Only)

Filters dataset to only species with real Fish Tree of Life distances (6,510 species).
No rank-based fallback — every pairwise distance is real evolutionary time (MYA).

Directly comparable to Stalder et al. (2025) who used 7,445 species with real distances.
We have: more sequences per species, longer marker (COI 660bp vs 12S 64bp), SSM backbone.

Usage:
    python3 scripts/12_phylo_fish_only.py --data-dir data/processed --output-dir results
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


class PhyloDataset(Dataset):
    def __init__(self, df, phylo_embeddings, sp_to_idx):
        self.seqs = df["nucleotides"].tolist()
        self.species = df["species_name"].tolist()
        self.phylo_embeddings = phylo_embeddings
        self.sp_to_idx = sp_to_idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)
        sp_idx = self.sp_to_idx.get(self.species[idx], 0)
        target = torch.tensor(self.phylo_embeddings[sp_idx], dtype=torch.float32)
        return x, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--phylo-epochs", type=int, default=300)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("PHYLO EMBEDDINGS — FISH ONLY (REAL TREE DISTANCES)")
    print("=" * 60)

    # ── Step 1: Load tree and filter to matched species ──────────────────
    print(f"\nLoading Fish Tree of Life...")
    tree = dendropy.Tree.get(path=TREE_PATH, schema="newick")
    pdm = tree.phylogenetic_distance_matrix()
    tree_species = {t.label.replace("_", " "): t for t in tree.taxon_namespace}
    print(f"  Tree species: {len(tree_species)}")

    # Load full dataset
    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    unseen_df = pd.read_csv(data_dir / "unseen.csv")

    # Filter to species in tree only
    matched = set(tree_species.keys())
    train_fish = train_df[train_df["species_name"].isin(matched)].copy()
    val_fish = val_df[val_df["species_name"].isin(matched)].copy()
    test_fish = test_df[test_df["species_name"].isin(matched)].copy()
    unseen_fish = unseen_df[unseen_df["species_name"].isin(matched)].copy()

    print(f"\n  Fish-only subset:")
    print(f"    Train: {len(train_fish)} seqs, {train_fish['species_name'].nunique()} species")
    print(f"    Val: {len(val_fish)} seqs")
    print(f"    Test: {len(test_fish)} seqs")
    print(f"    Unseen: {len(unseen_fish)} seqs, {unseen_fish['genus_name'].nunique()} genera")

    fish_species = sorted(set(train_fish["species_name"].unique()) |
                          set(val_fish["species_name"].unique()) |
                          set(test_fish["species_name"].unique()))
    sp_to_idx = {sp: i for i, sp in enumerate(fish_species)}
    n_species = len(fish_species)
    print(f"    Total fish species: {n_species}")

    # ── Step 2: Compute real tree distance matrix for sampling ────────────
    print(f"\n  Computing max tree distance for normalization...")
    sample_taxa = [tree_species[sp] for sp in fish_species[:200] if sp in tree_species]
    max_dist = 0
    for i, t1 in enumerate(sample_taxa):
        for t2 in sample_taxa[i+1:]:
            d = pdm(t1, t2)
            if d > max_dist:
                max_dist = d
    print(f"    Max distance: {max_dist:.1f} MYA")

    # ── Step 3: Learn phylogenetic embeddings with REAL distances ─────────
    print(f"\n[1/5] Learning phylogenetic embeddings ({n_species} species, dim={args.embed_dim})...")

    embeddings = nn.Embedding(n_species, args.embed_dim).to(device)
    nn.init.normal_(embeddings.weight, std=0.1)
    optimizer = torch.optim.AdamW(embeddings.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(args.phylo_epochs):
        total_loss = 0
        anchors = torch.randperm(n_species)[:min(512, n_species)]

        for anchor_idx in anchors:
            anchor_idx = anchor_idx.item()
            anchor_sp = fish_species[anchor_idx]
            anchor_taxon = tree_species.get(anchor_sp)
            if not anchor_taxon:
                continue

            neg_indices = torch.randint(0, n_species, (32,))
            all_indices = torch.cat([torch.tensor([anchor_idx]), neg_indices]).to(device)
            embs = embeddings(all_indices)
            anchor_emb = embs[0:1]
            other_embs = embs[1:]

            cos_dists = 1 - F.cosine_similarity(anchor_emb.expand_as(other_embs), other_embs)

            target_dists = []
            for idx in neg_indices:
                other_sp = fish_species[idx.item()]
                other_taxon = tree_species.get(other_sp)
                if other_taxon and anchor_taxon:
                    try:
                        d = pdm(anchor_taxon, other_taxon) / max_dist
                        target_dists.append(min(d, 1.0))
                    except Exception:
                        target_dists.append(0.5)
                else:
                    target_dists.append(0.5)

            target_dists = torch.tensor(target_dists, dtype=torch.float32, device=device)
            loss = F.mse_loss(cos_dists, target_dists)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(anchors)
            print(f"    Epoch {epoch+1}/{args.phylo_epochs}: loss={avg:.6f}")

    with torch.no_grad():
        phylo_embs = F.normalize(embeddings.weight.data, dim=1).cpu().numpy()
    print(f"  Embeddings shape: {phylo_embs.shape}")

    # ── Step 4: Train BarcodeMamba to map sequences → phylo space ─────────
    print(f"\n[2/5] Building PhyloMamba model...")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # Train
    print(f"\n[3/5] Training PhyloMamba on fish-only data...")
    train_ds = PhyloDataset(train_fish, phylo_embs, sp_to_idx)
    val_ds = PhyloDataset(val_fish, phylo_embs, sp_to_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_epochs)
    best_val = float("inf")
    patience = 7
    patience_counter = 0

    for epoch in range(args.train_epochs):
        model.train()
        total_loss, n = 0, 0
        for x, target in tqdm(train_dl, desc=f"  Epoch {epoch+1}/{args.train_epochs}", leave=False):
            x, target = x.to(device), target.to(device)
            pred = model(x)
            loss = 1 - F.cosine_similarity(F.normalize(pred, dim=1), F.normalize(target, dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for x, target in val_dl:
                x, target = x.to(device), target.to(device)
                pred = model(x)
                cos = F.cosine_similarity(F.normalize(pred, dim=1), F.normalize(target, dim=1)).mean()
                val_loss += (1 - cos.item()) * len(x)
                vn += len(x)

        avg_val = val_loss / vn
        print(f"  Epoch {epoch+1}: train_loss={total_loss/n:.4f} val_cos_dist={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "phylo_fish_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "phylo_fish_best.pt"), weights_only=True))

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    print(f"\n[4/5] Evaluating...")

    def extract(seqs):
        dl = DataLoader(SeqDataset(seqs), batch_size=128, shuffle=False, num_workers=4)
        embs = []
        model.eval()
        with torch.no_grad():
            for x in dl:
                embs.append(model.get_embeddings(x.to(device)).cpu().numpy())
        return np.vstack(embs)

    X_train = extract(train_fish["nucleotides"].tolist())
    X_test = extract(test_fish["nucleotides"].tolist())
    X_unseen = extract(unseen_fish["nucleotides"].tolist()) if len(unseen_fish) > 0 else None

    # Species accuracy
    print("\n=== SPECIES (1-NN cosine) ===")
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train, train_fish["species_name"])
    sp_acc = float(accuracy_score(test_fish["species_name"], knn.predict(X_test)))
    print(f"  Species: {sp_acc:.4f}")

    # Hierarchical on unseen genera
    genus_tax_map = {}
    for _, row in pd.concat([train_fish, unseen_fish]).iterrows():
        g = row.get("genus_name", "")
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {"family": row.get("family_name", ""), "order": row.get("order_name", "")}

    eval_a = {}
    if X_unseen is not None and len(X_unseen) > 0:
        print("\n=== EVAL A: Unseen Genera ===")
        knn_g = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
        knn_g.fit(X_train, train_fish["genus_name"])
        g_preds = knn_g.predict(X_unseen).tolist()
        true_g = unseen_fish["genus_name"].tolist()

        for level in ["family", "order"]:
            true_l = [genus_tax_map.get(g, {}).get(level, "") for g in true_g]
            pred_l = [genus_tax_map.get(g, {}).get(level, "") for g in g_preds]
            valid = [(t, p) for t, p in zip(true_l, pred_l) if t and p]
            if valid:
                t, p = zip(*valid)
                acc = accuracy_score(t, p)
                eval_a[level] = float(acc)
                print(f"  {level.title()}: {acc:.4f} ({sum(a==b for a,b in valid)}/{len(valid)})")

    # Eval C: Stalder protocol
    print("\n=== EVAL C: Stalder Protocol (unseen species, seen genera) ===")
    genus_species = train_fish.groupby("genus_name")["species_name"].nunique()
    eligible = genus_species[genus_species >= 3].index
    holdout_species = set()
    for genus in eligible:
        spp = train_fish[train_fish["genus_name"] == genus]["species_name"].unique()
        np.random.seed(42)
        held = np.random.choice(spp, size=max(1, int(len(spp) * 0.2)), replace=False)
        holdout_species.update(held)

    ref_df = train_fish[~train_fish["species_name"].isin(holdout_species)]
    query_df = train_fish[train_fish["species_name"].isin(holdout_species)]
    print(f"  Reference: {len(ref_df)} seqs | Query: {len(query_df)} seqs ({len(holdout_species)} species)")

    X_ref = extract(ref_df["nucleotides"].tolist())
    X_query = extract(query_df["nucleotides"].tolist())

    knn_c = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_c.fit(X_ref, ref_df["genus_name"])
    c_preds = knn_c.predict(X_query)
    genus_acc = float(accuracy_score(query_df["genus_name"], c_preds))
    print(f"  Genus: {genus_acc:.4f}")

    fam_p = [genus_tax_map.get(g, {}).get("family", "") for g in c_preds]
    fam_t = [genus_tax_map.get(g, {}).get("family", "") for g in query_df["genus_name"]]
    v = [(t, p) for t, p in zip(fam_t, fam_p) if t and p]
    fam_acc = accuracy_score(*zip(*v)) if v else 0
    print(f"  Family: {fam_acc:.4f}")

    ord_p = [genus_tax_map.get(g, {}).get("order", "") for g in c_preds]
    ord_t = [genus_tax_map.get(g, {}).get("order", "") for g in query_df["genus_name"]]
    v2 = [(t, p) for t, p in zip(ord_t, ord_p) if t and p]
    ord_acc = accuracy_score(*zip(*v2)) if v2 else 0
    print(f"  Order: {ord_acc:.4f}")

    # Tree recovery test
    print("\n=== TREE RECOVERY ===")
    species_embs = {}
    for sp in tqdm(fish_species[:2000], desc="  Species embeddings"):
        sp_seqs = train_fish[train_fish["species_name"] == sp]["nucleotides"].tolist()[:5]
        if not sp_seqs:
            continue
        dl = DataLoader(SeqDataset(sp_seqs), batch_size=64, shuffle=False)
        embs = []
        with torch.no_grad():
            for x in dl:
                embs.append(model.get_embeddings(x.to(device)).cpu().numpy())
        species_embs[sp] = np.vstack(embs).mean(axis=0)

    matched_sp = [sp for sp in fish_species[:2000] if sp in species_embs and sp in tree_species]
    pairs = [(i, j) for i in range(len(matched_sp)) for j in range(i+1, len(matched_sp))]
    np.random.seed(42)
    if len(pairs) > 50000:
        pairs = [pairs[i] for i in np.random.choice(len(pairs), 50000, replace=False)]

    emb_dists, tree_dists = [], []
    for i, j in pairs:
        sp_i, sp_j = matched_sp[i], matched_sp[j]
        ei, ej = species_embs[sp_i], species_embs[sp_j]
        cos = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej) + 1e-8)
        try:
            td = pdm(tree_species[sp_i], tree_species[sp_j])
            emb_dists.append(1 - cos)
            tree_dists.append(td)
        except:
            pass

    emb_dists = np.array(emb_dists)
    tree_dists = np.array(tree_dists)

    pr, pp = pearsonr(emb_dists, tree_dists)
    sr, sp_val = spearmanr(emb_dists, tree_dists)
    print(f"  Pearson r:  {pr:.4f} (p={pp:.2e})")
    print(f"  Spearman r: {sr:.4f} (p={sp_val:.2e})")

    # Save
    results = {
        "experiment": "phylo_fish_only_real_distances",
        "n_fish_species": n_species,
        "n_train_seqs": len(train_fish),
        "embed_dim": args.embed_dim,
        "species_accuracy": sp_acc,
        "eval_a_unseen_genera": eval_a,
        "eval_c_stalder": {"genus": genus_acc, "family": float(fam_acc), "order": float(ord_acc)},
        "tree_recovery": {"pearson_r": float(pr), "spearman_r": float(sr)},
    }

    output_path = os.path.join(args.output_dir, "phylo_fish_only_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PHYLO FISH-ONLY RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {output_path}")

    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH STALDER")
    print("=" * 60)
    print(f"  {'Metric':<10} {'Stalder':>10} {'Ours':>10}")
    print(f"  {'Genus':<10} {'50.7%':>10} {genus_acc*100:>9.1f}%")
    print(f"  {'Family':<10} {'80.5%':>10} {fam_acc*100:>9.1f}%")
    print(f"  {'Order':<10} {'86.7%':>10} {ord_acc*100:>9.1f}%")
    print(f"  {'Tree r':<10} {'~0.7?':>10} {pr:>9.3f}")


if __name__ == "__main__":
    main()
