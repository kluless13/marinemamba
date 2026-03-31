"""
Hierarchical zero-shot evaluation.

For unseen genera, evaluate whether models predict the correct:
  - Genus (hardest)
  - Family (medium)
  - Order (easiest)

This reveals whether models learn taxonomic structure at different levels,
even when exact genus classification fails.

Usage:
    python3 scripts/06_hierarchical_eval.py --data-dir data/processed --results-dir results
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import Counter
from itertools import product
from tqdm import tqdm
import multiprocessing as mp


def kmer_single(args):
    seq, kmer_to_idx, k = args
    counts = np.zeros(len(kmer_to_idx), dtype=np.float32)
    for j in range(len(seq) - k + 1):
        idx = kmer_to_idx.get(seq[j:j + k])
        if idx is not None:
            counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def extract_kmer_features(sequences, k=6):
    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    n_workers = mp.cpu_count() or 4
    args = [(seq, kmer_to_idx, k) for seq in sequences]
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(kmer_single, args, chunksize=500),
                           total=len(args), desc=f"  {k}-mers"))
    return np.stack(results)


def build_taxonomy_map(df):
    """Build genus -> family -> order mapping from dataframe."""
    tax = {}
    for _, row in df.iterrows():
        genus = row.get("genus_name", "")
        family = row.get("family_name", "")
        order = row.get("order_name", "")
        if genus and genus not in tax:
            tax[genus] = {"family": family, "order": order}
    return tax


def hierarchical_eval(train_labels, pred_labels, true_labels, tax_map, level):
    """Evaluate at a given taxonomic level (family or order)."""
    def get_level(genus, lvl):
        entry = tax_map.get(genus, {})
        return entry.get(lvl, "UNKNOWN")

    if level == "genus":
        true = true_labels
        pred = pred_labels
    else:
        true = [get_level(g, level) for g in true_labels]
        pred = [get_level(g, level) for g in pred_labels]

    # Filter out unknowns
    valid = [(t, p) for t, p in zip(true, pred) if t != "UNKNOWN" and t != "" and p != "UNKNOWN" and p != ""]
    if not valid:
        return {"accuracy": 0.0, "n_evaluated": 0}

    true_f, pred_f = zip(*valid)
    acc = accuracy_score(true_f, pred_f)
    return {
        "accuracy": float(acc),
        "n_evaluated": len(valid),
        "n_correct": int(sum(t == p for t, p in valid)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("HIERARCHICAL ZERO-SHOT EVALUATION")
    print("=" * 60)

    train = pd.read_csv(data_dir / "supervised_train.csv")
    unseen = pd.read_csv(data_dir / "unseen.csv")

    print(f"Train: {len(train)} sequences")
    print(f"Unseen: {len(unseen)} sequences ({unseen['genus_name'].nunique()} genera)")

    # Build taxonomy map from both train and unseen
    all_data = pd.concat([train, unseen])
    tax_map = build_taxonomy_map(all_data)

    n_families_train = train["family_name"].nunique()
    n_families_unseen = unseen["family_name"].nunique()
    n_orders_train = train["order_name"].nunique()
    n_orders_unseen = unseen["order_name"].nunique()

    # How many unseen families overlap with training?
    families_in_train = set(train["family_name"].dropna().unique())
    families_in_unseen = set(unseen["family_name"].dropna().unique())
    family_overlap = families_in_train & families_in_unseen
    orders_in_train = set(train["order_name"].dropna().unique())
    orders_in_unseen = set(unseen["order_name"].dropna().unique())
    order_overlap = orders_in_train & orders_in_unseen

    print(f"\nTaxonomic overlap:")
    print(f"  Families in train: {n_families_train} | unseen: {n_families_unseen} | shared: {len(family_overlap)}")
    print(f"  Orders in train: {n_orders_train} | unseen: {n_orders_unseen} | shared: {len(order_overlap)}")

    # k-mer features
    print("\nExtracting k-mer features...")
    X_train = extract_kmer_features(train["nucleotides"].tolist())
    X_unseen = extract_kmer_features(unseen["nucleotides"].tolist())

    results = {}

    # Evaluate k-NN at each taxonomic level
    for level in ["genus", "family", "order"]:
        print(f"\n--- {level.upper()} level ---")

        # Train k-NN on genus labels, then map predictions to family/order
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
        knn.fit(X_train, train["genus_name"].tolist())
        pred_genera = knn.predict(X_unseen)

        true_genera = unseen["genus_name"].tolist()

        eval_result = hierarchical_eval(
            train["genus_name"].tolist(),
            pred_genera.tolist(),
            true_genera,
            tax_map,
            level,
        )
        results[f"knn_{level}"] = eval_result
        print(f"  k-NN accuracy: {eval_result['accuracy']:.4f} ({eval_result.get('n_correct', 0)}/{eval_result['n_evaluated']})")

    # Also evaluate: for each unseen sequence, what's the nearest training genus's family?
    # This tells us if the embedding space is organized by family
    print("\n--- NEAREST-NEIGHBOR FAMILY ANALYSIS ---")
    knn_1 = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_1.fit(X_train, train["genus_name"].tolist())
    nn_genera = knn_1.predict(X_unseen)

    true_families = [tax_map.get(g, {}).get("family", "") for g in unseen["genus_name"]]
    pred_families = [tax_map.get(g, {}).get("family", "") for g in nn_genera]

    valid = [(t, p) for t, p in zip(true_families, pred_families) if t and p]
    if valid:
        t, p = zip(*valid)
        fam_acc = accuracy_score(t, p)
        n_correct = sum(a == b for a, b in valid)
        print(f"  Nearest-neighbor family match: {fam_acc:.4f} ({n_correct}/{len(valid)})")
        results["nn_family_match"] = {"accuracy": float(fam_acc), "n_correct": n_correct, "n_evaluated": len(valid)}

    true_orders = [tax_map.get(g, {}).get("order", "") for g in unseen["genus_name"]]
    pred_orders = [tax_map.get(g, {}).get("order", "") for g in nn_genera]

    valid_o = [(t, p) for t, p in zip(true_orders, pred_orders) if t and p]
    if valid_o:
        t, p = zip(*valid_o)
        ord_acc = accuracy_score(t, p)
        n_correct = sum(a == b for a, b in valid_o)
        print(f"  Nearest-neighbor order match: {ord_acc:.4f} ({n_correct}/{len(valid_o)})")
        results["nn_order_match"] = {"accuracy": float(ord_acc), "n_correct": n_correct, "n_evaluated": len(valid_o)}

    # Summary
    print(f"\n{'=' * 60}")
    print("HIERARCHICAL ZERO-SHOT SUMMARY (k-NN baseline)")
    print(f"{'=' * 60}")
    print(f"  Genus accuracy:  {results['knn_genus']['accuracy']:.4f}")
    print(f"  Family accuracy: {results.get('nn_family_match', {}).get('accuracy', 0):.4f}")
    print(f"  Order accuracy:  {results.get('nn_order_match', {}).get('accuracy', 0):.4f}")
    print(f"{'=' * 60}")

    # Save
    output_path = results_dir / "hierarchical_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
