"""Run BLAST and k-NN baselines on the processed fish barcode data."""
import pandas as pd
import numpy as np
import json
import subprocess
import tempfile
import time
from pathlib import Path
from collections import Counter
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_kmer_features(sequences, k=6):
    """Extract k-mer frequency vectors from DNA sequences."""
    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}

    features = np.zeros((len(sequences), len(all_kmers)), dtype=np.float32)

    for i, seq in enumerate(tqdm(sequences, desc=f"Extracting {k}-mers")):
        counts = Counter()
        for j in range(len(seq) - k + 1):
            kmer = seq[j : j + k]
            if kmer in kmer_to_idx:
                counts[kmer] += 1
        total = sum(counts.values())
        if total > 0:
            for kmer, count in counts.items():
                features[i, kmer_to_idx[kmer]] = count / total

    return features


def run_blast(train_df, test_df, label_col="species_name"):
    """BLASTn baseline: build DB from train, query with test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build reference DB with species labels encoded in headers
        db_fasta = f"{tmpdir}/db.fasta"
        with open(db_fasta, "w") as f:
            for i, (idx, row) in enumerate(train_df.iterrows()):
                f.write(f">ref_{i}___{row[label_col].replace(' ', '_')}\n{row['nucleotides']}\n")

        # Write query sequences with sequential IDs
        query_fasta = f"{tmpdir}/query.fasta"
        query_labels = []
        with open(query_fasta, "w") as f:
            for i, (idx, row) in enumerate(test_df.iterrows()):
                f.write(f">query_{i}\n{row['nucleotides']}\n")
                query_labels.append(row[label_col])

        subprocess.run(
            ["makeblastdb", "-in", db_fasta, "-dbtype", "nucl", "-out", f"{tmpdir}/blastdb"],
            capture_output=True, check=True,
        )

        start = time.time()
        result = subprocess.run(
            ["blastn", "-query", query_fasta, "-db", f"{tmpdir}/blastdb",
             "-outfmt", "6 qseqid sseqid pident evalue", "-max_target_seqs", "1",
             "-evalue", "1e-5", "-num_threads", "4"],
            capture_output=True, text=True, check=True,
        )
        elapsed = time.time() - start

        # Parse: extract species from subject ID (ref_N___Genus_species)
        predictions = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            qid = parts[0]  # e.g. "query_0"
            sid = parts[1]  # e.g. "ref_5___Amphiprion_ocellaris"
            label = sid.split("___", 1)[1].replace("_", " ") if "___" in sid else "Unknown"
            if qid not in predictions:
                predictions[qid] = label

        y_true, y_pred, no_hit = [], [], 0
        for i, true_label in enumerate(query_labels):
            qid = f"query_{i}"
            y_true.append(true_label)
            if qid in predictions:
                y_pred.append(predictions[qid])
            else:
                y_pred.append("NO_HIT")
                no_hit += 1

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "no_hit_rate": no_hit / len(test_df),
            "seqs_per_sec": len(test_df) / elapsed,
        }


def main():
    print("=" * 60)
    print("MARINEMAMBA BASELINES")
    print("=" * 60)

    train = pd.read_csv(PROC_DIR / "supervised_train.csv")
    test = pd.read_csv(PROC_DIR / "supervised_test.csv")
    unseen = pd.read_csv(PROC_DIR / "unseen.csv")
    print(f"Train: {len(train)} | Test: {len(test)} | Unseen: {len(unseen)}")

    results = {}

    # BLAST
    print("\n[1/3] Running BLAST baseline...")
    try:
        blast = run_blast(train, test)
        results["blast"] = blast
        print(f"  Species accuracy: {blast['accuracy']:.4f}")
        print(f"  No-hit rate: {blast['no_hit_rate']:.4f}")
        print(f"  Speed: {blast['seqs_per_sec']:.1f} seq/s")
    except FileNotFoundError:
        print("  BLAST not installed. Install with: brew install blast / apt install ncbi-blast+")
        results["blast"] = {"error": "not installed"}

    # k-mer features
    print("\n[2/3] Running k-NN baseline...")
    X_train = extract_kmer_features(train["nucleotides"].tolist())
    X_test = extract_kmer_features(test["nucleotides"].tolist())
    X_unseen = extract_kmer_features(unseen["nucleotides"].tolist())

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train, train["species_name"])

    start = time.time()
    knn_pred = knn.predict(X_test)
    knn_elapsed = time.time() - start

    knn_species_acc = float(accuracy_score(test["species_name"], knn_pred))

    knn_genus = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_genus.fit(X_train, train["genus_name"])
    knn_genus_pred = knn_genus.predict(X_unseen)
    knn_genus_acc = float(accuracy_score(unseen["genus_name"], knn_genus_pred))

    results["knn"] = {
        "species_accuracy_test": knn_species_acc,
        "genus_accuracy_unseen": knn_genus_acc,
        "seqs_per_sec": len(X_test) / knn_elapsed,
    }
    print(f"  Species accuracy (test): {knn_species_acc:.4f}")
    print(f"  Genus accuracy (unseen): {knn_genus_acc:.4f}")

    # Random Forest
    print("\n[3/3] Running Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, train["species_name"])
    rf_pred = rf.predict(X_test)
    rf_acc = float(accuracy_score(test["species_name"], rf_pred))

    results["random_forest"] = {"species_accuracy_test": rf_acc}
    print(f"  Species accuracy (test): {rf_acc:.4f}")

    # Save
    with open(RESULTS_DIR / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {RESULTS_DIR / 'baselines.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
