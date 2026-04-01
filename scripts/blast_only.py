"""Run BLAST baseline only. Installs BLAST if needed."""
import pandas as pd
import subprocess
import tempfile
import time
import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm

PROC = Path("data/processed")

def main():
    # Install BLAST
    print("Installing BLAST...")
    subprocess.run(["apt-get", "install", "-qq", "-y", "ncbi-blast+"], capture_output=True)

    train = pd.read_csv(PROC / "supervised_train.csv")
    test = pd.read_csv(PROC / "supervised_test.csv")
    print(f"Train: {len(train)} | Test: {len(test)}")

    with tempfile.TemporaryDirectory() as tmp:
        print("Building BLAST database...")
        with open(f"{tmp}/db.fasta", "w") as f:
            for i, (_, row) in enumerate(train.iterrows()):
                f.write(f">ref_{i}___{row['species_name'].replace(' ', '_')}\n{row['nucleotides']}\n")

        subprocess.run(
            ["makeblastdb", "-in", f"{tmp}/db.fasta", "-dbtype", "nucl", "-out", f"{tmp}/blastdb"],
            capture_output=True, check=True,
        )
        print("Database built.")

        n_threads = str(os.cpu_count() or 4)
        predictions = {}
        batch_size = 5000
        n_batches = (len(test) + batch_size - 1) // batch_size

        start = time.time()
        for batch_idx in tqdm(range(n_batches), desc="BLAST"):
            bs = batch_idx * batch_size
            be = min(bs + batch_size, len(test))
            with open(f"{tmp}/q.fasta", "w") as f:
                for i in range(bs, be):
                    f.write(f">query_{i}\n{test.iloc[i]['nucleotides']}\n")

            r = subprocess.run(
                ["blastn", "-query", f"{tmp}/q.fasta", "-db", f"{tmp}/blastdb",
                 "-outfmt", "6 qseqid sseqid pident evalue", "-max_target_seqs", "1",
                 "-evalue", "1e-5", "-num_threads", n_threads],
                capture_output=True, text=True, check=True,
            )

            for line in r.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                qid, sid = parts[0], parts[1]
                label = sid.split("___", 1)[1].replace("_", " ") if "___" in sid else "Unknown"
                if qid not in predictions:
                    predictions[qid] = label

        elapsed = time.time() - start

        y_true, y_pred = [], []
        for i in range(len(test)):
            y_true.append(test.iloc[i]["species_name"])
            y_pred.append(predictions.get(f"query_{i}", "NO_HIT"))

        acc = accuracy_score(y_true, y_pred)
        no_hit = sum(1 for p in y_pred if p == "NO_HIT") / len(y_pred)
        print(f"\nBLAST species accuracy: {acc:.4f}")
        print(f"No-hit rate: {no_hit:.4f}")
        print(f"Speed: {len(test)/elapsed:.1f} seq/s")

        results_path = Path("results/baselines.json")
        if results_path.exists():
            results = json.loads(results_path.read_text())
        else:
            results = {}
        results["blast"] = {
            "accuracy": float(acc),
            "no_hit_rate": float(no_hit),
            "seqs_per_sec": len(test) / elapsed,
        }
        results_path.write_text(json.dumps(results, indent=2))
        print("Saved to results/baselines.json")


if __name__ == "__main__":
    main()
