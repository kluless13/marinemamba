"""
Evo 2 embedding extraction + linear classifier for marine fish barcodes.
Model F in the experiment matrix.

Usage (run in Colab with A100):
    python scripts/05_evo2_embeddings.py --data-dir data/processed --output-dir results
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ── Embedding Extraction ────────────────────────────────────────────────────

def load_evo2(model_name="evo2_7b"):
    """Load Evo 2 model."""
    from evo2 import Evo2
    print(f"Loading Evo 2: {model_name}")
    model = Evo2(model_name)
    print("  Model loaded.")
    return model


def extract_embedding(model, sequence, layer_name="blocks.26", pool="mean"):
    """Extract a single embedding vector from Evo 2."""
    input_ids = torch.tensor(
        model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to("cuda:0")

    _, embeddings = model(
        input_ids,
        return_embeddings=True,
        layer_names=[layer_name],
    )

    emb = embeddings[layer_name][0]  # (seq_len, hidden_dim)

    if pool == "mean":
        vec = emb.mean(dim=0)
    elif pool == "last_token":
        vec = emb[-1, :]
    else:
        raise ValueError(f"Unknown pool: {pool}")

    return vec.cpu().to(torch.float32).numpy()


def extract_all(model, sequences, layer_name="blocks.26", pool="mean", cache_path=None):
    """Extract embeddings for all sequences. Caches to disk."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"  Extracting embeddings (layer={layer_name}, pool={pool})...")
    embeddings = []
    for seq in tqdm(sequences, desc="  Evo 2"):
        emb = extract_embedding(model, seq, layer_name, pool)
        embeddings.append(emb)

    result = np.stack(embeddings)
    if cache_path:
        np.save(cache_path, result)
        print(f"  Cached to {cache_path}: shape {result.shape}")

    return result


# ── Classification ──────────────────────────────────────────────────────────

def train_logistic(X_train, y_train, X_test, y_test):
    """Train logistic regression classifier."""
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    return {"accuracy": float(acc), "balanced_accuracy": float(bacc)}


def train_knn(X_train, y_train, X_test, y_test):
    """Train 1-NN cosine classifier."""
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    return {"accuracy": float(acc), "balanced_accuracy": float(bacc)}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default="evo2_7b")
    parser.add_argument("--layer", default="blocks.26")
    parser.add_argument("--pool", default="mean", choices=["mean", "last_token"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "evo2_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "supervised_train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "supervised_test.csv"))
    unseen_df = pd.read_csv(os.path.join(args.data_dir, "unseen.csv"))

    print("=" * 60)
    print("MODEL F: Evo 2 Embeddings + Linear Classifier")
    print("=" * 60)
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Unseen: {len(unseen_df)}")

    # Load model
    model = load_evo2(args.model_name)

    # Extract embeddings (cached per split)
    print("\n[1/3] Extracting train embeddings...")
    X_train = extract_all(
        model, train_df["nucleotides"].tolist(),
        layer_name=args.layer, pool=args.pool,
        cache_path=os.path.join(cache_dir, "train_embeddings.npy"),
    )

    print("\n[2/3] Extracting test embeddings...")
    X_test = extract_all(
        model, test_df["nucleotides"].tolist(),
        layer_name=args.layer, pool=args.pool,
        cache_path=os.path.join(cache_dir, "test_embeddings.npy"),
    )

    print("\n[3/3] Extracting unseen embeddings...")
    X_unseen = extract_all(
        model, unseen_df["nucleotides"].tolist(),
        layer_name=args.layer, pool=args.pool,
        cache_path=os.path.join(cache_dir, "unseen_embeddings.npy"),
    )

    # Encode labels
    species_enc = LabelEncoder()
    y_train_species = species_enc.fit_transform(train_df["species_name"])
    y_test_species = species_enc.transform(test_df["species_name"])

    genus_enc = LabelEncoder()
    y_train_genus = genus_enc.fit_transform(train_df["genus_name"])
    y_unseen_genus = genus_enc.fit_transform(unseen_df["genus_name"])  # Fresh fit for unseen

    # Train classifiers
    print("\n=== SPECIES CLASSIFICATION (test) ===")
    print("  Logistic Regression...")
    lr_species = train_logistic(X_train, y_train_species, X_test, y_test_species)
    print(f"  Accuracy: {lr_species['accuracy']:.4f}")

    print("  1-NN Cosine...")
    knn_species = train_knn(X_train, y_train_species, X_test, y_test_species)
    print(f"  Accuracy: {knn_species['accuracy']:.4f}")

    print("\n=== GENUS CLASSIFICATION (unseen genera) ===")
    print("  1-NN Cosine...")
    # For unseen genera: train on genus labels, test on held-out genera
    genus_enc_train = LabelEncoder()
    y_train_g = genus_enc_train.fit_transform(train_df["genus_name"])

    knn_genus = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_genus.fit(X_train, y_train_g)
    # Map unseen genus labels through the same encoder where possible
    unseen_preds = knn_genus.predict(X_unseen)
    # Evaluate: unseen genera won't match train labels exactly, so compare strings
    pred_labels = genus_enc_train.inverse_transform(unseen_preds)
    true_labels = unseen_df["genus_name"].tolist()
    genus_acc = accuracy_score(true_labels, pred_labels)
    genus_bacc = balanced_accuracy_score(true_labels, pred_labels)
    print(f"  Genus accuracy (unseen): {genus_acc:.4f}")
    print(f"  Genus balanced accuracy: {genus_bacc:.4f}")

    # Save results
    results = {
        "model": "evo2_7b",
        "layer": args.layer,
        "pool": args.pool,
        "embedding_dim": int(X_train.shape[1]),
        "species_logistic_regression": lr_species,
        "species_knn": knn_species,
        "genus_knn_unseen": {
            "accuracy": float(genus_acc),
            "balanced_accuracy": float(genus_bacc),
        },
    }

    output_path = os.path.join(args.output_dir, "model_f_evo2_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MODEL F RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
