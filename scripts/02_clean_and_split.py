"""Clean sequences, validate taxonomy, create train/test splits."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 660


def pad_or_truncate(seq):
    """Left-pad with N if shorter, truncate if longer."""
    if len(seq) >= MAX_LEN:
        return seq[:MAX_LEN]
    return "N" * (MAX_LEN - len(seq)) + seq


def main():
    print("=" * 60)
    print("MARINEMAMBA DATA CLEANING & SPLITS")
    print("=" * 60)

    df = pd.read_csv(RAW_DIR / "merged_barcodes.csv")
    initial = len(df)
    print(f"Loaded {initial} records")

    # Uppercase and clean
    df["nucleotides"] = df["nucleotides"].str.upper().str.strip()
    df["nucleotides"] = df["nucleotides"].str.replace(r"[^ACGTN]", "", regex=True)

    # Length filters
    df["seq_len"] = df["nucleotides"].str.len()
    df = df[df["seq_len"] >= 500].copy()
    print(f"After min length (500bp): {len(df)}")

    df = df[df["seq_len"] <= 700].copy()
    print(f"After max length (700bp): {len(df)}")

    # N filter
    df["n_frac"] = df["nucleotides"].str.count("N") / df["seq_len"]
    df = df[df["n_frac"] <= 0.05].copy()
    print(f"After N filter (<=5%): {len(df)}")

    # Require species name (binomial)
    df = df.dropna(subset=["species_name"])
    df = df[df["species_name"].str.contains(" ", na=False)].copy()
    print(f"After requiring binomial names: {len(df)}")

    # Extract genus from species name if missing
    if "genus_name" not in df.columns:
        df["genus_name"] = df["species_name"].str.split().str[0]
    else:
        df["genus_name"] = df["genus_name"].fillna(df["species_name"].str.split().str[0])

    print(f"\nRemoved {initial - len(df)} records ({100*(initial-len(df))/initial:.1f}%)")
    print(f"Remaining: {len(df)} records, {df['species_name'].nunique()} species, {df['genus_name'].nunique()} genera")

    # Pad/truncate to 660bp
    df["nucleotides"] = df["nucleotides"].apply(pad_or_truncate)

    # Identify holdout genera for zero-shot evaluation
    genus_species_count = df.groupby("genus_name")["species_name"].nunique()
    genus_total = df.groupby("genus_name").size()

    # Holdout: genera with 5+ species, not in top 50 most common
    candidates = genus_species_count[(genus_species_count >= 3) & (genus_species_count <= 50)]
    top50 = genus_total.nlargest(50).index
    candidates = candidates[~candidates.index.isin(top50)]

    np.random.seed(42)
    n_holdout = max(20, len(candidates) // 10)
    holdout_genera = np.random.choice(
        candidates.index, size=min(n_holdout, len(candidates)), replace=False
    )
    print(f"\nHolding out {len(holdout_genera)} genera for zero-shot evaluation")

    # Split
    unseen_mask = df["genus_name"].isin(holdout_genera)
    unseen = df[unseen_mask].copy()
    seen = df[~unseen_mask].copy()

    # Pretraining set: all seen data
    pretrain = seen[["nucleotides", "processid"]].copy()

    # Supervised: species with >= 5 samples (need enough for stratified splits)
    species_counts = seen["species_name"].value_counts()
    valid_species = species_counts[species_counts >= 5].index
    supervised = seen[seen["species_name"].isin(valid_species)].copy()

    # 70/10/20 split
    train_val, test = train_test_split(
        supervised, test_size=0.2, random_state=42, stratify=supervised["species_name"]
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=42, stratify=train_val["species_name"]
    )

    # Save
    output_cols = [c for c in ["nucleotides", "species_name", "genus_name", "family_name", "order_name", "processid"] if c in df.columns]

    pretrain.to_csv(PROC_DIR / "pre_training.csv", index=False)
    train[output_cols].to_csv(PROC_DIR / "supervised_train.csv", index=False)
    val[output_cols].to_csv(PROC_DIR / "supervised_val.csv", index=False)
    test[output_cols].to_csv(PROC_DIR / "supervised_test.csv", index=False)
    unseen[output_cols].to_csv(PROC_DIR / "unseen.csv", index=False)

    stats = {
        "total_sequences": len(df),
        "total_species": int(df["species_name"].nunique()),
        "total_genera": int(df["genus_name"].nunique()),
        "pretrain_size": len(pretrain),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "unseen_size": len(unseen),
        "unseen_genera": int(unseen["genus_name"].nunique()),
        "n_classes": int(supervised["species_name"].nunique()),
        "holdout_genera": list(holdout_genera),
    }

    with open(PROC_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("SPLITS")
    print("=" * 60)
    print(f"Pretrain:     {len(pretrain):,}")
    print(f"Train:        {len(train):,}")
    print(f"Val:          {len(val):,}")
    print(f"Test:         {len(test):,}")
    print(f"Unseen:       {len(unseen):,} ({unseen['genus_name'].nunique()} genera)")
    print(f"N classes:    {supervised['species_name'].nunique():,}")
    print(f"\nSaved to: {PROC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
