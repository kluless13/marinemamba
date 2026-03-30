# Data

Raw and processed data files are not tracked in git (too large). Data is fetched live during experiments.

## Source

**BOLD Systems v5 API** (Barcode of Life Data Systems, University of Guelph)
- Endpoint: `https://portal.boldsystems.org/api`
- Query: `tax:class:Teleostei` (all bony fishes)
- 420,773 raw records, 360,757 with COI marker + species-level ID

## To reproduce

### Option A: Vast.ai (recommended)
```bash
bash scripts/vast/02_fetch_data.sh
```

### Option B: Locally
```bash
python scripts/fetch_bold.py          # Download raw TSV from BOLD API
python scripts/01b_process_bold.py    # Convert to merged_barcodes.csv
python scripts/02_clean_and_split.py  # Quality filter + train/test/unseen splits
```

## Expected structure

```
data/
├── raw/
│   ├── bold_teleostei.tsv        # Raw BOLD download (~577 MB)
│   └── merged_barcodes.csv       # Filtered COI sequences with taxonomy
├── processed/
│   ├── pre_training.csv          # 299,430 sequences for NTP pretraining
│   ├── supervised_train.csv      # 194,457 sequences (70%)
│   ├── supervised_val.csv        #  27,780 sequences (10%)
│   ├── supervised_test.csv       #  55,560 sequences (20%)
│   ├── unseen.csv                #  19,399 sequences (173 held-out genera)
│   └── dataset_stats.json        # Summary statistics
└── README.md
```

## Data format (BarcodeMamba-compatible CSV)

```csv
nucleotides,species_name,genus_name,family_name,order_name,processid
NNNNATCGGATCN...,Amphiprion ocellaris,Amphiprion,Pomacentridae,Ovalentaria_incertae_sedis,ABFJ020-06
```

- Sequences: 660bp, left-padded with N if shorter, truncated if longer
- Quality filters: 500-700bp raw length, ≤5% ambiguous bases, binomial species name required
- Tokenization: character-level (A=3, C=4, G=5, T=6, N=7)

## Unseen genera holdout

173 genera are held out entirely for zero-shot evaluation. Selection criteria:
- 3-50 species per genus (enough diversity, not dominant)
- Not in the 50 most common genera (avoids removing too much training data)
- Random seed 42 for reproducibility
