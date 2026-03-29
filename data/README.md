# Data

Raw and processed data files are not tracked in git (too large).

## To reproduce

Run notebooks in order:
1. `notebooks/01_data_acquisition.ipynb` — downloads to `data/raw/`
2. `notebooks/02_data_cleaning.ipynb` — outputs to `data/processed/`

## Expected structure after running notebooks

```
data/
├── raw/
│   ├── bold_aus_actinopterygii_coi.tsv
│   ├── bold_<family>_coi.tsv (multiple)
│   ├── genbank_fish_coi.fasta
│   └── merged_barcodes.csv
├── processed/
│   ├── pre_training.csv
│   ├── supervised_train.csv
│   ├── supervised_val.csv
│   ├── supervised_test.csv
│   ├── unseen.csv
│   ├── dataset_stats.json
│   └── worms_cache.json
└── README.md
```

## Data format (BarcodeMamba-compatible CSV)

```csv
nucleotides,species_name,genus_name,family_name,order_name,processid
NNNNATCGGATCN...,Amphiprion ocellaris,Amphiprion,Pomacentridae,Perciformes,PROCESS001
```

- Sequences: 660bp, left-padded with N
- Tokenization: character-level (A=3, C=4, G=5, T=6, N=7) or 6-mer
