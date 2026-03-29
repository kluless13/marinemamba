# MarineMamba

State space models for marine fish DNA barcode classification. Fine-tuned and domain-adapted from [BarcodeMamba](https://github.com/bioscan-ml/BarcodeMamba) for tropical marine species identification from eDNA.

## Why

BarcodeMamba achieves 99.2% species accuracy on insect DNA barcodes. No equivalent exists for marine taxa. MarineMamba bridges this gap by domain-adapting SSMs to marine fish COI/12S barcodes, benchmarked against BLAST and traditional pipelines.

## Approach

Three-stage domain-adaptive pretraining:
1. **Start** from BarcodeMamba pretrained weights (insect COI)
2. **Continue pretraining** on marine fish barcodes (NTP objective)
3. **Fine-tune** with classification head for species ID

We compare this against training from scratch, direct transfer, and traditional baselines (BLAST, k-NN).

## Data Sources

- [Mare-MAGE](http://mare-mage.weebly.com/) — 231K curated fish barcode sequences (COI + 12S)
- [BOLD Systems](https://boldsystems.org) — 327K+ fish specimens with COI barcodes
- [meta-fish-lib](https://github.com/genner-lab/meta-fish-lib) — curated fish metabarcoding reference library
- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/nuccore) — gap-filling

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_acquisition.ipynb` | Download and merge marine fish barcode data |
| `02_data_cleaning.ipynb` | QC, taxonomy validation, train/test splits |
| `03_baselines.ipynb` | BLAST, k-NN, Random Forest baselines |
| `04_pretrain.ipynb` | Domain-adaptive pretraining on fish barcodes |
| `05_finetune.ipynb` | Fine-tune and evaluate all models |
| `06_analysis.ipynb` | Figures, comparison tables, failure analysis |
| `07_release.ipynb` | Package model for HuggingFace + Zenodo |

## Quick Start

```bash
# Clone
git clone https://github.com/kluless/marinemamba.git
cd marinemamba

# Install
pip install -r requirements.txt

# Or open notebooks in Colab
# https://colab.research.google.com/github/kluless/marinemamba/blob/main/notebooks/01_data_acquisition.ipynb
```

## Architecture

Built on [Mamba-2](https://github.com/state-spaces/mamba) SSM blocks. See [BarcodeMamba](https://arxiv.org/abs/2412.11084) for base architecture.

| Model | Params | Tokenizer |
|---|---|---|
| MarineMamba-mini | ~7.4M | 6-mer |
| MarineMamba-large | ~56.7M | 6-mer |

## Citation

If you use this work, please cite:

```bibtex
@article{marinemamba2026,
  title={MarineMamba: Domain-Adaptive State Space Models for Marine Fish DNA Barcode Classification},
  author={TODO},
  journal={TODO},
  year={2026}
}
```

## License

MIT
