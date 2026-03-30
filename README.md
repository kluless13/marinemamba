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

## Data

**360,757 COI barcode sequences** from [BOLD Systems v5 API](https://portal.boldsystems.org/api) (Barcode of Life Data), covering **25,663 species** across **4,145 genera** of ray-finned fishes (Teleostei).

After quality filtering (500-700bp, ≤5% ambiguous bases, species-level ID required):

| Split | Sequences | Species | Genera |
|---|---|---|---|
| Total | 318,829 | 23,663 | 4,017 |
| Train | 194,457 | 10,242 | — |
| Validation | 27,780 | 10,242 | — |
| Test | 55,560 | 10,242 | — |
| Unseen genera | 19,399 | — | 173 |

Unseen genera are held out entirely — no species from these genera appear in training. This tests whether models learn transferable taxonomic representations rather than memorizing species-specific patterns.

## Models

| Model | Method | Description |
|---|---|---|
| A | BLAST | BLASTn top-1 hit against training database |
| B | k-NN + 6-mer | 1-NN cosine similarity on 6-mer frequency vectors |
| C | BarcodeMamba (transfer) | Insect-pretrained SSM, fine-tuned on fish |
| D | BarcodeMamba (scratch) | SSM pretrained from scratch on fish COI, then fine-tuned |
| E | BarcodeMamba (adapted) | Insect SSM, continued pretraining on fish, then fine-tuned |
| F | Evo 2 (7B) | Embeddings from Arc Institute's genomic foundation model + linear classifier |

## Quick Start

```bash
# Run everything on Vast.ai A100 (recommended)
pip install vastai-sdk
export VAST_API_KEY="your-key"
# Then run scripts/vast/01_setup.sh through 09_save_results.sh

# Or open in Colab
# https://colab.research.google.com/github/kluless13/marinemamba/blob/main/notebooks/gpu_runner.ipynb
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
