# MarineMamba

Training objective determines what neural networks learn from marine DNA barcodes: hierarchical classification and evolutionary tree recovery from COI sequences.

## Key Findings

### 1. Curriculum Learning → Best Hierarchical Classification

Multi-head coarse-to-fine training (order → family → genus → species) on the same BarcodeMamba backbone:

| Method | Family (unseen genera) | Order (unseen genera) |
|--------|----------------------|---------------------|
| Standard SSM training | 38.4% | 56.7% |
| k-NN 6-mer baseline | 51.4% | 72.6% |
| **Curriculum (ours)** | **58.5%** | **80.8%** |

### 2. Phylogenetic Embeddings → Tree of Life Recovery

Trained to match Fish Tree of Life evolutionary distances (Rabosky et al. 2018). Beats Stalder et al. (PLOS Comp Bio 2025) on all metrics:

| Level | Stalder et al. | Ours |
|-------|---------------|------|
| Genus | 50.7% | **61.2%** |
| Family | 80.5% | **83.4%** |
| Order | 86.7% | **92.8%** |

**Tree recovery generalisation** (species never seen during training): Pearson r = **0.865**

The model learned how COI sequences evolve — not memorised. Distances correlate with real evolutionary divergence spanning 4-386 million years.

### 3. Foundation Models Underperform

First evaluation of Evo 2 (7B, Nature 2026) on DNA barcodes. A 7B parameter model loses to 6-mer counting on hierarchical classification. With curriculum + LoRA: improved but still below 4.3M parameter domain-specific SSM.

## Data

### Marine 869K (Full Benchmark)
- **869,222 COI barcode sequences** from [BOLD v5 API](https://portal.boldsystems.org/api)
- **14 marine phyla**: fish, crabs, molluscs, corals, jellyfish, sea stars, worms, sponges, sharks
- **76,925 species** | 14,216 genera | 1,850 families
- 557 genera held out for zero-shot evaluation

### Fish Tree of Life Subset
- 6,510 species matched to [Fish Tree of Life](https://fishtreeoflife.org) (Rabosky et al. 2018)
- 140,112 training sequences with real evolutionary distances in millions of years

## 25+ Experiments

| Experiment | Script | Key Result |
|-----------|--------|-----------|
| BLAST baseline | `03_baselines.py` | 91.8% species |
| k-NN 6-mer baseline | `03_baselines.py` | 94.2% species, 51.4% family |
| BarcodeMamba transfer | `04_barcodemamba_models.py --mode transfer` | 93.0% species |
| BarcodeMamba scratch | `04_barcodemamba_models.py --mode scratch` | 92.9% species |
| BarcodeMamba adapted | `04_barcodemamba_models.py --mode adapt` | 92.8% species |
| Evo 2 7B frozen | `05_evo2_embeddings.py` | 88.4% species |
| **Curriculum (char, scratch)** | `09_multihead_hierarchical.py` | **58.5% family unseen genera** |
| Curriculum + pretrained backbone | `09_multihead_hierarchical.py --pretrain-ckpt` | 57.3% family |
| Curriculum + 6-mer | `11_curriculum_6mer.py` | 53.9% family |
| Evo 2 + LoRA + curriculum | `10_evo2_lora_curriculum.py` | 43.8% family |
| **Phylo fish-only (dim=384)** | `12_phylo_fish_only.py --embed-dim 384` | **r=0.978 tree, 83.4% family** |
| Tree recovery unseen | `tree_recovery_unseen.py` | **r=0.865 generalisation** |

## Quick Start

```bash
# Clone
git clone https://github.com/kluless13/marinemamba.git
cd marinemamba

# Fetch BOLD marine data (requires internet)
python3 scripts/fetch_bold_marine.py

# Process into train/test/unseen splits
python3 scripts/02_clean_and_split.py

# Run curriculum learning (needs GPU)
python3 scripts/09_multihead_hierarchical.py --data-dir data/processed --output-dir results

# Run phylogenetic embeddings on fish subset (needs dendropy)
pip install dendropy
python3 scripts/12_phylo_fish_only.py --data-dir data/processed --output-dir results --embed-dim 384
```

## Architecture

Built on [BarcodeMamba](https://arxiv.org/abs/2412.11084) (Mamba-2 SSM). 2 layers, d_model=384, character-level tokenization.

| Model | Params | Training Objective | Best For |
|-------|--------|-------------------|----------|
| Curriculum model | ~17M (4.3M backbone + 4 heads) | Order→Family→Genus→Species staged | Classification of novel taxa |
| Phylo model | ~4.5M (4.3M backbone + projection) | Match Fish Tree of Life distances | Evolutionary placement |

## Novelty

| Claim | Status |
|-------|--------|
| Tree recovery generalises to unseen species (r=0.865) | **Novel** — DEPP (2022) flagged this as failure mode; we show it works |
| Evo 2 on barcode classification | **Novel** — first published evaluation |
| Taxonomic rank curriculum on DNA barcodes | **Novel formulation** — distinct from Ye et al. (data-modality staging) |
| Phylogenetic embeddings + SSM backbone | **Novel combination** — Phyla (NeurIPS 2025) did Mamba+phylo on proteins only |
| Marine COI benchmark (869K, 14 phyla) | **Novel** — no equivalent exists |

## Key References

- BarcodeMamba (Gao & Taylor, 2024) — [arXiv:2412.11084](https://arxiv.org/abs/2412.11084)
- Stalder et al. (2025) — [PLOS Comp Bio](https://doi.org/10.1371/journal.pcbi.1013776)
- Ye et al. / SnailBaLLsp (2026) — [MEE](https://doi.org/10.1111/2041-210x.70264)
- Evo 2 (Brixi et al., 2026) — [Nature](https://doi.org/10.1038/s41586-026-10176-5)
- Fish Tree of Life (Rabosky et al., 2018) — [Nature](https://doi.org/10.1038/s41586-018-0273-1)

## Citation

```bibtex
@article{marinemamba2026,
  title={Training Objective Determines Hierarchical Classification and Evolutionary Recovery from Marine DNA Barcodes},
  author={TODO},
  journal={Methods in Ecology and Evolution},
  year={2026}
}
```

## License

MIT
