# Literature Review: DNA Barcode Classification with SSMs and Foundation Models

## Barcode-Specific Models

### BarcodeMamba (Peng et al., 2024)
- **Architecture**: Mamba-2 SSM, 7.4M / 56.7M params
- **Data**: 1.5M insect COI barcodes (BIOSCAN-5M)
- **Results**: 99.2% species accuracy (linear probe), **70.2% genus zero-shot** (1-NN)
- **Gap**: Insects only. No marine fish. No foundation model comparison.
- [arXiv:2412.11084](https://arxiv.org/abs/2412.11084)

### BarcodeMamba+ (2025)
- Extended BarcodeMamba to **fungal ITS barcodes** (5.23M sequences)
- Added hierarchical label smoothing and multi-head taxonomic output
- **Gap**: Fungi only. No marine vertebrates.
- [arXiv:2512.15931](https://arxiv.org/abs/2512.15931)

### BarcodeBERT (Milczek et al., 2024)
- Masked language model pretrained on ~1M invertebrate barcodes
- Zero-shot 1-NN genus classification evaluated
- **Gap**: Transformers, not SSMs. Insects only.
- [arXiv:2311.02401](https://arxiv.org/abs/2311.02401)

### DNACSE (2025)
- Contrastive learning on DNABERT-2 for barcode tasks
- Zero-shot clustering AMI of 92.25%
- **Shows**: General DNA models CAN be adapted to barcodes
- [J. Chem. Inf. Model.](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02747)

### DeepCOI (2025)
- LLM pre-trained on 7M COI sequences across 8 phyla
- AU-ROC 0.958 for metabarcoding assignment
- Published in Genome Biology
- **Gap**: No zero-shot genus evaluation. No SSM comparison.

### BIOSCAN-CLIP (Gong et al., 2024)
- Multimodal: DNA barcode + specimen image + taxonomy text
- Contrastive learning, 8%+ improvement on zero-shot
- **Gap**: Requires images. Insects only.

## General Genomic Foundation Models

### Evo 2 (Brixi et al., 2026) — Nature
- 7B / 40B params, StripedHyena architecture
- Trained on 9.3 trillion nucleotides from 128K+ genomes (all domains of life)
- Zero-shot mutation impact prediction (AUROC 0.921)
- **Gap**: NOT tested on species classification or barcode identification
- [Nature](https://www.nature.com/articles/s41586-026-10176-5)

### HyenaDNA (Nguyen et al., 2023) — NeurIPS
- Hyena operator, up to 1M nucleotide context
- Pretrained on human reference genome
- Species classification: only 5-way (human/lemur/mouse/pig/hippo) from long fragments
- **Gap**: NOT tested on short barcodes (~660bp) or multi-species classification

### DNABERT-2 (Zhou et al., 2024) — ICLR
- BPE tokenization, 21x fewer params than Nucleotide Transformer
- Genomic benchmarks but NOT barcode classification
- DNACSE later adapted it for barcodes

### Nucleotide Transformer (Dalla-Torre et al., 2024) — Nature Methods
- 50M-2.5B params, 3,202 human genomes + 850 species
- Regulatory genomics tasks
- **Gap**: NOT tested on barcode classification

## Hierarchical / Novel Taxa Methods

### PROTAX (Somervuo et al., 2016)
- Probabilistic taxonomic assignment, allows unknown taxa at every rank
- Statistical (multinomial regression), not deep learning
- PROTAX-GPU (2024) added scalability
- **Gap**: Not neural. No learned embeddings.

### BayesANT (Zito et al., 2023) — Methods in Ecology and Evolution
- Bayesian nonparametric classifier with Pitman-Yor priors
- Probabilistic predictions at each rank, allows novel taxon discovery
- **Gap**: Statistical only. No SSM/transformer comparison.

### Orsholm et al. (2025) — bioRxiv
- Benchmark of EPA-ng, PROTAX, BayesANT, SINTAX, RDP-NBC, IDTAXA
- For "unknown" arthropod COI and fungal ITS barcodes
- **Gap**: Does NOT include any neural model (BarcodeBERT, BarcodeMamba, etc.)

## Fish-Specific Work

### Zhao et al. (2021)
- Elastic Net + Stacked Autoencoder for fish family classification from COI
- ~97.57% accuracy but small ad-hoc datasets
- **Gap**: Old architectures. No SSMs. No zero-shot.

### Australia's Marine Fish Barcode Reference Library (Ward et al., 2025)
- 9,767 specimens, 2,220+ species, 288 families
- Reference DATABASE, not ML benchmark
- [Scientific Data](https://www.nature.com/articles/s41597-025-04375-4)

### Benchmarking DNA Foundation Models (Dalla-Torre et al., 2025) — Nature Comms
- Compared DNABERT-2, Nucleotide Transformer V2, HyenaDNA, Caduceus across 49 tasks
- **Gap**: Barcode classification NOT included as a task

## Summary of Gaps Our Work Fills

| Gap | Severity | Our Contribution |
|-----|----------|-----------------|
| No SSM/foundation model on **marine fish** barcodes | HIGH | First benchmark: 6 models on 318K BOLD sequences |
| No **Evo 2** tested on barcode classification | HIGH | First Evo 2 evaluation on COI barcodes |
| No **hierarchical zero-shot** with neural models | HIGH | Family/order level evaluation on 173 held-out genera |
| No **marine fish COI benchmark** (equivalent to BIOSCAN for insects) | HIGH | 318K sequences, 23K species, standardized splits |
| No **general vs domain-specific** foundation model comparison on barcodes | MEDIUM-HIGH | Evo 2 (7B generalist) vs BarcodeMamba (8.2M specialist) |
| Neural models not included in unknown taxa benchmarks | MEDIUM | Bridge between statistical (PROTAX/BayesANT) and neural approaches |

## Critical Context: Dataset Size Comparison

| Dataset | Sequences | Species | Genera | Zero-shot Genus Acc |
|---------|-----------|---------|--------|-------------------|
| BIOSCAN-5M (insects) | 1,500,000 | ~50,000 | ~10,000 | 70.2% (BarcodeMamba) |
| **BOLD Teleostei (ours)** | **318,829** | **23,663** | **4,017** | **0.53% (Model E)** |

The 5x difference in dataset size may partly explain the genus accuracy gap.
Other factors: insect COI may have stronger inter-genus divergence than fish COI,
and the BIOSCAN evaluation uses a different holdout strategy.
