#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 5: MODEL D — BARCODEMAMBA FROM SCRATCH"
echo "============================================================"
echo "This is the longest run (pretrain + finetune)."
echo ""

python scripts/04_barcodemamba_models.py --mode scratch --data-dir data/processed --output-dir results

echo ""
echo "============================================================"
echo "MODEL D DONE — run 06_model_e.sh next"
echo "============================================================"
