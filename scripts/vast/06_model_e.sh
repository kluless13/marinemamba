#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 6: MODEL E — BARCODEMAMBA DOMAIN ADAPTATION"
echo "============================================================"

python scripts/04_barcodemamba_models.py --mode adapt --data-dir data/processed --output-dir results

echo ""
echo "============================================================"
echo "MODEL E DONE — run 07_model_f.sh next"
echo "============================================================"
