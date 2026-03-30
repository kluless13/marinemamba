#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 4: MODEL C — BARCODEMAMBA TRANSFER (insect → fish)"
echo "============================================================"

python scripts/04_barcodemamba_models.py --mode transfer --data-dir data/processed --output-dir results

echo ""
echo "============================================================"
echo "MODEL C DONE — run 05_model_d.sh next"
echo "============================================================"
