#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 3: MODELS A & B — BASELINES (BLAST + k-NN)"
echo "============================================================"

python scripts/03_baselines.py

echo ""
echo "============================================================"
echo "BASELINES DONE — run 04_model_c.sh next"
echo "============================================================"
