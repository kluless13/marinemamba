#!/bin/bash
cd /workspace/marinemamba

echo "============================================================"
echo "SAVING RESULTS TO GITHUB"
echo "============================================================"

git config user.email "angd1399@gmail.com"
git config user.name "kluless13"

git add -f results/*.json data/processed/dataset_stats.json
git commit -m "results: BOLD 320K dataset, all 6 models on Vast.ai A100" || echo "Nothing to commit"
git push

echo ""
echo "============================================================"
echo "RESULTS PUSHED — you can now destroy the instance on vast.ai"
echo "============================================================"
