#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 7: MODEL F — EVO 2 (7B) EMBEDDINGS"
echo "============================================================"

# Clear GPU memory from previous models
python -c "
import torch, gc
gc.collect()
torch.cuda.empty_cache()
free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
print(f'VRAM free: {free:.1f} GB')
"

python scripts/05_evo2_embeddings.py --data-dir data/processed --output-dir results

echo ""
echo "============================================================"
echo "MODEL F DONE — run 08_results.sh to see summary"
echo "============================================================"
