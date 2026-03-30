#!/bin/bash
cd /workspace/marinemamba

echo "============================================================"
echo "MARINEMAMBA EXPERIMENT RESULTS"
echo "============================================================"

python -c "
import json, glob

with open('data/processed/dataset_stats.json') as f:
    stats = json.load(f)

print(f'Dataset: BOLD Teleostei COI Barcodes')
print(f'  Total sequences: {stats[\"total_sequences\"]:,}')
print(f'  Species:         {stats[\"total_species\"]:,}')
print(f'  Genera:          {stats[\"total_genera\"]:,}')
print(f'  Train:           {stats[\"train_size\"]:,}')
print(f'  Test:            {stats[\"test_size\"]:,}')
print(f'  Unseen genera:   {stats[\"unseen_size\"]:,} ({stats[\"unseen_genera\"]} genera)')
print(f'  N classes:       {stats[\"n_classes\"]:,}')
print()
print(f'{\"Model\":<35} {\"Species Acc\":>12} {\"Unseen Genus\":>14}')
print('-' * 65)

for f_path in sorted(glob.glob('results/*.json')):
    name = f_path.split('/')[-1].replace('.json', '')
    with open(f_path) as f:
        r = json.load(f)
    if name == 'baselines':
        blast_acc = r.get('blast', {}).get('accuracy', None)
        knn_sp = r.get('knn', {}).get('species_accuracy_test', None)
        knn_g = r.get('knn', {}).get('genus_accuracy_unseen', None)
        rf_acc = r.get('random_forest', {}).get('species_accuracy_test', None)
        if blast_acc is not None:
            print(f'{\"A: BLAST\":<35} {blast_acc:>12.4f} {\"0.0000\":>14}')
        if knn_sp is not None:
            print(f'{\"B: k-NN (6-mer, cosine)\":<35} {knn_sp:>12.4f} {knn_g:>14.4f}')
        if rf_acc is not None:
            print(f'{\"B: Random Forest (6-mer)\":<35} {rf_acc:>12.4f} {\"—\":>14}')
    else:
        sp = r.get('linear_probe_accuracy', r.get('species_logistic_regression', {}).get('accuracy', '?'))
        g = r.get('genus_accuracy_unseen', r.get('genus_knn_unseen', {}).get('accuracy', '?'))
        label = name.replace('_results', '').replace('_', ' ').title()
        sp_s = f'{sp:>12.4f}' if isinstance(sp, float) else f'{sp:>12}'
        g_s = f'{g:>14.4f}' if isinstance(g, float) else f'{g:>14}'
        print(f'{label:<35} {sp_s} {g_s}')
print('=' * 65)
"

echo ""
echo "Result files:"
ls -la results/*.json 2>/dev/null || echo "  No results yet"

echo ""
echo "============================================================"
echo "DONE! Push results with: 09_save_results.sh"
echo "============================================================"
