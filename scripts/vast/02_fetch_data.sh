#!/bin/bash
set -e
cd /workspace/marinemamba

echo "============================================================"
echo "STEP 2: FETCH & PROCESS BOLD DATA"
echo "============================================================"

python3 -c "
import requests, csv, sys, os
from pathlib import Path
from collections import Counter
csv.field_size_limit(sys.maxsize)

RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)
BOLD_TSV = RAW_DIR / 'bold_teleostei.tsv'
MERGED_CSV = RAW_DIR / 'merged_barcodes.csv'

# Download from BOLD v5 API
if not BOLD_TSV.exists():
    print('Fetching Teleostei from BOLD v5 API...')
    BASE = 'https://portal.boldsystems.org/api'
    r = requests.get(f'{BASE}/query', params={'query': 'tax:class:Teleostei', 'extent': 'full'}, timeout=60)
    r.raise_for_status()
    query_id = r.json()['query_id']
    print(f'  Query ID: {query_id}')

    r = requests.get(f'{BASE}/documents/{query_id}/download', params={'format': 'tsv'}, timeout=600, stream=True)
    r.raise_for_status()
    total = 0
    with open(BOLD_TSV, 'wb') as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
            total += len(chunk)
    print(f'  Downloaded: {total / 1e6:.1f} MB')
else:
    print(f'BOLD data already exists: {BOLD_TSV}')

# Convert to merged_barcodes.csv
print('Converting BOLD TSV to merged_barcodes.csv...')
kept = 0
skipped = Counter()
with open(BOLD_TSV, 'r') as infile, open(MERGED_CSV, 'w', newline='') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    writer = csv.DictWriter(outfile, fieldnames=['processid', 'nucleotides', 'species_name', 'genus_name', 'family_name', 'order_name'])
    writer.writeheader()
    for row in reader:
        seq = row.get('nuc', '').strip()
        species = row.get('species', '').strip()
        marker = row.get('marker_code', '').strip()
        if 'COI' not in marker.upper():
            skipped['no_COI'] += 1; continue
        if not seq or len(seq) < 100:
            skipped['short_seq'] += 1; continue
        if not species or ' ' not in species:
            skipped['no_species'] += 1; continue
        genus = row.get('genus', '').strip() or species.split()[0]
        writer.writerow({
            'processid': row.get('processid', '').strip(),
            'nucleotides': seq.upper().replace('-', '').replace('.', ''),
            'species_name': species,
            'genus_name': genus,
            'family_name': row.get('family', '').strip(),
            'order_name': row.get('order', '').strip(),
        })
        kept += 1
print(f'  Kept: {kept:,}')
for reason, count in skipped.most_common():
    print(f'  Skipped ({reason}): {count:,}')
"

echo ""
echo "Running train/test/unseen splits..."
python3 scripts/02_clean_and_split.py

echo ""
echo "============================================================"
echo "DATA READY — run 03_baselines.sh next"
echo "============================================================"
