"""
Fetch ALL marine animal COI barcodes from BOLD v5 API.

Combined ~1.2M sequences across:
  - Teleostei (bony fish): 420K
  - Mollusca (snails, octopus, clams): 312K
  - Malacostraca (crabs, lobsters, shrimp): 233K
  - Polychaeta (marine worms): 55K
  - Echinodermata (sea stars, urchins): 49K
  - Cnidaria (corals, jellyfish): 44K
  - Elasmobranchii (sharks, rays): 29K
  - Porifera (sponges): 13K

Usage:
    python3 scripts/fetch_bold_marine.py
"""
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests

csv.field_size_limit(sys.maxsize)

BASE_URL = "https://portal.boldsystems.org/api"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MARINE_TAXA = [
    {"name": "Teleostei", "query": "tax:class:Teleostei", "phylum": "Chordata", "expected": 420000},
    {"name": "Mollusca", "query": "tax:phylum:Mollusca", "phylum": "Mollusca", "expected": 312000},
    {"name": "Malacostraca", "query": "tax:class:Malacostraca", "phylum": "Arthropoda", "expected": 233000},
    {"name": "Polychaeta", "query": "tax:class:Polychaeta", "phylum": "Annelida", "expected": 55000},
    {"name": "Echinodermata", "query": "tax:phylum:Echinodermata", "phylum": "Echinodermata", "expected": 49000},
    {"name": "Cnidaria", "query": "tax:phylum:Cnidaria", "phylum": "Cnidaria", "expected": 44000},
    {"name": "Elasmobranchii", "query": "tax:class:Elasmobranchii", "phylum": "Chordata", "expected": 29000},
    {"name": "Porifera", "query": "tax:phylum:Porifera", "phylum": "Porifera", "expected": 13000},
]


def fetch_taxon(taxon_info):
    """Download a single taxon from BOLD."""
    name = taxon_info["name"]
    query = taxon_info["query"]
    tsv_path = RAW_DIR / f"bold_{name.lower()}.tsv"

    if tsv_path.exists():
        with open(tsv_path) as f:
            count = sum(1 for _ in f) - 1
        print(f"  {name}: already downloaded ({count:,} records)")
        return tsv_path, count

    print(f"  {name}: querying {query}...")
    r = requests.get(f"{BASE_URL}/query", params={"query": query, "extent": "full"}, timeout=60)
    r.raise_for_status()
    query_id = r.json().get("query_id")

    if not query_id:
        print(f"  {name}: no query_id returned, skipping")
        return None, 0

    print(f"  {name}: downloading...")
    r = requests.get(
        f"{BASE_URL}/documents/{query_id}/download",
        params={"format": "tsv"},
        timeout=1200,
        stream=True,
    )
    r.raise_for_status()

    total = 0
    with open(tsv_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
            total += len(chunk)

    with open(tsv_path) as f:
        count = sum(1 for _ in f) - 1

    print(f"  {name}: {count:,} records ({total / 1e6:.1f} MB)")
    return tsv_path, count


def merge_all(tsv_paths, output_path):
    """Merge all taxon TSVs into a single merged_barcodes.csv."""
    print(f"\nMerging {len(tsv_paths)} taxa into {output_path}...")

    kept = 0
    skipped = Counter()
    species_set = set()
    phyla_set = set()

    with open(output_path, "w", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["processid", "nucleotides", "species_name", "genus_name",
                        "family_name", "order_name", "class_name", "phylum_name"],
        )
        writer.writeheader()

        for tsv_path in tsv_paths:
            with open(tsv_path, "r") as infile:
                reader = csv.DictReader(infile, delimiter="\t")
                for row in reader:
                    seq = row.get("nuc", "").strip()
                    species = row.get("species", "").strip()
                    marker = row.get("marker_code", "").strip()

                    if "COI" not in marker.upper():
                        skipped["no_COI"] += 1
                        continue
                    if not seq or len(seq) < 100:
                        skipped["short_seq"] += 1
                        continue
                    if not species or " " not in species:
                        skipped["no_species"] += 1
                        continue

                    genus = row.get("genus", "").strip() or species.split()[0]
                    phylum = row.get("phylum", "").strip()

                    writer.writerow({
                        "processid": row.get("processid", "").strip(),
                        "nucleotides": seq.upper().replace("-", "").replace(".", ""),
                        "species_name": species,
                        "genus_name": genus,
                        "family_name": row.get("family", "").strip(),
                        "order_name": row.get("order", "").strip(),
                        "class_name": row.get("class", "").strip(),
                        "phylum_name": phylum,
                    })
                    kept += 1
                    species_set.add(species)
                    phyla_set.add(phylum)

                    if kept % 100000 == 0:
                        print(f"    Processed {kept:,} records...")

    print(f"\n  Total kept: {kept:,}")
    print(f"  Unique species: {len(species_set):,}")
    print(f"  Phyla: {phyla_set}")
    for reason, count in skipped.most_common():
        print(f"  Skipped ({reason}): {count:,}")

    return kept


def main():
    print("=" * 60)
    print("BOLD v5 API: MARINE BIODIVERSITY DATA FETCH")
    print("=" * 60)

    tsv_paths = []
    total_records = 0

    for taxon in MARINE_TAXA:
        tsv_path, count = fetch_taxon(taxon)
        if tsv_path:
            tsv_paths.append(tsv_path)
            total_records += count

    print(f"\nTotal raw records across {len(tsv_paths)} taxa: {total_records:,}")

    output_path = RAW_DIR / "merged_marine_barcodes.csv"
    kept = merge_all(tsv_paths, output_path)

    print(f"\n{'=' * 60}")
    print(f"MARINE DATA FETCH COMPLETE")
    print(f"  Raw records: {total_records:,}")
    print(f"  After filtering: {kept:,}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
