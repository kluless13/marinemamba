"""
Convert BOLD v5 TSV to the merged_barcodes.csv format expected by 02_clean_and_split.py.

BOLD columns → Our columns:
  processid → processid
  nuc → nucleotides
  species → species_name
  genus → genus_name
  family → family_name
  order → order_name
  marker_code → (filter for COI)
"""
import csv
import sys
from pathlib import Path
from collections import Counter

csv.field_size_limit(sys.maxsize)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
BOLD_FILE = RAW_DIR / "bold_teleostei.tsv"
OUTPUT_FILE = RAW_DIR / "merged_barcodes.csv"


def main():
    print("=" * 60)
    print("BOLD → merged_barcodes.csv")
    print("=" * 60)

    total = 0
    kept = 0
    skipped = Counter()
    species_set = set()
    genera_set = set()

    with open(BOLD_FILE, "r") as infile, open(OUTPUT_FILE, "w", newline="") as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        writer = csv.DictWriter(
            outfile,
            fieldnames=["processid", "nucleotides", "species_name", "genus_name", "family_name", "order_name"],
        )
        writer.writeheader()

        for row in reader:
            total += 1

            seq = row.get("nuc", "").strip()
            species = row.get("species", "").strip()
            genus = row.get("genus", "").strip()
            family = row.get("family", "").strip()
            order = row.get("order", "").strip()
            marker = row.get("marker_code", "").strip()
            processid = row.get("processid", "").strip()

            # Filter: must have COI marker
            if "COI" not in marker.upper():
                skipped["no_COI_marker"] += 1
                continue

            # Filter: must have sequence
            if not seq or len(seq) < 100:
                skipped["short_or_no_sequence"] += 1
                continue

            # Filter: must have species-level ID (binomial)
            if not species or " " not in species:
                skipped["no_species_id"] += 1
                continue

            # Clean sequence: uppercase, remove gaps/non-ACGTN
            clean_seq = seq.upper().replace("-", "").replace(".", "")

            writer.writerow({
                "processid": processid,
                "nucleotides": clean_seq,
                "species_name": species,
                "genus_name": genus if genus else species.split()[0],
                "family_name": family,
                "order_name": order,
            })

            kept += 1
            species_set.add(species)
            genera_set.add(genus if genus else species.split()[0])

            if kept % 50000 == 0:
                print(f"  Processed {kept:,} records...")

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total BOLD records:  {total:,}")
    print(f"Kept:                {kept:,}")
    print(f"Unique species:      {len(species_set):,}")
    print(f"Unique genera:       {len(genera_set):,}")
    print(f"\nSkipped reasons:")
    for reason, count in skipped.most_common():
        print(f"  {reason}: {count:,}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
