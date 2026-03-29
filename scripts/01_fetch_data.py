"""Download marine fish DNA barcode data from NCBI GenBank."""
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_genbank_fish_coi(email="marinemamba@research.org", max_records=100000):
    """Fetch fish COI barcode sequences from NCBI GenBank."""
    from Bio import Entrez, SeqIO

    Entrez.email = email

    query = (
        '("COI"[Gene] OR "cox1"[Gene] OR "cytochrome c oxidase subunit I"[Gene]) '
        'AND "Actinopterygii"[Organism] '
        'AND 500:700[Sequence Length]'
    )

    print(f"Searching GenBank: {query[:80]}...")
    handle = Entrez.esearch(db="nuccore", term=query, retmax=0, usehistory="y")
    results = Entrez.read(handle)
    handle.close()

    total = min(int(results["Count"]), max_records)
    print(f"Found {results['Count']} sequences. Fetching {total}...")

    records = []
    batch_size = 500
    for start in tqdm(range(0, total, batch_size), desc="Fetching"):
        try:
            handle = Entrez.efetch(
                db="nuccore", rettype="fasta", retmode="text",
                retstart=start, retmax=batch_size,
                webenv=results["WebEnv"], query_key=results["QueryKey"],
            )
            batch = list(SeqIO.parse(handle, "fasta"))
            records.extend(batch)
            handle.close()
            time.sleep(0.35)
        except Exception as e:
            print(f"  Batch {start} error: {e}")
            time.sleep(5)
            continue

    print(f"Fetched {len(records)} sequences")
    return records


def parse_genbank_records(records):
    """Convert GenBank FASTA records to DataFrame."""
    rows = []
    for rec in records:
        parts = rec.description.split()
        seq = str(rec.seq).upper()

        # Skip if too many Ns
        if seq.count("N") / max(len(seq), 1) > 0.05:
            continue

        # Parse taxonomy from description
        # Typical: "ACC_ID Genus species gene description"
        species = " ".join(parts[1:3]) if len(parts) >= 3 else "Unknown"
        genus = parts[1] if len(parts) >= 2 else "Unknown"

        # Skip non-binomial names
        if " " not in species or species == "Unknown":
            continue

        rows.append({
            "processid": rec.id,
            "nucleotides": seq,
            "species_name": species,
            "genus_name": genus,
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("MARINEMAMBA DATA ACQUISITION (GenBank)")
    print("=" * 60)

    # Fetch from GenBank
    records = fetch_genbank_fish_coi(max_records=100000)

    # Parse to DataFrame
    print("\nParsing records...")
    df = parse_genbank_records(records)
    print(f"Parsed: {len(df)} valid sequences")

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["nucleotides"], keep="first")
    print(f"After dedup: {before} -> {len(df)}")

    # Save
    output = RAW_DIR / "merged_barcodes.csv"
    df.to_csv(output, index=False)

    # Also save raw FASTA
    from Bio import SeqIO
    SeqIO.write(records, RAW_DIR / "genbank_fish_coi.fasta", "fasta")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences:  {len(df):,}")
    print(f"Unique species:   {df['species_name'].nunique():,}")
    print(f"Unique genera:    {df['genus_name'].nunique():,}")
    print(f"Saved to: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
