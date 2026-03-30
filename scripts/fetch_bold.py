"""
Download marine fish COI barcodes from BOLD v5 API.
New API: https://portal.boldsystems.org/api/docs

Workflow:
  1. Preprocessor: validate query terms
  2. Query: get query_id (valid 24 hours)
  3. Download: get TSV/JSON with sequences + taxonomy
"""
import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "https://portal.boldsystems.org/api"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_query(query_str):
    """Validate query terms against BOLD controlled vocabularies."""
    print(f"  Preprocessing: {query_str}")
    r = requests.get(f"{BASE_URL}/query/preprocessor", params={"query": query_str}, timeout=30)
    r.raise_for_status()
    result = r.json()
    print(f"  Successful terms: {result.get('successful_terms', [])}")
    if result.get("failed_terms"):
        print(f"  WARNING - Failed terms: {result['failed_terms']}")
    return result


def run_query(query_str, extent="full"):
    """Submit query and get query_id."""
    print(f"  Querying: {query_str} (extent={extent})")
    r = requests.get(
        f"{BASE_URL}/query",
        params={"query": query_str, "extent": extent},
        timeout=60,
    )
    r.raise_for_status()
    result = r.json()
    query_id = result.get("query_id")
    print(f"  Query ID: {query_id}")
    return query_id


def download_results(query_id, fmt="tsv", output_path=None):
    """Download query results as TSV or JSON."""
    print(f"  Downloading (format={fmt})...")
    r = requests.get(
        f"{BASE_URL}/documents/{query_id}/download",
        params={"format": fmt},
        timeout=600,
        stream=True,
    )
    r.raise_for_status()

    if output_path:
        with open(output_path, "wb") as f:
            total = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)
        size_mb = total / (1024 * 1024)
        print(f"  Saved to {output_path} ({size_mb:.1f} MB)")
    else:
        return r.text


def fetch_bold_fish():
    """Download all Actinopterygii COI barcodes from BOLD."""
    print("=" * 60)
    print("BOLD v5 API: Fetching Fish COI Barcodes")
    print("=" * 60)

    # Try different query formats
    queries_to_try = [
        "tax:class:Actinopterygii",
        "tax:Actinopterygii",
    ]

    for query_str in queries_to_try:
        try:
            # Step 1: Preprocess
            print(f"\n[1/3] Preprocessing query: {query_str}")
            prep = preprocess_query(query_str)

            # Step 2: Query
            print(f"\n[2/3] Running query...")
            query_id = run_query(query_str, extent="full")

            if not query_id:
                print("  No query_id returned, trying next format...")
                continue

            # Step 3: Download TSV
            print(f"\n[3/3] Downloading results...")
            output_path = RAW_DIR / "bold_actinopterygii_coi.tsv"
            download_results(query_id, fmt="tsv", output_path=output_path)

            # Quick stats
            print(f"\n  Checking downloaded file...")
            with open(output_path) as f:
                header = f.readline()
                line_count = sum(1 for _ in f)
            print(f"  Header: {header.strip()[:100]}...")
            print(f"  Records: {line_count:,}")

            print("\n" + "=" * 60)
            print("BOLD DOWNLOAD COMPLETE")
            print("=" * 60)
            return output_path

        except Exception as e:
            print(f"  Error with query '{query_str}': {e}")
            continue

    print("\nAll query formats failed. Try downloading manually from https://portal.boldsystems.org")
    return None


if __name__ == "__main__":
    fetch_bold_fish()
