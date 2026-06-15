#!/usr/bin/env python3
"""
fetch_data.py — Download required data files for the GmR58A benchmark.

Downloads:
  - 2KUT.pdb              from RCSB PDB (10-model NMR ensemble)
  - bmrb16746_GmR58A.str  from BMRB FTP (chemical shifts + 3-media RDCs)

Usage:
    python fetch_data.py
"""

import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).parent

DOWNLOADS = {
    "2KUT.pdb": "https://files.rcsb.org/download/2KUT.pdb",
    "bmrb16746_GmR58A.str": (
        "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr16746/bmr16746_3.str"
    ),
}


def fetch(filename: str, url: str) -> None:
    dest = BENCH_DIR / filename
    if dest.exists():
        print(f"  {filename} already exists, skipping.")
        return
    print(f"  Downloading {filename} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb} KB)")
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print("Fetching GmR58A benchmark data files...")
    for filename, url in DOWNLOADS.items():
        fetch(filename, url)
    print("\nAll files ready. Run the benchmark with:")
    print("  python benchmark_GmR58A.py                     # Cα shifts only")
    print("  python benchmark_GmR58A.py --rdc all           # Cα + 3 RDC media")
    print("  python benchmark_GmR58A.py --rdc RDC_list_1   # Cα + 1 medium")


if __name__ == "__main__":
    main()
