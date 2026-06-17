#!/usr/bin/env python3
"""
fetch_data.py — Download required data files for the HR2876B benchmark.

Downloads:
  - 2LTM.pdb              from RCSB PDB (NMR ensemble)
  - bmrb18489_HR2876B.str from BMRB FTP (chemical shifts + RDCs)

Usage:
    python fetch_data.py
"""

import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).parent

DOWNLOADS = {
    "2LTM.pdb": "https://files.rcsb.org/download/2LTM.pdb",
    "bmrb18489_HR2876B.str": (
        "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr18489/bmr18489_3.str"
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
    print("Fetching HR2876B benchmark data files...")
    for filename, url in DOWNLOADS.items():
        fetch(filename, url)
    print("\nAll files ready.")


if __name__ == "__main__":
    main()
