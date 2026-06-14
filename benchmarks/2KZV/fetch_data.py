#!/usr/bin/env python3
"""
fetch_data.py — Download required data files for the 2KZV benchmark.

Downloads:
  - 2KZV.pdb        from RCSB PDB
  - bmrb17020.str   from BMRB FTP

Usage:
    python fetch_data.py
"""

import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).parent

DOWNLOADS = {
    "2KZV.pdb": "https://files.rcsb.org/download/2KZV.pdb",
    "bmrb17020.str": "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr17020/bmr17020_3.str",
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
    print("Fetching 2KZV benchmark data files...")
    for filename, url in DOWNLOADS.items():
        fetch(filename, url)
    print("\nAll files ready. Run the benchmark with:")
    print("  python benchmark_2KZV.py")
    print("\nFor Phase 2 (RDC), populate rdc_PAG.tsv then run:")
    print("  python benchmark_2KZV.py --rdc rdc_PAG.tsv")


if __name__ == "__main__":
    main()
