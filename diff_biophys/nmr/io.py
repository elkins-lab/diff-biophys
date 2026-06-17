"""
diff_biophys.nmr.io
===================
Generic NMR experimental data file I/O.

Currently supports whitespace-delimited RDC tables in the 4- or 5-column
format produced by PALES/DC and CYANA exporters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_rdc_table(path: Path | str) -> dict[str, dict[str, np.ndarray]]:
    """Load ¹⁵N–¹H RDC data from a whitespace-delimited file.

    Two column formats are accepted (comment lines begin with ``#``):

    **4-column** (medium inferred from the filename stem)::

        # res_id  res_name  rdc_hz  err_hz
        14        THR       -1.793  1.187
        15        SER        2.341  0.418

    **5-column** (medium embedded in the file)::

        14  THR  -1.793  1.187  PAG
        15  SER   2.341  0.418  PAG

    The medium name is always upper-cased.  For 4-column files the medium is
    taken from the last ``_``-delimited token of the filename stem, e.g.
    ``rdc_PAG.tsv`` → ``"PAG"``.

    Args:
        path: Path to the RDC table file.

    Returns:
        ``dict`` keyed by medium name (e.g. ``"PAG"``, ``"PEG"``).  Each
        value is a ``dict`` with keys:

        * ``"res_id"`` — ``np.ndarray`` of shape ``(M,)``, dtype int32
        * ``"rdc"``    — ``np.ndarray`` of shape ``(M,)``, dtype float32

        Returns an empty dict if the file does not exist or contains no
        valid data rows.
    """
    path = Path(path)
    if not path.exists():
        return {}

    # Derive medium from filename when not embedded in the file.
    # e.g. "rdc_PAG.tsv" → stem "rdc_PAG" → "PAG"
    stem = path.stem.upper()
    filename_medium = stem.split("_")[-1] if "_" in stem else stem

    groups: dict[str, tuple[list[int], list[float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            try:
                res_id = int(tokens[0])
                rdc_hz = float(tokens[2])
                medium = tokens[4].upper() if len(tokens) >= 5 else filename_medium
            except (ValueError, IndexError):
                continue
            ids, vals = groups.setdefault(medium, ([], []))
            ids.append(res_id)
            vals.append(rdc_hz)

    return {
        medium: {
            "res_id": np.array(ids, dtype=np.int32),
            "rdc": np.array(vals, dtype=np.float32),
        }
        for medium, (ids, vals) in groups.items()
    }
