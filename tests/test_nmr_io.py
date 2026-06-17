# mypy: disable-error-code="no-untyped-def"
"""Tests for diff_biophys.nmr.io.load_rdc_table."""

import textwrap
from pathlib import Path

import numpy as np

from diff_biophys.nmr.io import load_rdc_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_temp(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# 4-column format (medium from filename)
# ---------------------------------------------------------------------------


def test_load_rdc_table_4col_medium_from_filename(tmp_path):
    """4-column file: medium is inferred from the filename stem (rdc_PAG → PAG)."""
    p = write_temp(
        tmp_path,
        "rdc_PAG.tsv",
        """\
        # res_id  res_name  rdc_hz  err_hz
        14  THR  -1.793  1.187
        15  SER   2.341  0.418
        16  GLY  -0.512  0.418
        """,
    )
    result = load_rdc_table(p)
    assert "PAG" in result, f"Expected 'PAG' key, got {list(result.keys())}"
    data = result["PAG"]
    assert list(data["res_id"]) == [14, 15, 16]
    np.testing.assert_allclose(data["rdc"], [-1.793, 2.341, -0.512], atol=1e-3)


def test_load_rdc_table_4col_medium_lowercase_filename(tmp_path):
    """Filename medium inference is case-insensitive (rdc_peg → PEG)."""
    p = write_temp(
        tmp_path,
        "rdc_peg.tsv",
        "20  ALA  3.1  0.4\n",
    )
    result = load_rdc_table(p)
    assert "PEG" in result


# ---------------------------------------------------------------------------
# 5-column format (medium embedded)
# ---------------------------------------------------------------------------


def test_load_rdc_table_5col_medium_embedded(tmp_path):
    """5-column file: medium taken from column 5, overrides filename."""
    p = write_temp(
        tmp_path,
        "rdc_WRONG.tsv",
        """\
        14  THR  -1.793  1.187  PAG
        15  SER   2.341  0.418  PAG
        20  ALA   5.100  0.418  PEG
        """,
    )
    result = load_rdc_table(p)
    assert set(result.keys()) == {"PAG", "PEG"}
    assert list(result["PAG"]["res_id"]) == [14, 15]
    assert list(result["PEG"]["res_id"]) == [20]


# ---------------------------------------------------------------------------
# Comment and blank line handling
# ---------------------------------------------------------------------------


def test_load_rdc_table_skips_comments_and_blanks(tmp_path):
    """Lines starting with # and blank lines are ignored."""
    p = write_temp(
        tmp_path,
        "rdc_PAG.tsv",
        """\
        # This is a header comment
        14  THR  -1.793  1.187

        # Another comment
        15  SER   2.341  0.418
        """,
    )
    result = load_rdc_table(p)
    assert len(result["PAG"]["res_id"]) == 2


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_rdc_table_missing_file(tmp_path):
    """Returns an empty dict when the file does not exist."""
    result = load_rdc_table(tmp_path / "nonexistent.tsv")
    assert result == {}


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_load_rdc_table_values_dtype(tmp_path):
    """res_id is int32 and rdc is float32."""
    p = write_temp(
        tmp_path,
        "rdc_PAG.tsv",
        "14  THR  -1.793  1.187\n",
    )
    result = load_rdc_table(p)
    assert result["PAG"]["res_id"].dtype == np.int32
    assert result["PAG"]["rdc"].dtype == np.float32


def test_load_rdc_table_multiple_residues_correct_order(tmp_path):
    """Values appear in file order, not sorted."""
    p = write_temp(
        tmp_path,
        "rdc_PAG.tsv",
        """\
        30  LEU   1.0  0.4
        10  ALA  -2.0  0.4
        20  GLY   3.0  0.4
        """,
    )
    result = load_rdc_table(p)
    assert list(result["PAG"]["res_id"]) == [30, 10, 20]
    np.testing.assert_allclose(result["PAG"]["rdc"], [1.0, -2.0, 3.0], atol=1e-6)
