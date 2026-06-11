"""Static integrity checks for example notebooks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_example_notebooks_are_valid_json() -> None:
    """Notebook JSON corruption should be caught before release."""
    notebooks = sorted((ROOT / "examples").rglob("*.ipynb"))
    assert notebooks, "No example notebooks found"

    for notebook in notebooks:
        with notebook.open(encoding="utf-8") as handle:
            json.load(handle)
