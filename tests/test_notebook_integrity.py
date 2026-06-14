"""Static integrity checks for example notebooks."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_example_notebooks_are_valid_json() -> None:
    """Notebook JSON corruption should be caught before release."""
    notebooks = sorted((ROOT / "examples").rglob("*.ipynb"))
    assert notebooks, "No example notebooks found"

    for notebook in notebooks:
        with notebook.open(encoding="utf-8") as handle:
            json.load(handle)


@pytest.mark.parametrize(
    "notebook_path",
    [
        "examples/interactive_tutorials/01_hello_gradient_descent.ipynb",
        "examples/interactive_tutorials/02_nmr_fundamentals.ipynb",
        "examples/interactive_tutorials/04_protein_folding_nerf.ipynb",
    ],
)
def test_notebook_execution(notebook_path: str) -> None:
    """Smoke test: execute notebooks to ensure they run without errors."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        pytest.skip("nbformat or nbclient not installed")

    path = ROOT / notebook_path
    if not path.exists():
        pytest.skip(f"Notebook {notebook_path} not found")

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    # This will raise an exception if any cell fails
    client.execute()
