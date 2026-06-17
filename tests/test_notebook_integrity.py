"""Static integrity checks for example notebooks."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# On Windows the default Proactor event loop does not support the add_reader
# family of methods that zmq (and therefore nbclient) requires.  Switch to the
# Selector policy before any notebook execution so that kernel communication
# works reliably and does not produce spurious timeouts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# JAX's first import triggers XLA compilation and device detection.  On
# Windows CI with a cold XLA cache this can take well over 60 seconds.
# 300 s gives a generous margin without masking genuine hangs.
_NOTEBOOK_TIMEOUT = 300


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
        "examples/interactive_tutorials/05_hybrid_refinement_pdb.ipynb",
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

    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=_NOTEBOOK_TIMEOUT, kernel_name="python3")
    # This will raise an exception if any cell fails
    client.execute()
