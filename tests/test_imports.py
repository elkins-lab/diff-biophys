import sys
from importlib import reload
from typing import Any
from unittest.mock import patch

import pytest


def test_nmr_constants_fallback() -> None:
    """Verify that NMR constants fall back gracefully when synth-nmr is missing."""
    import diff_biophys.nmr.constants as constants

    # Mocking synth-nmr as missing
    with patch.dict(
        sys.modules,
        {"synth_nmr": None, "synth_nmr.chemical_shifts": None, "synth_nmr.j_coupling": None},
    ):
        reload(constants)

        # Verify defaults are still present
        assert constants.KARPLUS_A == 6.51
        assert "PHE" in constants.RING_INTENSITIES
        assert constants.RING_INTENSITIES["PHE"] == 1.2

    # Restore
    reload(constants)


def test_torch_interop_fallback() -> None:
    """Verify that jax_to_torch raises ImportError when torch is missing."""
    import diff_biophys.torch_interop as torch_interop

    with patch.dict(sys.modules, {"torch": None}):
        reload(torch_interop)
        assert torch_interop.HAS_TORCH is False

        def dummy_fn(x: Any) -> Any:
            return x

        with pytest.raises(ImportError, match="PyTorch must be installed"):
            torch_interop.jax_to_torch(dummy_fn)

    # Restore
    reload(torch_interop)


def test_init_version_fallback() -> None:
    """Verify that __init__ handles missing package metadata gracefully."""
    import diff_biophys

    with patch("importlib.metadata.version") as mock_version:
        from importlib.metadata import PackageNotFoundError

        mock_version.side_effect = PackageNotFoundError

        reload(diff_biophys)
        assert diff_biophys.__version__ == "unknown"

    # Restore
    reload(diff_biophys)
