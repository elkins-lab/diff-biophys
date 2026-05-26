import numpy as np
import pytest
from diff_biophys.nmr.chemical_shifts import RANDOM_COIL_CA


# Published Cα random coil shifts (ppm) from:
# Wishart et al. (1995) J. Biomol. NMR 5, 67–81, Table 1
# Values used as the ground truth; any deviation > 0.05 ppm is a data-entry error.
WISHART_1995_CA = {
    "ALA": 52.5,
    "ARG": 56.0,
    "ASN": 53.1,
    "ASP": 54.2,
    "CYS": 58.2,
    "GLN": 55.7,
    "GLU": 56.6,
    "GLY": 45.1,
    "HIS": 55.0,
    "ILE": 61.1,
    "LEU": 55.1,
    "LYS": 56.2,
    "MET": 55.3,
    "PHE": 57.7,
    "PRO": 63.3,
    "SER": 58.3,
    "THR": 61.8,
    "TRP": 57.5,
    "TYR": 57.9,
    "VAL": 62.2,
}


def test_all_20_amino_acids_present():
    """All canonical amino acids must have an entry."""
    missing = set(WISHART_1995_CA.keys()) - set(RANDOM_COIL_CA.keys())
    assert not missing, f"Missing amino acids in RANDOM_COIL_CA: {missing}"


@pytest.mark.parametrize("aa,expected", WISHART_1995_CA.items())
def test_random_coil_wishart_1995(aa, expected):
    """
    Spot-check each Cα random coil shift against Wishart et al. (1995).
    Tolerance: 0.05 ppm (sub-digitisation-error).
    """
    actual = RANDOM_COIL_CA[aa]
    np.testing.assert_allclose(
        actual, expected, atol=0.05,
        err_msg=f"{aa}: expected {expected} ppm (Wishart 1995), got {actual} ppm"
    )


if __name__ == "__main__":
    test_all_20_amino_acids_present()
    for aa, val in WISHART_1995_CA.items():
        test_random_coil_wishart_1995(aa, val)
    print("✅ All Wishart (1995) random coil Cα shifts verified!")
