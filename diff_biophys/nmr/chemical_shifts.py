import jax.numpy as jnp
from jax import jit, vmap

# Baseline Random Coil Shifts (Wishart et al. 1995, J. Biomol. NMR 5, 67–81)
# Cα chemical shifts in ppm (referenced to DSS)
RANDOM_COIL_CA = {
    "ALA": 52.5, "ARG": 56.0, "ASN": 53.1, "ASP": 54.2, "CYS": 58.2,
    "GLN": 55.7, "GLU": 56.6, "GLY": 45.1, "HIS": 55.0, "ILE": 61.1,
    "LEU": 55.1, "LYS": 56.2, "MET": 55.3, "PHE": 57.7, "PRO": 63.3,
    "SER": 58.3, "THR": 61.8, "TRP": 57.5, "TYR": 57.9, "VAL": 62.2
}

# Statistical Secondary Structure Offsets for Cα (SPARTA / SPARTA+ convention)
# Alpha Helix: ~ +3.1 ppm, Beta Sheet: ~ -1.5 ppm
OFFSET_HELIX = 3.1
OFFSET_SHEET = -1.5

# Width (σ²) of the Gaussian secondary-structure detectors (radians²).
# This controls the "softness" of the helix/sheet classification.
# At σ²=0.5, a residue 0.7 rad (~40°) from the helix center gets ~37% weight.
_SS_SIGMA_SQ = 0.5

@jit
def predict_ca_shifts(phi: jnp.ndarray, psi: jnp.ndarray, rc_shifts: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiable Cα chemical shift prediction based on backbone torsions.

    Uses Gaussian "soft detectors" in (Φ, Ψ) space to classify secondary
    structure and applies SPARTA-like offsets.  The detectors are normalised
    via a softmax so that helix, sheet, and coil contributions always sum to
    1.0, preventing unphysical double-counting.

    Reference centres (radians):
        * α-helix: Φ = −1.05 rad (−60°), Ψ = −0.78 rad (−45°)
        * β-sheet:  Φ = −2.09 rad (−120°), Ψ = +2.35 rad (+135°)
        * Random coil: treated as the baseline (weight = 1 − w_helix − w_sheet)

    Args:
        phi: (N,) backbone Φ angles in radians.
        psi: (N,) backbone Ψ angles in radians.
        rc_shifts: (N,) baseline random-coil Cα shifts (ppm).

    Returns:
        jnp.ndarray: (N,) predicted Cα chemical shifts (ppm).
    """
    # --- Unnormalised Gaussian affinities ---
    # Alpha-helix centre: Φ ~ −60°, Ψ ~ −45°
    helix_dist_sq = (phi + 1.05)**2 + (psi + 0.78)**2
    w_helix_raw = jnp.exp(-helix_dist_sq / _SS_SIGMA_SQ)

    # Beta-sheet centre: Φ ~ −120°, Ψ ~ +135°
    sheet_dist_sq = (phi + 2.09)**2 + (psi - 2.35)**2
    w_sheet_raw = jnp.exp(-sheet_dist_sq / _SS_SIGMA_SQ)

    # Coil baseline: all residues start with weight 1 (i.e. the neutral state)
    w_coil_raw = jnp.ones_like(phi)

    # --- Softmax normalisation ---
    # Ensures w_helix + w_sheet + w_coil = 1 for every residue,
    # preventing simultaneous helix + sheet double-counting.
    total = w_helix_raw + w_sheet_raw + w_coil_raw
    w_helix = w_helix_raw / total
    w_sheet = w_sheet_raw / total
    # w_coil = w_coil_raw / total  (implicit; contributes zero offset)

    # --- Weighted offset ---
    # Coil weight contributes 0 offset (it is the RC baseline).
    return rc_shifts + (w_helix * OFFSET_HELIX) + (w_sheet * OFFSET_SHEET)
