import jax.numpy as jnp
from jax import jit, vmap

# Baseline Random Coil Shifts (Wishart et al. 1995)
# Using a simplified set for CA (Alpha Carbon) for demonstration
RANDOM_COIL_CA = {
    "ALA": 52.5, "ARG": 56.0, "ASN": 53.1, "ASP": 54.2, "CYS": 58.2,
    "GLN": 55.7, "GLU": 56.6, "GLY": 45.1, "HIS": 55.0, "ILE": 61.1,
    "LEU": 55.1, "LYS": 56.2, "MET": 55.3, "PHE": 57.7, "PRO": 63.3,
    "SER": 58.3, "THR": 61.8, "TRP": 57.5, "TYR": 57.9, "VAL": 62.2
}

# Statistical Secondary Structure Offsets for CA
# Alpha Helix: ~ +3.1 ppm, Beta Sheet: ~ -1.5 ppm
OFFSET_HELIX = 3.1
OFFSET_SHEET = -1.5

@jit
def predict_ca_shifts(phi: jnp.ndarray, psi: jnp.ndarray, rc_shifts: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiable CA Chemical Shift prediction based on Backbone Torsions.
    
    This uses a "soft" classification of secondary structure based on Phi/Psi
    to apply SPARTA-like offsets.
    
    Args:
        phi, psi: (N,) backbone dihedrals in radians.
        rc_shifts: (N,) baseline random coil shifts.
        
    Returns:
        jnp.ndarray: (N,) predicted CA shifts.
    """
    # 1. Soft-classify secondary structure
    # Alpha Helix: Phi ~ -60 deg (-1.05 rad), Psi ~ -45 deg (-0.78 rad)
    helix_dist_sq = (phi + 1.05)**2 + (psi + 0.78)**2
    is_helix = jnp.exp(-helix_dist_sq / 0.5) # Soft mask
    
    # Beta Sheet: Phi ~ -120 deg (-2.09 rad), Psi ~ 135 deg (2.35 rad)
    sheet_dist_sq = (phi + 2.09)**2 + (psi - 2.35)**2
    is_sheet = jnp.exp(-sheet_dist_sq / 0.5) # Soft mask
    
    # 2. Combine offsets
    # Shift = RC + (is_helix * OFFSET_HELIX) + (is_sheet * OFFSET_SHEET)
    return rc_shifts + (is_helix * OFFSET_HELIX) + (is_sheet * OFFSET_SHEET)
