from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import jit

# Baseline Random Coil Shifts (Wishart et al. 1995, J. Biomol. NMR 5, 67–81)
# Cα chemical shifts in ppm (referenced to DSS)
RANDOM_COIL_CA = {
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
    helix_dist_sq = (phi + 1.05) ** 2 + (psi + 0.78) ** 2
    w_helix_raw = jnp.exp(-helix_dist_sq / _SS_SIGMA_SQ)

    # Beta-sheet centre: Φ ~ −120°, Ψ ~ +135°
    sheet_dist_sq = (phi + 2.09) ** 2 + (psi - 2.35) ** 2
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

    # NOTE — effective offset cap (Issue 3):
    # Because w_coil_raw is fixed at 1.0, at the *exact* helix or sheet centre
    # (where the Gaussian peak = 1 and the opposing class ≈ 0) the denominator
    # is 1 + 0 + 1 = 2, so w_helix_norm ≈ 0.5.  This means the maximum Cα
    # shift a perfectly helical residue receives is:
    #
    #   0.5 × OFFSET_HELIX  ≈  +1.55 ppm  (not the full +3.1 ppm)
    #
    # This is a deliberate approximation: the coil baseline acts as a Bayesian
    # prior that prevents runaway shifts.  It also means the predictor underestimates
    # pure-helix / pure-sheet shifts by ~50% relative to SPARTA+.  Users who need
    # quantitative SPARTA+ parity should either:
    #   (a) reduce _SS_SIGMA_SQ (sharper Gaussians → w_helix closer to 1), or
    #   (b) double OFFSET_HELIX / OFFSET_SHEET to compensate.

    # --- Weighted offset ---
    # Coil weight contributes 0 offset (it is the RC baseline).
    return rc_shifts + (w_helix * OFFSET_HELIX) + (w_sheet * OFFSET_SHEET)


# ---------------------------------------------------------------------------
# Loss builder
# ---------------------------------------------------------------------------


def make_ca_shift_loss(
    exp_res_ids: np.ndarray,
    exp_shifts: np.ndarray,
    struct_res_ids: np.ndarray,
    struct_res_names: list[str],
) -> tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], int]:
    """Build a differentiable Cα chemical shift RMSD loss.

    Matches experimental residues to the structure by residue ID and builds a
    JAX-differentiable closure that predicts Cα shifts from backbone torsions
    and returns the RMSD against the matched experimental values.

    Args:
        exp_res_ids: ``(M,)`` residue IDs from the experimental dataset.
        exp_shifts: ``(M,)`` experimental Cα chemical shifts in ppm.
        struct_res_ids: ``(N,)`` residue IDs present in the structure.
        struct_res_names: List of N three-letter residue codes, aligned with
            ``struct_res_ids``.

    Returns:
        Tuple ``(loss_fn, n_matched)`` where:

        * **loss_fn** ``(phi, psi) → scalar RMSD (ppm)`` — differentiable with
          respect to both torsion arrays.
        * **n_matched** — number of residues found in both datasets.

    Raises:
        ValueError: If no residues overlap between the experimental and
            structure datasets.
    """
    res_id_to_idx = {int(rid): i for i, rid in enumerate(struct_res_ids)}

    matched_struct_idx: list[int] = []
    matched_exp: list[float] = []
    matched_names: list[str] = []

    for rid, shift in zip(exp_res_ids, exp_shifts, strict=False):
        if int(rid) in res_id_to_idx:
            si = res_id_to_idx[int(rid)]
            matched_struct_idx.append(si)
            matched_exp.append(float(shift))
            matched_names.append(struct_res_names[si])

    if not matched_struct_idx:
        raise ValueError(
            "No residues overlap between exp_res_ids and struct_res_ids. "
            "Check that residue numbering conventions match."
        )

    rc = np.array([RANDOM_COIL_CA.get(name, 55.0) for name in matched_names], dtype=np.float32)
    rc_jax = jnp.array(rc)
    exp_jax = jnp.array(matched_exp, dtype=jnp.float32)
    idx_jax = jnp.array(matched_struct_idx, dtype=jnp.int32)
    n_matched = len(matched_struct_idx)

    def loss_fn(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
        """Cα shift RMSD loss.

        Args:
            phi: ``(N,)`` backbone φ angles in radians.
            psi: ``(N,)`` backbone ψ angles in radians.

        Returns:
            jnp.ndarray: Scalar RMSD in ppm.
        """
        pred = predict_ca_shifts(phi[idx_jax], psi[idx_jax], rc_jax)
        return jnp.sqrt(jnp.mean((pred - exp_jax) ** 2))

    return loss_fn, n_matched
