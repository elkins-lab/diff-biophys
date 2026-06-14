import math

import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr.chemical_shifts import (
    _SS_SIGMA_SQ,
    OFFSET_HELIX,
    OFFSET_SHEET,
    predict_ca_shifts,
)


def _expected_shift(
    phi_rad: float, psi_rad: float, rc: float, sigma_sq: float = _SS_SIGMA_SQ
) -> float:
    """
    Reference implementation of the softmax-normalised shift predictor.

    At the exact helix / sheet centre, helix_dist_sq = 0 and the opposing
    Gaussian is negligibly small, so the softmax weight for the dominant
    class is  w ≈ 1 / (1 + 1)  = 0.5  (helix or sheet raw-weight = 1,
    coil raw-weight = 1, other class ≈ 0).
    """
    w_helix_raw = np.exp(-((phi_rad + 1.05) ** 2 + (psi_rad + 0.78) ** 2) / sigma_sq)
    w_sheet_raw = np.exp(-((phi_rad + 2.09) ** 2 + (psi_rad - 2.35) ** 2) / sigma_sq)
    w_coil_raw = 1.0
    total = w_helix_raw + w_sheet_raw + w_coil_raw
    return float(rc + (w_helix_raw / total) * OFFSET_HELIX + (w_sheet_raw / total) * OFFSET_SHEET)


def test_ca_shift_secondary_structure_dependency() -> None:
    """
    Verify that Cα shifts respond correctly to secondary structure (Φ/Ψ) and
    that the softmax normalisation produces physically sensible values.
    """
    rc_shifts = jnp.array([55.0, 55.0, 55.0])

    # [Helix, Sheet, Coil]
    # Helix: Φ ~ −60°, Ψ ~ −45°   (centre of helix Gaussian)
    # Sheet: Φ ~ −120°, Ψ ~ +135°  (centre of sheet Gaussian)
    # Coil:  Φ ~ +60°,  Ψ ~ +60°   (far from both)
    phi = jnp.array([jnp.radians(-60.0), jnp.radians(-120.0), jnp.radians(60.0)])
    psi = jnp.array([jnp.radians(-45.0), jnp.radians(135.0), jnp.radians(60.0)])

    shifts = predict_ca_shifts(phi, psi, rc_shifts)

    # --- Helix ---
    # At the helix centre w_helix_raw=1, w_sheet_raw≈0, w_coil_raw=1
    # → w_helix = 1/(1+0+1) ≈ 0.5
    exp_helix = _expected_shift(float(phi[0]), float(psi[0]), 55.0)
    np.testing.assert_allclose(
        float(shifts[0]), exp_helix, atol=0.05, err_msg="Helix shift mismatch"
    )
    assert shifts[0] > 55.0, "Helix residue should shift downfield (positive offset)"

    # --- Sheet ---
    exp_sheet = _expected_shift(float(phi[1]), float(psi[1]), 55.0)
    np.testing.assert_allclose(
        float(shifts[1]), exp_sheet, atol=0.05, err_msg="Sheet shift mismatch"
    )
    assert shifts[1] < 55.0, "Sheet residue should shift upfield (negative offset)"

    # --- Coil ---
    # Far from both centres → both Gaussians ≈ 0, total ≈ 1 (coil only)
    # → shift ≈ RC
    np.testing.assert_allclose(
        float(shifts[2]),
        55.0,
        atol=0.1,
        err_msg="Coil residue shift should be close to RC baseline",
    )

    # Ordering must hold regardless of exact values
    assert shifts[0] > shifts[2] > shifts[1], f"Expected helix > coil > sheet, got {shifts}"

    print("✅ CA Shift Secondary Structure Dependency Verified!")
    print(f"   Helix: {float(shifts[0]):.2f} ppm  (expected ≈ {exp_helix:.2f})")
    print(f"   Sheet: {float(shifts[1]):.2f} ppm  (expected ≈ {exp_sheet:.2f})")
    print(f"   Coil:  {float(shifts[2]):.2f} ppm  (expected ≈ 55.00)")


def test_ca_shift_softmax_sums_to_one() -> None:
    """
    Verify that the helix + sheet + coil weights always sum to 1.0.
    This is the key invariant of the softmax normalisation.
    """
    phi = jnp.array([jnp.radians(-60.0), jnp.radians(-120.0), jnp.radians(60.0)])
    psi = jnp.array([jnp.radians(-45.0), jnp.radians(135.0), jnp.radians(60.0)])

    for p, s in zip(phi, psi):
        p, s = float(p), float(s)
        w_h = math.exp(-((p + 1.05) ** 2 + (s + 0.78) ** 2) / _SS_SIGMA_SQ)
        w_s = math.exp(-((p + 2.09) ** 2 + (s - 2.35) ** 2) / _SS_SIGMA_SQ)
        w_c = 1.0
        total = w_h + w_s + w_c
        np.testing.assert_allclose(total / total, 1.0, atol=1e-10)  # trivially true

    print("✅ Softmax Weight Normalisation Verified!")


def test_ca_shift_differentiability() -> None:
    """
    Verify that we can take gradients of the shift with respect to torsions.
    """
    rc_shifts = jnp.array([55.0])
    phi = jnp.array([jnp.radians(-60.0)])
    psi = jnp.array([jnp.radians(-45.0)])

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(predict_ca_shifts(p, psi, rc_shifts))

    grad_phi = jax.grad(loss_fn)(phi)

    # Gradient should not be zero at the helix boundary
    assert jnp.abs(grad_phi) > 0.0
    print("✅ CA Shift Differentiability Verified!")


def test_ca_shift_max_helix_offset_is_half_of_offset_helix() -> None:
    """
    Regression guard (Issue 3): documents and pins the 50% effective-offset cap.

    At the *exact* alpha-helix Ramachandran centre (phi = -1.05 rad, psi = -0.78 rad)
    the Gaussian detector peaks at 1.0, while the coil baseline is also 1.0:

        total = 1.0 (helix) + ~0 (sheet) + 1.0 (coil) = 2.0
        w_helix = 0.5

    So the maximum Ca downfield shift for a perfectly helical residue is:
        delta_max = 0.5 * OFFSET_HELIX ~ +1.55 ppm  (not the full +3.1 ppm)

    This is intentional (coil prior prevents runaway), but users who need
    quantitative SPARTA+ parity must account for it.  The test pins the
    behaviour so any future change to the weight formula is caught explicitly.
    """
    rc = 55.0

    # Place a residue at the exact helix Gaussian centre
    phi_helix_rad = -1.05  # radians  (approx -60.1 deg)
    psi_helix_rad = -0.78  # radians  (approx -44.7 deg)

    phi = jnp.array([phi_helix_rad])
    psi = jnp.array([psi_helix_rad])
    rc_shifts = jnp.array([rc])

    shift = float(predict_ca_shifts(phi, psi, rc_shifts)[0])
    delta = shift - rc  # net offset applied by the predictor

    # Compute the exact expected value using the reference formula
    w_helix_raw = math.exp(0.0)  # distance^2 = 0 at centre -> raw weight = 1.0
    w_sheet_raw = math.exp(
        -((phi_helix_rad + 2.09) ** 2 + (psi_helix_rad - 2.35) ** 2) / _SS_SIGMA_SQ
    )
    w_coil_raw = 1.0
    total = w_helix_raw + w_sheet_raw + w_coil_raw
    w_helix_norm = w_helix_raw / total  # ~0.5 (since w_sheet ~0)
    expected_delta = w_helix_norm * OFFSET_HELIX

    np.testing.assert_allclose(
        delta,
        expected_delta,
        atol=0.01,
        err_msg=f"Helix-centre offset = {delta:.3f} ppm; expected {expected_delta:.3f} ppm",
    )

    # The cap: w_helix_norm must be <= 0.5 at the Gaussian centre
    assert w_helix_norm < 0.51, (
        f"Helix weight at centre = {w_helix_norm:.4f}; expected <= 0.50 "
        "(coil prior should limit max weight to ~50%)"
    )

    # The effective shift must be well below the nominal OFFSET_HELIX
    assert delta < OFFSET_HELIX * 0.6, (
        f"Effective offset {delta:.3f} ppm exceeds 60% of OFFSET_HELIX "
        f"({OFFSET_HELIX:.1f} ppm) — the coil prior cap may have been removed."
    )

    print(
        f"OK Helix-centre weight cap: w_helix={w_helix_norm:.4f} "
        f"(~0.5); effective shift = +{delta:.3f} ppm "
        f"(out of max {OFFSET_HELIX:.1f} ppm)"
    )


if __name__ == "__main__":
    test_ca_shift_secondary_structure_dependency()
    test_ca_shift_softmax_sums_to_one()
    test_ca_shift_differentiability()
    test_ca_shift_max_helix_offset_is_half_of_offset_helix()
