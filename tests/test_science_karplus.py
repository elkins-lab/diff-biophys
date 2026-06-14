import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr import calculate_karplus_j

# Parameters from Vuister & Bax (1993), JACS 115, 7772-7777
A, B, C = 6.51, -1.76, 1.60


def test_bax_1993_karplus_validation() -> None:
    """
    Validate against Vuister & Bax (1993) JACS parameters for 3J(HN-HA).
    Reference: J. Am. Chem. Soc. 115, 7772-7777.
    """
    # 1. Alpha-helix check: phi ~ -60 deg -> theta = phi - 60 = -120 deg
    phi_helix = jnp.radians(-60.0)
    theta_helix = phi_helix - jnp.radians(60.0)
    j_helix = calculate_karplus_j(theta_helix, A, B, C)

    # Expected range for alpha-helix: 3.0 - 6.0 Hz
    assert 3.0 <= float(j_helix) <= 6.0
    print(f"✅ Karplus Alpha-Helix (phi=-60): {float(j_helix):.2f} Hz (in range 3-6)")

    # 2. Beta-sheet check: phi ~ -120 deg -> theta = phi - 60 = -180 deg
    phi_sheet = jnp.radians(-120.0)
    theta_sheet = phi_sheet - jnp.radians(60.0)
    j_sheet = calculate_karplus_j(theta_sheet, A, B, C)

    # Expected range for beta-sheet: 8.0 - 10.0 Hz
    assert 8.0 <= float(j_sheet) <= 10.0
    print(f"✅ Karplus Beta-Sheet (phi=-120): {float(j_sheet):.2f} Hz (in range 8-10)")


def test_karplus_b_parameter_sign() -> None:
    """
    Verify that the B parameter sign is correctly implemented.

    The B term (−1.76) creates an asymmetry: J is NOT equal at +θ and −θ.
    At θ=0°:   J = A + B + C = 6.51 − 1.76 + 1.60 = 6.35 Hz
    At θ=180°: J = A − B + C = 6.51 + 1.76 + 1.60 = 9.87 Hz

    A sign error in B would swap these values.
    """
    j_0 = float(calculate_karplus_j(jnp.array(0.0), A, B, C))
    j_180 = float(calculate_karplus_j(jnp.array(jnp.pi), A, B, C))

    np.testing.assert_allclose(j_0, 6.35, atol=0.01, err_msg="J(0°) mismatch — B sign may be wrong")
    np.testing.assert_allclose(
        j_180, 9.87, atol=0.01, err_msg="J(180°) mismatch — B sign may be wrong"
    )

    # J(θ=90°) = C only (cos²=0, cos=0)
    j_90 = float(calculate_karplus_j(jnp.array(jnp.pi / 2), A, B, C))
    np.testing.assert_allclose(j_90, C, atol=1e-5, err_msg="J(90°) should equal C")

    # With a negative B, J(180°) must be larger than J(0°)
    assert j_180 > j_0, "With B<0, J(180°) should exceed J(0°)"

    print(f"✅ Karplus B-parameter sign correct: J(0°)={j_0:.2f} Hz, J(180°)={j_180:.2f} Hz")


def test_karplus_requires_phi_offset_not_raw_phi() -> None:
    """
    Regression guard (Issue 2): the function accepts theta = phi - 60°, not
    raw phi.  This test quantifies the error from the common mistake of passing
    phi directly and confirms that the pre-offset theta yields correct values.

    For 3J(HN,HA) with Vuister & Bax 1993 parameters:
      - Alpha-helix (phi = -60°) → theta = -120° → J ≈ 3.9 Hz   (in-range)
      - Beta-sheet  (phi = -120°) → theta = -180° → J ≈ 9.87 Hz (in-range)

    Mistakenly passing raw phi:
      - phi = -60°  → J ≈ 5.5 Hz  (off by ~1.6 Hz from the correct 3.9 Hz)
      - phi = -120° → J ≈ 7.5 Hz  (off by ~2.4 Hz from the correct 9.87 Hz)

    These errors (>1 Hz) are large enough to misclassify secondary structure.
    """
    A, B, C = 6.51, -1.76, 1.60

    # --- Correct usage: theta = phi - 60° ---
    phi_helix = jnp.radians(-60.0)
    theta_helix = phi_helix - jnp.radians(60.0)
    j_helix_correct = float(calculate_karplus_j(theta_helix, A, B, C))

    phi_sheet = jnp.radians(-120.0)
    theta_sheet = phi_sheet - jnp.radians(60.0)
    j_sheet_correct = float(calculate_karplus_j(theta_sheet, A, B, C))

    assert 3.0 <= j_helix_correct <= 6.0, f"Correct helix J = {j_helix_correct:.2f} Hz out of range"
    assert 8.0 <= j_sheet_correct <= 10.5, (
        f"Correct sheet J = {j_sheet_correct:.2f} Hz out of range"
    )

    # --- Incorrect usage: raw phi (the common mistake) ---
    j_helix_wrong = float(calculate_karplus_j(phi_helix, A, B, C))
    j_sheet_wrong = float(calculate_karplus_j(phi_sheet, A, B, C))

    # The error from passing raw phi must be detectable (>1 Hz off)
    helix_error = abs(j_helix_correct - j_helix_wrong)
    sheet_error = abs(j_sheet_correct - j_sheet_wrong)
    assert helix_error > 1.0, (
        f"Helix: expected >1 Hz error from passing raw phi, got {helix_error:.3f} Hz. "
        "The theta = phi - 60° offset may have been baked into the function."
    )
    assert sheet_error > 1.0, (
        f"Sheet: expected >1 Hz error from passing raw phi, got {sheet_error:.3f} Hz."
    )

    print(
        f"✅ Karplus offset guard: "
        f"helix J(theta)={j_helix_correct:.2f} Hz vs J(phi)={j_helix_wrong:.2f} Hz "
        f"(Δ={helix_error:.2f} Hz); "
        f"sheet J(theta)={j_sheet_correct:.2f} Hz vs J(phi)={j_sheet_wrong:.2f} Hz "
        f"(Δ={sheet_error:.2f} Hz)"
    )


if __name__ == "__main__":
    test_bax_1993_karplus_validation()
    test_karplus_b_parameter_sign()
    test_karplus_requires_phi_offset_not_raw_phi()
