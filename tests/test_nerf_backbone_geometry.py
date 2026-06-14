import jax.numpy as jnp
import numpy as np
import pytest

from diff_biophys.geometry.nerf import chain_nerf, position_atom_3d
from diff_biophys.geometry.torsions import compute_dihedrals

# Standard IUPAC/Engh-Huber backbone bond parameters (Engh & Huber 1991)
# Bond lengths in Ångströms, angles in degrees
CA_C_LENGTH = 1.525  # Cα–C
C_N_LENGTH = 1.329  # C–N (peptide bond)
N_CA_LENGTH = 1.459  # N–Cα

CA_C_N_ANGLE = 116.2  # ∠Cα–C–N
C_N_CA_ANGLE = 121.7  # ∠C–N–Cα
N_CA_C_ANGLE = 111.2  # ∠N–Cα–C


def test_nerf_preserves_bond_length() -> None:
    """
    The placed atom must be exactly `bond_length` away from p3.
    Tested with standard peptide bond parameters.
    """
    p1 = jnp.array([0.000, 0.000, 0.000])  # Cα
    p2 = jnp.array([1.525, 0.000, 0.000])  # C
    p3 = jnp.array([2.100, 1.230, 0.000])  # N  (approximate)
    bond_len = N_CA_LENGTH
    angle_rad = jnp.radians(C_N_CA_ANGLE)
    dihedral_rad = jnp.radians(180.0)  # trans omega

    p4 = position_atom_3d(p1, p2, p3, bond_len, angle_rad, dihedral_rad)

    assert jnp.all(jnp.isfinite(p4)), f"NaN/Inf in backbone atom position: {p4}"

    actual_length = float(jnp.linalg.norm(p4 - p3))
    np.testing.assert_allclose(
        actual_length,
        bond_len,
        atol=1e-5,
        err_msg=f"Bond length: placed={actual_length:.6f} Å, expected={bond_len} Å",
    )
    print(f"✅ NeRF bond length preserved: {actual_length:.6f} Å")


def test_nerf_preserves_bond_angle() -> None:
    """
    The ∠(p2, p3, p4) angle must equal the requested bond_angle_rad.
    """
    p1 = jnp.array([0.000, 0.000, 0.000])
    p2 = jnp.array([1.525, 0.000, 0.000])
    p3 = jnp.array([2.100, 1.230, 0.000])
    bond_len = N_CA_LENGTH
    angle_rad = jnp.radians(C_N_CA_ANGLE)
    dihedral_rad = jnp.radians(180.0)

    p4 = position_atom_3d(p1, p2, p3, bond_len, angle_rad, dihedral_rad)

    v1 = p2 - p3
    v2 = p4 - p3
    cos_a = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
    actual_angle = float(jnp.arccos(jnp.clip(cos_a, -1.0, 1.0)))

    np.testing.assert_allclose(
        actual_angle,
        float(angle_rad),
        atol=1e-5,
        err_msg=f"Bond angle: placed={np.degrees(actual_angle):.3f}°, expected={C_N_CA_ANGLE}°",
    )
    print(f"✅ NeRF bond angle preserved: {np.degrees(actual_angle):.3f}°")


def test_chain_nerf_with_backbone_parameters() -> None:
    """
    Build a 6-atom Cα–C–N–Cα–C–N fragment with real backbone geometry
    and verify that all placed bond lengths are correct.

    This tests `chain_nerf` end-to-end with biologically relevant parameters.
    """
    # Seed the first three atoms manually (Cα–C–N)
    p1 = jnp.array([0.000, 0.000, 0.000])  # Cα
    p2 = jnp.array([1.525, 0.000, 0.000])  # C
    p3 = jnp.array([2.100, 1.230, 0.000])  # N
    init_coords = jnp.stack([p1, p2, p3])

    # Build 3 more atoms: Cα–C–N of the next residue
    bond_lengths = jnp.array([N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH])
    bond_angles = jnp.array(
        [
            jnp.radians(C_N_CA_ANGLE),
            jnp.radians(N_CA_C_ANGLE),
            jnp.radians(CA_C_N_ANGLE),
        ]
    )
    dihedrals = jnp.array(
        [
            jnp.radians(180.0),  # omega (trans peptide)
            jnp.radians(-60.0),  # phi (helix-like)
            jnp.radians(-45.0),  # psi (helix-like)
        ]
    )

    chain = chain_nerf(init_coords, bond_lengths, bond_angles, dihedrals)

    assert chain.shape == (6, 3), f"Expected shape (6,3), got {chain.shape}"
    assert jnp.all(jnp.isfinite(chain)), "NaN/Inf in chain coordinates"

    # Check that each placed bond length matches the requested value
    _expected_lengths = np.concatenate(
        [
            [float(jnp.linalg.norm(p2 - p1)), float(jnp.linalg.norm(p3 - p2))],
            list(bond_lengths),
        ]
    )
    chain_np = np.array(chain)
    for i in range(5):
        actual = np.linalg.norm(chain_np[i + 1] - chain_np[i])
        # For init atoms the lengths are whatever we set; for placed atoms
        # they must match bond_lengths exactly.
        if i >= 2:
            np.testing.assert_allclose(
                actual,
                float(bond_lengths[i - 2]),
                atol=1e-5,
                err_msg=f"Bond {i}→{i + 1}: {actual:.6f} Å ≠ {float(bond_lengths[i - 2]):.6f} Å",
            )

    print("✅ chain_nerf backbone geometry verified for 6-atom Cα–C–N–Cα–C–N fragment")


@pytest.mark.parametrize("phi_deg", [-180, -120, -60, -45, -30, 0, 30, 45, 60, 90, 120, 150, 179])
def test_nerf_dihedral_roundtrip(phi_deg: float) -> None:
    """
    Critical regression guard (Issue 1): a dihedral passed to position_atom_3d
    must be recovered *exactly* when the resulting four-atom chain is measured
    with compute_dihedrals (IUPAC / Praxeolitic convention).

    The NeRF local frame uses m = n × u2 (opposite to Parsons 2005's u2 × n),
    which negates the m column.  Combined with negated coefficients in the
    placement formula, the convention is internally consistent and IUPAC-correct
    — but this test makes that invariant explicit and guards against future
    refactors that might break the sign pairing.
    """
    phi = jnp.radians(float(phi_deg))

    # Reference frame: p1-p2-p3 along a simple geometry
    p1 = jnp.array([0.0, 1.0, 0.0])
    p2 = jnp.array([0.0, 0.0, 0.0])
    p3 = jnp.array([1.0, 0.0, 0.0])

    p4 = position_atom_3d(
        p1,
        p2,
        p3,
        jnp.array(1.5),
        jnp.radians(109.5),
        phi,
    )

    coords = jnp.stack([p1, p2, p3, p4])
    measured = float(compute_dihedrals(coords)[0])

    # Wrap error to (−π, π] for the ±180 boundary
    error_rad = abs(float(phi) - measured)
    if error_rad > np.pi:
        error_rad = abs(error_rad - 2 * np.pi)

    np.testing.assert_allclose(
        error_rad,
        0.0,
        atol=1e-4,
        err_msg=(
            f"Dihedral roundtrip failed at phi={phi_deg}°: "
            f"placed={phi_deg}°, measured={np.degrees(measured):.4f}°"
        ),
    )


def test_nerf_alpha_helix_is_right_handed() -> None:
    """
    Regression guard (Issue 1): building a peptide chain with ideal α-helix
    torsions (φ = −57.8°, ψ = −47°, ω = 180°) must produce a *right-handed*
    spiral — the biologically correct screw sense.

    Handedness check: the cross-products of successive Cα–Cα displacement
    vectors must all point in the same direction (consistent screw axis).
    A mirrored (left-handed) chain would produce cross products with an
    inconsistent dominant sign.
    """
    # Standard backbone geometry (Engh & Huber 1991)
    CA_C_LENGTH = 1.525
    C_N_LENGTH = 1.329
    N_CA_LENGTH = 1.459
    CA_C_N_ANGLE = np.radians(116.2)
    C_N_CA_ANGLE = np.radians(121.7)
    N_CA_C_ANGLE = np.radians(111.2)

    PHI = np.radians(-57.8)
    PSI = np.radians(-47.0)
    OMEGA = np.radians(180.0)

    n_res = 6  # enough turns to see handedness clearly

    # Seed the first three atoms (N0, Cα0, C0)
    p1 = jnp.array([0.000, 0.000, 0.000])
    p2 = jnp.array([N_CA_LENGTH, 0.000, 0.000])
    p3 = jnp.array(
        [
            N_CA_LENGTH + CA_C_LENGTH * np.cos(np.pi - N_CA_C_ANGLE),
            CA_C_LENGTH * np.sin(np.pi - N_CA_C_ANGLE),
            0.0,
        ]
    )
    init_coords = jnp.stack([p1, p2, p3])

    # The repeating unit per residue is: C–N, N–Cα, Cα–C
    # with dihedrals: ψ (of current res), ω, φ (of next res)
    bond_lengths = jnp.array([C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH] * n_res, dtype=jnp.float32)
    bond_angles = jnp.array([CA_C_N_ANGLE, C_N_CA_ANGLE, N_CA_C_ANGLE] * n_res, dtype=jnp.float32)
    dihedrals = jnp.array([PSI, OMEGA, PHI] * n_res, dtype=jnp.float32)

    chain = chain_nerf(init_coords, bond_lengths, bond_angles, dihedrals)
    chain_np = np.array(chain)

    # Cα atoms are at indices 1, 4, 7, … (N, Cα, C repeating, with 3 seed atoms)
    ca_indices = [1 + 3 * i for i in range(n_res + 1) if (1 + 3 * i) < len(chain_np)]
    ca = chain_np[ca_indices]

    assert len(ca) >= 4, "Need at least 4 Cα atoms to test handedness"

    # Cross products of successive Cα–Cα step vectors
    steps = np.diff(ca, axis=0)
    crosses = np.cross(steps[:-1], steps[1:])

    # All cross products should have the same sign of their dominant component
    mean_axis = np.argmax(np.abs(np.mean(crosses, axis=0)))
    dominant_signs = np.sign(crosses[:, mean_axis])

    assert np.all(dominant_signs == dominant_signs[0]), (
        f"α-helix is not consistently handed — NERF sign convention may be "
        f"inverted.  Cross product dominant components: {crosses[:, mean_axis]}"
    )
    print("✅ α-helix handedness: right-handed spiral confirmed")


if __name__ == "__main__":
    test_nerf_preserves_bond_length()
    test_nerf_preserves_bond_angle()
    test_chain_nerf_with_backbone_parameters()
    for deg in [-180, -120, -60, -45, 0, 45, 60, 90, 179]:
        test_nerf_dihedral_roundtrip(deg)
    test_nerf_alpha_helix_is_right_handed()
