import numpy as np
import jax.numpy as jnp
from diff_biophys.geometry.nerf import position_atom_3d, chain_nerf


# Standard IUPAC/Engh-Huber backbone bond parameters (Engh & Huber 1991)
# Bond lengths in Ångströms, angles in degrees
CA_C_LENGTH  = 1.525   # Cα–C
C_N_LENGTH   = 1.329   # C–N (peptide bond)
N_CA_LENGTH  = 1.459   # N–Cα

CA_C_N_ANGLE  = 116.2  # ∠Cα–C–N
C_N_CA_ANGLE  = 121.7  # ∠C–N–Cα
N_CA_C_ANGLE  = 111.2  # ∠N–Cα–C


def test_nerf_preserves_bond_length():
    """
    The placed atom must be exactly `bond_length` away from p3.
    Tested with standard peptide bond parameters.
    """
    p1 = jnp.array([0.000, 0.000, 0.000])   # Cα
    p2 = jnp.array([1.525, 0.000, 0.000])   # C
    p3 = jnp.array([2.100, 1.230, 0.000])   # N  (approximate)
    bond_len    = N_CA_LENGTH
    angle_rad   = jnp.radians(C_N_CA_ANGLE)
    dihedral_rad = jnp.radians(180.0)        # trans omega

    p4 = position_atom_3d(p1, p2, p3, bond_len, angle_rad, dihedral_rad)

    assert jnp.all(jnp.isfinite(p4)), f"NaN/Inf in backbone atom position: {p4}"

    actual_length = float(jnp.linalg.norm(p4 - p3))
    np.testing.assert_allclose(actual_length, bond_len, atol=1e-5,
                               err_msg=f"Bond length: placed={actual_length:.6f} Å, expected={bond_len} Å")
    print(f"✅ NeRF bond length preserved: {actual_length:.6f} Å")


def test_nerf_preserves_bond_angle():
    """
    The ∠(p2, p3, p4) angle must equal the requested bond_angle_rad.
    """
    p1 = jnp.array([0.000, 0.000, 0.000])
    p2 = jnp.array([1.525, 0.000, 0.000])
    p3 = jnp.array([2.100, 1.230, 0.000])
    bond_len     = N_CA_LENGTH
    angle_rad    = jnp.radians(C_N_CA_ANGLE)
    dihedral_rad = jnp.radians(180.0)

    p4 = position_atom_3d(p1, p2, p3, bond_len, angle_rad, dihedral_rad)

    v1 = p2 - p3
    v2 = p4 - p3
    cos_a = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
    actual_angle = float(jnp.acos(jnp.clip(cos_a, -1.0, 1.0)))

    np.testing.assert_allclose(actual_angle, float(angle_rad), atol=1e-5,
                               err_msg=f"Bond angle: placed={np.degrees(actual_angle):.3f}°, "
                                       f"expected={C_N_CA_ANGLE}°")
    print(f"✅ NeRF bond angle preserved: {np.degrees(actual_angle):.3f}°")


def test_chain_nerf_with_backbone_parameters():
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
    bond_angles  = jnp.array([
        jnp.radians(C_N_CA_ANGLE),
        jnp.radians(N_CA_C_ANGLE),
        jnp.radians(CA_C_N_ANGLE),
    ])
    dihedrals = jnp.array([
        jnp.radians(180.0),   # omega (trans peptide)
        jnp.radians(-60.0),   # phi (helix-like)
        jnp.radians(-45.0),   # psi (helix-like)
    ])

    chain = chain_nerf(init_coords, bond_lengths, bond_angles, dihedrals)

    assert chain.shape == (6, 3), f"Expected shape (6,3), got {chain.shape}"
    assert jnp.all(jnp.isfinite(chain)), "NaN/Inf in chain coordinates"

    # Check that each placed bond length matches the requested value
    expected_lengths = np.concatenate([
        [float(jnp.linalg.norm(p2 - p1)),
         float(jnp.linalg.norm(p3 - p2))],
        list(bond_lengths),
    ])
    chain_np = np.array(chain)
    for i in range(5):
        actual = np.linalg.norm(chain_np[i+1] - chain_np[i])
        # For init atoms the lengths are whatever we set; for placed atoms
        # they must match bond_lengths exactly.
        if i >= 2:
            np.testing.assert_allclose(actual, float(bond_lengths[i-2]), atol=1e-5,
                                       err_msg=f"Bond {i}→{i+1}: {actual:.6f} Å ≠ {float(bond_lengths[i-2]):.6f} Å")

    print(f"✅ chain_nerf backbone geometry verified for 6-atom Cα–C–N–Cα–C–N fragment")


if __name__ == "__main__":
    test_nerf_preserves_bond_length()
    test_nerf_preserves_bond_angle()
    test_chain_nerf_with_backbone_parameters()
