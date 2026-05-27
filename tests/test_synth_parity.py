import jax.numpy as jnp
import numpy as np
import pytest

# Optional dependencies for parity benchmarking
synth_nerf = pytest.importorskip("synth_pdb.geometry.nerf")
synth_sup = pytest.importorskip("synth_pdb.geometry.superposition")
synth_j = pytest.importorskip("synth_nmr.j_coupling")
synth_geom = pytest.importorskip("synth_pdb.geometry")

from diff_biophys.geometry import (  # noqa: E402
    compute_bond_lengths,
    compute_dihedrals,
    kabsch_alignment,
    position_atom_3d,
)
from diff_biophys.nmr import calculate_karplus_j, calculate_rdc  # noqa: E402


def test_nerf_synth_parity():
    """Verify parity with synth-pdb's NeRF implementation."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p3 = np.array([2.0, 1.4, 0.0])
    bond_len = 1.33
    angle_deg = 116.0
    dihedral_deg = 180.0

    # synth-pdb version (NumPy)
    expected = synth_nerf.position_atom_3d_from_internal_coords(
        p1, p2, p3, bond_len, angle_deg, dihedral_deg
    )

    # diff-biophys version (JAX)
    # Note: diff-biophys takes radians
    actual = position_atom_3d(
        jnp.array(p1),
        jnp.array(p2),
        jnp.array(p3),
        bond_len,
        jnp.radians(angle_deg),
        jnp.radians(dihedral_deg),
    )

    np.testing.assert_allclose(expected, np.array(actual), atol=1e-5)
    print("✅ NeRF Synth-PDB Parity Verified!")


def test_kabsch_synth_parity():
    """Verify parity with synth-pdb's Kabsch implementation."""
    P = np.random.randn(10, 3).astype(np.float64)
    Q = np.random.randn(10, 3).astype(np.float64)

    # synth-pdb version
    R_expected, t_expected = synth_sup.kabsch_superposition(P, Q)

    # diff-biophys version
    R_actual, t_actual = kabsch_alignment(jnp.array(P), jnp.array(Q))

    np.testing.assert_allclose(R_expected, np.array(R_actual), atol=1e-5)
    np.testing.assert_allclose(t_expected, np.array(t_actual), atol=1e-5)
    print("✅ Kabsch Synth-PDB Parity Verified!")


def test_torsion_synth_parity():
    """Verify parity with synth-pdb's torsion implementations."""
    coords = np.random.randn(10, 3)

    # 1. Bond Lengths
    expected_lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    actual_lengths = compute_bond_lengths(jnp.array(coords))
    np.testing.assert_allclose(expected_lengths, np.array(actual_lengths), atol=1e-5)

    # 2. Dihedrals
    # synth-pdb uses degrees, we use radians
    for i in range(len(coords) - 3):
        p1, p2, p3, p4 = coords[i : i + 4]
        expected_deg = synth_geom.calculate_dihedral(p1, p2, p3, p4)
        # Our compute_dihedrals takes a full chain
        actual_rad = compute_dihedrals(jnp.array(coords))
        np.testing.assert_allclose(np.deg2rad(expected_deg), float(actual_rad[i]), atol=1e-5)

    print("✅ Torsion Synth-PDB Parity Verified!")


def test_rdc_synth_parity():
    """Verify parity with synth-nmr's RDC implementation."""
    # RDC formula in synth-nmr:
    # D = Da * [ (3*cos^2(theta) - 1) + (3/2) * R * sin^2(theta) * cos(2*phi) ]
    # This is exactly what we have in diff_biophys.nmr.calculate_rdc

    Da, R = 15.0, 0.3
    vectors = np.random.randn(5, 3)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    # diff-biophys JAX version
    actual = calculate_rdc(jnp.array(vectors), Da, R)

    # synth-nmr uses a higher-level API with Biotite structures,
    # so we'll re-implement their core logic here to verify our kernel math.
    def synth_rdc_core(v, Da, R):
        x, y, z = v
        cos_theta = z
        sin_theta_sq = 1 - cos_theta**2
        if sin_theta_sq < 1e-9:
            cos_2phi = 1.0
        else:
            cos_2phi = (x**2 - y**2) / sin_theta_sq
        return Da * ((3 * cos_theta**2 - 1) + 1.5 * R * sin_theta_sq * cos_2phi)

    expected = np.array([synth_rdc_core(v, Da, R) for v in vectors])

    np.testing.assert_allclose(expected, np.array(actual), atol=1e-5)
    print("✅ RDC Synth-NMR Parity Verified!")


def test_karplus_synth_parity():
    """Verify parity with synth-nmr's Karplus implementation."""
    phi_deg = np.array([-60.0, -120.0, 0.0, 180.0])

    # synth-nmr version
    expected = synth_j.calculate_hn_ha_coupling_from_phi(phi_deg)

    # diff-biophys version
    # Params: A=6.51, B=-1.76, C=1.60
    # synth-nmr uses theta = phi - 60
    theta_rad = jnp.radians(phi_deg - 60.0)
    actual = calculate_karplus_j(theta_rad, 6.51, -1.76, 1.60)

    np.testing.assert_allclose(expected, np.array(actual), atol=1e-5)
    print("✅ Karplus Synth-NMR Parity Verified!")


if __name__ == "__main__":
    test_nerf_synth_parity()
    test_kabsch_synth_parity()
    test_torsion_synth_parity()
    test_rdc_synth_parity()
    test_karplus_synth_parity()
