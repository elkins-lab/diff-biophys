import jax.numpy as jnp
import numpy as np

from diff_biophys.geometry.nerf import position_atom_3d


def test_nerf_parity():
    # Test Data: Simple zigzag in XY plane
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, 1.0, 0.0])

    bond_len = 1.0
    angle_rad = np.deg2rad(90.0)
    dihedral_rad = np.deg2rad(180.0)

    # For p1=(0,0,0), p2=(1,0,0), p3=(1,1,0)
    # bond_len=1, angle=90, dihedral=180 (trans)
    # The atom should be at (2,1,0)
    expected_res = np.array([2.0, 1.0, 0.0])

    # diff-biophys version (JAX)
    res_jax = position_atom_3d(
        jnp.array(p1), jnp.array(p2), jnp.array(p3), bond_len, angle_rad, dihedral_rad
    )

    # Assert parity
    np.testing.assert_allclose(expected_res, np.array(res_jax), atol=1e-5)
    print("NeRF Parity Verified!")


if __name__ == "__main__":
    test_nerf_parity()
