import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr.rdc import calculate_rdc


def test_rdc_parity():
    # Test with a vector along the Z-axis
    # For a Z-aligned vector, RDC = da * (3*1^2 - 1) = 2 * da
    da = 10.0
    r = 0.2

    bond_vector = np.array([[0.0, 0.0, 1.0]])
    actual_rdcs = calculate_rdc(jnp.array(bond_vector), da, r)
    actual_val = actual_rdcs[0]

    expected_val = 20.0

    np.testing.assert_allclose(expected_val, float(actual_val), atol=1e-5)

    # Test with a vector along the X-axis
    # x=1, y=0, z=0
    # cos_theta = 0, sin_theta_sq = 1
    # cos_2phi = (1^2 - 0^2) / 1 = 1
    # axial = 3*0 - 1 = -1
    # rhombic = 1.5 * r * 1 * 1 = 1.5 * r
    # RDC = da * (-1 + 1.5 * r)
    bond_vector_x = np.array([[1.0, 0.0, 0.0]])
    actual_rdcs_x = calculate_rdc(jnp.array(bond_vector_x), da, r)
    expected_val_x = da * (-1 + 1.5 * r)

    np.testing.assert_allclose(expected_val_x, float(actual_rdcs_x[0]), atol=1e-5)
    print("RDC Parity Verified!")


if __name__ == "__main__":
    test_rdc_parity()
