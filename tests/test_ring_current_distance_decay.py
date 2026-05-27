import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr import calculate_ring_current_shift


def test_ring_current_inverse_cube_decay():
    """
    Verify the 1/r³ distance decay of the Johnson-Bovey ring current model.

    Along the ring axis (θ = 0°, cos²θ = 1), the geometric term simplifies to:
        δ = intensity · (1 − 3·cos²θ) / r³ = intensity · (−2) / r³

    Doubling the distance should therefore reduce |δ| by a factor of 2³ = 8.
    """
    center = jnp.zeros(3)
    normal = jnp.array([0.0, 0.0, 1.0])
    intensity = 1.0

    # Two points on the Z-axis: r = 2 Å and r = 4 Å
    c2 = jnp.array([[0.0, 0.0, 2.0]])
    c4 = jnp.array([[0.0, 0.0, 4.0]])

    s2 = float(calculate_ring_current_shift(c2, center, normal, intensity)[0])
    s4 = float(calculate_ring_current_shift(c4, center, normal, intensity)[0])

    # Both should be negative (shielding cone above ring)
    assert s2 < 0 and s4 < 0, "Axial points should be in the shielding cone (δ < 0)"

    # |s2| / |s4| should equal 2³ = 8
    np.testing.assert_allclose(
        abs(s2) / abs(s4),
        8.0,
        rtol=1e-4,
        err_msg=f"1/r³ ratio: got {abs(s2) / abs(s4):.4f}, expected 8.0",
    )
    print(
        f"✅ Ring current 1/r³ decay: δ(r=2)={s2:.4f}, δ(r=4)={s4:.4f}, ratio={abs(s2) / abs(s4):.4f}"
    )


def test_ring_current_equatorial_decay():
    """
    Same 1/r³ test for the equatorial (deshielding) region (θ = 90°).

    Here the geometric term is (1 − 0) / r³ = 1/r³ (positive, deshielding).
    """
    center = jnp.zeros(3)
    normal = jnp.array([0.0, 0.0, 1.0])
    intensity = 1.0

    c2 = jnp.array([[2.0, 0.0, 0.0]])
    c4 = jnp.array([[4.0, 0.0, 0.0]])

    s2 = float(calculate_ring_current_shift(c2, center, normal, intensity)[0])
    s4 = float(calculate_ring_current_shift(c4, center, normal, intensity)[0])

    assert s2 > 0 and s4 > 0, "Equatorial points should be in the deshielding region (δ > 0)"

    np.testing.assert_allclose(
        s2 / s4,
        8.0,
        rtol=1e-4,
        err_msg=f"1/r³ ratio in equatorial plane: got {s2 / s4:.4f}, expected 8.0",
    )
    print(f"✅ Equatorial 1/r³ decay: δ(r=2)={s2:.4f}, δ(r=4)={s4:.4f}, ratio={s2 / s4:.4f}")


if __name__ == "__main__":
    test_ring_current_inverse_cube_decay()
    test_ring_current_equatorial_decay()
