import jax.numpy as jnp
import numpy as np


def test_science_rdc_angular_dependence() -> None:
    """
    Scientific Validation: RDC Angular Dependence (Clore et al., 1998).

    The RDC magnitude in the Principal Axis Frame follows:
    D(theta, phi) = Da * [(3 cos^2 theta - 1) + 1.5 * R * sin^2 theta * cos(2*phi)]

    For an axial tensor (R = 0), this simplifies to:
    D(theta) = Da * (3 cos^2 theta - 1)

    This validates the angular physics of the dipolar coupling implementation.
    """
    from diff_biophys.nmr.rdc import calculate_rdc

    da = 10.0
    r = 0.0  # Perfectly axial

    # 1. Test at the "Magic Angle" (theta = 54.74 degrees)
    # cos^2(theta) = 1/3, so 3*cos^2(theta) - 1 = 0. RDC should be zero.
    magic_angle = jnp.arccos(jnp.sqrt(1 / 3.0))
    # Vector at magic angle from Z axis
    v_magic = jnp.array([[jnp.sin(magic_angle), 0.0, jnp.cos(magic_angle)]])
    rdc_magic = calculate_rdc(v_magic, da, r)[0]
    np.testing.assert_allclose(float(rdc_magic), 0.0, atol=1e-5)

    # 2. Test aligned with Z axis (theta = 0)
    # 3*cos^2(0) - 1 = 2. RDC should be 2 * Da.
    v_z = jnp.array([[0.0, 0.0, 1.0]])
    rdc_z = calculate_rdc(v_z, da, r)[0]
    np.testing.assert_allclose(float(rdc_z), 2.0 * da, atol=1e-5)

    # 3. Test aligned with X/Y plane (theta = 90)
    # 3*cos^2(90) - 1 = -1. RDC should be -Da.
    v_xy = jnp.array([[1.0, 0.0, 0.0]])
    rdc_xy = calculate_rdc(v_xy, da, r)[0]
    np.testing.assert_allclose(float(rdc_xy), -da, atol=1e-5)

    print("✅ RDC Angular Dependence (Magic Angle & Z-axis) Validation Successful!")


if __name__ == "__main__":
    test_science_rdc_angular_dependence()
