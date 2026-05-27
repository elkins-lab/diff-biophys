import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr import calculate_ring_current_shift


def test_ring_current_shielding_cone():
    """
    Validate the spatial shielding/deshielding cone characteristic of Johnson-Bovey rings.
    """
    center = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])  # Z-axis is normal to ring plane (XY)
    intensity = 1.0

    # 1. Above center (Shielding Cone)
    # Nuclei directly above the ring face should experience negative (upfield) shifts
    coords_above = jnp.array([[0.0, 0.0, 2.0]])
    shift_above = calculate_ring_current_shift(coords_above, center, normal, intensity)[0]
    assert shift_above < 0
    print(f"✅ Shielding Cone (above ring): {shift_above:.4f} ppm (upfield)")

    # 2. In-plane (Deshielding region)
    # Nuclei in the equatorial plane (XY) should experience positive (downfield) shifts
    coords_plane = jnp.array([[2.0, 0.0, 0.0]])
    shift_plane = calculate_ring_current_shift(coords_plane, center, normal, intensity)[0]
    assert shift_plane > 0
    print(f"✅ Deshielding Region (in-plane): {shift_plane:.4f} ppm (downfield)")


def test_trp_intensity_scaling():
    """
    Verify relative intensities of aromatic residues.
    Trp indole usually has higher intensity than Phe benzene.
    """
    center = jnp.array([0.0, 0.0, 0.0])
    normal = jnp.array([0.0, 0.0, 1.0])
    coord = jnp.array([[0.0, 0.0, 3.0]])

    # Intensities from literature (e.g. SHIFTX)
    intensity_phe = 1.0
    intensity_trp_6m = 1.2  # indole 6-m ring

    shift_phe = calculate_ring_current_shift(coord, center, normal, intensity_phe)
    shift_trp = calculate_ring_current_shift(coord, center, normal, intensity_trp_6m)

    np.testing.assert_allclose(shift_trp, 1.2 * shift_phe, atol=1e-5)
    print("✅ Ring Current Intensity Scaling Verified!")


if __name__ == "__main__":
    test_ring_current_shielding_cone()
    test_trp_intensity_scaling()
