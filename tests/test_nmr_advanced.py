import jax.numpy as jnp
import numpy as np
import pytest
from diff_biophys.nmr.rdc import calculate_rdc # Existing
# We will create these:
# from diff_biophys.nmr.karplus import calculate_karplus_j
# from diff_biophys.nmr.ring_currents import calculate_ring_current_shift

def test_karplus_parity():
    """
    Test Karplus J-coupling calculation.
    J = A cos^2(theta) + B cos(theta) + C
    """
    from diff_biophys.nmr import calculate_karplus_j
    
    # Standard 3J(HN, HA) parameters (Bax 1990)
    # A = 6.51, B = -1.25, C = 1.60
    params = (6.51, -1.25, 1.60)
    
    # theta = 0 -> J = A + B + C = 6.86
    # theta = 180 -> J = A - B + C = 9.36
    # theta = 90 -> J = C = 1.60
    
    thetas = jnp.array([0.0, jnp.pi, jnp.pi/2])
    expected = jnp.array([6.86, 9.36, 1.60])
    
    actual = calculate_karplus_j(thetas, *params)
    np.testing.assert_allclose(actual, expected, atol=1e-5)

def test_ring_current_parity():
    """
    Test Johnson-Bovey ring current shifts.
    Uses the geometric term (1 - 3 cos^2 theta) / r^3
    """
    from diff_biophys.nmr import calculate_ring_current_shift
    
    # Mock aromatic ring at origin in XY plane
    ring_center = jnp.array([0.0, 0.0, 0.0])
    ring_normal = jnp.array([0.0, 0.0, 1.0])
    
    # Test point on Z-axis (above center)
    # r = 3.0, cos_theta = 1.0 (aligned with normal)
    # term = (1 - 3*1) / 3^3 = -2 / 27
    point_z = jnp.array([[0.0, 0.0, 3.0]])
    
    # Test point on X-axis (in plane)
    # r = 3.0, cos_theta = 0.0 (perpendicular to normal)
    # term = (1 - 0) / 3^3 = 1 / 27
    point_x = jnp.array([[3.0, 0.0, 0.0]])
    
    # Scale factor (B in Johnson-Bovey) - arbitrary for parity check
    intensity = 100.0
    
    shift_z = calculate_ring_current_shift(point_z, ring_center, ring_normal, intensity)
    shift_x = calculate_ring_current_shift(point_x, ring_center, ring_normal, intensity)
    
    # Ratio should be -2
    np.testing.assert_allclose(shift_z / shift_x, -2.0, atol=1e-5)

def test_nmr_gradients():
    """Verify both kernels are differentiable."""
    from diff_biophys.nmr import calculate_karplus_j, calculate_ring_current_shift
    import jax
    
    # Karplus Grad
    grad_karplus = jax.grad(lambda x: jnp.sum(calculate_karplus_j(x, 6.5, -1.2, 1.6)))
    g_k = grad_karplus(jnp.array([0.5, 1.0]))
    assert jnp.all(jnp.isfinite(g_k))
    
    # Ring Current Grad (wrt coordinates)
    def loss_rc(coords):
        return jnp.sum(calculate_ring_current_shift(coords, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]), 1.0))
    
    grad_rc = jax.grad(loss_rc)
    g_rc = grad_rc(jnp.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]]))
    assert g_rc.shape == (2, 3)
    assert jnp.all(jnp.isfinite(g_rc))
