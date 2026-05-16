import numpy as np
import jax.numpy as jnp
from diff_biophys.saxs.kernels import debye_saxs
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor

def test_saxs_invariance():
    """SAXS intensities should be invariant to rotation and translation."""
    coords = np.random.randn(10, 3).astype(np.float32)
    q_values = np.linspace(0.01, 0.5, 20).astype(np.float32)
    form_factors = np.ones((10, 20), dtype=np.float32)
    
    # Base intensity
    i_base = debye_saxs(jnp.array(coords), jnp.array(q_values), jnp.array(form_factors))
    
    # 1. Translate
    coords_trans = coords + np.array([10.0, -5.0, 2.0])
    i_trans = debye_saxs(jnp.array(coords_trans), jnp.array(q_values), jnp.array(form_factors))
    np.testing.assert_allclose(i_base, i_trans, atol=1e-4)
    
    # 2. Rotate
    # Simple 90 deg rotation around Z
    theta = np.deg2rad(90)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords_rot = (rot_matrix @ coords.T).T
    i_rot = debye_saxs(jnp.array(coords_rot), jnp.array(q_values), jnp.array(form_factors))
    np.testing.assert_allclose(i_base, i_rot, atol=1e-4)
    print("✅ SAXS Invariance Verified!")

def test_rdc_tensor_invariance():
    """RDCs should be invariant to translation of bond vectors."""
    vectors = np.random.randn(10, 3).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    
    tensor = jnp.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, -0.02]
    ])
    
    rdcs_base = calculate_rdc_from_tensor(jnp.array(vectors), tensor)
    
    # Translating a bond vector doesn't change its orientation, 
    # and calculate_rdc_from_tensor only takes the vectors.
    # However, we can check if it's sensitive to vector scaling (it should be, quadratically)
    # But RDCs are typically defined for unit vectors.
    
    # Let's test rotational covariance: rotating both vectors and tensor should preserve RDCs.
    theta = np.deg2rad(45)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    vectors_rot = (R @ vectors.T).T
    # S' = R S R^T
    tensor_rot = jnp.array(R @ np.array(tensor) @ R.T)
    
    rdcs_rot = calculate_rdc_from_tensor(jnp.array(vectors_rot), tensor_rot)
    np.testing.assert_allclose(rdcs_base, rdcs_rot, atol=1e-5)
    print("✅ RDC Rotational Covariance Verified!")

if __name__ == "__main__":
    test_saxs_invariance()
    test_rdc_tensor_invariance()
