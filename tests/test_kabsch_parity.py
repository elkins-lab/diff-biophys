import numpy as np
import jax.numpy as jnp
from diff_biophys.geometry.superposition import kabsch_alignment

def test_kabsch_parity():
    # 1. Create dummy structures
    P = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    
    # Q is P rotated 90 deg around Z
    Q = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Expected Rotation Matrix (90 deg around Z)
    # [cos(90) -sin(90) 0]   [0 -1 0]
    # [sin(90)  cos(90) 0] = [1  0 0]
    # [0        0       1]   [0  0 1]
    # Since Q = R * P + t, R should be the rotation that takes P to Q.
    # Actually kabsch_alignment usually returns R such that R * P_centered approx Q_centered.
    
    # JAX version
    R_jax, t_jax = kabsch_alignment(jnp.array(P), jnp.array(Q))
    
    # Reconstruct Q from P
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    Q_reconstructed = (np.array(R_jax) @ P_centered.T).T
    
    # Assert parity
    np.testing.assert_allclose(Q_centered, Q_reconstructed, atol=1e-5)
    print('Kabsch Parity Verified!')

def test_kabsch_reflection():
    # 1. Create a structure P
    P = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # 2. Q is P reflected across the XY plane (z -> -z)
    Q = P.copy()
    Q[:, 2] = -Q[:, 2]
    
    # JAX version
    R_jax, t_jax = kabsch_alignment(jnp.array(P), jnp.array(Q))
    
    # Reconstruct Q from P
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    Q_reconstructed = (np.array(R_jax) @ P_centered.T).T
    
    # Even with reflection, Kabsch should find the best ROTATION.
    # For a pure reflection, the "best rotation" will still have a residual RMSD.
    # This test ensures the determinant logic in kabsch_alignment doesn't crash
    # and produces a valid rotation matrix (det=1).
    det = np.linalg.det(np.array(R_jax))
    np.testing.assert_allclose(det, 1.0, atol=1e-5)
    print('Kabsch Reflection Handling Verified!')

if __name__ == '__main__':
    test_kabsch_parity()
    test_kabsch_reflection()
