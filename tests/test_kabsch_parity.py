import sys
import os
import numpy as np
import jax.numpy as jnp
from diff_biophys.geometry.superposition import kabsch_alignment
from synth_pdb.geometry.superposition import kabsch_superposition

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
    
    # NumPy version
    R_np, t_np = kabsch_superposition(P, Q)
    
    # JAX version
    R_jax, t_jax = kabsch_alignment(jnp.array(P), jnp.array(Q))
    
    # Assert parity
    np.testing.assert_allclose(R_np, np.array(R_jax), atol=1e-5)
    np.testing.assert_allclose(t_np, np.array(t_jax), atol=1e-5)
    print('Kabsch Parity Verified!')

if __name__ == '__main__':
    test_kabsch_parity()
