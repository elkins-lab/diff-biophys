import numpy as np
import jax.numpy as jnp
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor, fit_saupe_tensor

def test_rdc_tensor_fitting():
    # 1. Create a known symmetric traceless tensor
    # Trace: 0.01 + 0.02 - 0.03 = 0
    true_tensor = np.array([
        [0.01, 0.005, 0.002],
        [0.005, 0.02, -0.001],
        [0.002, -0.001, -0.03]
    ], dtype=np.float32)
    
    # 2. Generate random bond vectors
    np.random.seed(42)
    n_bonds = 50
    vectors = np.random.randn(n_bonds, 3).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    
    # 3. Calculate "experimental" RDCs
    d_max = 20000.0 # Typical N-H max coupling
    rdcs = calculate_rdc_from_tensor(jnp.array(vectors), jnp.array(true_tensor), d_max=d_max)
    
    # 4. Fit the tensor back
    fitted_tensor = fit_saupe_tensor(jnp.array(vectors), rdcs, d_max=d_max)
    
    # 5. Assert parity
    np.testing.assert_allclose(true_tensor, np.array(fitted_tensor), atol=1e-5)
    print("✅ RDC Saupe Tensor Fitting Verified!")

if __name__ == '__main__':
    test_rdc_tensor_fitting()
