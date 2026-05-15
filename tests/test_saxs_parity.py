import sys
import os
import numpy as np
import jax.numpy as jnp
import jax

# Add parent directories to path for imports
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../synth-pdb'))

from diff_biophys.saxs.kernels import debye_saxs
from synth_pdb.saxs import get_form_factor

def test_saxs_parity_and_gradients():
    print('Testing SAXS Kernel Parity and Gradients...')
    
    # 1. Create a dummy structure (Small peptide)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.5, 0.0],
        [0.0, 1.5, 1.5]
    ], dtype=np.float32)
    
    # 2. Define q values
    q_values = np.linspace(0.01, 0.5, 10).astype(np.float32)
    
    # 3. Form factors
    f_c = get_form_factor('C', q_values)
    form_factors = np.tile(f_c, (len(coords), 1))
    
    # 4. Run NumPy version (Manual reference)
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1))
    expected_intensities = []
    for qi, fi in zip(q_values, form_factors.T):
        f_prod = fi[:, None] * fi[None, :]
        qr = qi * dist
        # Standard NumPy sinc implementation
        sinc_qr = np.sinc(qr / np.pi) 
        expected_intensities.append(np.sum(f_prod * sinc_qr))
    expected_intensities = np.array(expected_intensities)
    
    # 5. Run JAX version
    actual_intensities = debye_saxs(
        jnp.array(coords), 
        jnp.array(q_values), 
        jnp.array(form_factors)
    )
    
    # 6. Assert parity
    np.testing.assert_allclose(expected_intensities, np.array(actual_intensities), rtol=1e-5)
    print('✅ SAXS Kernel Parity Verified!')
    
    # 7. Gradient Check
    def loss_fn(c):
        intensities = debye_saxs(c, jnp.array(q_values), jnp.array(form_factors))
        return jnp.mean(intensities)
        
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(jnp.array(coords))
    
    assert grads.shape == coords.shape
    assert jnp.all(jnp.isfinite(grads))
    print(f'✅ SAXS Gradients Verified! (Sum of grads: {jnp.sum(grads):.6f})')

if __name__ == '__main__':
    test_saxs_parity_and_gradients()
