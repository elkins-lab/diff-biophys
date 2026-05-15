import jax.numpy as jnp
from jax import jit, vmap
from typing import Any

@jit
def debye_saxs(coords: jnp.ndarray, q_values: jnp.ndarray, form_factors: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiable Debye Formula in JAX.
    
    Args:
        coords: (N, 3) coordinates
        q_values: (M,) q points
        form_factors: (N, M) q-dependent atomic form factors
        
    Returns:
        jnp.ndarray: Scattering intensities I(q)
    """
    # 1. Pairwise distances (N, N)
    sq_norms = jnp.sum(coords**2, axis=-1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * jnp.dot(coords, coords.T)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 0.0) + 1e-12)
    
    def compute_intensity(q, f_q):
        f_prod = f_q[:, None] * f_q[None, :]
        qr = q * dist
        sinc_qr = jnp.where(qr < 1e-4, 1.0 - (qr**2) / 6.0, jnp.sin(qr + 1e-10) / (qr + 1e-10))
        return jnp.sum(f_prod * sinc_qr)
    
    return vmap(compute_intensity)(q_values, form_factors.T)
