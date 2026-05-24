import jax.numpy as jnp
from jax import jit, vmap
from typing import Any

# Atomic volumes (A^3) from Pavlov & Svergun (1997)
VOLUMES = {
    "H": 5.15,
    "C": 16.44,
    "N": 14.0,
    "O": 12.0,
    "S": 19.86,
    "P": 24.4,
}

@jit
def debye_saxs(coords: jnp.ndarray, 
               q_values: jnp.ndarray, 
               form_factors: jnp.ndarray, 
               volumes: jnp.ndarray = None,
               solvent_density: float = 0.334) -> jnp.ndarray:
    """
    Differentiable Debye Formula in JAX with optional Solvent Subtraction.
    
    Args:
        coords: (N, 3) coordinates
        q_values: (M,) q points
        form_factors: (N, M) q-dependent vacuum form factors
        volumes: (N,) atomic volumes for solvent displacement. If None, no solvent subtraction.
        solvent_density: Solvent electron density (default 0.334 e/A^3 for water)
        
    Returns:
        jnp.ndarray: Scattering intensities I(q)
    """
    # 1. Pairwise distances (N, N)
    sq_norms = jnp.sum(coords**2, axis=-1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * jnp.dot(coords, coords.T)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 0.0) + 1e-12)
    
    # 2. Solvent Correction (Hydration Shell)
    # f_eff(q) = f_vac(q) - rho_sol * V * exp(-q^2 * R^2 / 10)
    # We simplify the Gaussian decay for now or use a constant volume subtraction.
    if volumes is not None:
        # Effective radius for the volume (V = 4/3 * pi * R^3)
        r_eff = (3.0 * volumes / (4.0 * jnp.pi))**(1.0/3.0)
        
        # Gaussian decay term for the excluded volume
        # exp(- (q * r_eff)^2 / (4 * pi)) is a common approximation
        decay = jnp.exp(-(q_values[None, :] * r_eff[:, None])**2 / (4.0 * jnp.pi))
        f_eff = form_factors - (solvent_density * volumes[:, None] * decay)
    else:
        f_eff = form_factors

    def compute_intensity(q_idx):
        q = q_values[q_idx]
        f_q = f_eff[:, q_idx]
        f_prod = f_q[:, None] * f_q[None, :]
        qr = q * dist
        # Sinc function with safety for qr -> 0
        sinc_qr = jnp.where(qr < 1e-4, 1.0 - (qr**2) / 6.0, jnp.sin(qr + 1e-10) / (qr + 1e-10))
        return jnp.sum(f_prod * sinc_qr)
    
    return vmap(compute_intensity)(jnp.arange(len(q_values)))
