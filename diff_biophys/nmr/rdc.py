import jax.numpy as jnp
from jax import jit

@jit
def calculate_rdc(bond_vectors: jnp.ndarray, da: float, r: float) -> jnp.ndarray:
    """
    Differentiable RDC calculation in JAX.
    
    Args:
        bond_vectors: (N, 3) unit vectors for bond orientations
        da: Axial component of the alignment tensor in Hz
        r: Rhombicity of the alignment tensor (0 <= R <= 2/3)
        
    Returns:
        jnp.ndarray: Calculated RDC values in Hz
    """
    x, y, z = bond_vectors[:, 0], bond_vectors[:, 1], bond_vectors[:, 2]
    
    cos_theta = z
    sin_theta_sq = 1.0 - cos_theta**2
    
    cos_2phi = (x**2 - y**2) / (sin_theta_sq + 1e-10)
    
    axial = 3.0 * cos_theta**2 - 1.0
    rhombic = 1.5 * r * sin_theta_sq * cos_2phi
    
    return da * (axial + rhombic)
