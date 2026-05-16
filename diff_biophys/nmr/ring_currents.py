import jax.numpy as jnp
from jax import jit

@jit
def calculate_ring_current_shift(coords: jnp.ndarray, 
                                 ring_center: jnp.ndarray, 
                                 ring_normal: jnp.ndarray, 
                                 intensity: float) -> jnp.ndarray:
    """
    Calculate chemical shift changes due to aromatic ring currents using 
    the Johnson-Bovey dipolar approximation.
    
    delta = intensity * (1 - 3*cos^2(theta)) / r^3
    
    Args:
        coords: (N, 3) coordinates of the nuclei being shielded.
        ring_center: (3,) coordinates of the aromatic ring center.
        ring_normal: (3,) unit vector normal to the ring plane.
        intensity: Scaling factor (proportional to ring area and current).
        
    Returns:
        jnp.ndarray: (N,) shielding values in ppm.
    """
    # 1. Displacement vectors from ring center
    r_vec = coords - ring_center
    
    # 2. Distances
    r = jnp.linalg.norm(r_vec, axis=-1)
    
    # 3. cos(theta) where theta is the angle between r_vec and the ring normal
    # cos(theta) = (r_vec . normal) / (|r_vec| * |normal|)
    # Assume ring_normal is already a unit vector
    cos_theta = jnp.sum(r_vec * ring_normal, axis=-1) / (r + 1e-10)
    
    # 4. Johnson-Bovey geometric term
    # delta = intensity * (1 - 3 * cos^2(theta)) / r^3
    return intensity * (1.0 - 3.0 * cos_theta**2) / (r**3 + 1e-10)
