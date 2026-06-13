import jax.numpy as jnp
from jax import jit


@jit
def calculate_karplus_j(theta: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    """
    Calculate 3J coupling constants using the Karplus equation.

    J = a * cos^2(theta) + b * cos(theta) + c

    Args:
        theta: (N,) Dihedral angles in radians.
        a: Cosine-squared coefficient (Hz).
        b: Cosine coefficient (Hz).
        c: Constant offset (Hz).

    Returns:
        jnp.ndarray: (N,) Calculated J-couplings.
    """
    cos_theta = jnp.cos(theta)
    return a * (cos_theta**2) + b * cos_theta + c
