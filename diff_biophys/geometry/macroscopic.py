from typing import cast

import jax.numpy as jnp
from jax import jit


@jit
def compute_rg(coords: jnp.ndarray, masses: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Compute the Radius of Gyration (Rg) for a set of coordinates.

    Rg is a macroscopic measure of compactness, commonly used in SAXS and polymer physics.
    This implementation is fully differentiable.

    Args:
        coords: (N, 3) array of coordinates.
        masses: Optional (N,) array of weights (e.g., atomic masses or electron counts).
                If None, all points are weighted equally.

    Returns:
        A scalar jnp.ndarray representing the Radius of Gyration.
    """
    if masses is None:
        # Unweighted center of mass
        com = jnp.mean(coords, axis=0)
        # Mean squared distance from COM
        sq_dist = jnp.sum((coords - com) ** 2, axis=-1)
        rg_sq = jnp.mean(sq_dist)
    else:
        # Weighted center of mass
        total_mass = jnp.sum(masses)
        com = jnp.sum(coords * masses[:, None], axis=0) / total_mass
        # Weighted mean squared distance
        sq_dist = jnp.sum((coords - com) ** 2, axis=-1)
        rg_sq = jnp.sum(sq_dist * masses) / total_mass

    return cast(
        jnp.ndarray, jnp.sqrt(rg_sq + 1e-10)
    )  # Add epsilon for numerical stability of sqrt near 0
