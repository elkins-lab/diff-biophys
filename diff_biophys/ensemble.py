import jax.numpy as jnp
from jax import vmap, jit
from typing import Callable, Any

class Ensemble:
    """
    High-level API for ensemble-averaged biophysical observables.
    """
    def __init__(self, coordinates: jnp.ndarray, weights: jnp.ndarray = None):
        """
        Args:
            coordinates: (M, N, 3) array where M is ensemble size and N is atom count.
            weights: (M,) array of population weights. Defaults to uniform.
        """
        self.coords = coordinates
        self.m = coordinates.shape[0]
        if weights is None:
            self.weights = jnp.full((self.m,), 1.0 / self.m)
        else:
            self.weights = weights / jnp.sum(weights)

    def calculate_average(self, observable_fn: Callable[[jnp.ndarray], jnp.ndarray], *args, **kwargs) -> jnp.ndarray:
        """
        Calculate the population-weighted average of an observable.
        
        Args:
            observable_fn: Function that takes (N, 3) coords and returns (D,) observable.
            *args, **kwargs: Additional arguments for the observable_fn.
            
        Returns:
            jnp.ndarray: (D,) averaged observable.
        """
        # Vectorize the observable function over the ensemble dimension
        v_fn = vmap(lambda c: observable_fn(c, *args, **kwargs))
        ensemble_results = v_fn(self.coords) # (M, D)
        
        # Weighted average
        return jnp.sum(ensemble_results * self.weights[:, None], axis=0)

@jit
def calculate_ensemble_saxs(coords: jnp.ndarray, weights: jnp.ndarray, q_values: jnp.ndarray, form_factors: jnp.ndarray):
    """Utility for fast ensemble SAXS."""
    from diff_biophys.saxs import debye_saxs
    v_saxs = vmap(lambda c: debye_saxs(c, q_values, form_factors))
    return jnp.sum(v_saxs(coords) * weights[:, None], axis=0)
