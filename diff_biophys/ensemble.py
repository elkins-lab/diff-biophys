from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import jit, vmap


@jax.tree_util.register_pytree_node_class
class Ensemble:
    """
    High-level API for ensemble-averaged biophysical observables.
    """

    coords: jnp.ndarray
    weights: jnp.ndarray
    m: int

    def __init__(self, coordinates: jnp.ndarray, weights: jnp.ndarray | None = None) -> None:
        """
        Args:
            coordinates: (M, N, 3) array where M is ensemble size and N is atom count.
            weights: (M,) array of population weights. Defaults to uniform.
        """
        self.coords = coordinates
        self.m = int(coordinates.shape[0])
        if weights is None:
            self.weights = jnp.full((self.m,), 1.0 / self.m)
        else:
            self.weights = jnp.asarray(weights) / jnp.sum(weights)

    def calculate_average(
        self,
        observable_fn: Callable[..., jnp.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """
        Calculate the population-weighted average of an observable.

        Args:
            observable_fn: Function that takes (N, 3) coords and returns (D,) or scalar observable.
            *args, **kwargs: Additional arguments for the observable_fn.

        Returns:
            jnp.ndarray: (D,) or scalar averaged observable.
        """
        # Vectorize the observable function over the ensemble dimension
        v_fn = vmap(lambda c: observable_fn(c, *args, **kwargs))
        ensemble_results = v_fn(self.coords)

        # Handle both scalar and vector outputs from observable_fn
        if ensemble_results.ndim > 1:
            # Result is (M, D, ...). Weights are (M,).
            # We want to multiply by weights along the first dimension.
            # Create a broadcast-compatible shape for weights: (M, 1, 1, ...)
            weight_shape = (self.m,) + (1,) * (ensemble_results.ndim - 1)
            return jnp.sum(ensemble_results * self.weights.reshape(weight_shape), axis=0)
        else:
            # Result is (M,). Just a dot product with weights.
            return jnp.dot(ensemble_results, self.weights)

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[int]]:
        """Standard JAX PyTree flattening."""
        children = (self.coords, self.weights)
        aux_data = (self.m,)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[int], children: tuple[jnp.ndarray, jnp.ndarray]
    ) -> "Ensemble":
        """Standard JAX PyTree unflattening."""
        coords, weights = children
        m = aux_data[0]
        # Skip normalization in unflatten to preserve gradients exactly
        obj = cls.__new__(cls)
        obj.coords = coords
        obj.weights = weights
        obj.m = m
        return obj


@jit
def calculate_ensemble_saxs(
    coords: jnp.ndarray, weights: jnp.ndarray, q_values: jnp.ndarray, form_factors: jnp.ndarray
) -> jnp.ndarray:
    """Utility for fast ensemble SAXS."""
    from diff_biophys.saxs import debye_saxs

    v_saxs = vmap(lambda c: debye_saxs(c, q_values, form_factors))
    return cast(jnp.ndarray, jnp.sum(v_saxs(coords) * weights[:, None], axis=0))
