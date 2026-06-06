import jax
import jax.numpy as jnp
import numpy as np
from synth_core.signal_processing import compute_fsc as numpy_compute_fsc

from diff_biophys.cryo_em import compute_fsc


def test_fsc_parity() -> None:
    """Verify that JAX compute_fsc returns the same results as synth_core NumPy."""
    np.random.seed(42)
    shape = (16, 16, 16)
    map1 = np.random.normal(0, 1, shape)
    map2 = np.random.normal(0, 1, shape)
    voxel_size = (1.0, 1.0, 1.0)

    # Run numpy version
    freqs_np, fsc_np = numpy_compute_fsc(map1, map2, voxel_size)

    # Run jax version
    freqs_jax, fsc_jax = compute_fsc(jnp.array(map1), jnp.array(map2), voxel_size)

    # For bins that have data, the JAX array valid elements should match the NumPy elements.
    # Because NumPy dynamically sizes the output based on valid bins and JAX returns a statically sized array
    # with the same length as n_bins, we will slice the JAX array or mask it.

    # The first len(freqs_np) elements should be approximately equal for continuous identical data
    # We test on dense arrays so all bins should be valid.

    # We compare where JAX output is valid (not 0.0 frequency, as valid frequencies > 0)
    mask = freqs_jax > 0.0

    np.testing.assert_allclose(freqs_jax[mask], freqs_np, rtol=1e-5)
    np.testing.assert_allclose(fsc_jax[mask], fsc_np, rtol=1e-5)


def test_fsc_gradients() -> None:
    """Verify that gradients can flow through compute_fsc."""
    shape = (16, 16, 16)
    key = jax.random.PRNGKey(42)
    map1 = jax.random.normal(key, shape)
    map2 = jax.random.normal(key, shape)
    voxel_size = (1.0, 1.0, 1.0)

    # We define a loss function that tries to maximize the FSC across all frequencies
    def loss_fn(m1: jax.Array, m2: jax.Array) -> jax.Array:
        _, fsc = compute_fsc(m1, m2, voxel_size)
        return jnp.sum(fsc)

    # Compute the gradient with respect to map1
    grad_fn = jax.grad(loss_fn, argnums=0)
    gradients = grad_fn(map1, map2)

    assert gradients.shape == map1.shape
    assert not jnp.all(gradients == 0)
