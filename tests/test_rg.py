import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# We haven't implemented this yet, so it should fail on import
from diff_biophys.geometry.macroscopic import compute_rg


def test_compute_rg_basic() -> None:
    """Test Rg calculation against a known simple geometry."""
    # A square of side length 2 in the XY plane.
    # Center of mass is (1, 1, 0) if corners are (0,0,0), (2,0,0), (2,2,0), (0,2,0)
    # Distances to center are all sqrt(1^2 + 1^2) = sqrt(2)
    # Rg^2 = (2 + 2 + 2 + 2) / 4 = 2.0
    # Rg = sqrt(2.0)
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    rg = compute_rg(coords)
    np.testing.assert_allclose(float(rg), float(np.sqrt(2.0)), atol=1e-5)


def test_compute_rg_weighted() -> None:
    """Test Rg calculation with non-uniform weights (e.g., masses)."""
    coords = jnp.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Unequal weights
    weights = jnp.array([1.0, 3.0])

    # Weighted COM: (-1*1 + 1*3) / 4 = 2/4 = 0.5
    # Distances to COM:
    # pt 0: |-1 - 0.5| = 1.5 -> sq = 2.25
    # pt 1: |1 - 0.5| = 0.5 -> sq = 0.25
    # Weighted sum of squares: 1.0 * 2.25 + 3.0 * 0.25 = 2.25 + 0.75 = 3.0
    # Rg^2 = 3.0 / 4.0 = 0.75
    # Rg = sqrt(0.75)

    rg = compute_rg(coords, masses=weights)
    np.testing.assert_allclose(float(rg), float(np.sqrt(0.75)), atol=1e-5)


@settings(deadline=None)
@given(tx=st.floats(-100, 100), ty=st.floats(-100, 100), tz=st.floats(-100, 100))
def test_rg_translation_invariance(tx: float, ty: float, tz: float) -> None:
    """Rg should be invariant to rigid body translations."""
    np.random.seed(42)
    coords = jnp.array(np.random.randn(10, 3).astype(np.float32))

    rg_orig = compute_rg(coords)

    translated_coords = coords + jnp.array([tx, ty, tz])
    rg_trans = compute_rg(translated_coords)

    np.testing.assert_allclose(float(rg_trans), float(rg_orig), atol=1e-4)


def test_rg_differentiability() -> None:
    """Ensure we can take the gradient of Rg with respect to coordinates."""
    coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    # Rg^2 = 1.0, Rg = 1.0
    # Derivative of Rg wrt coords:
    grad_fn = jax.grad(compute_rg)
    grads = grad_fn(coords)

    assert grads.shape == coords.shape
    assert jnp.all(jnp.isfinite(grads))
