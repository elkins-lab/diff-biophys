import numpy as np
import jax.numpy as jnp
import pytest
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor, fit_saupe_tensor


def test_fitted_tensor_is_traceless_and_symmetric():
    """
    Verify the two algebraic constraints of the Saupe tensor after fitting:
      1. Traceless:  Sxx + Syy + Szz = 0
      2. Symmetric:  S = S^T

    Both properties are enforced by the construction in fit_saupe_tensor,
    but should be explicitly asserted to guard against future refactoring.
    """
    np.random.seed(7)
    vectors = np.random.randn(30, 3).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    true_tensor = np.array([
        [0.010,  0.003,  0.000],
        [0.003,  0.015,  0.000],
        [0.000,  0.000, -0.025],
    ], dtype=np.float32)

    d_max = 1000.0
    rdcs = calculate_rdc_from_tensor(jnp.array(vectors), jnp.array(true_tensor), d_max=d_max)
    S = np.array(fit_saupe_tensor(jnp.array(vectors), rdcs, d_max=d_max))

    np.testing.assert_allclose(np.trace(S), 0.0, atol=1e-5,
                               err_msg="Fitted Saupe tensor must be traceless (Sxx+Syy+Szz=0)")
    np.testing.assert_allclose(S, S.T, atol=1e-5,
                               err_msg="Fitted Saupe tensor must be symmetric (S=S^T)")

    print(f"✅ Saupe tensor trace = {np.trace(S):.2e} (≈0), symmetric: {np.allclose(S, S.T, atol=1e-5)}")


def test_fitted_tensor_recovers_ground_truth():
    """
    Full round-trip: generate RDCs from a known tensor, fit back, compare.
    Uses a well-conditioned set of 50 randomly oriented bond vectors.
    """
    np.random.seed(42)
    n_bonds = 50
    vectors = np.random.randn(n_bonds, 3).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    true_tensor = np.array([
        [0.01,  0.005,  0.002],
        [0.005, 0.02,  -0.001],
        [0.002, -0.001, -0.03],
    ], dtype=np.float32)

    d_max = 20000.0
    rdcs = calculate_rdc_from_tensor(jnp.array(vectors), jnp.array(true_tensor), d_max=d_max)
    fitted = np.array(fit_saupe_tensor(jnp.array(vectors), rdcs, d_max=d_max))

    np.testing.assert_allclose(true_tensor, fitted, atol=1e-5,
                               err_msg="Fitted tensor does not match the ground truth")
    print("✅ Saupe tensor round-trip recovery verified!")


if __name__ == "__main__":
    test_fitted_tensor_is_traceless_and_symmetric()
    test_fitted_tensor_recovers_ground_truth()
