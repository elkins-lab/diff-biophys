import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr.rdc import calculate_q_factor


def test_q_factor_zero_experimental() -> None:
    """Verify Q-factor handles zero experimental data."""
    # If experimental data is zero, but calculated is not, the error is large.
    # Current implementation returns 0.0 in this case (per comment).
    exp = jnp.zeros(3)
    calc = jnp.array([1.0, 1.0, 1.0])

    q = calculate_q_factor(calc, exp)
    np.testing.assert_allclose(float(q), 0.0, atol=1e-7)


def test_q_factor_all_zero() -> None:
    """Both are zero, should be 0.0."""
    exp = jnp.zeros(3)
    calc = jnp.zeros(3)
    q = calculate_q_factor(calc, exp)
    np.testing.assert_allclose(float(q), 0.0, atol=1e-7)


if __name__ == "__main__":
    test_q_factor_zero_experimental()
    test_q_factor_all_zero()
