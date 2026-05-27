import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr import calculate_q_factor


def test_q_factor_behavior():
    """
    Validate the RDC Q-factor (Cornilescu et al., 1998) behavior.
    """
    # 1. Perfect Match
    exp = jnp.array([10.0, -5.0, 2.0])
    calc_perfect = jnp.array([10.0, -5.0, 2.0])
    q_perfect = calculate_q_factor(calc_perfect, exp)
    np.testing.assert_allclose(float(q_perfect), 0.0, atol=1e-7)

    # 2. Total mismatch (zeros)
    calc_zeros = jnp.zeros_like(exp)
    q_zeros = calculate_q_factor(calc_zeros, exp)
    # Q = sqrt(sum(exp^2)/sum(exp^2)) = 1.0
    np.testing.assert_allclose(float(q_zeros), 1.0, atol=1e-7)

    # 3. Typical noisy fit
    # exp_rms ~ sqrt(129/3) ~ 6.5
    # diff_rms ~ sqrt(3/3) = 1.0
    # Q ~ 1.0 / 6.5 ~ 0.15
    calc_noisy = exp + jnp.array([1.0, -1.0, 1.0])
    q_noisy = calculate_q_factor(calc_noisy, exp)
    assert 0.1 <= float(q_noisy) <= 0.2
    print(f"✅ RDC Q-factor Validation: {float(q_noisy):.4f} (expected range 0.1-0.2)")


if __name__ == "__main__":
    test_q_factor_behavior()
