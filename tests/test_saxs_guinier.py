import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_saxs_guinier_rg():
    """
    Verify that the Guinier approximation recovers the correct radius of
    gyration from the low-q SAXS slope.

    Guinier law:  ln I(q) ≈ ln I(0) − (Rg² / 3) q²

    Geometry: two equal atoms at (0,0,0) and (4,0,0).
      CoM = (2, 0, 0)
      Rg² = (1/N) Σ |r_i − r_com|² = (2² + 2²) / 2 = 4  →  Rg = 2.0 Å
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    q_vals = jnp.linspace(0.001, 0.05, 60)
    ff = jnp.ones((2, 60))

    I = np.array(debye_saxs(coords, q_vals, ff))  # noqa: E741
    q = np.array(q_vals)

    # Use only the first 20 points (genuine Guinier region)
    ln_I = np.log(I[:20])
    q2 = q[:20] ** 2

    slope, _ = np.polyfit(q2, ln_I, 1)
    rg_calc = np.sqrt(-slope * 3)

    np.testing.assert_allclose(
        rg_calc, 2.0, atol=0.05, err_msg=f"Guinier Rg = {rg_calc:.3f} Å, expected 2.0 Å"
    )
    print(f"✅ Guinier Rg = {rg_calc:.3f} Å (expected 2.0 Å)")


if __name__ == "__main__":
    test_saxs_guinier_rg()
