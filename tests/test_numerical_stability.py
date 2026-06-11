import jax.numpy as jnp
import numpy as np

from diff_biophys.cd.kernels import simulate_cd_matrix
from diff_biophys.saxs.kernels import debye_saxs


def test_debye_saxs_singularities() -> None:
    """Verify debye_saxs handles q=0 and r=0 (overlapping atoms)."""
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Overlapping
    q_values = jnp.array([0.0, 0.1, 1.0])
    form_factors = jnp.ones((2, 3))

    # I(q) = sum_i sum_j f_i f_j sinc(q r_ij)
    # For overlapping atoms, r_ij = 0, sinc(0) = 1.
    # I(q) = f_1^2 + f_2^2 + 2 f_1 f_2 = (f_1 + f_2)^2 = 4.0

    intensities = debye_saxs(coords, q_values, form_factors)
    np.testing.assert_allclose(np.array(intensities), 4.0, atol=1e-5)


def test_cd_matrix_singularities() -> None:
    """Verify CD simulation handles overlapping chromophores."""
    # This shouldn't happen in real proteins but the kernel should be robust.
    # Overlapping chromophores will have r_ij = 0, which would lead to 1/0.
    # simulate_cd_matrix uses jnp.where(dist_sq > 0, ...) to handle this.

    pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dipoles = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wavelengths = jnp.array([190.0, 200.0])

    spectrum = simulate_cd_matrix(pos, dipoles, wavelengths)
    # Result should be finite (not NaN)
    assert jnp.all(jnp.isfinite(spectrum))
    # For zero distance, interaction V is zeroed out by the mask,
    # so we get independent response sum.
    # Since R_ij involves diff = 0, cd_val should be 0.
    np.testing.assert_allclose(np.array(spectrum), 0.0, atol=1e-5)


if __name__ == "__main__":
    test_debye_saxs_singularities()
    test_cd_matrix_singularities()
