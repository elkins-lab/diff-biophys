import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_saxs_forward_scattering_identity():
    """
    Verify the q→0 limit of the Debye formula:

        I(0) = (Σ_i f_i)²

    This follows directly from sinc(0) = 1, so all cross-terms contribute
    fully at zero scattering angle.  The result must hold regardless of
    molecular geometry.
    """
    # Three atoms with different form factors; positions are arbitrary
    coords = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    # Use a very small but finite q to avoid numerical issues at exactly 0
    q_vals = jnp.array([1e-4])
    f_vals = jnp.array([[6.0], [7.0], [8.0]])  # C, N, O electrons

    I0 = float(debye_saxs(coords, q_vals, f_vals)[0])
    expected = (6.0 + 7.0 + 8.0) ** 2  # = 441.0

    np.testing.assert_allclose(
        I0, expected, rtol=1e-3, err_msg=f"I(q→0) = {I0:.3f}, expected {expected:.3f}"
    )
    print(f"✅ Forward scattering I(q→0) = {I0:.3f} (expected {expected:.3f})")


def test_saxs_forward_scattering_geometry_independence():
    """
    I(q→0) = (Σ f_i)² must be independent of the molecular geometry —
    confirmed by comparing two very different configurations of the same atoms.
    """
    f_vals = jnp.array([[6.0], [7.0], [8.0]])
    q_vals = jnp.array([1e-4])
    expected = (6.0 + 7.0 + 8.0) ** 2

    # Compact cluster
    coords_compact = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Extended chain
    coords_extended = jnp.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    I_compact = float(debye_saxs(coords_compact, q_vals, f_vals)[0])
    I_extended = float(debye_saxs(coords_extended, q_vals, f_vals)[0])

    np.testing.assert_allclose(I_compact, expected, rtol=1e-3)
    np.testing.assert_allclose(I_extended, expected, rtol=1e-3)
    print(f"✅ I(q→0) is geometry-independent: compact={I_compact:.2f}, extended={I_extended:.2f}")


if __name__ == "__main__":
    test_saxs_forward_scattering_identity()
    test_saxs_forward_scattering_geometry_independence()
