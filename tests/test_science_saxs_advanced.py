import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_saxs_hydration_shell_impact():
    """
    Verify that the hydration shell (solvent subtraction) reduces scattering
    intensity at low q as expected.
    """
    # 1. Setup a simple dimer
    coords = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    q_values = jnp.array([0.01, 0.05, 0.1])
    form_factors = jnp.ones((2, 3)) * 6.0  # Mock Carbon-like

    # 2. Vacuum Intensity
    i_vac = debye_saxs(coords, q_values, form_factors, volumes=None)

    # 3. Solvated Intensity (with displaced volumes)
    # Using typical Carbon volume ~16.4 A^3
    volumes = jnp.array([16.4, 16.4])
    i_sol = debye_saxs(coords, q_values, form_factors, volumes=volumes, solvent_density=0.334)

    # Check that intensity is reduced (I_sol < I_vac)
    # This is the "contrast" effect: protein is less dense relative to solvent than to vacuum.
    assert jnp.all(i_sol < i_vac)
    print(f"Vacuum I(q=0.01): {i_vac[0]:.4f}")
    print(f"Solvent I(q=0.01): {i_sol[0]:.4f}")

    # 4. Correctness check: I(q=0) should be approx (Total_Electrons - Displaced_Electrons)^2
    # For a point at q=0, the sinc is 1.0.
    # f_eff = 6.0 - (0.334 * 16.4) = 6.0 - 5.4776 = 0.5224
    # Total effective electrons = 0.5224 * 2 = 1.0448
    # I(0) = (1.0448)^2 = 1.0916
    # Our q=0.01 should be very close to this.
    np.testing.assert_allclose(i_sol[0], 1.0916, atol=0.1)

    print("✅ SAXS Hydration Shell Impact Verified!")


def test_saxs_solvated_differentiability():
    """
    Verify that we can take gradients through the solvated SAXS model.
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    q_values = jnp.array([0.1])
    form_factors = jnp.ones((2, 1)) * 6.0
    volumes = jnp.array([16.4, 16.4])

    def loss_fn(c):
        return jnp.sum(debye_saxs(c, q_values, form_factors, volumes=volumes))

    grads = jax.grad(loss_fn)(coords)

    # Gradients should be non-zero (atoms should "feel" each other via the profile)
    assert jnp.any(jnp.abs(grads) > 0.0)
    print("✅ SAXS Solvated Differentiability Verified!")


if __name__ == "__main__":
    test_saxs_hydration_shell_impact()
    test_saxs_solvated_differentiability()
