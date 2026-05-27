import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_saxs_parity_and_gradients():
    print("Testing SAXS Kernel Parity and Gradients...")

    # 1. Create a dummy structure (Small peptide)
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.5, 0.0], [0.0, 1.5, 1.5]], dtype=np.float32
    )

    # 2. Define q values
    q_values = np.linspace(0.01, 0.5, 10).astype(np.float32)

    # 3. Form factors (Assume constant 1.0 for testing parity)
    form_factors = np.ones((len(coords), len(q_values)), dtype=np.float32)

    # 4. Run NumPy version (Manual reference)
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    expected_intensities = []
    for qi in q_values:
        f_prod = np.ones((len(coords), len(coords)))
        qr = qi * dist
        # Use np.errstate to suppress division by zero warnings in the reference calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc_qr = np.where(qr < 1e-4, 1.0 - (qr**2) / 6.0, np.sin(qr) / qr)
        expected_intensities.append(np.sum(f_prod * sinc_qr))
    expected_intensities = np.array(expected_intensities)

    # 5. Run JAX version
    actual_intensities = debye_saxs(jnp.array(coords), jnp.array(q_values), jnp.array(form_factors))

    # 6. Assert parity
    np.testing.assert_allclose(expected_intensities, np.array(actual_intensities), rtol=1e-5)
    print("✅ SAXS Kernel Parity Verified!")

    # 7. Gradient Check
    def loss_fn(c):
        intensities = debye_saxs(c, jnp.array(q_values), jnp.array(form_factors))
        return jnp.mean(intensities)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(jnp.array(coords))

    assert grads.shape == coords.shape
    assert jnp.all(jnp.isfinite(grads))
    print(f"✅ SAXS Gradients Verified! (Sum of grads: {jnp.sum(grads):.6f})")


if __name__ == "__main__":
    test_saxs_parity_and_gradients()
