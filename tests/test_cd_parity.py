import jax
import jax.numpy as jnp

from diff_biophys.cd.kernels import simulate_cd_matrix


def test_cd_simulation_runs() -> None:
    """
    Verify that the CD matrix-method kernel runs and produces plausible output.
    """
    n_residues = 10
    wavelengths = jnp.linspace(190, 250, 61)

    # Random but reproducible coordinates
    positions = jax.random.normal(jax.random.PRNGKey(42), (n_residues, 3)) * 5.0
    orientations = jax.random.normal(jax.random.PRNGKey(43), (n_residues, 3))
    orientations /= jnp.linalg.norm(orientations, axis=-1, keepdims=True)

    spectrum = simulate_cd_matrix(positions, orientations, wavelengths)

    assert spectrum.shape == (61,)
    assert not jnp.any(jnp.isnan(spectrum))
    # CD signal should be non-zero for random chiral arrangement
    assert jnp.any(spectrum != 0.0)


def test_cd_differentiable() -> None:
    """
    Verify that we can take gradients through the CD simulation.
    """
    n_residues = 5
    wavelengths = jnp.array([200.0, 210.0, 222.0])
    positions = jax.random.normal(jax.random.PRNGKey(44), (n_residues, 3))
    orientations = jax.random.normal(jax.random.PRNGKey(45), (n_residues, 3))
    orientations /= jnp.linalg.norm(orientations, axis=-1, keepdims=True)

    def loss(pos: jnp.ndarray) -> jnp.ndarray:
        spec = simulate_cd_matrix(pos, orientations, wavelengths)
        return jnp.sum(spec**2)

    grads = jax.grad(loss)(positions)
    assert grads.shape == positions.shape
    assert not jnp.any(jnp.isnan(grads))


def test_cd_chirality_flip() -> None:
    """
    Verify that flipping the chirality flips the CD signal sign.
    """
    n_residues = 4
    wavelengths = jnp.array([200.0])
    positions = jax.random.normal(jax.random.PRNGKey(46), (n_residues, 3))
    orientations = jax.random.normal(jax.random.PRNGKey(47), (n_residues, 3))
    orientations /= jnp.linalg.norm(orientations, axis=-1, keepdims=True)

    spec1 = simulate_cd_matrix(positions, orientations, wavelengths)

    # Mirror image: negate one coordinate axis
    positions_mirror = positions.at[:, 0].set(-positions[:, 0])
    orientations_mirror = orientations.at[:, 0].set(-orientations[:, 0])

    spec2 = simulate_cd_matrix(positions_mirror, orientations_mirror, wavelengths)

    # CD is a pseudo-scalar, so mirror image should have opposite sign
    assert jnp.allclose(spec1, -spec2, atol=1e-5)
