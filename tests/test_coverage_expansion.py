import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.ensemble import Ensemble
from diff_biophys.geometry.macroscopic import compute_rg
from diff_biophys.geometry.torsions import compute_dihedrals
from diff_biophys.nmr.rdc import calculate_q_factor
from diff_biophys.saxs.kernels import debye_saxs


def test_ensemble_weight_differentiability() -> None:
    """Verify that Ensemble class allows gradients through weights (PyTree test)."""
    coords = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    weights = jnp.array([0.5, 0.5])

    def loss(w: jnp.ndarray) -> jnp.ndarray:
        ens = Ensemble(coords, weights=w)

        # Distance observable
        def get_dist(c: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(jnp.linalg.norm(c[0] - c[1]))

        return ens.calculate_average(get_dist)

    grad = jax.grad(loss)(weights)
    assert jnp.all(jnp.isfinite(grad))
    # Expected: (w0*1 + w1*2)/(w0+w1) -> d/dw0 = -0.5, d/dw1 = 0.5
    np.testing.assert_allclose(grad, jnp.array([-0.5, 0.5]), atol=1e-5)


def test_q_factor_zero_exp_stability() -> None:
    """Verify Q-factor gradient is finite even with zero experimental data."""
    calc = jnp.array([1.0, 2.0])
    exp = jnp.array([0.0, 0.0])

    def loss(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(calculate_q_factor(c, exp))

    grad = jax.grad(loss)(calc)
    assert jnp.all(jnp.isfinite(grad)), f"Q-factor grad should be finite, got {grad}"
    assert jnp.all(grad == 0.0), "Grad should be zero because Q-factor is clamped to 0"


def test_collinear_dihedral_stability() -> None:
    """Verify dihedral gradients are finite for collinear atoms."""
    p1 = jnp.array([0.0, 0.0, 0.0])
    p2 = jnp.array([1.0, 0.0, 0.0])
    p3 = jnp.array([2.0, 0.0, 0.0])  # Collinear
    p4 = jnp.array([2.0, 1.0, 0.0])
    coords = jnp.stack([p1, p2, p3, p4])

    grad = jax.grad(lambda c: jnp.sum(compute_dihedrals(c)))(coords)
    assert jnp.all(jnp.isfinite(grad)), "Collinear dihedral gradient contains NaNs"


def test_saxs_guinier_limit() -> None:
    """
    Science Sanity: I(q) at small q should follow the Guinier approximation:
    I(q) approx I(0) * exp(-q^2 Rg^2 / 3)
    """
    # Create a simple structure (sphere-like)
    n_atoms = 50
    key = jax.random.PRNGKey(42)
    coords = jax.random.normal(key, (n_atoms, 3)) * 10.0

    rg = compute_rg(coords)
    q_values = jnp.linspace(0.001, 0.01, 10)
    ff = jnp.ones((n_atoms, 10))

    iq = debye_saxs(coords, q_values, ff)

    # I(0) = (sum f_i)^2 = (n_atoms)^2
    i0 = n_atoms**2

    # Guinier prediction
    log_iq_pred = jnp.log(i0) - (q_values**2 * rg**2 / 3.0)
    log_iq_actual = jnp.log(iq)

    # Check agreement at very low q
    np.testing.assert_allclose(log_iq_actual[:3], log_iq_pred[:3], rtol=1e-2)


def test_saxs_debye_high_q_limit() -> None:
    """
    Science Sanity: I(q) at high q should approach the sum of squared form factors
    (the independent atom model), as sinc(qr) terms vanish.
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    q_values = jnp.array([100.0])  # Very high q
    ff = jnp.array([[1.0], [1.0]])

    iq = debye_saxs(coords, q_values, ff)
    # Expected: f1^2 + f2^2 = 1^2 + 1^2 = 2.0
    # The interference term is 2*f1*f2*sinc(q*r) = 2*1*1*sinc(1000) approx 0
    np.testing.assert_allclose(iq, 2.0, atol=1e-2)


def test_ensemble_pytree_jit() -> None:
    """Verify that Ensemble can be passed through JIT (PyTree flattening/unflattening)."""
    coords = jnp.zeros((2, 3, 3))
    weights = jnp.array([0.5, 0.5])
    ens = Ensemble(coords, weights=weights)

    @jax.jit
    def get_coords(e: Ensemble) -> jnp.ndarray:
        return e.coords

    res = get_coords(ens)
    assert jnp.all(res == 0.0)
    assert res.shape == (2, 3, 3)
