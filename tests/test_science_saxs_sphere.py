import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def analytic_sphere_intensity(q: jnp.ndarray, R: float, rho: float = 1.0) -> jnp.ndarray:
    """
    Analytic scattering intensity for a uniform sphere of radius R.
    I(q) = [V * rho * 3 * (sin(qR) - qR cos(qR)) / (qR)^3]^2
    """
    V = (4 / 3) * jnp.pi * R**3
    qR = q * R
    # Limit for qR -> 0 to avoid division by zero
    f = jnp.where(qR < 1e-4, 1.0 - (qR**2) / 10.0, 3 * (jnp.sin(qR) - qR * jnp.cos(qR)) / (qR**3))
    return (V * rho * f) ** 2


def test_saxs_debye_vs_analytic_sphere() -> None:
    """
    Validate that a dense collection of points approximating a sphere
    reproduces the analytic sphere scattering profile.
    """
    # 1. Create a dense sphere of points
    R = 10.0
    n_points = 500
    np.random.seed(42)

    # Simple rejection sampling for a sphere
    points: list[np.ndarray] = []
    while len(points) < n_points:
        p = np.random.uniform(-R, R, size=3)
        if np.linalg.norm(p) <= R:
            points.append(p)
    coords = jnp.array(points)

    # 2. Setup q-values
    q_values = jnp.linspace(0.01, 0.5, 30)

    # 3. Each point has a form factor representing its volume element
    # Total volume V = 4/3 * pi * R^3
    # Each point represents V / n_points
    vol_element = ((4 / 3) * np.pi * R**3) / n_points
    form_factors = jnp.ones((n_points, len(q_values))) * vol_element

    # 4. Calculate using Debye formula (vacuum)
    i_debye = debye_saxs(coords, q_values, form_factors, volumes=None)

    # 5. Calculate Analytic
    i_analytic = analytic_sphere_intensity(q_values, R)

    # 6. Compare (normalized at q=0)
    # The Debye sum will have a self-term (N * vol_element^2) and pairwise terms.
    # At q=0, Debye = (N * vol_element)^2 = V^2.
    # Analytic at q=0 is also V^2.

    # We expect good agreement in the low-q (Guinier) region.
    # At high q, the discrete sampling will deviate from the continuous analytic form.
    guinier_mask = q_values < 0.15

    # Relative error check
    rel_error = jnp.abs(i_debye[guinier_mask] - i_analytic[guinier_mask]) / i_analytic[guinier_mask]

    avg_error = float(jnp.mean(rel_error))
    print(f"Average relative error in Guinier region: {avg_error:.4f}")

    # 10% tolerance for a discrete approximation of a continuous sphere
    assert avg_error < 0.10

    print("✅ SAXS Debye vs Analytic Sphere Validation Successful!")


if __name__ == "__main__":
    test_saxs_debye_vs_analytic_sphere()
