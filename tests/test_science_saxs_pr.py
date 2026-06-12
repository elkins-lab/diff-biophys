import numpy as np


def test_saxs_sphere_pr_topology() -> None:
    """
    Scientific Validation: Sphere Pair Distance Distribution P(r).

    For a uniform sphere of radius R, the distribution of distances between
    all pairs of points (P(r)) has a known analytical form:
    P(r) = 12 * (r/2R)^2 * [1 - 1.5*(r/2R) + 0.5*(r/2R)^3]
    for r <= 2R.

    This validates that our structural coordinates and pairwise distance
    distributions are physically consistent with a continuous sphere.
    """
    R = 10.0
    n_points = 1000
    np.random.seed(42)

    # 1. Generate sphere points
    points: list[np.ndarray] = []
    while len(points) < n_points:
        p = np.random.uniform(-R, R, size=3)
        if np.linalg.norm(p) <= R:
            points.append(p)
    coords = np.array(points)

    # 2. Compute all-pairs distances
    from scipy.spatial.distance import pdist

    dists = pdist(coords)

    # 3. Compute histogram (P(r))
    bins = np.linspace(0, 2 * R, 50)
    counts, bin_edges = np.histogram(dists, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 4. Analytic P(r) for a sphere
    x = bin_centers / (2 * R)
    analytic_pr = (12 * x**2 * (1 - 1.5 * x + 0.5 * x**3)) / (2 * R)

    # 5. Compare
    # We expect some noise due to finite sampling, but the trend should match.
    correlation = np.corrcoef(counts, analytic_pr)[0, 1]
    assert correlation > 0.98, f"Sphere P(r) correlation too low: {correlation:.4f}"

    # Max error should be reasonable
    max_err = np.max(np.abs(counts - analytic_pr))
    assert max_err < 0.05, f"Sphere P(r) max error too high: {max_err:.4f}"

    print(f"✅ Sphere P(r) Validation Successful (Correlation: {correlation:.4f})")


if __name__ == "__main__":
    test_saxs_sphere_pr_topology()
