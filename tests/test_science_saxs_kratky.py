import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_saxs_kratky_plot_topology() -> None:
    """
    Scientific Validation: Kratky Plot Topology (Kratky & Porod, 1949).

    A Kratky plot maps I(q) * q^2 against q.
    It is a classic biophysical test for protein fold state:
    1. Globular (compact) structures produce a distinct bell-shaped peak.
    2. Random coil (unfolded) structures do not peak; they plateau or rise at high q.
    """
    # 1. Generate a globular structure (dense sphere)
    np.random.seed(42)
    n_points = 200
    glob_list: list[np.ndarray] = []
    while len(glob_list) < n_points:
        p = np.random.uniform(-10.0, 10.0, size=3)
        if np.linalg.norm(p) <= 10.0:
            glob_list.append(p)
    globular_coords = jnp.array(glob_list)

    # 2. Generate a random coil (unfolded string-like structure)
    # A simple 1D random walk
    steps = np.random.randn(n_points, 3) * 3.8  # ~CA-CA distance
    coil_coords = jnp.array(np.cumsum(steps, axis=0))

    q_values = jnp.linspace(0.01, 0.5, 100)
    ff = jnp.ones((n_points, 100))

    # Calculate scattering
    i_globular = np.array(debye_saxs(globular_coords, q_values, ff))
    i_coil = np.array(debye_saxs(coil_coords, q_values, ff))

    # Kratky transform: I(q) * q^2
    kratky_globular = i_globular * (np.array(q_values) ** 2)
    kratky_coil = i_coil * (np.array(q_values) ** 2)

    # Validate Globular behavior (Must have a clear peak)
    max_idx_glob = np.argmax(kratky_globular)
    # The peak should not be at the very end of the q-range
    assert max_idx_glob < len(q_values) - 10, "Globular Kratky plot failed to peak and descend."

    # Validate Random Coil behavior (Should plateau/rise, peak much later if at all)
    max_idx_coil = np.argmax(kratky_coil)
    # For a random walk, the max of q^2 I(q) in this range is typically at the very end
    assert max_idx_coil > max_idx_glob, "Random coil should peak at much higher q than globular."

    # Specifically, at high q, globular should drop significantly from its peak
    drop_ratio = kratky_globular[-1] / kratky_globular[max_idx_glob]
    assert drop_ratio < 0.5, f"Globular Kratky did not drop sufficiently: {drop_ratio:.2f}"

    print("✅ Kratky Plot Topology Validation Successful!")


if __name__ == "__main__":
    test_saxs_kratky_plot_topology()
