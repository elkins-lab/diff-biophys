import jax.numpy as jnp

from diff_biophys.geometry.superposition import kabsch_alignment


def test_kabsch_single_point() -> None:
    P = jnp.array([[0.0, 0.0, 0.0]])
    Q = jnp.array([[1.0, 1.0, 1.0]])

    R, t = kabsch_alignment(P, Q)
    print(f"R:\n{R}")
    print(f"t: {t}")

    P_aligned = P @ R.T + t
    print(f"P_aligned: {P_aligned}")
    assert jnp.all(jnp.isfinite(R))
    assert jnp.all(jnp.isfinite(t))
    assert jnp.allclose(P_aligned, Q, atol=1e-5)


if __name__ == "__main__":
    test_kabsch_single_point()
