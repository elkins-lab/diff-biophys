import jax
import jax.numpy as jnp

from diff_biophys.geometry.superposition import kabsch_alignment


def test_kabsch_perfect_grad() -> None:
    P = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = P

    R, t = kabsch_alignment(P, Q)
    P_aligned = P @ R.T + t
    diff = P_aligned - Q
    print(f"Diff:\n{diff}")
    print(f"Sum diff sq: {jnp.sum(diff**2)}")

    def loss(mobile: jnp.ndarray) -> jnp.ndarray:
        R, t = kabsch_alignment(mobile, Q)
        P_aligned = mobile @ R.T + t
        # Return squared RMSD to avoid sqrt singularity
        return jnp.mean(jnp.sum((P_aligned - Q) ** 2, axis=-1))

    grad = jax.grad(loss)(P)
    print(f"Squared RMSD Gradient:\n{grad}")


if __name__ == "__main__":
    test_kabsch_perfect_grad()
