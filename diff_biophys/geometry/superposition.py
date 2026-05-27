import jax.numpy as jnp
from jax import jit


@jit
def kabsch_alignment(P: jnp.ndarray, Q: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimal superposition of P onto Q using Kabsch algorithm in JAX.

    Args:
        P: (N, 3) mobile coordinates
        Q: (N, 3) reference coordinates

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: (3x3 rotation matrix, 3-element translation vector)
    """
    p_center = jnp.mean(P, axis=0)
    q_center = jnp.mean(Q, axis=0)

    P_c = P - p_center
    Q_c = Q - q_center

    H = jnp.dot(P_c.T, Q_c)

    U, S, Vt = jnp.linalg.svd(H)

    d = jnp.linalg.det(jnp.dot(Vt.T, U.T))
    step = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, jnp.where(d > 0, 1.0, -1.0)]])

    R = jnp.dot(Vt.T, jnp.dot(step, U.T))
    t = q_center - jnp.dot(R, p_center)

    return R, t
