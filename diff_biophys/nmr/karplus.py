import jax.numpy as jnp
from jax import jit


@jit
def calculate_karplus_j(theta: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    """
    Calculate 3J coupling constants using the Karplus equation.

    J = a * cos^2(theta) + b * cos(theta) + c

    .. important::
        ``theta`` is **not** the same as the raw backbone dihedral ``phi``.
        For ³J(H_N, H_α) couplings, the Karplus dihedral is offset from
        the IUPAC backbone angle by 60°::

            theta = phi - 60°   (i.e. subtract 60° from phi before calling)

        Passing raw ``phi`` directly produces errors of ~3–4 Hz, invalidating
        the α-helix / β-sheet discrimination.

        The default parameters (A=6.51, B=−1.76, C=1.60) are for H_N–H_α
        (Vuister & Bax 1993).  Different spin pairs require different offsets
        and parameters.

    Args:
        theta: (N,) Karplus dihedral angle in radians.
               For ³J(HN,HA): pass (phi − 60°), **not** phi.
        a: Cosine-squared coefficient (Hz).
        b: Cosine coefficient (Hz).
        c: Constant offset (Hz).

    Returns:
        jnp.ndarray: (N,) Calculated J-couplings.
    """
    cos_theta = jnp.cos(theta)
    return a * (cos_theta**2) + b * cos_theta + c
