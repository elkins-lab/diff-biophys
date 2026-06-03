from typing import Any, cast

import jax.numpy as jnp
from jax import vmap

# Atomic volumes (A^3) from Pavlov & Svergun (1997)
VOLUMES = {
    "H": 5.15,
    "C": 16.44,
    "N": 14.0,
    "O": 12.0,
    "S": 19.86,
    "P": 24.4,
}


def debye_saxs(
    coords: jnp.ndarray,
    q_values: jnp.ndarray,
    form_factors: jnp.ndarray,
    volumes: jnp.ndarray | None = None,
    solvent_density: float = 0.334,
) -> jnp.ndarray:
    """
    Differentiable Debye Formula in JAX with optional solvent subtraction.

    Note: This function is NOT decorated with ``@jit`` because the
    ``volumes`` argument may be ``None`` (a Python sentinel that is
    resolved at trace time, not at runtime).  JIT-compile the *call site*
    instead, e.g.::

        jitted_debye = jax.jit(lambda c: debye_saxs(c, q, ff, volumes=vols))

    Args:
        coords: (N, 3) atomic coordinates in Ångströms.
        q_values: (M,) scattering vector magnitudes (Å⁻¹).
        form_factors: (N, M) q-dependent vacuum atomic form factors.
        volumes: (N,) atomic volumes (Å³) for excluded-volume correction.
            Pass ``None`` (default) to skip solvent subtraction.
        solvent_density: Bulk solvent electron density (e/Å³).
            Default 0.334 e/Å³ for water.

    Returns:
        jnp.ndarray: Scattering intensities I(q), shape (M,).
    """
    # 1. Pairwise distances (N, N)
    sq_norms = jnp.sum(coords**2, axis=-1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * jnp.dot(coords, coords.T)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 0.0) + 1e-12)

    # 2. Effective form factors with optional solvent correction.
    # When volumes=None we use a zero-volume array so the code path is
    # identical (JIT-safe) and the correction term vanishes.
    if volumes is None:
        f_eff = form_factors
    else:
        # Effective radius for excluded volume: V = (4/3) π R³  →  R = (3V/4π)^(1/3)
        r_eff = (3.0 * volumes / (4.0 * jnp.pi)) ** (1.0 / 3.0)

        # Gaussian decay for the excluded-volume envelope (Fraser et al. 1978)
        # f_eff(q) = f_vac(q) - ρ_sol · V · exp(−(q·r_eff)² / (4π))
        decay = jnp.exp(-((q_values[None, :] * r_eff[:, None]) ** 2) / (4.0 * jnp.pi))
        f_eff = form_factors - (solvent_density * volumes[:, None] * decay)

    # 3. Debye sum: I(q) = Σ_i Σ_j f_i(q) f_j(q) sinc(q r_ij)
    def compute_intensity(q_idx: Any) -> Any:
        q = q_values[q_idx]

        f_q = f_eff[:, q_idx]
        f_prod = f_q[:, None] * f_q[None, :]
        qr = q * dist
        # Taylor expansion for qr→0; standard formula elsewhere.
        # The epsilon is in the denominator only, *not* inside sin(), to
        # avoid introducing a phase error at large qr.
        sinc_qr = jnp.where(
            qr < 1e-4,
            1.0 - (qr**2) / 6.0,
            jnp.sin(qr) / (qr + 1e-10),
        )
        return jnp.sum(f_prod * sinc_qr)

    return cast(jnp.ndarray, vmap(compute_intensity)(jnp.arange(len(q_values))))
