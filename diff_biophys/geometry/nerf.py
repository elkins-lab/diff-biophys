from typing import Any, cast

import jax.numpy as jnp
from jax import jit, lax


@jit
def position_atom_3d(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    bond_length: jnp.ndarray,
    bond_angle_rad: jnp.ndarray,
    dihedral_angle_rad: jnp.ndarray,
) -> jnp.ndarray:
    """
    Differentiable NeRF implementation in JAX for a single atom.

    Places atom p4 given three reference atoms (p1, p2, p3) and the internal
    coordinates (bond length, bond angle, dihedral angle) that define its
    position relative to p3.

    Args:
        p1: (3,) first reference atom coordinate.
        p2: (3,) second reference atom coordinate.
        p3: (3,) third reference atom coordinate (parent of p4).
        bond_length: Scalar distance p3→p4 in Ångströms.
        bond_angle_rad: Scalar bond angle ∠(p2, p3, p4) in radians.
        dihedral_angle_rad: Scalar dihedral angle ∠(p1, p2, p3, p4) in radians.

    Returns:
        jnp.ndarray: (3,) Cartesian coordinates of the new atom p4.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    u2 = v2 / (jnp.linalg.norm(v2) + 1e-10)

    n = jnp.cross(v1, u2)
    n /= jnp.linalg.norm(n) + 1e-10

    m = jnp.cross(n, u2)

    p4 = p3 + bond_length * (
        -jnp.cos(bond_angle_rad) * u2
        - jnp.sin(bond_angle_rad) * jnp.cos(dihedral_angle_rad) * m
        - jnp.sin(bond_angle_rad) * jnp.sin(dihedral_angle_rad) * n
    )
    return cast(jnp.ndarray, p4)


@jit
def chain_nerf(
    init_coords: jnp.ndarray,
    bond_lengths: jnp.ndarray,
    bond_angles: jnp.ndarray,
    dihedrals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build a chain of atoms using the NeRF algorithm.

    Args:
        init_coords: (3, 3) initial coordinates for the first 3 atoms
        bond_lengths: (N,) bond lengths for atoms 4 to N+3
        bond_angles: (N,) bond angles (in radians) for atoms 4 to N+3
        dihedrals: (N,) dihedral angles (in radians) for atoms 4 to N+3

    Returns:
        jnp.ndarray: (N+3, 3) coordinates for the entire chain
    """

    def body_fun(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], i: Any
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]:
        p1, p2, p3 = carry

        p4 = position_atom_3d(p1, p2, p3, bond_lengths[i], bond_angles[i], dihedrals[i])
        return (p2, p3, p4), p4

    # Use .shape[0] instead of len() so this works correctly under vmap
    # and with dynamically-shaped arrays during JAX tracing.
    indices = jnp.arange(bond_lengths.shape[0])
    init_carry = (init_coords[0], init_coords[1], init_coords[2])
    _, final_coords = lax.scan(body_fun, init_carry, indices)

    return cast(jnp.ndarray, jnp.concatenate([init_coords, final_coords], axis=0))
