import jax.numpy as jnp
from jax import jit

@jit
def compute_bond_lengths(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute bond lengths between adjacent atoms.
    """
    vectors = coords[1:] - coords[:-1]
    return jnp.linalg.norm(vectors, axis=-1)

@jit
def compute_bond_angles(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute bond angles (in radians) between three adjacent atoms.
    """
    v1 = coords[:-2] - coords[1:-1]
    v2 = coords[2:] - coords[1:-1]
    
    v1_norm = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-10)
    v2_norm = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-10)
    
    cos_angle = jnp.sum(v1_norm * v2_norm, axis=-1)
    return jnp.acos(jnp.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))

@jit
def compute_dihedrals(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute dihedral angles (in radians) for four adjacent atoms.
    Follows the IUPAC convention and matches synth-pdb.
    Uses the robust Praxeolitic formula.
    """
    # Vectors: p1-p2, p3-p2, p4-p3
    b0 = coords[:-3] - coords[1:-2]
    b1 = coords[2:-1] - coords[1:-2]
    b2 = coords[3:] - coords[2:-1]
    
    # Normalize b1
    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True)
    u1 = b1 / (b1_norm + 1e-10)
    
    # v = orthogonal component of b0 with respect to b1
    v = b0 - jnp.sum(b0 * u1, axis=-1, keepdims=True) * u1
    # w = orthogonal component of b2 with respect to b1
    w = b2 - jnp.sum(b2 * u1, axis=-1, keepdims=True) * u1
    
    # x = dot product of v and w
    x = jnp.sum(v * w, axis=-1)
    # y = dot product of cross(u1, v) and w
    y = jnp.sum(jnp.cross(u1, v) * w, axis=-1)
    
    return jnp.atan2(y, x)
