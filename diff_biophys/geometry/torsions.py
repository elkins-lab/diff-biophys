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
    Follows the IUPAC convention (0 is cis, 180 is trans).
    """
    b0 = coords[1:-2] - coords[:-3]
    b1 = coords[2:-1] - coords[1:-2]
    b2 = coords[3:] - coords[2:-1]
    
    n1 = jnp.cross(b0, b1)
    n2 = jnp.cross(b1, b2)
    
    # Normalize normals
    n1 /= (jnp.linalg.norm(n1, axis=-1, keepdims=True) + 1e-10)
    n2 /= (jnp.linalg.norm(n2, axis=-1, keepdims=True) + 1e-10)
    
    # Unit vector along b1
    u1 = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-10)
    
    # m is perpendicular to n1 and u1
    m1 = jnp.cross(n1, u1)
    
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    
    return jnp.atan2(y, x)
