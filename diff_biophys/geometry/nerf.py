import jax.numpy as jnp
from jax import jit

@jit
def position_atom_3d(p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, 
                     bond_length: float, bond_angle_rad: float, dihedral_angle_rad: float) -> jnp.ndarray:
    """
    Differentiable NeRF implementation in JAX.
    
    Args:
        p1: Position of atom 1
        p2: Position of atom 2
        p3: Position of atom 3
        bond_length: Length of the bond p3-p4
        bond_angle_rad: Angle formed by p2-p3-p4 in radians
        dihedral_angle_rad: Dihedral angle formed by p1-p2-p3-p4 in radians
        
    Returns:
        jnp.ndarray: 3D position of atom 4
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    u2 = v2 / (jnp.linalg.norm(v2) + 1e-10)
    
    n = jnp.cross(v1, u2)
    n /= (jnp.linalg.norm(n) + 1e-10)
    
    m = jnp.cross(n, u2)
    
    p4 = p3 + bond_length * (
        -jnp.cos(bond_angle_rad) * u2 
        - jnp.sin(bond_angle_rad) * jnp.cos(dihedral_angle_rad) * m 
        + jnp.sin(bond_angle_rad) * jnp.sin(dihedral_angle_rad) * n
    )
    return p4
