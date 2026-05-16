import jax.numpy as jnp
from jax import jit

@jit
def calculate_rdc_from_tensor(bond_vectors: jnp.ndarray, saupe_tensor: jnp.ndarray, d_max: float = 1.0) -> jnp.ndarray:
    """
    Calculate RDCs from a full 3x3 Saupe alignment tensor.
    D = d_max * sum_ij (v_i * S_ij * v_j)
    
    Args:
        bond_vectors: (N, 3) unit vectors
        saupe_tensor: (3, 3) symmetric traceless Saupe tensor
        d_max: Maximum dipolar coupling constant (Hz)
        
    Returns:
        jnp.ndarray: Calculated RDCs (N,)
    """
    # Vectorized computation of v^T S v
    return d_max * jnp.einsum('ni,ij,nj->n', bond_vectors, saupe_tensor, bond_vectors)

@jit
def fit_saupe_tensor(bond_vectors: jnp.ndarray, experimental_rdcs: jnp.ndarray, d_max: float = 1.0) -> jnp.ndarray:
    """
    Fit a Saupe alignment tensor to experimental RDCs using SVD (least squares).
    
    The RDC formula can be rewritten as D = A * s
    where s = [Sxx, Syy, Sxy, Sxz, Syz] (5 independent components)
    
    Args:
        bond_vectors: (N, 3) unit vectors
        experimental_rdcs: (N,) measured RDCs in Hz
        d_max: Maximum dipolar coupling constant (Hz)
        
    Returns:
        jnp.ndarray: (3, 3) Fitted Saupe tensor
    """
    x = bond_vectors[:, 0]
    y = bond_vectors[:, 1]
    z = bond_vectors[:, 2]
    
    # Basis functions for the 5 independent components
    # Using the identity Szz = -Sxx - Syy
    # D = d_max * [ Sxx*x^2 + Syy*y^2 + Szz*z^2 + 2Sxy*xy + 2Sxz*xz + 2Syz*yz ]
    # D = d_max * [ Sxx(x^2 - z^2) + Syy(y^2 - z^2) + 2Sxy*xy + 2Sxz*xz + 2Syz*yz ]
    
    A = d_max * jnp.stack([
        x**2 - z**2,
        y**2 - z**2,
        2 * x * y,
        2 * x * z,
        2 * y * z
    ], axis=1)
    
    # Solve A * s = experimental_rdcs
    s, _, _, _ = jnp.linalg.lstsq(A, experimental_rdcs)
    
    sxx, syy, sxy, sxz, syz = s
    szz = -(sxx + syy)
    
    tensor = jnp.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ])
    
    return tensor

@jit
def calculate_q_factor(calculated_rdcs: jnp.ndarray, experimental_rdcs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the RDC Q-factor (Cornilescu et al., 1998).
    Q = sqrt( sum((D_calc - D_exp)^2) / sum(D_exp^2) )
    
    Args:
        calculated_rdcs: (N,) calculated couplings.
        experimental_rdcs: (N,) measured couplings.
        
    Returns:
        jnp.ndarray: Scalar Q-factor.
    """
    diff_sq = jnp.sum((calculated_rdcs - experimental_rdcs)**2)
    exp_sq = jnp.sum(experimental_rdcs**2)
    return jnp.sqrt(diff_sq / (exp_sq + 1e-10))

@jit
def calculate_rdc(bond_vectors: jnp.ndarray, da: float, r: float) -> jnp.ndarray:
    """
    Differentiable RDC calculation in the principal frame.
    
    Args:
        bond_vectors: (N, 3) unit vectors in the tensor's principal frame
        da: Axial component in Hz
        r: Rhombicity (0 <= R <= 2/3)
    """
    x, y, z = bond_vectors[:, 0], bond_vectors[:, 1], bond_vectors[:, 2]
    
    axial = 3.0 * z**2 - 1.0
    rhombic = 1.5 * r * (x**2 - y**2)
    
    return da * (axial + rhombic)
