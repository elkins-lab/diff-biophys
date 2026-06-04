import jax
import jax.numpy as jnp


def simulate_cd_matrix(
    peptide_positions: jnp.ndarray,
    dipole_orientations: jnp.ndarray,
    wavelengths: jnp.ndarray,
    f_osc: float = 0.2,
    gamma: float = 10.0,
    lambda_0: float = 190.0,
) -> jnp.ndarray:
    """
    Matrix-Method CD Simulation (DeVoe Theory).

    Implements the coupled-oscillator model for transition dipole coupling.
    Calculates the interaction matrix and solves for the complex polarizability
    response to determine molar ellipticity.

    Args:
        peptide_positions: (N, 3) positions of amide chromophores in Angstroms.
        dipole_orientations: (N, 3) unit vectors for transition dipoles.
        wavelengths: (M,) wavelengths in nm to simulate.
        f_osc: Oscillator strength of the transition (default 0.2 for pi->pi*).
        gamma: Linewidth parameter in nm (default 10.0).
        lambda_0: Resonance wavelength in nm (default 190.0).

    Returns:
        Molar ellipticity [θ] in deg cm^2 / dmol (M,).
    """
    n_chromophores = peptide_positions.shape[0]

    # 1. Compute dipole-dipole interaction matrix V_ij
    # V_ij = (1/r^3) * [ mu_i . mu_j - 3(mu_i . r_ij)(mu_j . r_ij) ]
    diff = peptide_positions[:, None, :] - peptide_positions[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    # Safe distance for gradients (avoid sqrt(0) and 1/0)
    # 1e-9 is a safe epsilon for float32
    mask = dist_sq > 0
    safe_dist_sq = jnp.where(mask, dist_sq, 1.0)
    r_ij = jnp.sqrt(safe_dist_sq)
    r_ij_inv3 = jnp.where(mask, 1.0 / r_ij**3, 0.0)

    # Unit vectors between chromophores
    r_hat = diff * jnp.where(mask[:, :, None], 1.0 / r_ij[:, :, None], 0.0)

    # Dot products
    mu_i_mu_j = jnp.sum(dipole_orientations[:, None, :] * dipole_orientations[None, :, :], axis=-1)
    mu_i_r = jnp.sum(dipole_orientations[:, None, :] * r_hat, axis=-1)
    mu_j_r = jnp.sum(dipole_orientations[None, :, :] * r_hat, axis=-1)

    # Interaction energy V (N, N)
    V = r_ij_inv3 * (mu_i_mu_j - 3 * mu_i_r * mu_j_r)

    # 2. Frequency-dependent response
    def compute_at_wavelength(lmbda: jnp.ndarray) -> jnp.ndarray:
        # Complex polarizability alpha(lambda)
        # Lorentzian-like response
        denom = (1.0 / lmbda**2 - 1.0 / lambda_0**2) + 1j * (gamma / (lmbda * lambda_0**2))
        alpha = f_osc / denom

        # Interaction matrix (I - alpha * V)
        # alpha is scalar for all identical chromophores here
        M = jnp.eye(n_chromophores) - alpha * V

        # We'll use the matrix inverse to find the coupled response
        # Note: jnp.linalg.inv is differentiable but can be sensitive
        inv_M = jnp.linalg.inv(M)

        # Geometric factor for CD (Scalar triple product mu_i x mu_j . r_ij)
        # This represents the chiral arrangement.
        cross_mu = jnp.cross(dipole_orientations[:, None, :], dipole_orientations[None, :, :])
        R_ij = jnp.sum(cross_mu * diff, axis=-1)

        # Total CD response at this wavelength
        coupled_V = inv_M @ (alpha * V)
        cd_val = jnp.imag(jnp.sum(coupled_V * R_ij))

        return cd_val

    # Vectorize over wavelengths
    cd_spectrum = jax.vmap(compute_at_wavelength)(wavelengths)

    # Scale to molar ellipticity (arbitrary units for this kernel,
    # should be calibrated to exp data)
    return cd_spectrum * 1e5
