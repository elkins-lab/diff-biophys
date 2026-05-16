import jax.numpy as jnp
from diff_biophys.cd.kernels import simulate_cd_matrix

def test_cd_placeholder():
    # Test that the placeholder runs and returns the expected shape
    n_residues = 10
    wavelengths = jnp.linspace(190, 250, 61)
    
    positions = jnp.zeros((n_residues, 3))
    orientations = jnp.zeros((n_residues, 3))
    
    intensities = simulate_cd_matrix(positions, orientations, wavelengths)
    
    assert intensities.shape == wavelengths.shape
    assert jnp.all(intensities == 0.0)
