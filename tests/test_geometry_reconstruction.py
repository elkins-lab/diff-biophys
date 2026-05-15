import numpy as np
import jax.numpy as jnp
from diff_biophys.geometry.nerf import chain_nerf
from diff_biophys.geometry.torsions import compute_bond_lengths, compute_bond_angles, compute_dihedrals

def test_geometry_consistency():
    # 1. Create a random chain of 20 atoms
    np.random.seed(42)
    coords = np.random.randn(20, 3).astype(np.float32)
    
    # 2. Extract internal coordinates
    lengths = compute_bond_lengths(jnp.array(coords))
    angles = compute_bond_angles(jnp.array(coords))
    dihedrals = compute_dihedrals(jnp.array(coords))
    
    init_coords = jnp.array(coords[:3])
    
    # Reconstruct using NeRF
    reconstructed = chain_nerf(
        init_coords,
        lengths[2:],
        angles[1:],
        dihedrals[:]
    )
    
    # 4. Assert parity
    np.testing.assert_allclose(coords, np.array(reconstructed), atol=1e-4)
    print("✅ Geometry Reconstruction (NeRF <-> Torsions) Verified!")

if __name__ == '__main__':
    test_geometry_consistency()
