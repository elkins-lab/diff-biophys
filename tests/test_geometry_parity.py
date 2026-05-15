import sys
import os
import numpy as np
import jax.numpy as jnp
import pytest

# Add parent directories to path for imports
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../synth-pdb'))

from diff_biophys.geometry.nerf import position_atom_3d
from synth_pdb.geometry.nerf import position_atom_3d_from_internal_coords

def test_nerf_parity():
    # Test Data
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.46, 0.0, 0.0])
    p3 = np.array([2.01, 1.34, 0.0])
    
    bond_len = 1.33
    angle_deg = 116.0
    dihedral_deg = 180.0
    
    # synth-pdb version (NumPy)
    res_numpy = position_atom_3d_from_internal_coords(
        p1, p2, p3, bond_len, angle_deg, dihedral_deg
    )
    
    # diff-biophys version (JAX)
    res_jax = position_atom_3d(
        jnp.array(p1), 
        jnp.array(p2), 
        jnp.array(p3), 
        bond_len, 
        np.deg2rad(angle_deg), 
        np.deg2rad(dihedral_deg)
    )
    
    # Assert parity
    np.testing.assert_allclose(res_numpy, np.array(res_jax), atol=1e-5)
    print('NeRF Parity Verified!')

if __name__ == '__main__':
    test_nerf_parity()
