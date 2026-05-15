import sys
import os
import numpy as np
import jax.numpy as jnp

# Add parent directories to path for imports
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../synth-pdb'))

from diff_biophys.nmr.rdc import calculate_rdc
from synth_nmr.rdc import calculate_rdcs as calculate_rdcs_numpy
import biotite.structure as struc

def test_rdc_parity():
    # 1. Create a dummy structure
    # We need a Biotite structure for the numpy version
    atom1 = struc.Atom(coord=[0.0, 0.0, 0.0], atom_name='N', res_id=1, res_name='ALA')
    atom2 = struc.Atom(coord=[0.0, 0.0, 1.0], atom_name='H', res_id=1, res_name='ALA')
    structure = struc.array([atom1, atom2])
    
    da = 10.0
    r = 0.2
    
    # NumPy version
    # calculate_rdcs returns {res_id: value}
    expected_rdcs_dict = calculate_rdcs_numpy(structure, da, r)
    expected_val = expected_rdcs_dict[1]
    
    # JAX version
    # calculate_rdc takes (bond_vectors, da, r)
    # bond_vector is (H - N) / |H - N|
    bond_vector = np.array([[0.0, 0.0, 1.0]])
    actual_rdcs = calculate_rdc(jnp.array(bond_vector), da, r)
    actual_val = actual_rdcs[0]
    
    # Assert parity
    # Note: NumPy version rounds to 2 decimal places
    np.testing.assert_allclose(expected_val, float(actual_val), atol=1e-2)
    print('RDC Parity Verified!')

if __name__ == '__main__':
    test_rdc_parity()
