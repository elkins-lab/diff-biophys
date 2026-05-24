import jax
import jax.numpy as jnp
import numpy as np
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts, OFFSET_HELIX, OFFSET_SHEET

def test_ca_shift_secondary_structure_dependency():
    """
    Verify that CA shifts respond correctly to secondary structure (Phi/Psi).
    """
    # 1. Baseline RC shifts
    rc_shifts = jnp.array([55.0, 55.0, 55.0])
    
    # 2. Test Cases: [Helix, Sheet, Coil]
    # Helix: Phi ~ -60, Psi ~ -45
    # Sheet: Phi ~ -120, Psi ~ 135
    # Coil:  Phi ~ 60, Psi ~ 60 (arbitrary coil)
    phi = jnp.array([jnp.radians(-60.0), jnp.radians(-120.0), jnp.radians(60.0)])
    psi = jnp.array([jnp.radians(-45.0), jnp.radians(135.0), jnp.radians(60.0)])
    
    shifts = predict_ca_shifts(phi, psi, rc_shifts)
    
    # Assertions based on SPARTA+ / synth-nmr logic:
    # Helix should have +OFFSET_HELIX
    # Sheet should have +OFFSET_SHEET
    # Coil should be close to RC
    
    # Use a tolerance because of the "soft" Gaussian classification
    np.testing.assert_allclose(shifts[0], 55.0 + OFFSET_HELIX, atol=0.2)
    np.testing.assert_allclose(shifts[1], 55.0 + OFFSET_SHEET, atol=0.2)
    np.testing.assert_allclose(shifts[2], 55.0, atol=0.1)
    
    print("✅ CA Shift Secondary Structure Dependency Verified!")

def test_ca_shift_differentiability():
    """
    Verify that we can take gradients of the shift with respect to torsions.
    """
    rc_shifts = jnp.array([55.0])
    phi = jnp.array([jnp.radians(-60.0)])
    psi = jnp.array([jnp.radians(-45.0)])
    
    def loss_fn(p):
        return jnp.sum(predict_ca_shifts(p, psi, rc_shifts))
    
    grad_phi = jax.grad(loss_fn)(phi)
    
    # Gradient should not be zero at the helix boundary
    assert jnp.abs(grad_phi) > 0.0
    print("✅ CA Shift Differentiability Verified!")

if __name__ == "__main__":
    test_ca_shift_secondary_structure_dependency()
    test_ca_shift_differentiability()
