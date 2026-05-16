import jax.numpy as jnp
import numpy as np
from diff_biophys.nmr import calculate_karplus_j

def test_bax_1993_karplus_validation():
    """
    Validate against Vuister & Bax (1993) JACS parameters for 3J(HN-HA).
    Reference: J. Am. Chem. Soc. 115, 7772-7777.
    """
    # Parameters from Bax 1993
    A, B, C = 6.51, -1.76, 1.60
    
    # 1. Alpha-helix check
    # Typical phi ~ -60 deg
    phi_helix = jnp.radians(-60.0)
    # The Pardi/Bax formula is often J = A cos^2(phi-60) + B cos(phi-60) + C
    # But our kernel is a generic J = a cos^2(theta) + b cos(theta) + c.
    # To use the Bax formula, theta = phi - 60 deg
    theta_helix = phi_helix - jnp.radians(60.0)
    j_helix = calculate_karplus_j(theta_helix, A, B, C)
    
    # Expected range for alpha-helix: 3.0 - 6.0 Hz
    assert 3.0 <= float(j_helix) <= 6.0
    print(f"✅ Karplus Alpha-Helix (phi=-60): {float(j_helix):.2f} Hz (in range 3-6)")
    
    # 2. Beta-sheet check
    # Typical phi ~ -120 deg
    phi_sheet = jnp.radians(-120.0)
    theta_sheet = phi_sheet - jnp.radians(60.0)
    j_sheet = calculate_karplus_j(theta_sheet, A, B, C)
    
    # Expected range for beta-sheet: 8.0 - 10.0 Hz
    assert 8.0 <= float(j_sheet) <= 10.0
    print(f"✅ Karplus Beta-Sheet (phi=-120): {float(j_sheet):.2f} Hz (in range 8-10)")

if __name__ == "__main__":
    test_bax_1993_karplus_validation()
