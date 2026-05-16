import jax.numpy as jnp
import numpy as np
from diff_biophys.ensemble import Ensemble
from diff_biophys.saxs import debye_saxs

def test_ensemble_averaging():
    """Verify that Ensemble API correctly averages observables."""
    # 1. Create a 2-member ensemble
    # Member 1: atoms at (0,0,0) and (1,0,0) -> dist = 1
    # Member 2: atoms at (0,0,0) and (2,0,0) -> dist = 2
    coords = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    ])
    
    # 2. Setup SAXS
    q_values = jnp.array([0.1, 0.2])
    form_factors = jnp.ones((2, 2))
    
    # 3. Manual Averaging
    saxs1 = debye_saxs(coords[0], q_values, form_factors)
    saxs2 = debye_saxs(coords[1], q_values, form_factors)
    expected_avg = (saxs1 + saxs2) / 2.0
    
    # 4. Ensemble API
    ens = Ensemble(coords)
    actual_avg = ens.calculate_average(debye_saxs, q_values, form_factors)
    
    np.testing.assert_allclose(actual_avg, expected_avg, atol=1e-5)
    print("✅ Ensemble Averaging Verified!")

def test_weighted_ensemble():
    """Verify weighted averaging."""
    coords = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    ])
    weights = jnp.array([0.75, 0.25])
    
    # Simple observable function: just return the distance between atoms
    def get_dist(c):
        return jnp.array([jnp.linalg.norm(c[0] - c[1])])
        
    ens = Ensemble(coords, weights=weights)
    avg_dist = ens.calculate_average(get_dist)
    
    expected_dist = 0.75 * 1.0 + 0.25 * 2.0
    np.testing.assert_allclose(avg_dist, jnp.array([expected_dist]), atol=1e-5)
    print("✅ Weighted Ensemble Verified!")

def test_ensemble_saxs_utility():
    """Verify the convenience utility for ensemble SAXS."""
    from diff_biophys.ensemble import calculate_ensemble_saxs
    coords = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    ])
    weights = jnp.array([0.5, 0.5])
    q_values = jnp.array([0.1])
    form_factors = jnp.ones((2, 1))
    
    actual_saxs = calculate_ensemble_saxs(coords, weights, q_values, form_factors)
    
    s1 = debye_saxs(coords[0], q_values, form_factors)
    s2 = debye_saxs(coords[1], q_values, form_factors)
    expected_saxs = (s1 + s2) / 2.0
    
    np.testing.assert_allclose(actual_saxs, expected_saxs, atol=1e-5)
    print("✅ Ensemble SAXS Utility Verified!")

if __name__ == "__main__":
    test_ensemble_averaging()
    test_weighted_ensemble()
    test_ensemble_saxs_utility()
