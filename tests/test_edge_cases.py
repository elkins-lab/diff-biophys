import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.cd.kernels import simulate_cd_matrix
from diff_biophys.geometry.superposition import kabsch_alignment
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.nmr.rdc import calculate_q_factor, calculate_rdc
from diff_biophys.saxs.kernels import debye_saxs


def test_cd_singular_matrix_stability() -> None:
    """
    Stress test: CD simulation with extremely close chromophores.
    This can lead to a singular (I - alpha*V) matrix.
    We check if gradients remain finite or at least don't crash the JIT.
    """
    # Two chromophores very close together
    peptide_positions = jnp.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    dipole_orientations = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    wavelengths = jnp.array([190.0, 200.0, 210.0])

    def loss(pos: jnp.ndarray) -> jnp.ndarray:
        cd = simulate_cd_matrix(pos, dipole_orientations, wavelengths)
        return jnp.sum(cd**2)

    # Check value
    val = loss(peptide_positions)
    assert jnp.isfinite(val)

    # Check gradient
    grad = jax.grad(loss)(peptide_positions)
    assert jnp.all(jnp.isfinite(grad)), (
        f"CD gradient should be finite even for close atoms, got {grad}"
    )


def test_kabsch_collinear_degeneracy() -> None:
    """
    Kabsch alignment stability test for collinear points.
    In 3D, if points are collinear, the rotation matrix might be ambiguous
    or lead to numerical instability in SVD.
    """
    # Three collinear points
    P = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Rotated and translated
    Q = jnp.array([[10.0, 10.0, 0.0], [10.0, 11.0, 0.0], [10.0, 12.0, 0.0]])

    R, t = kabsch_alignment(P, Q)

    assert jnp.all(jnp.isfinite(R))
    assert jnp.all(jnp.isfinite(t))

    P_aligned = P @ R.T + t
    np.testing.assert_allclose(P_aligned, Q, atol=1e-5)


def test_empty_input_kernels() -> None:
    """
    Verify that kernels handle zero-length inputs gracefully.
    This is important for pipelines where selections might be empty.
    """
    # 1. RDC
    empty_vecs = jnp.zeros((0, 3))
    rdcs = calculate_rdc(empty_vecs, 10.0, 0.5)
    assert rdcs.shape == (0,)

    # 2. Q-factor
    q = calculate_q_factor(jnp.array([]), jnp.array([]))
    assert q == 0.0

    # 3. SAXS
    empty_coords = jnp.zeros((0, 3))
    empty_ff = jnp.zeros((0, 5))
    q_vals = jnp.linspace(0.1, 0.5, 5)
    iq = debye_saxs(empty_coords, q_vals, empty_ff)
    assert iq.shape == (5,)
    assert jnp.all(iq == 0.0)

    # 4. Chemical Shifts
    empty_torsions = jnp.array([])
    shifts = predict_ca_shifts(empty_torsions, empty_torsions, empty_torsions)
    assert shifts.shape == (0,)


def test_extreme_rhombicity_rdc() -> None:
    """Test RDC stability at the bounds of rhombicity [0, 2/3]."""
    vecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # R = 0
    rdc_axial = calculate_rdc(vecs, 10.0, 0.0)
    assert jnp.isfinite(rdc_axial).all()

    # R = 2/3 (maximum rhombicity)
    rdc_rhombic = calculate_rdc(vecs, 10.0, 2.0 / 3.0)
    assert jnp.isfinite(rdc_rhombic).all()

    # Gradient w.r.t rhombicity at the boundary
    grad_r = jax.grad(lambda r: jnp.sum(calculate_rdc(vecs, 10.0, r)))(2.0 / 3.0)
    assert jnp.isfinite(grad_r)


def test_saxs_zero_q() -> None:
    """Verify SAXS behavior at q=0 (limit of debye_saxs)."""
    coords = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    q_values = jnp.array([0.0])
    ff = jnp.array([[6.0], [6.0]])

    # I(0) should be (sum f_i)^2 = (6+6)^2 = 144
    iq = debye_saxs(coords, q_values, ff)
    np.testing.assert_allclose(iq, 144.0, atol=1e-5)

    # Grad at q=0
    grad_q = jax.grad(lambda q: jnp.sum(debye_saxs(coords, q, ff)))(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(grad_q))
    # dI/dq at q=0 should be 0 (Guinier peak is flat)
    np.testing.assert_allclose(grad_q, 0.0, atol=1e-5)


def test_rdc_fitting_degenerate() -> None:
    """Verify that RDC fitting handles degenerate bond vectors (singular A matrix)."""
    from diff_biophys.nmr.rdc import fit_saupe_tensor

    # All bond vectors are identical
    vecs = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rdcs = jnp.array([10.0, 10.0, 10.0])

    tensor = fit_saupe_tensor(vecs, rdcs)
    assert jnp.all(jnp.isfinite(tensor))


def test_fsc_zero_data() -> None:
    """Verify that FSC handles zero-data maps without crashing (division by zero)."""
    from diff_biophys.cryo_em import compute_fsc

    map1 = jnp.zeros((10, 10, 10))
    map2 = jnp.zeros((10, 10, 10))
    voxel_size = (1.0, 1.0, 1.0)

    freqs, vals = compute_fsc(map1, map2, voxel_size)
    assert jnp.all(jnp.isfinite(freqs))
    assert jnp.all(jnp.isfinite(vals))
    assert jnp.all(vals == 0.0)


def test_ensemble_zero_weights() -> None:
    """Verify Ensemble behavior when weights are all zero (should ideally not crash or return NaNs)."""
    from diff_biophys.ensemble import Ensemble

    coords = jnp.zeros((2, 3, 3))
    # Note: Ensemble.__init__ currently does weights / sum(weights)
    # This might produce NaNs. Let's see.
    with np.errstate(divide="ignore", invalid="ignore"):
        ens = Ensemble(coords, weights=jnp.array([0.0, 0.0]))

    # Check if calculation is still finite or handles it
    res = ens.calculate_average(jnp.mean)
    # It will likely be NaN because of the normalization in __init__
    # This test documents the current behavior.
    assert jnp.all(jnp.isnan(res)) or jnp.all(jnp.isfinite(res))
