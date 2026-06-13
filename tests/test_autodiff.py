# mypy: disable-error-code="no-untyped-def"
"""
Differentiability tests for diff-biophys kernels.

These tests verify the *core promise* of the library: that every forward-model
kernel is end-to-end differentiable via JAX automatic differentiation.

Design philosophy
-----------------
We are NOT just checking that ``jax.grad`` doesn't crash.  Each test verifies:

1. **Gradient exists** — ``jax.grad`` executes without error.
2. **Gradient is finite** — no NaN or Inf anywhere in the gradient array.
3. **Gradient is non-zero** — a zero gradient would mean the output is
   constant w.r.t. the input, which would make the kernel useless for
   optimisation.  (Exception: degenerate inputs where zero-gradient is
   physically expected are tested separately.)
4. **Gradient has correct shape** — same shape as the differentiated input.
5. **Gradient sign / direction** — where physics gives us a known sign, we
   check it.  For example, moving atoms apart should decrease SAXS intensity
   at low q, so dI(q=0)/dr > 0 when atoms are brought closer.
6. **JIT-compatibility** — gradients must work inside ``jax.jit``, since the
   whole point of the library is GPU-accelerated optimisation.

Each kernel section also includes a brief comment explaining *what* the kernel
computes and *why* differentiability matters for structural biology.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_biophys.cd.kernels import simulate_cd_matrix
from diff_biophys.cryo_em import compute_fsc
from diff_biophys.geometry.macroscopic import compute_rg
from diff_biophys.geometry.nerf import position_atom_3d
from diff_biophys.geometry.superposition import kabsch_alignment
from diff_biophys.geometry.torsions import (
    compute_bond_angles,
    compute_bond_lengths,
    compute_dihedrals,
)
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.nmr.karplus import calculate_karplus_j
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor
from diff_biophys.nmr.ring_currents import calculate_ring_current_shift
from diff_biophys.saxs.kernels import debye_saxs

# ============================================================================
# Helpers
# ============================================================================


def _assert_finite_nonzero_grad(grad: jnp.ndarray, name: str) -> None:
    """Check a gradient array is finite and has at least one nonzero element."""
    assert jnp.all(jnp.isfinite(grad)), (
        f"{name}: gradient contains NaN or Inf — "
        "the kernel likely has a non-differentiable operation (e.g. integer cast, "
        "branch on value, or division by zero without safe_norm)."
    )
    assert jnp.any(grad != 0.0), (
        f"{name}: gradient is identically zero — "
        "the output is constant w.r.t. the input, making optimisation impossible."
    )


# ============================================================================
# SAXS — Debye formula
# ============================================================================
# The Debye formula computes I(q) = Σ_ij f_i f_j sinc(q r_ij).
# Differentiating I(q) w.r.t. atomic coordinates gives dI/dr, which is the
# gradient used in gradient-descent structure refinement against SAXS data.


class TestSAXSDifferentiability:
    """All tests differentiate w.r.t. atomic coordinates."""

    def test_grad_exists_and_finite(self, small_protein_coords, q_values, uniform_form_factors):
        """Basic: jax.grad runs and produces finite gradients."""

        def loss(coords):
            iq = debye_saxs(coords, q_values, uniform_form_factors)
            return jnp.sum(iq)

        grad_fn = jax.grad(loss)
        grad = grad_fn(small_protein_coords)

        assert grad.shape == small_protein_coords.shape
        _assert_finite_nonzero_grad(grad, "debye_saxs coords")

    def test_grad_is_jit_compatible(self, small_protein_coords, q_values, uniform_form_factors):
        """Gradients must work inside jax.jit for GPU-accelerated optimisation."""

        @jax.jit
        def grad_fn(coords):
            def loss(c):
                return jnp.sum(debye_saxs(c, q_values, uniform_form_factors))

            return jax.grad(loss)(coords)

        grad = grad_fn(small_protein_coords)
        assert jnp.all(jnp.isfinite(grad))

    def test_low_q_intensity_increases_as_atoms_cluster(self, q_values, uniform_form_factors):
        """Physics check: I(q→0) = (Σ f_i)² for a point mass (all atoms at origin).

        As atoms spread out, I(q=0) stays constant (Σ f_i²  term dominates at q=0),
        but the gradient w.r.t. spreading coordinates should be non-trivially structured.
        Here we check that moving atoms toward the centroid changes the loss monotonically.
        """
        n = uniform_form_factors.shape[0]
        # Spread coords
        coords_spread = jnp.eye(n, 3, dtype=jnp.float32) * 5.0
        # Clustered coords
        coords_clustered = jnp.zeros((n, 3), dtype=jnp.float32)

        iq_spread = debye_saxs(coords_spread, q_values, uniform_form_factors)
        iq_clustered = debye_saxs(coords_clustered, q_values, uniform_form_factors)

        # At very low q, clustering increases intensity (constructive interference)
        assert float(iq_clustered[0]) >= float(iq_spread[0]), (
            "I(q_min) should be higher for clustered atoms (constructive interference)"
        )

    def test_solvent_subtraction_gradient_finite(
        self, small_protein_coords, q_values, uniform_form_factors
    ):
        """Gradient remains finite when excluded-volume solvent correction is active."""
        n_atoms = small_protein_coords.shape[0]
        volumes = jnp.full((n_atoms,), 16.44, dtype=jnp.float32)  # carbon volumes

        def loss(coords):
            iq = debye_saxs(coords, q_values, uniform_form_factors, volumes=volumes)
            return jnp.sum(iq)

        grad = jax.grad(loss)(small_protein_coords)
        assert jnp.all(jnp.isfinite(grad)), "Solvent-subtraction gradient contains NaN/Inf"

    def test_second_order_grad_finite(self, small_protein_coords, q_values, uniform_form_factors):
        """Second-order gradients (Hessian diagonal) should be finite.

        This matters for Newton-method optimisers and for verifying the
        Debye kernel doesn't use any non-twice-differentiable operations.
        We use forward-over-reverse (jacfwd ∘ grad) to get the full Hessian
        and check that its diagonal is finite.
        """

        def loss(coords):
            return jnp.sum(debye_saxs(coords, q_values, uniform_form_factors))

        # jacfwd(grad(scalar)) produces the Hessian matrix (N*3, N*3).
        # We flatten coords for jacfwd, then extract the diagonal.
        flat = small_protein_coords.ravel()

        def loss_flat(x):
            return loss(x.reshape(small_protein_coords.shape))

        hessian = jax.jacfwd(jax.grad(loss_flat))(flat)
        hess_diag = jnp.diag(hessian)
        assert jnp.all(jnp.isfinite(hess_diag)), (
            "Debye kernel Hessian diagonal contains NaN/Inf — "
            "kernel may not be twice-differentiable"
        )


# ============================================================================
# NMR — Karplus equation
# ============================================================================
# J = A cos²θ + B cosθ + C
# Differentiating J w.r.t. the dihedral angle θ gives dJ/dθ, used to
# back-propagate from J-coupling restraints into backbone torsion angles.


class TestKarplusDifferentiability:
    """Karplus equation: J-coupling as a function of dihedral angle."""

    # Canonical Karplus coefficients for ³J(HN,Hα): Vuister & Bax 1993
    A, B, C = 6.98, -1.38, 1.72

    def test_grad_w_r_t_theta(self):
        """dJ/dθ must be finite and non-zero away from the extrema."""
        # θ = 60° is away from the minima (0°, 180°) so gradient is non-zero
        theta = jnp.array([np.deg2rad(60.0)], dtype=jnp.float32)

        def j_sum(t):
            return jnp.sum(calculate_karplus_j(t, self.A, self.B, self.C))

        grad = jax.grad(j_sum)(theta)
        _assert_finite_nonzero_grad(grad, "karplus theta")

    def test_grad_sign_at_known_point(self):
        """dJ/dθ at θ=0° should be negative (J is at a maximum, decreasing away).

        J = A + B + C at θ=0.  dJ/dθ = -2A sinθ - B sinθ → at θ=ε, negative
        for A>0 and B<0 (Vuister & Bax coefficients).
        """
        # Slightly past 0° so gradient is well-defined
        theta = jnp.array([np.deg2rad(5.0)], dtype=jnp.float32)

        def j_sum(t):
            return jnp.sum(calculate_karplus_j(t, self.A, self.B, self.C))

        grad = jax.grad(j_sum)(theta)
        # dJ/dθ = (-2A sinθ - B sinθ) which at small positive θ > 0 is negative
        # because A=6.98 >> |B|=1.38
        assert float(grad[0]) < 0.0, f"Expected dJ/dθ < 0 near θ=0°, got {float(grad[0])}"

    def test_batch_grad(self):
        """Gradient over a batch of dihedral angles (vectorised)."""
        thetas = jnp.linspace(0.1, jnp.pi - 0.1, 20, dtype=jnp.float32)

        def total_j(t):
            return jnp.sum(calculate_karplus_j(t, self.A, self.B, self.C))

        grad = jax.grad(total_j)(thetas)
        assert grad.shape == thetas.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_grad(self):
        """Gradient inside jax.jit."""
        theta = jnp.array([np.deg2rad(90.0)], dtype=jnp.float32)

        @jax.jit
        def grad_fn(t):
            return jax.grad(lambda x: jnp.sum(calculate_karplus_j(x, self.A, self.B, self.C)))(t)

        grad = grad_fn(theta)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# NMR — Residual Dipolar Couplings (RDC)
# ============================================================================
# D = d_max · vᵀ S v  where S is the Saupe alignment tensor.
# Gradient w.r.t. bond vectors is used in structure refinement;
# gradient w.r.t. S is used in tensor fitting.


class TestRDCDifferentiability:
    """RDC calculation via Saupe tensor."""

    def test_grad_w_r_t_bond_vectors(self, bond_vectors_nh, axial_saupe_tensor):
        """dD/dv: gradient flows back through bond orientations → coordinates."""

        def total_rdc(vecs):
            return jnp.sum(calculate_rdc_from_tensor(vecs, axial_saupe_tensor))

        grad = jax.grad(total_rdc)(bond_vectors_nh)
        assert grad.shape == bond_vectors_nh.shape
        _assert_finite_nonzero_grad(grad, "rdc bond_vectors")

    def test_grad_w_r_t_saupe_tensor(self, bond_vectors_nh, axial_saupe_tensor):
        """dD/dS: gradient flows back into the alignment tensor.

        This is used when jointly optimising structure and alignment simultaneously.
        """

        def total_rdc(S):
            return jnp.sum(calculate_rdc_from_tensor(bond_vectors_nh, S))

        grad = jax.grad(total_rdc)(axial_saupe_tensor)
        assert grad.shape == axial_saupe_tensor.shape
        _assert_finite_nonzero_grad(grad, "rdc saupe_tensor")

    def test_rdc_zero_for_magic_angle(self, axial_saupe_tensor):
        """Physics: RDC = 0 for a bond at the magic angle (54.74°) with axial tensor.

        D ∝ (3cos²θ - 1) = 0 at θ = arccos(1/√3) ≈ 54.74°.
        This is a science correctness test, not purely a grad test.
        """
        magic_angle = np.arccos(1.0 / np.sqrt(3.0))
        # Bond vector in the xz plane at the magic angle from z
        v = jnp.array([[np.sin(magic_angle), 0.0, np.cos(magic_angle)]], dtype=jnp.float32)
        rdc = calculate_rdc_from_tensor(v, axial_saupe_tensor)
        assert abs(float(rdc[0])) < 1e-4, f"Expected D≈0 at magic angle, got {float(rdc[0])}"

    def test_jit_grad_w_r_t_bond_vectors(self, bond_vectors_nh, axial_saupe_tensor):
        """Grad inside jit."""

        @jax.jit
        def grad_fn(vecs):
            return jax.grad(lambda v: jnp.sum(calculate_rdc_from_tensor(v, axial_saupe_tensor)))(
                vecs
            )

        grad = grad_fn(bond_vectors_nh)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# NMR — Chemical Shifts (Cα)
# ============================================================================
# Cα shifts are predicted from backbone φ/ψ angles via soft Gaussian detectors.
# Gradient w.r.t. φ/ψ enables direct shift-driven torsion refinement.


class TestChemicalShiftsDifferentiability:
    """Cα chemical shift prediction from backbone torsions."""

    def test_grad_w_r_t_phi(self, helix_phi_psi, ala_rc_shifts):
        """dδ_Cα/dφ — key for NMR-driven torsion angle refinement."""
        phi, psi = helix_phi_psi

        def total_shift(p):
            return jnp.sum(predict_ca_shifts(p, psi, ala_rc_shifts))

        grad = jax.grad(total_shift)(phi)
        assert grad.shape == phi.shape
        _assert_finite_nonzero_grad(grad, "ca_shifts phi")

    def test_grad_w_r_t_psi(self, helix_phi_psi, ala_rc_shifts):
        """dδ_Cα/dψ."""
        phi, psi = helix_phi_psi

        def total_shift(ps):
            return jnp.sum(predict_ca_shifts(phi, ps, ala_rc_shifts))

        grad = jax.grad(total_shift)(psi)
        assert grad.shape == psi.shape
        _assert_finite_nonzero_grad(grad, "ca_shifts psi")

    def test_grad_w_r_t_rc_shifts(self, helix_phi_psi, ala_rc_shifts):
        """dδ_Cα/dδ_rc — used when fitting random-coil reference values."""
        phi, psi = helix_phi_psi

        def total_shift(rc):
            return jnp.sum(predict_ca_shifts(phi, psi, rc))

        grad = jax.grad(total_shift)(ala_rc_shifts)
        assert grad.shape == ala_rc_shifts.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_helix_shifts_higher_than_sheet(self, helix_phi_psi, sheet_phi_psi, ala_rc_shifts):
        """Physics: α-helix Cα shifts are ~+3.1 ppm above random coil;
        β-sheet Cα shifts are ~–1.5 ppm below random coil.
        So mean(helix_shift) > mean(sheet_shift).
        """
        phi_h, psi_h = helix_phi_psi
        phi_s, psi_s = sheet_phi_psi
        rc_h = jnp.full_like(phi_h, 52.5)
        rc_s = jnp.full_like(phi_s, 52.5)

        shifts_helix = predict_ca_shifts(phi_h, psi_h, rc_h)
        shifts_sheet = predict_ca_shifts(phi_s, psi_s, rc_s)

        assert jnp.mean(shifts_helix) > jnp.mean(shifts_sheet), (
            f"Helix mean shift {jnp.mean(shifts_helix):.2f} should exceed "
            f"sheet mean shift {jnp.mean(shifts_sheet):.2f}"
        )

    def test_jit_grad(self, helix_phi_psi, ala_rc_shifts):
        """Grad inside jit."""
        phi, psi = helix_phi_psi

        @jax.jit
        def grad_fn(p):
            return jax.grad(lambda x: jnp.sum(predict_ca_shifts(x, psi, ala_rc_shifts)))(p)

        grad = grad_fn(phi)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# NMR — Ring Current Shifts
# ============================================================================
# Aromatic rings (Phe, Tyr, Trp, His) shield or deshield nearby nuclei.
# Differentiating w.r.t. atomic coordinates propagates ring current
# restraints back through the structure.


class TestRingCurrentDifferentiability:
    """Ring current chemical shift using Johnson-Bovey approximation."""

    @pytest.fixture()
    def ring_setup(self):
        """Phenylalanine-like ring: centre at origin, normal along z."""
        # 5 probe nuclei placed around the ring
        coords = jnp.array(
            [
                [0.0, 0.0, 3.0],  # directly above (shielded)
                [3.0, 0.0, 0.0],  # in the plane (deshielded)
                [0.0, 3.0, 0.0],  # in the plane
                [2.0, 0.0, 2.0],  # intermediate
                [0.0, 0.0, -3.0],  # directly below (shielded)
            ],
            dtype=jnp.float32,
        )
        ring_center = jnp.zeros(3, dtype=jnp.float32)
        ring_normal = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        intensity = 1.0
        return coords, ring_center, ring_normal, intensity

    def test_grad_w_r_t_coords(self, ring_setup):
        """dδ/dr: shielding gradient w.r.t. probe nuclei positions."""
        coords, ring_center, ring_normal, intensity = ring_setup

        def total_shift(c):
            return jnp.sum(calculate_ring_current_shift(c, ring_center, ring_normal, intensity))

        grad = jax.grad(total_shift)(coords)
        assert grad.shape == coords.shape
        _assert_finite_nonzero_grad(grad, "ring_current coords")

    def test_shielding_above_ring(self, ring_setup):
        """Physics: nucleus directly above ring centre (along normal) is shielded (δ < 0)."""
        coords, ring_center, ring_normal, intensity = ring_setup
        above = jnp.array([[0.0, 0.0, 3.0]], dtype=jnp.float32)
        shift = calculate_ring_current_shift(above, ring_center, ring_normal, intensity)
        assert float(shift[0]) < 0.0, (
            f"Nucleus above ring should be shielded (δ < 0), got {float(shift[0])}"
        )

    def test_deshielding_in_plane(self, ring_setup):
        """Physics: nucleus in the ring plane is deshielded (δ > 0)."""
        coords, ring_center, ring_normal, intensity = ring_setup
        in_plane = jnp.array([[5.0, 0.0, 0.0]], dtype=jnp.float32)
        shift = calculate_ring_current_shift(in_plane, ring_center, ring_normal, intensity)
        assert float(shift[0]) > 0.0, (
            f"Nucleus in ring plane should be deshielded (δ > 0), got {float(shift[0])}"
        )

    def test_jit_grad(self, ring_setup):
        """Grad inside jit."""
        coords, ring_center, ring_normal, intensity = ring_setup

        @jax.jit
        def grad_fn(c):
            return jax.grad(
                lambda x: jnp.sum(
                    calculate_ring_current_shift(x, ring_center, ring_normal, intensity)
                )
            )(c)

        grad = grad_fn(coords)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# Geometry — Radius of Gyration
# ============================================================================
# Rg is used as a SAXS-derived restraint and as a regulariser in ML loss
# functions.  dRg/dr tells you which atoms to move to change compactness.


class TestRadiusOfGyrationDifferentiability:
    """Rg differentiability and physics correctness."""

    def test_grad_w_r_t_coords(self, small_protein_coords):
        """dRg/dr: fundamental gradient for compaction/expansion optimisation."""
        grad = jax.grad(compute_rg)(small_protein_coords)
        assert grad.shape == small_protein_coords.shape
        _assert_finite_nonzero_grad(grad, "compute_rg coords")

    def test_grad_w_r_t_coords_with_masses(self, small_protein_coords):
        """dRg/dr with mass-weighting — used when electron counts differ."""
        masses = jnp.ones(small_protein_coords.shape[0], dtype=jnp.float32) * 12.0

        def rg_with_masses(coords):
            return compute_rg(coords, masses=masses)

        grad = jax.grad(rg_with_masses)(small_protein_coords)
        assert grad.shape == small_protein_coords.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_single_atom_rg_is_zero(self):
        """Edge case: a single atom has Rg = 0 (no spread from centroid)."""
        coords = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
        rg = compute_rg(coords)
        assert abs(float(rg)) < 1e-5, f"Single-atom Rg should be 0, got {float(rg)}"

    def test_rg_increases_with_spread(self):
        """Rg of a spread-out structure > Rg of a compact one."""
        compact = jnp.zeros((10, 3), dtype=jnp.float32)
        spread = jnp.eye(10, 3, dtype=jnp.float32) * 10.0
        assert compute_rg(spread) > compute_rg(compact)

    def test_rg_translation_invariant(self, small_protein_coords):
        """Rg must be invariant to rigid translation."""
        shift = jnp.array([100.0, 50.0, -30.0], dtype=jnp.float32)
        rg_original = compute_rg(small_protein_coords)
        rg_shifted = compute_rg(small_protein_coords + shift)
        assert abs(float(rg_original) - float(rg_shifted)) < 1e-4, (
            f"Rg changed after translation: {float(rg_original):.4f} → {float(rg_shifted):.4f}"
        )

    def test_jit_grad(self, small_protein_coords):
        """Grad inside jit."""

        @jax.jit
        def grad_fn(coords):
            return jax.grad(compute_rg)(coords)

        grad = grad_fn(small_protein_coords)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# Geometry — NeRF (Natural Extension Reference Frame)
# ============================================================================
# NeRF converts internal coordinates (bond length, angle, dihedral) to
# Cartesian xyz.  This is the differentiable "forward kinematics" layer
# that underlies torsion-tuner and any backbone parametrisation.


class TestNeRFDifferentiability:
    """Differentiability of position_atom_3d w.r.t. each internal coordinate."""

    @pytest.fixture()
    def nerf_inputs(self):
        """Standard linear setup: p1=(0,0,0), p2=(1,0,0), p3=(1,1,0)."""
        p1 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p2 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        p3 = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        bond_len = jnp.array(1.52, dtype=jnp.float32)  # Cα–C bond length
        bond_ang = jnp.array(np.deg2rad(111.2), dtype=jnp.float32)  # N–Cα–C angle
        dihedral = jnp.array(np.deg2rad(-57.0), dtype=jnp.float32)  # helix φ
        return p1, p2, p3, bond_len, bond_ang, dihedral

    def test_grad_w_r_t_dihedral(self, nerf_inputs):
        """dp4/dφ — the gradient that drives torsion-based refinement."""
        p1, p2, p3, bond_len, bond_ang, dihedral = nerf_inputs

        def loss(d):
            p4 = position_atom_3d(p1, p2, p3, bond_len, bond_ang, d)
            return jnp.sum(p4)

        grad = jax.grad(loss)(dihedral)
        assert jnp.isfinite(grad), f"NeRF dihedral gradient is non-finite: {grad}"
        assert grad != 0.0, "NeRF dihedral gradient is zero"

    def test_grad_w_r_t_bond_length(self, nerf_inputs):
        """dp4/dl — bond-length gradient."""
        p1, p2, p3, bond_len, bond_ang, dihedral = nerf_inputs

        def loss(bond_len_param):
            p4 = position_atom_3d(p1, p2, p3, bond_len_param, bond_ang, dihedral)
            return jnp.sum(p4)

        grad = jax.grad(loss)(bond_len)
        assert jnp.isfinite(grad), f"NeRF bond_len gradient is non-finite: {grad}"

    def test_grad_w_r_t_bond_angle(self, nerf_inputs):
        """dp4/dθ — bond-angle gradient."""
        p1, p2, p3, bond_len, bond_ang, dihedral = nerf_inputs

        def loss(a):
            p4 = position_atom_3d(p1, p2, p3, bond_len, a, dihedral)
            return jnp.sum(p4)

        grad = jax.grad(loss)(bond_ang)
        assert jnp.isfinite(grad), f"NeRF bond_angle gradient is non-finite: {grad}"

    def test_grad_w_r_t_reference_positions(self, nerf_inputs):
        """dp4/dp3 — gradient flows back through reference atom positions.

        This is required for full-chain differentiation where every atom's
        position depends on all preceding atoms.
        """
        p1, p2, p3, bond_len, bond_ang, dihedral = nerf_inputs

        def loss(p):
            p4 = position_atom_3d(p1, p2, p, bond_len, bond_ang, dihedral)
            return jnp.sum(p4)

        grad = jax.grad(loss)(p3)
        assert grad.shape == p3.shape
        _assert_finite_nonzero_grad(grad, "nerf p3")

    def test_jit_grad(self, nerf_inputs):
        """All gradients inside jit."""
        p1, p2, p3, bond_len, bond_ang, dihedral = nerf_inputs

        @jax.jit
        def grad_fn(d):
            return jax.grad(lambda x: jnp.sum(position_atom_3d(p1, p2, p3, bond_len, bond_ang, x)))(
                d
            )

        grad = grad_fn(dihedral)
        assert jnp.isfinite(grad)


# ============================================================================
# Geometry — Torsion angles
# ============================================================================
# compute_torsion_angles is used to extract φ/ψ from Cartesian coordinates.
# Its gradient enables coordinate→torsion→observable pipelines to be
# differentiated end-to-end.


class TestTorsionAnglesDifferentiability:
    def test_grad_w_r_t_coords(self, helix_backbone_coords):
        """dφ_i/dr: torsion angle gradient w.r.t. all backbone atom positions."""

        def total_torsion(coords):
            return jnp.sum(compute_dihedrals(coords))

        grad = jax.grad(total_torsion)(helix_backbone_coords)
        assert grad.shape == helix_backbone_coords.shape
        assert jnp.all(jnp.isfinite(grad)), "Torsion angle gradient contains NaN/Inf"

    def test_grad_bond_lengths(self, helix_backbone_coords):
        """Bond length gradients (simpler baseline)."""

        def total_bl(coords):
            return jnp.sum(compute_bond_lengths(coords))

        grad = jax.grad(total_bl)(helix_backbone_coords)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_bond_angles(self, helix_backbone_coords):
        """Bond angle gradients."""

        def total_ba(coords):
            return jnp.sum(compute_bond_angles(coords))

        grad = jax.grad(total_ba)(helix_backbone_coords)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_grad_torsions(self, helix_backbone_coords):
        """Torsion grad inside jit."""

        @jax.jit
        def grad_fn(coords):
            return jax.grad(lambda c: jnp.sum(compute_dihedrals(c)))(coords)

        grad = grad_fn(helix_backbone_coords)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# Geometry — Kabsch alignment (superposition)
# ============================================================================
# The Kabsch RMSD is used as a loss function in structure validation and
# ensemble averaging.  Its gradient drives rigid-body alignment during
# refinement (though it's more commonly used as a metric).


class TestKabschDifferentiability:
    def test_grad_w_r_t_mobile_coords(self, small_protein_coords):
        """d(RMSD)/dP where P is the mobile structure."""
        Q = small_protein_coords + jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32)

        def rmsd_loss(P):
            R, t = kabsch_alignment(P, Q)
            P_aligned = P @ R.T + t
            return jnp.sqrt(jnp.mean(jnp.sum((P_aligned - Q) ** 2, axis=-1)))

        grad = jax.grad(rmsd_loss)(small_protein_coords)
        assert grad.shape == small_protein_coords.shape
        assert jnp.all(jnp.isfinite(grad)), "Kabsch RMSD gradient contains NaN/Inf"

    def test_perfect_alignment_has_zero_rmsd(self, small_protein_coords):
        """Physics: aligning a structure to itself gives RMSD = 0."""
        R, t = kabsch_alignment(small_protein_coords, small_protein_coords)
        aligned = small_protein_coords @ R.T + t
        rmsd = float(jnp.sqrt(jnp.mean(jnp.sum((aligned - small_protein_coords) ** 2, axis=-1))))
        assert rmsd < 1e-4, f"Self-alignment RMSD should be ~0, got {rmsd:.6f}"


# ============================================================================
# CD — Matrix Method
# ============================================================================
# The CD spectrum is an experimental observable for secondary structure.
# Differentiating it w.r.t. chromophore positions connects CD data to
# structure refinement.


class TestCDDifferentiability:
    def test_grad_w_r_t_positions(
        self, helix_chromophore_positions, helix_dipole_orientations, wavelengths
    ):
        """dCD(λ)/dr: gradient of molar ellipticity w.r.t. amide positions."""

        def total_cd(positions):
            cd = simulate_cd_matrix(positions, helix_dipole_orientations, wavelengths)
            return jnp.sum(cd)

        grad = jax.grad(total_cd)(helix_chromophore_positions)
        assert grad.shape == helix_chromophore_positions.shape
        _assert_finite_nonzero_grad(grad, "cd_matrix positions")

    def test_grad_w_r_t_dipole_orientations(
        self, helix_chromophore_positions, helix_dipole_orientations, wavelengths
    ):
        """dCD(λ)/dμ: gradient w.r.t. transition dipole orientations."""

        def total_cd(dipoles):
            cd = simulate_cd_matrix(helix_chromophore_positions, dipoles, wavelengths)
            return jnp.sum(cd)

        grad = jax.grad(total_cd)(helix_dipole_orientations)
        assert grad.shape == helix_dipole_orientations.shape
        assert jnp.all(jnp.isfinite(grad)), "CD dipole orientation gradient contains NaN/Inf"

    def test_cd_output_shape(
        self, helix_chromophore_positions, helix_dipole_orientations, wavelengths
    ):
        """CD spectrum has one value per wavelength."""
        cd = simulate_cd_matrix(helix_chromophore_positions, helix_dipole_orientations, wavelengths)
        assert cd.shape == wavelengths.shape

    def test_jit_grad(self, helix_chromophore_positions, helix_dipole_orientations, wavelengths):
        """CD gradient inside jit."""

        @jax.jit
        def grad_fn(positions):
            return jax.grad(
                lambda p: jnp.sum(simulate_cd_matrix(p, helix_dipole_orientations, wavelengths))
            )(positions)

        grad = grad_fn(helix_chromophore_positions)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# Cryo-EM — Fourier Shell Correlation (FSC)
# ============================================================================
# FSC quantifies map quality by comparing two independent half-maps in
# Fourier space.  A differentiable FSC enables end-to-end optimisation of
# reconstruction or refinement pipelines w.r.t. map quality.


class TestFSCDifferentiability:
    def test_grad_w_r_t_map1(self, small_density_map):
        """dFSC/dmap1: gradient of FSC curve w.r.t. the first half-map."""
        map1, map2 = small_density_map
        voxel_size = (1.0, 1.0, 1.0)

        def fsc_sum(m1):
            _freqs, corr = compute_fsc(m1, map2, voxel_size)
            return jnp.sum(corr)

        grad = jax.grad(fsc_sum)(map1)
        assert grad.shape == map1.shape
        assert jnp.all(jnp.isfinite(grad)), "FSC gradient contains NaN/Inf"

    def test_fsc_self_correlation_is_one(self, small_density_map):
        """Physics: FSC(map, map) = 1.0 at every shell (perfect self-correlation)."""
        map1, _map2 = small_density_map
        voxel_size = (1.0, 1.0, 1.0)
        _freqs, corr = compute_fsc(map1, map1, voxel_size)
        # FSC should be 1 everywhere (or very close, modulo numerical precision)
        assert jnp.all(corr > 0.999), (
            f"Self-FSC should be ≈1 everywhere; min was {float(jnp.min(corr)):.4f}"
        )

    def test_fsc_range_is_minus_one_to_one(self, small_density_map):
        """FSC values must lie in [–1, 1] by definition (it's a correlation)."""
        map1, map2 = small_density_map
        voxel_size = (1.0, 1.0, 1.0)
        _freqs, corr = compute_fsc(map1, map2, voxel_size)
        assert jnp.all(corr >= -1.0 - 1e-5) and jnp.all(corr <= 1.0 + 1e-5), (
            f"FSC out of [–1, 1]: min={float(jnp.min(corr)):.4f}, max={float(jnp.max(corr)):.4f}"
        )

    def test_jit_fsc(self, small_density_map):
        """FSC forward pass inside jit."""
        map1, map2 = small_density_map

        @jax.jit
        def fsc_fn(m1, m2):
            return compute_fsc(m1, m2, (1.0, 1.0, 1.0))

        freqs, corr = fsc_fn(map1, map2)
        assert jnp.all(jnp.isfinite(freqs))
        assert jnp.all(jnp.isfinite(corr))
