# mypy: disable-error-code="no-untyped-def"
"""
End-to-end integration tests: gradient descent using diff-biophys kernels.

These tests verify that the *full optimisation pipeline* works correctly —
not just that individual kernels produce finite gradients, but that those
gradients are actually *useful* for minimising a physically-meaningful loss.

A gradient that is finite but points in the wrong direction (e.g. due to a
sign error in the physics formula) would pass all individual autodiff tests
but fail here.

Test structure
--------------
Each test runs a short gradient-descent loop (10–30 steps with optax Adam)
and asserts that the loss decreases monotonically or at least reaches a value
below the initial loss.  We do NOT test for convergence to the global minimum
— that would require many more steps and is the domain of scientific papers.

What we test:
  1. SAXS-driven structure refinement: move atoms to match a target SAXS curve.
  2. NMR chemical-shift restraints: adjust torsion angles to match target δ_Cα.
  3. Rg restraint: drive a structure toward a target radius of gyration.
  4. Composite SAXS + Rg loss: multi-observable joint refinement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from diff_biophys.geometry.macroscopic import compute_rg
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.saxs.kernels import debye_saxs

# ============================================================================
# Helpers
# ============================================================================


def run_gradient_descent(
    loss_fn,
    params: jnp.ndarray,
    n_steps: int = 20,
    learning_rate: float = 1e-3,
) -> tuple[list[float], jnp.ndarray]:
    """Run Adam gradient descent and return the loss trajectory and final params."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    losses = []

    for _ in range(n_steps):
        loss_val, grads = grad_fn(params)
        losses.append(float(loss_val))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    return losses, params


# ============================================================================
# Test 1: SAXS-driven coordinate refinement
# ============================================================================


class TestSAXSRefinement:
    """Minimise χ² between computed and target SAXS profile by moving atoms."""

    @pytest.fixture()
    def saxs_setup(self, small_protein_coords, q_values, uniform_form_factors):
        """Create a target SAXS curve from slightly perturbed coordinates."""
        # Target: small random displacement of each atom
        rng = np.random.default_rng(42)
        noise = jnp.array(rng.normal(0, 0.5, small_protein_coords.shape).astype(np.float32))
        target_coords = small_protein_coords + noise
        target_iq = debye_saxs(target_coords, q_values, uniform_form_factors)
        return small_protein_coords, target_iq, q_values, uniform_form_factors

    def test_loss_decreases(self, saxs_setup):
        """Core test: SAXS χ² should decrease under gradient descent."""
        starting_coords, target_iq, q_values, form_factors = saxs_setup

        def saxs_loss(coords):
            iq = debye_saxs(coords, q_values, form_factors)
            # Normalised χ²: insensitive to overall scale
            iq_norm = iq / (jnp.sum(iq) + 1e-8)
            target_norm = target_iq / (jnp.sum(target_iq) + 1e-8)
            return jnp.sum((iq_norm - target_norm) ** 2)

        losses, _ = run_gradient_descent(saxs_loss, starting_coords, n_steps=30, learning_rate=5e-3)

        assert losses[-1] < losses[0], (
            f"SAXS loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        )

    def test_loss_trajectory_is_finite(self, saxs_setup):
        """All intermediate losses must be finite — no NaN explosions."""
        starting_coords, target_iq, q_values, form_factors = saxs_setup

        def saxs_loss(coords):
            iq = debye_saxs(coords, q_values, form_factors)
            return jnp.mean((iq - target_iq) ** 2)

        losses, _ = run_gradient_descent(saxs_loss, starting_coords, n_steps=20, learning_rate=1e-3)
        assert all(np.isfinite(lv) for lv in losses), (
            f"SAXS refinement produced NaN/Inf loss at step: "
            f"{[i for i, lv in enumerate(losses) if not np.isfinite(lv)]}"
        )

    def test_final_coords_are_closer_to_target(self, saxs_setup):
        """The refined coordinates should produce a smaller SAXS residual than the start."""
        starting_coords, target_iq, q_values, form_factors = saxs_setup

        def saxs_loss(coords):
            iq = debye_saxs(coords, q_values, form_factors)
            return jnp.mean((iq - target_iq) ** 2)

        _losses, final_coords = run_gradient_descent(
            saxs_loss, starting_coords, n_steps=30, learning_rate=1e-3
        )

        initial_residual = float(saxs_loss(starting_coords))
        final_residual = float(saxs_loss(final_coords))
        assert final_residual < initial_residual, (
            f"Refinement made SAXS residual worse: {initial_residual:.6f} → {final_residual:.6f}"
        )


# ============================================================================
# Test 2: NMR chemical-shift-driven torsion refinement
# ============================================================================


class TestNMRShiftRefinement:
    """Drive φ/ψ torsions toward target Cα chemical shifts."""

    @pytest.fixture()
    def shift_setup(self):
        """Start from a β-strand, drive toward helix chemical shifts."""
        # Starting point: β-strand torsions
        phi_start = jnp.full((6,), np.deg2rad(-120.0), dtype=jnp.float32)
        psi_start = jnp.full((6,), np.deg2rad(120.0), dtype=jnp.float32)
        rc_shifts = jnp.full((6,), 52.5, dtype=jnp.float32)  # alanine random coil

        # Target: helix-like shifts (φ=-57°, ψ=-47°)
        phi_target = jnp.full((6,), np.deg2rad(-57.0), dtype=jnp.float32)
        psi_target = jnp.full((6,), np.deg2rad(-47.0), dtype=jnp.float32)
        target_shifts = predict_ca_shifts(phi_target, psi_target, rc_shifts)

        return phi_start, psi_start, rc_shifts, target_shifts

    def test_phi_loss_decreases(self, shift_setup):
        """Minimising shift MSE w.r.t. φ should decrease the loss."""
        phi_start, psi_start, rc_shifts, target_shifts = shift_setup

        def shift_loss_phi(phi):
            predicted = predict_ca_shifts(phi, psi_start, rc_shifts)
            return jnp.mean((predicted - target_shifts) ** 2)

        losses, _ = run_gradient_descent(shift_loss_phi, phi_start, n_steps=50, learning_rate=1e-2)
        assert losses[-1] < losses[0], (
            f"NMR shift loss (φ) did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_joint_phi_psi_loss_decreases(self, shift_setup):
        """Joint optimisation of φ and ψ together as a stacked parameter."""
        phi_start, psi_start, rc_shifts, target_shifts = shift_setup

        # Stack φ and ψ into a single parameter array shape (2, 6)
        params_start = jnp.stack([phi_start, psi_start], axis=0)

        def shift_loss(params):
            phi, psi = params[0], params[1]
            predicted = predict_ca_shifts(phi, psi, rc_shifts)
            return jnp.mean((predicted - target_shifts) ** 2)

        losses, _ = run_gradient_descent(shift_loss, params_start, n_steps=50, learning_rate=1e-2)

        assert losses[-1] < losses[0], (
            f"Joint NMR shift loss (φ, ψ) did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ============================================================================
# Test 3: Rg restraint
# ============================================================================


class TestRgRestraint:
    """Drive structure compaction or expansion toward a target Rg."""

    def test_compaction_decreases_rg(self, small_protein_coords):
        """Loss = (Rg - target_Rg)² with target_Rg < initial_Rg should compact the structure."""
        initial_rg = float(compute_rg(small_protein_coords))
        target_rg = initial_rg * 0.7  # 30% more compact

        def rg_loss(coords):
            rg = compute_rg(coords)
            return (rg - target_rg) ** 2

        losses, final_coords = run_gradient_descent(
            rg_loss, small_protein_coords, n_steps=30, learning_rate=1e-2
        )

        final_rg = float(compute_rg(final_coords))
        assert final_rg < initial_rg, (
            f"Compaction failed: Rg went from {initial_rg:.2f} to {final_rg:.2f} "
            f"(target was {target_rg:.2f})"
        )

    def test_expansion_increases_rg(self, small_protein_coords):
        """Loss = (Rg - target_Rg)² with target_Rg > initial_Rg should expand the structure."""
        initial_rg = float(compute_rg(small_protein_coords))
        target_rg = initial_rg * 1.3  # 30% more expanded

        def rg_loss(coords):
            rg = compute_rg(coords)
            return (rg - target_rg) ** 2

        losses, final_coords = run_gradient_descent(
            rg_loss, small_protein_coords, n_steps=30, learning_rate=1e-2
        )

        final_rg = float(compute_rg(final_coords))
        assert final_rg > initial_rg, (
            f"Expansion failed: Rg went from {initial_rg:.2f} to {final_rg:.2f} "
            f"(target was {target_rg:.2f})"
        )

    def test_rg_loss_trajectory_finite(self, small_protein_coords):
        """No NaN/Inf in the loss trajectory during Rg-driven refinement."""
        initial_rg = float(compute_rg(small_protein_coords))
        target_rg = initial_rg * 0.8

        def rg_loss(coords):
            return (compute_rg(coords) - target_rg) ** 2

        losses, _ = run_gradient_descent(rg_loss, small_protein_coords, n_steps=30)
        assert all(np.isfinite(lv) for lv in losses), "Rg loss trajectory contains NaN/Inf"


# ============================================================================
# Test 4: Composite SAXS + Rg loss (joint multi-observable refinement)
# ============================================================================


class TestCompositeRefinement:
    """The real-world use case: simultaneously fit SAXS and an Rg restraint.

    This tests that gradients from two different physics kernels add together
    coherently — i.e. that there are no cancellation bugs or sign inversions
    when combining multiple loss terms.
    """

    def test_composite_loss_decreases(self, small_protein_coords, q_values, uniform_form_factors):
        """Joint SAXS + Rg loss should decrease under gradient descent."""
        # Target: perturbed structure
        rng = np.random.default_rng(99)
        noise = jnp.array(rng.normal(0, 0.3, small_protein_coords.shape).astype(np.float32))
        target_coords = small_protein_coords + noise
        target_iq = debye_saxs(target_coords, q_values, uniform_form_factors)
        target_rg = float(compute_rg(target_coords))

        saxs_weight = 1.0
        rg_weight = 10.0

        def composite_loss(coords):
            # SAXS term
            iq = debye_saxs(coords, q_values, uniform_form_factors)
            iq_norm = iq / (jnp.sum(iq) + 1e-8)
            target_norm = target_iq / (jnp.sum(target_iq) + 1e-8)
            saxs_term = saxs_weight * jnp.sum((iq_norm - target_norm) ** 2)

            # Rg term
            rg = compute_rg(coords)
            rg_term = rg_weight * (rg - target_rg) ** 2

            return saxs_term + rg_term

        losses, _ = run_gradient_descent(
            composite_loss, small_protein_coords, n_steps=30, learning_rate=2e-3
        )

        assert losses[-1] < losses[0], (
            f"Composite SAXS+Rg loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        )
        assert all(np.isfinite(lv) for lv in losses), "Composite loss contains NaN/Inf"

    def test_rg_contribution_drives_compaction(
        self, small_protein_coords, q_values, uniform_form_factors
    ):
        """With a strong Rg restraint, the final structure should be more compact."""
        initial_rg = float(compute_rg(small_protein_coords))
        target_rg = initial_rg * 0.75

        def rg_dominated_loss(coords):
            rg_term = 100.0 * (compute_rg(coords) - target_rg) ** 2
            iq = debye_saxs(coords, q_values, uniform_form_factors)
            saxs_term = 0.01 * jnp.mean(iq)  # very weak SAXS term
            return rg_term + saxs_term

        _losses, final_coords = run_gradient_descent(
            rg_dominated_loss, small_protein_coords, n_steps=40, learning_rate=5e-3
        )

        final_rg = float(compute_rg(final_coords))
        assert final_rg < initial_rg, (
            f"Strong Rg restraint failed to compact structure: "
            f"{initial_rg:.2f} → {final_rg:.2f} (target {target_rg:.2f})"
        )
