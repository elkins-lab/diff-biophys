# mypy: disable-error-code="no-untyped-def"
"""Tests for diff_biophys.geometry.backbone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_biophys.geometry.backbone import (
    C_N_LENGTH,
    CA_C_LENGTH,
    N_CA_C_ANGLE,
    N_CA_LENGTH,
    compute_phi_psi,
    make_backbone_builder,
)
from diff_biophys.geometry.torsions import compute_bond_lengths

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_seed() -> jnp.ndarray:
    """Minimal 3-atom seed in ideal (non-collinear) N–CA–C geometry."""
    n0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    ca0 = jnp.array([N_CA_LENGTH, 0.0, 0.0], dtype=jnp.float32)
    c0 = ca0 + CA_C_LENGTH * jnp.array(
        [float(np.cos(np.pi - N_CA_C_ANGLE)), float(np.sin(np.pi - N_CA_C_ANGLE)), 0.0],
        dtype=jnp.float32,
    )
    return jnp.stack([n0, ca0, c0])


# ---------------------------------------------------------------------------
# make_backbone_builder
# ---------------------------------------------------------------------------


def test_make_backbone_builder_output_shape():
    """Builder returns (3*N, 3) array for N residues."""
    for n_res in [3, 6, 10]:
        build = make_backbone_builder(n_res, _make_seed())
        phi = jnp.zeros(n_res, dtype=jnp.float32)
        psi = jnp.zeros(n_res, dtype=jnp.float32)
        coords = build(phi, psi)
        assert coords.shape == (3 * n_res, 3), f"Expected ({3 * n_res}, 3), got {coords.shape}"


def test_make_backbone_builder_bond_lengths_near_ideal(backbone_coords_6res):
    """All inter-atom bonds are within 1 % of their ideal Engh & Huber value."""
    coords = backbone_coords_6res
    lengths = np.array(compute_bond_lengths(coords))

    # Pattern: N-CA, CA-C, C-N, N-CA, CA-C, C-N, ...
    ideal = [N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH] * 5 + [N_CA_LENGTH, CA_C_LENGTH]
    for i, (measured, expected) in enumerate(zip(lengths, ideal)):
        assert abs(measured - expected) / expected < 0.01, (
            f"Bond {i}: expected {expected:.3f} Å, got {measured:.3f} Å"
        )


def test_make_backbone_builder_seed_atoms_unchanged(backbone_coords_6res):
    """The first three atoms (seed) are exactly preserved."""
    seed = _make_seed()
    np.testing.assert_allclose(np.array(backbone_coords_6res[:3]), np.array(seed), atol=1e-5)


def test_make_backbone_builder_is_differentiable():
    """jax.grad flows through the builder without NaN or error."""
    n_res = 4
    build = make_backbone_builder(n_res, _make_seed())
    phi = jnp.full((n_res,), np.deg2rad(-57.0), dtype=jnp.float32)
    psi = jnp.full((n_res,), np.deg2rad(-47.0), dtype=jnp.float32)

    def scalar_loss(phi, psi):
        coords = build(phi, psi)
        return jnp.sum(coords**2)

    grads = jax.grad(scalar_loss, argnums=(0, 1))(phi, psi)
    assert not jnp.any(jnp.isnan(grads[0])), "NaN in phi gradient"
    assert not jnp.any(jnp.isnan(grads[1])), "NaN in psi gradient"


# ---------------------------------------------------------------------------
# compute_phi_psi
# ---------------------------------------------------------------------------


def test_compute_phi_psi_output_lengths(backbone_coords_6res):
    """phi and psi each have length == n_residues."""
    n_res = backbone_coords_6res.shape[0] // 3
    phi, psi = compute_phi_psi(backbone_coords_6res)
    assert phi.shape == (n_res,)
    assert psi.shape == (n_res,)


def test_compute_phi_psi_terminal_padding(backbone_coords_6res):
    """N-terminal phi and C-terminal psi are padded to 0."""
    phi, psi = compute_phi_psi(backbone_coords_6res)
    assert float(phi[0]) == pytest.approx(0.0, abs=1e-5), "N-terminal phi should be 0 (padding)"
    assert float(psi[-1]) == pytest.approx(0.0, abs=1e-5), "C-terminal psi should be 0 (padding)"


def test_compute_phi_psi_helix_interior_values(backbone_coords_6res):
    """Interior phi/psi angles approximate canonical helix values (±15°)."""
    phi, psi = compute_phi_psi(backbone_coords_6res)
    helix_phi = np.deg2rad(-57.0)
    helix_psi = np.deg2rad(-47.0)
    tol = np.deg2rad(15.0)  # allow 15° tolerance for NERF discretisation
    for i in range(1, 5):  # skip terminal padded residues
        assert abs(float(phi[i]) - helix_phi) < tol, (
            f"phi[{i}]={np.rad2deg(float(phi[i])):.1f}° far from helix"
        )
    for i in range(0, 5):
        assert abs(float(psi[i]) - helix_psi) < tol, (
            f"psi[{i}]={np.rad2deg(float(psi[i])):.1f}° far from helix"
        )


# ---------------------------------------------------------------------------
# phi/psi round-trip
# ---------------------------------------------------------------------------


def test_phi_psi_builder_roundtrip():
    """Extract phi/psi → rebuild → re-extract → same interior angles."""
    n_res = 8
    phi_in = jnp.array([0.0] + [np.deg2rad(-57.0)] * 6 + [0.0], dtype=jnp.float32)
    psi_in = jnp.array([np.deg2rad(-47.0)] * 7 + [0.0], dtype=jnp.float32)
    build = make_backbone_builder(n_res, _make_seed())
    coords = build(phi_in, psi_in)
    phi_out, psi_out = compute_phi_psi(coords)

    # Interior residues only (skip terminal padding)
    np.testing.assert_allclose(
        np.array(phi_out[1:-1]), np.array(phi_in[1:-1]), atol=np.deg2rad(1.0)
    )
    np.testing.assert_allclose(np.array(psi_out[:-1]), np.array(psi_in[:-1]), atol=np.deg2rad(1.0))
