# mypy: disable-error-code="no-untyped-def"
"""Tests for diff_biophys.nmr.rdc.nh_bond_vectors."""

import jax
import jax.numpy as jnp
import numpy as np

from diff_biophys.nmr.rdc import nh_bond_vectors

# ---------------------------------------------------------------------------
# Basic shape and unit-length properties
# ---------------------------------------------------------------------------


def test_nh_bond_vectors_shape(backbone_coords_6res):
    """Output shape is (N_residues, 3)."""
    n_res = backbone_coords_6res.shape[0] // 3
    nh = nh_bond_vectors(backbone_coords_6res)
    assert nh.shape == (n_res, 3)


def test_nh_bond_vectors_unit_length(backbone_coords_6res):
    """All output vectors have norm == 1 (within float32 tolerance)."""
    nh = nh_bond_vectors(backbone_coords_6res)
    norms = jnp.linalg.norm(nh, axis=-1)
    np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Geometric plausibility
# ---------------------------------------------------------------------------


def test_nh_bond_vectors_angle_to_nca(backbone_coords_6res):
    """Angle between NH and N→CA is approximately 119° (±10°) for residues 1+.

    The peptide-plane bisector places H at ~119° from both N–CA and
    N–C(i-1) bonds.  Residue 0 uses the N→CA fallback so is excluded.
    """
    coords = backbone_coords_6res
    n_atoms = coords[0::3]
    ca_atoms = coords[1::3]

    # Unit vector N→CA for each residue
    n_to_ca = ca_atoms - n_atoms
    n_to_ca = n_to_ca / jnp.linalg.norm(n_to_ca, axis=-1, keepdims=True)

    nh = nh_bond_vectors(coords)

    for i in range(1, coords.shape[0] // 3):
        cos_theta = float(jnp.dot(nh[i], n_to_ca[i]))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))
        assert 100.0 < angle_deg < 140.0, (
            f"Residue {i}: NH–NCA angle = {angle_deg:.1f}°, expected ~119°"
        )


def test_nh_bond_vectors_in_peptide_plane(backbone_coords_6res):
    """For residues 1+, H lies approximately in the C(i-1)–N–CA plane.

    The scalar triple product (NH · [N→C_prev × N→CA]) should be small
    relative to the product of the vector norms, indicating near-coplanarity.
    """
    coords = backbone_coords_6res
    n_atoms = coords[0::3]
    ca_atoms = coords[1::3]
    c_atoms = coords[2::3]
    nh = nh_bond_vectors(coords)

    for i in range(1, coords.shape[0] // 3):
        n = np.array(n_atoms[i])
        ca = np.array(ca_atoms[i])
        c_prev = np.array(c_atoms[i - 1])

        v_nca = ca - n
        v_ncprev = c_prev - n
        v_nh = np.array(nh[i])

        # Normal to the C(i-1)–N–CA plane
        normal = np.cross(v_ncprev, v_nca)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            continue  # degenerate geometry; skip

        # Out-of-plane component of NH
        out_of_plane = abs(np.dot(v_nh, normal / normal_norm))
        assert out_of_plane < 0.25, (
            f"Residue {i}: NH out-of-plane by {out_of_plane:.3f} (>0.25 sin-units)"
        )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_nh_bond_vectors_gradient_no_nan(backbone_coords_6res):
    """jax.grad flows through nh_bond_vectors without NaN."""

    def scalar_fn(coords):
        return jnp.sum(nh_bond_vectors(coords) ** 2)

    grad = jax.grad(scalar_fn)(backbone_coords_6res)
    assert not jnp.any(jnp.isnan(grad)), "NaN in nh_bond_vectors gradient"


def test_nh_bond_vectors_gradient_nonzero(backbone_coords_6res):
    """Gradient of nh_bond_vectors w.r.t. coords is non-trivially nonzero.

    We sum the raw (unnormalised) dot products to get a larger gradient signal
    than the unit-vector squared-sum produces in float32.
    """

    def scalar_fn(coords):
        # Sum all NH vector components — avoids near-cancellation from
        # unit-length normalisation that yields ~1e-8 gradients in float32.
        return jnp.sum(nh_bond_vectors(coords))

    grad = jax.grad(scalar_fn)(backbone_coords_6res)
    assert jnp.any(jnp.abs(grad) > 1e-8), (
        "All gradients are zero — nh_bond_vectors may be detached from coords"
    )
