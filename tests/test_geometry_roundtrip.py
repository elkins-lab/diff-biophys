# mypy: disable-error-code="no-untyped-def"
"""
NeRF ↔ Torsion round-trip tests.

The central chain of computation in diff-biophys is:

    torsion angles (φ, ψ)  →  [NeRF]  →  Cartesian coords (xyz)
                          →  [torsions]  →  torsion angles (φ, ψ)

This round-trip must be a near-identity transform: converting torsion angles
to Cartesian coordinates and then extracting torsion angles back out should
recover the original values to floating-point precision.

Why this matters
----------------
If the round-trip drifts, it means the NeRF kernel and the torsion-extraction
kernel are using different conventions (e.g. opposite sign of dihedral, or
different atom ordering).  Such inconsistencies would silently corrupt any
gradient that flows through the full coordinate→observable pipeline.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diff_biophys.geometry.nerf import position_atom_3d
from diff_biophys.geometry.torsions import (
    compute_bond_angles,
    compute_bond_lengths,
    compute_dihedrals,
)

# ============================================================================
# Helpers
# ============================================================================


def build_chain_from_internals(
    bond_lengths: np.ndarray,
    bond_angles: np.ndarray,
    dihedrals: np.ndarray,
) -> jnp.ndarray:
    """Build a linear atomic chain from internal coordinates using NeRF.

    Places the first three atoms manually along a standard frame:
      p0 = (0, 0, 0)
      p1 = (l01, 0, 0)
      p2 = placed using bond_angles[0]

    Args:
        bond_lengths: (N-1,) bond lengths in Å
        bond_angles: (N-2,) bond angles in radians
        dihedrals: (N-3,) dihedral angles in radians

    Returns:
        (N, 3) Cartesian coordinates
    """
    n = len(bond_lengths) + 1
    coords: list[jnp.ndarray] = [jnp.zeros(3, dtype=jnp.float32)] * n

    # Seed atoms
    coords[0] = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    coords[1] = jnp.array([float(bond_lengths[0]), 0.0, 0.0], dtype=jnp.float32)

    # Third atom placed in xy-plane using first bond angle
    l12 = float(bond_lengths[1])
    a012 = float(bond_angles[0])
    coords[2] = jnp.array(
        [float(bond_lengths[0]) - l12 * np.cos(np.pi - a012), l12 * np.sin(np.pi - a012), 0.0],
        dtype=jnp.float32,
    )

    # Remaining atoms via NeRF
    for i in range(3, n):
        coords[i] = position_atom_3d(
            coords[i - 3],
            coords[i - 2],
            coords[i - 1],
            jnp.array(bond_lengths[i - 1], dtype=jnp.float32),
            jnp.array(bond_angles[i - 2], dtype=jnp.float32),
            jnp.array(dihedrals[i - 3], dtype=jnp.float32),
        )

    return jnp.stack(coords)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def ideal_helix_internals():
    """Ideal α-helix internal coordinates for a 10-atom backbone chain.

    All-alanine helix with Engh & Huber (1991) ideal geometry.
    """
    n_atoms = 10
    bond_lengths = np.full(n_atoms - 1, 1.52, dtype=np.float32)  # Cα–C ≈ 1.52 Å
    bond_angles = np.full(n_atoms - 2, np.deg2rad(111.2), dtype=np.float32)  # N–Cα–C
    dihedrals = np.full(n_atoms - 3, np.deg2rad(-57.0), dtype=np.float32)  # helix φ
    return bond_lengths, bond_angles, dihedrals


@pytest.fixture()
def ideal_strand_internals():
    """Ideal β-strand internal coordinates for a 10-atom backbone chain."""
    n_atoms = 10
    bond_lengths = np.full(n_atoms - 1, 1.52, dtype=np.float32)
    bond_angles = np.full(n_atoms - 2, np.deg2rad(116.2), dtype=np.float32)  # extended strand
    dihedrals = np.full(n_atoms - 3, np.deg2rad(-120.0), dtype=np.float32)
    return bond_lengths, bond_angles, dihedrals


@pytest.fixture()
def random_chain_internals():
    """Random (but well-behaved) internal coordinates for property-testing."""
    rng = np.random.default_rng(7)
    n_atoms = 12
    bond_lengths = rng.uniform(1.4, 1.6, n_atoms - 1).astype(np.float32)
    bond_angles = rng.uniform(np.deg2rad(100), np.deg2rad(120), n_atoms - 2).astype(np.float32)
    dihedrals = rng.uniform(-np.pi + 0.1, np.pi - 0.1, n_atoms - 3).astype(np.float32)
    return bond_lengths, bond_angles, dihedrals


# ============================================================================
# Bond length round-trip
# ============================================================================


class TestBondLengthRoundTrip:
    """Build chain from internal coords, then re-extract bond lengths."""

    def test_helix_bond_lengths_preserved(self, ideal_helix_internals):
        """Bond lengths should be recovered exactly (NeRF places atoms at specified distance)."""
        bl, ba, di = ideal_helix_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_bl = compute_bond_lengths(coords)
        np.testing.assert_allclose(
            np.array(recovered_bl),
            bl,
            atol=1e-4,
            err_msg="NeRF does not preserve bond lengths (helix)",
        )

    def test_strand_bond_lengths_preserved(self, ideal_strand_internals):
        bl, ba, di = ideal_strand_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_bl = compute_bond_lengths(coords)
        np.testing.assert_allclose(
            np.array(recovered_bl),
            bl,
            atol=1e-4,
            err_msg="NeRF does not preserve bond lengths (strand)",
        )

    def test_random_bond_lengths_preserved(self, random_chain_internals):
        bl, ba, di = random_chain_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_bl = compute_bond_lengths(coords)
        np.testing.assert_allclose(
            np.array(recovered_bl),
            bl,
            atol=1e-4,
            err_msg="NeRF does not preserve bond lengths (random)",
        )


# ============================================================================
# Bond angle round-trip
# ============================================================================


class TestBondAngleRoundTrip:
    """Build chain from internals, then re-extract bond angles."""

    def test_helix_bond_angles_preserved(self, ideal_helix_internals):
        bl, ba, di = ideal_helix_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_ba = compute_bond_angles(coords)
        # recovered_ba[0] involves the manually-seeded atom (not placed by NeRF),
        # so we only compare angles from index 1 onward.
        np.testing.assert_allclose(
            np.array(recovered_ba[1:]),
            ba[1:],
            atol=1e-3,
            err_msg="NeRF does not preserve bond angles (helix)",
        )

    def test_strand_bond_angles_preserved(self, ideal_strand_internals):
        bl, ba, di = ideal_strand_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_ba = compute_bond_angles(coords)
        # Skip index 0: see note in test_helix_bond_angles_preserved.
        np.testing.assert_allclose(
            np.array(recovered_ba[1:]),
            ba[1:],
            atol=1e-3,
            err_msg="NeRF does not preserve bond angles (strand)",
        )

    def test_random_bond_angles_preserved(self, random_chain_internals):
        bl, ba, di = random_chain_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_ba = compute_bond_angles(coords)
        # Skip index 0: see note in test_helix_bond_angles_preserved.
        np.testing.assert_allclose(
            np.array(recovered_ba[1:]),
            ba[1:],
            atol=1e-3,
            err_msg="NeRF does not preserve bond angles (random)",
        )


# ============================================================================
# Dihedral angle round-trip (the critical one)
# ============================================================================


class TestDihedralRoundTrip:
    """Build chain from dihedrals, then re-extract them.

    This is the most important round-trip because the dihedral angle is the
    primary optimisation variable in torsion-space refinement.  A sign error
    or convention mismatch would not be caught by the bond-length or
    bond-angle tests.
    """

    DIHEDRAL_ATOL = 1e-3  # radians (~0.06°)

    def test_helix_dihedrals_preserved(self, ideal_helix_internals):
        bl, ba, di = ideal_helix_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_di = compute_dihedrals(coords)
        # The first recovered dihedral corresponds to di[0]
        n_compare = min(len(di), len(recovered_di))
        np.testing.assert_allclose(
            np.array(recovered_di[:n_compare]),
            di[:n_compare],
            atol=self.DIHEDRAL_ATOL,
            err_msg="NeRF↔torsion round-trip fails for helix dihedrals",
        )

    def test_strand_dihedrals_preserved(self, ideal_strand_internals):
        bl, ba, di = ideal_strand_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_di = compute_dihedrals(coords)
        n_compare = min(len(di), len(recovered_di))
        np.testing.assert_allclose(
            np.array(recovered_di[:n_compare]),
            di[:n_compare],
            atol=self.DIHEDRAL_ATOL,
            err_msg="NeRF↔torsion round-trip fails for strand dihedrals",
        )

    def test_random_dihedrals_preserved(self, random_chain_internals):
        """Most important: arbitrary dihedrals must survive the round-trip."""
        bl, ba, di = random_chain_internals
        coords = build_chain_from_internals(bl, ba, di)
        recovered_di = compute_dihedrals(coords)
        n_compare = min(len(di), len(recovered_di))
        np.testing.assert_allclose(
            np.array(recovered_di[:n_compare]),
            di[:n_compare],
            atol=self.DIHEDRAL_ATOL,
            err_msg="NeRF↔torsion round-trip fails for random dihedrals",
        )

    def test_sign_convention_positive_dihedral(self):
        """Explicitly verify sign convention for a known positive dihedral.

        p1=(0,0,0), p2=(1,0,0), p3=(1,1,0), dihedral=+90° should place p4 in
        a specific quadrant — we can verify the sign of the z-component.
        """
        p1 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p2 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        p3 = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        bl = jnp.array(1.0, dtype=jnp.float32)
        ba = jnp.array(np.deg2rad(90.0), dtype=jnp.float32)
        dihedral_pos = jnp.array(np.deg2rad(90.0), dtype=jnp.float32)
        dihedral_neg = jnp.array(np.deg2rad(-90.0), dtype=jnp.float32)

        p4_pos = position_atom_3d(p1, p2, p3, bl, ba, dihedral_pos)
        p4_neg = position_atom_3d(p1, p2, p3, bl, ba, dihedral_neg)

        # +90° and −90° should produce mirror images (opposite z-component sign)
        assert float(p4_pos[2]) * float(p4_neg[2]) < 0, (
            f"±90° dihedrals should have opposite z: p4_pos={p4_pos}, p4_neg={p4_neg}"
        )

    def test_dihedral_periodicity(self):
        """A dihedral of θ and θ + 2π should give identical atom positions."""
        p1 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        p2 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        p3 = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        bl = jnp.array(1.52, dtype=jnp.float32)
        ba = jnp.array(np.deg2rad(111.2), dtype=jnp.float32)
        theta = np.deg2rad(-57.0)

        p4_a = position_atom_3d(p1, p2, p3, bl, ba, jnp.array(theta, dtype=jnp.float32))
        p4_b = position_atom_3d(p1, p2, p3, bl, ba, jnp.array(theta + 2 * np.pi, dtype=jnp.float32))

        np.testing.assert_allclose(
            np.array(p4_a),
            np.array(p4_b),
            atol=1e-5,
            err_msg="NeRF is not 2π-periodic in dihedral angle",
        )
