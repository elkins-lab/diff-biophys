# mypy: disable-error-code="no-untyped-def"
"""
Shared pytest fixtures for diff-biophys tests.

Provides small, canonical atomic geometries that are biophysically meaningful
so every test starts from the same well-understood coordinates rather than each
file inventing its own ad-hoc arrays.

Geometry conventions
--------------------
* Coordinates are in Ångströms, dtype float32 (JAX default).
* "Backbone atoms" follow the PDB order  N – Cα – C  per residue.
* Bond lengths and angles are taken from Engh & Huber (1991) ideal geometry.
"""

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Small Cartesian geometries
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_chain_coords() -> jnp.ndarray:
    """Five atoms spaced 1.5 Å apart along the x-axis.

    Useful for bond-length / bond-angle tests where the answer is trivially
    known: all bond lengths = 1.5 Å, all bond angles = 180°.
    """
    return jnp.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )


@pytest.fixture()
def helix_backbone_coords() -> jnp.ndarray:
    """Backbone (N–Cα–C) coordinates for a canonical α-helix segment.

    Generated from ideal α-helix parameters:
      * φ = –57°, ψ = –47°
      * Rise per residue ≈ 1.5 Å, radius ≈ 2.3 Å, pitch ≈ 5.4 Å

    Returns (18, 3) array: 6 residues × 3 backbone atoms each.
    """
    # Ideal α-helix: parameterised analytically for 6 residues
    # Using the cylindrical helix formula from Pauling & Corey (1951)
    n_residues = 6
    coords = []
    rise_per_atom = 1.5 / 3.0  # ~0.5 Å per backbone atom
    radius = 2.3  # Å
    atoms_per_turn = 10.8  # residues per turn → ~3.6 atoms between backbone atoms / residue

    for res in range(n_residues):
        for atom_idx in range(3):  # N, CA, C
            i = res * 3 + atom_idx
            angle = 2 * np.pi * i / atoms_per_turn
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = i * rise_per_atom
            coords.append([x, y, z])
    return jnp.array(coords, dtype=jnp.float32)


@pytest.fixture()
def sheet_backbone_coords() -> jnp.ndarray:
    """Backbone (N–Cα–C) coords for a simple extended β-strand segment.

    φ = –120°, ψ = +120° (canonical parallel β-sheet geometry).
    Returns (12, 3): 4 residues × 3 backbone atoms.
    """
    # Extended strand: backbone atoms lie nearly in a plane
    # Cα positions spaced ~3.5 Å apart along z; N and C offset in x
    coords = []
    for res in range(4):
        z_ca = res * 3.5
        coords.append([-0.8, 0.0, z_ca - 1.2])  # N
        coords.append([0.0, 0.0, z_ca])  # Cα
        coords.append([0.8, 0.0, z_ca + 1.2])  # C
    return jnp.array(coords, dtype=jnp.float32)


@pytest.fixture()
def small_protein_coords() -> jnp.ndarray:
    """20 atoms placed on a helix — minimal but realistic for SAXS / Rg tests.

    Returns (20, 3) array in Ångströms.
    """
    n = 20
    t = jnp.linspace(0, 4 * jnp.pi, n)
    x = 5.0 * jnp.cos(t)
    y = 5.0 * jnp.sin(t)
    z = 2.0 * t
    return jnp.stack([x, y, z], axis=-1).astype(jnp.float32)


@pytest.fixture()
def sphere_coords() -> jnp.ndarray:
    """50 atoms uniformly distributed on a sphere of radius 10 Å.

    Useful for analytical Rg checks: Rg of a uniform spherical shell = R.
    """
    rng = np.random.default_rng(42)
    # Marsaglia method for uniform sphere sampling
    theta = rng.uniform(0, 2 * np.pi, 50)
    phi = np.arccos(rng.uniform(-1, 1, 50))
    r = 10.0
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return jnp.array(np.stack([x, y, z], axis=-1), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# SAXS fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def q_values() -> jnp.ndarray:
    """Standard SAXS q-grid: 50 points from 0.01 to 0.5 Å⁻¹."""
    return jnp.linspace(0.01, 0.5, 50, dtype=jnp.float32)


@pytest.fixture()
def uniform_form_factors(small_protein_coords, q_values) -> jnp.ndarray:
    """Constant form factor = 6.0 (carbon) for every atom and every q.

    Returns (N, M) array where N = number of atoms, M = number of q points.
    """
    n_atoms = small_protein_coords.shape[0]
    n_q = q_values.shape[0]
    return jnp.full((n_atoms, n_q), 6.0, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# NMR fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def helix_phi_psi() -> tuple[jnp.ndarray, jnp.ndarray]:
    """φ/ψ angles for a 6-residue ideal α-helix.

    φ = –57°, ψ = –47° (Pauling & Corey ideal values), in radians.
    Returns (phi, psi) each of shape (6,).
    """
    phi = jnp.full((6,), np.deg2rad(-57.0), dtype=jnp.float32)
    psi = jnp.full((6,), np.deg2rad(-47.0), dtype=jnp.float32)
    return phi, psi


@pytest.fixture()
def sheet_phi_psi() -> tuple[jnp.ndarray, jnp.ndarray]:
    """φ/ψ angles for a 4-residue β-strand.

    φ = –120°, ψ = +120°, in radians.
    """
    phi = jnp.full((4,), np.deg2rad(-120.0), dtype=jnp.float32)
    psi = jnp.full((4,), np.deg2rad(120.0), dtype=jnp.float32)
    return phi, psi


@pytest.fixture()
def ala_rc_shifts(helix_phi_psi) -> jnp.ndarray:
    """Random-coil Cα shifts for 6 alanine residues (52.5 ppm each)."""
    phi, _ = helix_phi_psi
    n = phi.shape[0]
    return jnp.full((n,), 52.5, dtype=jnp.float32)


@pytest.fixture()
def bond_vectors_nh() -> jnp.ndarray:
    """10 unit bond vectors for N–H bonds, spread across the unit sphere.

    Uses a deterministic Fibonacci lattice so the test is reproducible and
    the vectors have good angular coverage (important for Saupe tensor fitting).
    """
    n = 10
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vecs = np.stack([x, y, z], axis=-1).astype(np.float32)
    # Normalise (already unit vectors, but be explicit)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    return jnp.array(vecs)


@pytest.fixture()
def axial_saupe_tensor() -> jnp.ndarray:
    """A simple axially-symmetric Saupe tensor aligned along z.

    S_zz = 0.1 (weak alignment), S_xx = S_yy = –0.05, off-diagonal = 0.
    This is the simplest physically-meaningful tensor (axial symmetry).
    """
    return jnp.array(
        [[-0.05, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, 0.10]],
        dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# CD fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def helix_chromophore_positions() -> jnp.ndarray:
    """Amide N positions for a 6-residue α-helix (one per residue).

    Derived from helix_backbone_coords by taking every 3rd atom (N atoms).
    Returns (6, 3).
    """
    n_residues = 6
    coords = []
    rise_per_atom = 1.5 / 3.0
    radius = 2.3
    atoms_per_turn = 10.8
    for res in range(n_residues):
        i = res * 3  # N atom index
        angle = 2 * np.pi * i / atoms_per_turn
        coords.append([radius * np.cos(angle), radius * np.sin(angle), i * rise_per_atom])
    return jnp.array(coords, dtype=jnp.float32)


@pytest.fixture()
def helix_dipole_orientations(helix_chromophore_positions) -> jnp.ndarray:
    """Unit tangent vectors along the helix as transition dipole orientations.

    Returns (6, 3) unit vectors.
    """
    positions = np.array(helix_chromophore_positions)
    # Tangent: forward difference, wrap the last one
    tangents = np.roll(positions, -1, axis=0) - positions
    tangents[-1] = tangents[-2]  # repeat last
    norms = np.linalg.norm(tangents, axis=-1, keepdims=True)
    return jnp.array((tangents / norms).astype(np.float32))


@pytest.fixture()
def wavelengths() -> jnp.ndarray:
    """CD wavelength range: 180–250 nm, 36 points."""
    return jnp.linspace(180.0, 250.0, 36, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Cryo-EM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_density_map() -> tuple[jnp.ndarray, jnp.ndarray]:
    """A 16×16×16 synthetic Gaussian density blob centred in the box.

    This is a minimal but physically-shaped density map for FSC tests.
    """
    rng = np.random.default_rng(0)
    grid = np.indices((16, 16, 16)).astype(np.float32)
    centre = 7.5
    sigma = 3.0
    r2 = sum((g - centre) ** 2 for g in grid)
    density = np.exp(-r2 / (2 * sigma**2))
    # Add small noise so FSC < 1.0 between two independent half-maps
    map1 = density + 0.05 * rng.standard_normal(density.shape).astype(np.float32)
    map2 = density + 0.05 * rng.standard_normal(density.shape).astype(np.float32)
    return jnp.array(map1, dtype=jnp.float32), jnp.array(map2, dtype=jnp.float32)
