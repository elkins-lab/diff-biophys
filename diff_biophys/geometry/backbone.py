"""
diff_biophys.geometry.backbone
==============================
Backbone structure I/O and geometry utilities for gradient-based NMR refinement.

Provides helpers to:
  * Load a PDB model into a biotite AtomArray
  * Extract N–CA–C backbone coordinates as a JAX array
  * Extract residue ID / residue name metadata
  * Recover φ/ψ torsion angles from backbone coordinates
  * Build a differentiable (φ, ψ) → coordinates closure using NERF

Ideal bond geometry follows Engh & Huber (1991), Acta Cryst. A47, 392–400.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import jax.numpy as jnp
import numpy as np

from diff_biophys.geometry.nerf import chain_nerf
from diff_biophys.geometry.torsions import compute_dihedrals

# ---------------------------------------------------------------------------
# Ideal backbone geometry (Engh & Huber 1991)
# ---------------------------------------------------------------------------

#: N–Cα bond length (Å)
N_CA_LENGTH: float = 1.459
#: Cα–C bond length (Å)
CA_C_LENGTH: float = 1.525
#: C–N peptide bond length (Å)
C_N_LENGTH: float = 1.329

#: Cα–C–N bond angle (radians)
CA_C_N_ANGLE: float = float(np.radians(116.2))
#: C–N–Cα bond angle (radians)
C_N_CA_ANGLE: float = float(np.radians(121.7))
#: N–Cα–C bond angle (radians)
N_CA_C_ANGLE: float = float(np.radians(111.2))


# ---------------------------------------------------------------------------
# PDB loading
# ---------------------------------------------------------------------------


def load_pdb_model(path: Path | str, model_id: int = 1) -> struc.AtomArray:
    """Load a single model from a multi-model PDB file.

    Args:
        path: Path to the ``.pdb`` file.
        model_id: 1-based model index (default 1, the first NMR model).

    Returns:
        biotite ``AtomArray`` for the requested model.
    """
    f = pdb_io.PDBFile.read(str(path))
    stack = f.get_structure()
    return stack[model_id - 1]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Backbone coordinate extraction
# ---------------------------------------------------------------------------


def get_backbone_coords(struct: struc.AtomArray) -> jnp.ndarray:
    """Extract N–CA–C backbone coordinates from a biotite ``AtomArray``.

    Atoms are sorted into strict N–CA–C order within each residue so the
    output is safe to pass to :func:`compute_dihedrals` and :func:`chain_nerf`.

    Args:
        struct: Full-atom ``AtomArray`` (or a backbone subset).

    Returns:
        jnp.ndarray: ``(3 × N_residues, 3)`` coordinates in Å, dtype float32.
    """
    mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    backbone = struct[mask]
    order = {"N": 0, "CA": 1, "C": 2}
    sort_key = np.array([order[a] for a in backbone.atom_name])
    res_ids = backbone.res_id
    idx = np.lexsort((sort_key, res_ids))
    return cast(jnp.ndarray, jnp.array(backbone.coord[idx], dtype=jnp.float32))


def get_residue_info(struct: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Return residue IDs and three-letter names from a structure.

    Selects one entry per residue via CA atoms.

    Args:
        struct: ``AtomArray`` (any atom set; CA atoms are used).

    Returns:
        Tuple ``(res_ids, res_names)`` where both are 1-D numpy arrays of
        length ``N_residues``.  ``res_ids`` is int32; ``res_names`` is object
        (str).
    """
    ca_mask = struct.atom_name == "CA"
    ca_atoms = struct[ca_mask]
    return ca_atoms.res_id, ca_atoms.res_name


# ---------------------------------------------------------------------------
# Torsion angle helpers
# ---------------------------------------------------------------------------


def compute_phi_psi(
    coords: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract backbone φ and ψ torsion angles from N–CA–C coordinates.

    The coordinate array must be ordered as::

        [N₀, CA₀, C₀,  N₁, CA₁, C₁,  …,  Nₙ, CAₙ, Cₙ]

    For a chain of N residues, the mapping from :func:`compute_dihedrals`
    output is::

        dihedrals[3i]   = ψᵢ   (C–N–CA–C rotation within residue i)
        dihedrals[3i+1] = ωᵢ   (peptide plane, ≈ π)
        dihedrals[3i+2] = φᵢ₊₁  (next residue's phi)

    Terminal residues that lack a partner are padded with 0.

    Args:
        coords: ``(3N, 3)`` backbone atom coordinates.

    Returns:
        Tuple ``(phi, psi)`` each of shape ``(N,)``.
    """
    d = compute_dihedrals(coords)
    n_res = coords.shape[0] // 3
    psi = d[0::3]
    phi = d[2::3]
    # C-terminal ψ and N-terminal φ are undefined — pad with 0
    psi = jnp.concatenate([psi, jnp.zeros(1)])[:n_res]
    phi = jnp.concatenate([jnp.zeros(1), phi])[:n_res]
    return phi, psi


# ---------------------------------------------------------------------------
# Differentiable backbone builder
# ---------------------------------------------------------------------------


def make_backbone_builder(
    n_residues: int,
    seed_coords: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a differentiable ``(phi, psi) → coords`` function.

    Uses :func:`~diff_biophys.geometry.nerf.chain_nerf` with ideal Engh &
    Huber bond lengths and angles, and fixes ω = π (trans peptide bond).

    The ``seed_coords`` anchor the global frame — they are the N, CA, C
    coordinates of residue 0 and are held fixed.

    Args:
        n_residues: Number of residues in the chain.
        seed_coords: ``(3, 3)`` coordinates of atoms N₀, CA₀, C₀.

    Returns:
        Callable ``build(phi, psi) → jnp.ndarray`` of shape ``(3N, 3)``.
    """
    bond_lengths = jnp.array(
        [C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH] * (n_residues - 1),
        dtype=jnp.float32,
    )
    bond_angles = jnp.array(
        [CA_C_N_ANGLE, C_N_CA_ANGLE, N_CA_C_ANGLE] * (n_residues - 1),
        dtype=jnp.float32,
    )

    def build(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
        """Build backbone from torsion angles.

        Args:
            phi: ``(N,)`` φ angles in radians.
            psi: ``(N,)`` ψ angles in radians.

        Returns:
            jnp.ndarray: ``(3N, 3)`` backbone atom coordinates in Å.
        """
        omega = jnp.full(n_residues - 1, jnp.pi)
        # Dihedral sequence per inter-residue step: ψᵢ, ωᵢ, φᵢ₊₁
        d = jnp.stack([psi[:-1], omega, phi[1:]], axis=1).ravel()
        return cast(jnp.ndarray, chain_nerf(seed_coords, bond_lengths, bond_angles, d))

    return build
