"""
benchmark_GmR58A.py — diff-biophys benchmark: GmR58A from Geobacter metallireducens.

Target:
    NESG target GmR58A | PDB 2KUT | BMRB 16746
    Benchmark protein for the diff-biophys NMR refinement library.

Observables (all from BMRB 16746, fully public):
    Phase 1 — Cα chemical shifts (114 residues)
    Phase 2 — ¹⁵N-¹H RDCs in 3 independent alignment media:
               RDC_list_1  43 values  (high alignment — stretched gel)
               RDC_list_2  59 values  (low alignment — negative gel / bicelles)
               RDC_list_3  53 values  (medium alignment — PEG)

What this demonstrates:
    - diff_biophys.nmr.chemical_shifts.predict_ca_shifts
    - diff_biophys.nmr.rdc.{fit_saupe_tensor, calculate_rdc_from_tensor, calculate_q_factor}
    - diff_biophys.geometry.nerf.chain_nerf (backbone rebuilding)
    - optax.adam for gradient-based multi-observable refinement

Reference:
    Chemical shift assignment: Bhatt DL et al. BMRB 16746 / PDB 2KUT
    NESG: Northeast Structural Genomics Consortium

Usage:
    # Phase 1 only (Cα shifts):
    python benchmark_GmR58A.py

    # Phase 1 + all 3 RDC media:
    python benchmark_GmR58A.py --rdc all --steps 500

    # Phase 1 + specific medium:
    python benchmark_GmR58A.py --rdc RDC_list_1 --steps 300
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import jax
import jax.numpy as jnp
import numpy as np
import optax
from parse_nmrstar import load_bmrb_rdcs, load_bmrb_shifts

# diff-biophys imports
from diff_biophys.geometry.nerf import chain_nerf
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
from diff_biophys.nmr.rdc import (
    calculate_q_factor,
    calculate_rdc_from_tensor,
    fit_saupe_tensor,
)

BENCH_DIR = Path(__file__).parent

# ── Bond geometry constants (IUPAC 2012) ─────────────────────────────────────
C_N_LENGTH = 1.329  # Å
N_CA_LENGTH = 1.460  # Å
CA_C_LENGTH = 1.525  # Å
CA_C_N_ANGLE = 2.028  # rad (116.2°)
C_N_CA_ANGLE = 2.124  # rad (121.7°)
N_CA_C_ANGLE = 1.941  # rad (111.2°)

# ── Random coil Cα shifts (Wishart 2011) ─────────────────────────────────────
RANDOM_COIL_CA = {
    "ALA": 52.5,
    "ARG": 56.4,
    "ASN": 53.1,
    "ASP": 54.2,
    "CYS": 58.2,
    "GLN": 55.7,
    "GLU": 56.6,
    "GLY": 45.1,
    "HIS": 56.6,
    "ILE": 61.1,
    "LEU": 55.1,
    "LYS": 56.2,
    "MET": 55.4,
    "PHE": 57.7,
    "PRO": 63.3,
    "SER": 58.3,
    "THR": 61.8,
    "TRP": 57.5,
    "TYR": 57.9,
    "VAL": 62.2,
}

# ── Torsion geometry ──────────────────────────────────────────────────────────


def compute_dihedrals(coords: jnp.ndarray) -> jnp.ndarray:
    """Compute all dihedral angles along a backbone coordinate array (N×3)."""
    p0 = coords[:-3]
    p1 = coords[1:-2]
    p2 = coords[2:-1]
    p3 = coords[3:]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - jnp.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
    w = b2 - jnp.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_norm, v) * w, axis=-1)
    return jnp.arctan2(y, x)


# ── Structure loading ─────────────────────────────────────────────────────────


def get_backbone_coords(struct: struc.AtomArray) -> jnp.ndarray:  # type: ignore[return]
    """Extract N–CA–C backbone coordinates as (3N × 3) array in residue order."""
    mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    backbone = struct[mask]
    # Sort by residue then N→CA→C
    order = {"N": 0, "CA": 1, "C": 2}
    idx = np.argsort(
        [backbone.res_id[i] * 3 + order.get(backbone.atom_name[i], 3) for i in range(len(backbone))]
    )
    return cast(jnp.ndarray, jnp.array(backbone.coord[idx], dtype=jnp.float32))


def load_pdb_model(path: Path, model_id: int = 1) -> struc.AtomArray:  # type: ignore[return]
    f = pdb_io.PDBFile.read(str(path))
    stack = f.get_structure()
    return stack[model_id - 1]


def get_residue_info(struct: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (res_ids, res_names) arrays from Cα atoms."""
    ca_mask = struct.atom_name == "CA"
    ca_atoms = struct[ca_mask]
    return ca_atoms.res_id, ca_atoms.res_name


# ── Torsion extraction ────────────────────────────────────────────────────────


def compute_phi_psi(coords: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[return]
    """Extract φ/ψ from N-CA-C backbone coordinates."""
    d = compute_dihedrals(coords)
    psi = d[0::3]
    phi = d[2::3]
    n_res = coords.shape[0] // 3
    psi = jnp.concatenate([psi, jnp.zeros(1)])[:n_res]
    phi = jnp.concatenate([jnp.zeros(1), phi])[:n_res]
    return phi, psi


# ── Structure builder ─────────────────────────────────────────────────────────


def make_builder(
    n_residues: int, seed_coords: jnp.ndarray
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    bond_lengths = jnp.array([C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH] * (n_residues - 1))
    bond_angles = jnp.array([CA_C_N_ANGLE, C_N_CA_ANGLE, N_CA_C_ANGLE] * (n_residues - 1))

    def build(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:  # type: ignore[return]
        omega = jnp.full(n_residues - 1, jnp.pi)
        d = jnp.stack([psi[:-1], omega, phi[1:]], axis=1).ravel()
        return cast(jnp.ndarray, chain_nerf(seed_coords, bond_lengths, bond_angles, d))

    return build


# ── Observable: Cα chemical shifts ───────────────────────────────────────────


def make_ca_shift_loss(
    exp_res_ids: np.ndarray,
    exp_shifts: np.ndarray,
    struct_res_ids: np.ndarray,
    struct_res_names: list[str],
) -> tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], int]:
    """Build a JAX-differentiable Cα shift RMSD loss: (phi, psi) → scalar ppm."""
    res_id_to_idx = {int(rid): i for i, rid in enumerate(struct_res_ids)}
    matched_idx, matched_exp, matched_names = [], [], []
    for i, rid in enumerate(exp_res_ids):
        if int(rid) in res_id_to_idx:
            si = res_id_to_idx[int(rid)]
            matched_idx.append(si)
            matched_exp.append(exp_shifts[i])
            matched_names.append(struct_res_names[si])
    if not matched_idx:
        raise ValueError("No Cα residues matched between BMRB and PDB.")
    rc = np.array([RANDOM_COIL_CA.get(name, 55.0) for name in matched_names], dtype=np.float32)
    rc_jax = jnp.array(rc)
    exp_jax = jnp.array(matched_exp, dtype=jnp.float32)
    idx_jax = jnp.array(matched_idx, dtype=jnp.int32)

    def loss_fn(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
        pred = predict_ca_shifts(phi[idx_jax], psi[idx_jax], rc_jax)
        return jnp.sqrt(jnp.mean((pred - exp_jax) ** 2))

    print(f"  Cα shift loss: {len(matched_idx)} matched residues")
    return loss_fn, len(matched_idx)


# ── Observable: ¹⁵N-¹H RDCs ─────────────────────────────────────────────────


def make_rdc_loss(
    rdc_data: dict[str, np.ndarray],
    struct_res_ids: np.ndarray,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """
    Build a differentiable RDC Q-factor loss for one alignment medium.
    Uses SVD Saupe tensor fitting at each step (fully differentiable via JAX).
    Returns: (loss_fn: coords → Q-factor, exp_jax)
    """
    res_id_to_idx = {int(rid): i for i, rid in enumerate(struct_res_ids)}
    matched_idx, matched_rdc = [], []
    for i, rid in enumerate(rdc_data["res_id"]):
        if int(rid) in res_id_to_idx:
            matched_idx.append(res_id_to_idx[int(rid)])
            matched_rdc.append(float(rdc_data["rdc"][i]))
    if not matched_idx:
        raise ValueError("No RDC residues matched the PDB sequence.")
    exp_jax = jnp.array(matched_rdc, dtype=jnp.float32)

    def get_nh_vectors(coords: jnp.ndarray) -> jnp.ndarray:
        """Approximate N–H bond vector as unit(N[i] − CA[i])."""
        n_atoms = coords[0::3]
        ca_atoms = coords[1::3]
        raw = n_atoms - ca_atoms
        norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / jnp.maximum(norms, 1e-8)

    idx_jax = jnp.array(matched_idx, dtype=jnp.int32)

    def loss_fn(coords: jnp.ndarray) -> jnp.ndarray:
        nh = get_nh_vectors(coords)[idx_jax]
        tensor = fit_saupe_tensor(nh, exp_jax, d_max=21.7)
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=21.7)
        return calculate_q_factor(calc, exp_jax)  # type: ignore[no-any-return,return-value]

    print(
        f"    {len(matched_idx)} matched residues | "
        f"RDC range {min(matched_rdc):.1f}–{max(matched_rdc):.1f} Hz"
    )
    return loss_fn, exp_jax


# ── Main benchmark ────────────────────────────────────────────────────────────


def run_benchmark(
    start_model: int = 1,
    rdc_media: list[str] | None = None,
    n_steps: int = 500,
    lr: float = 0.01,
    w_ca: float = 1.0,
    w_rdc: float = 1.0,
) -> tuple[list[float], tuple[jnp.ndarray, jnp.ndarray]]:
    print("=" * 70)
    print("diff-biophys Benchmark: GmR58A (BMRB 16746, PDB 2KUT)")
    print("NESG target — Geobacter metallireducens, 122 residues")
    print("3 RDC alignment media: Stretch Gel, Negative Gel, PEG")
    print("=" * 70)

    # ── Load structure ──────────────────────────────────────────────────────
    pdb_path = BENCH_DIR / "2KUT.pdb"
    bmrb_path = BENCH_DIR / "bmrb16746_GmR58A.str"
    for p in (pdb_path, bmrb_path):
        if not p.exists():
            print(f"\nMissing: {p.name} — run fetch_data.py first.", file=sys.stderr)
            sys.exit(1)

    print(f"\n[1] Loading PDB 2KUT model {start_model}...")
    struct = load_pdb_model(pdb_path, model_id=start_model)
    struct_res_ids, struct_res_names = get_residue_info(struct)
    struct_res_names_list = list(struct_res_names)
    print(f"    {len(struct_res_ids)} residues, residues {struct_res_ids[0]}–{struct_res_ids[-1]}")

    # ── Load experimental data ──────────────────────────────────────────────
    print("\n[2] Loading BMRB 16746...")
    shifts = load_bmrb_shifts(bmrb_path)
    rdcs = load_bmrb_rdcs(bmrb_path)
    ca_shifts = shifts.get("CA")
    if ca_shifts is None:
        print("ERROR: No Cα shifts found.", file=sys.stderr)
        sys.exit(1)
    print(f"    Cα shifts: {len(ca_shifts['res_id'])} residues")
    for name, d in rdcs.items():
        print(f"    RDC '{name}': {len(d['res_id'])} values")

    # ── Build initial backbone ──────────────────────────────────────────────
    print("\n[3] Building backbone from torsions...")
    coords_init = get_backbone_coords(struct)
    n_res = len(struct_res_ids)
    phi0, psi0 = compute_phi_psi(coords_init)
    build = make_builder(n_res, coords_init[:3])
    print(f"    {n_res} residues, {len(coords_init)} backbone atoms")

    # ── Build loss functions ────────────────────────────────────────────────
    print("\n[4] Building loss functions...")
    print("  Cα shifts:")
    ca_loss_fn, n_ca = make_ca_shift_loss(
        ca_shifts["res_id"], ca_shifts["shift"], struct_res_ids, struct_res_names_list
    )

    rdc_loss_fns: list[tuple[str, Callable[[jnp.ndarray], jnp.ndarray]]] = []
    if rdc_media:
        rdc_names = list(rdcs.keys()) if rdc_media == ["all"] else rdc_media
        print(f"  RDC ({len(rdc_names)} media):")
        for name in rdc_names:
            if name not in rdcs:
                print(f"    Warning: RDC saveframe '{name}' not found; skipping.")
                continue
            print(f"  RDC '{name}':")
            loss_fn, _ = make_rdc_loss(rdcs[name], struct_res_ids)
            rdc_loss_fns.append((name, loss_fn))

    # ── Baseline scores ─────────────────────────────────────────────────────
    print("\n[5] Baseline scores (before refinement)...")
    coords0 = build(phi0, psi0)
    baseline_ca = float(ca_loss_fn(phi0, psi0))
    print(f"    Cα RMSD baseline : {baseline_ca:.3f} ppm")
    baseline_qs: dict[str, float] = {}
    for name, rfn in rdc_loss_fns:
        q = float(rfn(coords0))
        baseline_qs[name] = q
        print(f"    Q-factor baseline ({name}): {q:.3f}")

    # ── Combined loss ───────────────────────────────────────────────────────
    @jax.jit
    def total_loss(params: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        phi, psi = params
        coords = build(phi, psi)
        loss = w_ca * ca_loss_fn(phi, psi)
        for _, rfn in rdc_loss_fns:
            loss = loss + w_rdc * rfn(coords)
        return loss

    # ── Optimiser ───────────────────────────────────────────────────────────
    optimizer = optax.adam(lr)
    params = (phi0, psi0)
    opt_state = optimizer.init(params)
    history: list[float] = []

    @jax.jit
    def step(
        params: tuple[jnp.ndarray, jnp.ndarray], opt_state: optax.OptState
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(total_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ── Main loop ───────────────────────────────────────────────────────────
    print(f"\n[6] Gradient descent: {n_steps} steps, Adam lr={lr}")
    for i in range(n_steps):
        params, opt_state, loss_val = step(params, opt_state)
        history.append(float(loss_val))
        if (i + 1) % 50 == 0:
            print(f"    step {i + 1:4d}/{n_steps}  loss={float(loss_val):.4f}")

    # ── Final scores ────────────────────────────────────────────────────────
    phi_final, psi_final = params
    coords_final = build(phi_final, psi_final)
    final_ca = float(ca_loss_fn(phi_final, psi_final))

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Cα RMSD  before: {baseline_ca:.3f} ppm")
    print(f"  Cα RMSD   after: {final_ca:.3f} ppm  (Δ = {baseline_ca - final_ca:+.3f} ppm)")
    for name, rfn in rdc_loss_fns:
        q_final = float(rfn(coords_final))
        print(f"  Q-factor before ({name}): {baseline_qs[name]:.3f}")
        print(
            f"  Q-factor  after ({name}): {q_final:.3f}  (Δ = {baseline_qs[name] - q_final:+.3f})"
        )
    print("=" * 70)

    # Save loss history
    (BENCH_DIR / "loss_history.txt").write_text("\n".join(f"{v:.6f}" for v in history))
    return history, params


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="diff-biophys benchmark: GmR58A (BMRB 16746, PDB 2KUT)"
    )
    parser.add_argument("--model", type=int, default=1, help="PDB model number (default: 1)")
    parser.add_argument(
        "--steps", type=int, default=300, help="Gradient descent steps (default: 300)"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate (default: 0.01)")
    parser.add_argument(
        "--w-ca", type=float, default=1.0, help="Cα shift loss weight (default: 1.0)"
    )
    parser.add_argument(
        "--w-rdc", type=float, default=1.0, help="RDC loss weight per medium (default: 1.0)"
    )
    parser.add_argument(
        "--rdc",
        nargs="+",
        metavar="MEDIUM",
        default=None,
        help=(
            "RDC saveframe name(s) to include. Use 'all' for all 3 media. "
            "Options: RDC_list_1  RDC_list_2  RDC_list_3  all  (default: none)"
        ),
    )
    args = parser.parse_args()

    run_benchmark(
        start_model=args.model,
        rdc_media=args.rdc,
        n_steps=args.steps,
        lr=args.lr,
        w_ca=args.w_ca,
        w_rdc=args.w_rdc,
    )


if __name__ == "__main__":
    main()
