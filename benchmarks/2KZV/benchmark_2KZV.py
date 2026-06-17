"""
benchmark_2KZV.py — diff-biophys benchmark against 2KZV experimental NMR data.

Ground truth: Li, Spaman, Tejero, Montelione et al. (2023), PMID 37257257.
  "Blind assessment of monomeric AlphaFold2 protein structure models with
   experimental NMR data."

Protein: CV_0373(175-257) from Chromobacterium violaceum
         NESG target CvR118A, PDB 2KZV, BMRB 17020
         83 well-defined residues (BMRB sequence 1–92, comparison range 10–80)

Observables used:
  Phase 1:  Cα chemical shifts (BMRB 17020, 91 residues)
  Phase 2:  ¹⁵N-¹H RDCs in PAG medium (23 residues, data: rdc_PAG.tsv)
            ¹⁵N-¹H RDCs in PEG medium (16 residues, data: rdc_PEG.tsv)
            Published Q-factors (Table 5, Li et al.):
              PAG: AF2=0.22  NMR medoid=0.18
              PEG: AF2=0.35 (0.24*)  NMR medoid=0.36 (0.20*)
              * excluding Thr14 outlier

Benchmark design:
  1. Load PDB 2KZV NMR model (default: model 1).
  2. Apply gradient descent using diff_biophys.nmr.chemical_shifts and/or
     diff_biophys.nmr.rdc as loss functions.
  3. Report Q-factor per medium and Cα RMSD before and after refinement.

Usage:
    # Phase 1: chemical shifts only
    python benchmark_2KZV.py

    # Phase 2: one medium
    python benchmark_2KZV.py --rdc rdc_PAG.tsv

    # Phase 2: both media simultaneously (recommended)
    python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv

    # Tune weights
    python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv --w-ca 0.5 --w-rdc 1.0
"""

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

# ── diff-biophys imports ────────────────────────────────────────────────────
from diff_biophys.geometry.nerf import chain_nerf
from diff_biophys.geometry.torsions import compute_dihedrals
from diff_biophys.nmr.chemical_shifts import RANDOM_COIL_CA, predict_ca_shifts
from diff_biophys.nmr.rdc import (
    calculate_q_factor,
    calculate_rdc_from_tensor,
    fit_saupe_tensor,
)

# ── local helpers ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from parse_bmrb import load_bmrb_shifts, load_rdc_table

# ── Constants ───────────────────────────────────────────────────────────────
BENCH_DIR = Path(__file__).parent
PDB_PATH = BENCH_DIR / "2KZV.pdb"
BMRB_PATH = BENCH_DIR / "bmrb17020.str"

# 2KZV NMR well-defined range (from Li et al. Table 2):
COMPARISON_RANGE = set(
    list(range(10, 21))
    + list(range(27, 39))
    + list(range(53, 60))
    + list(range(63, 69))
    + list(range(73, 81))
)

# Ideal backbone geometry (standard NERF values)
CA_C_LENGTH = 1.525
C_N_LENGTH = 1.329
N_CA_LENGTH = 1.459
CA_C_N_ANGLE = np.radians(116.2)
C_N_CA_ANGLE = np.radians(121.7)
N_CA_C_ANGLE = np.radians(111.2)


# ── Structure loading ────────────────────────────────────────────────────────


def get_backbone_coords(struct: struc.AtomArray) -> jnp.ndarray:  # type: ignore[return]
    """Extract N–CA–C backbone coordinates (3N × 3) in N-CA-C order."""
    mask = np.isin(struct.atom_name, ["N", "CA", "C"])
    backbone = struct[mask]
    # Sort within each residue: N < CA < C
    order = {"N": 0, "CA": 1, "C": 2}
    sort_key = np.array([order[a] for a in backbone.atom_name])
    res_ids = backbone.res_id
    idx = np.lexsort((sort_key, res_ids))
    return cast(jnp.ndarray, jnp.array(backbone.coord[idx], dtype=jnp.float32))


def load_pdb_model(path: Path, model_id: int = 1) -> struc.AtomArray:  # type: ignore[return]
    f = pdb_io.PDBFile.read(str(path))
    stack = f.get_structure()
    # stack is AtomArrayStack; index by model
    return stack[model_id - 1]


def get_residue_info(struct: struc.AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (res_ids, res_names) for unique residues in struct."""
    ca_mask = struct.atom_name == "CA"
    ca_atoms = struct[ca_mask]
    return ca_atoms.res_id, ca_atoms.res_name


# ── Torsion extraction ───────────────────────────────────────────────────────


def compute_phi_psi(coords: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[return]
    """Extract φ/ψ from N-CA-C backbone coordinates."""
    d = compute_dihedrals(coords)
    # Pattern: psi[i]=d[3i], omega[i]=d[3i+1], phi[i+1]=d[3i+2]
    psi = d[0::3]
    phi = d[2::3]
    n_res = coords.shape[0] // 3
    psi = jnp.concatenate([psi, jnp.zeros(1)])[:n_res]
    phi = jnp.concatenate([jnp.zeros(1), phi])[:n_res]
    return phi, psi


# ── Structure builder ────────────────────────────────────────────────────────


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


# ── Observable: Cα chemical shifts ──────────────────────────────────────────


def make_ca_shift_loss(
    exp_res_ids: np.ndarray,
    exp_shifts: np.ndarray,
    struct_res_ids: np.ndarray,
    struct_res_names: list[str],
) -> tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], int]:
    """
    Build a JAX-differentiable Cα shift RMSD loss.

    Returns a function: (phi, psi) → scalar RMSD (ppm).
    Also returns the rc_shifts array and residue mask.
    """
    # Map BMRB res_ids → structure indices
    res_id_to_idx = {rid: i for i, rid in enumerate(struct_res_ids)}

    matched_idx = []  # indices into struct
    matched_exp = []  # experimental Cα shifts
    matched_names = []  # 3-letter codes for RC lookup

    for i, rid in enumerate(exp_res_ids):
        if rid in res_id_to_idx:
            struct_idx = res_id_to_idx[rid]
            matched_idx.append(struct_idx)
            matched_exp.append(exp_shifts[i])
            matched_names.append(struct_res_names[struct_idx])

    if not matched_idx:
        raise ValueError("No residues matched between BMRB and PDB.")

    # Random-coil shifts for matched residues
    rc = np.array([RANDOM_COIL_CA.get(name, 55.0) for name in matched_names], dtype=np.float32)
    rc_jax = jnp.array(rc)
    exp_jax = jnp.array(matched_exp, dtype=jnp.float32)
    idx_jax = jnp.array(matched_idx, dtype=jnp.int32)

    def loss_fn(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
        # Select matched residue torsions
        phi_m = phi[idx_jax]
        psi_m = psi[idx_jax]
        pred = predict_ca_shifts(phi_m, psi_m, rc_jax)
        return jnp.sqrt(jnp.mean((pred - exp_jax) ** 2))

    print(
        f"  Cα shift loss: {len(matched_idx)} residues matched "
        f"(BMRB residues {min(matched_exp):.1f}–{max(matched_exp):.1f} ppm)"
    )
    return loss_fn, len(matched_idx)


# ── Observable: ¹⁵N-¹H RDCs ─────────────────────────────────────────────────


def make_rdc_loss(
    rdc_data: dict[str, np.ndarray], struct_res_ids: np.ndarray
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """
    Build a JAX-differentiable RDC Q-factor loss for the given medium.
    Uses fit_saupe_tensor (SVD) to find the best alignment tensor each step.

    Returns a function: (coords) → scalar Q-factor.
    """
    res_id_to_idx = {rid: i for i, rid in enumerate(struct_res_ids)}

    matched_idx = []
    matched_rdc = []
    for i, rid in enumerate(rdc_data["res_id"]):
        if rid in res_id_to_idx:
            matched_idx.append(res_id_to_idx[int(rid)])
            matched_rdc.append(float(rdc_data["rdc"][i]))

    if not matched_idx:
        raise ValueError("No RDC residues matched PDB sequence.")

    exp_jax = jnp.array(matched_rdc, dtype=jnp.float32)

    def get_nh_vectors(coords: jnp.ndarray) -> jnp.ndarray:
        """Extract N–H bond vectors from N-CA-C coordinates.
        For each residue i (0-indexed): N is at index 3i, CA at 3i+1.
        The amide H is approximately in the plane of N–CA–C(i-1):
        we approximate the NH vector as the unit vector from N[i] toward
        a position displaced from N[i] along (N[i]-CA[i]) direction.
        A simpler proxy: use N[i+1]-C[i] (inter-residue peptide vector).
        Here we use the canonical approximation: unit(N[i] - CA[i]).
        """
        n_atoms = coords[0::3]  # (N_res, 3)
        ca_atoms = coords[1::3]  # (N_res, 3)
        # NH direction approximated as N→(N - CA) unit vector
        # (good proxy for backbone NH orientation)
        raw = n_atoms - ca_atoms
        norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        unit = raw / jnp.maximum(norms, 1e-8)
        return unit  # (N_res, 3)

    matched_idx_jax = jnp.array(matched_idx, dtype=jnp.int32)

    def loss_fn(coords: jnp.ndarray) -> jnp.ndarray:
        all_nh = get_nh_vectors(coords)  # (N_res, 3)
        nh_matched = all_nh[matched_idx_jax]  # (N_matched, 3)
        # Fit Saupe tensor (differentiable via SVD)
        tensor = fit_saupe_tensor(nh_matched, exp_jax, d_max=21.7)  # ¹⁵N-¹H d_max
        calc = calculate_rdc_from_tensor(nh_matched, tensor, d_max=21.7)
        return calculate_q_factor(calc, exp_jax)  # type: ignore[no-any-return,return-value]

    print(f"  RDC loss: {len(matched_idx)} residues matched")
    return loss_fn, exp_jax


# ── Main benchmark ───────────────────────────────────────────────────────────


def run_benchmark(
    start_model: int = 1,
    rdc_paths: list[Path] | None = None,
    n_steps: int = 500,
    lr: float = 0.01,
    w_ca: float = 1.0,
    w_rdc: float = 1.0,
) -> tuple[list[float], tuple[jnp.ndarray, jnp.ndarray]]:
    print("diff-biophys Benchmark: 2KZV (CvR118A, BMRB 17020)")
    print("Li, Spaman, Tejero, Montelione et al. (PMID 37257257)")
    print("=" * 65)

    # ── Load structure ──────────────────────────────────────────────────────
    print(f"\n[1] Loading PDB 2KZV model {start_model}...")
    struct = load_pdb_model(PDB_PATH, model_id=start_model)
    res_ids, res_names = get_residue_info(struct)
    n_residues = len(res_ids)
    print(f"    {n_residues} residues (res_id {res_ids[0]}–{res_ids[-1]})")

    coords = get_backbone_coords(struct)
    seed_coords = coords[:3]  # N0, CA0, C0 — anchors global frame
    build_structure = make_builder(n_residues, seed_coords)

    # Extract initial torsions from the NMR structure
    init_phi, init_psi = compute_phi_psi(coords)

    # ── Load experimental data ───────────────────────────────────────────────
    print("\n[2] Loading experimental data...")

    # Chemical shifts
    bmrb_data = load_bmrb_shifts(BMRB_PATH)
    ca_exp = bmrb_data.get("CA", {})
    if not ca_exp:
        raise RuntimeError("No Cα shifts found in BMRB file.")
    ca_loss_fn, n_ca = make_ca_shift_loss(
        ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names)
    )

    # RDCs (optional, Phase 2) — one loss function per medium, summed during optimisation
    rdc_loss_fns: list[tuple[str, Callable[[jnp.ndarray], jnp.ndarray], int]] = []
    for rdc_path in rdc_paths or []:
        rdc_all = load_rdc_table(rdc_path)
        if not rdc_all:
            print(f"  WARNING: RDC file {rdc_path} is empty or not found.")
            continue
        medium = list(rdc_all.keys())[0]
        print(f"  RDC medium '{medium}' from {rdc_path.name}")
        loss_fn, _ = make_rdc_loss(rdc_all[medium], res_ids)
        n_rdc = len(rdc_all[medium]["res_id"])
        rdc_loss_fns.append((rdc_path.stem, loss_fn, n_rdc))

    # ── Compute baseline scores ──────────────────────────────────────────────
    print(f"\n[3] Baseline scores (NMR structure, model {start_model})...")
    init_ca_rmsd = float(ca_loss_fn(init_phi, init_psi))
    print(f"    Cα shift RMSD : {init_ca_rmsd:.3f} ppm  ({n_ca} residues)")

    if rdc_loss_fns:
        init_coords = build_structure(init_phi, init_psi)
        for label, loss_fn, n_rdc in rdc_loss_fns:
            init_q = float(loss_fn(init_coords))
            print(f"    RDC Q ({label:12s}): {init_q:.4f}  ({n_rdc} residues)")

    # ── Combined loss function ───────────────────────────────────────────────
    @jax.jit
    def total_loss(params: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        phi, psi = params
        loss = w_ca * ca_loss_fn(phi, psi)
        if rdc_loss_fns:
            coords = build_structure(phi, psi)
            for _, loss_fn, _ in rdc_loss_fns:
                loss = loss + w_rdc * loss_fn(coords)
        return loss

    # ── Optimization ─────────────────────────────────────────────────────────
    print(f"\n[4] Optimizing ({n_steps} steps, lr={lr})...")
    optimizer = optax.adam(learning_rate=lr)
    params = (init_phi, init_psi)
    opt_state = optimizer.init(params)
    history = []

    @jax.jit
    def step(
        params: tuple[jnp.ndarray, jnp.ndarray], opt_state: optax.OptState
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(total_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_steps + 1):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))
        if i % 100 == 0:
            phi_r, psi_r = params
            ca_rmsd = float(ca_loss_fn(phi_r, psi_r))
            line = f"  Step {i:4d} | total loss={float(loss):.4f} | Cα RMSD={ca_rmsd:.3f} ppm"
            if rdc_loss_fns:
                mid_coords = build_structure(phi_r, psi_r)
                for label, loss_fn, _ in rdc_loss_fns:
                    q = float(loss_fn(mid_coords))
                    line += f" | Q({label})={q:.4f}"
            print(line)

    # ── Final scores ─────────────────────────────────────────────────────────
    final_phi, final_psi = params
    final_ca_rmsd = float(ca_loss_fn(final_phi, final_psi))

    PUBLISHED = {
        "rdc_PAG": {"af2": 0.22, "nmr": 0.18},
        "rdc_PEG": {"af2": 0.35, "nmr": 0.36},
    }

    print("\n[5] Results summary")
    print("=" * 65)
    print(f"  Cα shift RMSD  before: {init_ca_rmsd:.3f} ppm")
    print(f"  Cα shift RMSD  after : {final_ca_rmsd:.3f} ppm")
    print(f"  Δ Cα RMSD            : {init_ca_rmsd - final_ca_rmsd:+.3f} ppm")

    if rdc_loss_fns:
        init_coords_final = build_structure(init_phi, init_psi)
        final_coords = build_structure(final_phi, final_psi)
        for label, loss_fn, n_rdc in rdc_loss_fns:
            q_before = float(loss_fn(init_coords_final))
            q_after = float(loss_fn(final_coords))
            pub = PUBLISHED.get(label, {})
            print(f"\n  RDC Q ({label}, {n_rdc} residues)")
            print(f"    before : {q_before:.4f}")
            print(f"    after  : {q_after:.4f}")
            print(f"    Δ Q    : {q_before - q_after:+.4f}")
            if pub:
                print(f"    published AF2 Q = {pub['af2']:.2f}  |  NMR medoid Q = {pub['nmr']:.2f}")

    # Save history
    np.savetxt(BENCH_DIR / "loss_history.txt", history, header="total_loss per step", comments="# ")
    print("\n  Loss history saved to: loss_history.txt")

    return history, params


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="diff-biophys 2KZV benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark_2KZV.py                              # Phase 1: Ca shifts only\n"
            "  python benchmark_2KZV.py --rdc rdc_PAG.tsv           # + PAG RDCs\n"
            "  python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv  # both media\n"
        ),
    )
    parser.add_argument(
        "--rdc",
        type=Path,
        nargs="+",
        default=None,
        metavar="TSV",
        help="RDC table TSV file(s). Pass multiple files to use both media simultaneously.",
    )
    parser.add_argument(
        "--start-model",
        type=int,
        default=1,
        help="Which NMR ensemble model to start from (1–20, default 1)",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of optimization steps (default 500)"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate (default 0.01)")
    parser.add_argument(
        "--w-ca", type=float, default=1.0, help="Weight for Cα chemical shift loss (default 1.0)"
    )
    parser.add_argument(
        "--w-rdc",
        type=float,
        default=1.0,
        help="Weight for RDC Q-factor loss per medium (default 1.0)",
    )
    args = parser.parse_args()

    run_benchmark(
        start_model=args.start_model,
        rdc_paths=args.rdc,
        n_steps=args.steps,
        lr=args.lr,
        w_ca=args.w_ca,
        w_rdc=args.w_rdc,
    )
