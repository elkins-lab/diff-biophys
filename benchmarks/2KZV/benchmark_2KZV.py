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

RDC refinement design (fixed-tensor approach):
  The Saupe alignment tensor is fit once to the current structure, then
  held FIXED during gradient descent (jax.lax.stop_gradient). This prevents
  the optimizer from driving Q→0 by learning to fool the tensor fit.
  The tensor is re-fit every --tensor-update-interval steps (default 500).
  Q-factors reported during optimization are computed with a fresh tensor
  fit (best-achievable Q), for honest monitoring.

Usage:
    # Phase 1: chemical shifts only
    python benchmark_2KZV.py

    # Phase 2: one medium
    python benchmark_2KZV.py --rdc rdc_PAG.tsv

    # Phase 2: both media simultaneously (recommended)
    python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv

    # Control tensor update frequency
    python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv --tensor-update-interval 200
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

# Published Q-factors (Table 5, Li et al. 2023)
PUBLISHED = {
    "rdc_PAG": {"af2": 0.22, "nmr": 0.18},
    "rdc_PEG": {"af2": 0.35, "nmr": 0.36},
}

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
    order = {"N": 0, "CA": 1, "C": 2}
    sort_key = np.array([order[a] for a in backbone.atom_name])
    res_ids = backbone.res_id
    idx = np.lexsort((sort_key, res_ids))
    return cast(jnp.ndarray, jnp.array(backbone.coord[idx], dtype=jnp.float32))


def load_pdb_model(path: Path, model_id: int = 1) -> struc.AtomArray:  # type: ignore[return]
    f = pdb_io.PDBFile.read(str(path))
    stack = f.get_structure()
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
    """Build a JAX-differentiable Cα shift RMSD loss: (phi, psi) → scalar RMSD."""
    res_id_to_idx = {rid: i for i, rid in enumerate(struct_res_ids)}

    matched_idx: list[int] = []
    matched_exp: list[float] = []
    matched_names: list[str] = []

    for i, rid in enumerate(exp_res_ids):
        if rid in res_id_to_idx:
            struct_idx = res_id_to_idx[rid]
            matched_idx.append(struct_idx)
            matched_exp.append(exp_shifts[i])
            matched_names.append(struct_res_names[struct_idx])

    if not matched_idx:
        raise ValueError("No residues matched between BMRB and PDB.")

    rc = np.array([RANDOM_COIL_CA.get(name, 55.0) for name in matched_names], dtype=np.float32)
    rc_jax = jnp.array(rc)
    exp_jax = jnp.array(matched_exp, dtype=jnp.float32)
    idx_jax = jnp.array(matched_idx, dtype=jnp.int32)

    def loss_fn(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
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


def make_rdc_fns(
    rdc_data: dict[str, np.ndarray],
    struct_res_ids: np.ndarray,
) -> tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],  # fixed-tensor loss
    Callable[[jnp.ndarray], jnp.ndarray],  # Q evaluator (monitoring)
    Callable[[jnp.ndarray], jnp.ndarray],  # tensor fitter
    int,  # n_matched
]:
    """
    Build three functions for RDC-based refinement:

      loss_fn(coords, fixed_tensor) → scalar MSE loss
          Gradient flows through coords only; the tensor is blocked via
          stop_gradient. This prevents the optimizer from driving Q→0 by
          distorting the structure to match any tensor.

      q_eval_fn(coords) → scalar Q-factor
          Re-fits the Saupe tensor and returns the best-achievable Q.
          Used ONLY for monitoring — never called inside a gradient.

      make_tensor_fn(coords) → Saupe tensor (5,)
          Fit the alignment tensor to the current structure.
          Call this outside of jit to update the fixed tensor periodically.
    """
    res_id_to_idx = {rid: i for i, rid in enumerate(struct_res_ids)}

    matched_idx: list[int] = []
    matched_rdc: list[float] = []
    for i, rid in enumerate(rdc_data["res_id"]):
        if rid in res_id_to_idx:
            matched_idx.append(res_id_to_idx[int(rid)])
            matched_rdc.append(float(rdc_data["rdc"][i]))

    if not matched_idx:
        raise ValueError("No RDC residues matched PDB sequence.")

    exp_jax = jnp.array(matched_rdc, dtype=jnp.float32)
    matched_idx_jax = jnp.array(matched_idx, dtype=jnp.int32)
    n_matched = len(matched_idx)

    def _nh_vectors(coords: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct amide N–H unit vectors using peptide-plane geometry.

        H lies in the C(i-1)–N–CA plane, anti-parallel to the bisector of
        the N→CA and N→C(i-1) unit vectors (placing H at ~119° from each).
        """
        n_atoms = coords[0::3]
        ca_atoms = coords[1::3]
        c_atoms = coords[2::3]

        n_to_ca = ca_atoms - n_atoms
        n_to_ca = n_to_ca / jnp.maximum(jnp.linalg.norm(n_to_ca, axis=-1, keepdims=True), 1e-8)

        c_prev = jnp.concatenate([c_atoms[:1], c_atoms[:-1]], axis=0)
        n_to_cprev = c_prev - n_atoms
        n_to_cprev = n_to_cprev / jnp.maximum(
            jnp.linalg.norm(n_to_cprev, axis=-1, keepdims=True), 1e-8
        )

        bisector = n_to_ca + n_to_cprev
        bisector = bisector / jnp.maximum(jnp.linalg.norm(bisector, axis=-1, keepdims=True), 1e-8)
        nh = -bisector
        # Residue 0 fallback (no C(i-1) available; rarely in RDC set)
        nh = jnp.concatenate([-n_to_ca[:1], nh[1:]], axis=0)
        return nh

    def loss_fn(coords: jnp.ndarray, fixed_tensor: jnp.ndarray) -> jnp.ndarray:
        """Fixed-tensor MSE loss. Gradient flows through coords only."""
        tensor = jax.lax.stop_gradient(fixed_tensor)
        nh = _nh_vectors(coords)[matched_idx_jax]
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=21.7)
        return jnp.mean((calc - exp_jax) ** 2)

    def q_eval_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Re-fit tensor and return honest Q-factor (monitoring only)."""
        nh = _nh_vectors(coords)[matched_idx_jax]
        tensor = fit_saupe_tensor(nh, exp_jax, d_max=21.7)
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=21.7)
        return calculate_q_factor(calc, exp_jax)  # type: ignore[no-any-return,return-value]

    def make_tensor_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Fit and return the Saupe tensor for the given coords."""
        nh = _nh_vectors(coords)[matched_idx_jax]
        return fit_saupe_tensor(nh, exp_jax, d_max=21.7)  # type: ignore[no-any-return,return-value]

    # Saupe tensor has 5 free parameters; flag underdetermined media.
    ratio = n_matched / 5.0
    status = "✓ adequate" if ratio >= 4.0 else "⚠ underdetermined"
    print(
        f"  RDC loss: {n_matched} residues matched  ({ratio:.1f}× overdetermined vs tensor — {status})"
    )
    return loss_fn, q_eval_fn, make_tensor_fn, n_matched


# ── Main benchmark ───────────────────────────────────────────────────────────


def run_benchmark(
    start_model: int = 1,
    rdc_paths: list[Path] | None = None,
    n_steps: int = 500,
    lr: float = 0.01,
    w_ca: float = 1.0,
    w_rdc: float = 1.0,
    tensor_update_interval: int = 500,
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
    seed_coords = coords[:3]
    build_structure = make_builder(n_residues, seed_coords)

    init_phi, init_psi = compute_phi_psi(coords)

    # ── Load experimental data ───────────────────────────────────────────────
    print("\n[2] Loading experimental data...")

    bmrb_data = load_bmrb_shifts(BMRB_PATH)
    ca_exp = bmrb_data.get("CA", {})
    if not ca_exp:
        raise RuntimeError("No Cα shifts found in BMRB file.")
    ca_loss_fn, n_ca = make_ca_shift_loss(
        ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names)
    )

    # RDC entries: (label, loss_fn, q_eval_fn, make_tensor_fn, n_rdc)
    rdc_entries: list[tuple[str, Callable, Callable, Callable, int]] = []
    for rdc_path in rdc_paths or []:
        rdc_all = load_rdc_table(rdc_path)
        if not rdc_all:
            print(f"  WARNING: RDC file {rdc_path} is empty or not found.")
            continue
        medium = list(rdc_all.keys())[0]
        print(f"  RDC medium '{medium}' from {rdc_path.name}")
        loss_fn, q_eval_fn, make_tensor_fn, n_rdc = make_rdc_fns(rdc_all[medium], res_ids)
        rdc_entries.append((rdc_path.stem, loss_fn, q_eval_fn, make_tensor_fn, n_rdc))

    # ── Baseline scores ──────────────────────────────────────────────────────
    print(f"\n[3] Baseline scores (NMR structure, model {start_model})...")
    init_ca_rmsd = float(ca_loss_fn(init_phi, init_psi))
    print(f"    Cα shift RMSD : {init_ca_rmsd:.3f} ppm  ({n_ca} residues)")

    init_coords = build_structure(init_phi, init_psi)

    # Fit initial alignment tensors (one per medium)
    tensors: list[jnp.ndarray] = [
        make_tensor_fn(init_coords) for _, _, _, make_tensor_fn, _ in rdc_entries
    ]

    if rdc_entries:
        for (label, _, q_eval_fn, _, n_rdc), _t in zip(rdc_entries, tensors, strict=False):
            init_q = float(q_eval_fn(init_coords))
            print(f"    RDC Q ({label:12s}): {init_q:.4f}  ({n_rdc} residues)")

    # ── Combined loss (fixed-tensor RDC + Cα shifts) ─────────────────────────
    # tensors is a Python list captured by reference; updates outside jit are visible.
    @jax.jit
    def total_loss(
        params: tuple[jnp.ndarray, jnp.ndarray],
        current_tensors: list[jnp.ndarray],
    ) -> jnp.ndarray:
        phi, psi = params
        loss = w_ca * ca_loss_fn(phi, psi)
        if rdc_entries:
            c = build_structure(phi, psi)
            for (_, rdc_loss, _, _, _), t in zip(rdc_entries, current_tensors, strict=False):
                loss = loss + w_rdc * rdc_loss(c, t)
        return loss

    # ── Optimization ─────────────────────────────────────────────────────────
    print(
        f"\n[4] Optimizing ({n_steps} steps, lr={lr}"
        + (f", tensor updated every {tensor_update_interval} steps)" if rdc_entries else ")")
    )
    optimizer = optax.adam(learning_rate=lr)
    params = (init_phi, init_psi)
    opt_state = optimizer.init(params)
    history = []

    @jax.jit
    def step(
        params: tuple[jnp.ndarray, jnp.ndarray],
        opt_state: optax.OptState,
        current_tensors: list[jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(total_loss, argnums=0)(params, current_tensors)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for i in range(n_steps + 1):
        # Periodically re-fit alignment tensors outside the gradient
        if i > 0 and rdc_entries and i % tensor_update_interval == 0:
            phi_u, psi_u = params
            cur_coords = build_structure(phi_u, psi_u)
            tensors = [make_tensor_fn(cur_coords) for _, _, _, make_tensor_fn, _ in rdc_entries]

        params, opt_state, loss = step(params, opt_state, tensors)
        history.append(float(loss))

        if i % 100 == 0:
            phi_r, psi_r = params
            ca_rmsd = float(ca_loss_fn(phi_r, psi_r))
            line = f"  Step {i:4d} | total loss={float(loss):.4f} | Cα RMSD={ca_rmsd:.3f} ppm"
            if rdc_entries:
                mid_coords = build_structure(phi_r, psi_r)
                for label, _, q_eval_fn, _, _ in rdc_entries:
                    q = float(q_eval_fn(mid_coords))
                    line += f" | Q({label})={q:.4f}"
            print(line)

    # ── Final scores ─────────────────────────────────────────────────────────
    final_phi, final_psi = params
    final_ca_rmsd = float(ca_loss_fn(final_phi, final_psi))

    print("\n[5] Results summary")
    print("=" * 65)
    print(f"  Cα shift RMSD  before: {init_ca_rmsd:.3f} ppm")
    print(f"  Cα shift RMSD  after : {final_ca_rmsd:.3f} ppm")
    print(f"  Δ Cα RMSD            : {init_ca_rmsd - final_ca_rmsd:+.3f} ppm")

    if rdc_entries:
        init_coords_final = build_structure(init_phi, init_psi)
        final_coords = build_structure(final_phi, final_psi)
        for label, _, q_eval_fn, _, n_rdc in rdc_entries:
            q_before = float(q_eval_fn(init_coords_final))
            q_after = float(q_eval_fn(final_coords))
            pub = PUBLISHED.get(label, {})
            ratio = n_rdc / 5.0
            primary = label == "rdc_PAG"  # PAG is the primary benchmark metric
            tag = " [PRIMARY]" if primary else " [supplementary]"
            print(f"\n  RDC Q ({label}, {n_rdc} residues, {ratio:.1f}× overdetermined){tag}")
            print(f"    before : {q_before:.4f}")
            print(f"    after  : {q_after:.4f}")
            print(f"    Δ Q    : {q_before - q_after:+.4f}")
            if pub:
                print(f"    published AF2 Q = {pub['af2']:.2f}  |  NMR medoid Q = {pub['nmr']:.2f}")
            # Warn when Q is implausibly below the published target
            if pub and q_after < pub["nmr"] * 0.5:
                print(
                    f"    ⚠  Q after ({q_after:.3f}) is well below the published NMR target "
                    f"({pub['nmr']:.2f}), indicating overfitting. "
                    f"With only {n_rdc} RDCs ({ratio:.1f}× the tensor's 5 free parameters), "
                    "this medium is too data-sparse to constrain the backbone reliably. "
                    "Treat this result as supplementary only."
                )

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
            "  python benchmark_2KZV.py                                      # Ca shifts only\n"
            "  python benchmark_2KZV.py --rdc rdc_PAG.tsv                   # + PAG RDCs\n"
            "  python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv      # both media\n"
            "  python benchmark_2KZV.py --rdc rdc_PAG.tsv --tensor-update-interval 200\n"
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
        help="Weight for RDC MSE loss per medium (default 1.0)",
    )
    parser.add_argument(
        "--tensor-update-interval",
        type=int,
        default=500,
        help="Re-fit alignment tensors every N steps (default 500; 0 = never update)",
    )
    args = parser.parse_args()

    run_benchmark(
        start_model=args.start_model,
        rdc_paths=args.rdc,
        n_steps=args.steps,
        lr=args.lr,
        w_ca=args.w_ca,
        w_rdc=args.w_rdc,
        tensor_update_interval=args.tensor_update_interval,
    )
