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

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ── diff-biophys imports ────────────────────────────────────────────────────
from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)
from diff_biophys.nmr.chemical_shifts import make_ca_shift_loss
from diff_biophys.nmr.io import load_rdc_table
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

# ── local helpers ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from parse_bmrb import load_bmrb_shifts

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
    build_structure = make_backbone_builder(n_residues, seed_coords)

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
    print(
        f"  Cα shift loss: {n_ca} residues matched "
        f"(BMRB residues {ca_exp['shift'].min():.1f}–{ca_exp['shift'].max():.1f} ppm)"
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
        rdc_data = rdc_all[medium]
        n_rdc = len(rdc_data["res_id"])
        ratio = n_rdc / 5.0
        status = "✓ adequate" if ratio >= 4.0 else "⚠ underdetermined"
        print(
            f"  RDC loss: {n_rdc} residues matched  ({ratio:.1f}× overdetermined vs tensor — {status})"
        )
        loss_fn, q_eval_fn, make_tensor_fn, n_matched = make_rdc_refinement_fns(
            rdc_data["res_id"], rdc_data["rdc"], res_ids
        )
        rdc_entries.append((rdc_path.stem, loss_fn, q_eval_fn, make_tensor_fn, n_matched))

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
            primary = label == "rdc_PAG"
            tag = " [PRIMARY]" if primary else " [supplementary]"
            print(f"\n  RDC Q ({label}, {n_rdc} residues, {ratio:.1f}× overdetermined){tag}")
            print(f"    before : {q_before:.4f}")
            print(f"    after  : {q_after:.4f}")
            print(f"    Δ Q    : {q_before - q_after:+.4f}")
            if pub:
                print(f"    published AF2 Q = {pub['af2']:.2f}  |  NMR medoid Q = {pub['nmr']:.2f}")
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
