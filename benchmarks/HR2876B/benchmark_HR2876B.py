"""
benchmark_HR2876B.py — diff-biophys benchmark against HR2876B experimental NMR data.

Target: HR2876B (PDB 2LTM), BMRB 18489.
        Human NFU1 N-terminal domain (107 residues).
        CASD-NMR 2013 blind assessment target.

Reference for NMR medoid Q-factor:
  Rosato et al. (2015), "Blind testing of routine, fully automated determination
  of protein structures from NMR data."
  Reported NMR medoid Q-factor: 0.32

Observables used:
  Phase 1: Cα chemical shifts (BMRB 18489)
  Phase 2: ¹⁵N-¹H RDCs in two independent alignment media (RDC_list_1, RDC_list_2)

Usage:
    # Phase 1: chemical shifts only
    python benchmark_HR2876B.py

    # Phase 2: specific medium
    python benchmark_HR2876B.py --rdc RDC_list_1

    # Phase 2: all media simultaneously
    python benchmark_HR2876B.py --rdc all
"""

import argparse
import sys
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
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

# ── local helpers ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from parse_nmrstar import load_bmrb_rdcs, load_bmrb_shifts

# ── Constants ───────────────────────────────────────────────────────────────
BENCH_DIR = Path(__file__).parent
PDB_PATH = BENCH_DIR / "2LTM.pdb"
BMRB_PATH = BENCH_DIR / "bmrb18489_HR2876B.str"

PUBLISHED_NMR_Q = 0.32


def run_benchmark(
    start_model: int = 1,
    rdc_media: list[str] | None = None,
    n_steps: int = 500,
    lr: float = 0.01,
    w_ca: float = 1.0,
    w_rdc: float = 1.0,
    w_restraint: float = 0.0,
    tensor_update_interval: int = 500,
) -> tuple[list[float], tuple[jnp.ndarray, jnp.ndarray]]:
    print("diff-biophys Benchmark: HR2876B (PDB 2LTM, BMRB 18489)")
    print("CASD-NMR 2013 target (Rosato et al. 2015)")
    print("=" * 65)

    if not PDB_PATH.exists() or not BMRB_PATH.exists():
        raise FileNotFoundError("Missing data files. Run `python fetch_data.py` first.")

    # ── Load structure ──────────────────────────────────────────────────────
    print(f"\n[1] Loading PDB 2LTM model {start_model}...")
    struct = load_pdb_model(PDB_PATH, model_id=start_model)
    res_ids, res_names = get_residue_info(struct)
    n_residues = len(res_ids)
    print(f"    {n_residues} residues (res_id {res_ids[0]}–{res_ids[-1]})")

    coords = get_backbone_coords(struct)
    seed_coords = coords[:3]
    build_structure = make_backbone_builder(n_residues, seed_coords)

    init_phi, init_psi = compute_phi_psi(coords)

    # ── Load experimental data ───────────────────────────────────────────────
    print("\n[2] Loading experimental data from BMRB 18489...")

    shifts = load_bmrb_shifts(BMRB_PATH)
    all_rdcs = load_bmrb_rdcs(BMRB_PATH)

    ca_exp = shifts.get("CA", {})
    if not ca_exp:
        raise RuntimeError("No Cα shifts found in BMRB file.")

    ca_loss_fn, n_ca = make_ca_shift_loss(
        ca_exp["res_id"], ca_exp["shift"], res_ids, list(res_names)
    )
    print(
        f"  Cα shift loss: {n_ca} residues matched "
        f"(BMRB residues {ca_exp['shift'].min():.1f}–{ca_exp['shift'].max():.1f} ppm)"
    )

    rdc_entries = []
    if rdc_media:
        media_to_use = list(all_rdcs.keys()) if rdc_media == ["all"] else rdc_media
        for medium in media_to_use:
            if medium not in all_rdcs:
                print(f"  WARNING: RDC medium '{medium}' not found in BMRB file.")
                continue
            rdc_data = all_rdcs[medium]
            n_rdc = len(rdc_data["res_id"])
            ratio = n_rdc / 5.0
            status = "✓ adequate" if ratio >= 4.0 else "⚠ underdetermined"
            print(
                f"  RDC medium '{medium}': {n_rdc} residues matched ({ratio:.1f}× overdetermined — {status})"
            )

            loss_fn, q_eval_fn, make_tensor_fn, n_matched = make_rdc_refinement_fns(
                rdc_data["res_id"], rdc_data["rdc"], res_ids
            )
            rdc_entries.append((medium, loss_fn, q_eval_fn, make_tensor_fn, n_matched))

    # ── Baseline scores ──────────────────────────────────────────────────────
    print(f"\n[3] Baseline scores (NMR structure, model {start_model})...")
    init_ca_rmsd = float(ca_loss_fn(init_phi, init_psi))
    print(f"    Cα shift RMSD : {init_ca_rmsd:.3f} ppm  ({n_ca} residues)")

    init_coords = build_structure(init_phi, init_psi)

    # Fit initial alignment tensors
    tensors = [make_tensor_fn(init_coords) for _, _, _, make_tensor_fn, _ in rdc_entries]

    if rdc_entries:
        for (label, _, q_eval_fn, _, n_rdc), _t in zip(rdc_entries, tensors, strict=False):
            init_q = float(q_eval_fn(init_coords))
            print(f"    RDC Q ({label:12s}): {init_q:.4f}  ({n_rdc} residues)")

    # ── Combined loss ────────────────────────────────────────────────────────
    @jax.jit
    def total_loss(
        params: tuple[jnp.ndarray, jnp.ndarray],
        current_tensors: list[jnp.ndarray],
    ) -> jnp.ndarray:
        phi, psi = params
        loss = w_ca * ca_loss_fn(phi, psi)

        # Harmonic restraint to prevent overfitting (penalize deviation from initial structure)
        if w_restraint > 0.0:
            loss = loss + w_restraint * (
                jnp.mean((phi - init_phi) ** 2) + jnp.mean((psi - init_psi) ** 2)
            )

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
        if i > 0 and rdc_entries and tensor_update_interval > 0 and i % tensor_update_interval == 0:
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
            print(f"\n  RDC Q ({label}, {n_rdc} residues)")
            print(f"    before : {q_before:.4f}")
            print(f"    after  : {q_after:.4f}")
            print(f"    Δ Q    : {q_before - q_after:+.4f}")
            print(f"    published NMR medoid Q ≈ {PUBLISHED_NMR_Q:.2f}")
            if q_after < PUBLISHED_NMR_Q * 0.5:
                print(
                    f"    ⚠  Q after ({q_after:.3f}) is well below the published NMR target "
                    f"({PUBLISHED_NMR_Q:.2f}), indicating extreme overfitting. "
                    f"The backbone is being contorted to satisfy RDCs because the system "
                    f"is globally underdetermined (214 DOFs vs {n_rdc} constraints)."
                )

    np.savetxt(BENCH_DIR / "loss_history.txt", history, header="total_loss per step", comments="# ")
    print("\n  Loss history saved to: loss_history.txt")

    return history, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="diff-biophys HR2876B benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark_HR2876B.py                                 # Ca shifts only\n"
            "  python benchmark_HR2876B.py --rdc RDC_list_1                # + one medium\n"
            "  python benchmark_HR2876B.py --rdc all                       # all media\n"
        ),
    )
    parser.add_argument(
        "--rdc",
        nargs="+",
        metavar="MEDIUM",
        default=None,
        help="RDC saveframe name(s) to include. Use 'all' for all media.",
    )
    parser.add_argument("--start-model", type=int, default=1, help="NMR ensemble model (default 1)")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps (default 500)")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate (default 0.01)")
    parser.add_argument("--w-ca", type=float, default=1.0, help="Cα shift weight (default 1.0)")
    parser.add_argument("--w-rdc", type=float, default=1.0, help="RDC weight (default 1.0)")
    parser.add_argument(
        "--w-restraint",
        type=float,
        default=0.0,
        help="Harmonic backbone restraint weight (default 0.0)",
    )
    parser.add_argument(
        "--tensor-update-interval", type=int, default=500, help="Tensor update interval"
    )
    args = parser.parse_args()

    run_benchmark(
        start_model=args.start_model,
        rdc_media=args.rdc,
        n_steps=args.steps,
        lr=args.lr,
        w_ca=args.w_ca,
        w_rdc=args.w_rdc,
        w_restraint=args.w_restraint,
        tensor_update_interval=args.tensor_update_interval,
    )
