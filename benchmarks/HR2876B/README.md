# HR2876B diff-biophys Benchmark

**Protein**: N-terminal domain of human NFU1 (Iron-Sulfur Cluster Scaffold Homolog)
**NESG Target**: HR2876B
**PDB**: [2LTM](https://www.rcsb.org/structure/2LTM) | **BMRB**: [18489](https://bmrb.io/data_library/summary/index.php?bmrbId=18489)
**Residues**: 107

---

## Overview

This benchmark validates the `diff-biophys` NMR kernels against **HR2876B**, a target
from the CASD-NMR 2013 blind assessment of automated NMR structure determination
(Rosato et al., *J Biomol NMR* 2015). All experimental data is fully public in a single
NMR-STAR v3 file from BMRB, parseable by the library's standard `parse_nmrstar.py`.

**Reference:**
> Rosato, A., et al. (2015). "The second round of critical assessment of automated
> structure determination of proteins by NMR: CASD-NMR-2013."
> *J Biomol NMR* **62**, 413–424. DOI: 10.1007/s10858-015-9928-5.

---

## Experimental Data

| Observable | Count | Notes |
|---|---|---|
| Cα chemical shifts | 97 residues | From `save_assigned_chem_shift_list_1` |
| RDC list 1 (PEG) | 72 ¹⁵N–¹H values | `save_RDC_list_1` |
| RDC list 2 (Pf1 phage) | 75 ¹⁵N–¹H values | `save_RDC_list_2` |

---

## Quick Start

```bash
# 1. Download the PDB and BMRB data files
python fetch_data.py

# 2. Baseline only (0 optimization steps) — prints ground-truth Q-factors
python -m benchmarks.HR2876B.benchmark_HR2876B --rdc all --steps 0

# 3. Phase 1: Cα shifts only
python -m benchmarks.HR2876B.benchmark_HR2876B

# 4. Phase 2: joint refinement (Cα shifts + both RDC media)
python -m benchmarks.HR2876B.benchmark_HR2876B --rdc all
```

> Run from the repository root so that `diff_biophys` is on the Python path.

---

## Ground-Truth Baseline Q-Factors

These values are computed directly from the deposited 2LTM model 1 Cartesian
coordinates and the BMRB 18489 RDC data — no NERF reconstruction involved.
They are the stored reference values for regression testing.

| Medium | Residues | Q (raw PDB model 1) | Tensor ratio |
|---|---|---|---|
| RDC_list_1 (PEG) | 72 | **0.440** | 14.4× |
| RDC_list_2 (Pf1 phage) | 75 | **0.444** | 15.0× |

The published NMR medoid Q reported in CASD-NMR literature is approximately **0.32**.
The model-1 baseline of ~0.44 is higher because model 1 of the NMR ensemble is not
the medoid conformer — the medoid is the centroid of the ensemble.

---

## Design Notes

### NERF reconstruction drift

`make_backbone_builder` uses **ideal bond lengths and angles** (from the NeRF
parameterization) rather than the actual PDB bond geometry. Over 107 residues this
accumulates to a **~14 Å Cα RMSD** between the raw PDB structure and the NERF-rebuilt
starting point. Consequently:

- All **baseline Q-factors are reported from raw PDB Cartesian coordinates**, not from
  the NERF-rebuilt starting structure.
- The benchmark prints an explicit warning with the NERF drift magnitude at startup.
- Optimization begins from the NERF-parameterized backbone (Q ≈ 0.64), which is
  substantially worse than the raw PDB (Q ≈ 0.44).

### Why optimization overshoots the true NMR Q

Even with 14× overdetermined alignment tensors, gradient descent on backbone torsions
will drive Q well below the physically meaningful NMR baseline. The cause is a DOF
imbalance that applies regardless of how many RDC constraints are available:

| Quantity | Value |
|---|---|
| Backbone torsional DOFs (φ, ψ × 107 residues) | 214 |
| RDC constraints per medium | 72–75 |
| DOFs / constraints ratio | ~3× |

The optimizer has roughly three times more free parameters than RDC constraints per
medium. Without additional structural restraints (Ramachandran priors, steric repulsion,
NOE distances) it finds backbone conformations that satisfy the RDC data better than
the actual NMR structure ever could — but those conformations are physically unrealistic.

This is the same reason that NMR structure calculation programs (X-PLOR/CNS, CYANA)
always combine RDCs with NOE distance restraints and dihedral angle priors.
See *"The Need for Structural Priors"* in the repository root `README.md` for the
general discussion.

### Loss scale normalization

The RDC MSE term (in Hz²) is normalized by `rms(exp_rdcs)²` before entering the total
loss, making it dimensionless and comparable in magnitude to the Cα RMSD term (in ppm).
Without normalization the optimizer weights RDCs ~120× more heavily than chemical shifts.

### Alignment tensor update interval

The Saupe tensor is re-fit every **50 steps** (default), more frequently than in the
2KZV benchmark (500 steps). Because the NERF-rebuilt starting backbone already differs
substantially from the PDB structure, longer intervals create a feedback loop: the
tensor goes stale, the optimizer moves the backbone to fit the stale tensor, and the
next re-fit finds an even worse alignment — artificially driving Q lower than the data
supports.

---

## Expected Output (500 steps, both RDC media)

```
[3] Baseline scores (NMR structure, model 1, raw Cartesian)...
    Cα shift RMSD : 1.710 ppm  (97 residues)
    RDC Q (RDC_list_1  ): 0.4399 (raw PDB)  [reference: 0.440]
      ↳ NERF-rebuilt start: 0.6390  (Δ=+0.1992 from NERF drift; optimization starts here)
    RDC Q (RDC_list_2  ): 0.4435 (raw PDB)  [reference: 0.444]
      ↳ NERF-rebuilt start: 0.6910  (Δ=+0.2475 from NERF drift; optimization starts here)

[5] Results summary
  Cα shift RMSD  before: 1.710 ppm
  Cα shift RMSD  after : ~1.60 ppm
  RDC Q (RDC_list_1): raw PDB 0.440 → NERF start 0.639 → optimized ~0.076  ← overfitting
  RDC Q (RDC_list_2): raw PDB 0.444 → NERF start 0.691 → optimized ~0.086  ← overfitting
```

The benchmark emits a warning when the optimized Q falls more than 30% below the
raw-PDB baseline, flagging that the NERF backbone has overfit the RDC data.

---

## NMR Observable Modules Used

| Observable | diff-biophys function |
|---|---|
| Cα chemical shifts | `diff_biophys.nmr.chemical_shifts.predict_ca_shifts` |
| RDC back-calculation | `diff_biophys.nmr.rdc.calculate_rdc_from_tensor` |
| Alignment tensor fit | `diff_biophys.nmr.rdc.fit_saupe_tensor` |
| Q-factor | `diff_biophys.nmr.rdc.calculate_q_factor` |
| Backbone builder | `diff_biophys.geometry.backbone.make_backbone_builder` |
| BMRB NMR-STAR parser | `parse_nmrstar.load_bmrb_shifts`, `parse_nmrstar.load_bmrb_rdcs` |
| Optimizer | `optax.adam` |
