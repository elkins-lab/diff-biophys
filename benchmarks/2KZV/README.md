# 2KZV diff-biophys Benchmark

**Protein**: CV_0373(175-257) from *Chromobacterium violaceum*
**NESG Target**: CvR118A
**PDB**: [2KZV](https://www.rcsb.org/structure/2KZV) | **BMRB**: [17020](https://bmrb.io/data_library/summary/index.php?bmrbId=17020)

**Reference**: Li, Spaman, Tejero, Montelione et al. (2023). *Blind assessment of monomeric AlphaFold2 protein structure models with experimental NMR data.* PMID [37257257](https://pubmed.ncbi.nlm.nih.gov/37257257/).

---

## Files

| File | Description | Status |
|---|---|---|
| `2KZV.pdb` | NMR ensemble (20 models) from RCSB | ✅ Downloaded |
| `bmrb17020.str` | NMR-STAR v3 file (chem. shifts + NOESY) from BMRB | ✅ Downloaded |
| `parse_bmrb.py` | Parser: extracts Cα/N/HN shifts and RDC tables | ✅ Ready |
| `benchmark_2KZV.py` | Main benchmark script | ✅ Ready |
| `rdc_PAG.tsv` | ¹⁵N-¹H RDCs in PAG medium | ⏳ Awaiting from Tejero/Montelione |
| `rdc_PEG.tsv` | ¹⁵N-¹H RDCs in PEG medium | ⏳ Awaiting from Tejero/Montelione |
| `loss_history.txt` | Per-step loss values from last run | ✅ Auto-generated |

---

## Benchmark Design

### Phase 1: Cα Chemical Shift Refinement (runs now)
Uses `diff_biophys.nmr.chemical_shifts.predict_ca_shifts` against 91 measured Cα shifts from BMRB 17020.

```bash
python benchmark_2KZV.py --steps 500
```

**Phase 1 results (300 steps):**
| | Cα RMSD (ppm) |
|---|---|
| NMR model 1 (before) | 1.542 |
| After refinement | 1.356 |
| Improvement | **+0.186 ppm** |

### Phase 2: RDC Q-factor Refinement (needs RDC data)
Uses `diff_biophys.nmr.rdc.fit_saupe_tensor` + `calculate_rdc_from_tensor` against ¹⁵N-¹H RDCs.

**Published benchmarks (PAG medium, Table 5 of Li et al.):**
| Model | r² | Q-factor |
|---|---|---|
| NMR medoid | 0.95 | **0.18** ← target |
| AF2 Model 1 | 0.92 | 0.22 ← starting point |
| diff-biophys (expected) | TBD | TBD |

```bash
# Once rdc_PAG.tsv is populated:
python benchmark_2KZV.py --rdc rdc_PAG.tsv --steps 500 --w-ca 0.5 --w-rdc 1.0
```

---

## Data Acquisition

The RDC data is in the RPI GitHub repository:
> **[github.rpi.edu/RPIBioinformatics/BlindAssessmentMonomericAF2Data](https://github.rpi.edu/RPIBioinformatics/BlindAssessmentMonomericAF2Data)**

This is RPI's GitHub Enterprise (requires RPI login). Contact **Tejero** or **Montelione** for:
- RDC table for 2KZV in PAG medium (primary target)
- RDC table for 2KZV in PEG medium (secondary validation)

---

## Expected Gradient Behavior

The published data tells us what to expect:
- **AF2 model** has Q=0.22 in PAG, **NMR structure** has Q=0.18
- The AF2 bond-vector orientations are close but not optimal for the alignment tensor
- `fit_saupe_tensor` in the loss function re-fits the tensor each step (SVD, differentiable)
- Gradient descent should incrementally rotate local backbone segments toward better N-H alignment
- Known outliers: **Thr14** and **Ser54** (both in α-helices, unusual PEG dynamics)

---

## NMR Observable Modules Used

| Observable | diff-biophys function |
|---|---|
| Cα chemical shifts | `diff_biophys.nmr.chemical_shifts.predict_ca_shifts` |
| RDC (general tensor) | `diff_biophys.nmr.rdc.calculate_rdc_from_tensor` |
| Alignment tensor fit | `diff_biophys.nmr.rdc.fit_saupe_tensor` |
| Q-factor | `diff_biophys.nmr.rdc.calculate_q_factor` |
| Backbone builder | `diff_biophys.geometry.nerf.chain_nerf` |
| Optimizer | `optax.adam` |
