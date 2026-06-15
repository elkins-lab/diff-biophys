# GmR58A diff-biophys Benchmark

**Protein**: GmR58A from *Geobacter metallireducens*
**PDB**: [2KUT](https://www.rcsb.org/structure/2KUT) (10-model NMR ensemble) | **BMRB**: [16746](https://bmrb.io/data_library/summary/index.php?bmrbId=16746)

**Why this is a better benchmark than 2KZV**: All data is fully public in BMRB 16746 — no institutional GitHub access required. The dataset includes **3 independent alignment media**, which over-determines the Saupe tensor and makes the benchmark much harder to overfit than a single-medium measurement.

---

## Files

| File | Description | Status |
|---|---|---|
| `parse_nmrstar.py` | Universal NMR-STAR v3 parser (shifts + RDC saveframes) | ✅ Ready |
| `benchmark_GmR58A.py` | Main benchmark (Phase 1 + Phase 2, `--rdc` flag) | ✅ Ready |
| `fetch_data.py` | Downloads 2KUT.pdb + bmrb16746.str | ✅ Ready |
| `2KUT.pdb` | NMR ensemble from RCSB | ⬇️ Run `fetch_data.py` |
| `bmrb16746_GmR58A.str` | NMR-STAR v3 file from BMRB | ⬇️ Run `fetch_data.py` |
| `loss_history.txt` | Per-step loss values from last run | Auto-generated |

---

## Quick Start

```bash
# 1. Download data files
python fetch_data.py

# 2. Cα shifts only (Phase 1)
python benchmark_GmR58A.py --steps 300

# 3. Cα shifts + all 3 RDC alignment media (full benchmark)
python benchmark_GmR58A.py --steps 500 --rdc all

# 4. Single RDC medium (e.g., only stretched gel)
python benchmark_GmR58A.py --steps 300 --rdc RDC_list_1
```

---

## Benchmark Design

### Observables (all from BMRB 16746)

| Observable | Residues | Module |
|---|---|---|
| Cα chemical shifts | 114 | `diff_biophys.nmr.chemical_shifts.predict_ca_shifts` |
| ¹⁵N-¹H RDC (RDC_list_1, stretched gel) | 43 | `diff_biophys.nmr.rdc.*` |
| ¹⁵N-¹H RDC (RDC_list_2, negative gel) | 59 | `diff_biophys.nmr.rdc.*` |
| ¹⁵N-¹H RDC (RDC_list_3, PEG) | 53 | `diff_biophys.nmr.rdc.*` |

### Why 3 Alignment Media Matters

With a single alignment medium, the 5 Saupe tensor parameters are determined from N RDC values — any set of bond vectors can always be fit better than the data. With 3 media, the 3×5 = 15 tensor parameters must be simultaneously satisfied by the same set of bond vectors. This is the gold standard for RDC-based structure refinement.

### Phase 1 Result (Cα shifts only, 300 steps)

| | Cα RMSD (ppm) |
|---|---|
| Before | 1.279 |
| After  | 1.199 |
| Δ | **+0.080 ppm** |

### Phase 2 Result (Cα + 3 RDC media, 300 steps)

| Observable | Before | After | Δ |
|---|---|---|---|
| Cα RMSD (ppm) | 1.279 | 1.199 | +0.080 |
| Q (RDC_list_1, stretched gel) | 0.932 | 0.092 | +0.840 |
| Q (RDC_list_2, negative gel)  | 0.916 | 0.400 | +0.517 |
| Q (RDC_list_3, PEG)           | 0.897 | 0.291 | +0.606 |

> **Note on baseline Q-factors**: The high baseline Q (≈0.9) reflects that the NMR ensemble model 1
> has not been aligned to any tensor. Starting from an AF2 prediction (run `--model` from a separate
> AF2 PDB) will give more interpretable before/after comparisons matching the Li et al. (2023) framing.

---

## Comparison with 2KZV Benchmark

| | **2KZV** (CvR118A) | **GmR58A** |
|---|---|---|
| Residues | 83 | 122 |
| RDC media | 2 (PAG, PEG) | **3** (Gel+, Gel−, PEG) |
| RDC data location | RPI GitHub (private) | **BMRB 16746 (public)** |
| Chemical shifts | BMRB 17020 ✅ | BMRB 16746 ✅ |
| Published AF2 comparison | Li et al. 2023 ✅ | None yet (new) |

---

## NMR-STAR Parser

`parse_nmrstar.py` is a universal parser for any BMRB entry. It handles:
- `save_assigned_chem_shift_list_1` → Cα, N, HN shifts
- Any `save_RDC_*` saveframe → ¹⁵N-¹H RDC values with errors

This replaces the 2KZV-specific `parse_bmrb.py` and can be used for any future benchmark.
