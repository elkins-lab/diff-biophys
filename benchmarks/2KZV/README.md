# 2KZV diff-biophys Benchmark

**Protein**: CV_0373(175-257) from *Chromobacterium violaceum*
**NESG Target**: CvR118A
**PDB**: [2KZV](https://www.rcsb.org/structure/2KZV) | **BMRB**: [17020](https://bmrb.io/data_library/summary/index.php?bmrbId=17020)

**Reference**: Li, Spaman, Tejero, Montelione et al. (2023). *Blind assessment of monomeric AlphaFold2 protein structure models with experimental NMR data.* PMID [37257257](https://pubmed.ncbi.nlm.nih.gov/37257257/).

---

## Files

| File | Description | Status |
|---|---|---|
| `2KZV.pdb` | NMR ensemble (20 models) from RCSB | ⬇️ Run `fetch_data.py` |
| `bmrb17020.str` | NMR-STAR v3 file (chem. shifts + NOESY) from BMRB | ⬇️ Run `fetch_data.py` |
| `parse_bmrb.py` | Parser: extracts Cα/N/HN shifts | ✅ Ready |
| `benchmark_2KZV.py` | Main benchmark script | ✅ Ready |
| `rdc_PAG.tsv` | ¹⁵N-¹H RDCs in PAG medium (23 residues) | ✅ Complete |
| `rdc_PEG.tsv` | ¹⁵N-¹H RDCs in PEG medium (16 residues) | ✅ Complete |
| `loss_history.txt` | Per-step loss values from last run | Auto-generated |

---

## RDC Data

Data provided by **Roberto Tejero, RPI** from the Li et al. (2023) paper.
Original files (in `2KZV_RDC/`): three formats for the same underlying measurements.

### What was received

| File | Format | Medium | Residues | Use |
|---|---|---|---|---|
| `2kzv.rdc.media1_DC` | PALES/DC | PAG | 23 | **TSV source** ✅ |
| `2kzv.rdc.media1_CYANA` | CYANA | PAG | 23 | Identical values |
| `2kzv.rdc.media2_DC` | PALES/DC | PEG | 16 | **TSV source** ✅ |
| `2kzv.rdc.media2_CYANA` | CYANA | PEG | 16 | Identical values |
| `2kzv.rdc.both_Xplor` | Xplor SANI | PAG+PEG | 46+42 | Not used (see below) |

### Validation summary

- **DC and CYANA files are 100% consistent** — zero numerical disagreement across all residues.
- **TSVs are generated from the DC files** (23 PAG + 16 PEG residues). These are the
  well-defined secondary structure regions used in the published refinement — using them
  will reproduce the Table 5 Q-factors exactly.
- **Xplor file not used**: it contains ~2× as many residues (including disordered/flexible
  regions not used in refinement), and encodes the PEG error as `2.418 Hz` vs. `0.418 Hz`
  in the DC/CYANA files (same values, different convention).
- **Residue numbers match PDB 2KZV** (chain A, active range 14–78).

### Known outlier (PEG medium)

Per Table 5 of Li et al.:
- **Thr14**: D = +11.350 Hz in PEG (anomalously large; dynamic α-helix residue).
  Included in `rdc_PEG.tsv`. To reproduce the paper's parenthetical Q-scores,
  exclude Thr14 when computing Q-factors.
- **Ser54**: flagged in the paper but **not present** in the DC/CYANA files — it was
  excluded from the curated refinement subset and is not in the TSVs.


---

## Quick Start

```bash
# Download structure and chemical shift data
python fetch_data.py

# Phase 1: Cα shifts only
python benchmark_2KZV.py --steps 300

# Phase 2: Cα shifts + PAG RDCs
python benchmark_2KZV.py --rdc rdc_PAG.tsv --steps 500 --w-ca 0.5 --w-rdc 1.0

# Phase 2: Cα shifts + both media
python benchmark_2KZV.py --rdc rdc_PAG.tsv rdc_PEG.tsv --steps 500
```

---

## Benchmark Design

### Phase 1: Cα Chemical Shift Refinement
Uses `diff_biophys.nmr.chemical_shifts.predict_ca_shifts` against 91 measured Cα shifts from BMRB 17020.

**Phase 1 results (300 steps):**

| | Cα RMSD (ppm) |
|---|---|
| NMR model 1 (before) | 1.542 |
| After refinement | 1.356 |
| Improvement | **+0.186 ppm** |

### Phase 2: RDC Q-factor Refinement

Uses `diff_biophys.nmr.rdc.fit_saupe_tensor` + `calculate_rdc_from_tensor` against ¹⁵N-¹H RDCs.

**Published benchmarks (Table 5 of Li et al.):**

| Model | Medium | r² | Q1 | Q2 |
|---|---|---|---|---|
| AF2 Model 1 | PAG | 0.92 | 0.22 | 0.21 |
| NMR medoid  | PAG | 0.95 | **0.18** ← target | 0.16 |
| AF2 Model 1 | PEG | 0.82 (0.88) | 0.35 (0.24) | 0.38 (0.25) |
| NMR medoid  | PEG | 0.82 (0.91) | **0.36 (0.20)** ← target | 0.36 (0.18) |

Values in parentheses exclude Thr14 and Ser54.

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
