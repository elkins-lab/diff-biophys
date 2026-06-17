# HR2876B diff-biophys Benchmark

**Protein**: N-terminal domain of human NFU1
**Target**: HR2876B (CASD-NMR 2013 Blind Assessment Target)
**PDB**: [2LTM](https://www.rcsb.org/structure/2LTM) | **BMRB**: [18489](https://bmrb.io/data_library/summary/index.php?bmrbId=18489)
**Residues**: 107

## Overview

This benchmark validates the `diff-biophys` geometry and NMR kernels against **HR2876B**, a target from the CASD-NMR 2013 blind assessment of automated NMR structure determination. The data is exceptionally clean, publicly available in a single NMR-STAR file, and provides a rigorous test of joint refinement against both chemical shifts and multiple Residual Dipolar Coupling (RDC) datasets.

**References:**
> Rosato, A., et al. (2015). "Blind testing of routine, fully automated determination of protein structures from NMR data." *Structure*.
> Reported NMR Medoid Q-factor: **0.32**

## Experimental Data Used
- **Cα Chemical Shifts:** 97 assigned residues.
- **RDC Media 1 (RDC_list_1):** 72 ¹⁵N-¹H values.
- **RDC Media 2 (RDC_list_2):** 75 ¹⁵N-¹H values.

*Note: With >70 RDCs for a 107-residue protein, the alignment tensors are highly overdetermined (~14x the 5 free parameters), making this an ideal system for evaluating backbone refinement without the severe risk of overfitting seen in sparse datasets.*

## Quick Start

```bash
# 1. Download the PDB and BMRB data files
python fetch_data.py

# 2. Run Phase 1: Refinement against Cα shifts only
python benchmark_HR2876B.py

# 3. Run Phase 2: Joint refinement (Cα shifts + both RDC media)
python benchmark_HR2876B.py --rdc all
```

## Expected Results (Model 1)

### Phase 1 Result (Cα shifts only, 500 steps)

| | Cα RMSD (ppm) |
|---|---|
| Before | 1.710 |
| After  | 1.530 |
| Δ | **+0.181 ppm** |

### Phase 2 Result (Cα + 2 RDC media, 500 steps)

| Observable | Before | After | Δ | Target |
|---|---|---|---|---|
| **Cα RMSD (ppm)** | 1.710 | 1.716 | -0.005 | - |
| **Q-factor (List 1)** | 0.639 | **0.092** | +0.547 | 0.32 |
| **Q-factor (List 2)** | 0.691 | **0.129** | +0.562 | 0.32 |

### ⚠️ The Overfitting Phenomenon

You will notice that the refined Q-factors (~0.10) drop significantly below the published NMR medoid target (0.32). **This is an example of mathematical overfitting, not a superior physical model.**

Why does this happen?
1. **Degrees of Freedom:** HR2876B has 107 residues, giving the optimizer **214 backbone angles** (φ and ψ) to manipulate.
2. **Constraints:** We are only providing 147 RDC constraints (72 + 75) and 97 loose Cα shift constraints.

The system is globally **underdetermined**. Because `diff-biophys` is a pure mathematical optimization engine without strict physical priors (like steric repulsion or hydrogen bond terms by default), the optimizer will happily contort the backbone into physically unrealistic conformations just to perfectly align the N-H vectors with the target RDC values.

This benchmark serves as an excellent educational example: while RDCs are incredibly powerful for global orientation, relying *solely* on differentiable physics without regularizing physical forces will inevitably lead to overfitting in underdetermined systems.
