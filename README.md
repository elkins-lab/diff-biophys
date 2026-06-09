# 🧬 DiffBiophys: Differentiable Biophysics for the AI Era

[![Tests](https://github.com/elkins/diff-biophys/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-biophys/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/elkins/diff-biophys/branch/main/graph/badge.svg)](https://codecov.io/gh/elkins/diff-biophys)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/diff-biophys/blob/main/examples/diff_biophys_showcase.ipynb)

**DiffBiophys** is a high-performance Python library for differentiable biophysical modeling. Built on **JAX**, it re-implements core structural biology and spectroscopy observables (SAXS, NMR, CD) as hardware-accelerated, auto-differentiable kernels.

**[Documentation Website](https://elkins.github.io/diff-biophys/)** | **[Use Cases](https://elkins.github.io/diff-biophys/use_cases/)**

---

## 🎯 Vision

To bridge the gap between static structural models and experimental solution-state data by providing a "differentiable bridge." This allows researchers to:
1. **Optimize** protein structures directly against experimental spectra via gradient descent.
2. **Train** machine learning models using physics-informed loss functions.
3. **Accelerate** large-scale biophysical simulations on GPUs and TPUs.

---

## 🏗️ Core Components

### 1. `diff_biophys.geometry` (Differentiable Structural Engine)
- **NeRF (Natural Extension Reference Frame):** Differentiable conversion from internal coordinates ($\phi, \psi, \omega$, bond lengths/angles) to Cartesian XYZ.
- **Kabsch Alignment:** Differentiable optimal superposition using SVD.
- **Torsion Analysis:** Vectorized calculation of all backbone and side-chain dihedrals.

### 2. `diff_biophys.saxs` (Differentiable Scattering)
- **Debye Formula:** $O(N^2)$ inter-atomic interference summation.
- **Hydration Shell Correction:** Excluded-volume solvent subtraction (Fraser et al. 1978).
- **Hardware Acceleration:** GPU-optimized pairwise distance kernels via JAX `vmap`.
- **Use Case:** Fitting structure compactness and radius of gyration to solution-state X-ray scattering curves.

### 3. `diff_biophys.nmr` (Differentiable Spectroscopy)
- **Residual Dipolar Couplings (RDCs):** Differentiable Saupe tensor alignment and coupling calculation. Includes SVD-based tensor fitting.
- **Chemical Shifts:** Differentiable ring-current (Johnson-Bovey) shielding and softmax-weighted secondary structure Cα shift predictor.
- **Karplus J-coupling:** Parameterizable 3J coupling equation (Vuister & Bax 1993 defaults).
- **Use Case:** Refining side-chain packing and domain orientations against high-resolution NMR data.

### 4. `diff_biophys.cd` (Differentiable Dichroism)
- **Matrix-Method Simulation:** Differentiable simulation of peptide bond transition dipole coupling via DeVoe theory.
- **Status:** ✅ Implemented. Supports frequency-dependent coupled-oscillator response.

---

## ⚡ Technical Architecture

- **Backend:** JAX (XLA-compiled) — supports CPU, GPU, and TPU.
- **Parallelism:** Native support for `vmap` (vectorization across ensembles/trajectories) and `pmap` (multi-device execution).
- **Differentiability:** Forward and reverse-mode autodiff through all kernels.
- **Interoperability:** JAX arrays are compatible with NumPy and can be exchanged with PyTorch via `dlpack` (user-managed conversion).

---

## 🚀 Roadmap

### Phase 1: Foundations (Alpha)
- [x] Differentiable NeRF and Kabsch alignment.
- [x] GPU-accelerated Debye formula for SAXS with hydration shell correction.
- [x] Unit tests verifying parity with `synth-pdb` NumPy implementations.

### Phase 2: NMR & Spectroscopy (Beta)
- [x] Differentiable RDC and Karplus kernels.
- [x] Differentiable Johnson-Bovey ring current model.
- [x] Integration with `synth-nmr` parameter libraries (optional dependency).

### Phase 3: Integration & Optimization (v1.0)
- [x] Full CD matrix-method implementation (DeVoe theory).
- [ ] Example notebooks for structure refinement via gradient descent.
- [ ] Plugin for `torch`-based AI models to use biophysical loss functions.
- [ ] Full support for BinaryCIF streaming.

---

## 📂 Repository Structure

```text
diff-biophys/
├── diff_biophys/          # Core package
│   ├── geometry/          # NeRF, Kabsch, Torsions
│   ├── saxs/              # Debye kernels, form factors
│   ├── nmr/               # RDCs, Karplus, Ring Currents, Chemical Shifts
│   ├── cd/                # CD simulation (DeVoe Matrix Method)
│   └── ensemble.py        # Ensemble averaging API
├── tests/                 # Parity, gradient, and scientific validation checks
├── examples/              # Jupyter notebooks (Refinement Lab)
├── docs/                  # API and Theory
├── pyproject.toml         # Modern build config
└── README.md
```

## 🚀 Installation

```bash
pip install diff-biophys
```

For GPU support (CUDA):
```bash
pip install "jax[cuda12]" diff-biophys
```

## 🤝 Contributing

Contributions are welcome from both ML and structural biology communities! Please open an issue or pull request on [GitHub](https://github.com/elkins/diff-biophys). Run `pre-commit run --all-files` before submitting.

## 🔗 Related Projects

diff-biophys is the **differentiable engine** powering the higher-level tools in this ecosystem:

- [synth-pdb](https://github.com/elkins/synth-pdb) — Synthetic structure generation (uses NumPy implementations)
- [synth-nmr](https://github.com/elkins/synth-nmr) — NMR observables (optional dependency)
- [synth-saxs](https://github.com/elkins/synth-saxs) — SAXS profile simulator
- [diff-fret](https://github.com/elkins/diff-fret) — Differentiable FRET (new)
- [diff-hdx](https://github.com/elkins/diff-hdx) — Differentiable HDX-MS (new)
- [diff-epr](https://github.com/elkins/diff-epr) — Differentiable EPR/DEER (new)
- [diff-ensemble](https://github.com/elkins/diff-ensemble) — IDP ensemble VAE (depends on diff-biophys)
- [TorsionTuner](https://github.com/elkins/TorsionTuner) — GNN refinement (depends on diff-biophys)
- [resonance-flow](https://github.com/elkins/resonance-flow) — NMR-guided folding (depends on diff-biophys)

## ⚖️ License

MIT License — see [LICENSE](LICENSE) for details.

## 📖 Citation

```bibtex
@software{diff_biophys,
  author  = {Elkins, George},
  title   = {diff-biophys: Differentiable biophysics kernels for JAX},
  year    = {2024},
  url     = {https://github.com/elkins/diff-biophys},
  version = {0.1.2}
}
```
