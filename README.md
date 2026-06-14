# 🧬 Diff-Biophys: Differentiable Biophysics for the AI Era

[![Tests](https://github.com/elkins-lab/diff-biophys/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/diff-biophys/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/elkins-lab/diff-biophys/branch/main/graph/badge.svg)](https://codecov.io/gh/elkins-lab/diff-biophys)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13178/badge)](https://www.bestpractices.dev/projects/13178)

**Diff-Biophys** is a high-performance Python library for differentiable biophysical modeling. Built on **JAX**, it re-implements core structural biology and spectroscopy observables (SAXS, NMR, CD) as hardware-accelerated, auto-differentiable kernels.

**[Documentation Website](https://elkins-lab.github.io/diff-biophys/)** | **[Use Cases](https://elkins-lab.github.io/diff-biophys/use_cases/)** | **[Tutorials](#-interactive-tutorials)**

---

## 🎯 Vision

To bridge the gap between static structural models and experimental solution-state data by providing a "differentiable bridge." This allows researchers to:
1. **Optimize** protein structures directly against experimental spectra via gradient descent.
2. **Train** machine learning models using physics-informed loss functions.
3. **Accelerate** large-scale biophysical simulations on GPUs and TPUs.

---

## 🌉 The Interdisciplinary Bridge

`diff-biophys` sits at the intersection of **Machine Learning** and **Structural Biology**. If you find the terminology confusing, please read our **[Concepts & Context Guide](https://elkins-lab.github.io/diff-biophys/concepts/)**! It acts as a "Rosetta Stone" to explain:
* **For ML Engineers:** What SAXS and NMR are, and why traditional physics code can't be used in PyTorch/JAX loss functions.
* **For Biologists:** What automatic differentiation is, why JAX is used instead of traditional Monte Carlo/Simulated Annealing, and how it enables optimization on GPUs.

---

## 📚 Interactive Tutorials

Experience **Diff-Biophys** directly in your browser with our Colab tutorials:

| Tutorial | Audience | Description | Action |
| :--- | :--- | :--- | :--- |
| [**🎓 Hello, Gradient Descent!**](examples/interactive_tutorials/01_hello_gradient_descent.ipynb) | Undergrad (any) | No biology needed. Learn what a gradient is, how gradient descent works, and how JAX computes gradients automatically — then fit a real Karplus curve. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/01_hello_gradient_descent.ipynb) |
| [**🔬 NMR Fundamentals**](examples/interactive_tutorials/02_nmr_fundamentals.ipynb) | Undergrad (bio/chem) | Chemical shifts, the Karplus equation, RDCs, and the magic angle — computed differentiably and connected back to protein backbone torsion angles. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/02_nmr_fundamentals.ipynb) |
| [**💡 CD Spectroscopy**](examples/interactive_tutorials/03_cd_spectroscopy.ipynb) | Undergrad (bio/chem) | Build an α-helix from scratch, simulate its CD spectrum via the DeVoe model, watch it change as the helix unwinds, and compute the gradient of [θ]₂₂₂ w.r.t. atomic positions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/03_cd_spectroscopy.ipynb) |
| [**🧪 Diff-Biophys Showcase**](examples/interactive_tutorials/diff_biophys_showcase.ipynb) | Graduate / researcher | A complete overview of the JAX-accelerated SAXS and NMR kernels. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/diff_biophys_showcase.ipynb) |
| [**⚗️ Structure Refinement Lab**](examples/interactive_tutorials/structure_refinement.ipynb) | Graduate / researcher | Use gradient descent to optimize protein structures against experimental SAXS profiles. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/structure_refinement.ipynb) |

---

## 🏗️ Core Components

### 1. `diff_biophys.geometry` (Differentiable Structural Engine)
- **NeRF (Natural Extension Reference Frame):** Differentiable conversion from internal coordinates ($\phi, \psi, \omega$, bond lengths/angles) to Cartesian XYZ.
- **Kabsch Alignment:** Differentiable optimal superposition using SVD.
- **Torsion Analysis:** Vectorized calculation of all backbone and side-chain dihedrals.
- **Macroscopic Properties:** Differentiable Radius of Gyration ($R_g$) for driving compaction/expansion during structural optimization.

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

## 🧪 Scientific Validation

DiffBiophys is validated against foundational biophysical principles and analytical solutions to ensure physical realism:

* **SAXS Guinier Approximation:** Recovers correct $R_g$ from low-q scattering slopes (`test_saxs_guinier.py`).
* **SAXS Analytic Sphere:** Reproduces the theoretical scattering profile of a uniform sphere (`test_science_saxs_sphere.py`).
* **SAXS Kratky Topology:** Correctly distinguishes between globular and unfolded topologies via Kratky plot signatures (`test_science_saxs_kratky.py`).
* **SAXS $P(r)$ Distribution:** Matches analytical pair-distance distribution for spheres (Guinier 1939) with $>0.98$ correlation (`test_science_saxs_pr.py`).
* **NMR RDC Physics:** Verified 1/r³ distance scaling and $(3\cos^2\theta - 1)$ angular dependence, including zero coupling at the Magic Angle (`test_science_rdc_angular.py`).
* **NMR Ring Currents:** Reproduces shielding/deshielding cones of the Johnson-Bovey model (`test_science_ring_currents.py`).

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
- [x] Example notebooks for structure refinement via gradient descent.
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
├── examples/interactive_tutorials/              # Jupyter notebooks (Refinement Lab)
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

Contributions are welcome from both ML and structural biology communities! Please open an issue or pull request on [GitHub](https://github.com/elkins-lab/diff-biophys). Run `pre-commit run --all-files` before submitting.

## 🔗 Related Projects

diff-biophys is the **differentiable engine** powering the higher-level tools in this ecosystem:

- [synth-pdb](https://github.com/elkins-lab/synth-pdb) — Synthetic structure generation (uses NumPy implementations)
- [synth-nmr](https://github.com/elkins-lab/synth-nmr) — NMR observables (optional dependency)
- [synth-saxs](https://github.com/elkins-lab/synth-saxs) — SAXS profile simulator
- [diff-fret](https://github.com/elkins-lab/diff-fret) — Differentiable FRET (new)
- [diff-hdx](https://github.com/elkins-lab/diff-hdx) — Differentiable HDX-MS (new)
- [diff-epr](https://github.com/elkins-lab/diff-epr) — Differentiable EPR/DEER (new)
- [diff-ensemble](https://github.com/elkins-lab/diff-ensemble) — IDP ensemble VAE (depends on diff-biophys)
- [torsion-tuner](https://github.com/elkins-lab/torsion-tuner) — GNN refinement (depends on diff-biophys)
- [resonance-flow](https://github.com/elkins-lab/resonance-flow) — NMR-guided folding (depends on diff-biophys)

## ⚖️ License

MIT License — see [LICENSE](LICENSE) for details.

## 📖 Citation

```bibtex
@software{diff_biophys,
  author  = {Elkins, George},
  title   = {diff-biophys: Differentiable biophysics kernels for JAX},
  year    = {2026},
  url     = {https://github.com/elkins-lab/diff-biophys},
  version = {0.1.6}
}
```
