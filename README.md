# 🧬 DiffBiophys: Differentiable Biophysics for the AI Era

**DiffBiophys** is a high-performance Python library for differentiable biophysical modeling. Built on **JAX**, it re-implements core structural biology and spectroscopy observables (SAXS, NMR, CD) as hardware-accelerated, auto-differentiable kernels.

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
- **Hardware Acceleration:** GPU-optimized pairwise distance kernels.
- **Use Case:** Fitting structure "compactness" and "radius of gyration" to solution-state X-ray scattering curves.

### 3. `diff_biophys.nmr` (Differentiable Spectroscopy)
- **Residual Dipolar Couplings (RDCs):** Differentiable Saupe tensor alignment and coupling calculation.
- **Chemical Shifts:** Differentiable Ring-Current (Johnson-Bovey) shielding and Karplus J-coupling kernels.
- **Use Case:** Refining side-chain packing and domain orientations against high-resolution NMR data.

### 4. `diff_biophys.cd` (Differentiable Dichroism)
- **Matrix-Method Simulation:** Differentiable simulation of peptide bond transition dipole coupling.
- **Use Case:** Predicting secondary structure content and verifying fold stability.

---

## ⚡ Technical Architecture

- **Backend:** JAX (XLA-compiled).
- **Parallelism:** Native support for `vmap` (vectorization across ensembles/trajectories) and `pmap` (multi-device execution).
- **Differentiability:** Support for both Forward and Reverse-mode autodiff.
- **Interoperability:** Seamless integration with PyTorch/TensorFlow (via DLPack) and standard structural formats (mmCIF/BCIF).

---

## 🚀 Roadmap

### Phase 1: Foundations (Alpha)
- [ ] Differentiable NeRF and Kabsch alignment.
- [ ] GPU-accelerated Debye formula for SAXS.
- [ ] Unit tests verifying parity with `synth-pdb` NumPy implementations.

### Phase 2: NMR & Spectroscopy (Beta)
- [ ] Differentiable RDC and Karplus kernels.
- [ ] Differentiable Johnson-Bovey ring current model.
- [ ] Integration with `synth-nmr` parameter libraries.

### Phase 3: Integration & Optimization (v1.0)
- [ ] Example notebooks for structure refinement via gradient descent.
- [ ] Plugin for `torch`-based AI models to use biophysical loss functions.
- [ ] Full support for BinaryCIF streaming.

---

## 📂 Repository Structure (Proposed)

```text
diff-biophys/
├── diff_biophys/          # Core package
│   ├── geometry/          # NeRF, Kabsch, Torsions
│   ├── saxs/              # Debye kernels, form factors
│   ├── nmr/               # RDCs, Karplus, Ring Currents
│   ├── cd/                # CD simulation
│   └── utils/             # Constants, JAX-NumPy shims
├── tests/                 # Parity and gradient checks
├── examples/              # Jupyter notebooks (Refinement Lab)
├── docs/                  # API and Theory
├── pyproject.toml         # Modern build config
└── README.md
```
