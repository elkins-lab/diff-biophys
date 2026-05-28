# 🧬 DiffBiophys: Differentiable Biophysics Kernels

[![PyPI version](https://img.shields.io/pypi/v/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/diff-biophys.svg)](https://pypi.org/project/diff-biophys/)
[![Tests](https://github.com/elkins/diff-biophys/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-biophys/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DiffBiophys is the foundational engine for the Differentiable Biophysics suite. It provides high-performance JAX kernels for back-calculating experimental observables from atomic coordinates.

---

### 🧪 For Structural Biologists
*   **Comprehensive Kernels:** Includes validated models for NMR (Chemical Shifts, RDCs, Karplus, Ring Currents), SAXS (Debye formula, Hydration shells), and CD.
*   **Physical Accuracy:** Every kernel is benchmarked against standard non-differentiable packages like SHIFTX2 and CRYSOL.

### 🤖 For Machine Learning Geeks
*   **Pure JAX:** Every function is end-to-end differentiable and JIT-compilable.
*   **Plug-and-Play Losses:** Easily integrate SAXS or NMR constraints as loss terms in your PyTorch or JAX molecular models.

---

## 🚀 Supported Kernels

*   **NMR:** Chemical Shifts (Coil, Secondary Structure, Ring Currents), RDCs (Saupe Tensor), J-Couplings (Karplus).
*   **SAXS:** Debye Equation, Hydration Shells, Rg calculation.
*   **Geometry:** Differentiable NeRF (Kinematics), Kabsch Alignment, PBC utilities.

## 📦 Installation

```bash
pip install diff-biophys
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
