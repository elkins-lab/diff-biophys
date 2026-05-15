# 🧬 DiffBiophys

Welcome to **DiffBiophys**, a high-performance Python library for **differentiable biophysical modeling**.

Built on **JAX**, it re-implements core structural biology and spectroscopy observables (SAXS, NMR, CD) as hardware-accelerated, auto-differentiable kernels.

## Why Differentiable?

Traditional biophysics libraries provide "forward models" (Structure -> Observable). DiffBiophys provides the gradient, enabling "inverse modeling" (Observable -> Structure).

- **Refine** protein loops directly against experimental NMR data.
- **Fit** structure ensembles to SAXS scattering curves.
- **Integrate** physical constraints into machine learning loss functions.

