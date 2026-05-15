# 🔬 Core Theory

DiffBiophys is based on several foundational biophysical principles:

## 1. Natural Extension Reference Frame (NeRF)
NeRF converts internal coordinates (bond lengths, angles, dihedrals) to Cartesian coordinates. By using JAX, we can differentiate through this conversion to optimize protein backbones.

## 2. Debye Scattering
The Debye formula calculates SAXS intensity by summing interference between all atom pairs:
$$I(q) = \sum_i \sum_j f_i(q) f_j(q) \frac{\sin(qr_{ij})}{qr_{ij}}$$
Our GPU-accelerated implementation makes this $O(N^2)$ calculation viable for large complexes.

## 3. Saupe Alignment Tensor
RDCs are calculated using the Saupe tensor approach, which describes the partial alignment of a molecule in an anisotropic medium.

