# 🔬 Core Theory

DiffBiophys is based on several foundational biophysical principles:

## 1. Natural Extension Reference Frame (NeRF)
NeRF converts internal coordinates (bond lengths, angles, dihedrals) to Cartesian coordinates. By using JAX, we can differentiate through this conversion to optimize protein backbones.

## 2. Debye Scattering
The Debye formula calculates SAXS intensity by summing interference between all atom pairs:
$$I(q) = \sum_i \sum_j f_i(q) f_j(q) \frac{\sin(qr_{ij})}{qr_{ij}}$$
Our GPU-accelerated implementation makes this $O(N^2)$ calculation viable for large complexes.

### Scientific Validation: Kratky Plot Topology
A classic test for protein fold state is the Kratky plot ($q^2 I(q)$ vs $q$).
* **Globular (compact) structures** exhibit a distinct bell-shaped peak that drops towards zero.
* **Random coil (unfolded) structures** plateau or rise at high $q$.
DiffBiophys correctly reproduces these characteristic topological signatures, as verified in `tests/test_science_saxs_kratky.py`.

### Scientific Validation: Distance Distribution $P(r)$
The pair distance distribution function $P(r)$ is the inverse Fourier transform of the scattering intensity. For a uniform sphere of radius $R$, $P(r)$ has a known analytical form (Guinier 1939). DiffBiophys reproduces this analytical distribution for simulated spheres with $>0.98$ correlation (`tests/test_science_saxs_pr.py`).

## 3. Saupe Alignment Tensor
RDCs are calculated using the Saupe tensor approach, which describes the partial alignment of a molecule in an anisotropic medium.

### Scientific Validation: Angular Dependence
In the principal axis frame, the RDC follows the relationship:
$D(\theta) \propto 3\cos^2\theta - 1$
DiffBiophys correctly predicts zero coupling at the **Magic Angle** ($\sim 54.74^\circ$) and maximum coupling when the bond is parallel to the Z-axis, as verified in `tests/test_science_rdc_angular.py`.
