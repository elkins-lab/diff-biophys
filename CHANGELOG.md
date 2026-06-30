# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-30

### Changed
- **Test Infrastructure**: Enabled parallel testing via `pytest-xdist`, isolated slow interactive notebook tests with `@pytest.mark.slow`, and correctly configured the JAX compilation cache.
- **Linting & Typing**: Fixed minor docstring formatting issues in `diff_biophys/nmr/chemical_shifts.py` for MkDocs strict builds, and resolved `mypy` typing errors for `tests/` and `examples/` directories.

## [0.2.0] - 2026-06-17

### Added
- **Benchmarking Suite**: Validated `diff-biophys` against experimental NMR data for proteins 2KZV and GmR58A (Li et al. 2023).
- **Refinement Factories**: New high-level APIs `make_rdc_refinement_fns` and `make_ca_shift_loss` to simplify experimental data integration.
- **Robust Backbone I/O**: Direct support for loading PDB models and extracting N–CA–C coordinates into JAX arrays via `biotite`.
- **Edge Case Test Suite**: Comprehensive tests for numerical singularities (CD), collinearity (Kabsch), and empty input handling.

### Fixed
- **RDC Overfitting**: Implemented fixed-tensor refinement (periodically updated) to prevent trivial Q-factor minimization.
- **NH Bond Reconstruction**: Improved peptide-plane bisector method for more accurate NH vector orientations from backbone coordinates.
- **Test Coverage**: Achieved 100% coverage across all core biophysical modules.

## [0.1.6] - 2026-06-14

### Fixed
- Fixed syntax error in `examples/interactive_tutorials/03_cd_spectroscopy.ipynb`:
  a stray backslash continuation in the notebook generator fused the docstring
  of `cd_at_222nm` with its first statement, causing a `SyntaxError` in Colab.
  Fixed in both the notebook and the generator source.

### Added
- **Regression test suite for four scientific-review findings** (Issues 1–4):
  - *Issue 1 (NeRF geometry):* `test_nerf_dihedral_roundtrip` (parametrized
    over 13 angles, verifies `position_atom_3d → compute_dihedrals` roundtrip
    to < 0.01°) and `test_nerf_alpha_helix_is_right_handed` (cross-product
    handedness check on ideal α-helix chain).
  - *Issue 2 (Karplus offset):* `test_karplus_requires_phi_offset_not_raw_phi`
    quantifies the > 1 Hz error from passing raw φ instead of θ = φ − 60°.
  - *Issue 3 (chemical shift cap):* `test_ca_shift_max_helix_offset_is_half_of_offset_helix`
    pins the 50% effective-weight cap that is intrinsic to the softmax coil prior.
  - *Issue 4 (SAXS excluded volume):* `tests/test_saxs_excluded_volume_fraser.py`
    validates the Fraser (1978) 4π Gaussian denominator via q → 0 limit,
    q → ∞ limit, 1/e decay point, and linear ln(EV) vs q² slope test.
- Expanded `calculate_karplus_j` docstring with an `.. important::` block
  documenting the mandatory θ = φ − 60° offset for ³J(HN,Hα).
- Added inline `NOTE` comment in `chemical_shifts.py` explaining the 50%
  maximum helical weight and how to compensate for quantitative SPARTA+ parity.

### Removed
- Removed `examples/interactive_tutorials/generate_notebooks.py` from version
  control (added to `.gitignore`). The `.ipynb` files are the primary source;
  keeping both caused two-source-of-truth drift (the notebook syntax error
  originated from a bug in the generator).

## [0.1.5] - 2026-06-12

### Added
- **Differentiable Radius of Gyration (Rg) kernel** (`diff_biophys.saxs.rg`):
  fully differentiable Rg calculation, compatible with JAX `grad` and `vmap`.
- **Differentiable `Ensemble` class** (`diff_biophys.ensemble`): population-weighted
  ensemble averaging now supports `jax.grad` end-to-end, enabling ensemble
  refinement against experimental observables.
- **Comprehensive autodiff and geometry test suites**: new tests covering
  autodifferentiation through all kernels, geometry round-trips, and end-to-end
  gradient-descent refinement scenarios.
- **SAXS topology and RDC physics validation tests** for scientific correctness.
- **Three undergraduate interactive tutorials** (Jupyter notebooks):
  `01_hello_gradient_descent`, `02_nmr_fundamentals`, `03_cd_spectroscopy`;
  all configured for one-click launch on Google Colab.
- **Comprehensive documentation overhaul**: expanded `theory.md` (~950 lines),
  `use_cases.md` (~550 lines), `concepts.md` (~290 lines); new API reference
  pages for geometry, NMR, CD/cryo-EM, and SAXS modules.
- OpenSSF Best Practices badge and associated documentation.

### Fixed
- Corrected documentation URLs to the `elkins-lab` GitHub organization.
- Fixed `codecov-action` v5 syntax (`file` → `files`) in CI workflow.
- Pinned GitHub Actions to Node.js 24 to resolve deprecation warnings.
- Kernel numerical stability improvements in `ensemble.py` and `rdc.py`.

## [0.1.4] - 2026-06-11

### Added
- Comprehensive property-based testing using `hypothesis` to verify Kabsch alignment and NeRF kinematics geometry.
- New numerical stability checks testing edge cases (e.g. `sinc` limits in Debye scattering and overlaps in CD matrix method).
- Added `hypothesis` to `pyproject.toml` `dev` optional-dependencies.
- Added an Interdisciplinary "Concepts & Context" guide to documentation to clarify jargon for ML researchers and biologists.

### Fixed
- Fixed notebook integrity test relying on strict directory structure by recursively globbing for notebooks (`rglob`).
- RDC Q-factor edge case when experimental data is strictly zero.

## [0.1.3] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.
