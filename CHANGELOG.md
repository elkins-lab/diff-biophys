# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
