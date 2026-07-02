# TODO: Dans_Diffraction validation oracle

Implement the backend that lets `Dans_Diffraction` act as an optional,
independent validation oracle without becoming production physics.

Scope for this branch:

- Add lazy optional loading/building of `Dans_Diffraction.Crystal` objects from
  internal `CRYSTALS` entries and CIF paths.
- Compare lattice parameters, unit-cell volume, reciprocal-vector magnitudes,
  and `|F_hkl|^2` for selected HKLs.
- Add fast unit tests using fake oracle objects so the core test suite does not
  require `Dans-Diffraction`.
- Add a checks-level example script for a few representative materials.
- Keep cxr_mc atomic form factors and structure-factor calculations as the
  production implementation; use `Dans_Diffraction` only as a comparator.
