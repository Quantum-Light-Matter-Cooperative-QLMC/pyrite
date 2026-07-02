# Research: Dans_Diffraction as a validation oracle for cxr-mc

**Date:** 2026-07-02  
**Depth:** standard  
**Branch:** `codex/dans-diffraction-research`  
**Question:** Should `Dans_Diffraction` be examined beyond the earlier alternatives scan, and if
so, what role should it play in `cxr-mc` after the `diffpy.structure` structural importer?
**Method:** Context7 lookup, then primary sources: GitHub README/repository, PyPI project
metadata, generated API documentation, and Zenodo DOI record. This is a research report only;
no implementation was performed.

---

## Executive Summary

`Dans_Diffraction` is a strong candidate for **independent validation checks**, not for the
core structural importer and not for replacing `cxr_mc` physics.

Its useful distinction from `diffpy.structure` is that it is not just a structure container:
it can read CIF files, expand crystallographic symmetry, calculate reflection lists, and compute
X-ray/neutron structure-factor intensities using its own scattering-factor tables and formulas.
That makes it valuable as a second implementation to compare against `cxr_mc.structure_factor`,
`chi_g`, and reflection-selection behavior.

The recommended role is:

1. Keep `diffpy.structure` as the production importer path.
2. Use `Dans_Diffraction` only in `checks/` or optional validation tests.
3. Compare a small material set, starting with NaCl/Si/sapphire, on lattice, expanded basis,
   selected HKLs, `|F_hkl|^2`, and powder/reflection list consistency.
4. Treat magnetic/resonant functionality as out of scope for initial validation because the
   project documentation itself cautions that magnetic structures are still under development.

The package is permissively licensed as Apache-2.0 in PyPI/GitHub metadata, but there is a
minor packaging-license caveat: the PyPI page says the Apache notice covers a named subset of
files and that other files may have their own license or no reuse license. This does not block
using it as an installed test dependency, but it argues against copying code or data from it into
`cxr_mc`.

**Recommendation:** Add an optional, dev/checks-only validation spike later. Do not add
`Dans_Diffraction` to the core runtime dependency set.

---

## Source Findings

### Package Identity and Maintenance

- GitHub repository: `DanPorter/Dans_Diffraction`, public, default branch `master`, described
  as reading crystallographic CIF files and simulating diffraction. The README currently presents
  "Version 3.4" and credits Dan Porter / Diamond Light Source 2025.
  Source: [GitHub README](https://github.com/DanPorter/Dans_Diffraction).
- PyPI package: `Dans-Diffraction`, current listed release `3.4.0` dated 2025-09-15.
  Source: [PyPI project page](https://pypi.org/project/Dans-Diffraction/).
- Zenodo DOI record exists for version 3.0.0, published 2023-07-02, with DOI
  `10.5281/zenodo.8106031`.
  Source: [Zenodo record](https://zenodo.org/records/8106031).
- Context7 does not have a usable `Dans_Diffraction` entry. The closest match was an unrelated
  optical diffraction package, `diffractio`, so primary sources are required for this library.

Confidence: **high** on package identity and release metadata.

### License and Dependency Profile

- PyPI metadata reports Apache Software License / Apache License and Python `>=3.7`.
  It also classifies the project as Development Status "3 - Alpha".
  Source: [PyPI project details](https://pypi.org/project/Dans-Diffraction/).
- The GitHub repository has an Apache-2.0 `LICENSE`.
  Source: [GitHub LICENSE](https://github.com/DanPorter/Dans_Diffraction/blob/master/LICENSE).
- The generated package docs include an Apache-2.0 notice for many core modules, including
  `classes_crystal.py`, `classes_scattering.py`, `functions_crystallography.py`, and
  `functions_scattering.py`, but state that other files may be under separate or missing reuse
  licenses.
  Source: [generated package docs](https://danporter.github.io/Dans_Diffraction/).
- README requirements are Python 3.7+ with NumPy, Matplotlib, and Tkinter; the package also uses
  built-in modules such as `sys`, `os`, `re`, `glob`, `warnings`, `json`, and `itertools`.
  Source: [GitHub README](https://github.com/DanPorter/Dans_Diffraction).

Confidence: **high** on metadata; **medium** on full transitive dependency behavior until an
installed-build smoke test is run.

### Relevant API Surface

High-level workflow:

- `import Dans_Diffraction as dif`
- `xtl = dif.Crystal("some_file.cif")`
- `xtl.info()` to inspect structure parameters.
- `xtl.Scatter.print_all_reflections(energy_kev=...)` for reflection lists.
- `xtl.Scatter.intensity([h, k, l])` for intensity.
- `xtl.Plot.simulate_powder(...)` for powder-pattern plotting.

Source: [GitHub README](https://github.com/DanPorter/Dans_Diffraction).

Structural model:

- `Crystal` owns `Cell`, `Symmetry`, `Atoms`, and `Structure`.
- `Atoms` stores asymmetric atomic positions with fractional `u`, `v`, `w`, element `type`,
  label, occupancy, displacement factor, and optional magnetic moment.
- `Structure` is the full P1-expanded structure after symmetry expansion.
- `Crystal.generate_structure()` combines atomic positions with symmetry operations.
- `Crystal.write_cif()` can write basic CIF/MCIF output.

Sources:
- [classes_crystal docs](https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.classes_crystal.html)
- [PyPI operation examples](https://pypi.org/project/Dans-Diffraction/)

Scattering model:

- `Scatter.setup_scatter(scattering_type="x-ray", energy_keV=...)` supports x-ray, neutron,
  x-ray magnetic, neutron magnetic, x-ray resonant, and x-ray dispersion modes.
- `Scatter.intensity()` returns `|F|^2` and documents the calculation as a sum over atomic
  scattering factor, occupancy, Debye-Waller term, and a crystallographic phase factor.
- Lower-level `functions_scattering` exposes `phase_factor`, `phase_factor_qr`,
  `autostructurefactor`, `autointensity`, and `intensity`.
- `functions_crystallography` documents Waasmaier-Kirfel X-ray scattering factors and points to
  DiffPy's `f0_WaasKirf.dat` as a data source.

Sources:
- [classes_scattering docs](https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.classes_scattering.html)
- [functions_scattering docs](https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.functions_scattering.html)
- [functions_crystallography docs](https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.functions_crystallography.html)

Confidence: **medium-high**. The docs are generated and detailed, but some pages identify version
3.3.0 while PyPI/GitHub list 3.4.0, so exact method signatures should be confirmed against an
installed build before writing checks.

---

## Fit for cxr-mc

### Good Fit: Independent Validation

`cxr_mc` already has its own X-ray physics through `xraydb` and internal formulas. A validation
oracle should be independent enough to catch mistakes but close enough in physics scope to compare
numbers. `Dans_Diffraction` fits that role because:

- It has independent CIF parsing and symmetry expansion.
- It computes X-ray reflection intensities from a full unit-cell structure.
- It includes reflection-list and powder-pattern helpers that can stress reciprocal-space
  enumeration.
- It is permissively licensed for installed dependency use.

Best first comparisons:

1. **Geometry parity:** `Cell.lp()`, `Cell.volume()`, reciprocal vectors, and selected `d_hkl`
   or `Qmag` values for simple cells.
2. **Basis parity:** compare `xtl.Structure.get()` against `cxr_mc`'s basis after normalizing
   fractional coordinates modulo 1 and sorting by element/position.
3. **Structure-factor magnitude:** compare `|F_hkl|^2` for small HKL sets. Use magnitude or
   intensity first because phase sign conventions can produce complex conjugates while preserving
   intensity.
4. **Reflection list sanity:** compare allowed/strong reflection sets, not exact ranking on the
   first pass.

### Poor Fit: Production Importer

`diffpy.structure` is a better fit for the production importer because it is a focused structural
container/parser dependency. `Dans_Diffraction` brings a broader application surface:

- GUI entry points and Tkinter/Matplotlib workflows.
- Plotting and powder simulation interfaces.
- Resonant, magnetic, multiple-scattering, diffractometer, and FDMNES-related features.
- Larger API surface with more room for convention mismatch.

That breadth is useful for validation, but excessive for a runtime structural-data adapter.

### Poor Fit: Physics Replacement

Do not move `cxr_mc` physics into `Dans_Diffraction`.

The repo's validation rules require derivations and local ownership of physics behavior. Replacing
internal formulas with a broad external package would make the scientific contract harder to audit.
Using Dans_Diffraction as a comparator preserves independence and keeps physics provenance local.

---

## Risks and Watch Items

1. **Convention mismatch:** `Dans_Diffraction` docs for `Scatter.intensity()` show a negative
   exponential sign in the crystallographic phase, while lower-level `phase_factor` docs show a
   positive sign. `cxr_mc` comparisons should start with `|F|^2` and only compare complex phase
   after a convention audit.
2. **Displacement factors:** `Atoms` includes `uiso`, and scattering intensity includes a
   Debye-Waller term. Initial validation should pin or neutralize displacement factors so a
   mismatch does not get mistaken for a form-factor or phase bug.
3. **Occupancy:** `Atoms` carries occupancy; `cxr_mc` basis entries currently do not appear to
   model partial occupancy. Restrict first checks to full-occupancy structures.
4. **Magnetic/resonant scope:** The PyPI page cautions that magnetic structure support is in
   development. Exclude it from initial validation.
5. **Generated docs may lag:** API docs show some module versions lower than the package version.
   Run a local import/API smoke test before coding checks.
6. **License granularity:** Installed dependency use is fine under Apache metadata, but avoid
   copying source tables or implementation into `cxr_mc` unless every copied file's license is
   verified.

---

## Recommended Validation Spike

Create a dev/checks-only experiment, not a runtime dependency:

1. Add an optional extra or isolated script path for `Dans-Diffraction==3.4.0`.
2. Load the same CIFs through `diffpy.structure`, `Dans_Diffraction`, and `cxr_mc`.
3. Use three fixtures:
   - NaCl or Si: simple, high-confidence baseline.
   - Sapphire/corundum: symmetry-expansion stress case.
   - One TMD already in `crystal_structures.toml`: domain-relevant layered material.
4. Compare:
   - lattice parameters and cell volume,
   - expanded fractional basis,
   - selected `Q`/`d` values,
   - `|F_hkl|^2` for a small fixed HKL set,
   - top-N reflection list agreement under a fixed energy and cutoff.
5. Record any successful comparisons in `docs/physics-validation-ledger.md` only after a fresh
   implementation/review context checks the equations and conventions.

Suggested file location for a future spike: `checks/dans_diffraction_oracle.py` or a narrowly
marked pytest module skipped unless the optional package is installed.

---

## Decision

**Use later as an optional validation oracle. Do not add as a production dependency now.**

`diffpy.structure` should remain the structural-importer path that feeds the existing
`CRYSTALS`-style shape. `Dans_Diffraction` is more valuable as an independent physics-aware
comparison target, especially for structure-factor magnitudes and reflection enumeration.

---

## Sources

- GitHub repository and README:
  <https://github.com/DanPorter/Dans_Diffraction>
- PyPI package metadata and README render:
  <https://pypi.org/project/Dans-Diffraction/>
- Generated package docs:
  <https://danporter.github.io/Dans_Diffraction/>
- Crystal API docs:
  <https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.classes_crystal.html>
- Scattering API docs:
  <https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.classes_scattering.html>
- Lower-level scattering functions:
  <https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.functions_scattering.html>
- Crystallography functions:
  <https://danporter.github.io/Dans_Diffraction/Dans_Diffraction.functions_crystallography.html>
- Zenodo DOI record:
  <https://zenodo.org/records/8106031>
- Local prior context:
  `claudedocs/research_crystals_alternatives_20260702.md`

**Overall confidence:** High that `Dans_Diffraction` is worth using as a validation oracle; medium
on exact API details until confirmed against an installed 3.4.0 build.
