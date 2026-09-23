# Relativistic PXR/CBS and polarization: implementation assessment

**Status:** Research assessment, 12 September 2026; code inspected at `a017a50f`. Proposed capabilities below are not implemented or independently validated by this assessment. No existing validation status is advanced.

## Recommendation

Extend the existing vector radiation kernel rather than introducing relativity or polarization as if both were absent. First expose angular and polarization observables and establish the kinematic model's validity envelope. Then add dynamical diffraction for geometries where the first-order field is inadequate. Treat physically normalized [superradiant PXR](superradiant-pxr.md) as a separate extension.

The deciding variables are electron energy, photon energy, diffraction detuning, crystal thickness and orientation, formation length, and detector acceptance. There is no single electron-energy threshold at which every calculation needs the same replacement model.

## Source audit

The assessment used the full text of {cite:t}`feranchuk2000` and the main paper and supplement of {cite:t}`zhai2025`, retrieved from the local Zotero library. The source records are `VE2PXG8N` and `6M7RCW9K`; the inspected attachments are `6GZCPKZQ`, `NWBR3CF2`, and `IJ2F7V4J`. These are Zotero provenance identifiers, not bibliography keys or portable attachment links. PDFs are not vendored in the repository. Equation numbers below were read from those source texts, rather than copied from the repository ledger. Indexed mathematical text can lose typography; exact normalization still requires equation-level verification against the typeset papers.

Feranchuk's Eqs. (13)–(14) retain vector PXR and CBS amplitudes. The later discussion preceding Eq. (16) introduces the nearly isotropic approximation when simplifying their relative contributions. Production retains vector amplitudes, not that scalar simplification. The printed nonrelativistic CBS Eq. (14) does not contain the full relativistic structure now used by PyRITE.

Zhai's supplement explicitly includes relativistic corrections. Its Eqs. (6)–(7) contain the Lorentz factor, polarization vectors, a velocity-aligned relativistic contraction, and both finite-time branches, $Q$ and $Q'$. PyRITE instead organizes the calculation by signed reciprocal vectors and retained positive resonances. Establishing equivalence requires tracking potential Fourier conventions, branch counting, relative signs, and discarded off-resonant terms. This assessment does not certify literal source transcription.

The two-wave diffraction treatment in {cite:t}`feranchuk2022`, Eqs. (37)–(39), provides a starting point for the next model tier. Its boundary-value problem is more extensive than adding a relativistic prefactor to a kinematic amplitude.

## Current implementation and actual gaps

The maintained descriptions of the current model remain [coherent radiation](../../physics/radiation-physics/coherent-radiation.md), [coherent tracking](../../physics/radiation-physics/coherent-emission.md), and [photon propagation](../../physics/radiation-physics/photon-escape-and-dispersion.md). The following is an implementation snapshot supporting the proposed extension.

| Aspect | Evidence at the inspected revision | Remaining work |
| --- | --- | --- |
| Relativistic velocity | `transport/kinematics.py::_beta_array` uses kinetic energy and electron rest energy | Validate high-energy numerics and transport inputs |
| CBS amplitude | `_per_hkl.py` includes a single inverse Lorentz factor and velocity-projector contractions | Verify source conventions and the domain of the perturbation |
| Polarization | `_kernels.py::_polarization_pair` builds two transverse vectors; amplitudes are evaluated for both | Preserve polarization observables in output |
| PXR/CBS interference | Complex amplitudes are added before squaring within each polarization | Carry the same convention into any replacement diffraction solver |
| Direction dependence | Resonance, amplitudes and escape depend on arbitrary `n_hat` | Expose detector azimuth and roll through the standard scene |
| Detector integration | `mc_spectrum_solid_angle` and `run_case_directions` reuse transported trajectories | Integrate acceptance consistently into results, units and response |
| Photon field | First-order diffraction plus bulk dispersion and passive absorption | Coupled modes, extinction and boundary conditions where required |

Source owners: [line kernels](../../../src/pyrite/montecarlo/spectrum/lines/_kernels.py), [per-reflection amplitudes](../../../src/pyrite/montecarlo/spectrum/lines/_per_hkl.py), [batched amplitudes](../../../src/pyrite/montecarlo/spectrum/lines/_batched.py), [geometry](../../../src/pyrite/montecarlo/geometry.py), and [detector specification](../../../src/pyrite/detectors/spec.py).

### Polarization summation is not an isotropy assumption

Adding intensities of orthogonal polarizations is appropriate for a detector that does not analyze polarization. It does not assert equal polarization populations or azimuthally constant intensity. The current output discards information needed to predict an analyzer measurement.

The standard geometry places the detector at lab azimuth zero. This is a coordinate convention with a restricted interface, not an average over detector azimuth. For a fixed crystal and beam, changing detector azimuth generally changes the vector products. Rotating all geometry together should leave total intensity unchanged. Special axial symmetries can also give azimuthal invariance; such a result alone would not demonstrate a missing polarization term.

The flat-solid-angle approximation concerns variation *across a detector face*. It is distinct from global angular isotropy. A small detector can still resolve substantial variation across a narrow relativistic lobe. See [detector solid-angle integration](../../physics/detectors/detector-solid-angle.md).

### Documentation discrepancies

At the inspected revision, the package-level `montecarlo/__init__.py` description says there are no gamma corrections; production code contradicts this statement. The public line-spectrum docstring also labels the calculation nonrelativistic. These descriptions should be reconciled with a source-verified model statement. Their age is not evidence that the corresponding production terms are absent.

## Proposed extension sequence

### 1. Angular and polarization output

Add detector azimuth and roll to the public geometry, case serialization and checkpoint identity. Reuse arbitrary-direction evaluation rather than repeating electron transport for each viewing direction.

Retain either complex transverse fields or the two-by-two polarization coherency matrix. Define a common detector basis and rotate each reflection's local basis into it before accumulating polarization observables. Stokes parameters and analyzer responses can then be derived with a documented handedness convention. Scalar intensity must recover existing output.

For detector-face integration, distinguish per-steradian spectra from acceptance-integrated spectra. Remove the flat solid-angle multiplier and analytic aperture broadening only for components already integrated over the face; retain instrumental energy redistribution. Choose direction grids by angular convergence, including both polar and azimuthal variation.

This is a moderate API, result-schema and kernel-output extension. Full Stokes support is broader than simply returning two polarization intensities because it also needs their complex cross-correlation.

### 2. Validated relativistic kinematic radiation

Derive the emitted field and CBS trajectory perturbation in one common convention. Check the source branch bookkeeping, normalization, finite-time phase and dielectric factors, including the differences already discussed in the [CBS validation record](../../validation/radiation-physics/cbs-amplitude.md). Existing ledger coverage of `chi_g` is not by itself a validation of every PXR angular numerator or the complete spectral normalization.

Establish when amplitudes can be frozen at resonance across a finite line and when evaluating the full frequency-dependent field is necessary. Bound the CBS perturbation where the velocity–reciprocal-vector projection is small; an accepted resonance root does not alone establish perturbative validity.

Test coherence across physical flights. The default independent-flight sum needs a random-phase justification; an elastic collision does not automatically destroy phase. Prefer the existing per-electron field path when the formation length spans multiple flights, subject to its own phase and convergence checks.

This stage is primarily derivation, reference calculations and validation. Thin or sufficiently off-Bragg configurations can remain kinematic even for relativistic electrons; that domain must be demonstrated rather than assumed.

### 3. Dynamical diffraction

Introduce a coupled-mode field solver using the complex mean susceptibility and opposite reciprocal harmonics, polarization coupling and crystal–vacuum boundary conditions. Use the two-wave construction of {cite:t}`feranchuk2022` as an initial reference, with its assumptions stated explicitly. Multiple simultaneously coupled reflections require a later extension.

Integrate the electron current against the reciprocal outgoing-field solution. The thin, weak-coupling or off-Bragg limit should recover the validated kinematic model. Include absorption consistently within the complex modes; do not also apply the old escape attenuation to the same propagation without a derivation.

CBS must use the same outgoing-field convention. If periodic electron deflection is explicitly included in a trajectory, do not also add its radiation through the existing perturbative CBS term. Interface radiation likewise needs an include-or-bound decision; see the [transition-radiation study](transition-radiation-recommendations.md).

This is a major new radiation component. Start with prescribed straight tracks and a perfect single crystal before coupling it to stochastic transport, mosaicity, layered targets or device kernels.

### 4. High-energy transport and precision

The inspected scattering data and documented transport checks target energies through roughly 300 keV. Extending radiation formulas does not extend those data. Higher-energy work requires suitable scattering, collision stopping and radiative-loss treatment, or externally supplied trajectories. Coordinate this with the [relativistic transport proposal](relativistic-electron-transport.md).

Compute the Lorentz factor directly from energy rather than reconstructing it from a rounded speed. `_setup.py` currently performs the latter operation and the GPU backend defaults to float32. Near light speed this loses the small difference from unity. Small angular denominators and long-path phase accumulation also need stable evaluation and precision checks; enabling float64 alone is not a convergence argument.

## Acceptance evidence and limits

The first implementation should establish:

- scalar intensity recovery after summing polarization channels, transverse basis-rotation invariance, and known analyzer limits;
- covariance under a common rotation of beam, crystal and detector, plus detector-azimuth dependence in deliberately nonsymmetric geometries;
- low-energy source benchmarks and relativistic spectral-angular benchmarks, including PXR/CBS interference rather than isolated component magnitudes;
- convergence in energy, detector angle, trajectory subdivision and precision;
- weak-coupling recovery of the kinematic model and finite, boundary-consistent dynamical results near diffraction peaks;
- parity between the reference route and each accelerated implementation.

During the assessment, the existing `test_solid_angle.py`, `test_tilt_convention.py` and `test_coherent_emission.py` suites passed (41 tests). They exercise existing geometry and coherence contracts; they do not establish MeV/GeV accuracy, polarization-resolved output or dynamical diffraction. New equations require derivation records, ledger entries and fresh-context validation before scientific acceptance. Only a human may mark claims signed off.
