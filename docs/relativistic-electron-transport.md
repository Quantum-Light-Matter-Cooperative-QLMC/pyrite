# Relativistic 1–10 MeV electron transport and quantum channeling

## Summary

This design adds optional Geant4 electromagnetic transport through `g4ppyy`,
native low-energy quantum channeling, and coherent per-electron PXR/CBS
amplitude summation. Current native transport, output fields, and checkpoints
remain the default.

Target capability:

- ordinary electron/photon transport from 0.1 to 10 MeV through Geant4;
- validated quantum planar channeling from 1 to 10 MeV, initially for silicon
  and diamond;
- opt-in experimental axial channeling;
- PXR, CBS, channeling radiation, bremsstrahlung, and atomic-relaxation output
  as separately inspectable components.

## Source audit

### G4ppyy is a binding layer, not a physics model

[`g4ppyy`](https://pypi.org/project/g4ppyy/) is a `cppyy` loader and helper for
a separately installed Geant4. Its current PyPI release is `0.1.0`. It neither
ships Geant4 nor adds channeling physics or channeling databases. The
integration must therefore pin and validate both G4ppyy and an external Geant4
installation.

### Geant4 channeling does not support this regime defensibly

Geant4 11.4.2 contains `G4ChannelingFastSimModel` and `G4BaierKatkov`, but its
production evidence does not extend to sub-100 MeV electrons:

- `G4ChannelingFastSimModel` has a
  [default 200 MeV kinetic-energy threshold](https://github.com/Geant4/geant4/blob/v11.4.2/source/parameterisations/channeling/include/G4ChannelingFastSimModel.hh).
- The official channeling-radiation example is based on validation with
  [855 MeV electrons](https://github.com/Geant4/geant4/blob/v11.4.2/examples/extended/exoticphysics/channeling/ch2/README.md).
- Its radiation macro explicitly warns that
  [quantum particle-dynamics effects may become important below 100 MeV](https://github.com/Geant4/geant4/blob/v11.4.2/examples/extended/exoticphysics/channeling/ch2/special_macros/run_Radiation.mac).
- The Baier–Katkov implementation is quasiclassical. It cannot resolve the
  discrete transverse states responsible for low-energy channeling radiation.

`G4CHANNELING 2.0` contains continuum electric-field, atomic-density,
electron-density, and ionization-threshold splines for diamond, silicon,
germanium, and tungsten. It contains no quantum eigenstates, occupation
probabilities, transition rates, linewidths, or low-energy validation records.

Conclusion: neither G4ppyy, Geant4 channeling classes, nor `G4CHANNELING 2.0`
explicitly supports defensible electron channeling below roughly 50–100 MeV.
Geant4 remains suitable for ordinary electromagnetic transport, while cxr-mc
must own low-energy quantum channeling.

### Low-energy theory and validation anchors

Relevant sources include:

- Andersen, Bonderup, and Pantell,
  [“Channeling Radiation”](https://doi.org/10.1146/annurev.ns.33.120183.002321),
  for quantum planar-channeling formalism;
- Uggerhøj,
  [“The interaction of relativistic particles with strong crystalline fields”](https://doi.org/10.1103/RevModPhys.77.1131),
  for regime boundaries and crystalline-field effects;
- Watson,
  [1–3 MeV silicon and gold measurements](https://doi.org/10.2172/6115542),
  which explicitly report quantized channeling orbits and X-ray transitions;
- Kephart et al.,
  [17 and 54 MeV silicon occupation lengths](https://doi.org/10.1103/PhysRevB.40.4249);
- Gouanère et al.,
  [54–110 MeV planar spectra in diamond and silicon](https://doi.org/10.1103/PhysRevB.38.4352).

## Interfaces and data contracts

### Sweep configuration

Add central fields to `Sweep`:

```python
transport_backend: Literal["native", "geant4"] = "native"
channeling_model: Literal[
    "off", "quantum-planar", "experimental-axial"
] = "off"
channeling_direction: tuple[int, int, int] | None = None
crystal_temperature_K: float = 293.15
```

Validation rules:

- channeling requires an explicit direction and crystalline radiator;
- `quantum-planar` interprets the direction as a plane;
- `experimental-axial` interprets it as an axis and marks results experimental;
- MeV runs using current Browning/Joy transport fail with a validity error
  instead of silently extrapolating;
- current defaults remain bit-for-bit compatible.

### Transport result

Introduce a backend-neutral result carrying current segment fields plus:

- absolute segment-start position and time;
- parent, track, and primary-electron identity;
- exit state and termination reason;
- emitted-photon phase-space records;
- backend version, physics list, datasets, and model-validity metadata.

### Radiation result

Store canonical `radiation_components` arrays:

- `pxr_cbs`;
- `channeling`;
- `bremsstrahlung`;
- `atomic_relaxation`.

Preserve compatibility fields:

```python
spec = radiation_components["pxr_cbs"] + radiation_components["channeling"]
brem = (
    radiation_components["bremsstrahlung"]
    + radiation_components["atomic_relaxation"]
)
```

Old records without components retain their current interpretation. All new
components represent sample-exit yield in `photons/eV/sr/electron`. Photons
already transported through the sample by Geant4 receive no second
Beer–Lambert attenuation.

## Geant4 backend through G4ppyy

1. Add an optional locked environment using `g4ppyy==0.1.0`, compatible
   `cppyy`, and Geant4 11.4.2. Keep default installation Geant4-free.
2. Add startup probe verifying Geant4 version, required datasets, class
   loading, compiled callbacks, and one 3 MeV electron slab event. Missing or
   incompatible installations fail with actionable diagnostics.
3. Use `G4EmStandardPhysics_option4` for ordinary 0.1–10 MeV electromagnetic
   transport.
4. Compile stepping and event collectors through G4ppyy. Avoid Python callback
   execution for every transport step.
5. Export primary trajectories, transmitted/backscattered/stopped counts,
   secondaries, bremsstrahlung photons, and atomic-relaxation photons.
6. Score photons in the configured detector angular bin and normalize by
   primary count, energy-bin width, and solid angle.
7. Never activate `G4ChannelingFastSimModel` or `G4BaierKatkov` below 100 MeV.
   Permit them only in validation tools at 200 MeV or above.

## Coherent relativistic PXR/CBS

Replace segment-intensity addition with complex amplitude addition grouped by
primary electron, reflection, polarization, and photon energy.

For each segment:

- reconstruct segment start from `r_mid`, `v_hat`, and `L_ang`;
- use recorded `t_ang` as segment-start time;
- apply phase `exp[i(omega*t_start - k·r_start)]`;
- apply absorption to field amplitude as `exp(-tau/2)`;
- preserve PXR/CBS amplitude interference;
- square only after summing the complete electron trajectory.

Separate electrons and mosaic crystallites remain incoherent. Artificially
splitting a straight segment must not change the result. Legacy incoherent
summation remains a private validation oracle.

Quantum channeling radiation and PXR/CBS remain separate components. Initial
implementation sums their photon yields and makes no claim of cross-process
amplitude interference.

## Native quantum planar channeling

1. Build planar continuum potential from catalog CIF structure, reciprocal
   vectors, source-backed electron scattering factors, thermal vibration
   amplitude, and Debye–Waller data. Do not treat `G4CHANNELING` splines as
   quantum truth.
2. Solve the relativistic transverse Schrödinger equation for bound and
   continuum states using converged sparse finite-difference or plane-wave
   diagonalization.
3. Calculate incident-state occupation from beam angle, divergence,
   impact-parameter averaging, and crystal orientation.
4. Propagate state populations through depth using scattering-induced
   transition rates. Track dechanneling into continuum states.
5. Calculate dipole transition energies, angular spectra, polarization,
   natural and occupation broadening, recoil, and photon attenuation.
6. Route initially unchanneled and dechanneled phase-space electrons into
   Geant4 for remaining bulk transport. Native bound-state propagation owns
   the channeling path, preventing amorphous-scattering and bremsstrahlung
   double counting.
7. Add source-backed material profiles recording temperature, RMS thermal
   displacement, potential model, supported planes, and provenance.
8. Mark silicon and diamond planar profiles validated only after physics
   anchors pass. Other catalog crystals remain experimental until independently
   validated.

## Experimental axial channeling

- Add opt-in 2D transverse Hamiltonian using axial periodic potential and
  sparse eigenstates.
- Reuse population propagation and radiation machinery with 2D transition
  matrix elements.
- Initially support silicon axes covered by the 1–3 MeV observations.
- Store explicit `experimental` validity metadata in checkpoints and plots.
- Require numerical convergence and qualitative peak-order checks, but make no
  quantitative intensity claim until an independent axial dataset is
  reproduced.

## Validation and tests

Every new physics equation requires a derivation docstring, `Validation: <id>`
marker, ledger row, derivation note, units, assumptions, and limiting case. A
fresh context performs source-to-code review; only a human may mark a claim
`signed-off`.

### Fast regression tests

- backend selection, missing-install diagnostics, case-name separation,
  serialization, and old-checkpoint compatibility;
- segment identity and monotonic time ordering;
- one-segment coherent result equals the analytic sinc result;
- straight-segment subdivision invariance;
- randomized phases approach the legacy incoherent sum;
- zero potential yields no bound states or channeling radiation;
- eigenstate normalization, orthogonality, parity, grid convergence, and
  dipole selection rules;
- conservation across bound, continuum, stopped, and exited populations;
- deterministic adapter normalization using fixture photon/trajectory data.

### Geant4 integration checks

- compare ESTAR collision/radiative stopping powers and CSDA ranges for carbon
  and silicon at 0.5, 1, 3, 5, and 10 MeV;
- compare transmitted, backscattered, and stopped fractions with published
  thin-target benchmarks;
- compare bremsstrahlung yield and endpoint with Seltzer–Berger/Geant4
  references;
- enforce electron/photon energy conservation including secondaries.

### Physics anchors

- reproduce 1–3 MeV silicon planar and axial peak energies from Watson;
- reproduce 17 MeV silicon occupation lengths;
- reproduce 54–110 MeV silicon/diamond planar spectra as continuity check;
- compare 200 MeV and higher planar output with Geant4/Baier–Katkov only as a
  regime-overlap oracle.

Acceptance criteria:

- transport quantities agree with reference uncertainty or 5%, whichever is
  larger;
- planar peak energies agree within experimental calibration or bin
  uncertainty;
- intensity comparisons include statistical and digitization uncertainty;
- no `validated` status until independent derivation and published-spectrum
  anchor both pass.

## Validity map and rollout

| Capability | Intended regime | Status |
| --- | --- | --- |
| Current native transport | Existing keV regime | Preserved |
| Geant4 ordinary EM transport | 0.1–10 MeV | Production after anchors |
| Quantum planar channeling | 1–10 MeV, initially Si/Diamond | Validated per profile |
| Quantum axial channeling | Initially Si | Experimental |
| Geant4 channeling/Baier–Katkov | 200 MeV and above | Validation oracle only |

Run focused CPU tests first, optional Geant4 integration suite in a pinned
container second, then canonical repository verification. Energies below
1 MeV require separate channeling validation before any support claim.
