# Relativistic 1–10 MeV electron transport and quantum channeling

**Status:** Partly superseded (2026-10-14). This page retains the source audit, the low-energy anchors and the validation targets. It is not the current plan for transport or channeling.

| Original component | Current disposition |
| --- | --- |
| Geant4 electromagnetic transport through `g4ppyy` | **Not pursued.** Native transport is extended instead: 30–100 MeV in [#439](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/439), above 100 MeV in [#440](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/440). Geant4 remains an external validation reference only. |
| Native low-energy quantum channeling | Redesigned in [ADR-0018](../../adr/0018-quantum-channeling-zone.md) (Proposed) and the [channeling radiation walkthrough](channeling-radiation-physics.md): a quantum entrance zone with Lindblad evolution and one-way handoff to *native* transport. Tracked in [#56](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/56). |
| Coherent per-electron PXR/CBS amplitude summation | Implemented as the `coherent` emission policy; see [coherent emission](../../physics/radiation-physics/coherent-emission.md). The section below is the original design rationale. |
| `radiation_components` output schema | Replaced by the existing slots (`spec`, `spec_coherent`, `spec_characteristic`, continuum) plus a future additive `spec_channeling` (ADR-0018). |

## Summary

The original design proposed three things: optional Geant4 electromagnetic transport through `g4ppyy`, native low-energy quantum channeling, and coherent per-electron PXR/CBS amplitude summation. Current native transport, output fields and checkpoints remain the default. The table above records which parts survive.

Original target capability (historical):

- ordinary electron/photon transport from 0.1 to 10 MeV through Geant4 (*not pursued*);
- validated quantum planar channeling from 1 to 10 MeV, initially for silicon and diamond (*now 1–100 MeV under ADR-0018*);
- opt-in experimental axial channeling;
- PXR, CBS, channeling radiation, bremsstrahlung and atomic-relaxation output as separately inspectable components.

## Source audit

### G4ppyy is a binding layer, not a physics model

[`g4ppyy`](https://pypi.org/project/g4ppyy/) is a `cppyy` loader and helper for a separately installed Geant4. Its current PyPI release is `0.1.0`. It neither ships Geant4 nor adds channeling physics or channeling databases. An integration would therefore have had to pin and validate both G4ppyy and an external Geant4 installation.

### Geant4 channeling does not support this regime defensibly

Geant4 11.4.2 contains `G4ChannelingFastSimModel` and `G4BaierKatkov`, but its production evidence does not extend to sub-100 MeV electrons:

- `G4ChannelingFastSimModel` has a [default 200 MeV kinetic-energy threshold](https://github.com/Geant4/geant4/blob/v11.4.2/source/parameterisations/channeling/include/G4ChannelingFastSimModel.hh).
- The official channeling-radiation example is based on validation with [855 MeV electrons](https://github.com/Geant4/geant4/blob/v11.4.2/examples/extended/exoticphysics/channeling/ch2/README.md).
- Its radiation macro explicitly warns that [quantum particle-dynamics effects may become important below 100 MeV](https://github.com/Geant4/geant4/blob/v11.4.2/examples/extended/exoticphysics/channeling/ch2/special_macros/run_Radiation.mac).
- The Baier–Katkov implementation is quasiclassical. It cannot resolve the discrete transverse states responsible for low-energy channeling radiation.

`G4CHANNELING 2.0` contains continuum electric-field, atomic-density, electron-density, and ionization-threshold splines for diamond, silicon, germanium, and tungsten. It contains no quantum eigenstates, occupation probabilities, transition rates, linewidths, or low-energy validation records.

Conclusion: neither G4ppyy, Geant4 channeling classes, nor `G4CHANNELING 2.0` explicitly supports defensible electron channeling below roughly 50–100 MeV. PyRITE must own low-energy quantum channeling. Geant4 is suitable for ordinary electromagnetic transport, but the project extends native transport instead; Geant4 serves only as an external validation reference, as in the existing TestEm5 transport benchmarks.

### Low-energy theory and validation anchors

Relevant sources include:

- Andersen, Bonderup, and Pantell, [“Channeling Radiation”](https://doi.org/10.1146/annurev.ns.33.120183.002321), for quantum planar-channeling formalism;
- Uggerhøj, [“The interaction of relativistic particles with strong crystalline fields”](https://doi.org/10.1103/RevModPhys.77.1131), for regime boundaries and crystalline-field effects;
- Watson, [1–3 MeV silicon and gold measurements](https://doi.org/10.2172/6115542), which explicitly report quantized channeling orbits and X-ray transitions;
- Kephart et al., [17 and 54 MeV silicon occupation lengths](https://doi.org/10.1103/PhysRevB.40.4249);
- Gouanère et al., [54–110 MeV planar spectra in diamond and silicon](https://doi.org/10.1103/PhysRevB.38.4352).

## Interfaces and data contracts

Superseded by [ADR-0018](../../adr/0018-quantum-channeling-zone.md):

- **Selection:** `Physics.channeling_model` (default `"off"`), with orientation taken from the independent face/lattice orientation of [#25](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/25). There is no `transport_backend` field.
- **Transport records:** the existing native segment and event records, plus an `EVENT_DECHANNEL` code at the handoff.
- **Radiation output:** the existing output slots, plus `spec_channeling` once anchored. Characteristic emission and the continuum keep their current owners.

The remaining validity rules carry over: channeling requires a crystalline radiator; planar runs refuse or warn near low-index axes; the axial tier is marked experimental; current defaults stay bit-for-bit compatible.

## Geant4 backend through G4ppyy

Not pursued (decision 2026-10-14, recorded in ADR-0018). Higher-energy native transport is tracked in #439 and #440. The audit above remains the reason Geant4 channeling classes are excluded below 100 MeV; quasiclassical Geant4 channeling at ≥200 MeV is admissible only as an external validation oracle.

## Coherent relativistic PXR/CBS

Replace segment-intensity addition with complex amplitude addition grouped by primary electron, reflection, polarization, and photon energy.

For each segment:

- use stored `r_mid` as the representative position;
- preserve recorded `t_ang` as segment-start time and derive the matching midpoint time `t_mid = t_ang + L_ang/(2 beta)`;
- apply the centered-segment phase `exp[i(omega*t_mid - k·r_mid)]` with the centered finite-time integral;
- apply absorption to field amplitude as `exp(-tau/2)`;
- preserve PXR/CBS amplitude interference;
- square only after summing the complete electron trajectory.

Separate electrons and mosaic crystallites remain incoherent. Artificially splitting a straight segment must not change the result. Legacy incoherent summation remains a private validation oracle.

Quantum channeling radiation and PXR/CBS remain separate components. Initial implementation sums their photon yields and makes no claim of cross-process amplitude interference.

## Native quantum planar channeling

Superseded in detail by the [channeling radiation walkthrough](channeling-radiation-physics.md) and ADR-0018. In outline:

1. Build the planar continuum potential from the catalog CIF structure, reciprocal vectors, Mott–Bethe electron scattering factors, and per-site, temperature-dependent thermal displacements. Do not treat `G4CHANNELING` splines as quantum truth.
2. Solve the relativistic transverse Schrödinger equation for bound and continuum states.
3. Build the incident transverse state from beam angle, divergence and crystal orientation.
4. Evolve the transverse state through depth with Lindblad jump operators (thermal diffuse and electronic scattering): a deterministic density matrix for planar channeling, MCWF for axial/keV. Coherences are retained, and the dechanneling flux is recorded.
5. Calculate dipole transition energies, angular spectra, polarization, occupation broadening, recoil and photon attenuation. v1 emits bound→bound lines only.
6. Hand dechanneled electrons, sampled from the flux, one way to the native transport cores. Native multiple scattering is off inside the zone, preventing double counting.
7. Add source-backed material profiles recording temperature, per-site thermal displacement, potential model, supported planes and provenance.
8. Mark silicon and diamond planar profiles validated only after physics anchors pass. Other catalog crystals remain ineligible or experimental until independently validated.

## Experimental axial channeling

- Add opt-in 2D transverse Hamiltonian using axial periodic potential and sparse eigenstates.
- Reuse population propagation and radiation machinery with 2D transition matrix elements.
- Initially support silicon axes covered by the 1–3 MeV observations.
- Store explicit `experimental` validity metadata in checkpoints and plots.
- Require numerical convergence and qualitative peak-order checks, but make no quantitative intensity claim until an independent axial dataset is reproduced.

## Validation and tests

Every new physics equation requires a derivation docstring, `Validation: <id>` marker, ledger row, derivation note, units, assumptions, and limiting case. A fresh context performs source-to-code review; only a human may mark a claim `signed-off`.

### Fast regression tests

- model selection, refusals, case-identity separation, serialization, and old-checkpoint compatibility;
- segment identity and monotonic time ordering;
- one-segment coherent result equals the analytic sinc result;
- straight-segment subdivision invariance;
- randomized phases approach the legacy incoherent sum;
- zero potential yields no bound states or channeling radiation;
- eigenstate normalization, orthogonality, parity, grid convergence, and dipole selection rules;
- conservation across bound, handed-off, stopped and exited populations (trace + dechanneling flux = 1);
- density-matrix/MCWF ensemble equivalence and the rate-equation limit;
- deterministic adapter normalization using fixture photon/trajectory data.

### Native-transport checks at MeV energies

Owned by #439/#440: stopping powers and CSDA ranges against ESTAR, transmitted/backscattered/stopped fractions against thin-target benchmarks (including Geant4 TestEm5 as an external reference), bremsstrahlung yield and endpoint, and energy conservation including secondaries.

### Physics anchors

- reproduce 1–3 MeV silicon planar and axial peak energies from Watson;
- reproduce 17 MeV silicon occupation lengths;
- reproduce 54–110 MeV silicon/diamond planar spectra as continuity check;
- compare 200 MeV and higher planar output with Geant4/Baier–Katkov only as a regime-overlap oracle;
- compare depth-resolved populations with frozen-phonon multislice (abTEM) as an optional external oracle.

Acceptance criteria:

- transport quantities agree with reference uncertainty or 5%, whichever is larger;
- planar peak energies agree within experimental calibration or bin uncertainty;
- intensity comparisons include statistical and digitization uncertainty;
- no `validated` status until independent derivation and published-spectrum anchor both pass.

## Validity map and rollout

| Capability | Intended regime | Status |
| --- | --- | --- |
| Current native transport | keV–MeV; extensions in #439/#440 | Preserved |
| Geant4 ordinary EM transport | — | Not pursued; external validation reference only |
| Quantum planar channeling | 1–100 MeV, initially Si/diamond | Validated per profile (ADR-0018) |
| Quantum axial channeling, including vortex beams | Initially Si; keV TEM vortex probes | Experimental |
| Geant4 channeling/Baier–Katkov | 200 MeV and above | Validation oracle only |

Run focused CPU tests first, then canonical repository verification. Energies below 1 MeV require separate channeling validation (the keV Bloch-wave tier) before any support claim.
