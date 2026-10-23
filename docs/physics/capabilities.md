# Capabilities and maturity

This inventory describes available paths and their limits, using the maintained
model references below. **Available does not mean validated:** the generated
[validation summary](../validation/status-summary.md) reports evidence and human
sign-off separately; the [domain inventories](../validation/domain-inventories.md)
link each claim to its authoritative ledger record.

| Model or effect | Availability | Scope and limits | Model and evidence |
|---|---|---|---|
| Elastic electron scattering and collision stopping | Default | Independent electron flights; ELSEPA elastic tables and SBETHE material stopping. Upstream beamline transport and space charge are excluded. | [Electron transport](beam-transport/electron-transport.md); [transport evidence](../validation/ledger-transport-background.md) |
| Shell soft/hard inelastic events | Conditional default | Automatic selection requires covered shell data for every layer and the supported transport configuration. Other cases use continuous loss; missing required reference tables raise. | [Selection and supported paths](beam-transport/shell-soft-hard-transport.md#configuration); [shell evidence](../validation/beam-transport/shell-soft-hard-transport.md) |
| Secondary electrons | Opt-in | Shell-collision secondaries use generation-based transport with explicit tracking thresholds and resource limits. Coherent spectra with secondary transport are refused. | [Secondary electron transport](beam-transport/shell-soft-hard-transport.md#secondary-electron-transport); [secondary evidence](../validation/beam-transport/shell-secondary-transport.md) |
| Pair conversion and positron transport | Opt-in | Coupled hard-photon cascades can create pairs; positron transport requires its own tables and explicit selection. This is not general photon multiple-scattering transport. | [Pair conversion](radiation-physics/hard-bremsstrahlung-events.md#pair-conversion-of-hard-photons), [positrons](beam-transport/positron-transport.md); [transport ledger](../validation/ledger-transport-background.md) |
| PXR and CBS lines | Default | First-order kinematic emission. Physical flights and electrons normally add as intensities; dynamical diffraction is excluded. | [Coherent radiation](radiation-physics/coherent-radiation.md); [coherent-physics evidence](../validation/ledger-core-coherent-physics.md) |
| Trajectory and bunch coherence | Opt-in | Phased tracking preserves trajectory phase with physical bunch-population weighting and geometry-dependent offset averaging; consumer restrictions apply. | [Coherent-emission tracking](radiation-physics/coherent-emission.md); [coherence evidence](../validation/radiation-physics/coherent-emission.md) |
| Bremsstrahlung and characteristic radiation | Default, with data-dependent backend | BremsLib is preferred when installed; otherwise bremsstrahlung warns and falls back to EEDL. Characteristic lines are a separate incoherent component. | [Bremsstrahlung](radiation-physics/bremsstrahlung.md), [characteristic radiation](radiation-physics/characteristic-radiation.md); [radiation evidence](../validation/ledger-transport-background.md) |
| Planar quantum channeling | Opt-in, bounded run integration | Fixed-count runs, native continuation, persisted additive spectra and line recompute are supported. The zone solve and radiation source run on CPU; native continuation can use CUDA. Adaptive precision, capture/block transport and several combined physics/geometry modes are refused. Consult the current run-surface restrictions before enabling it. | [Architecture and implemented outcomes](../adr/0018-quantum-channeling-zone.md#run-surfaces-521); [channeling evidence](../validation/ledger-channeling.md) |
| Axial channeling and vortex inputs | Experimental diagnostic | Bounded diagnostic solvers and app views; no standard run integration. Experimental checks do not establish general predictive accuracy. | [Axial evidence](../validation/channeling/cr-axial-2d-eigenproblem.md), [real-space propagation](../validation/channeling/cr-axial-realspace-propagation.md); [diagnostic workflow](../guides/axial-diagnostics.md) |
| Photon escape and detector response | Default source escape; downstream response | Straight-ray attenuation and passive bulk dispersion. Detector acceptance, efficiency and measured-energy redistribution follow source scoring; general photon scattering, interface optics and feedback on electrons are excluded from this escape model. | [Photon escape](radiation-physics/photon-escape-and-dispersion.md), [detector response](detectors/detector-response.md); [detector evidence](../validation/ledger-detector-forward-models.md) |
| Transition radiation | Opt-in planar-stack TR and CTR; off by default | Flat interfaces, dielectric-band and coherence limits apply; default omission bounds are observable-dependent. | [Model and limits](radiation-physics/transition-radiation.md); [TR/CTR demo](../guides/transition-radiation-demo.md); [detected omission bound](../validation/radiation-physics/tr-detected-omission-bound.md) |

## Beam-energy validity

Electron (and positron) beams outside channeling. "Benchmarked" names the
external comparison behind the tier; the ledger rows carry the evidence and
none is human signed-off.

| Beam energy | Physics state | Benchmarked against | Known limits |
|---|---|---|---|
| ≤ 1 MeV | ELSEPA elastic, SBETHE stopping, shell soft/hard events, coupled BremsLib radiative loss | Geant4 TestEm5 full track, W/Si at 300 and 800 keV; ESTAR stopping | EUV/soft X-ray transition radiation not omissible ([TR decisions](radiation-physics/transition-radiation.md#decisions)) |
| 1–30 MeV | As above on BremsLib's complete grid, with the Seltzer–Berger e–e term and Ter-Mikaelian suppression | TestEm5 W 100 µm / Si 1 mm at 3, 10, 30 MeV (`mev-full-track-transport`); ESTAR collision/radiative stopping and CSDA range; Seltzer–Berger spectra (`brem-source-comparison`); EPDL pair totals | Forward soft X-rays need the opt-in TR component; photons are attenuated, not transported (thick-target warning above 0.05 X₀, #341) |
| 30–100 MeV | BremsLib extension rows (library SDCS to `k/T` = 0.95; tip and `k > 0` angles modelled, `bremslib-high-energy-extension`); uncoupled radiative loss refused unless `radiative_model="uncoupled"` | TestEm5 at 50 and 100 MeV (W θ_rms excepted: Geant4's point-nucleus Mott correction); ESTAR to 100 MeV; Seltzer–Berger reported, not gated | Filled tip `k/T > 0.95` 0.6–1.9× Seltzer–Berger at 100 MeV; inner-shell ionization may double-count the density effect above ~10 MeV ([closure note](../validation/beam-transport/penelope-shell-rate-closure.md)); TR dominates forward keV continua below 25–30 keV |
| > 100 MeV | No coupled radiative tables; LPM not modelled | — | #440 |

Channeling ([#56](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/56)) and coherent PXR/CBS line couplings above ~1 MeV photon energy (#249) are tracked separately.

Research designs may describe later tiers alongside implemented outcomes. Use
the linked run-surface restrictions for execution support, and the ledger for
the evidence behind a numerical claim. No maturity label in this table promotes
a ledger status or replaces formal human sign-off.
