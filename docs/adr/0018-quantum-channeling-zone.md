# 0018 — Model electron channeling as a quantum entrance zone

- **Status:** Accepted — 2026-11-10 (planar implemented and persisted in #503–#507 and #521; first axial cases admitted in #994)
- **Date:** 2026-10-14
- **Issue:** [#502](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/502) (parent [#56](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/56))
- **Design:** [Channeling radiation physics walkthrough](../research/physics/channeling-radiation-physics.md)

## Context

Native transport treats every medium as amorphous. It scatters electrons from
random-atom ELSEPA/Mott tables, and the crystal lattice enters only through the
PXR/CBS emission kernels. Electrons aligned with a crystal plane or axis instead
occupy discrete transverse states in the lattice's continuum potential. Below
roughly 100 MeV those states are few: Si(110) at room temperature has 1 bound
state at 1 MeV, 3 at 10 MeV and 9 at 100 MeV. Such states need a quantum treatment. Quasiclassical models
(Baier–Katkov, Geant4 `G4ChannelingFastSimModel`) are not defensible in that
regime, and the documented Geant4 production threshold is 200 MeV.

The earlier research design routed ordinary 0.1–10 MeV transport and all
dechanneled electrons through Geant4 via `g4ppyy`. That route was never
implemented. Native transport now reaches the 100 MeV ELSEPA limit, and
[#439](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/439)
and [#440](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/440)
extend it further.

Under [ADR-0017](0017-in-tree-physics-extension-contract.md), channeling is a
tier-3 change: a new native event plus an additive spectral channel.

## Decision

### Scope and selection

Channeling is opt-in through `Physics.channeling_model`, which defaults to
`"off"`. Off means no case key, no identity marker, and bit-for-bit unchanged
results. The first validated tier is planar channeling of electrons at
1–100 MeV in silicon and diamond. Axial channeling (including vortex/OAM beam
inputs), the keV Bloch-wave regime and CUDA execution are later tiers.

Crystal orientation comes from the independent face/lattice orientation of
[#25](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/25).
Channeling adds no second orientation convention.

### Quantum entrance zone with one-way handoff

The channeling module owns each aligned electron from crystal entry until it
dechannels:

- the transverse state evolves in the continuum potential under an effective
  mass $\gamma m$;
- thermal diffuse and electronic scattering enter as Lindblad jump operators,
  the Einstein-model Markov limit of the frozen-phonon/QEP description;
- native multiple scattering is off inside the zone, because the jump
  operators *are* the multiple scattering there.

After each jump, the state is projected onto the bound subspace (bound states
plus a declared near-barrier margin) and the free subspace. The free part leaves
the zone. A dechanneled electron becomes an ordinary launch state that the
existing native transport cores take over through the secondaries launch seam,
under a new `EVENT_DECHANNEL` event code. Rechanneling is omitted and recorded
as a validity limit.
*Outcome (#506, maintainer decisions 2026-10-16):* each electron's fate
(entry, free, overflow, bound) and handoff depth are sampled from one
density-matrix solve per case with keyed per-electron streams. Free `E_⊥` is
sampled separably from depth, the plane-angle sign is symmetric, and
`ψ = √(2E_⊥/pv)` (from `U₀ = 0`) tilts the electron's own in-plane direction
toward the plane normal. Overflow leaves at the declared band-edge angle,
with a warning above 10%. Stopping is constant at `S(E₀)` in the solve and the
zone row, and zone caps shorter than the path are refused. The primaries'
entrance states come from the same seed-pure draws as amorphous transport and
relaunch with their own stream keys. Mosaic crystals, coupled radiative
transport, secondaries, coherent emission, stacks and grooves are refused.
CUDA native continuation is described under Run surfaces below. The 300 eV margin sets the dechanneled angular distribution (mean
4.7 `ψ_c` at 5 MeV Si(110)); see `cr-zone-handoff-accounting`.

*Scope correction (#522, 2026-10-20):* the original description of the
overflow angle as a lower bound is withdrawn. The legacy residual includes
omitted transfer-kernel weight that can land inside the particle band, so
the band-edge angle is an assignment, not a bound for that entire channel.
The independent [handoff review](../validation/channeling/cr-zone-handoff-accounting.md#scope-correction-the-legacy-overflow-angle-is-an-assignment)
records the counterexample. The [boundary investigation](../research/physics/dechanneling-boundary.md#recommendation-after-boundary-and-tail-prototypes)
retains the 300 eV model after negative boundary and tail-window tests;
the assignment and production output are unchanged.

### Evolution backends

Both backends share one operator layer and must agree in the ensemble mean.

| Use | Backend |
| --- | --- |
| Planar (production) | Deterministic density matrix, integrated split-operator style: a coherent step in the transverse eigenbasis, then an elementwise dissipator on $\rho(x,x')$. It outputs the dechanneling flux $dP/dz\,dE_\perp$. Per-electron handoff states are sampled from that flux with keyed RNG streams. |
| Axial, keV, two-electron | Monte Carlo wave-function (MCWF) unravelling on a real-space grid, with the same collapse rule. |

Neither backend uses an absorptive-only (imaginary-potential) model. Such
models delete scattered electrons instead of redistributing them among states
or letting them rechannel. Frozen-phonon multislice (abTEM) is an optional
external validation oracle, not production code.

### Emission

Channeling radiation (CR) is scored with an expected-value estimator,
$\langle\hat d_\omega^\dagger\hat d_\omega\rangle$ per transition-frequency
cluster. The estimator is built on a generic first-order transition operator,
whose long-wavelength bound→bound limit is the dipole rate.

- v1 scores bound→bound lines only.
- Free→bound radiative capture is estimated as an upper bound first, and
  scored only if that bound exceeds a few percent of the line yield.
  *Outcome (#505, maintainer decision 2026-10-15):* the estimate is about 40%
  of the line yield at Kephart's Si(110) 17 MeV, 2.2 mrad configuration, so
  capture from the tracked above-barrier states is scored, reported as a
  separate spectral component (`cr-capture-yield`).
- Free→free transitions are coherent bremsstrahlung (CBS), which the PXR/CBS
  kernel already produces, so they are never scored here.
  *Outcome (#1055, maintainer decision 2026-11-10):* superseded for the zone.
  The native kernel scored zone rows as free plane waves at the beam angle and
  counted the zone twice: 5.4× the in-cut free→free block for Si(110) at
  0.8 mrad, 17 MeV, more than the whole CR yield. Native PXR/CBS now drops the
  plane's own reciprocal row on zone rows; other rows stay native. The
  persisted source scores the in-cut free→free block between tracked
  above-barrier states (`cr-planar-free-free-block`,
  `cr-planar-zone-row-ownership`); above-cut free→free stays unscored.
  Handed-off electrons keep native CBS.
- Squaring per-trajectory field amplitudes across MCWF trajectories is not a
  valid estimator.

*Outcome (#1148, planar v9):* zone rows carry depth-averaged flux weights
for incoherent bremsstrahlung and shell-resolved characteristic emission.
The weights are normalized expectations of localization kernels in the retained
transverse density; native continuation rows retain unit weight. Shell
localization uses the shell GOS and Slater form factor, with the distant
transverse term uniform. The brem kernel uses normalized screened-Born
transfer shapes with nuclear/electron strengths $Z^2:Z$; it neglects
photon-energy-dependent localization and distinct screening logarithms.
These are model assumptions, without a measured orientation-dependent error
bound. Axial fibre zone rows (axial v4) carry the same weights over the
2D cell, from the retained density of the fibre primaries. See
[`cr-zone-incoherent-flux-weight`](../validation/ledger-channeling.md#cr-zone-incoherent-flux-weight).

Straight zone rows can produce lines narrower than the energy-grid spacing.
Planar cases with off-own-row reflections warn when using node sampling;
`channeling_demo` selects bin-mean quadrature to preserve bin-integrated yield.
This does not bound omitted channeled-state PXR modulation
(`cr-planar-zone-pxr-modulation`, #1166). The rows' omitted angular spread is a
measured second-order omission: for `channeling_demo` it shifts zone-row
`spec` and `brem` by ≈2% forward and ≤0.5% at the Bragg and detector probes,
and characteristic emission by ~3e-6 (`cr-zone-row-angular-spread`, #1148).

CR is first exposed through a diagnostic API. It becomes the persisted additive
channel `spec_channeling` only after its anchors pass.
*Outcome (#507):* planar Python runs produce `spec_channeling` separately from PXR/CBS and characteristic emission, on the native line grid. Checkpoints store it in optional `channeling.h5`; common totals and the analysis component selector add it once. `CHANNELING_MODEL` v2 distinguishes enabled output identities, with off unchanged. The source reuses the constant-energy diagnostic estimator, slab-normal attenuation in a laterally infinite photon slab, and independently averaged finite electron path limits. Measured linewidth gaps remain explicit; persistence does not certify them.

Channeled-zone path segments are passed to the existing PXR/CBS and
bremsstrahlung scorers as straight mean segments. Channeled-state modulation of
PXR/CBS is recorded as an omission.

### Placement and data

- Static physics lives in `materials/channeling/`: potential, temperature
  dependence of thermal displacement $B(T)$, and the eigensolve.
  *Outcome (#1018, maintainer decision 2026-11-07):* `materials/channeling/`
  keeps only the beam-independent crystal properties (`thermal`, `potential`,
  `axial`). The beam-dependent Bloch eigensolve, dipole elements, entry
  populations, Doppler line energies and `longitudinal_kinematics` moved to
  `montecarlo/channeling/bloch.py`, and the sample-tilt entry geometry to
  `montecarlo/channeling/entry.py`, with the dynamics they feed. No alias
  modules remain (experimental surface); `hbar^2/2m` is now
  `materials.crystal.HBAR2_2M_EV_ANG2`.
- Dynamics, emission and handoff live in `montecarlo/channeling/`.
- The crystal catalog gains per-site thermal displacement, a reference
  temperature and a Debye temperature. Existing consumers keep reading the scalar
  `B_ang2`.
- Crystals without the new fields, or with placeholder `B_ang2`, are refused.
- A dechanneling-rate calibration factor is allowed only if it is declared and
  carried in identity and provenance.
  *Outcome (#504, maintainer decision 2026-10-15):* the declared calibration is
  the dechanneling cutoff `DECHANNELING_MARGIN_EV = 300` eV above the planar
  barrier, with `DECHANNELING_MARGIN_PROVENANCE` (`montecarlo/channeling/density.py`).
  It is calibrated on Kephart et al. (1989) through the forward-detector
  observable. Five of six Table I occupation lengths fall within ±20%, and
  Si(100) at 54.5 MeV is a documented 1.35× outlier. Persisted output (#507)
  must carry the cutoff in its identity.

## Run surfaces (#521)

Profiles select the zone in a `[channeling]` table (`model = "planar"`,
`plane`, and the zone resolution controls `basis_size`, `bloch_cells`,
`step_ang`, `zone_max_ang`); Python passes the same section as
`simulate(..., channeling=pyrite.quantum.Channeling(...))` (#1017). Both lower
to the unchanged flat `channeling_*` case record. Fixed-count CPU/CUDA sweeps,
physical-detector directions and observations score the same deterministic
ensemble source. Replay and line recompute rescore it on the requested grid;
bremsstrahlung repair preserves the stored channel. Channeling-off identities
and outputs retain their historical layout.

The ensemble zone solve and keyed handoff sampling run on the CPU. Native
continuation accepts the same `LaunchState` on the per-electron CPU core or
CUDA core. CUDA rows are downloaded once for the host zone/native join;
`keep_segments_on_device` therefore does not keep channeling rows resident.
The deterministic radiation source remains on the CPU. Existing stream keys
and channeling-off identities are unchanged by the backend choice.

The following consumers refuse channeling before transport or writes:

- Adaptive precision: the channel is a deterministic ensemble expectation,
  rather than an independent per-electron segment estimate. No CR variance
  observable is defined for the ADR-0017 stopping contract; use fixed counts.
- Trajectory capture and electron-block transport: the zone source and initial
  entrance population are not represented by the current block/capture
  contract. Use fixed-count execution without capture.

## Axial outcomes (#883–#886, #992–#994)

Axial channeling reached standard runs through a narrower route than the
planar tier. Each outcome below is recorded with its ledger claim; see the
[channeling ledger](../validation/ledger-channeling.md) for current status.

- **Zone model (#883).** The axial zone is a carrier-fibred, absorbing MCWF
  model with TDS and electronic jumps, rather than the periodic real-space
  grid and numerical sink of the diagnostic API. Its terminal fates (retained, outgoing, unresolved) form a physical
  outgoing instrument; outgoing electrons launch native transport with their
  own stream keys (`cr-axial-zone-handoff-physical`).
- **Depth budget (#884, #993).** A case runs only with a measured,
  repository-held bound ε on its on-axis band yield (`cr-axial-depth-budget`,
  `cr-axial-depth-calibration`). The first calibrations give ε = 3.97% for
  `si-fibre/A` and 9.71% for `si-fibre/B` (95%, 800 eV–10 keV band).
- **Source and partition (#885, #992).** The photon source is an
  expected-value two-time correlation estimator over fibre histories
  (`cr-axial-fiber-regression`). Bound→bound and capture belong to channeling,
  free→free to CBS (`cr-axial-source-partition`). The Si evidence is compared
  with Andersen et al., NIM 194, 209 (1982); see the
  [#992 note](../research/physics/axial-material-source-992.md).
- **Zone-row ownership (#994).** Native PXR/CBS scores
  zero-order reflections (`h·axis_uvw = 0`) only outside the zone. Since
  #1148 (axial marker v3), nonzero-order reflections retain zone rows because
  they are absent from the transverse continuum source. This partitions
  ownership without certifying the native channeled-state approximation.
  Standard off-axis/finite-aperture axial runs remain refused; under
  `admission="accept-inaccurate"` they lower to the unadmitted `directional`
  observable and score the fibre's two-current correlation at each direction
  with exact linear-interpolation depth weights
  (`cr-axial-directional-observation`, #1148). The in-cut free→free block that CBS would have carried is omitted and
  stored as a declared one-sided bound: 8.06% for A, 18.9% for B
  (`cr-axial-zone-row-native-lines`).
- **Admission (#886, #994).** Standard runs admit only cases in the
  repository lookup whose run claims are all `anchored`
  (`cr-axial-persisted-channel`): Si ⟨100⟩, 3.5 MeV, 0.3 µm, CPU, incoherent
  emission, on-axis zero-aperture `PointDetector`. Every other axial case,
  and replay, recompute, CUDA, block, adaptive, coherent and remote
  consumers, refuse before work. The
  [first envelope](../research/physics/axial-first-envelope-886.md) lists the
  evidence needed to widen it.

## Admission tables (#1105)

Planar and axial standard runs both refuse, before transport, every case that
is not in a repository admission table with a measured error budget. The axial
table is the #994 lookup above. The planar table
(`pyrite.quantum._planar_envelope`) holds one entry: the `channeling_demo` case
(Si (1-10), 17 MeV, 2 µm, beam in the planes, default zone numerics,
293.15 K, 300 eV margin). Its ε = 4.9% (95%) bounds the band-weighted
on-axis `spec_channeling` line source only. It does not bound native PXR/CBS,
incoherent bremsstrahlung or characteristic emission at a physical detector.
It was measured with the #993 paired-arm method on the
carrier-fibred planar reference (#1056,
[dechanneling boundary note](../research/physics/dechanneling-boundary.md#paired-arm-source-budgets-993-method)).
Admission is curated rather than set by an ε threshold. The measured Kephart
configurations (ε 113–572%) are not admitted.

`Channeling(admission="accept-inaccurate")` (profile `[channeling]`
`admission`) is the shared opt-in. The
user acknowledges that an unadmitted case may give very inaccurate results.
The case then runs, records an `unadmitted` admission with no ε, and never
shares a content key with an admitted case. The opt-in relaxes only the table
match. Energy, bound-state, thermal-data, solver-window, geometry and
observable refusals still apply.

## Consequences

- **Default refusal (#1105):** planar runs that previously ran on generic
  gates alone now need the opt-in unless they match the admission table.
  Planar case records gained an `admission` field (`CHANNELING_MODEL`
  `planar-density-matrix-emission-v8`). Admitted axial records and
  channeling-off identities are unchanged.

- **Layered-material thermal data (#525):** keep the existing isotropic
  Si/diamond rows. Layered support requires site identifiers tied to the CIF
  basis and temperature-dependent displacement covariances in a declared
  Cartesian frame. An element-only or plane-normal scalar loses inequivalent
  sites and cannot supply both transverse directions of the TDS kernel.
  Convert crystallographic ADPs with the lattice metric before rotation;
  retain the full plane-local covariance for scattering. Do not extrapolate
  a single-temperature refinement with an unsourced Debye model. This is the
  schema requirement for a future implementation; the present isotropic loader
  and kernels continue to refuse crystals without sourced thermal rows.
  Directional electronic loss and HOPG mosaic quadrature are separate gates.
  See the [material survey](../research/physics/channeling-materials.md).

- Default outputs, RNG consumption and cache identities are unchanged.
- Channeling gets a native, testable path that does not depend on an external
  transport engine.
- Planar results can be validated to solver tolerance:
  - unitarity with no scattering;
  - trace plus flux equals 1;
  - MCWF/density-matrix equivalence;
  - the rate-equation limit;
  - literature anchors for line energies, widths and occupation lengths.
- Geant4 remains only an external validation reference, not a transport backend.
- New event, channel and identity surfaces follow the ADR-0017 tier-3 checklist:
  replay, recompute and directional paths either support channeling or refuse
  it before work starts.
- Above 30 MeV, physical-support claims also depend on radiative loss after
  handoff (#439). Above about 100 MeV, quasiclassical methods take over (#440).
- Interfaces are chosen so that quantum coherent-emission research
  ([#501](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/501))
  can reuse the engine:
  - no structural basis truncation;
  - a generic emission operator;
  - arbitrary initial and product states;
  - a first-class density-matrix backend.

## Alternatives

| Alternative | Why rejected |
| --- | --- |
| Diagonal population (rate) equations | Cheapest and matches published codes, but drops coherences, so it is invalid for crystals thinner than the inter-level beat length. Kept as a limiting-case test. |
| Geant4 transport handoff | Unimplemented, adds an external engine and dataset pinning, and native transport already covers the energy range. |
| Quantum state embedded in every native core | Rechanneling for a large CPU/CUDA/LUT regression surface. |
| MCWF for planar production | Needs ~$10^5$ trajectories to match one noise-free density-matrix solve. |
| Frozen-phonon multislice in production | About $10^4$ slices per µm times $10^2$ configurations; retained as an oracle. |
| Classical or quasiclassical trajectories | Cannot produce discrete CR lines below about 100 MeV. |
