# 0018 — Model electron channeling as a quantum entrance zone

- **Status:** Proposed
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
- Free→free transitions are coherent bremsstrahlung (CBS), which the PXR/CBS
  kernel already produces, so they are never scored here.
- Squaring per-trajectory field amplitudes across MCWF trajectories is not a
  valid estimator.

CR is first exposed through a diagnostic API. It becomes the persisted additive
channel `spec_channeling` only after its anchors pass.

Channeled-zone path segments are passed to the existing PXR/CBS and
bremsstrahlung scorers as straight mean segments. Channeled-state modulation of
PXR/CBS is recorded as an omission.

### Placement and data

- Static physics lives in `materials/channeling/`: potential, temperature
  dependence of thermal displacement $B(T)$, and the eigensolve.
- Dynamics, emission and handoff live in `montecarlo/channeling/`.
- The crystal catalog gains per-site thermal displacement, a reference
  temperature and a Debye temperature. Existing consumers keep reading the scalar
  `B_ang2`.
- Crystals without the new fields, or with placeholder `B_ang2`, are refused.
- A dechanneling-rate calibration factor is allowed only if it is declared and
  carried in identity and provenance.

## Consequences

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
