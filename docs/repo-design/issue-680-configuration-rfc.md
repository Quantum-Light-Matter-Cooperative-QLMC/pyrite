# Issue #680 — Coherent simulation configuration: quantum API plus sampling/numerics cleanup

**Status:** Proposal for maintainer review; no runtime changes in this slice
**Target:** PyRITE `main`, inspected 2026-10-24
**Issue:** https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/680
**Related:** ADR-0018 (quantum channeling zone), ADR-0019 (CLI ownership), ADR-0021 (#663, named physics/numerics presets), #623/#624 (experimental axial configuration), #361 (adaptive sampling), #508 (transverse source states).

## Decision requested

Adopt a **typed, composed configuration** with orthogonal ownership and explicit sampling modes. Make quantum source, quantum dynamics, and quantum propagation **separate reusable objects**. Preserve the existing classical `Scene` as the composition root, adding an opt-in `QuantumSetup`. Do not create another mutable, loosely typed super-object or expose arbitrary Python callables through the catalog.

Most importantly: `Numerics` describes *algorithms and discretization*, `Physics` describes *physical approximations and interactions*, `Sampling` describes *statistical estimator budgets/stopping*, `Execution` describes *hardware/scheduling/resource settings*, and quantum entry states describe *the source*. Named catalog objects are immutable **resolved values** and can be referenced from profiles, per ADR-0021.

## 1. Current implementation audit

The public API currently has the following issues (code locations are for the current `main`, not proposed new files):

| Current element | Observed behavior | Proposed owner |
| --- | --- | --- |
| `Numerics.cxr_trials=450`, `.brem_trials=100` | Positive fixed counts exist even with an adaptive `Precision`; adaptive lowering intentionally ignores them (`tests/scan/test_adaptive_api.py::test_lowering_uses_policy_maximum_and_ignores_fixed_defaults`). | Mutually exclusive `FixedSampling` / `AdaptiveSampling` |
| `Numerics.precision` | Hidden alternative to two fixed-count fields; rejects `lockstep` while the fields remain visible. | `Sampling(mode='adaptive')` selection; retain `Precision` as deprecated compatibility alias for stopping rule |
| `Precision.observables=('cxr', 'brem')`, `.brem_target_rse`, `.brem_min_electrons`, `.brem_max_electrons` | A CXR/brem-specific adaptive controller masquerades as general precision. Brem may use a separately stopped **prefix** under the independent mode; only two estimators are implemented. | `AdaptiveSampling` with explicit CXR and optional brem stopping components; generalized observables later **only after** per-observable variance estimators exist |
| `Numerics.cxr_chunk`, `.brem_chunk`, `.transport_core` | Backend and memory/batch choices intermixed with scientific discretization. | `Execution`; never hashed when genuinely result-invariant, but recorded in execution provenance |
| `Numerics.energy_model`, `.max_dE_frac` | Transport integration algorithms. | `TransportNumerics` |
| `Numerics.cxr_reflections`, `.energy_bins` | CXR-specific reflection and photon-grid quadrature. | `RadiationNumerics` (`cxr`, `brem` child options) |
| Four `Numerics.channeling_*` settings | Planar basis/grid/depth integration with no typed quantum sub-object, validated even when channeling is off. | `QuantumNumerics` in a reusable `QuantumSetup` |
| `Physics.channeling_model`, `.channeling_plane`, `.channeling_axial` | Physical selection and plane geometry plus an experimental JSON mapping carrying source/solver/limits. | Typed `QuantumDynamics` / `QuantumRegion`, source state, solver and execution controls |
| `Physics.emission` | CXR emission estimator selection, **not** a declaration of electron wavefunction coherence. | Continue as radiation physics/observable selection; distinguish explicitly from `Beam.state` |
| Profile `precision` | CLI refuses explicit fixed counts alongside adaptive, and defaults an eligible profile without counts to adaptive. | Same tagged `sampling` policy as API; no implicit cross-route semantic drift |
| Bare `Numerics()` | Uses fixed `450/100`, unlike the eligible CLI default adaptive stopping policy. | One documented default for **new** v2 configurations; migrate legacy configurations without changing their results |

Additional constraints:

- `campaign/lowering.py` ultimately constructs `Case` through `build_cases`; the `Case` and dataset digests are sensitive to scientific inputs. Keep this as a centralized lowering boundary.
- `campaign/channeling_cases.py` uses four planar parameters to construct a versioned channeling case record. Existing channeling-off outputs deliberately have no extra marker. **Keep off identical.**
- `campaign/axial_spec.py` is a versioned *candidate* axial specification with `entry`, mesh, limits, pending physical boundary/source; current `refuse_unverified_axial_run()` intentionally prevents production axial runs. New configuration must not bypass that gate or equate FFT boundary loss with physical dechanneling.
- `montecarlo/channeling/states.py` already defines `TransverseState`, `SampledState`, `Superposition`, mixtures, Gaussian/HG/LG/Bessel/Airy/aperture sources, `from_array`, and `from_abtem`. This is a **foundation to reuse**, not code to reimplement.
- `docs/adr/0021-named-physics-numerics-presets.md` already establishes catalog identities, user/bundled layering, preset references and fixed-vs-adaptive exclusivity. Extend it; do not create a competing registry.

## 2. abTEM lessons worth copying (and not copying)

abTEM distinguishes `PlaneWave`/`Probe` input builders, `Waves` arrays, `Potential` construction, multislice propagation, detectors and measurements. Its `Probe.multislice(potential, detectors=..., max_batch='auto', lazy=...)` makes source and propagation explicit, while Dask-backed lazy arrays separate construction from execution. Device/FFT-cache/chunking settings belong to configuration or execution, not the physical initial state. Reference: [abTEM Waves/PlaneWave](https://abtem.github.io/doc/reference/api/_autosummary/abtem.waves.PlaneWave.html), [Potential](https://abtem.github.io/doc/reference/api/_autosummary/abtem.potentials.iam.Potential.html), [Parallelization](https://abtem.github.io/doc/user_guide/walkthrough/parallelization.html).

Adopt the *separation of responsibilities*, abTEM adapters, inspectable resolved computation graphs/budgets and lazy asset loading. **Do not** make PyRITE `Scene`/`Numerics` into a `Waves` array; PyRITE's electron transport, mixed-state jumps, radiation and physical handoff are not simply a multislice problem. Do not make Dask a hard dependency of state definitions or solver semantics; if adopted, it belongs to `Execution`.

## 3. Proposed object model

**Composition root:** retain `Scene(beam, target, detector, physics, numerics, ...)`, and add `quantum: QuantumSetup | None = None` plus `execution: Execution = Execution()` after feature-gating. `QuantumSetup` owns a **reference** to the dynamics, propagation numerics and requested quantum observables; initial state belongs to `Beam` and the crystal potential is resolved from `Target` (no duplicated physical energy/geometry). Both quantum and classical solvers can consume the same `Scene`, but `quantum=None` is strictly legacy-equivalent.

| Type (tentative public name) | Contents | Identity/provenance classification |
| --- | --- | --- |
| `Beam.state: TransverseStateSpec | None` | Entrance pure state, coherent superposition, or statistical mixed state; units and beam-frame coordinate conventions explicit. Uses #508 factories. | Scientific, serialized spec and content hash of sampled data |
| `QuantumRegion` | Planar/axial crystal alignment, potential source, physical zone bounds, handoff/escape definition. Plane/axis selectors use target-crystal orientation. | Scientific |
| `QuantumDynamics` | Hamiltonian/potential recipe and physical dissipation/scattering mechanisms (e.g., validated Lindblad operator families), plus model compatibility. | Scientific |
| `QuantumNumerics` | Representation (`BlochBasis`, `RealSpaceGrid`, etc.), solver (`density-matrix`, `MCWF`, coherent-only), propagation step, tolerance, quadrature, numerical/FFT boundary. | Scientific when affecting numerical approximation |
| `QuantumObservables` | Population/coherence/flux/CR requests, saved snapshots, supported estimator contracts. Distinguish physical observables from presentation toggles. | Scientific if changes computed/stored arrays; display-only fields elsewhere |
| `QuantumSetup` | `region`, `dynamics`, `numerics`, `observables`; optional when no quantum stage. | A typed, inspectable composition of the preceding values |
| `TransportNumerics` | Classical electron integration and bounded-step options. | Scientific |
| `RadiationNumerics` | CXR reflection/mosaic quadrature, photon grids, component-specific integration settings. | Scientific |
| `FixedSampling` | Explicit counts for line and continuum estimators, along with a defined count/correlation model. Preserve current fixed-path behavior. | Scientific |
| `AdaptiveSampling` | Existing error targets, block estimator, stabilization and heavy-tail guards, supported CXR/brem stopping choices. | Scientific |
| `Execution` | Core/backend/device, CPU workers, scheduler, chunk/batch sizes, memory ceilings, streaming/spill policy and cluster profile. | Execution provenance; identity only for settings proven result-affecting |

`Precision` describes **statistical accuracy**, not float32/float64 arithmetic; give it a nonambiguous new name. GPU dtype/FP64 requirements are separate `Execution`/numerical representation constraints. Do not promise FP32 parity.

**Important separation within quantum settings:** physical loss/handoff is in `QuantumRegion`/`QuantumDynamics`; periodic grids, FFT buffers, absorbing numerical layers and discretization are in `QuantumNumerics`. An FFT-box exit is **not** physical dechanneling without a validated flux model. A planar plane index is region geometry, not integration numerics. A solver limit like `max_working_bytes` is execution, not material physics.

Avoid a general-purpose `Hamiltonian` *plugin code registry* in phase 1. Provide typed supported recipes, a `to_spec()`/`from_spec()` interface and a backend protocol that can later admit additional validated operators. Arbitrary state samples are allowed via a **content-addressed array artifact** with units, grid/FFT convention and digest, not embedded Python functions in portable profile TOML.

## 4. Illustrative target Python API (NOT yet implemented)

```python
import pyrite as pr

state = pr.states.laguerre_gauss(p=0, ell=1, waist_ang=80.0)
beam = pr.Beam(
    energy_keV=5_000.0,
    transverse=pr.Spot(),
    longitudinal=pr.PointBunch(),
    state=pr.TransverseStateSpec.from_runtime(state),
)
scene = pr.Scene(
    beam=beam,
    target=pr.Slab("si", thickness_ang=100_000.0, tilt_deg=0.0),
    physics=pr.Physics(radiative_model="uncoupled"),
    numerics=pr.Numerics(
        sampling=pr.FixedSampling(cxr_electrons=500, brem_electrons=200),
        transport=pr.TransportNumerics(energy_model="midpoint", max_dE_frac=0.01),
        radiation=pr.RadiationNumerics(cxr=pr.CxrNumerics(reflections=pr.CxrReflections())),
    ),
    quantum=pr.QuantumSetup(
        region=pr.PlanarChannelingRegion(plane_hkl=(1, 1, 0)),
        dynamics=pr.ChannelingDynamics(scattering=("tds", "electronic")),
        numerics=pr.QuantumNumerics(
            basis=pr.BlochBasis(half_width=16, cells=16),
            method="density-matrix",
            step_ang=250.0,
        ),
        observables=(pr.StatePopulations(), pr.ChannelingRadiation()),
    ),
    execution=pr.Execution(transport_core="auto", cxr_chunk=None, brem_chunk=None),
)
```

This is a **shape proposal**: source factories and `Scene` exist, but several names/constructors above do not. It intentionally cannot be run as-is. `ChannelingDynamics` and `QuantumSetup` should express a generic quantum problem with a *specialized* channeling recipe, not assume every quantum calculation is channeling. Later recipes can include `FreePropagation`, crystal-scattering potentials and custom validated operators without new top-level numerical kwargs.

Optional convenience `pr.Sampling.fixed(...)` / `pr.Sampling.adaptive(...)` factories can return the two discriminated frozen dataclasses; a separate polymorphic base class is not necessary.

## 5. Sampling resolution contract (priority 1)

Make `sampling` a **tagged union** rather than a bag of nullable fields:

```python
pr.Numerics(sampling=pr.FixedSampling(cxr_electrons=450, brem_electrons=100))
# OR
pr.Numerics(sampling=pr.AdaptiveSampling(
    target_rse=0.05, min_electrons=200,
    max_electrons=20_000, block_electrons=100,
    observables=("cxr", "brem"),
))
```

**Invariants:**

1. Exactly one sampling mode is active in a resolved v2 `Scene`/profile. `FixedSampling` never has adaptive tolerances; `AdaptiveSampling` never has fixed requested counts. `cxr_trials`/`brem_trials` cannot silently override or be ignored alongside adaptive mode. Legacy API kwargs get a deprecation warning or a clear error **when explicitly conflicting**, but unmodified old default construction continues through the compatibility resolver.
2. Adaptive sampling retains its present scope: only the validated CXR/brem incoherent estimators. Unsupported modes (coherent scoring, channeling CR, cascades, GDF, physical detector) refuse as today **until corresponding per-primary variance estimators and stopping contracts are implemented**. Never expose a generic `observables=('cr', 'dose')` API ahead of support.
3. Existing independent-brem adaptive stopping has brem-prefix behavior; do not assume fixed-count continuum sampling shares identical correlation semantics. Keep the two reducers' identity behavior and RNG streams unchanged until a dedicated statistical audit.
4. One shared `resolve_sampling(scene, profile, overrides) -> ResolvedSampling` returns requested policy, effective per-estimator work budgets, reason/source, and gate decisions; it is the sole lowering path for CLI and API. For adaptive, `max_electrons` is a cap, not a promised realized count. Expose `.explain()`/CLI `--explain` without running.
5. **Default migration:** old API's implicit 450/100 and eligible CLI profile's implicit adaptive 5% / 200–20,000 / 100 remain under legacy schema handling, to protect reproducibility. New v2 profile creation writes an explicit sampling reference/value; choose one well-documented default preset (rather than route-dependent hidden defaults). Do not silently change existing profile or case hashes.
6. Preserve `Precision.to_dict()`'s historical `line` observable serialization within legacy case/checkpoint identity until an explicit schema-version migration. An alias or adapter may be needed; changing the observable spelling in serialized historical objects is not cosmetic.

## 6. CLI and reusable catalog proposals

**Extend ADR-0021's named-object mechanism; do not create a second configuration store.** Implement one unified `pr.catalog` resolution service with typed references, clone/adopt/validate/diff, user/bundled layers, immutability rules and provenance. Add catalog `sampling`, `execution`, `quantum_state`, `quantum_dynamics`, `quantum_solver` objects. A `quantum_setup` named preset is optional composition, not a new model registry. Keep physical regions explicit in the setup/profile when they depend on target orientation; reusable region templates may be added later.

Illustrative commands (names provisional):

```sh
pyrite sampling preset create adaptive-5pct --mode adaptive --target-rse 0.05 --min-electrons 200 --max-electrons 20000 --block-electrons 100 --observables cxr,brem
pyrite quantum state create vortex1 --kind laguerre-gauss --ell 1 --p 0 --waist-ang 80
pyrite quantum dynamics create planar-tds --kind channeling --region planar --plane 1,1,0 --mechanisms tds,electronic
pyrite quantum solver create planar-dm16 --method density-matrix --basis bloch --half-width 16 --cells 16 --step-ang 250
pyrite execution preset create gpu-worker --core cuda --cxr-chunk auto --memory-limit 4GiB
pyrite profile set my-run --sampling adaptive-5pct --execution gpu-worker
pyrite profile quantum set my-run --state vortex1 --dynamics planar-tds --solver planar-dm16
pyrite profile validate my-run --explain
pyrite profile show my-run --resolved --output json
```

Example conceptual TOML (exact syntax subject to parser implementation):

```toml
[sampling.adaptive-5pct]
mode = "adaptive"
target_rse = 0.05
min_electrons = 200
max_electrons = 20000
block_electrons = 100
observables = ["cxr", "brem"]

[quantum_state.vortex1]
kind = "laguerre_gauss"
version = 1
ell = 1
p = 0
waist_ang = 80.0

[quantum_solver.planar-dm16]
method = "density-matrix"
representation = "bloch"
half_width = 16
cells = 16
step_ang = 250.0

[profiles.my-run]
beam = "beam5mev"
numerics = "electron-standard"
sampling = "adaptive-5pct"

[profiles.my-run.quantum]
state = "vortex1"
dynamics = "planar-tds"
solver = "planar-dm16"
```

A profile with `quantum` absent follows classical transport without allocating a quantum mesh. `quantum` present **does not** mean an experimental axial run is ready. `--explain` reports each field's origin, e.g., `preset adaptive-5pct`, `profile override`, `built-in`, alongside resolved unsupported-science gates. The CLI must not falsely advertise compatibility: validation should refuse adaptive sampling with the current deterministic channeling-CR path and respect #623's axial refusal.

Keep old `pyrite numerics precision enable|disable` and old `pyrite physics/numerics preset` commands as migration aliases with warnings; do not redirect them to an incompatible policy silently. Do not overload the words `precision` (numerical floating type, statistical stopping, grid accuracy).

## 7. Identity, serialization, and safety

Separate three representations:

1. **User specification:** named references/partial overrides, explicit units and schema version.
2. **Resolved scientific specification:** full typed, immutable, JSON-safe values with validated model support and content-addressed input arrays. This alone determines scientific case identity and checkpoints (subject to existing compatibility marker/version rules).
3. **Execution plan/provenance:** normalized backend, worker scheduler, chunks, cache/spill budget and hardware availability. Execution choices recorded per run; they influence content key only where a documented numerical result dependency exists.

Compile once: `profile/Scene -> resolve references/defaults -> validate applicability -> lower Case -> execute`. Use the **same** validator and lowering contracts in Python API, CLI, remote serialization and sweeps. Never hash preset *names*; do record name/digest as non-hashed provenance per ADR-0021.

For old scenes/configurations, canonical lowering and case serialization should be **byte-for-byte equivalent**. New quantum cases get an explicit schema/model/version marker. Do not reuse a classical cached result for an enabled quantum simulation. `off` must add no extra scientific keys or RNG use. Arrays must carry shape, dtype, grid convention, physical units and SHA-256 digest; require known portability and optional abTEM import at the edge. Reject unsupported backend/observable combinations *before* transport or output writes.

## 8. Incremental implementation plan (separately reviewable PRs)

**Slice A — Contract and diagnostic tests (safe foundation).** Approve this RFC, enumerate all current profile/preset/numerics/physics keys and values, add an exhaustive effective-settings matrix for API vs CLI, fixed vs adaptive, coherent vs incoherent, and channeling off/planar/axial. Record pinned legacy case/dataset keys and representative hashes. Add a `describe/explain` resolver with explicit inactive/ignored fields, scientific/execution classification and source. No run semantics change.

**Slice B — Sampling-policy union.** Add frozen `FixedSampling`/`AdaptiveSampling` with `Precision` adapter/alias, choose a single tagged mode at API construction and after catalog resolution, move lowerers to `ResolvedSampling`. On old API calls preserve positional/keyword compatibility and emitted identities; new constructions with conflicting explicit sampling values should raise. CLI new `sampling` commands and presets; migrate only when user explicitly requests `profile migrate`. Test independent brem, peak/band RSE, pilot/stability guards, and all unsupported modes.

**Slice C — Separate execution and classical numerics.** Add `Execution`, `TransportNumerics`, `RadiationNumerics` and compatibility accessors/constructors to `Numerics`. Move chunk/core from scientific spec without changing backend policy or current cache exclusion. Add `validate` and resolved/config inspection. Tests across CPU, GPU options (GPU skipped without hardware), sweeps, remote jobs, and resuming checkpoints.

**Slice D — Shared quantum types and import adapters.** Lift/alias the current `montecarlo.channeling.states` abstractions into stable public `pr.states` / `pr.quantum` entrypoints; add typed/JSON-safe `TransverseStateSpec`, file-backed sampled-state identity, schema and units, pure/mixed source handling. Do not rewrite amplitude evaluators. Add `QuantumRegion`, `QuantumDynamics`, `QuantumNumerics`, `QuantumSetup` with explicit supported types and no-op/off behavior. Separate source/state, physical boundary and numerical boundary.

**Slice E — Planar production adapter.** Rewrite only the planar case resolver to take `QuantumSetup`, lowering to the **unchanged** versioned channeling record and respecting ADR-0018 gates. Keep legacy `Physics(channeling_model='planar')` and `Numerics(channeling_*)` through deprecating adapters. Add roundtrip CLI spec/preset tests and equivalence golden tests for density evolution, dechanneling flux, spectra, RNG and identity.

**Slice F — Axial diagnostic adapter and capabilities.** Map #624's versioned entry/mesh to the new quantum objects; retain #623's refusal on unverified physical runs and pending sources. Expose diagnostics as explicitly nonproduction. Before enabling quantum transmission/radiation, validate source, depth budget, boundary flux and physical matching. Add solver/device capability matrices and explain reasons for unsupported combinations.

**Slice G — Replace legacy storage at a version boundary.** Add `pyrite profile migrate --configuration-v2 --dry-run` and a deterministic migration report. Support old profile/preset schema through the agreed deprecation window; annotate release notes and Sphinx API docs. Keep old checkpoint readers, data provenance and reproducibility available.

## 9. Acceptance gates

- `pr.Numerics()` and an equivalent **new** CLI profile resolve the same documented sampling mode. Legacy stored profiles preserve historical default rules and hashes.
- A resolved configuration cannot contain both adaptive stopping and effective fixed requested counts. Conflicts either raise before work or are explicitly marked as legacy-only in `.explain()`.
- Every active setting can be traced through public API, named catalog object, CLI setter, validation, lowered Case, and scientific/execution provenance.
- All new configuration types are immutable with defensive copies or frozen normalized data where necessary; no hidden mutable `Mapping` breaks immutability.
- Planar legacy fixed runs are numerically and identity equivalent; channeling-off has identical payload and RNG use; unsupported axial and adaptive channeling paths still refuse.
- `QuantumStateSpec` supports source factories, mixtures/superpositions, sampled inputs and optional abTEM import without requiring abTEM in core runtime or storing arbitrary callbacks.
- Tests cover invalid combinations, source-frame units, mixed-state normalization, reflection settings, plane/axis orientation, solver boundary interpretation, checkpoint/remote serialization, and old-version migration.
- `pyrite profile show --resolved` and `validate --explain` expose which settings are active, sources, fallback reasons, capability limitations and resource estimates before run.

## Unresolved decisions for the maintainers

1. Prefer an explicit fixed default for **new v2** scenes/profiles, or a named default adaptive preset? Both can be user-friendly if *identical across interfaces*, resolved visibly and pinned. Current differences should not propagate.
2. Preserve current public name `Precision` as a long-lived alias for `AdaptiveSampling`, or deprecate after the next major API version?
3. Do named quantum state recipes belong to beam-object references or the profile's optional quantum composition? This RFC keeps `Beam.state` as the runtime physical owner and uses `profiles.*.quantum.state` as a reference resolved into that beam state, but the parser must enforce that the two sources never conflict.
4. Should `Execution` have user presets in the material catalog (convenient) or in a separate machine-local catalog? Prefer machine-local *defaults*, transferable requested execution policies, and explicit hardware capability negotiation, with execution excluded from scientific identities unless demonstrated necessary.
5. Which first generic nonchanneling quantum propagation target is useful and readily validated (free-space evolution with plane-wave/Gaussian input is a low-risk candidate)? Include it in the acceptance gate so these classes are genuinely general purpose.

**Bottom line:** Do not migrate all legacy fields in one PR. Start with sampling semantics and a canonical resolution/explanation layer, then add the quantum object model by reusing #508, and only afterwards attach planar and axial execution paths. This avoids a large facade-only reorganization or an accidental scientific-regression surface.
