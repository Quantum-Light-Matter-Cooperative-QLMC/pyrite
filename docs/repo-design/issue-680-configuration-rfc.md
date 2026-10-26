# Simulation composition RFC: physical scenes and independent run configuration

**Status:** Target contract; shared resolution and independent quantum and
observation composition are implemented. Scene's run/acquisition settings are
compatibility-only `InitVar`s excluded from its canonical dataclass fields;
angular and planar detector responses and planar scoring do as well.
Configured profile adapters accept independent run and observation arguments;
downstream consumer migration remains. Unit-bearing physical inputs use the
normalization introduced by #709.
**Audit base:** `main` at `2ce1f174`, 2026-10-25.
**Scope:** [#680](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/680),
[#695](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/695),
and the composition boundary with
[#703](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/703).

This revision supersedes the earlier proposal to make `Scene` the simulation
composition root. It specifies the intended contract; examples marked proposed
are not executable API documentation. Current usage remains in
[the API reference](../api.md#scene-simulation). Implementation progress belongs
in the issues.

## Implemented resolution boundary

`campaign/resolution.py` now owns shared Scene/component composition and
model validation. `simulate` and `build_case` accept independent physics,
numerics and quantum settings. Omitted quantum inherits the legacy scene value;
explicit `None` disables it. The internal resolved request retains requested
inputs and effective quantum-adapted physics/numerics separately, so case
lowering does not repeat quantum adaptation. Observation planning and production
use the same resolution; configured sweeps share its model validation while
retaining their historical expansion and defaults.

`Observation` now independently composes detector references, response,
acquisition and pixel scoring through `simulate`, case lowering, observation
planning, production and Scene-based sweeps. Geometry references and matching planar values resolve
to the registered physical detector. The current virtual angular adapter uses
`Detector`, and sampling uses `PixelScorer`; renaming those concepts stays with
their owning issues. Legacy scoring omission is tracked through constructor
provenance, including dataclass replacement, outside scientific dataclass fields.
Scene-based sweeps apply independent physics/numerics/quantum overrides before
axes and bind the observation after expansion. Converted configured sweeps retain
their historical expansion, count grids, defaults and case names while accepting
independent physics, numerics, quantum and observation arguments. The resolved
observation supplies the detector direction before case expansion. Per-point
validation uses requested controls, so quantum adaptation is never applied to
already adapted settings.

This foundation retains the existing Scene and detector constructors and
compatibility attribute access while `physics`, `numerics`, `quantum` and
`acquisition` are excluded from `dataclasses.fields(Scene)` and
`dataclasses.asdict(scene)`. `Detector.response` and `PlanarDetector.response`
and `.scorer` are likewise excluded from their dataclass values. Compatibility
values survive `replace()` and pickle round-trips but do not enter physical
dataclass representations. The
independent execution argument in the proposed example below remains future
work; current execution controls use `Numerics.execution`.
No new public request type, units package, case schema or numerical model is
introduced by the resolver migration.

## Current implementation

The previous RFC predated work already present on the audit base:

- `Numerics.sampling` accepts `FixedSampling` and `AdaptiveSampling`, while
  retaining the historical count/`Precision` interface and serialization.
- `TransportNumerics`, `RadiationNumerics` and `ExecutionConfig` provide typed
  sections; `Numerics` still lowers them through legacy fields.
- `QuantumSetup` and `Beam.state` already have supported planar adapters and
  compatibility tests. General quantum production support remains bounded.
- `Scene` keeps physics, numerics, quantum setup and acquisition as
  constructor/access compatibility values outside its physical dataclass
  representation. Detector specifications do the same for read-time response
  and pixel scoring.
  `api._simulation_scene`, case lowering, profile adapters and observation
  planning still consume these compatibility values at different boundaries.

Reuse these implementations. `ExecutionConfig` describes local kernel/core
and chunk choices; it is distinct from `pyrite.execution.Executor` and resource
requests. A common composition boundary must retain that distinction rather
than introduce another scheduler or configuration store.

## Physical Scene contract

`Scene` describes an immutable physical system. Running it under a different
model, solver, observation, or backend must not require replacing its physical
fields. Its canonical fields contain sources, target geometry and material,
physical downstream bodies and positioned detector geometry.

| Concern | Intended owner | Boundary |
| --- | --- | --- |
| Energy, charge, source distributions, physical entrance state | `Beam` in `Scene` | State coordinates, units, normalization and sampled-state content are scientific inputs. |
| Material, layers, cut, orientation, shape, physical region | Target/bodies in `Scene` | Resolve material references before lowering; avoid a second copy of the geometry. |
| Passive downstream filters | Geometry in `Scene` | Preserve physical order; attenuation model selection belongs to physics. |
| Detector pose, active area and physical pixel layout | Detector geometry in `Scene` | Physical pixels are independent of angular scoring tiles. |
| Interactions, emission models, quantum dynamics | Physics/run configuration | Model enablement is independent of the existence of physical components. |
| Trial counts, stopping rules, grids, integration, quantum solver | Numerical/run configuration | Numerical approximations participate in affected scientific identities. |
| Virtual angular direction/acceptance, response, acquisition | `Observation` | A mathematical angular observation needs no positioned physical detector. |
| Angular sampling/reconstruction and response construction budgets | Observation numerical configuration | Owned by `pyrite.config`, composed by the observation. |
| Backend, chunks, workers, memory, scheduler | Execution configuration | Record provenance; preserve existing identity exclusions. |

A canonical scene contains no `physics`, `numerics`, `quantum` setup,
`acquisition`, response model, scorer or execution settings. Its physical
components must also satisfy this separation: merely removing fields from
`Scene` while leaving response/scoring on `PlanarDetector` is insufficient.
Compatibility constructors may still accept those values during migration.

A physical quantum entrance state remains on `Beam.state`. Crystal alignment
and actual physical region bounds belong to target/region geometry. Interaction
recipes, solver choice, FFT boundaries and requests for quantum observables
belong to run configuration. The current `QuantumRegion.zone_max_ang` needs an
explicit classification during migration: a computational handoff cap is not
made a physical boundary by moving it to geometry. Preserve its present lowered
meaning until that distinction is resolved. An absorbing FFT boundary never
becomes evidence of physical dechanneling.

## Composition through simulate

Keep `simulate` as the straightforward entry point. Compose a physical scene
with independent physics, numerics, observations, quantum setup and execution
arguments. Reuse typed values and named catalog objects under ADR-0021.

Proposed shape, with constructor spellings subject to their owning issues:

```python
scene = Scene(beam=beam, target=target, detector=panel, filters=filters)

observation = Observation(
    name="counts",
    detector="detector",
    response=response,
    acquisition=acquisition,
    sampling=angular_sampling,
)
result = simulate(
    scene,
    physics=physics,
    numerics=numerics,
    observation=observation,
    quantum=quantum_setup,
    execution=execution,
)
```

Calling this again with different run arguments reuses `scene` unchanged.
`simulate(beam, target, detector, ...)` remains a supported convenience adapter.
New geometry-only scenes without a detector can be lowered for source workflows;
`simulate` still requires a supported observation. Preserve historical defaults
for legacy construction rather than silently making a default angular detector
part of the new physical scene.

**Decision: defer a public `SimulationSpec`/`RunSpec`.** Current typed
configuration values and catalog references already provide reusable settings.
A new public wrapper would duplicate `simulate` arguments without demonstrated
campaign or serialization value. Use an internal immutable resolved request to
share validation and lowering. It must carry physical inputs, effective
scientific settings, observation requests, execution settings, schema origin and
per-field resolution provenance. Public promotion requires a concrete consumer
that benefits from a portable reusable request, plus API-governance review.

## Observation references and names

Use the proposed `pyrite.observations.Observation` from #703. It composes a
reference to physical detector geometry **or** a virtual angular specification,
response, acquisition, and numerical scoring configuration. Its resolved rays,
tiles and internal plan remain derived representations.

For the existing single-detector scene, reserve the reference `"detector"` for
`scene.detector`. The singular `observation=` form needs no user name; a supplied
name is a display/routing label. A future multiple-observation form must require
unique nonempty names and explicit detector references where ambiguous. Multiple
observations can reference the same physical detector with independent response,
acquisition and scoring settings. This contract does not promise a new plural
return type or multi-detector execution in the initial implementation.

A convenience `Observation(detector=panel, ...)` may be normalized to the scene
reference only when its canonical physical geometry matches the scene's
registered detector. Validate values, not Python object identity. A mismatched
pose, size or pixel layout raises an error naming the conflicting reference;
never allow an observation to override scene geometry. Portable requests store
the reference, avoiding independently mutable copies of physical geometry.

Use `AngularObservation` as the proposed name for the virtual direction and
acceptance specification, pending #703's export review. Retain root `Detector`
and its defaults through the compatibility adapter. No angular geometry is
inferred from an unrelated physical detector. Keep scalar source densities,
solid-angle-weighted averages, spatial accepted flux and acquired counts in
their existing units; composition does not change their meanings.

## One resolution path

The internal resolver is the authoritative boundary for `simulate`,
`build_case`, sweeps, profiles, observation producers, directional APIs and
photon APIs. It performs no transport, response-matrix generation, table fetch
or output writes. It may resolve local configuration/catalog metadata;
execution preparation performs expensive asset loading afterwards.

1. Adapt the input schema and record explicitly supplied values. Normalize
   physical inputs and approved quantities to existing unit-suffixed numeric
   values; resolve references and content-addressed state descriptors.
2. Resolve physics and quantum model selection, preserving supported capability
   gates and validating source/model compatibility.
3. Resolve numerical and sampling settings into one effective policy per
   estimator. Preserve schema-specific defaults and RNG/seed rules.
4. Resolve observation references, response capabilities, acquisition requests
   and scoring policies against the physical scene.
5. Resolve execution policy and validate backend compatibility before work.
6. Produce the immutable resolved request; existing case and observation
   lowerers consume its values without reapplying defaults.

Native acquisition edges remain an unresolved request until the supported
response/source information is available. Resolve the complete edge union
before acquisition identity, persistence or Poisson realization; the shared
resolver must not generate a response matrix merely to construct a request.

All entry points must use the same applicability and conflict validation.
Equivalent explicit API and CLI inputs resolve identically. Historical implicit
API fixed `450/100` and eligible CLI adaptive defaults remain tagged legacy
policies until an explicit profile/schema migration; they are not equivalent
inputs. New-schema profiles must write an explicit sampling policy so the
interface cannot silently choose the mode.

### Compatibility and precedence

Retain supported imports, positional construction, old scene fields, serialized
cases and checkpoints throughout the established compatibility window. The
migration adapter extracts legacy run fields into a private compatibility
record; they are not canonical physical fields. Deprecated accessors may expose
that record for old consumers. Dataclass replacement, sweep expansion and
provenance reconstruction must preserve it until those consumers migrate.
Implementation must demonstrate this behavior before replacing the constructor.

Omission must be distinguishable from explicit values, including explicit
`None` where it disables an optional request. Do not infer explicitness by
comparing to a default instance. Apply precedence to complete typed settings,
not an undocumented recursive merge:

| Input combination | Resolution |
| --- | --- |
| Explicit `physics`/`numerics` on a scene call | Override corresponding legacy scene settings, preserving today's behavior. |
| Omitted run argument with legacy scene settings | Use legacy values; retain historical defaults and serialized output. |
| Omitted run argument with canonical scene | Use the documented schema default. |
| Canonical `observation` plus embedded legacy response/acquisition/scorer | Accept if their effective values agree; reject differing explicit values with both paths in the diagnostic. |
| Explicit execution plus legacy numerical execution fields | Accept agreeing values; reject differing explicitly supplied fields. Omitted legacy defaults do not conflict. |
| Scene plus component `target`, `detector` or `filters` | Keep the existing mixed-input error; physical changes require a new scene. |
| Conflicting quantum setup and legacy channeling fields | Preserve refusal of contradictory explicit settings; never silently choose one. |
| Invalid detector reference or unsupported response/acquisition capability | Refuse before transport or writes. |

Continue rejecting `simulate(scene, acquisition=...)` under the existing API;
introduce independent acquisition via the new observation argument. The new
observation route must not weaken mixed-component diagnostics.

Warn when explicitly using migrated configuration fields in Scene or detector
constructors; implicit historical defaults must remain usable. Define field
warnings and removal dates in the implementation using the project's
[API compatibility policy](../development/api-policy.md); this RFC neither
starts a removal clock nor changes existing import-deprecation schedules.
Historical stored artifacts load through schema adapters without warnings for
each field. Never rewrite stored identities merely to normalize the Python API.

## Sampling and quantum requirements retained from #680

`FixedSampling` and `AdaptiveSampling` remain mutually exclusive effective
policies. `Precision` keeps its legacy serialization, including the historical
`line` observable spelling. Adaptive maxima are caps, not realized counts.
Independent brem stopping retains its prefix semantics; do not assume fixed and
adaptive populations have identical correlations.

Only supported CXR/brem estimators can use adaptive stopping. Preserve refusals
for unsupported coherent, channeling, cascade, GDF and physical-detector modes
until their variance/stopping contracts are implemented. New API names cannot
imply additional solver or estimator support.

Reuse current `QuantumSetup`, `QuantumDynamics`, `QuantumNumerics` and source
state implementations; do not recreate the already implemented #680 foundation.
Move the setup composition off Scene through the adapter. Initial state belongs
to the physical beam; sampled state arrays retain shape, dtype, units, coordinate
conventions and content digest. Keep optional source adapters lazy.

Production quantum support remains the supported planar Bloch density-matrix
path. Unsupported free/axial/MCWF configurations continue refusing where they do
today. Preserve axial production gates, channeling-off payloads and RNG use.
Separating configuration does not validate another propagation model.

## Scientific identity and serialization

Do not hash a dataclass dump of the new composition. Lower equivalent requests
to the established case and observation payloads, retaining schema versions,
conditional model markers, omission rules and response type identifiers.

| Layer | Inputs that must matter | Reuse constraint |
| --- | --- | --- |
| Physical/source inputs | Source distributions/state, material, geometry | There is no new all-purpose physical-scene cache key in this proposal. |
| Source computation | Existing resolved case, numerical policy, seed and model/table markers | Preserve `case_content_key` and dataset identity independently; they gate different storage. |
| True-spatial observation | Source digest, detector geometry, filters, numerical scorer, representative rays and attenuation arrays | Response/acquisition changes preserve compatible factors. |
| Response | Canonical response config, implementation identity and event semantics | Preserve historical qualified type strings through namespace migration. |
| Acquisition | Resolved bins, exposure, threshold, normalization and realization policy | Native edges resolve first; retain coordinate-stable Poisson RNG semantics. |
| Full observation | Existing links between true-spatial, response and acquisition digests | Names/references route values; labels alone cannot change scientific digests. |
| Execution | Backend and resource provenance | Preserve established chunk/resource exclusions; add identity inputs only for documented numerical/reproducibility effects. |

Current scalar source cases include observation direction; source reuse across
arbitrary angular observations is not promised. Likewise, changed numerical
angular sampling can require new directional source evaluations. Preserve the
existing factorization boundaries rather than assuming every observation can
reuse the same computed arrays.

Existing identity owners are
{source}`src/pyrite/campaign/profiles.py::case_content_key`,
{source}`src/pyrite/campaign/profiles.py::dataset_identity`, and
{source}`src/pyrite/instrument/observation.py::observation_identity`.
Response identity currently embeds the qualified Python type, and acquisition
identity includes beam rate/charge normalization. These dependencies require
explicit compatibility adapters if types move or configuration splits.

Preserve old checkpoint/case readers and resolved values. A new portable
request schema, if needed internally, must distinguish specification version
from historical case/observation schemas. Normalize approved unit-bearing
quantities at the boundary under the dedicated units contract; this RFC selects
no units package. No quantity objects enter kernels, CUDA arrays or existing
checkpoint payloads. Equivalent numeric and quantity inputs must yield the same
canonical numerical values and identities.

## Consumer boundaries and verification

The current implementation audit identifies these migration seams. This table
describes the required integration contract; issue checklists own scheduling.

| Current seam | Required migration and focused evidence |
| --- | --- |
| `campaign/model.py::Scene`, `api.py::_simulation_scene` | Extract compatibility settings, retain constructor/access behavior and override semantics; `tests/scan/test_public_api.py`. |
| `campaign/lowering.py::build_case` and `build_sweep_cases` | Consume resolved inputs once; preserve case payloads, seed order and legacy sweep handling. |
| `Sweep` and optimization parameter paths | Keep physical axes on scene; route run axes to run settings. Preserve legacy `physics.*`/`numerics.*` paths, label order, Cartesian expansion and override-before-axis precedence through adapters. |
| `campaign/legacy.py` and profile/settings adapters | Map existing profiles into the same resolver; preserve their schema defaults and named-preset layering under ADR-0021. |
| `api.py::observation_plan`, `observations/plan.py`, `runs/observe.py` | Build plans from independent observation specs; preserve native bins and reuse checks; `tests/observations/test_produce.py`, `test_sweep.py`. |
| `quantum.py` and channeling lowering | Move composition without changing versioned records; `tests/scan/test_quantum_configuration.py`, `test_quantum_state_spec.py`. |
| Directional, photon-event and photon-continuum entry points | Use shared source/model validation while retaining each pathway's capability refusals and distinct tally identities. |
| Result provenance and reconstruction | Record physical scene and resolved run settings separately; retain legacy `Result.scene` access and historical artifact reconstruction. |
| Apps, notebooks, CLI examples | Teach the independent composition after runtime support exists; regenerate CLI references only if CLI contracts change. |
| Public imports/dependencies | Register implemented contracts in `api-surface.toml`, preserve alias object identity and lightweight imports, run API checks and import-linter. |

Before migrating a seam, capture representative legacy case mappings and both
case/dataset digests. Cover scalar and pixel cases, fixed/adaptive defaults,
explicit overrides, conflicts, quantum off/on and converted sweeps. Compare
resolved values and serialized identities first; use focused execution tests
where the consumer changes. Existing identity-layer and rescoring tests in
{source}`tests/instrument/test_observation.py` and
{source}`tests/observations/test_produce.py` protect response/acquisition reuse.

The contract is implemented only when one immutable physical scene can be used
with independent run and observation settings, all consumers share resolution,
legacy calls retain their behavior, and the configured API/dependency checks
pass. Publishing this proposal alone does not satisfy those runtime gates.
