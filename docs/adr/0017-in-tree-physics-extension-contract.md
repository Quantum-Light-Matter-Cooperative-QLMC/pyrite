# 0017 — Extend physics through in-tree preparation and batch scoring contracts

- **Status:** Proposed
- **Date:** 2026-10-13
- **Issue:** [#485](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/485)

## Context

`Physics.validate_transport` in `src/pyrite/_numerics.py` owns public model
compatibility. `campaign/lowering.py` preserves deprecated `Numerics` inputs;
`campaign/sweep.py::build_cases` resolves selectors into `montecarlo/case.py::Case`.
Transport preparation lives in `montecarlo/runner/case_tables.py`, native dispatch
in `montecarlo/transport/api.py`, and scoring in `montecarlo/runner/`.
These are different boundaries: registering metadata cannot implement a new
collision scheduler or a persistent spectral channel.

This ADR defines the proposed migration contract. The current implementation
still uses those owners and their explicit branches. No descriptor catalog or
adapter runtime is introduced by this record.

## Decision

### Ownership and dependencies

Put immutable descriptor metadata and the data-only resolved view in an
import-light private root owner, `pyrite._physics_contract`. It may import the
standard library and import-light shared types, but must not import campaign,
runner, scientific material loaders, device backends, or compiled kernels.
Use primitive immutable fields for IDs, capabilities, parameters and data
markers; keep any type-only `Case` dependency under `TYPE_CHECKING`.

`_numerics.py` may consume metadata while retaining validation policy;
`campaign/` derives the resolved view from its resolved `Case`. Runtime adapters
remain in `montecarlo/runner/case_tables.py` and `runner/emission.py` and consume
that view. They own table loading, arrays and backend resources. Metadata must
never reach upward into runner for validation. Identity encoding remains in
`campaign/profiles.py`; provenance remains in `results/provenance.py`.
There is no second public selection authority or new public `PhysicsPlan`.

Use an explicit immutable in-tree catalog, rejecting duplicate IDs within a
family. Family-qualified IDs distinguish `transport.elastic/mott`,
`transport.elastic/elsepa`, `emission.continuum/eedl`,
`emission.continuum/bremslib`, `transport.radiative/bremslib-soft-hard`,
`emission.cxr/cxr` and `emission.characteristic/characteristic`.
These internal IDs describe existing selectors; they do not replace strings
in historical cases. Shell, pair and positron dependencies are explicit
requirements, not independently combinable scoring producers.

Keep stable model IDs distinct from numerical-generation markers. Reuse the
existing `*_MODEL` constants and xsgen table key/manifest digests. A descriptor
revision documenting capabilities is not a physics-generation change. A real
change in equations, estimators, selected parameters or data must enter both
existing identity surfaces and follow the validation/model-marker rules.
Backend availability and resource/chunk choices stay outside physical identity
when they implement the same numerical contract. Do not hash the whole catalog:
adding an unused descriptor cannot invalidate a case.

### Resolution sequence

1. Public facade/deprecated lowering validates selector syntax and integration
   compatibility through `_numerics.py`.
2. Campaign resolution checks **all layer materials/elements** before choosing
   `auto` in `campaign/inelastic_cases.py` and `bremsstrahlung_cases.py`.
   Keep existing warnings and fallback rules. Explicit unsupported requests
   fail; do not turn them into a new implicit fallback.
3. `Case` validates and serializes the resolved choice with its historical
   omission rules. An absent elastic key means Mott even though the direct
   transport API defaults to ELSEPA. An absent continuum key means EEDL;
   absent radiative keys mean uncoupled. Legacy decoding must not fill these
   keys with current public defaults.
4. Derive the immutable resolved view from that case and the actual consumed
   table inventory. Report physical applicability separately from execution
   support and from ledger-validated domain. Preflight a requested backend,
   streaming or adaptive capability before transport/output writes.
5. Runtime preparation loads each required table set and selects native
   arguments once; workers reconstruct resources locally from data-only
   selections. Remote staged resolution must agree on model/data digests
   before checkpoint reuse.
6. Feed unchanged built-ins through the **existing canonical identity
   payloads**. `dataset_identity` hashes a dataset's union of consumed tables;
   `case_content_key` and the API source key hash the case's subset. An additive
   descriptive view alone must not rekey old checkpoints or CAS blobs.

### Transport contract

A transport descriptor declares geometry/layer coverage, material and energy
requirements, required tables, integration/core support and coupling
requirements. Its adapter validates those requirements and returns native
selectors/prepared arguments currently assembled by `_case_elastic_kwargs`,
`_case_inelastic_kwargs` and `_case_radiative_kwargs`.

It does not call Python on each collision or flight, change RNG streams or
reorder native dispatch. Mott/ELSEPA adapters must preserve model codes, energy
coverage checks, per-layer element ordering and missing-table failures. Exact
CUDA and LUT execution have different supported combinations; metadata must
preserve the current dispatch/refusal or documented fallback, rather than
advertise universal CUDA support.

A genuinely new sampler still needs table packing and CPU/CUDA kernel work
under `montecarlo/transport/`, including `cores.py`, `batching.py` and the
`_jit_*` implementations; a new event also needs event records, loss accounting,
secondary/cascade scheduling and fresh-context physics validation. Registry
selection is not sufficient.

### Emission contract

Select batch adapters outside segment and energy-bin loops. A typed batch
context carries the resolved case/view, segment representation (host or shared
device upload), observation direction, native energy grids, layer radiators
and absorption stack, line/continuum electron populations and cutoffs,
coherent/temporal policy, and chunk/resource policy. Prepared table resources
remain runtime-only.

Outputs occupy the existing slots: CXR `spec` (and optional
`spec_coherent`), `spec_characteristic`, and continuum `brem_wide` with its
native `E_grid_brem` and existing interpolation to `brem`. Preserve units,
per-electron normalization, grid/bin quadrature, layer accumulation order,
attenuation, shared uploads and OOM retry behavior. Characteristic emission is
added once through `_spectral_components`; coherent and incoherent CXR are
alternative evaluations of the same source, not two additive channels.

Uncoupled EEDL/BremsLib are alternative continuum producers.
`runner/emission.py::_coupled_brem_from_segments` consumes the transported
radiative state and expected-value estimator; it cannot be summed with a
second uncoupled continuum contribution. A producer requiring coupled rows
must declare that dependency.

Live scoring, full and bounded artifact replay, recompute, directions and
adaptive scoring must consume consistent selections. Streaming declares an
additive reduction over whole electrons; coherent/temporal evaluation retains
`runner/artifacts.py::STREAM_REFUSED_KEYS`. Adaptive capability includes the
per-electron observable, not just a final averaged spectrum.

### Compatibility matrix

This table records constraints to preserve, not new capabilities. Native core
support is still decided by `transport/api.py` and the existing backend owners.

| Route | Preparation/state requirement | Existing scoring contract / restriction |
| --- | --- | --- |
| Legacy Mott | Absent case elastic key; explicit Mott dispatch; SRD 64 table requirement | Historical omitted keys and case/identity payload retained |
| ELSEPA | Tables for every layer element, covering transport energies | Native tables/samplers unchanged; preparation tested with synthetic arrays |
| EEDL fallback | `auto` chooses EEDL when complete BremsLib inventory is unavailable | No BremsLib/radiative keys; fixed resolved choice at scoring |
| Coupled BremsLib | Midpoint integration, resolved BremsLib and hard-photon cutoff | Coupled expected-value continuum only; repair replays the same transport |
| Multilayer | Ordered layer compositions, stopping/elastic tables and radiator orientations | Each layer emits with its composition and absorbs through the whole stack |
| Grooves | Lockstep transport; existing coupling fallback/warning | Preserve vacuum legs and groove attenuation; no new native event support |
| Coherent/temporal CXR | Whole required segment population and bunch metadata | Separate coherent alternative; bounded streaming refused |
| Shell/cascade | Shell mode/cutoff; secondary threshold; pairs require coupled photons; positron mode requires pairs | Preserve track ancestry, pair/annihilation records and energy accounting |
| Adaptive precision | Counter-addressed electron blocks and supported per-electron observables | Extension must supply observable or explicitly refuse before execution |

### Extension tiers and consequences

1. **Alternative implementation in an existing family:** descriptor, preparation
   or batch adapter, domain/capability checks, selected markers/data identity,
   and focused compatibility/physics evidence.
2. **Producer in an existing output slot:** the above plus normalization,
   replay/recompute/directional and adaptive/streaming support or explicit
   refusals. A test-only producer can prove dispatch without new physics.
3. **New native event or additive channel:** separate design and validation.
   Events require scheduler/kernel work; channels require `_spectral_components`,
   checkpoint splitting/reconstruction, results/spatial/observation factors,
   total/plot semantics and artifact/recompute migration.
4. **External plugin ABI/discovery:** deferred. No entry points, arbitrary import
   strings, mutable override registry or plugin-shipping promise.

The first implementation adapts existing physics. It must preserve default
outputs, reduction order, RNG consumption, case omission and cache identities.
The metadata route introduces no per-segment/collision/bin callback or repeated
load/upload. The final implementation gate measures warmed transport/scoring
against the parent on matched hardware (5% median overhead target), with cold
initialization measured separately; this contract-only slice makes no benchmark
or GPU-parity claim.

## Baseline evidence

Fixtures in `tests/data/physics-baseline/` record the pre-adapter parent commit,
seven resolved payload/identity families and small seeded CPU outputs. Their
README explains synthetic data and the distinction between exact compatibility
and physics validation. Checks extend existing `test_case`,
`test_trajectory_capture`, `test_elsepa_elastic`, `test_coupled_radiative_runner`
and `test_pair_production` owners.

Existing tests additionally guard public/deprecated selection, table resolution,
API/sweep source-key parity, table-subset versus dataset-union identity, coupled
repair, coherent limiting cases, cascade energy closure and artifact round trips.
They remain necessary: baseline snapshots cannot establish a physical validity
domain or prove CUDA, remote staging, arbitrary producers or every replay route.

## Alternatives

A second public plan object would duplicate `Physics` and campaign policy.
A mutable global registry would make resolution depend on import order and
undermine reproducibility. Arbitrary callbacks in native loops would expand the
scheduler contract and add overhead. A universal channel dictionary would skip
persistence and observation semantics. Immutable metadata with preparation and
batch adapters addresses the actual seams while leaving these larger decisions
explicitly scoped.
