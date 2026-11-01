# Select and extend in-tree physics

PyRITE has two internal extension boundaries: prepare an existing native
transport model, or score a batch of trajectories into an existing spectral
family. [ADR-0017](../../adr/0017-in-tree-physics-extension-contract.md) defines
their ownership and compatibility rules. These are private source interfaces;
there is no external plugin ABI, import-string profile selector, mutable global
registry, or per-collision Python callback.

## Select existing implementations

Use `Physics` through the [Python API](../../reference/python-api/core.md), or the existing physics
fields in [profile settings](../profile-settings.md). `Numerics` owns grids,
trial counts, precision and execution controls. Deprecated physics inputs on
`Numerics` still lower through the existing compatibility path.

| Choice | Selection and execution |
| --- | --- |
| Elastic | `elastic_model="elsepa"` uses per-element tabulated scattering; `"mott"` uses SRD 64. Exact native cores implement ELSEPA; enabling a screening-parameter LUT does not make it an ELSEPA LUT. |
| Continuum | `bremsstrahlung_model="eedl"` or `"bremslib"` selects one continuum implementation. `"auto"` checks every layer element, choosing BremsLib only with complete data and retaining the documented EEDL warning/fallback. |
| Inelastic | Continuous loss or supported shell soft/hard transport; existing energy integration, shell data and secondary-threshold constraints apply. |
| Coupled radiation | Supported coupled BremsLib scoring consumes the transported radiative state and its expected-value estimator. It replaces uncoupled continuum scoring; the two are never independent additive sources. |
| CXR and characteristic | CXR produces incoherent `spec` and optionally the alternative `spec_coherent`. Characteristic emission remains separate and is added once by `_spectral_components`. |

`Physics.validate_transport` validates combinations; campaign lowering resolves
material-dependent choices before serialization or execution. A resolved `Case`
has historical omission rules: missing elastic means Mott, missing continuum
means EEDL, missing radiative keys mean uncoupled, and a missing `channeling`
record means amorphous transport (channeling off; #506). The `channeling` record
and its `CHANNELING_MODEL` identity marker join a case and its content key only
when `Physics.channeling_model="planar"`, so channeling-off payloads, keys and
RNG streams are unchanged. Profiles and checkpoints cannot select it until the
persisted channel (#507). The direct transport API's
ELSEPA default does not change an old case. Do not resolve `auto` again at scoring
time or fill absent keys while reading a checkpoint.

## Boundary, units, shapes and normalization

Transport preparation selects native arguments and borrowed table arrays through
`runner/case_tables.py`. The fixed elastic map demonstrates Mott/ELSEPA
alternatives; native samplers, model codes, RNG addressing and event ordering
remain in transport. Follow the [preparation reference](physics-preparation-adapters.md)
for the exact CPU/CUDA/LUT and table-coverage responsibilities.

`EmissionBatch` borrows one already staged segment mapping. Segment positions
and lengths are angstrom; electron energies and cutoffs are keV; photon grids
are eV. Directions are sample-frame unit vectors. `abs_layers` describes the
complete attenuation stack; each material layer scores with its own composition
and radiator orientation. Producers receive existing population limits, table
caches, temporal/coherent accumulators and resource policies by reference.
They must retain their scorer's documented spectral units, grid/bin quadrature,
solid-angle convention and normalization by the requested electron population,
including electrons with zero contribution. Do not normalize by the number of
segments or only electrons present in a block.

| Output | Required shape and meaning |
| --- | --- |
| `spec`, `spec_coherent` | Native CXR grid shape; alternative incoherent/coherent evaluations of the same source. |
| `spec_characteristic` | CXR grid shape; separate atomic characteristic component. |
| `brem_wide` | Native continuum grid shape; interpolation to `brem` remains in the caller. |
| Adaptive continuum observable | When `electron_band_weights` is supplied, one band integral per electron: `(electron_limit,)`, including zeros. |
| Adaptive CXR observable | Populate `truncation_audit["electron_mass"]` with one value per electron, preserving the band and population requested by the stopping rule. |

Adapters are selected outside segment/bin loops. Ordinary scoring and full
artifact replay share the spectrum owner; bounded replay, line recompute and
adaptive scoring use the same selected adapters. Directional scoring reuses one
transport and invokes the spectrum owner at each direction. Continuum recompute
retains its existing wrapper route. CPU workers reconstruct process-local
adapters from data-only case selections; callables, arrays and device handles
are not case fields.

Streaming support declares `"whole-electron blocks"` and requires an additive
reduction with a fixed population denominator. CXR also declares `"incoherent"`
and `"no temporal profile"`; coherent/temporal bounded replay remains refused.
Empty or unknown streaming requirements fail preflight. Adaptive declarations
must identify the observables the implementation actually provides; final
averaged spectra alone cannot support a stopping rule. Backend capability
metadata records implementation support, while native owners perform admission
and resource checks. It is not a device probe or a validated physical domain.
Keep the caller's one shared upload, chunk/OOM retry path and reduction order.

## Complete example in an existing slot

This CPU diagnostic replaces characteristic emission with zeros. It exercises
registration, selection, shape and streaming admission without proposing a
physical model. Zero is additive and has the same zero value under the existing
per-electron normalization. It needs no tables, supports no precision observable,
and must not be used as a physical production selection.

The following block runs as written through the project environment. The
regression in `tests/montecarlo/spectrum/test_emission_adapters.py` executes this exact
block and checks that its selection reaches ordinary scoring.

<!-- executable-example: existing-slot -->
```python
from dataclasses import replace

import numpy as np

from pyrite._physics_contract import ModelDescriptor
from pyrite.montecarlo.runner.emission_adapters import (
    BUILTIN_EMISSION,
    EmissionAdapter,
    EmissionBatch,
    EmissionOutput,
    select_emission,
)


def zero_characteristic(batch: EmissionBatch) -> EmissionOutput:
    return EmissionOutput("spec_characteristic", np.zeros(batch.energy_eV.shape))


descriptor = ModelDescriptor(
    family="emission.characteristic",
    model="example-zero",
    owner="example",
    implementations=("cpu",),
    output_slots=("spec_characteristic",),
    streaming_requirements=("whole-electron blocks",),
)
adapter = EmissionAdapter(descriptor, zero_characteristic)
catalog = BUILTIN_EMISSION.with_adapter(adapter)
selection = replace(select_emission({}), characteristic=catalog.get(descriptor.id))
selection.preflight({}, streaming=True)

batch = EmissionBatch(
    segments={"Ne": 2, "L_ang": np.empty(0)},
    energy_eV=np.array([1500.0, 2000.0, 2500.0]),
    case={},
    n_hat=np.array([1.0, 0.0, 0.0]),
    electron_limit=2,
)
output = selection.characteristic.evaluate(batch, "spec_characteristic")
np.testing.assert_array_equal(output.values, np.zeros(3))
assert all(entry.descriptor.id != descriptor.id for entry in BUILTIN_EMISSION.adapters)
```

The local `selection` is explicit; this block does not install a production
override or add a public selector. For a real in-tree producer, add its descriptor
and adapter to the explicit catalogs, connect its selection through the owning
case/facade policy, and carry only data selectors/parameters into the case.
Choose the adapter before transport or writes and reject unsupported routes.
Test every consumer it supports, including empty blocks and population limits.

## Add an alternative implementation

1. Choose an existing process/source family and stable family-qualified ID.
   Declare required datasets, geometry/layer and material/energy restrictions,
   exact versus LUT/core support, coupling and reduction/observable capabilities.
   A catalog rejects duplicate IDs/routes; registration does not certify a
   ledger-validated domain.
2. Prepare data once through the existing table owner, with element order,
   coverage, units and manifest digests preserved. An alternative native sampler
   still needs compatible table packing and kernel implementations; a batch
   producer must use the established output slot and normalization.
3. Extend the existing public validation and campaign `auto` resolution only
   where needed. Preserve explicit-failure versus documented-fallback behavior,
   historical omissions and deprecated lowering. Serialize no runtime resource.
4. For a numerical/model/data change, update the selected generation marker or
   table tag in its existing owner and ensure both dataset and per-case identities
   invalidate. A descriptive adapter revision or an unused catalog entry must
   not rekey results. Keep dataset table unions separate from per-case subsets.
5. Exercise live/replay/recompute/directional/worker paths, and supported
   streaming/adaptive reductions. Add refusals for unsupported capabilities
   before work begins. Verify staged remote resolution against submission.
6. Record the equation, assumptions, units, limiting case, `Validation: <id>`
   and ledger row for changed physical claims. Obtain fresh-context validation
   to the requested `rederived`/`anchored` status; only a human signs off in #277.
   Seeded compatibility fixtures establish preservation, not physical validity.

Resolved descriptor/model/parameter/table snapshots appear in API provenance
and new trajectory captures. [Storage identity](../storage/dataset-identity-and-storage.md)
retains its canonical hash owners; [trajectory replay](../storage/trajectory-artifacts.md#compatibility)
checks new snapshots before scoring/writes while preserving legacy reading.

## Add a native event or additive channel

A new event needs a separate design for competition/scheduling, CPU and CUDA
integration, RNG streams, event records, soft/hard loss partition, secondary
species and energy accounting. The preparation reference lists the owning
files; metadata registration cannot implement the event.

A new additive spectral family needs `_spectral_components` semantics,
checkpoint splitting/reconstruction, result arrays/grids/units, spatial and
observation factors, directional stacking, totals/plots, replay and recompute
migration. Reusing an existing slot avoids that migration only when its actual
meaning and component accounting match. Arbitrary output dictionaries and
external discovery/shipping remain outside this contract.

See [dispatch overhead evidence](physics-adapter-overhead.md) for the measured
CPU scope and reproduction command. GPU, larger grids and heavy sweeps require
the remote compute workflow and matched hardware measurements.
