# In-tree transport preparation adapters

The private contract in [ADR-0017](../../adr/0017-in-tree-physics-extension-contract.md)
now has an elastic preparation proof at
`src/pyrite/montecarlo/runner/case_tables.py`. This page describes that current
boundary. [Emission batch adapters](emission-batch-adapters.md) have their own
contract. The [extension guide](extending-physics.md) connects preparation,
execution, replay, identity and validation responsibilities.

## Current elastic path

`_case_elastic_kwargs` selects a fixed preparation callable using the resolved
`transport.elastic/mott` or `transport.elastic/elsepa` descriptor ID. An absent
case key selects Mott, regardless of the direct transport API's ELSEPA default.
Unresolved `auto` and unknown case model strings are rejected before loading
ELSEPA tables. The direct API's developer-only `sr` model is not a Case selector.

The adapters prepare the existing native arguments. Mott explicitly names its
model; SRD 64 table lookup, coverage checks and packing remain in
`transport/scattering.py` and `layer_tables.py`. ELSEPA uses the existing
`xsgen/elsepa/catalog.py::resolve_layer_tables` per ordered layer composition and
passes each entry's arrays by reference. It neither copies/reorders arrays nor
selects another material catalog. Material authoring still resolves the selected
catalog into case layer compositions before this boundary; custom objects follow
the same per-element table requirements as bundled materials.

`_case_elastic_entries`, `_case_elastic_table_records`, `_case_elastic_kwargs`
and the runner's imported kwargs seam retain their roles. Table-record flattening
and identity markers are unchanged. Atomic-electron deflection remains the same
optional kwargs addition. Missing ELSEPA/SRD64 and energy-coverage failures come
from their existing owners; adapters do not swallow them or silently choose a
replacement model.

The runtime adapter map is immutable and process-local. It contains callables;
it is not a serialized case or resolved view. Selection occurs once per
preparation call, outside native collision/flight/segment/bin loops. Existing
block and retry preparation call patterns remain unchanged. No full resolved
view, stopping/continuum inventory, backend probing or extra table loads are
introduced merely to select an elastic adapter.

## Preserve native route distinctions

`transport/api.py` still owns model codes (`sr=0`, `mott=1`, `elsepa=2`), native
core dispatch and LUT admission. The descriptor catalog records those existing
limitations; it does not add a kernel or replace runtime validation.

| Selection | Native behavior to preserve |
| --- | --- |
| Mott, continuous loss, enabled LUT, ungrooved | Existing CPU/CUDA LUT routes remain eligible. |
| ELSEPA | Uses exact cores even with an enabled LUT configuration; an SR screening-parameter LUT cannot represent its tabulated angular distribution. |
| Coupled BremsLib | Bypasses LUT construction and uses exact cores; hard-photon rows are completed on the host. |
| Shell soft/hard with Mott on direct CUDA API and enabled LUT | Fails with the existing CUDA-LUT refusal before constructing the device kernel. |
| Shell case through runner on CUDA | Runner disables the unsupported CUDA LUT and selects exact CUDA. This is the existing production fallback, not a new adapter fallback. |
| Grooves | Lockstep route; explicit unsupported shell/coupled requests retain their existing refusal or campaign warning/fallback contract. |
| Pair/positron cascade | Requires shell transport, secondary threshold and coupled hard photons. Positron mode has no Mott implementation and retains species-specific tables and host annihilation completion. |

`tests/montecarlo/test_elsepa_elastic.py` exercises CUDA factory selection with
CPU stubs that stop at the factory. `test_shell_soft_hard_config.py` distinguishes
the direct native refusal from runner exact-core selection;
`test_coupled_radiative_runner.py` proves coupled selection bypasses the LUT
factory. These tests execute host preparation, not GPU transport or parity.
The seeded ELSEPA adapter path also matches the pre-adapter CPU fixture exactly.

## Files a new sampler or event still needs

An alternative table producer using an existing sampler can prepare the same
array contract. A genuinely different native sampler needs more than descriptor
registration. Review these specific owners and change those its semantics affect:

| Concern | Current files under `src/pyrite/` |
| --- | --- |
| Public selector/compatibility and serialized omissions | `_numerics.py`; `campaign/sweep.py`; `montecarlo/case.py` |
| Metadata and preparation | `_physics_contract.py`; `montecarlo/runner/case_tables.py`; the relevant `xsgen/*/catalog.py` and table parser/store |
| Native model code, table contracts and coverage | `montecarlo/transport/api.py`, `arguments.py`, `layer_tables.py`, `scattering.py` (`pack_elsepa_tables` for ELSEPA) |
| CPU samplers/flight integration | `montecarlo/transport/cores.py`, `batching.py` and the relevant sampler helpers |
| Exact CUDA sampler/integration | `montecarlo/transport/_jit_kernel.py`, `_jit_device.py`, `_jit_launch.py` and any event-specific device helpers |
| LUT representation and implementations, if meaningful for the sampler | `montecarlo/transport/lut.py`, `cores.py`, `batching.py`, `_jit_lut_kernel.py` |
| New event records, units and energy-loss accounting | `montecarlo/transport/events.py`; `hard_inelastic.py` or `hard_radiative.py`; affected `cores.py` and `_jit_*` event helpers |
| Secondary/coupled scheduling and species accounting | `montecarlo/transport/secondaries.py`, `pair_production.py`, `annihilation.py` and affected table/species preparation |
| Persisted transport fields/replay compatibility | `montecarlo/trajectories.py`, `runner/artifacts.py`, `trajectory_export.py` and affected checkpoint recompute/validation owners |
| Reproducible identities and validation evidence | Existing `*_MODEL` owners, `campaign/profiles.py`, `runner/case_tables.py`; relevant physics derivation, validation ledger and regression owners |

Preserve RNG stream addressing and existing event-order/reduction semantics in
a transparent adapter move. Changing a sampler, equation, estimator or numerical
output is physical-model work: document the affected claim and model marker and
obtain the required fresh-context validation. Metadata registration and matching
fixtures are compatibility evidence, not independent physical validation or
human sign-off. A new persistent additive spectral channel requires its own
component/results/checkpoint/observation migration; this transport proof supplies
no external plugin ABI or generic event callback mechanism.
