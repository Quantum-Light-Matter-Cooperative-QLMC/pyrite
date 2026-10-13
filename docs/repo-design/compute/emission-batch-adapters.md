# Batch emission adapters in existing components

[ADR-0017](../../adr/0017-in-tree-physics-extension-contract.md) separates native
transport preparation from trajectory-based emission. The live scoring proof
now lives in `src/pyrite/montecarlo/runner/emission_adapters.py` and the existing
`runner/emission.py` wrappers.

## Runtime contract

`EmissionBatch` borrows already staged segments, a native photon grid in eV,
the resolved case, observation direction, layer/absorption geometry, population,
coherent policy and runtime cache/audit/temporal accumulators. Segment positions
remain angstrom and electron energies remain keV. Scorers retain their existing
units, per-electron normalization and cutoff interpretation. The frozen context
protects its field assignments; arrays and accumulators are deliberately shared
runtime resources. It is not a serialized manifest or a second public plan.

`EmissionOutput` holds one declared existing slot and its native-grid array.
Adapters validate slot and shape without copying or converting values. The
internal per-electron continuum observable uses the selected electron-population
shape when band weights are supplied. No arbitrary persistent channel is added.

The immutable `EmissionCatalog` stores explicit in-tree descriptor/route entries;
duplicate keys fail. Adding a test producer returns a new local catalog through
`with_adapter`, leaving production registration unchanged. Local selection can
substitute an existing-slot producer for tests; it is not a public selector,
profile syntax, global override or external plugin ABI.

## Existing producers and compatibility seams

| Entry | Existing implementation | Output / dependency |
| --- | --- | --- |
| CXR | `runner._cxr_for_segments` | Incoherent `spec`, and optional alternative `spec_coherent`; shared reflection tables/audits |
| Characteristic | `runner._characteristic_from_segments` | Separate `spec_characteristic`, scored once on the continuum electron population |
| EEDL continuum, uncoupled | `emission._uncoupled_brem_from_segments` and `runner.mc_brem_spectrum` | `brem_wide` on native continuum grid, resolved EEDL |
| BremsLib continuum, uncoupled | Same per-layer reduction with prepared BremsLib kwargs | `brem_wide`, tables loaded once for the complete layer inventory |
| BremsLib continuum, coupled | `emission._coupled_brem_from_segments` | Existing expected-value estimator over coupled radiative state; excludes uncoupled continuum evaluation |

Live scoring selects adapters outside segment/bin loops, then evaluates in the
existing order: incoherent CXR, optional coherent CXR, truncation/coherent audit
checks, characteristic, continuum and native-grid interpolation to `brem`.
All producers receive the same single segment upload. CXR uses `Ne_lines`;
characteristic and continuum use `Ne_brem`, with their existing cutoffs.
Coherent/incoherent CXR remain alternatives, and characteristic is added once by
`_spectral_components`. No sums or reduction order change inside the estimators.

Default adapter scores resolve the existing runner façade wrappers at call time,
preserving import/monkeypatch seams and component OOM tagging. The continuum
wrapper constructs the same batch context and selects its native `from_segments`
producer; this keeps ordinary replay/recompute wrapper calls compatible without
changing signatures. It admits the same chunk and passes the same layer ordering,
full-stack attenuation and population kwargs as before. Descriptor selection
uses case model strings; it neither constructs a full resolved view nor loads a
whole identity/table inventory merely to dispatch. EEDL loads no BremsLib tables.

`tests/montecarlo/test_emission_adapters.py` registers a test-only constant CXR
producer in a local catalog and exercises the existing `spec` slot through live
scoring without a new branch in `_spectrum_case_impl`. It also checks shared
staging, different populations, native interpolation, CXR accumulator references,
per-layer continuum preparation, coupling refusals and closed output slots.
Seeded pre-adapter fixtures and existing coherent/characteristic/temporal,
continuum, staging and OOM regressions remain compatibility gates.

## Remaining integration boundary

[Issue #485](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/485)
still owns adapter selection across full/bounded artifact replay, recompute,
directional and adaptive dispatch. Existing wrappers continue to work, but
this slice does not claim that a newly registered test producer reaches every
one of those routes. Preserve coherent/temporal streaming refusals, whole-electron
streaming reductions and adaptive observable support when integrating them.
Identity/provenance/remote shipping and final overhead/hardware gates also remain
separate work. These adapters introduce no new physics-validity claim and change
no equations, model-generation markers or saved component schema.
