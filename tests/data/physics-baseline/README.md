# Physics extension compatibility baseline

Captured from unmodified runtime at commit
`7d89582178adddf0e76f639854a5b997a0adf647`, before issue #485 adapter work,
on CPU/NumPy fp64, Python 3.14.4. These are compatibility fixtures, not independent
physics references. Do not regenerate merely to make a refactor pass.

`resolved.json` stores full resolved case mappings, `case_content_key` and
`dataset_identity.parameter_sha256` for legacy Mott, ELSEPA, missing-BremsLib
EEDL fallback, coupled BremsLib, multilayer, coherent and cascade selections.
`tests/helpers/physics_baseline.py::resolved_baseline` defines the inputs:
40 keV silicon, four line/continuum trials, explicit selectors and small grids.
The fallback loader deliberately raises `TableNotFoundError`; the warning and
omitted EEDL/radiative keys are asserted. Table identity uses fixed fake manifest
markers (`a`/`b` repeated 64 times) to avoid depending on installed releases.
The mappings do not claim those fake tables are suitable for transport.

`outputs.json` pins externally meaningful seeded CPU transport counts, total
path length in angstrom, and the first eight flight lengths, positions,
row-end energies (keV), event kinds and electron IDs. Legacy, fallback,
multilayer and coherent cases additionally pin complete small CXR,
characteristic and continuum arrays in the runner's existing spectral units;
coherent uses a point source, an infinite lateral footprint and two physical
bunch electrons so the four-trial pair estimate does not amplify noise.
The fallback has the exact same resolved/output contract as explicit EEDL.
Their seed is the case builder's deterministic seed, recorded in `resolved.json`.
Transport uses the lockstep core for these small cases.

Other output inputs reuse existing test owners:

- ELSEPA: seed 485, four 30 keV Si electrons, 2000 angstrom slab, 5 keV cutoff,
  lockstep. `test_elsepa_elastic::_sr_table` supplies analytic
  screened-Rutherford-shaped ELSEPA-format arrays on the tracked ELSEPA angular
  grid, not a fetched ELSEPA release.
- Coupled: `test_coupled_radiative_runner::_coupled_case` (seed recorded by
  the production case builder), twelve 40 keV Si electrons and a 1 keV photon
  cutoff. `synthetic_si_tables` supplies deliberately inflated synthetic
  BremsLib arrays. Complete native continuum output is pinned.
- Cascade: `test_pair_production::_cascade`, seed 11, six 3 MeV Si electrons,
  100 keV secondary threshold, synthetic boosted BremsLib/pair rates, existing
  pinned shell/stopping data. Pins converted photon/electron energies and
  secondary launch kinds as well as transport. Positrons are recorded, not
  transported in this numeric fixture; the resolved mapping exercises the
  positron selection, and existing `test_positron_cascade` checks its accounting.
  This fixture skips if the existing shell reference is unavailable; no fetch.

Mott output uses the suite's tracked `mott_srd64_synthetic` tables, which reproduce
analytic Joy/Bishop screening. No licensed NIST tables were fetched or added.
Coupled/cascade arrays deliberately increase rare-event rates; their success
says nothing about a physical validity domain.

Output comparisons require identical component keys, array shapes, event counts
and integer IDs. Floating-point outputs use `rtol=1e-12`, `atol=0`: these small
fp64 cases accumulate up to a few thousand contributions, so the allowance is
of the order of thousands of machine epsilons. Only `spec_coherent` uses
`rtol=1e-10`: keV photons over flights of order 1e4-1e5 angstrom accumulate
phases of order 1e5 radians, amplifying fp64 error before interference and
squaring. CPU math/SIMD implementations
can differ in their final bits even with the same seed and reduction order;
disabling NumPy's `X86_V3` dispatch reproduces an exact-comparison failure in
the coherent fixture. A zero absolute tolerance keeps zeros exact and protects
the small spectra. The captured fixtures remain unchanged. Existing
replay/streaming and backend comparisons retain their documented precision
tolerances. For a deliberate numerical/model change, review the
changed expectations and model-generation/validation obligations together;
these snapshots are not a reason to loosen tolerances.

Run only these additions (including the slow cascade fixture):

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --slow \
  tests/montecarlo/test_case.py \
  tests/montecarlo/test_trajectory_capture.py \
  tests/montecarlo/test_elsepa_elastic.py \
  tests/montecarlo/test_coupled_radiative_runner.py \
  tests/montecarlo/test_pair_production.py -k pre_adapter --durations=0
```

Existing supporting evidence remains in the elastic/continuum case-resolution,
source-content-key, xsgen run-identity, coupled repair/layer scoring, coherent
limiting-case, trajectory replay and pair/positron energy-accounting tests.
Hardware/backend parity, staged remote identity and dispatch overhead are later
integration gates; these fixtures do not claim to have measured them.

## Planar channeling family (#506)

`channeling.json` adds one family captured after #506's selection plumbing,
not at the #485 parent: the resolved case mapping and `case_content_key` of a
`Physics(channeling_model="planar", channeling_plane=(1, -1, 0))` scene built by
`tests/helpers/physics_baseline.py::resolved_channeling_baseline` (5 MeV Si,
2 um slab, beam 3 deg from the (110) face normal inside the (1-10) planes, every
selector explicit). There is no `parameter_sha256`: profiles and checkpoints
cannot select channeling before #507. The seven #485 families above are
unchanged by #506, which is the channeling-off compatibility proof; they must not
be regenerated for it.
