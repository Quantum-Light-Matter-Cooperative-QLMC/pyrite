# Physics extension compatibility baseline

Captured from unmodified runtime at commit
`7d89582178adddf0e76f639854a5b997a0adf647`, before issue #485 adapter work,
on CPU/NumPy fp64, Python 3.14.4. These are compatibility fixtures, not independent
physics references. Do not regenerate merely to make a refactor pass.

## CBS v4 generation (#513)

The deliberate static `Z - f0(g)` CBS potential correction in `e7c66c6f`
changed `CXR_ESCAPE_MODEL` from v3 to v4. The resolved-case and channeling
hashes, plus the legacy/fallback/multilayer/coherent line spectra, were
re-pinned against that generation at `c4ab4179` (CPU/NumPy fp64, Python
3.14.4). The acquisition-parser refactor in #565 produces identical results
to its parent; these failures were inherited from the model change.

Before re-pinning, restoring only `materials/crystal.py` and
`montecarlo/spectrum/cxr/_setup.py` from the parent of `e7c66c6f` made all
15 failing transport-runner/core compatibility checks pass against the
original expectations. Under v4, the resolved case payloads, seeded transport
signatures, characteristic spectra, and continuum spectra retain their
original expectations. Only the CXR-containing `spec`/`spec_coherent` arrays
and generation-dependent identity hashes changed. Other output families and
all comparison tolerances remain unchanged.
The coherent-roundoff acceptance test applies representative relative errors
from the recorded CI sample to the current spectrum, instead of embedding
absolute v3 intensities.

## Line kinematics v5 generation (#439)

Forming `gamma`, the Doppler factor `1 - v.n_hat` and the refractive
decrement `delta = 1 - Re n` cancellation-free changed `CXR_ESCAPE_MODEL`
from v4 to v5. Only the generation-dependent `case_content_key` and
`parameter_sha256` hashes in `resolved.json` and `channeling.json` were
re-pinned. On CPU/NumPy fp64 the line spectra move at rounding level only, so
every recorded output keeps its expectation and tolerance.

## Always-on Ter–Mikaelian coupling (#439)

Applying the dielectric factor `k²/(k² + (γħω_p)²)` to every coupled BremsLib
run (not only runs reaching 30 MeV) changed `BREMSSTRAHLUNG_BREMSLIB_MODEL` to
`…/tm-coupled-v2`. Re-pinned: the `coupled` and `cascade` identity hashes in
`resolved.json`, and the `coupled` output in `outputs.json`. With the 1 keV
photon cutoff and `ħω_p ≈ 31 eV`, Si soft radiative stopping falls about 5 %
at 10–40 keV and the hard rate about 2e-4; on the 1e4-inflated synthetic
tables the seeded tracks diverge (rows 1355 → 1613, path +3.4 %, continuum
sum +1.6 %). Other outputs and tolerances are unchanged.

## Electron–electron bremsstrahlung (#439)

Staging every BremsLib table with five `k/T1` sub-nodes in `(0, 0.1)` and
the Seltzer–Berger (1985) factor `1 + η/Z` changed
`BREMSSTRAHLUNG_BREMSLIB_MODEL` to `…/lin-x18-…-v2/tm-coupled-v2/ee-sb85-v1`.
Re-pinned: the `coupled` and `cascade` identity hashes in `resolved.json`,
and the `coupled` and `cascade` outputs in `outputs.json`. With the factor
switched off, the sub-nodes alone reproduce the previous `coupled` transport
bit for bit and its continuum to 4e-16. At 5–200 keV the factor is
1.002–1.02, but on the 1e4-inflated synthetic tables the seeded tracks
diverge (`coupled` rows 1613 → 2272, path +25 %, continuum sum +3.1 %;
`cascade` rows 596 → 542). Other outputs and tolerances are unchanged.

## Channeling content key after rebase (#439)

After rebasing onto `main` (#536/#541), `channeling.json`'s
`case_content_key` was re-pinned for the v5 line-kinematics marker on the
rebased generation. Its resolved case payload is identical to `main`'s.

## Corrected SBETHE v2 generation (2026-10-24)

Commit `1f22f89c` weights compound shell corrections by electron fraction and
advances `STOPPING_MODEL` to `sbethe-corrected-v2`. Restoring only the v1
marker reproduces every prior hash in `resolved.json` and `channeling.json`.
All resolved case payloads remain identical. The dependent acceptance and
fixed-count identity pins advance with the same marker; installed BremsLib
selection is kept fixed for the acceptance fixture.

Only `outputs.json["multilayer"]` changes numerically: substituting the prior
SiO₂ stopping columns (cross section and eV/angstrom) reproduces the complete
old transport and spectra at the existing fp64 tolerance. With v2, the four
seeded electrons still produce 501 rows, one backscatter and three transmissions;
total path changes from 156417.592494 to 156372.061811 angstrom (−0.0291%).
The downstream line, characteristic and continuum spectra change with those
tracks. Other output families and all tolerances remain unchanged. This
isolates the inherited corrected-stopping change; it is not independent
physics validation.

## Exact explicit brem grids (#815)

Commit `00f659ea` keeps an explicit uniform `EnergyBins.brem` grid exact
instead of stretching its stop to `E0 + step`. Every fixture requests
`brem=[5000, 10000, 15000]` eV, so `E_grid_brem` changes from
`[5000, E0 + 5000, 5000]` (45 keV for the 40 keV families, 5.005 MeV for
channeling) to `[5000, 20000, 5000]`. Every resolved case payload is
otherwise unchanged. Re-pinned (2026-10-30, #819, CPU/NumPy fp64, Python
3.14): `E_grid_brem`, `case_content_key` and `parameter_sha256` in
`resolved.json`; `E_grid_brem` and `case_content_key` in `channeling.json`;
and `brem_wide` in `outputs.json` for legacy, eedl-fallback, multilayer and
coherent.

Each `brem_wide` shrinks from 8 nodes (5--40 keV) to the 3 requested nodes.
The new values equal the first three old nodes to at most 8.4e-16 relative,
as photon-node independence requires (#819). Transport signatures and the
line, characteristic, coherent and detected `brem` arrays keep their
expectations. All tolerances are unchanged.

## Fixture inputs

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

## Photon-transport mode-off compatibility (#341)

The fixtures above retain their values and tolerances. Before checking the
seven resolved-case/content-key/dataset-identity families, the tests execute
the experimental weighted photon source, vacuum queue and aperture tally.
The ELSEPA and pair-cascade output goldens also run after that diagnostic.

For the legacy/EEDL-fallback, multilayer, coherent and coupled BremsLib output
families, `tests/helpers/photon_mode_off.py` additionally runs real production
`run_case` transport and scoring on both sides of the diagnostic, with the
same fixed electron seed. Photon-MC entry points are replaced with failure
guards only during the production runs; electron transport and emission are
not stubbed. Every output array must keep its shape, dtype and bytes; output
metadata, transport signatures and case-content identity must match exactly.
Expected output arrays are copied before the diagnostic to expose aliasing
or mutation. The historical golden comparison still uses its original
cross-platform roundoff tolerances; it is not regenerated or tightened to a
platform-specific byte snapshot.

The coupled full-run check is explicitly slow because the first invocation
can spend more than five seconds loading/compiling transport inputs. Use the
focused `--slow ... -k pre_adapter` command above to include it. This is CPU
compatibility evidence for the existing default path, not validation of
photon interactions, charged feedback, GPU photon transport or full showers.

## Planar channeling family (#506)

`channeling.json` adds one family captured after #506's selection plumbing,
not at the #485 parent: the resolved case mapping and `case_content_key` of a
`Physics(channeling_model="planar", channeling_plane=(1, -1, 0))` scene built by
`tests/helpers/physics_baseline.py::resolved_channeling_baseline` (5 MeV Si,
2 um slab, beam 3 deg from the (110) face normal inside the (1-10) planes, every
selector explicit). There is no `parameter_sha256`: profiles and checkpoints
cannot select channeling before #507. `outputs.json["channeling"]` pins the
same scene on a small zone grid (`channeling_output_baseline`: 16 electrons,
`H = 12`, four Bloch cells, `dz = 500 Å`, per-electron core, synthetic Mott
tables): the joined zone and native transport signature, the zone-row count,
the sampled fates and the small line, characteristic and continuum spectra.
The seven #485 families above are
unchanged by #506, which is the channeling-off compatibility proof; they must not
be regenerated for it.

For #507, only `channeling.json`'s content key advanced with the enabled
`CHANNELING_MODEL` v2 marker. It advances again to v3 for #523's diamond
electronic-kernel input. Its resolved case payload and all seven
channeling-off family snapshots remain unchanged. The new CR component is
covered by public-producer/checkpoint/totals tests; the PXR/CBS, characteristic
and continuum arrays retain their separate meanings.
