# Equal-precision benchmark protocol and matrix freeze

This page records the repo-local freeze stage of
[#510](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/510):
the full parameter matrix, a table-coverage audit, a physics-capability audit
across comparators, and the engine-neutral comparison schema. It runs no
transport and no cross-engine comparison, so it supports no speed or accuracy
claim. Measured pilot results stay in the
[adapter pilot report](equal-precision-pilot.md); comparator access and control
research is in the [comparator feasibility notes](../../research/beam-transport/equal-precision-comparators.md).

## Frozen matrix definition

[`matrix-spec.json`](../../../checks/equal_precision/matrix-spec.json) is the
hand-edited definition; [`matrix.json`](../../../checks/equal_precision/matrix.json)
is generated from it by `build_matrix.py` and records the spec SHA-256, every
resolved case, observable applicability and a per-axis coverage table. A test
fails if the committed `matrix.json` no longer regenerates byte-for-byte.

- **Core:** Si and W × 100 keV, 800 keV, 5 MeV × `t/R` = 0.03, 0.3, 3, with
  `t/R` capped at 1 for 5 MeV (`t_over_r_cap_by_energy_keV`), giving 18
  scenes at normal incidence with a 10 × 10 mm prism, pencil monoenergetic beam
  and the pilot's numerical controls (10 keV electron threshold, 1 keV hard
  photon threshold, photons scored at ≥10 keV in 10 keV bins).
- **Anchors:** the six retained adapter-pilot scenes, with their absolute
  dimensions and an `anchor_of` link to the pilot manifest id. No `t/R` is
  claimed for them.
- **Sparse extensions (20):** 45° and 75° incidence at Si 800 keV `t/R` 0.3;
  angle×thickness checks at W 800 keV 75° and `t/R` 0.03 and 3; diamond (low Z)
  and SiO₂ (compound) at 800 keV `t/R` 0.3; one-factor beam phase space at Si
  800 keV `t/R` 0.3 (100 µm RMS spot, 5 mrad RMS divergence, 1% RMS energy
  spread); a 1 × 1 mm footprint at `t/R` 3 for Si and W; five detector cones
  (90°/5° for Si and W, 0°/1°, 0°/15°, 135°/5°); straggling on for Si and W;
  and explicit-secondary arms for Si and W.

Case ids are built only from material, energy, `t/R` or absolute thickness,
incidence and variant, for example `tungsten-800kev-tr3-75deg-oblique`.
Energy×thickness interactions are covered by the core product. Oblique cases
keep the surface-normal thickness of the matching normal case; their longer
path length is reported, not compensated.

### Thickness from one pinned range source

`t/R` is converted to an absolute surface-normal thickness exactly once, by
log-log interpolation of one pinned CSDA-range tabulation per material, and
every engine then receives that same thickness. The source is the NIST ESTAR
CSDA range (ICRU Report 37 methods), downloaded on 2026-10-22 with all seven
columns, space-delimited, on the default 10 keV–1 GeV grid. The text files are
committed under `checks/reference/estar/` (provenance in its `README.md`)
and pinned in `matrix-spec.json`, so all 37 range-scaled cases are `frozen`.
Nothing is estimated from PyRITE's or any other engine's stopping model.
Rebuilding needs no arguments:

```bash
uv run python checks/equal_precision/build_matrix.py
```

`matrix.json` records the repository path and SHA-256 of each input file and of
its canonical two-column CSV. The parser rejects any row without exactly seven
columns, because a table missing a stopping-power column would shift the CSDA
range out of column 5. ESTAR's densities and density-effect treatment differ
slightly from the matrix densities; the range in g/cm² is taken as tabulated
and that difference is a declared approximation, not a per-engine adjustment.
ESTAR has no diamond entry, so `diamond` uses amorphous carbon (2.0 g/cm³), the
densest carbon entry; graphite (1.7 g/cm³) gives ranges 0.5–0.6% shorter
between 0.1 and 5 MeV.

At 5 MeV, `t/R` = 3 would make the silicon slab 38.5 mm thick, so the
stopping-dominated point is capped at `t/R` = 1. That keeps essentially no
transmission, but `silicon-5000kev-tr1-0deg` is still 12.8 mm thick, deeper than
the 10 mm lateral prism, so side exits remain a large share of its terminal
fractions. `tungsten-100kev-tr0.03-0deg` is 0.46 µm thick.

### Observable applicability

`observable_applicability` marks each observable per case. Terminal fractions
and photon bands are headline gates; any band starting at or above the beam
endpoint is excluded (so `photons_ge100kev` is excluded at 100 keV). Transmitted
energy moments are diagnostics when `t/R ≥ 1`, where transmission is expected
to be unresolved, and side exits are headline only for the 1 mm footprint.
Rare-bin absolute floors and minimum effective counts are still uncalibrated.

## Table-coverage audit

`audit_coverage.py` resolved the installed released tables
(`pyrite tables fetch elsepa sbethe-tables bremslib`) for every matrix material
without running transport. The
[coverage record](../check-records/equal-precision/coverage-audit.json) keys
each case to the exact table digests.

| Material | ELSEPA (keV) | SBETHE (keV) | BremsLib (keV) | Released SBETHE table |
| --- | --- | --- | --- | --- |
| Si (catalog crystal) | 0.1–100000 | 1–10⁶ | 0.01–30000 | yes |
| W (catalog crystal, 19.261 g/cm³) | 0.1–100000 | 1–10⁶ | 0.01–30000 | yes (`tables-sbethe-3`) |
| C (diamond) | 0.1–100000 | 1–10⁶ | 0.01–30000 | yes |
| SiO₂ (medium) | 0.1–100000 | 1–10⁶ | 0.01–30000 | yes |

All 44 cases are covered by released tables. Every 5 MeV case lies inside all
three table domains, so the pilot manifest's 5 MeV deferral no longer has a
table-coverage reason; the physical validity of each model at 5 MeV is a
separate question this audit does not answer.

Tungsten moved to the catalog `tungsten` crystal (19.261 g/cm³) on 2026-10-22,
and its SBETHE table was published in `tables-sbethe-3` and
`tables-sbethe-positron-3`, so every W case reproduces from released inputs.
The retained pilot (`manifest.json`) used anonymous W at 19.3 g/cm³, matching
Geant4's `G4_W`, with a locally generated table; its evidence is unchanged. W
anchors rerun from this matrix use the catalog density, so a Geant4 arm must set
its tungsten density to match, as already done for silicon. The 0.2% density
change alters W thicknesses derived from `t/R` by the same fraction.

No table gap remains. The five detector cases were gaps only while no adapter
called the production weighted per-segment scorer; the
[detector arm](#pyrite-detector-arm) now does. Each audit entry also lists
`pyrite_arms`, the adapter arms that can run the case, and
`pyrite_unsupported`, the scene features the adapter cannot yet express. These
are adapter limits, not table gaps: the finite spot, divergence and energy-spread
beams and the two explicit-secondary arms have no PyRITE arm yet, and the adapter
writes an explicit `unsupported-physics` record for them.

## Physics-capability audit

What each arm can express today. "Match" means the same physical effect can be
switched on or off, not that the models agree.

| Switch | PyRITE (`simulate_trajectories`) | Geant4 TestEm5 pilot | PenRed / PENELOPE | EGSnrc |
| --- | --- | --- | --- | --- |
| Elastic | ELSEPA flights, every collision resolved | `empenelope` multiple scattering; `emstandardSS` DPWA single scattering | Class II mixed: hard events above `C1`/`C2`-set mean free path, soft multiple scattering between | Condensed history (PRESTA-II), single scattering only near boundaries with the EXACT crossing algorithm |
| Atomic-electron deflection | `kawrakow` rate `Z(Z+ξ)` | inside model | inelastic angular deflection in model | inside model |
| Stopping | SBETHE tables, continuous | Penelope ionisation | PENELOPE GOS model | collision stopping with `AE` threshold |
| Straggling | off by default; Urban `straggling=True` | fluctuations disabled in pilot | **always on**: soft losses sampled with correct mean and variance | none below `AE`/`AP` |
| Radiative | BremsLib soft/hard at 1 keV | Penelope brems; photon cut via range cut | soft/hard at `WCR` | NRC or BH brems, `AP` threshold |
| Delta rays | opt-in `shell-soft-hard` + `secondary_threshold_eV` | suppressed by 10 mm e⁻ cut | suppressed by `WCC` > E₀ and `EABS` | `AE`/`ECUT` |
| Relaxation | not in transport (separate characteristic path) | fluorescence off | optional | optional |
| Photon interactions | none in source tally; straight-ray escape in scorers | secondaries killed | `EABS(γ)` > E₀ absorbs at creation | `PCUT` > E₀ |
| PXR/CBS | excluded; audited by call tracing | absent | absent | absent |

Consequences for matching:

1. The PyRITE core's `straggling=False` arm has no PENELOPE-family
   counterpart, because PENELOPE's mixed scheme always samples soft-loss
   fluctuations. A PenRed comparison needs the PyRITE straggling-on variant as
   its partner, and that variant matches PENELOPE's straggling only in its first
   two moments. EGSnrc is the engine closest to a continuous-loss arm.
2. Suppressing delta rays by raising `WCC` above E₀ moves all inelastic loss
   into the soft channel, which changes PENELOPE's soft step control. It is a
   declared configuration, not a physics-equivalent switch.
3. Production cuts in Geant4 are range cuts converted per material; PENELOPE's
   `EABS`, `C1`, `C2`, `WCC`, `WCR` and `DSMAX`, and EGSnrc's `ECUT`, `PCUT`,
   `AE`, `AP` and `ESTEPE`, are recorded verbatim per case. None is assumed
   equal to PyRITE's `max_dE_frac` or hard-photon cutoff.

## Common comparison schema

`pyrite.perf.comparison` defines `pyrite.equal-precision.comparison.v1`. An
adapter supplies one score per incident primary per observable; zero-score
primaries are kept. `validate_primary_scores` rejects arrays where a primary
has no terminal state or more than one, transmitted energy is outside
`[0, E₀]` or is scored for a non-transmitted primary, photon counts are not
nonnegative integers, the fixed 10–50/50–100/≥100 keV bands do not partition
the ≥10 keV total, or a band above the endpoint scores anything.

Records have a `kind`. `source` carries the creation-level photon counts above.
`transport` carries only terminal fractions and transmitted energy, and refuses
any photon or cone field. `detector` adds weighted cone yields per primary. Each
10–50/50–100/≥100 keV band, and the ≥10 keV total they sum to, is scored at
creation (`cone_source_photons_*`) and after uncollided escape
(`cone_escape_photons_*`). Yields must be nonnegative, bands must sum to the
total, nothing may score above the endpoint, and escape may never exceed the
yield at creation.

`comparison_record` adds primary moments (via `pyrite.perf.precision`) and,
for photon kinds, the covariance of the band means, so integrated yields keep
their within-primary correlation. Every kind also records the moments of
transmitted energy over transmitted primaries: their count, mean, second
moment and variance, with a delta-method SE for the ratio mean. Outcomes other than `ok` (`failed`, `timeout`, `oom`,
`precision-unmet`, `coverage-gap`, `unsupported-physics`) must carry a reason
and no scores, so they are retained rather than dropped.

`integrate_cone` integrates a point-direction estimator over a cone with
Gauss–Legendre nodes in cos θ and a periodic azimuth rule; constant integrands
reproduce `2π(1 − cos α)` to rounding. `converged_cone_integral` doubles both
orders until successive results agree to `rtol`, and reports an unconverged or
repeatedly zero result as unresolved instead of raising, so a narrow emission
feature missed by a coarse rule cannot pass silently. A point value times solid
angle never stands in for this convergence evidence.
`converged_cone_scores` does the same for per-primary arrays. It refines
until, for every field, both the primary sum and the primary sum of squares
(the inputs of the mean and its variance) change by at most `rtol`. A band
that is zero by construction is declared `exact-zero` and must stay zero.

## PyRITE transport-only and detector arms

`checks/equal_precision/pyrite_adapter.py` takes a pilot `manifest.json` id or a
frozen `matrix.json` case id. Matrix cases use exact catalog compositions,
including catalog `tungsten`, and the case's footprint, electron and photon
cutoffs and straggling switch. `--arm source` on a manifest id writes the
retained `pyrite.equal-precision.sample.v1` pilot record unchanged. Every other
combination writes a `pyrite.equal-precision.comparison.v1` record of the
matching kind. Each record keeps the raw per-primary `.npz` with its SHA-256,
the table digests, the warmup policy and the timings.

### PyRITE transport-only arm

`--arm transport` scores only terminal fractions and transmitted energy. The
headline arm is `--radiative-loss bremslib-soft-hard` (the default), described
below; `--radiative-loss none` is a diagnostic variant. With
`--radiative-loss none`, transport runs with
`radiative_model="uncoupled"`. It applies SBETHE collision stopping only, with
no bremsstrahlung energy loss and no hard-photon sampling, and loads no BremsLib
table. A comparator matches this by removing electron bremsstrahlung from
transport. With `--radiative-loss bremslib-soft-hard`, transport keeps the
coupled BremsLib soft loss and sampled hard-photon debits, but discards the photons
unscored. A comparator then keeps bremsstrahlung at the same production
threshold and kills photons at creation. The selected treatment is recorded
verbatim under `provenance.radiative_energy_loss`. The `--audit-cxr` call
trace, which already refuses PXR/CBS work in every arm, also refuses any
`pyrite.montecarlo.spectrum` or detector scorer call in this arm. The one
exception is the BremsLib table module that coupled transport samples hard
photons from when loss is retained. Blocks carrying radiative rows without
retained loss are refused.

### PyRITE detector arm

`--arm detector` runs coupled BremsLib transport for a matrix case with a
`detector` cone. It scores each block with
`pyrite.perf.detector_scoring.cone_detector_scores`, which calls the production
expected-value estimator `mc_coupled_brem_spectrum` (BremsLib DDCS by track
length along the coupled tracks) in its per-electron band mode, one call per
direction, stage and band. Each primary's segments are summed before any
moment. Primaries with no emission keep zero scores, and successive blocks keep
global primary indices. The cone axis is at `center_deg` from the slab normal,
in the x–z plane of incidence, as frozen in the matrix spec's `detector_frame`
and recorded with every detector record.

Both stages use one estimator and the same geometry, and differ only in the
absorber:

- **Source at creation:** the absorbing layer has zero number density, so
  every segment's escape transmission is exactly one.
- **Uncollided escape:** straight-ray EPDL2025 narrow-beam attenuation through the
  slab and its finite footprint, the estimator's production escape model.

Photon energy is integrated per band with composite Gauss–Legendre panels.
Panels are split at every EPDL2025 photoelectric edge of the slab and are at most
100 keV wide, with `2 × order` nodes each. The ≥100 keV band ends at the beam
energy. Energy and angle are refined jointly by `converged_cone_scores`
(`--cone-rtol`, default 10⁻³; `--cone-max-order`, default 32). The record keeps
each block's per-field convergence evidence under `provenance.acceptance`, and
`provenance.acceptance_gate` reads `unconverged` if any block failed to converge.
Terminal fractions and transmitted energy come from the same coupled transport.
In this arm, the `--audit-cxr` trace allows only the CXR package's shared
segment-clipping, observation-direction and groove-validation helpers, which the
bremsstrahlung estimator imports. Any other PXR/CBS call fails the run.

Local CPU smoke runs of a few histories only checked the plumbing (see
`checks/equal_precision/README.md`). Estimator cost grows with segment count ×
directions × bands, so precision runs belong on the lab GPU through
`pyrite remote`.

## Remaining gates

This stage leaves the following open: PenRed and EGSnrc adapters with
creation-level photon tallies, Geant4 transport-only and cone-detector tallies
matching the arms above, adapter support for finite beams and explicit
secondaries, numerical refinement (including the electron cutoff and the
estimator's straight-ray escape against full photon transport), equivalence
margins, equal-core resources and held-out time-to-precision runs. No physics default, model
marker or version changes here, and no new physics-validation claim is made.
