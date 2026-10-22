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
- **Sparse extensions (19):** 45° and 75° incidence at Si 800 keV `t/R` 0.3;
  angle×thickness checks at W 800 keV 75° and `t/R` 0.03 and 3; diamond (low Z)
  and SiO₂ (compound) at 800 keV `t/R` 0.3; one-factor beam phase space at Si
  800 keV `t/R` 0.3 (100 µm RMS spot, 5 mrad RMS divergence, 1% RMS energy
  spread); a 1 × 1 mm footprint at `t/R` 3 for Si and W; five detector cones
  (90°/5° for Si and W, 0°/1°, 0°/15°, 135°/5°); straggling on for Si and W;
  and one explicit-secondary arm for Si.

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

38 of 43 cases are covered by released tables. Every 5 MeV case lies inside all
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

The remaining gaps are:

- **5 detector cases:** the uncollided-detector arm needs the production
  weighted per-segment scorer, which no adapter calls yet.
- The tungsten secondary-transport arm is not yet declared. Catalog `tungsten`
  now supplies the per-layer key that `inelastic_model="shell-soft-hard"`
  needs, so it can be added as an extension.

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

`comparison_record` adds primary moments (via `pyrite.perf.precision`) and the
covariance of the band means, so integrated yields keep their within-primary
correlation. Outcomes other than `ok` (`failed`, `timeout`, `oom`,
`precision-unmet`, `coverage-gap`, `unsupported-physics`) must carry a reason
and no scores, so they are retained rather than dropped.

`integrate_cone` integrates a point-direction estimator over a cone with
Gauss–Legendre nodes in cos θ and a periodic azimuth rule; constant integrands
reproduce `2π(1 − cos α)` to rounding. `converged_cone_integral` doubles both
orders until successive results agree to `rtol`, and reports an unconverged or
repeatedly zero result as unresolved instead of raising, so a narrow emission
feature missed by a coarse rule cannot pass silently. A point value times solid
angle never stands in for this convergence evidence.

## Remaining gates

This stage leaves the following open: the production weighted scorer for
detector cases, PenRed and EGSnrc adapters with creation-level photon tallies,
transport-only scoring, numerical refinement and equivalence margins, equal-
core resources and held-out time-to- precision runs. No physics default, model
marker or version changes here, and no new physics-validation claim is made.
