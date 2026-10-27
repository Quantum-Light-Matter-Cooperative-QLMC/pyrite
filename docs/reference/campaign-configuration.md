# Campaign configuration contracts

For a runnable edit-and-inspect exercise, use [Configure a campaign](../guides/configuration-cookbook.md).
The [profile settings reference](../repo-design/profile-settings.md) owns field
names, units, setters, creation/cloning and reporting-bin forms. The
[catalog schema](../repo-design/materials-catalog-schema.md) owns TOML structure;
[configuration resolution](../repo-design/configuration-resolution.md) owns
precedence. This page covers the behavior of those settings together.

## Calculation numerics and presets

`pyrite numerics show PROFILE` reports sampling, reflection/mosaic sampling and
transport integration; `pyrite physics show PROFILE` reports interaction models
and cutoffs. Each reports explicit and effective values and their sources.
`set` writes profile fields; `reset` removes explicit fields; `--dry-run`
previews without writing. Per-run overrides win over explicit profile values,
then named presets, then built-in workload/model defaults.

Named physics presets hold emission, model selectors and physical cutoffs.
Numerics presets hold scalar trial counts, integration, reflection/mosaic
quadrature and optional precision and line-grid policies. Bundled presets are
read-only. Clone one and attach it to an editable profile:

```bash
pyrite physics preset create low-angle --from electron-standard
pyrite physics preset set low-angle --elastic-model mott --straggling
pyrite profile set lab-scan --physics low-angle
pyrite physics show lab-scan
```

Numerics uses `pyrite numerics preset` and `profile set PROFILE --numerics NAME`.
Explicit profile physics fields override their preset. Any profile sampling
value supersedes the preset's entire sampling policy; fixed counts and adaptive
stopping are never mixed. Trial-count presets hold scalar values; a profile can
sweep counts with `profile set|add --cxr-trials/--brem-trials`.
One trial is one Monte Carlo history for that spectrum, distinct from physical
bunch charge. Line and brem counts select shared transport histories and are
not added to obtain a transport count.

`preset adopt PROFILE NAME --dry-run` previews moving effective settings into
a reusable preset without changing result identity; multi-value count grids
stay on the profile. Edits affect future runs of all referencing profiles.
Renames update references, deletion refuses referenced objects, and
`profile remove PROFILE --physics` (or `--numerics`) detaches a preset.
Names do not enter scientific identity; runs record preset names and definition
digests as provenance. Python uses `pyrite.config.load_physics` / `pyrite.config.load_numerics`
to obtain frozen configurations. See [named presets ADR](../adr/0021-named-physics-numerics-presets.md).

## Adaptive electron counts

Eligible profiles without fixed counts or a `precision` table use adaptive
sampling: target relative standard error 0.05 on line and bremsstrahlung yields,
200–20,000 electrons in blocks of 100, with heavy-tail guards. A case stops when
watched yields meet targets and guards pass, or at the maximum with a
statistics-limited flag. Both components use one transported population.
`pyrite numerics precision show PROFILE` reports the policy, fixed counts,
blockers and line-grid source.

Fixed counts apply when a profile or material override sets
`cxr_trials`/`brem_trials`, or for unsupported adaptive routes: coherent or
`both` emission, particle cascades (secondaries, pairs, positrons), GDF beams,
grooved entrances, physical pixel detectors, or channeling. An explicit policy
on an unsupported route is refused. Trajectory capture also needs fixed counts.
Bundled `trajectory_demo`, `coherent_demo`, and detector demos provide bounded
fixed-count examples; clone a demo before editing it.

```bash
pyrite numerics precision enable lab-scan
pyrite numerics precision set lab-scan --target-rse 0.02
pyrite numerics precision show lab-scan
pyrite numerics precision disable lab-scan --cxr-trials 500 --brem-trials 100
```

`enable` removes fixed profile counts, then stores supplied policy options;
it never changes beam, physics, geometry or grids. It refuses material count
overrides, which must be removed from catalog TOML by hand, and unsupported
routes, writing nothing. `set` retains unspecified policy fields, starts new
policies from defaults, and refuses fixed-count profiles with an `enable` hint.
`--observable` replaces the watched set. `reset PROFILE FIELD...` restores
individual defaults; `reset PROFILE` drops the table and returns an eligible
profile to implicit adaptive sampling. It does not add fixed counts.

`disable` removes the policy and stores fixed counts. Supplied counts replace
existing ones; unspecified existing counts, including sweep grids, remain.
Missing counts prompt on a terminal; non-interactive, `--yes` and `--dry-run`
use the workload defaults (300 line, 150 brem). Setting counts with
`numerics set` while a policy exists is refused; use `precision disable`.
All precision commands support `--dry-run`.

The minimum, maximum and optional pilot count must be block-size multiples.
Default guards are maximum electron share 0.05, minimum effective electrons
100, stability blocks 3 and stability fraction 0.5. `--band-ev START,STOP`
selects the monitor band (default: case line band); `--batch-means-band-ev`
adds reported bin errors without controlling stopping.

For independent brem stopping, keep both observables and set
`--brem-target-rse`, `--brem-min-electrons`, and `--brem-max-electrons`.
Brem bounds default to shared bounds and must be block multiples; its maximum
cannot exceed the transport maximum. Brem uses a prefix of shared transport,
freezes when its target/guards pass or its maximum is reached, and line work
continues until both finish. Reset all three `brem-*` fields to restore shared
stopping. Guards are shared; the pilot is clipped to brem bounds.

The requested policy enters dataset/case identity. Realized counts and errors
live separately in `adaptive_sampling`, so stopping does not change resume
identity. Count grids cannot combine with adaptive sampling. The Python
policy is `Numerics(precision=Precision(...))`; configured campaigns also accept
`Settings(precision=policy)`. See [statistical methods](../computation/statistical-methods.md#adaptive-electron-counts)
for normalization, rare-electron limits and replay semantics.

## Photon grids

Ordinary runs need no preparation batch. Automatic line grids use the case's
kinematic bandwidth at measured sinc spacing, capped by kinetic energy and the
Chantler coupling-data limit (about 966 keV). MeV beams can require millions of
nodes. Measured bandwidth/local-resolution selectors provide a bounded
alternative; their allowed combinations and tolerances belong to
[profile settings](../repo-design/profile-settings.md#line-grid-policy).

Per-case line-grid precedence, highest first:

1. Any explicit profile line-grid policy, even default-valued selectors: resolve
   automatically from that case, ignoring inline/stored line coordinates.
2. Explicit `E_grid_cxr` on the profile or material override.
3. Stored rows from `energy_grid_refs` or the catalog grid store. Automatic-grid
   environment controls outrank this layer, but not explicit coordinates.
4. Default automatic resolution when no stored row covers the energy.

A policy changes line coordinates only; stored brem grids still apply.
`pyrite numerics cxr-grid show PROFILE` displays sources; `set` changes
selectors and `reset` removes them. Cloning copies the policy; fresh creation
does not set one. Policy edits enter dataset identity.

An omitted continuum grid uses 25 eV uniform spacing. Each case raises its
start to the medium's photon-continuum floor on that lattice and replaces its
stop with beam energy plus one step. Explicit uniform grids select spacing and
can request a higher start; their stop is still case-local. Nonuniform grids
are used as stored. This is not automatic continuum refinement. Profile
inspection groups defaults, inline overrides and artifacts under Energy grids.

Optional derivation requires material and energy selections:
`pyrite numerics energy-grid derive --material hopg --energy 30,60 --profile lab-scan`.
Geometry/spacing controls are per invocation; omitted angles use profile
geometry. Use remote derivation for heavy cases. Install full artifacts with
`pyrite-dev energy-grid add PATH --profile lab-scan --material hopg` and verify
with `pyrite-dev energy-grid verify`. Do not paste reduced bounds into TOML.
Manual brem edits preserve resolved spacing unless `--step` is supplied.

Automatic coherent grids require `[cxr_grid_policy] windows = true` in the
profile object file and an adequate `max_points` budget (default 600000).
Plans above the budget refuse with their node count. `bin-mean` refuses positive
`max_dE_frac`. The windowed coherent path warns that complete-spectrum sampling
and dispersion remain uncertified; see its [validation record](../validation/beam-transport/coherent-line-grid-windowed-resolution.md).
Measured-bandwidth runs record a truncation audit and per-electron yield error
in `line_grid_resolved`; relative error above 0.1 raises
`CxrYieldStatisticsWarning` and flags statistics-limited output.

## Named beams and explicit instruments

A profile attaches `beam = "NAME"`; an inline `[profiles.NAME.beam]` table has
the same meaning. `pyrite beam create|set` owns phase-space edits;
`profile set --beam NAME` attaches a reference. Rename updates references and
delete refuses referenced beams. Labels and object names are excluded from
resolved identity. Identical resolved cases may share cached arrays; distinct
profile datasets need not share a directory stem.

Longitudinal/transverse tables decode into `BeamSpec`. Their units, defaults
and exclusions belong to [beam phase space](../physics/beam-transport/beam-phase-space.md).
Non-default policies enter identity; inert defaults preserve it. CLI flags
include `--emittance`, `--twiss-beta`, `--twiss-alpha`, `--energy-spread` and
`--transverse-fwhm-mm`. An emittance table replaces the default spot FWHM;
explicitly declaring both is an error. `emittance_demo` supplies a normalized
0.1 mm·mrad waist at beta 0.05 m with 0.1% energy spread;
`hopg_emittance_demo` uses that same physical beam at 30 and 100 keV.
For imported particles and charge normalization, see the [GDF input contract](gdf-beams.md).

Complete-catalog profiles must name a beam; profiles read over the bundled
catalog retain the example-beam fallback. Runs require explicit scalar
observation angle, polar acceptance and solid angle, or pixel geometry.
The bundled `reference` detector is an ideal scalar Timepix3 aperture;
[profile settings](../repo-design/profile-settings.md#instruments-and-emission)
owns its exact acceptance. Partial custom scalar geometry must have its missing
fields filled without replacing its existing pose. There is no implicit count
acquisition. A complete catalog must supply any referenced detector object.

## Detector collections, filters, and counting observations

`[profiles.NAME.detectors]` maps IDs to named detectors, or
`[profiles.NAME.detectors.ID]` declares inline geometry. IDs begin with a
lowercase letter, followed by lowercase letters, digits, `_` or `-`. Scalar
acceptance and pixel geometry cannot mix in one detector. Pixel pose/grid
fields and optional scorer, response and acquisition belong to that ID;
profile filters are shared and ordered. Pixel acceptance derives from geometry,
including without acquisition. `profile show` prints the projection.

```bash
pyrite detector create planar sensor --distance-mm 400 --shape 256 256 \
  --angular-shape 2 2 --response ideal --exposure-s 1 --reporting-bins native
pyrite profile detector migrate lab-scan --dry-run
pyrite profile detector migrate lab-scan --yes
pyrite profile detector add lab-scan camera --detector sensor
pyrite profile detector show lab-scan camera -o json
pyrite profile detector set lab-scan camera --inline-copy --roll-deg 5
```

Named detector creation uses `scalar`, `planar`, or `clone --from SOURCE`;
`set` edits reusable objects. `profile detector list|show` inspects ID-resolved
geometry/references. `add` attaches an ID; `set --detector REF` replaces a
reference. Profile-local field edits of a named reference require
`--inline-copy`, preserving all settings before the local edit.
`detector reset REF scorer|response|acquisition` clears optional sections;
`profile detector reset PROFILE ID SECTION...` clears inline sections.
`rm PROFILE ID` must leave one detector. Removal and migration require
confirmation (`--yes` for scripts); `--dry-run` previews without writing.

Legacy `detector` and `physical_detector` are exposed as IDs `default` and
`physical`; they cannot coexist with a collection. Targeting those IDs preserves
storage until explicit `migrate`. Migration preserves detector settings/order,
previews exact TOML and checkpoint variants, and confirms before writing.
Converting one legacy detector into a collection adds `detector-ID` to the
variant and changes identity. Multiple legacy detectors already use these
variants. Existing checkpoints remain in place. Profiles with no declaration
still need explicit acceptance after migration. The `profile set --detector`
shortcut updates a sole collection entry; multiple entries need ID-targeted edits.

`list`/`show` JSON use `cxr.profile.detector.list` / `cxr.profile.detector.show`
envelopes. List contains `profile` and ID-keyed `detectors`; show contains
`profile`, `id`, `kind`, `reference`, `settings` and projected `acceptance`.
Each `[[profiles.NAME.filters]]` is one `FilterPlate`, edited in declared order
with `profile filter add|set|rm|list|show`.

A run executes every detector ID for each material, with separate identity-
qualified datasets. Acquisition adds a factorized observation under
`pyrite-output/observations/<stem>/`. Pose and grid changes alter projection
and dataset; angular shape and filters re-evaluate observations with new
transport while scalar cache records remain; response, acquisition and beam
normalization rescore observations without transport. Remote observation
transfer and garbage collection are not implemented.

`two_detector_demo` provides scalar apertures at 90° and 135° with identical
ideal Timepix3 acceptance, without counts. `pixelated_detector_demo` uses a
256×256, 0.055 mm pitch camera at 400 mm/90°, 2×2 angular scoring and ideal
counts for 1 s in 100 eV bins over 0–30 keV. Those reporting bins do not broaden
the scalar source spectrum. `detector_demo` adds filtering and Poisson counts.
Each demo uses one 30 keV HOPG case, 500 line and 100 brem trials. Inspect
pixel observations with `pyrite app pixels launch hopg`.

## Single-scene pixel runs

`pyrite run NAME -m MATERIAL --ephemeral` resolves one local in-memory scene.
Thickness, energy, polar, azimuth and electron-count axes must be singleton.
It selects the sole pixel detector; `--detector ID` selects among several.
Put beam, geometry and workload inputs in the profile. Output is a compact
summary, tab-separated `-o wide`, or the versioned `cxr.material.simulate` v1
JSON envelope (including total counts for acquisitions).

`--output-file PATH` writes complete factorized spatial arrays to NPZ:
per-tile intrinsic lines, pixel-to-tile mapping, solid angles, filter paths and
line attenuation coefficients. Background remains a 1D summary. The exact path
is used, with no implicit suffix and no overwrite. Ephemeral mode never reads
or writes campaign checkpoints, shared case cache or observation stores.
An ordinary `--no-cache` run still writes its campaign checkpoints.

`--detector`/`--output-file` require ephemeral mode, which requires explicit
`-m MATERIAL`. Catalog/profile/output controls work; remote/preset workflows,
checkpoint/cache controls, trajectories, profiling, quick-grid changes,
worker/reflection overrides, resume/wall-clock controls and CLI beam/GDF
overrides are refused. Usage errors exit 2, runtime failures exit 1; JSON
failures use the same envelope and diagnostics stay on stderr.

Compare matched observations and resolved grids: checkpoint source spectra
use one projected direction, while ephemeral pixel spectra average angular
tiles with solid-angle weights and finite-filter attenuation. Automatic pixel
line grids resolve jointly over tile directions. Their totals can differ even
at equal seeds/counts. Keep explicit grids and transport controls matched when
testing command parity; do not infer parity from unlike scalar/pixel totals.

## Identity and storage

Resolved physics, workload, grids and detector variants determine dataset
identity; labels and execution-only tuning have separate rules. Checkpoint
manifests retain exact inputs. Use [dataset identity and storage](../repo-design/storage/dataset-identity-and-storage.md)
for hashing, compatibility stems, case reuse, archive and cleanup contracts.
For experimental channeling settings/restrictions and its separate component,
use [planar channeling](../guides/experimental-planar-channeling.md) and
[ADR-0018](../adr/0018-quantum-channeling-zone.md), rather than a generic sweep recipe.
