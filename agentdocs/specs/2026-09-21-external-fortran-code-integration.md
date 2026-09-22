# External Fortran code integration (ELSEPA, SBETHE, BremsLib)

Date: 2026-09-21
Status: accepted 2026-09-21; implementation tracked by #161
Issues: #161 (subsystem), #84 (roadmap parent), #86, #87, #89, #90, #93, #94, #95

Source paths below are relative to `src/pyrite/` unless stated otherwise.

## Problem

Three external Fortran codes must feed PyRITE's electron-transport and
photon-emission physics:

| code | supplies | issue |
| --- | --- | --- |
| ELSEPA 2020 | elastic DCS, integrated/transport elastic cross sections | #89 |
| SBETHE | shell- and density-corrected collisional + radiative stopping | #90 |
| BremsLib 2.0.8 | bremsstrahlung SDCS and DDCS/shape function | #86, #87, #95 |

None is a library. All three are batch programs driven by input decks or
prompts, writing fixed-name text files into the working directory. Run times
range from seconds (ELSEPA) to minutes (BREMS), so none can be called inside a
Monte Carlo loop. The integration is therefore a table-generation pipeline
feeding existing LUT machinery, not a runtime binding. No FFI layer
(f2py/ctypes/ISO_C_BINDING) is warranted or planned.

Users must be able to generate tables for materials PyRITE does not ship, so
the generator ships as part of the package rather than living in a
maintainer-only directory.

## Licensing

Verified from source headers and upstream READMEs on 2026-09-21.

- **ELSEPA 2020** — CC BY-NC 3.0.
- **SBETHE** — CC BY-NC 3.0.
- **BremsLib Fortran sources** — **GPL-3.0-or-later**. GPL headers are present
  in `Interpolate_DCS/Interpolate_DCS.f90` and in `Brems/{Brems,
  Bremsstrahlung, FitExp, Born_SM_appr, Brems_common}.f90` plus
  `Read_S_integrals.f90`. The CC BY 4.0 terms apply to the published *dataset*
  deposit, not the code — **confirmed 2026-09-21** from the deposit's own
  licence metadata: CC BY 4.0 International. Citation: Poškus, Andrius (2025),
  “BremsLib v2.0.8”, Mendeley Data, V9, DOI `10.17632/6zfsc9xsz8.9`. The
  deposit's terms themselves caution that further permission may be required
  for third-party content within it, which is how the bundled GPL-3 sources
  are to be read.

PyRITE is distributed under the UCLA Academic Software License
(academic/nonprofit use only), so the NC clauses impose no additional
restriction. None of the three carries ShareAlike, so no copyleft reaches
PyRITE's own code from the CC-licensed material.

Consequences:

1. Running these programs and consuming their **output** is unrestricted; GPL
   does not reach program output.
2. Translating `Interpolate_DCS.f90` or `Brems_CS_interp.f90` into Python
   creates a derivative work under GPL-3, which cannot be distributed under a
   nonprofit-only license (GPL-3 §7 forbids adding field-of-use restrictions).
   **Do not port BremsLib code.** Clean-room reimplementation from
   `Interpolate_DCS.pdf` and `Brems.pdf` is permitted.
3. **Do not vendor the BremsLib Fortran** into the wheel. Subprocess
   invocation is arm's-length, but bundling GPL-3 sources inside a
   nonprofit-only distribution raises a question that is avoidable.
4. All three BY clauses require attribution: author, title, license name and
   link, and an indication of modifications. Derived/resampled tables are
   adaptations and must be marked as such.

### Required repository changes

- Add `THIRD-PARTY-NOTICES.md` at the repository root, shipped in the wheel,
  listing all three codes with author, citation, license, link, and the nature
  of PyRITE's modifications.
- Set `[project] license` and license classifiers in `pyproject.toml`, which
  are currently unset.

## Decisions

### D1 — Ships as a domain package, not a maintainer tool

New package `src/pyrite/xsgen/`. A top-level `src/datagen/` would not be
packaged (the wheel takes `packages = ["src/pyrite"]`) and would claim a second
import name.

Rejected alternatives: separate repository per code (triplicates identical
plumbing for one consumer); one separate repository (cross-repo CI and release
coordination, and the provenance pin ends up in a different repo from the table
it pins); direct inclusion in the runtime physics packages (makes gfortran an
install dependency and puts minute-scale subprocesses behind a runtime import).

### D2 — One table store, one key

A single namespace keyed on a hash of a normalized request record:

```python
key = sha256(normalize({
    "code":         "elsepa" | "sbethe" | "bremslib",
    "code_version": "<source tree SHA-256>",
    "target":       Element(Z) | Material(<materials._identity hash>),
    "model":        {...code-specific deck parameters...},
}))
```

The element/material dichotomy does not survive the physics: ELSEPA in
muffin-tin mode (`MUFFIN 1`, the mode wanted for solids) takes `RMUF` from the
nearest-neighbour distance and `MABS`/`VABSA`/`VABSD` from density and band
gap, so it is already material-dependent. Only free-atom ELSEPA
(`MUFFIN 0`) is purely per-Z. A two-namespace design would encode a
distinction that is wrong on its most important consumer.

One invalidation rule — key changes, table is a different table — covers
free-atom ELSEPA (`Element`), muffin-tin ELSEPA and SBETHE (`Material`), and
BremsLib (`Element`). The material identity hash comes from the existing
`pyrite.materials._identity`.

Rejected: treating SBETHE `.mat` files as a catalog-adjacent artifact. It
splits provenance across two mechanisms and misses the checkpoint-identity
wiring.

### D3 — Two-tier table resolution

`xsgen.store.resolve(key)` checks the user table directory first, then the
packaged `paths.data_dir()` fallback. Every consumer (`montecarlo/transport/lut.py`,
`montecarlo/transport/scattering.py`, `montecarlo/transport/stopping.py`,
`montecarlo/spectrum/brem.py`) goes through that single
function and never learns where the bytes came from.

User tables live under `paths.user_data_dir()` — per-user, shared across
workspaces — not in the workspace. Tables are expensive, target-scoped, and
not run-specific; regenerating W for each new workspace is wasteful. A
workspace-level override remains available for reproducibility-pinned
campaigns.

`paths.user_data_dir()` currently exists and is documented as reserved; this
is its first use.

### D4 — Vendoring policy is decided by size, not license

Measured 2026-09-21:

| artifact | size | decision |
| --- | --- | --- |
| ELSEPA source + `database/` | 4.6 MB | vendor |
| SBETHE `sbethe.f` | 0.13 MB | vendor |
| SBETHE `sdbase/` | 18.8 MB, 599 files | fetch on demand |
| SBETHE `docs/`, `sbethe.exe` | 9.8 MB | neither vendored nor extracted |
| BREMS `*.f90` sources | 1.7 MB | not redistributed (GPL, see above) |
| BREMS `V/` + `CS_int/` | 24 MB | not redistributed |
| BremsLib precomputed library | 810 MB payload (1.5 GB on disk) | not redistributed (size) |
| BremsLib-*derived* tables | small, per material | **ship for the built-in catalogue** |

Rule: always vendor the generator source where licensing permits; vendor a
reference database only when small; ship derived tables for the built-in
catalog; fetch large reference databases on demand into `user_data_dir()`,
SHA-256 pinned.

**Revised 2026-09-21.** The three BremsLib artifacts are now treated
separately, because they are under different terms and hit different limits:

- **GPL-3 Fortran sources** — never redistributed, never ported (D7). A
  licence constraint, and not negotiable under PyRITE's nonprofit-only terms.
- **The precomputed data library** — not redistributed. A *size* constraint
  (810 MB payload), not a licence one. CC BY 4.0 would permit it.
- **Tables derived from that library** — **shipped** for the built-in
  catalogue, which brings BremsLib into line with the general rule above
  rather than leaving it as the one exception to it.

This supersedes the earlier "nothing from BremsLib is redistributed" position.
It rested on the dataset terms being unverified; they were confirmed CC BY 4.0
on 2026-09-21 (see Licensing). CC BY 4.0 carries neither NonCommercial nor
ShareAlike, so adaptations — derived tables — may be redistributed under
PyRITE's own terms provided Poškus is credited, the licence is linked, and the
tables are marked as modified. Those three obligations are discharged by
`THIRD-PARTY-NOTICES.md` plus the per-table provenance manifest of D5, which
already carries a modifications note; shipped tables must not be added without
them.

Generating a table for a material *outside* the built-in catalogue still needs
a local checkout, defaulting to `../BremsLib_v2.0.8` relative to the PyRITE
checkout and overridable through `xsgen.sources`. Absence is an actionable
error naming the expected path, never a silent fallback. This is the same
shape as ELSEPA and SBETHE: shipped tables cover the catalogue, generation
covers everything else.

Net wheel growth ≈ 4.8 MB (ELSEPA 4.6 MB + `sbethe.f` 0.13 MB) against 26 MB
of existing packaged data (of which
`EEDL.endf` is 25 MB — the precedent that a single large vendored data file is
already accepted practice here).

Vendoring ELSEPA outright is the high-value case: 4.6 MB buys both the code
and its database, so `pyrite tables generate --code elsepa` works offline and
zero-config, which matters for the `pyrite remote` cluster workflow.

Any vendored fixed-width table whose bytes are SHA-256 pinned needs the
`.gitattributes` `-text` treatment that `EEDL.endf` already has; `eol=lf`
normalization would otherwise break the pin.

### D5 — Runtime manifest replaces the committed provenance README

Each generated table gets a sidecar JSON manifest: Fortran source SHA-256,
gfortran version, deck hash, model parameters, PyRITE version, timestamp, and
a modifications note for CC BY attribution. The manifest hash feeds the model
identity marker, so checkpoints invalidate when a table changes.

This is the highest-risk piece. Without it, a user regenerating an element
with different deck parameters (`MABS 2` versus `MABS 0`) silently reuses
checkpoints computed from the old table.

**Prerequisite for D6:** `BREMSSTRAHLUNG_MODEL` (`montecarlo/spectrum/brem.py:44`) is a
module-level constant, not a function of the selected model, and
`campaign/profiles.py:548,633` and `api.py:257,369` record that constant into
run identity.

This is correct today and is not a live defect. The production path
(`montecarlo/runner/emission.py:38,54`) never passes `cross_section_model`, so every
checkpointed run uses EEDL; `"bethe-heitler"` is reachable only through
`montecarlo/spectrum/diagnostics.py`, an analysis surface that writes no identity or
checkpoints. The constant therefore describes what production actually
computes.

It becomes wrong the moment D6 makes a second model selectable from the run
path. Making the marker depend on the actual selection is part of D6, not a
standalone fix, and must land in the same change that exposes the selector —
otherwise runs using different bremsstrahlung data share a checkpoint
identity.

### D6 — BremsLib becomes the single source for bremsstrahlung cross sections

Supersedes the standing constraint in #84 ("Preserve the EEDL bremsstrahlung
energy spectrum") and in #86 ("EEDL remains the supported baseline").

Rationale: EEDL supplies an energy spectrum only. `montecarlo/spectrum/brem.py:260` rejects any
EEDL panel with `NA != 0`, so the packaged data carries no angular information
at all — which is precisely why emission is isotropic today (#87). BremsLib
supplies both SDCS and DDCS from one evaluation. Taking the spectrum from
EEDL and the angular shape from BremsLib would mean the shape function no
longer integrates to its own parent SDCS, requiring a renormalization that
discards part of what makes the BremsLib DDCS accurate.

Sequencing constraint: the default flips **after** #86 demonstrates the
accuracy claim, not before. #86's existing comparison work is the gate.
`"eedl"` remains a selectable model for regression and continuity;
`"bethe-heitler"` remains as the analytic fallback.

There were two independent reasons `"eedl"` had to stay the packaged default.
**Availability** — BremsLib-backed emission required every user to obtain the
library themselves — is removed by shipping derived tables (D4, revised).
**Accuracy** — #86 has not yet demonstrated the claim — stands, and is now the
only gate. Do not read the first being lifted as the second being lifted.

### D7 — Do not reproduce `Interpolate_DCS`

`Interpolate_DCS.f90` is 3558 lines, 96 parameters, 38 aliases, and ships
`DGELS.f`/`DMINV.FOR` — it is a weighted least-squares spline fit over selected
knots, not an interpolating spline. Its substance is: log–log fitting in
(T1, k/T1); interpolation of the *ratio* to a reference sub-library or an
analytic Schiff/Born form over parts of the plane rather than of the DCS
itself; shape-function renormalization against the SDCS; 11-point Newton–Cotes
θ integration; and error propagation through the fit.

PyRITE needs something narrower: SDCS and normalized shape function on the
PyRITE grid, from which it builds its own sampling CDFs (needed for #87
regardless). Plan: read the library tables, interpolate onto the PyRITE grid
with our own code, and use the shipped sample outputs
(`Interpolate_DCS/Samples/`, `Linux_exe/Samples/`) as the accuracy oracle
rather than reimplementing the path that produced them.

Adopt the ratio-to-analytic-reference technique — interpolating DDCS/Schiff is
far better conditioned than interpolating DDCS — as a documented method, not
as ported code (see licensing item 2).

### D8 — BremsLib library converter (deferred)

The 86300-file DDCS tree is read in place from the local checkout. Converting
it to a compressed store is **deferred**; it is not needed to land #87 or #95.

Measurements taken 2026-09-21 on a 300-file random sample, recorded so the
decision need not be re-derived:

| representation | full-set size |
| --- | --- |
| ascii payload (as distributed) | 810 MB |
| float64 dcs + float32 err, zstd-19 | 203 MB (bit-for-bit lossless) |
| `tar` of the ascii, zstd-19 | 173 MB |
| float32 dcs + float32 err, zstd-19 | 131 MB |
| float32 dcs, byte-shuffled, zstd-19 | 50 MB + ~15 MB for relErr |

float64 reproduces the DCS field exactly under `%.8e` in 300/300 sampled
files, so bit-for-bit round-trip is a checkable property if this is revisited.
Naive binary loses to `tar | zstd`; the entire win comes from the shuffle,
which only pays at float32. Only byte-shuffle was measured — bitshuffle
(blosc2) and delta-along-θ were not.

Revisit if read latency from the raw tree becomes a practical problem. Any
float32 variant would first need the relErr distribution checked across the
full library; the sample covered one file's range.

θ grids are ragged — 181/441/221 rows in the sample — so any future store
needs a grid-index table rather than a single 4-D block.

### D9 — gfortran is an optional runtime dependency

Not a Python dependency. Detect at runtime; on absence or unconfigured
sources, fail with the exact compile command and the upstream location. Never
silently fall back to a surrogate model — #89's acceptance criteria already
forbid this for elastic scattering, and the same rule applies throughout.

## Architecture

```
src/pyrite/xsgen/
  sources.py                 # resolve vendored vs. user-configured code trees
  toolchain.py               # gfortran detection + build, cached per source SHA
  _run.py                    # scratch dir, data-dir symlinks, cwd, subprocess
  store.py                   # key, manifest, two-tier resolve, table I/O
  elsepa/{deck,parse}.py
  sbethe/{deck,parse}.py
  bremslib/{convert,read}.py # library converter + table reader; no runner
```

### Execution model

All three codes hardcode relative data paths and fixed output filenames:
ELSEPA needs `./database`, SBETHE needs `./sdbase`, BREMS needs `./V` and
`./CS_int`; ELSEPA writes `dcs_xpyyyezz.dat` and `dcs.dat` into the working
directory. Every run therefore gets its own scratch directory with those data
directories symlinked in and `cwd` set to it. This is mandatory for
concurrency, not a convenience — concurrent runs in a shared directory
overwrite each other's output.

Verified for SBETHE on 2026-09-21: every database path in `sbethe.f` is a
cwd-relative literal — `'./sdbase/eshcor-'` etc. at lines 270-274, and
`'./sdbase/'` at 905, 984, 1425, 2349. The upstream readme's phrasing ("the
subdirectory './sdbase' of its own directory") is loose; Fortran `OPEN`
resolves against the process cwd, so setting `cwd` to the scratch directory and
symlinking `sdbase` into it is sufficient and the built binary may live
anywhere.

SBETHE is prompt-driven rather than deck-driven; its input is fed on stdin and
its `<mname>.mat` cache file is written into the scratch directory — which is
also why runs must not share one: the `.mat` file is read back in preference to
the prompts whenever it exists, so a stale sibling silently overrides the
requested material parameters. ELSEPA reads a deck on stdin
(`elscata < deck.in`).

Runs are cached content-addressed on the same key as the resulting table, so
an expensive sweep runs once.

### Grid policy

Committed and cached tables use each code's native dense grid, not PyRITE's
electron energy grid. PyRITE's grid is configurable (`energy_grid/`), so
baking it into the data file makes every table stale on a grid change.
Resampling happens at load in `montecarlo/transport/lut.py`, where interpolation already
lives.

Interpolate cross sections in log–log. For ELSEPA, build the normalized
angular CDF **before** interpolating; interpolating the DCS and integrating
afterwards produces non-monotone CDFs near the forward peak.

### Layering

`xsgen` sits at driver tier: it imports `materials` (composition and lattice →
deck scalars), and nothing in the physics core imports it. Transport and
spectrum read tables through `store.resolve` only.

Required `pyproject.toml` import-linter changes:

- add `pyrite.xsgen` to `forbidden_modules` in
  `physics-core-stays-below-drivers`
- add `pyrite.xsgen` to `forbidden_modules` in `materials-imports-no-peer-domain`

### CLI

New command group, subject to the documented command/help/output/exit
contracts and requiring `docs/repo-design/cli/cli-reference.md` regeneration:

```
pyrite tables generate --code <c> [--element Z | --material NAME] [deck opts]
pyrite tables list
pyrite tables show <key>          # prints the manifest
pyrite tables path
pyrite tables sources set <code> <path>
pyrite tables fetch <code>        # pinned download of a large reference DB
```

## Per-code integration

### ELSEPA → #89

Mode: `elscata` with `MUFFIN 1` for solids, per `Al.in` and
`Atoms-in-solids.pdf`. **Not** `elscatm`: the molecular path is an
independent-atom approximation with coherent summation over a randomly
oriented molecule. Applying it to a crystal cluster produces diffraction,
which PyRITE models separately, and would double-count.

The lattice database feeds ELSEPA as derived scalars, not as an atom-position
list: `RMUF` from the nearest-neighbour distance, and density plus band gap
into `MABS`/`VABSA`/`VABSD`.

Output replaces `src/pyrite/data/mott_transport_cross_sections/`, whose
consumer in `montecarlo/transport/lut.py` is already wired.

Energy coverage must be decided against #94 and #13, not only #89.

The upstream README's "about 5 eV" is the code's numerical operating floor,
not a physical validity claim. Reported validity for the muffin-tin/DHFS
treatment in solids is roughly 100 eV at the optimistic end and 1 keV at the
conservative end, depending on source and potential model. So the low end is a
**documented floor**, not something to extend into: generating tables below
~100 eV would produce numbers the model does not support. #94 must either set
its secondary tracking threshold at or above that floor, or carry an explicit
elastic-model boundary below it.

The binding constraint is the upper end. #13 targets ~100 MeV, which produces
delta rays up to ~E/2 — tens of MeV. Two things need checking before claiming
that range: whether ELSEPA's high-energy factorization (`IHEF`) covers it with
the partial-wave series still converging, and whether the angular CDF grid
resolves a DCS that becomes extremely forward-peaked at those energies.
Neither is answered by the 1–300 keV range #89 currently states.

### SBETHE → #90, and secondary outputs for #93 and #86

#### Upstream source (resolved 2026-09-21)

The whole of SBETHE — program, database and documentation — is published as a
**single zip** in one Mendeley Data deposit, not as separately addressable
parts. There is no independent `sdbase` URL to pin; the fetch pulls the zip and
extracts from it.

| field | value |
| --- | --- |
| landing page | <https://data.mendeley.com/datasets/7zw25f428t/2> |
| DOI | `10.17632/7zw25f428t.2` (version-pinned, immutable) |
| file | `sbethe.zip`, 12 878 659 bytes |
| SHA-256 | `d5d4879c2073ada3bd799fe0727054549cd6ec65cc699acbd0ff25fd5c3c4144` |
| direct URL | `https://data.mendeley.com/public-files/datasets/7zw25f428t/files/a7d2eed6-9aab-4462-936d-b15adf0e7c16/file_downloaded` |
| licence | CC BY-NC 3.0, confirmed from the deposit's own `data_licence` field |
| authors | F. Salvat (U. Barcelona), P. Andreo (Karolinska) |

Mendeley's public API publishes that SHA-256 *before* the download, at
`https://data.mendeley.com/public-api/datasets/7zw25f428t/files?folder_id=root&version=2`,
so the pin is verifiable against upstream rather than against whatever bytes we
happened to receive first. Both the hash and the licence above were checked
against a real download on 2026-09-21. Prefer the versioned API endpoint for
re-verification; treat the opaque file-id URL as a cache, not as the identity.

Extraction rules:

- Extract `sdbase/` only. **Do not extract or redistribute `sbethe.exe`** — a
  1.16 MB prebuilt Windows binary of unverified provenance. `xsgen` builds from
  `sbethe.f` through `toolchain.py`.
- Skip `docs/` (8.6 MB of PDFs). `rpwba.pdf` is the PWBA/GOS reference cited in
  the ionization-scope section; cite it, do not ship it.
- 28.7 MB extracted for a 12.9 MB download, of which 18.8 MB is the database we
  actually want.

`sdbase/` holds five per-element families for Z=1-99 — `oos<Z>.tab`,
`shcor-<Z>.tab`, `pshcor-<Z>.tab`, `eshcor-<Z>.tab`, `rmuon<Z>.tab` — plus the
Seltzer-Berger bremsstrahlung tables `pdebr<Z>.p08` that D8 and #86 want, and
five shared files: `atparams.tab`, `exp-param.tab`, `pdatconf.p14`,
`pdcompos.pen` and `shparams.tab`. Six families of 99 plus five shared is the
599-file total above.

`pdcompos.pen` carries SBETHE's own catalogue of 280 predefined materials
(indexed by `material-list.txt`).

**Decided 2026-09-21: PyRITE's materials catalogue stays the source, for now.**
`xsgen` always supplies composition, density and mean excitation energy
explicitly and never passes an SBETHE material ID. Keying off those IDs would
introduce a second identity namespace for the subset of materials both
catalogues happen to contain, which is exactly the split D2 exists to prevent —
and it would make a table's key depend on an upstream numbering PyRITE does not
control.

The 280 entries remain useful as a **cross-check oracle**: for any material in
both catalogues, SBETHE's composition, density and mean excitation energy are an
independent second opinion on ours. Worth a test; not worth a dependency.

Revisit only if a concrete disagreement shows up — if our I-value for a shared
material diverges from SBETHE's enough to move stopping power outside #90's
tolerance, that is a finding about our catalogue, and the question of which one
is authoritative becomes real rather than hypothetical.

#### Interface

Material-scoped: needs composition, density, and mean excitation energy, and
emits `<mname>.mat`. Couples to `materials/catalog.py` through the material
identity hash of D2. `sbethe/mos2.mat` in the upstream tree is present but
empty.

Outputs, from the `OPEN` statements in `sbethe.f`:

| file | content |
| --- | --- |
| `stp.dat`, `stp-low.dat`, `mstp.dat`, `lstp.dat`, `PENstp.dat` | stopping powers on various grids |
| `asymptotic.dat` | `CS0A` total inelastic cross section [cm²], `CS1A` stopping cross section [eV cm²], `CS2A` energy-straggling cross section [(eV cm)²] |
| `OOS.dat` | oscillator strengths |
| `depth-dose.dat` | depth-dose curve |

Two outputs beyond #90's stopping power are worth claiming:

- `CS0A`/`CS1A`/`CS2A` give the inelastic mean free path, stopping, and
  straggling from one consistent DHFS/PWBA evaluation — the inputs #93
  (soft/hard microscopic inelastic transport) needs, obtained as a byproduct
  of #90.
- SBETHE's radiative stopping is built from Seltzer–Berger bremsstrahlung
  tables (`sdbase/pdebr*.p08`), the independent reference #86 already wanted.
  With BremsLib and EEDL that gives #86 three independent bremsstrahlung
  sources from codes already being integrated.

## Ionization cross sections are out of scope for all three codes

None of the three can replace EEDL's subshell electroionization cross
sections (MT 534–572), which drive characteristic line production:

- ELSEPA is elastic only.
- BremsLib is bremsstrahlung only.
- SBETHE emits a **material-level total inelastic** cross section summed over
  subshells (`CS0A`), not subshell-resolved σ_ion. Its OOS database is
  subshell-resolved and its PWBA machinery computes generalized oscillator
  strengths (`rpwba.pdf`), but the program's outputs are material sums.
  Extracting per-subshell cross sections would mean reworking the code's
  internals, not parsing its output.

Shell ionization stays with EEDL, with Bote–Salvat (#92) as the alternative
backend. The single-source policy of D6 covers bremsstrahlung only.

### BremsLib → #86, #87, #95

No runner. The `Brems.exe` path recomputes what `BremsLib_v2.0/` already
contains; PyRITE reads the precomputed library in place and interpolates (D7).
Supplies both SDCS and shape function per D6.

Located at `../BremsLib_v2.0.8` relative to the PyRITE checkout by default,
overridable through `xsgen.sources`. Neither the GPL-3 Fortran nor the 810 MB
data library is vendored or redistributed; **tables derived from the library
are shipped** for the built-in catalogue, attributed and marked as adaptations
per D4 (revised) and the Licensing section.

So BremsLib-backed bremsstrahlung is available to every user for catalogue
materials, and requires a local checkout only for materials outside it.
`"eedl"` remains selectable for regression and continuity, and stays the
packaged default until #86 demonstrates the accuracy claim — now on accuracy
grounds alone.

## Testing

CI cannot run Fortran. Layers:

- Parser tests against committed vendor reference outputs —
  `elsepa-2020/test-run-output/dcs_1p000e03.dat` and `Brems/Test_runs/` are
  small and published.
- Subprocess/scratch-directory/symlink layer tested against a fake binary.
- Converter round-trip test asserting bit-for-bit ASCII reproduction under
  `%.8e` (D8).
- An `extern_codes` pytest marker beside the existing `hardware`, `oracle`,
  and `intel_sycl` markers for tests that actually compile and run the codes.

## Validation and ledger

Each generated table needs a `Validation: <id>` marker and a ledger row per
the repository physics contract. The natural anchor is reproducing each
vendor's own published test-run output, which is cheap and strong. New or
changed physics still requires source equation, assumptions, limiting case,
marker, and ledger row; only a human marks `signed-off`.

## Sequencing

1. `THIRD-PARTY-NOTICES.md`, `pyproject.toml` license metadata, import-linter
   contract updates.
2. `xsgen` skeleton: `store`, `sources`, `toolchain`, `_run`, `pyrite tables`.
3. ELSEPA (#89) — most mechanical, consumer already wired, vendored and
   offline.
4. SBETHE (#90) — needs the material-identity coupling.
5. BremsLib converter and reader (#86 validation, then #87, then #95).
   `BREMSSTRAHLUNG_MODEL` becomes selection-dependent (D5) in the same change
   that exposes the model selector, not before.

## Open items

- **Decided 2026-09-21: ship BremsLib-derived tables** for the built-in
  catalogue. Recorded in D4 (revised) and the BremsLib section; supersedes the
  earlier blanket no-redistribution position, which rested on dataset terms
  that are now confirmed CC BY 4.0. The GPL-3 sources and the 810 MB library
  remain non-redistributable, and D7's no-port rule is untouched.

  Still to settle inside that decision:

  - **Unmeasured:** the on-disk size of the derived SDCS + shape-function
    tables across the built-in catalogue, and therefore the real wheel cost.
    D4's growth figure (≈4.8 MB) predates this decision and does not include
    them. Measure before committing tables, not after.
  - **Unbuilt:** the maintainer-side generation path. Shipped tables have to
    be produced by someone holding a BremsLib checkout and refreshed when the
    upstream deposit versions. That is a release step this spec does not yet
    describe, and it needs the provenance manifest to record the deposit
    version (V9, `10.17632/6zfsc9xsz8.9`) so a stale table is detectable.

- Whether the `xsgen` scratch directory should fall back to copying when the
  filesystem does not support symlinks (Windows without developer mode, some
  network mounts). Symlinks are the plan of record; a copy fallback is cheap
  but untested against the codes' relative-path assumptions.
