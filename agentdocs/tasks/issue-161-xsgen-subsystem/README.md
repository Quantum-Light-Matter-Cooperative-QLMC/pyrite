# issue-161-xsgen-subsystem

`pyrite.xsgen`: generate, store, and resolve cross-section and stopping-power
tables produced by external Fortran codes (ELSEPA, SBETHE, BremsLib).

GitHub issue: #161. Branch: `issue-161-xsgen-subsystem`.
Worktree: `/tmp/pyrite-issue-161-xsgen-subsystem` (reattached here because the
standard worktree directory is read-only to the current editing sandbox).
Design: `agentdocs/specs/2026-09-21-external-fortran-code-integration.md`.

Prerequisite #162 (third-party notices, license metadata, import-linter
`forbidden_modules` entries, `extern_codes` marker) landed on `main` at
`5e12f8f2`; this branch starts from it.

## Session authority

Direct `/lead-task` invocation. Task-local checkpoint commits and
branch/worktree creation only. **No push, no GitHub issue edits, no
delegation.** Issue checkboxes stay unticked; this file is the local record.

## Milestones

#161 is not one slice. Decomposition, in the spec's sequencing order:

| M | Scope | State |
| --- | --- | --- |
| M1 | `sources.py`, `toolchain.py`, `_run.py`, `store.py` + fake-binary tests | complete |
| M2 | `pyrite tables` group + `cli-reference.md` regeneration | complete |
| M3 | manifest hash into run identity; stale checkpoints cannot be served | complete |
| M4 | vendor ELSEPA + `sbethe.f`, `.gitattributes` `-text`, `tables fetch` | complete |
| M5 | `elsepa/`, `sbethe/`, `bremslib/` deck+parse + vendor-reference parser tests | complete |
| M6 | shipped BremsLib-derived tables + maintainer refresh path | gated, see below |

M6 is gated on a measurement the issue requires *before* committing tables:
derived-table size across the built-in catalogue. That measurement is now
taken -- see F13 -- and it needs a human decision before any table is
committed.

## Local environment

The vendored ELSEPA/SBETHE sources and a local BremsLib checkout are available:

- `src/pyrite/data/xsgen/{elsepa,sbethe}` and `/home/alex/dev/BremsLib_v2.0.8`
  (the BremsLib anchors need no compiler, only the library; point at it with
  `PYRITE_XSGEN_BREMSLIB_SOURCE` when running from this worktree, whose parent
  is `/tmp`)
- gfortran 15.2.0 was available again from the M5b session onward, so both
  codes were built and run for real; SBETHE's `sdbase/` was fetched through
  `pyrite tables fetch sbethe` against the re-pinned deposit.

## Findings against the spec

### F1 — D2's material identity hash does not exist

D2 says the material identity hash "comes from the existing
`pyrite.materials._identity`". It does not. That module derives a *display*
identity -- formula, phase, full name, slab cut, bracket notation -- and
contains no hash of any kind. Nothing else in `materials/` hashes a material
record either (`atomic.py` blake2b digests are energy-grid cache keys).

Consequence for the store key: the `Material` arm of D2 needs a hash defined by
this issue, over the fields the Fortran decks actually consume -- composition,
density, mean excitation energy, and for muffin-tin ELSEPA the
nearest-neighbour distance and band gap -- not over the display label. Hashing
the catalogue key alone would be wrong: it would not change when the
catalogue's composition or density for that key changed.

### F2 — the identity marker must be conditional, not a constant

`campaign/profiles.py:_identity_v1` records model markers as *unconditional*
constants (`stopping_model`, `characteristic_model`, `bremsstrahlung_model`,
`line_kinematics`). Each such addition perturbs every digest exactly once and
orphans every existing checkpoint -- the file documents this as a deliberate
rev-and-re-run cost paid when the physics actually changed.

No consumer resolves an xsgen table until #89/#90. Adding an unconditional
xsgen marker now would orphan every checkpoint in exchange for zero change in
computed numbers. So M3 follows the *divergence-only key* pattern already in
the same function (`if emission != "incoherent"`, `transport_numerics`): the
marker appears only when a run actually resolves an xsgen table, and a run that
resolves none keeps its `parameter_sha256` bit-for-bit.

D5's `BREMSSTRAHLUNG_MODEL` prerequisite is untouched by this and stays with
#86/#87, per the spec: it must land in the same change that exposes the model
selector.

### F3 - ELSEPA compiles from `elscata.f` alone

`elscata.f` opens with `INCLUDE 'radial.f'` and `INCLUDE 'elsepa2020.f'`.
Naming all three on the compiler command line compiles the included bodies
twice and the link fails on dozens of duplicate symbols (`sfas0_`, `dbas_`,
`splset_`, ...). The build therefore passes `elscata.f` only, plus
`-I<tree root>` so the INCLUDE lines resolve from the cache directory the
build runs in.

All three files stay in `digest_sources`: they are part of the binary, so
patching `radial.f` is a different code even though the compiler is never
pointed at it directly.

Found by the `extern_codes` anchor, not by review. The spec's architecture
section does not mention it.

### F4 - the symlink copy fallback is implemented

The spec leaves "should the scratch directory fall back to copying when the
filesystem refuses symlinks" open. Implemented in the safe direction:
`_link_dir` tries `symlink_to` and falls back to `copytree`, recording which
was used in `RunResult.link_modes`. The codes only ever `OPEN` paths *beneath*
the directory, so a copy is behaviourally identical and differs only in cost.
Still untested against a real symlink-refusing filesystem.

### Vendor anchor result

`elscata.in` (Z=80, the shipped deck) reproduces
`test-run-output/dcs_1p000e03.dat`: 649/649 lines, 24 differing only in the
last of six significant digits, confined to the interference column
(~1e-9 against a 1e-12 to 1e-15 DCS scale, 1 part in 5.6e5). That is a libm
difference between our gfortran 15.2.0 and the vendor's compiler, so the
anchor compares numerically at `rtol=1e-5` -- one last-digit unit at six
significant figures -- rather than byte-for-byte. A byte comparison would be
an assertion about gfortran, not about ELSEPA.

### F5 - run identity has two surfaces, not one

D5 names `campaign/profiles.py` and `api.py`. The identity that gates
*checkpoint stems* is `_identity_v1` -> `parameter_sha256`, but there is a
second, independent cache: `case_content_key` keys the content-addressable
blob store (`checkpoints/_checkpoint_store.py`, `runs/run.py`,
`checkpoints/persistence.py`). Its own docstring already records why the
stopping/characteristic/bremsstrahlung constants are in it -- "without them a
blob from an earlier physics/data generation could be served silently".

A table marker in `dataset_identity` alone would move the stem while the blob
store still served arrays computed from the superseded table. Both take
`xsgen_tables`, and both take it conditionally.

`api.py`'s `bremsstrahlung_model` entries are provenance *record* fields
written into results, not identity inputs, so they need nothing here.

### F6 - the manifest had to cover the payload, not just the recipe

The first `identity_markers` test failed: regenerating a table with different
numbers produced the *same* manifest digest. Every "how it was made" field
agreed (same code, deck, shapes, dtypes) and `created_utc` has one-second
resolution, so two writes in the same second collided -- and that digest is
exactly what invalidates checkpoints.

The manifest now carries `arrays_sha256` over the stored arrays, normalized to
little-endian so a shipped manifest verifies on a big-endian host. Found by
the test, not by review.

### F7 - the pinned SBETHE archive contains 599 sdbase files

The issue and spec say 498 files, but the extracted archive pinned by SHA-256
contains 599: six 99-element families plus five shared files. Fetch validation
therefore checks the archive digest and required shared markers rather than an
incorrect file-count constant. The CLI reports the observed count.

### F8 - the SBETHE deposit had to move from version 1 to version 2

The issue, and this branch through M4, pinned deposit version 1
(`10.17632/7zw25f428t.1`). That pin is wrong for the source we vendor.

Both versions were downloaded and compared on 2026-09-21:

| | v1 | v2 |
| --- | --- | --- |
| `sbethe.zip` | 12 046 759 B | 12 878 659 B |
| SHA-256 | `693d447d...0b70845d` | `d5d4879c...5c3c4144` |
| `sdbase/` files | 497 | 599 |
| `atparams.tab`, `exp-param.tab`, `shparams.tab` | absent | present |

The vendored `src/pyrite/data/xsgen/sbethe/sbethe.f` is byte-identical to v2's
copy (`c2eda61b...cc1b12a7`), and it `OPEN`s `./sdbase/exp-param.tab`
(line 424) and `./sdbase/atparams.tab` (line 2243). Neither file exists in v1.
Fetching v1's `sdbase/` for the source we actually compile would therefore fail
part-way through a Fortran run, not at install time.

So the version bump is a correctness fix, not a refresh. Consequences:

- `_REQUIRED_SBETHE_FILES` gained `atparams.tab`, `exp-param.tab` and
  `shparams.tab`. The first two are the decisive markers -- they are the ones
  `sbethe.f` opens and the ones v1 lacks -- so a leftover v1 directory is now
  rejected by `_installed_sbethe` instead of being served as complete.
  `shparams.tab` is shared and v2-only but never opened; it is a completeness
  marker only.
- `_download` now sends an explicit `User-Agent`. The deposit URL answers 302
  to an S3 object; `urlopen` follows that by default, and the real fetch was
  verified end to end against both the API-published digest and the bytes.
- F7's 599-file count is a v2 figure. v1 holds 497, so the issue's "498" was
  never right for either version.
- Spec figures re-measured from the v2 archive rather than carried over:
  `sdbase/` 18.8 MB, `docs/` 8.6 MB, `sbethe.exe` 1.16 MB, 28.7 MB extracted
  from a 12.9 MB download. The sdbase inventory is five per-element families
  plus `pdebr<Z>.p08` and five shared files, not "four families and two global
  files".

The issue body still records the v1 digest, `10.17632/7zw25f428t.1` and 498
files. Correcting it needs issue-write authority this session does not hold.

### F9 - two M5a defects sat behind the anchor's shape

`pyrite tables generate --code elsepa` did not work on the committed branch,
and neither did `generate_element`. The M5a anchors passed anyway because they
drive `run_program` with a deck spelled out in the test, so nothing ever fed
`ElsepaDeck.render()` to `elscata`.

1. `elscata` reads every deck line as `(A6,1X,A12)`. The value field is
   *twelve* characters and a longer one is truncated, not rejected, so
   `EV     1.00000000E+03` arrived as `1.00000000E` and the program stopped
   with "Bad real number in item 1 of list input". Energies now render at
   `.5E` -- six significant digits, which fits, and which is the precision the
   `dcs_*.dat` names already carry.
2. `parse_dcs` required an `Absorption cross section` scalar. `elscata` writes
   that line only for `MABS > 0`, and the deck's default is `MABS 0`, so the
   common case has no such line. It now defaults to `0.0`: no absorption
   potential means no absorption, not a missing field.

Both are covered by unit regressions plus a new `extern_codes` anchor that
drives the real binary from the generator's own rendered deck, for both codes.
The issue's first acceptance bullet was therefore not met before this slice,
despite M5a being recorded complete.

### F10 - SBETHE's input is a prompt sequence, established by driving it

SBETHE reads no deck file: it consumes the answers a user would type, and
several prompts appear only given an earlier answer. The sequence was
established by building the program and driving it, not by reading its source,
because the conditional prompts are not obvious from the `READ` statements.

The order, for the keyboard-composition branch `xsgen` always takes:

| # | answer | note |
| --- | --- | --- |
| 1 | material name | `A15`, no blanks; also names the `<name>.mat` it writes |
| 2 | `1` | composition from the keyboard, never a `pdcompos.pen` ID |
| 3 | number of elements | |
| 4 | `1` | stoichiometric formula; **omitted when there is one element** |
| 5 | `Z N` per element | one line each; bare `Z` in the single-element branch |
| 6 | mass density | |
| 7 | `Y` | always override the proposed I |
| 8 | mean excitation energy | must exceed 1 eV |
| 9 | `Y`/`N` | insulator or semiconductor |
| 10 | band gap | **only when 9 is `Y`** |
| 11 | projectile number | |

A wrong order does not fail loudly: the program reads the next answer for a
different question and tabulates a material nobody asked for. The
`extern_codes` anchor therefore asserts that the material SBETHE echoes into
its output headers is the one the deck described.

Outputs parsed: `stp.dat` (collision stopping), `asymptotic.dat` (the
`sigma^0`/`sigma^1`/`sigma^2` moments #93 wants, on a grid reaching below
`stp.dat`'s corrected-Bethe floor) and `OOS.dat`. `stp-low.dat` holds the
sub-ECUT empirical extrapolation and is a different quantity, so it is not
concatenated onto `stp.dat`.

Two details found only by running it:

- `OOS.dat` repeats an abscissa at each shell binding edge, carrying the step
  in the oscillator-strength density there. Its grid is non-decreasing, not
  strictly increasing; the other two are strict.
- `sdbase/shparams.tab` is never opened by `sbethe.f`. It stays in the
  completeness markers as a v2 marker only; `atparams.tab` and `exp-param.tab`
  are the ones that matter (F8).

Against published values, SBETHE's liquid-water collision stopping at I=75 eV
reproduces ESTAR to well inside 2% at both 1 keV and 1 GeV. That is anchored
as evidence the right column in the right units reached the caller; the
physics comparison itself belongs to #90.

### F11 - the BremsLib source marker never matched a real checkout

`sources.py` accepted a BremsLib tree on `BremsLib_v2.0` existing inside it.
The unpacked deposit holds `BremsLib_v2.0.8/` -- the patch version is part of
the directory name, and the manual's own prose ("main library data folder
BremsLib_v2.0") is what the marker was written from. So `resolve_source
("bremslib")` failed on the real checkout at `../BremsLib_v2.0.8`, and the
only test covering it built a directory named after the marker rather than
after the deposit.

Fixed by generalizing `CodeSpec.marker` to `markers: tuple[str, ...]`, with
glob patterns allowed and any one match accepting the tree. BremsLib now takes
`("BremsLib_v2.0*/SDCS/SDCS_*.txt", "SDCS/SDCS_*.txt")`: a data file rather
than a source file, because a tree holding the GPL-3 codes but not the library
cannot answer a single request, and two patterns because a user who kept only
the data directory should not be told they have no BremsLib.

Not found by review -- found by pointing the generator at the real checkout.

### F12 - Simpson, not the trapezoid, reproduces the vendor's own integrals

The shape function is the DDCS over its angular integral, so the integration
rule is the whole of the derived physics. Upstream publishes its own integral
of every node (`SDCS/DDCS_int/DDCS_int_<Z>.txt`, 10th-order Newton-Cotes),
which makes the rule checkable rather than a matter of taste:

| rule | agreement with the published integral |
| --- | --- |
| trapezoid on the native grid | 1.1e-4 relative |
| composite Simpson on the native grid | 2.0e-5 worst, 3.4e-9 median |

Measured over all 858 nodes of Z=79 below 30 MeV and at sampled nodes from
Z=1 to Z=100. The native angular grids have 181, 221 or 441 points -- always
an even number of intervals -- so composite Simpson applies across the whole
range, and `scipy.integrate.simpson` handles their non-uniform spacing.
Simpson it is, anchored at `rtol=1e-4`.

Two smaller things the same comparison settled:

- The vendor's `DDCS_int` file carries a *shorter* energy grid than the SDCS
  file (integrals to 100 MeV, cross sections to 300 MeV). Reusing the SDCS row
  index would read a different energy's integral, and nothing in the stored
  numbers would show it. There is a regression for it.
- Node file names are not sortable by photon energy: a two-digit exponent puts
  `8.000E-01` after `2.000E+00`. The first anchor written did sort by name, and
  compared Z=100's top node against a DDCS 2.5x away from it.

### F13 - the M6 size measurement: 10 to 55 MB, not 4.8 MB

The issue requires the derived-table size across the built-in catalogue to be
measured before any table is committed. It is, with the converter that landed
here.

The catalogue holds 50 materials over **24 distinct elements**
(Z = 5, 6, 7, 8, 13, 14, 15, 16, 22, 23, 26, 32, 34, 40, 41, 42, 46, 52, 72,
73, 74, 75, 78, 83). Table size is essentially Z-independent, because the
node and angular grids depend on `T1` alone -- Z=6 and Z=79 differ by 4%:

| variant | per element | x24 elements |
| --- | --- | --- |
| stored schema, T1 <= 30 MeV | 2.3 MB | 55 MB |
| float32 DDCS, uncertainty columns dropped | 0.75 MB | 18 MB |
| stored schema, T1 <= 1 MeV | 1.3 MB | 32 MB |
| float32, no uncertainties, T1 <= 1 MeV | 0.46 MB | 11 MB |
| float32, no uncertainties, T1 <= 0.3 MeV | 0.41 MB | 10 MB |
| SDCS only, no angular data | 8 kB | 0.2 MB |

`.npz` deflate, measured. Bounding the energy range buys less than it looks
like: 50 of the 66 grid energies below 30 MeV are below 1 MeV, so the
low-energy nodes are most of the payload.

So every variant that ships angular information costs 10 MB or more -- two to
eleven times the spec's whole ~4.8 MB wheel-growth projection, which D4 (revised)
already noted excluded these tables. Only the SDCS-only table fits that
figure, and shipping the SDCS alone would leave emission isotropic for a user
without a checkout, which is the thing D6 and #87 exist to fix.

The decision belongs to a human. The options, as measured:

1. Accept a 10-18 MB wheel for float32 angular tables over the catalogue.
2. Ship the SDCS only, and require a checkout for the angular model.
3. Fetch derived tables on demand into the user data directory, the way
   SBETHE's `sdbase/` already is (`pyrite tables fetch`), so the wheel does
   not carry them but a user without a BremsLib checkout still gets them.
   The spec does not list this option; the machinery for it exists.

M6 stays gated, and no table is committed.

### F14 - a float32 payload is not free, and the store does not know about it

Two of the three options above store the DDCS as float32. `store.arrays_digest`
hashes dtype and bytes, so a dtype change re-keys every table -- which is
correct, but it means the shipped-table dtype is a decision to take once,
before tables exist, not after.

The converter stores float64 DDCS today because that is what the library
publishes (`%.8e`, which float32 does not hold) and because a locally
generated table costs nothing to keep exact. If M6 ships float32, it should be
a *shipped-table* transform in the maintainer path, not a change to what
`build_table` produces.

## Checklist

- [x] M1 `sources.py` / `toolchain.py` / `_run.py` / `store.py`
- [x] M1 tests: scratch isolation, symlinks, cwd, fake binary; store round-trip
- [x] M1 `extern_codes` anchors against the real ELSEPA tree (opt in with
      `PYRITE_EXTERN_CODES_TESTS=1`; ~84 s)
- [x] M2 `pyrite tables` group (`path`, `list`, `show`, `sources list|set`),
      `cli-reference --check` clean. `generate`/`fetch` deferred to M5/M4.
- [x] M3 conditional identity marker on both surfaces + regression tests
- [x] M4 vendored ELSEPA source/database/reference anchor and `sbethe.f`
- [x] M4 pinned `tables fetch sbethe`, selective/atomic extraction, fetched-data overlay
- [x] M5a ELSEPA free-atom deck writer, vendor-output parser, native-grid CDF,
      cached generation path, `tables generate`, CLI reference, and regression anchors
- [x] M4 SBETHE deposit re-pinned to version 2 after a two-version comparison (F8)
- [x] M5b SBETHE deck writer, three output parsers, `generate_material`,
      `tables generate --code sbethe`, CLI reference, and real-binary anchors
- [x] M5b fixed two M5a defects that blocked `tables generate` entirely (F9)
- [x] M5c BremsLib library reader (`read.py`), converter (`convert.py`),
      per-element generator, `tables generate --code bremslib`, CLI reference,
      `bremslib-library-reference` ledger row, unit tests and real-library
      anchors
- [x] M5c fixed the BremsLib source marker, which never matched a real
      checkout (F11)
- [x] M6 gate measurement taken: 24 catalogue elements, 10-55 MB depending on
      variant (F13). Decision is the human's; nothing committed.
- [x] import-linter contracts pass with `xsgen` populated (11 kept, 0 broken)
- [x] `pyrite-dev verify` for M1-M3 (4361 passed, 90 skipped)
- [x] `pyrite-dev verify` after M4 (4369 passed, 90 skipped)
- [x] After M5b: 4456 passed, 55 skipped; lint, typecheck and
      `cli-reference --check` clean; import-linter 11 kept, 0 broken. The 6
      failures are pre-existing and environmental -- `dpctl`/`cupy` absent
      against this shell's `PYRITE_MC_BACKEND=sycl`, and `mp_api` absent for
      the live materials-database comparisons.
- [x] All 9 `extern_codes` anchors pass against real gfortran 15.2.0, the
      vendored ELSEPA tree and the fetched SBETHE `sdbase/`
