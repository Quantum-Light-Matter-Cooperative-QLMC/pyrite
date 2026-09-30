# 0014 — Packaged data layout

- **Status:** Proposed
- **Date:** 2026-09-30
- **Issue:** #263. Measurements are in
  [Data distribution and repository size](../repo-design/data-distribution-and-repository-size.md).

## Context

`src/pyrite/data` is about 46 MiB installed and about 16.1 MiB of the 17.65 MiB
wheel. Three unrelated kinds of data share it:

- runtime inputs,
- upstream datasets that are already pinned by SHA-256 to public URLs,
- generator inputs that only `pyrite tables generate` reads.

Tables follow two different policies. The 49 SBETHE tables are packaged. The
ELSEPA and BremsLib tables are fetched through the hash-pinned
`xsgen/*-tables.json` release indexes, whose archive URLs are still `null`
(#167).

Table resolution searches three tiers: the workspace, the legacy
`~/.local/share/pyrite/xsgen/tables`, then the packaged directory. The lab
box's tables currently sit only in the legacy tier (#261).

A default run already needs fetched data. The default `elastic_model="elsepa"`
needs the ELSEPA tables, and every run needs EEDL (electroionization and the
bremsstrahlung fallback) and the SBETHE stopping tables.

Geant4 and pooch keep large evaluated data separate from the code and pin it
by hash. PENELOPE, EGSnrc, and MCNP ship their data with the code. All five
treat a missing required dataset as fatal; see
[How other codes distribute data](../repo-design/data-distribution-and-repository-size.md#how-other-codes-distribute-data).

## Decision

Every `src/pyrite/data` entry belongs to exactly one of three classes.

- **(a) In-wheel runtime data.** Small data (at most about 1 MiB compressed
  per entry) that is authored in this repository or tightly coupled to the
  code version, and read by a run.
- **(b) Hash-pinned fetched dataset.** Upstream or upstream-derived data that
  is large (over about 1 MiB compressed) or re-keyed whenever it is
  regenerated. It is pinned by SHA-256 in an in-wheel index and installed once
  with `pyrite tables fetch …` (or `--archive PATH` offline) into the
  workspace data root. The box receives it through content-addressed sync
  (#261). A missing required dataset is an error that names the fetch command
  (#262). No unpinned download is allowed.
- **(c) Generator input.** Data that only table generation reads. It is not in
  the wheel. It ships with the source checkout and the sdist, and reaches the
  box through code sync.

Licence gates apply on top of the classes. Anything under CC BY or CC BY-NC
ships only with its `THIRD-PARTY-NOTICES.md` entry. A derived table ships only
with a provenance manifest that carries a modifications note. BremsLib GPL-3
sources and the BremsLib library are never redistributed. Anything whose
redistribution terms are unconfirmed must not ship.

```{list-table} Proposed classification of each data entry. The Now column gives today's class; "wheel" means the file ships in the wheel without a clean fit to any class.
:name: tbl-adr0014-classification
:header-rows: 1

* - Entry
  - Now
  - Proposed
  - Basis
* - `catalog/`, `cifs/`, `line_grid_defaults.toml`, `line_grid_provenance.toml`, `conduction_band.toml`, `eaglexo_qe.csv`
  - a
  - a
  - Authored here, under 0.1 MiB, read by every run.
* - `xsgen/elsepa-tables.json`, `xsgen/bremslib-tables.json` (and a future `sbethe-tables.json`)
  - a
  - a
  - The pin registry (the pooch-registry analogue) must travel with the code.
* - `xsgen/tables/` (49 SBETHE tables)
  - a
  - b
  - 2.6 MiB and re-keyed on every regeneration, so each regeneration adds permanent Git history. Same policy as ELSEPA and BremsLib.
* - `characteristic_cross_sections/EEDL.endf`, `EADL2025.ALL`
  - a
  - b
  - 8.7 MiB compressed (33 MiB installed), upstream-verbatim, already pinned to the public LLNL URLs, CC BY 4.0.
* - `xsgen/elsepa/` sources and `database/`
  - a
  - c
  - Read only by `tables generate --code elsepa`.
* - `xsgen/sbethe/sbethe.f`
  - a
  - c
  - Read only by `tables generate --code sbethe`.
* - `xsgen/elsepa/test-run-output/dcs_1p000e03.dat`
  - a
  - test fixture
  - Read only by tests.
* - `BELLS_gpt.out.gdf`
  - wheel
  - out of the wheel
  - Not read by `src/`, `tests/`, or `checks/`; only the GPT guide uses it as an example. Provenance and licence are not recorded.
* - `atomic_scattering_factors/`
  - wheel
  - retire
  - No code reads it; the physics docs already call it legacy and unused.
* - `mott_transport_cross_sections/`
  - a
  - a until retired
  - Read by the opt-in `elastic_model="mott"` and by the catalog warning. Licence is unresolved (open decision 6).
```

For code sync, "source checkout" means that class (c) moves to a repository
path outside the package, for example `vendor/xsgen/{elsepa,sbethe}/`. That
path is added to `pyrite remote` `SYNC_PATHS`. It is resolved as a new
"checkout vendor" tier, placed where the wheel-vendored tier sits today. D4's
"vendored ⇒ offline zero-config generation" guarantee is restated as: offline
generation works from any checkout, any sdist, and the box. A bare wheel
install needs `xsgen.<code>_source` or a sibling checkout. `gfortran` is a
prerequisite in all cases.

## Recommendations on the open decisions

1. **SBETHE vs ELSEPA vs BremsLib tables.** Fetch all three through release
   indexes. Move SBETHE only after #167 hosts the archives and #261 syncs
   tables to the box. Until then, SBETHE stays packaged, because moving it
   first would break default runs on a fresh install. Rejected alternative:
   package all three. ELSEPA and BremsLib add 48 MB of archives, and every
   future regeneration would permanently grow Git history.
2. **EEDL and EADL.** Make them a fetched dataset from the LLNL URLs pinned in
   the characteristic-cross-sections README. These are the only class (b)
   entries with a working public URL today. Move them after #261 so the box
   receives them once rather than in every 19.9 MiB sync.
   The runner-up is to keep them in the wheel. That is zero-configuration
   offline, and the data changes only on a deliberate EPICS update. But it
   keeps 8.7 MiB compressed in every wheel and every sync, and it leaves the
   class rule with an exception.
3. **ELSEPA sources and D4.** Move them to class (c) as described above, and
   move `test-run-output` to `tests/data/`. Keep the source-digest pins
   unchanged. The paths change, but the bytes do not.
4. **BELLS gdf.** Take it out of the package. Move it to `examples/gpt/`
   only if its owner (commit 02d89fb5) confirms it may be redistributed.
   Otherwise remove it and have the guide name where to obtain it.
5. **Legacy directories.** Remove `atomic_scattering_factors/`. Keep
   `mott_transport_cross_sections/` exactly as long as `elastic_model="mott"`
   remains selectable, unless open decision 6 forbids redistribution.
6. **Licence gaps found (not resolved here).**
   - `EEDL.endf` is redistributed but has no `THIRD-PARTY-NOTICES.md` entry.
     The EPICS2025 data is CC BY 4.0 (distribution page), so attribution is
     owed.
   - The EADL notice still says "the current model does not read it"; it now
     does.
   - The Mott tables are NIST SRD 64 output. NIST states that SRD "may not be
     reproduced … without prior permission"
     ([NIST SRD law](https://www.nist.gov/srd/public-law)). Whether computed
     output tables count as SRD is unconfirmed.
   - The CXRO factors and the BELLS gdf have no recorded terms.
7. **Legacy `~/.local/share/pyrite/xsgen/tables` tier** (and the legacy
   `reference-data` sdbase path). Keep it read-only until #261 ships its
   one-time migration. Then have a resolution from that tier warn for one
   release, and remove the tier afterwards.
8. **Repository history.** Choose option A: leave history as it is. Moving
   entries out of the tree cuts shallow and blobless clones from about
   23–25 MiB to about 7–9 MiB. Recommend `git clone --filter=blob:none` for
   new clones and run `git gc` on the local object store (53 MiB in 17 packs
   plus 35 MiB loose).
   A `filter-repo` rewrite (option B3) would take the full clone from
   41.4 MiB to about 16 MiB. It is rejected: it would orphan the SHAs in 20
   committed validation check-record files, remote `code_revision` metadata,
   260 PRs, and four worktrees, and GitHub Support must purge 94 `refs/pull/*`
   refs. No Git LFS: class (b) already keeps large data out of Git, and LFS
   would add a client dependency and metered bandwidth. Revisit only if a file
   over 5 MiB must be tracked.

## Consequences

Projected wheel sizes, taken from the per-entry compressed sizes:

```{list-table} Projected wheel size.
:name: tbl-adr0014-wheel-projection
:header-rows: 1

* - Stage
  - Wheel, compressed (MiB)
  - Installed (MiB)
* - Before
  - 17.65
  - 50.6
* - Recommendations 3–5
  - about 13.0
  - about 40.3
* - Plus recommendation 1
  - about 10.3
  - about 37.5
* - Plus recommendation 2
  - about 1.6
  - about 4.7
```

- A clean install needs one documented step, `pyrite tables fetch` with all
  datasets or `--archive DIR` offline, before a default run. Today that step
  is already needed for ELSEPA. Missing data fails with a named fix, following
  Geant4 and MCNP practice.
- Offline generation depends on a checkout, not on the wheel.
- Full clones do not shrink, but the growth caused by table regeneration
  stops.
- Implementation is sequenced after #261, which owns table sync, and #167,
  which owns hosting. The strict-failure wording belongs to #262.

## Decisions required from a human

1. Approve the three-class rule and the ~1 MiB in-wheel threshold.
2. EEDL/EADL: fetched (recommended) or kept in the wheel.
3. SBETHE tables: fetched after #167/#261 (recommended) or kept packaged.
4. ELSEPA and SBETHE generator sources: move to `vendor/` outside the package,
   restating D4 (recommended), or stay in the wheel.
5. BELLS gdf: its owner confirms the terms and it moves to `examples/`, or it
   is removed.
6. Licences: approve adding the EEDL notice and fixing the stale EADL notice.
   Decide the Mott tables (seek NIST permission, retire them with the `mott`
   model, or accept the risk). Decide the CXRO factors (retire them, as
   recommended).
7. History: leave it as it is (recommended), or explicitly approve a
   `filter-repo` rewrite, run as a separate task with a force-push window.
