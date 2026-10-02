# 0014 — Packaged data layout

- **Status:** Accepted
- **Date:** 2026-09-30
- **Issue:** #263. Measurements are in
  [Data distribution and repository size](../repo-design/data-distribution-and-repository-size.md).

## Context

Before this decision, `src/pyrite/data` was about 46 MiB installed and about
16.1 MiB of the 17.65 MiB wheel. Three unrelated kinds of data shared it:

- runtime inputs,
- upstream datasets already pinned by SHA-256 to public URLs,
- generator inputs that only `pyrite tables generate` reads.

Tables followed two different policies. The 49 SBETHE tables are packaged.
The ELSEPA and BremsLib tables are fetched through the hash-pinned
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

Every packaged data entry belongs to exactly one of three classes.

- **(a) In-wheel runtime data.** Small data (at most about 1 MiB compressed
  per entry) that is authored in this repository or tightly coupled to the
  code version, and read by a run. It lives in `src/pyrite/data/`.
- **(b) Hash-pinned fetched dataset.** Upstream or upstream-derived data that
  is large (over about 1 MiB compressed) or re-keyed whenever it is
  regenerated.
  - It is pinned by SHA-256 in an in-wheel index.
  - It is installed once, with `pyrite tables fetch …` (or `--archive PATH`
    offline), into the workspace data root, and cached there.
  - The box receives it through content-addressed sync (#261).
  - A missing required dataset is an error that names the fetch command
    (#262).
  - No unpinned download is allowed.
- **(c) Generator input.** Data that only table generation reads. It lives in
  `vendor/xsgen/<code>/`, outside the package, so it is not in the wheel. It
  ships with the checkout and the sdist, and reaches the box through code
  sync, because `vendor` is one of the remote `SYNC_PATHS`.

Licence gates apply on top of the classes:

- Anything under CC BY or CC BY-NC ships only with its
  `THIRD-PARTY-NOTICES.md` entry.
- A derived table ships only with a provenance manifest that carries a
  modifications note.
- BremsLib GPL-3 sources and the BremsLib library are never redistributed.
- Anything whose redistribution terms are unconfirmed must not ship.

**D4 restated.** The external-code spec's D4 said "vendored ⇒ offline
zero-config generation from any install". That guarantee now reads: offline
generation works from any checkout, any sdist, and any synced box. A bare
wheel install needs `xsgen.<code>_source` or a sibling checkout. `gfortran`
is a prerequisite in every case. Resolution order is unchanged: explicit path,
then configured path, then vendored tree (now `vendor/xsgen/<code>`), then
sibling checkout. Source digests cover the same bytes, so table keys do not
change.

```{list-table} Classification of each data entry.
:name: tbl-adr0014-classification
:header-rows: 1

* - Entry
  - Class
  - State after #282
* - `catalog/`, `cifs/`, `line_grid_defaults.toml`, `line_grid_provenance.toml`, `conduction_band.toml`, `eaglexo_qe.csv`
  - a
  - Unchanged. Authored here, under 0.1 MiB, read by every run.
* - `xsgen/elsepa-tables.json`, `xsgen/bremslib-tables.json`, `xsgen/sbethe-tables.json`
  - a
  - The pin registry (the pooch-registry analogue) travels with the code. Each records an ordered `archive.urls` list (#282); the digest decides what installs.
* - `xsgen/tables/` (49 SBETHE tables)
  - b
  - Hosted on the private release `tables-sbethe-1` and fetched with `pyrite tables fetch sbethe-tables` (#282). The packaged copy is removed once CI installs the hosted archives end to end.
* - `characteristic_cross_sections/EEDL.endf`, `EADL2025.ALL`
  - b
  - Moved out of the wheel (decision 2): fetched with `pyrite tables fetch eedl` / `eadl` into `<data root>/datasets/` from LLNL, then the private mirror releases `mirror-eedl2025-1` / `mirror-eadl2025-1` (#282); the README stays as provenance.
* - `photon_cross_sections/epdl2025_mf23.npz` (added by #274)
  - b
  - Hosted on the private release `tables-epdl-1` and fetched with `pyrite tables fetch epdl` (#282), the only location since it is derived data; provenance and modifications note travel in the release notes and its README. The packaged copy is read until it is removed once CI installs the hosted file end to end.
* - ELSEPA sources and `database/`; `sbethe.f`
  - c
  - Moved to `vendor/xsgen/elsepa/` and `vendor/xsgen/sbethe/`.
* - ELSEPA `test-run-output/dcs_1p000e03.dat`
  - test fixture
  - Moved to `tests/data/xsgen/elsepa/test-run-output/`.
* - `BELLS_gpt.out.gdf`
  - none (removed)
  - Removed in #264; the GPT guide now uses a user-supplied file or a generated fixture.
* - `atomic_scattering_factors/` (CXRO/Henke f1/f2)
  - none (removed)
  - Deleted; nothing read it.
* - `mott_transport_cross_sections/`
  - none (removed)
  - Deleted: NIST SRD 64 may not be redistributed (decision 6). `elastic_model="mott"` reads user-downloaded tables from `mott.tables_dir`.
```

### Accepted decisions

1. **Classes and threshold.** The three-class rule and the ~1 MiB compressed
   in-wheel threshold are adopted.
2. **EEDL and EADL** become a class (b) dataset. They will be fetched from the
   LLNL URLs pinned in the characteristic-cross-sections README and cached in
   the workspace data root.
   - *Implemented in #263 after #261:* `pyrite/datasets.py` pins both files;
     `pyrite tables fetch eedl|eadl` downloads them (or copies
     `--archive PATH`), verifies the SHA-256, and installs them at
     `<data root>/datasets/<name>/`, the data root being the one xsgen
     tables use. The EEDL, EADL, characteristic and bremsstrahlung loaders
     resolve them there, verify the pin before first use, and fail naming
     the fetch command. The pins and the model identity markers are
     unchanged.
   - The published EEDL file ends with one more CRLF than the vetted bytes
     the pin covers; the fetch accepts either and installs the vetted form
     (see the characteristic-cross-sections README).
   - `pyrite remote sync` ships both files once by content digest to
     `<REMOTE_DIR>/datasets/`, first adopting a pre-#263 code-synced copy on
     the box; the job preflight is
     `pyrite tables verify --require bremslib,elsepa,sbethe-tables,eedl,eadl,epdl`
     (#282).
   - Nothing downloads implicitly at run time: compute nodes may have no
     network.
3. **SBETHE tables** become class (b), with a `sbethe-tables.json` release
   index like ELSEPA's and BremsLib's. #282 hosts the archive on a private
   GitHub Release (interim to the Zenodo deposit, #167); the packaged copy
   goes only after CI installs it end to end, because moving them first would
   break default runs on a fresh install.
4. **ELSEPA and SBETHE sources** are class (c), in `vendor/`. `vendor` is
   added to `pyrite remote` `SYNC_PATHS`, and `test-run-output` becomes a test
   fixture. Implemented in #263.
5. **BELLS gdf** is removed from the package under #264, together with the
   GPT guide rework.
6. **Licences.**
   - The EEDL attribution is added to `THIRD-PARTY-NOTICES.md` (EPICS2025 is
     CC BY 4.0).
   - The stale EADL notice is corrected: the relaxation cascade now reads the
     file.
   - The ELSEPA and SBETHE notices now describe the source-distribution
     location.
   - `atomic_scattering_factors/` is deleted.
   - **Mott tables:** redistribution appears prohibited without permission.
     *Decided 2026-09-30 (#263), implemented:* the five CSVs are removed from
     the tree and the wheel and stay only in Git history (decision 7). The
     opt-in `elastic_model="mott"` is kept: it reads SRD 64 exports the user
     downloads into the directory named by the `mott.tables_dir` config key
     (`PYRITE_MOTT_TABLES_DIR`), and a run fails naming that key and the
     element when a table is absent. The former silent per-element fallback
     to analytic screening is gone. Tests use synthetic SRD 64-format tables
     (`tests/data/mott_srd64_synthetic/`). The finding that led here:
     - The five `DisplayCalcTCSTableFor<El>.csv` files are transport
       cross-section tables exported from NIST SRD 64 (Electron
       Elastic-Scattering Cross-Section Database). The code docstring
       (`_load_mott_transport`) says so, and so do the file names and headers.
     - NIST states that SRD "are copyrighted by the U.S. Secretary of Commerce
       … None of our SRD may be reproduced, stored in a retrieval system or
       transmitted, in any form or by any means … without prior permission"
       ([NIST SRD public law](https://www.nist.gov/srd/public-law)). NIST
       lists SRD as the exception to its public-domain data policy
       ([NIST licence](https://www.nist.gov/open/license)).
     - The SRD 64 site carries "©2023 … All rights reserved. Copyright for
       NIST Standard Reference Data is governed by the Standard Reference Data
       Act" ([SRD 64](https://srdata.nist.gov/srd64/)).
     - The SRD 64 v3.2 Users' Guide says "No part of this database may be
       reproduced … without the prior written permission of the distributor".
     - Recommendation: stop redistributing them. Either retire them together
       with the opt-in `elastic_model="mott"` (ELSEPA is the default), or
       replace the download with a user-side fetch. Keep them only if NIST
       grants written permission (contact `data@nist.gov`).
     - The files stay in Git history in either case (see decision 7).
7. **Repository history.** Leave history as it is now.
   - When the repository goes public (#167), archive and start fresh:
     1. Push the full history to an archived `pyrite-history` repository.
     2. Give the existing repository an orphan-commit `main` holding the
        cleaned tree.
     3. Delete the old branches.
     4. Document the `git replace --graft` recipe that rejoins the archived
        history locally.
   - This runs only after #263 and #264 have removed the files. It uses
     neither `git filter-repo` nor Git LFS.
   - Costs:
     - every clone must be re-cloned;
     - open branches must be rebased onto the orphan `main`;
     - old SHAs resolve only in the archive. That covers issues, PRs,
       `agentdocs/`, validation check-records, and remote `code_revision`
       metadata.

The legacy `~/.local/share/pyrite/xsgen/tables` tier, and the legacy
`reference-data` sdbase path, stay read-only until #261 ships its one-time
migration. After that, a resolution from the legacy tier warns for one
release, and then the tier is removed.

*Implemented for the table tier (#263):* #261's sync migrates the box's legacy
tier on every inventory. Locally, a table served from the legacy tier, which
exists only when an explicit workspace is selected, emits one `FutureWarning`
per process naming `pyrite tables migrate`. That command copies every legacy
table the workspace lacks, payload first and atomically, and never deletes or
overwrites anything. `pyrite tables list` labels those rows `legacy`. The
tier stops being searched in 0.5.0 (`LEGACY_TABLE_TIER_REMOVE_IN`, held by
`tests/test_deprecation_schedule.py`). The legacy sdbase path is unchanged.

## Consequences

Measured wheel sizes are in
[Data distribution and repository size](../repo-design/data-distribution-and-repository-size.md#wheel-baseline).
Stages not yet built are projections from the per-entry compressed sizes.

```{list-table} Wheel size by stage.
:name: tbl-adr0014-wheel-projection
:header-rows: 1

* - Stage
  - Wheel, compressed (MiB)
  - Installed (MiB)
* - Before #263
  - 17.65 (measured)
  - 50.6
* - After #263 (decisions 4 and 6)
  - 16.06 (measured)
  - 45.9
* - Plus decision 2 (EEDL and EADL) and the Mott removal, #263
  - 7.36 (measured)
  - 13.1
* - Plus #264 (BELLS)
  - about 4.2
  - about 7.5
* - Plus decision 3 (SBETHE tables)
  - about 1.6
  - about 4.7
```

- A clean install needs one documented step, `pyrite tables fetch` for all
  datasets (or `--archive DIR` offline), before a default run. Today that
  step is already needed for ELSEPA. Missing data fails with a named fix,
  following Geant4 and MCNP practice.
- Offline generation depends on a checkout, an sdist, or a synced box, not
  on the wheel.
- A box that was synced before this change keeps a stale
  `src/pyrite/data/xsgen/elsepa` copy, because code sync overlays files and
  does not delete them. Nothing resolves that path any more.
- Full clones do not shrink until the fresh start in decision 7. The growth
  caused by table regeneration stops once decision 3 lands.
