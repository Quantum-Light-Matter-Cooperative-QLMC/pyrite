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
  - State after #263
* - `catalog/`, `cifs/`, `line_grid_defaults.toml`, `line_grid_provenance.toml`, `conduction_band.toml`, `eaglexo_qe.csv`
  - a
  - Unchanged. Authored here, under 0.1 MiB, read by every run.
* - `xsgen/elsepa-tables.json`, `xsgen/bremslib-tables.json` (and a future `sbethe-tables.json`)
  - a
  - Unchanged. The pin registry (the pooch-registry analogue) travels with the code.
* - `xsgen/tables/` (49 SBETHE tables)
  - b
  - Still packaged; they move once #167 hosts the archives (decision 3).
* - `characteristic_cross_sections/EEDL.endf`, `EADL2025.ALL`
  - b
  - Still packaged; they move after #261 (decision 2).
* - ELSEPA sources and `database/`; `sbethe.f`
  - c
  - Moved to `vendor/xsgen/elsepa/` and `vendor/xsgen/sbethe/`.
* - ELSEPA `test-run-output/dcs_1p000e03.dat`
  - test fixture
  - Moved to `tests/data/xsgen/elsepa/test-run-output/`.
* - `BELLS_gpt.out.gdf`
  - none (removed)
  - Removal and the GPT guide rework are #264.
* - `atomic_scattering_factors/` (CXRO/Henke f1/f2)
  - none (removed)
  - Deleted; nothing read it.
* - `mott_transport_cross_sections/`
  - a, pending licence
  - Unchanged; blocked on the NIST finding in decision 6.
```

### Accepted decisions

1. **Classes and threshold.** The three-class rule and the ~1 MiB compressed
   in-wheel threshold are adopted.
2. **EEDL and EADL** become a class (b) dataset. They will be fetched from the
   LLNL URLs pinned in the characteristic-cross-sections README and cached in
   the workspace data root.
   - *Follow-up, not yet implemented:* the move waits until #261 lands, so the
     box receives the files once through content-addressed sync rather than
     in every 19.9 MiB code sync.
   - The follow-up must repoint `montecarlo/eedl_ionization.py` and the
     bremsstrahlung loader, keep the SHA-256 pins in the model identity
     markers, and extend the remote preflight.
3. **SBETHE tables** become class (b), with a `sbethe-tables.json` release
   index like ELSEPA's and BremsLib's. *Follow-up:* they move once #167 hosts
   the release archives. Until then they stay packaged, because moving them
   first would break default runs on a fresh install.
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
     No action has been taken; a human decision is required.
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
* - Plus #264 (BELLS)
  - about 13.0
  - about 40.3
* - Plus decision 3 (SBETHE tables)
  - about 10.3
  - about 37.5
* - Plus decision 2 (EEDL and EADL)
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
