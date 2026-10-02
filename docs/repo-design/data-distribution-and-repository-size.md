# Data distribution and repository size

This page holds the measurements and comparisons behind
[ADR-0014](../adr/0014-packaged-data-layout.md). It records the state at
`main` 135105de (2026-09-30). Numbers go stale as the tree changes; the method
is kept so they can be re-measured.

## Wheel baseline

`uv build --wheel` at 135105de produces `pyrite_xray-0.4.0-py3-none-any.whl`:
**18.5 MB (17.65 MiB)** on disk, 922 files, **50.6 MiB** installed.
After #263 moved the generator sources to `vendor/` and deleted
`atomic_scattering_factors/`, the wheel is 16.8 MB (16.06 MiB), 693 files,
45.9 MiB installed. After EEDL and EADL became fetched datasets and the NIST
Mott tables were removed (also #263), it is 7.7 MB (7.36 MiB), 688 files,
13.1 MiB installed.

Issue #264 subsequently removed `BELLS_gpt.out.gdf` from the tree and wheel.
With #264 and the #274 EPDL table on `main`, the wheel measured on
2026-10-01 is 5.97 MB (5.69 MiB), 690 files: 2.64 MiB SBETHE tables, 1.49 MiB
EPDL table, 1.41 MiB code. That is a 68% cut from the 17.65 MiB baseline.
The baseline tables below retain its measurements to document the decision;
the blob remains in Git history.

Issue #282 then moved the SBETHE tables and the EPDL table to hosted releases:
the wheel measured on 2026-10-02 is 1.65 MB (1.57 MiB), 4.78 MiB installed,
down from 5.73 MiB (9.11 MiB installed) on the same branch before the removal.
That is a 91% cut from the 17.65 MiB baseline.

```{list-table} Wheel contents by data entry (MiB).
:name: tbl-data-wheel-baseline
:header-rows: 1

* - Entry
  - Files
  - Installed
  - Compressed in wheel
* - `data/characteristic_cross_sections/EEDL.endf`
  - 1
  - 24.70
  - 7.35
* - `data/BELLS_gpt.out.gdf`
  - 1
  - 5.64
  - 3.16
* - `data/xsgen/tables/` (49 SBETHE tables)
  - 98
  - 2.82
  - 2.64
* - `data/characteristic_cross_sections/EADL2025.ALL`
  - 1
  - 8.03
  - 1.34
* - `data/xsgen/elsepa/database/`
  - 206
  - 3.90
  - 1.31
* - `data/xsgen/elsepa/` Fortran sources and `elscata.in`
  - 4
  - 0.39
  - 0.08
* - `data/xsgen/elsepa/test-run-output/`
  - 1
  - 0.04
  - 0.01
* - `data/xsgen/sbethe/sbethe.f`
  - 1
  - 0.14
  - 0.03
* - `data/atomic_scattering_factors/`
  - 17
  - 0.24
  - 0.11
* - `data/mott_transport_cross_sections/`
  - 5
  - 0.05
  - 0.02
* - catalog, CIFs, line grids, QE curve, conduction band, release indexes
  - 169
  - 0.11
  - 0.05
* - Python code, apps, and metadata
  - 418
  - 4.55
  - 1.40
```

Issue #274 (2026-10-01) adds `data/photon_cross_sections/epdl2025_mf23.npz`,
1.49 MiB installed and about 1.4 MiB compressed in the wheel (already
deflated). It is derived from the 86 MB upstream `EPDL2025.ALL` by
lin-lin knot thinning to `5e-4` relative; at `1e-3` it would be 1.30 MiB, and a
log-log thinning reached 0.73 MiB but departed from the upstream lin-lin law
by up to 67% across single steep intervals, so it was rejected. It is classed
(b) beside EEDL and EADL in ADR-0014.

## Hosted table releases

PyRITE's own pinned table archives are published as GitHub Release assets of
this repository, one release per dataset (issues #284, #282). While the
repository is private, downloading one needs a GitHub token with read access;
the move to public Zenodo records (#167) swaps the URL and drops the token.

```{list-table}
:header-rows: 1

* - Release
  - Asset
  - Pinned by
* - `tables-elsepa-1`
  - `elsepa-tables.zip` (25,178,906 B, 28 tables)
  - `src/pyrite/data/xsgen/elsepa-tables.json`
* - `tables-bremslib-1`
  - `bremslib-tables.zip` (23,066,359 B, 24 tables)
  - `src/pyrite/data/xsgen/bremslib-tables.json`
* - `tables-sbethe-1`
  - `sbethe-tables.zip` (49 catalogue stopping tables)
  - `src/pyrite/data/xsgen/sbethe-tables.json`
* - `tables-epdl-1`
  - `epdl2025_mf23.npz` (derived EPDL2025 photon table)
  - `pyrite.datasets.EPDL`
* - `mirror-eedl2025-1`
  - `EEDL.endf` (fallback after LLNL)
  - `pyrite.datasets.EEDL`
* - `mirror-eadl2025-1`
  - `EADL2025.ALL` (fallback after LLNL)
  - `pyrite.datasets.EADL`
```

Each index and dataset lists its download locations in order (`archive.urls`
in the table indexes, `Dataset.urls` in `pyrite.datasets`): upstream first
where one exists, then PyRITE's release, later a Zenodo record. A location
that fails or serves other bytes is skipped; the pinned SHA-256 decides what
installs, so adding a mirror cannot change it. The SBETHE `sdbase/` database
stays upstream-only (its redistribution terms are unconfirmed), and NIST SRD 64
Mott tables are never hosted. `pyrite tables fetch` without a code installs
everything.

`pyrite tables fetch` resolves a `github.com/.../releases/download/<tag>/<asset>`
URL through the GitHub API with the first token found in `PYRITE_GITHUB_TOKEN`,
`GITHUB_TOKEN`, or `gh auth token`. The token goes only to `api.github.com`;
it is not forwarded on the redirect to the asset store and never appears in
output or manifests. Without a token the plain URL is tried, which fails with a
message naming the token variables and `--archive PATH`. CI passes the job's
`GITHUB_TOKEN`. The remote box has no token; `pyrite remote sync` ships the
installed tables instead.

To cut or refresh a release (maintainer):

1. Build the archive and index from the stored tables, for example
   `uv run python scripts/release_elsepa_tables.py --out build/xsgen-release`
   (likewise `release_bremslib_tables.py`, `release_sbethe_tables.py`;
   `release_epdl_table.py` writes the EPDL npz)
   (add `--generate` for missing tables). The archive is deterministic: an
   unchanged table set reproduces the pinned SHA-256.
2. If the digest differs from the pin, use a new tag (`tables-elsepa-<n+1>`);
   never replace an asset under an existing tag, since older indexes pin it.
3. Publish with
   `gh release create <tag> build/xsgen-release/elsepa-tables.zip --title ... --notes-file NOTES.md`.
   The notes carry the upstream attribution, licence, and a modifications
   statement (see `THIRD-PARTY-NOTICES.md`).
4. Download the asset back (`gh release download <tag> -p <asset> -O - | sha256sum`)
   and check it matches the index.
5. Rebuild with `--url <release download URL> --pin` (`--url` repeats, in
   fetch order; or edit `archive.urls`) and commit the index; CI's cache key
   follows it. A dataset's URLs and SHA-256 live in `pyrite.datasets`.

## Repository size

```{list-table} Clone and pack sizes.
:name: tbl-data-clone-sizes
:header-rows: 1

* - Measurement
  - Size (MiB)
* - Local object store (`git count-objects -vH`): 17 packs plus 5979 loose objects, not repacked
  - 53.3 packed + 35.4 loose
* - Full `git clone --bare` from GitHub (4 branches, 2036 commits)
  - 41.4
* - The same objects repacked locally (`git repack -adf --window=250 --depth=50`)
  - 29.9
* - `git clone --depth=1`
  - 22.7
* - `git clone --filter=blob:none`, no checkout (all history, no blobs)
  - 2.3
* - `git clone --filter=blob:none` with `main` checked out
  - 24.9
```

GitHub serves about 11.5 MiB more than a fresh repack of the same objects
needs. Most of that is `EEDL.endf`. Commits 791d75cb and 238f79a7 stored two
revisions that differ by 2 bytes. GitHub's pack stores both as full 7.35 MiB
objects instead of storing one as a delta of the other. After a local repack,
the second revision costs 687 bytes and the older LF-normalized revision costs
0.98 MiB. Server-side repacking is up to GitHub; this repository cannot
trigger it (unverified whether a support request would).

### Per-path history cost

Method: run `git rev-list --objects --all` on the bare clone, pipe it into
`git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize)
%(objectsize:disk) %(rest)'`, and sum by path prefix. Each blob counts once,
under the first path that `rev-list` reports for it. "History-only" means
blobs that are not in the `main` tree.

```{list-table} History cost of the larger paths in GitHub's served pack (MiB).
:name: tbl-data-history-cost
:header-rows: 1

* - Path
  - Blobs
  - Raw
  - Packed
  - History-only packed
* - `src/pyrite/data/characteristic_cross_sections/EEDL.endf`
  - 3
  - 73.77
  - 15.62
  - 8.28
* - `src/pyrite/data/BELLS_gpt.out.gdf`
  - 1
  - 5.64
  - 3.16
  - 0
* - `src/pyrite/data/xsgen/tables/`
  - 98
  - 2.82
  - 2.60
  - 0
* - `checks/full_track_bremslib/`
  - 75
  - 2.34
  - 2.21
  - 0.01
* - `src/pyrite/data/characteristic_cross_sections/EADL2025.ALL`
  - 1
  - 8.03
  - 1.34
  - 0
* - `src/pyrite/data/xsgen/elsepa/` (all)
  - 211
  - 4.33
  - 1.40
  - 0
* - `uv.lock`
  - 60
  - 33.63
  - 1.14
  - 0.93
* - Other `src/`, `tests/`, `docs/`, `agentdocs/`, `checks/`, and so on
  - about 11 000
  - about 230
  - about 11
  - about 9
* - Trees, commits, and tags
  - n/a
  - n/a
  - 1.90
  - n/a
```

At the recorded baseline, no large blob except the duplicate EEDL revisions
existed only in history. Removing `BELLS_gpt.out.gdf` from the current tree
makes shallow and blobless clones smaller at once. Full clones keep its blob
unless history is rewritten.

### Options and projected savings

For each rewrite scenario, the listed paths were removed from the object list
and the remaining objects were packed with fresh deltas
(`git pack-objects --no-reuse-delta`, window 10, which is Git's default).
This approximates the pack GitHub builds after a rewrite. The filtering ran on
a scratch copy; no history was rewritten.

```{list-table} Projected full-clone size by option (MiB; the current GitHub clone is 41.4).
:name: tbl-data-rewrite-savings
:header-rows: 1

* - Option
  - Full clone
  - Saving
* - A. Leave history (fresh-delta equivalent of today)
  - 41.4 served (30.5 if repacked)
  - 0, or about 11 if GitHub repacks
* - B1. Rewrite: `BELLS_gpt.out.gdf`
  - 27.3
  - 14.1
* - B2. Also `xsgen/elsepa/`
  - 25.7
  - 15.7
* - B3. Also `EEDL.endf`, `EADL2025.ALL`
  - 16.1
  - 25.3
* - B4. Also `xsgen/tables/`
  - 13.7
  - 27.7
* - B5. Also `checks/full_track_bremslib/`
  - 11.7
  - 29.7
```

Measured against a clean repack instead of GitHub's current pack, B3 saves
14.4 MiB. The rest of B3's headline saving comes from the repack that any
rewrite forces.

If the recommended entries leave the tree without a rewrite, the checkout
drops by about 16 MiB compressed; #264 accounts for 3.16 MiB of that total.
That cuts a `--depth=1` clone from 22.7 to
about 7 MiB and a blobless clone from 24.9 to about 9 MiB. Full clones grow
only by future commits.

The `pyrite remote sync` payload also shrinks. Each sync tars `src/`,
`checks/`, and the root metadata into a 19.9 MiB `.tgz`. Excluding EEDL, EADL,
the removed BELLS file, the ELSEPA tree, and the SBETHE tables gives 4.0 MiB.
That holds only
if the box receives the fetched datasets once through content-addressed sync
(#261), not through every code sync.

### Costs of each option

**A. Leave history.** No SHA changes. New clones that do not need history use
`git clone --filter=blob:none` (full history, blobs fetched on demand) or
`--depth=1`. Old blobs stay in full clones.

**B. `git filter-repo --invert-paths --path … --sensitive-data-removal`.**
Every commit SHA from the first touched commit onward changes. Costs:

- All four branches must be force-pushed. The three task worktrees
  (`issue-237`, `issue-261`, `issue-263`) and every clone must be re-cloned,
  or reset and rebased with care.
- 20 committed files under `docs/validation/check-records/` hold 30
  full-length commit SHAs as `code_revision` or similar provenance. Remote job
  `meta` files and `.pyrite-sync` stamps record `code_revision`
  (`remote/transport.py`, `validation/check_records.py`). After a rewrite
  these point at commits that no longer exist in the canonical history.
- SHAs cited in issues and pull requests (260 so far) and in `agentdocs/` stop
  resolving on `main`.
- GitHub keeps 94 read-only `refs/pull/*` refs. Old objects stay on the server,
  and closed-PR diffs break, until GitHub Support dereferences them and runs
  garbage collection. GitHub documents this in
  [Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
  Clones do not fetch `refs/pull/*` by default, so clone size still falls.
- The repository is private with no forks, so there are no third-party forks
  to chase.

**C. Git LFS for future large files.** History is unchanged. New large files
are stored as roughly 130-byte pointers, and the content lives in LFS storage
under a per-plan quota. The GitHub Free personal allowance is 10 GiB of
storage and 10 GiB of bandwidth; the organization's plan was not checked; see
[Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).
Costs: every clone, CI job, remote box, and `uv` git install needs `git-lfs`,
and bandwidth is metered. It is also redundant with ADR-0014's rule that large
upstream-derived data is a fetched dataset, not a tracked file. `git-lfs` is
not installed on the development machine.

**C′. `git lfs migrate import --everything`.** This rewrites history. It has
option B's costs, and the files still download on checkout.

**D. Archive and fresh start (chosen; ADR-0014 decision 7).** This happens at
go-public time (#167), after #263 and #264 have removed the files:

1. Push the full history to an archived `pyrite-history` repository.
2. Give the existing repository an orphan-commit `main` of the cleaned tree.
3. Delete the old branches.
4. Rejoin history locally with `git replace --graft`.

The new repository's clone is roughly the size of the cleaned tree: well under
the B-series figures, because it has no history. The costs follow B's list,
except that no rewrite tool runs: every clone is re-cloned, open branches are
rebased onto the orphan `main`, and old SHAs resolve only in the archive.

## How other codes distribute data

- **Geant4.** Datasets such as G4EMLOW and G4LEDATA are separate archives,
  versioned per toolkit release.
  [CMake](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/installguide.html)
  downloads missing ones with `GEANT4_INSTALL_DATA=ON` into
  `GEANT4_INSTALL_DATADIR`. `cmake/Modules/Geant4DatasetDefinitions.cmake`
  pins an MD5 sum for each dataset version (as reported in secondary sources;
  not read directly). At runtime, per-dataset environment variables such as
  `G4LEDATA` locate each dataset. A missing dataset raises a fatal
  `G4Exception`, per the
  [Geant4 HyperNews](https://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1511.html).
  Missing data does not break the build.
- **PENELOPE.** The NEA Data Bank distributes the database with the code
  ([PENELOPE-2018](https://www.oecd-nea.org/upload/docs/application/pdf/2020-10/penelope-2018__a_code_system_for_monte_carlo_simulation_of_electron_and_photon_transport.pdf)).
  The bundled `MATERIAL` program pre-generates the material data files that a
  simulation reads. The claim that a missing material file aborts the run is
  unverified.
- **EGSnrc.** Cross-section data files such as
  `photon_xsections_*.data` and `msnew.data` are tracked in the source
  repository under `$HEN_HOUSE/data`
  ([EGSnrc manual](https://nrc-cnrc.github.io/EGSnrc/doc/pirs701-egsnrc.pdf)).
  The input selects the cross-section set explicitly (for example `si`, `epdl`,
  or `xcom`). Media come from PEGS4 files or are defined in pegsless mode.
- **MCNP.** The `xsdir` directory file, found through the `DATAPATH`
  environment variable, maps each table identifier to a library file. MCNP
  stops with a fatal error when `xsdir` is not found
  ([MCNP6.2 manual](https://mcnp.lanl.gov/pdf_files/TechReport_2017_LANL_LA-UR-17-29981_WernerArmstrongEtAl.pdf)).
  The data libraries ship with the code distribution from RSICC (from general
  knowledge; not verified).
- **pooch.** A registry file maps each file name to a hash. Files download
  once into a cache and are verified against the registry. The `env=`
  argument lets an environment variable override the cache location
  ([user-defined cache](https://www.fatiando.org/pooch/latest/user-defined-cache.html)).
  PyRITE's `xsgen/*-tables.json` release indexes already follow this model.

All five treat a missing required dataset as an error rather than something to
degrade around. Geant4 and pooch separate large evaluated data from the code
and pin it by hash. Geant4's data is several gigabytes, much larger than its
code. PyRITE's largest upstream dataset (EEDL) compresses to 7.35 MiB, about
five times PyRITE's compressed code.
