# Releasing and versioning

PyRITE versions are bumped only in a dedicated release PR, never per feature PR.
Per-PR bumps are rejected: they conflict across parallel worktrees. Between
releases, result provenance still identifies the exact code through `git_sha` /
`git_dirty` next to `versions.pyrite`.

## 0.x semantics

Before 1.0 the minor number carries the breaking-change signal.

- **Minor (`0.X.0`)**: a breaking API, CLI, schema, or checkpoint change; a
  deprecation removal; or any change to default-run numerical output.
- **Patch (`0.x.Y`)**: fixes, docs, performance, or internal work whose default
  output is unchanged.

The deprecation window is `SUPPORT_WINDOW_MINORS = 2` minor releases
(`pyrite.cli._deprecations`). It counts minors, so a release is what advances
scheduled removals.

## Physics identity

Physics identity is carried by the model markers, not by the release version.
A PR that changes default numerical output bumps the relevant `*_MODEL` marker
(for example `STOPPING_MODEL`, `CHARACTERISTIC_MODEL`, `BREMSSTRAHLUNG_MODEL`,
`CXR_ESCAPE_MODEL`, `COHERENT_POPULATION_MODEL`) or `tables-*-N` table tag
**in the same PR**. The release version only bundles markers; it never replaces
them.

## When to release

- Before kept campaign, paper, or collaborator-handoff runs.
- When deprecation removals are due.
- When a breaking change has landed and anything external installs the package
  (lab box, collaborators).
- Otherwise roughly every 2-4 weeks, if anything changed.

## The release PR

Title and commit: `chore(release): bump version to X.Y.Z`. Produce it with:

```bash
pyrite-dev release X.Y.Z --notes-file release-notes.md
```

The command:

1. Rejects a target that is not a strictly greater `X.Y.Z`.
2. Fails, before editing anything, while any scheduled deprecation removal
   (module shims, command/option spellings, implicit defaults, `Sweep.from_legacy`,
   the legacy xsgen table tier) targets a minor at or before `X.Y`. Do those
   removals first, with call sites, tests, and docs.
3. Bumps `pyproject.toml`, `src/pyrite/__init__.py` and the tutorial
   `pyrite-mc[...]==X.Y.Z` script pins in `docs/notebooks/*.py`, then refreshes
   `pixi.lock` (CI syncs with `--locked`).
4. Regenerates `docs/repo-design/cli/cli-reference.md` and
   `docs/repo-design/cli/cli-deprecations.md`.
5. Prints release notes from conventional commits since the last `v*` tag
   (without a tag, since the last `chore(release): bump version to` commit;
   override with `--base REV`). Breaking (`!` or `BREAKING CHANGE`) and
   physics-changing entries are listed in their own sections; physics-changing
   is detected from the physics scopes in `pyrite.devtools.release` or a
   `Physics-Changing` body line, so review that list by hand.

A release does not regenerate tutorial captures. Their recorded
`pyrite_version` is provenance only, and capture fingerprints ignore the
PyRITE pin version, so the bumped tree passes `verify` as is. Tutorials whose
numerics change are regenerated in the PR that changes them; see
[the documentation workflow](documentation.md).

`--check` validates and prints notes without editing files. Put the notes in
the PR description, then run `pyrite-dev verify`. After the PR merges,
tag `vX.Y.Z` on the merge commit and push the tag. Tagging and pushing need
explicit human authorization. Pushing the tag also authorizes the workflow's
automatic PyPI publication.

## Publishing

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes the
GitHub Release for it, then publishes the package to PyPI:

1. It refuses a tag that is not `v{__version__}` for the tagged commit, so a
   tag on an unbumped commit publishes nothing.
2. It renders notes with `python -m pyrite.devtools.release vX.Y.Z` (also
   `pyrite-dev release-notes vX.Y.Z`): the same grouping as above, from the
   previous `v*` tag to the new one.
3. It creates the Release titled `PyRITE X.Y.Z` and marks it **Latest**.
4. Only after the release job succeeds, it builds the tagged commit's wheel
   and source distribution, checks them with Twine, and uploads them as a
   workflow artifact.
5. Only after the build job succeeds, it downloads those distributions and
   publishes them to PyPI using Trusted Publishing in the `pypi` environment.

Configure the `pyrite-mc` project's PyPI Trusted Publisher for this repository,
workflow filename `release.yml`, and environment `pypi`. When migrating from
the separate publish workflow, replace the `publish-pypi.yml` publisher with
`release.yml` before pushing the next release tag. Any required reviewers on
the GitHub `pypi` environment still gate publication; omit that requirement
if publication should proceed without manual approval.

### Zenodo authorship

The repository's Zenodo GitHub integration currently archives code, table and
mirror releases into the same concept DOI, `10.5281/zenodo.23487182`. That DOI
can therefore resolve to a table release. For publications, select the exact
software version and cite its version DOI; see
[Citing PyRITE](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#citing-pyrite).

Future archive creators come from the committed `.zenodo.json`, never from
GitHub's contributor list: agent commits carry `Co-Authored-By: Claude …`
trailers, which GitHub counts as contributors, and Zenodo would otherwise
credit them as authors. Only humans are listed. Add a new human author to
`.zenodo.json` and `CITATION.cff` in a normal PR, preserving author order and
affiliations; never add an AI agent or `[bot]` account. `CITATION.cff` enables
GitHub's citation interface. When both files exist,
[Zenodo uses only `.zenodo.json`](https://help.zenodo.org/docs/github/describe-software/).
The licence is `other-nc`, Zenodo's nearest vocabulary entry for the UCLA
Academic Software License in `LICENSE.txt`, which the archived source carries.

The committed `description` supplies a general software abstract instead of
the GitHub release changelog. Maintain it alongside README capabilities;
release-specific changes stay in GitHub release notes. The title, English
language, keywords and related identifiers support discovery. Documentation is
linked with `isDocumentedBy`; the runtime tables concept DOI
`10.5281/zenodo.23220627` is linked with `requires` as a dataset. Keep these
links stable, and let ingestion derive each release's version, date and DOI
rather than freezing them in `.zenodo.json`.

For multiple affiliations, use `affiliations: [{"name": "…"}, …]` on a creator,
with one full institution/department/address per entry. Sergio has four entries,
matching the four semicolon-separated texts in `CITATION.cff`. Do not also set
the singular `affiliation`: current ingestion converts that value to one entry
and overwrites the plural list. A singular string remains valid for one
affiliation; an array under the singular key is invalid, and semicolons inside
a string do not create separate affiliations. This plural form is supported by
the current [GitHub metadata loader](https://github.com/zenodo/zenodo-rdm/blob/c0ad6c384218530072e39c0e0901c9fd37ce6659/site/zenodo_rdm/github/release.py#L22-L41)
and [creator deserializer](https://github.com/zenodo/zenodo-rdm/blob/c0ad6c384218530072e39c0e0901c9fd37ce6659/site/zenodo_rdm/legacy/deserializers/metadata.py#L80-L109),
although the older documented legacy schema shows only a singular string.
The upstream creator deserializer was checked locally against all five creators;
confirm the published result after ingestion as well.

After each software release:

1. Open the exact version's Zenodo record. Check the abstract, keywords,
   documentation link, runtime-tables **Requires** link, version and creator
   order/ORCIDs. Confirm Sergio occurs once with four separate affiliations.
2. If a correction is needed, choose **Edit**. Update the description from
   `.zenodo.json` and edit Sergio's creator entry. The
   [creator form](https://help.zenodo.org/docs/deposit/describe-records/creators/)
   accepts multiple affiliations: add each of the four full texts separately,
   using text entries where institution suggestions lose department/address
   details. Remove any combined affiliation, save the creator, then **Publish**.
   This [metadata correction preserves the DOI](https://help.zenodo.org/docs/deposit/manage-records/).
3. Check the record's citation/metadata exports and a fresh Zotero import for
   the general abstract. Inspect structured affiliations on the record even
   if the chosen citation format omits them.

Repository metadata changes affect future archives; they do not update already
published records. Apply the same correction procedure to affected versions,
including the reported v0.36.1 abstract. Existing Zotero items may retain the
old abstract: refresh their metadata or compare a fresh import before replacing
any locally edited fields.

(zenodo-archive-separation)=

#### Separating software and dataset archives

`--latest=false` does **not** exclude a release from Zenodo or change its
version family: the account's GitHub integration archives every GitHub
release of this repository into the software family. Software and tables are
therefore separated by where they are published, not by integration
settings:

- **Software** stays on automatic ingestion: each `vX.Y.Z` release becomes a
  version of the software family, with creators from `.zenodo.json`.
- **Tables** publish only as versions of the separate Zenodo dataset family,
  concept DOI `10.5281/zenodo.23220627`, deposited by a maintainer (see
  {ref}`the table record <zenodo-table-record>`). No new `tables-*` or
  `mirror-*` GitHub release is created, so no table can advance software
  latest. The dataset record credits upstream deposits as `isDerivedFrom`
  relations and keeps their licences and modification statements in the
  bundled notices.

Earlier `tables-*` and `mirror-*` releases (through `tables-bremslib-4`)
were ingested into the software family, so concept DOI
`10.5281/zenodo.23487182` can resolve to a table version. Those records are
left as published: do not delete them or relabel them as software, and keep
their download URLs, which older indexes pin. Moving them out of the family
would need Zenodo support; a metadata edit does not change grouping. Until
the next code release becomes the family's latest version, cite exact
software-version DOIs.

Existing data and table releases (`tables-*`, `mirror-*`) share the
repository's Releases page and were created with `--latest=false`, so the
Latest badge points at the newest code release; if one shows Latest, mark the
newest `vX.Y.Z` release Latest (`gh release edit vX.Y.Z --latest`).

`git clone` checks out `main`, not the latest release. To use a released
version, check out its tag (`git checkout vX.Y.Z`) or pin it when installing
(`pip install "git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite@vX.Y.Z"`).

## Majors

There is no major policy at 0.x. 1.0 is ready when:

- there is a public API surface the project commits to;
- the deprecation process has been exercised across at least two removal cycles;
- result and checkpoint schemas are versioned, and results carry SHA provenance.

After 1.0, a major is any breaking change to the public API.
