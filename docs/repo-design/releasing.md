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
   `uv.lock` (CI syncs with `--locked`).
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

Releases for data and table archives (`tables-*`, `mirror-*`) share the
repository's Releases page. Create them with `--latest=false` (see
[data distribution](data-distribution-and-repository-size.md)) so the Latest
badge always points at the newest code release.

`git clone` checks out `main`, not the latest release. To use a released
version, check out its tag (`git checkout vX.Y.Z`) or pin it when installing
(`pip install "git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite@vX.Y.Z"`).

## Majors

There is no major policy at 0.x. 1.0 is ready when:

- there is a public API surface the project commits to;
- the deprecation process has been exercised across at least two removal cycles;
- result and checkpoint schemas are versioned, and results carry SHA provenance.

After 1.0, a major is any breaking change to the public API.
