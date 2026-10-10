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
`LINE_ESCAPE_MODEL`, `COHERENT_POPULATION_MODEL`) or `tables-*-N` table tag
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
3. Bumps `pyproject.toml` and `src/pyrite/__init__.py`, then refreshes `uv.lock` (CI syncs with `--locked`).
4. Regenerates `docs/repo-design/cli/cli-reference.md` and
   `docs/repo-design/cli/cli-deprecations.md`.
5. Prints release notes from conventional commits since the last `v*` tag
   (without a tag, since the last `chore(release): bump version to` commit;
   override with `--base REV`). Breaking (`!` or `BREAKING CHANGE`) and
   physics-changing entries are listed in their own sections; physics-changing
   is detected from the physics scopes in `pyrite.devtools.release` or a
   `Physics-Changing` body line, so review that list by hand.

`--check` validates and prints notes without editing files. Put the notes in
the PR description, then run `pyrite-dev verify`. After the PR merges,
tag `vX.Y.Z` on the merge commit. Tagging and pushing need explicit human
authorization. PyPI publishing is out of scope.

## Majors

There is no major policy at 0.x. 1.0 is ready when:

- there is a public API surface the project commits to;
- the deprecation process has been exercised across at least two removal cycles;
- result and checkpoint schemas are versioned, and results carry SHA provenance.

After 1.0, a major is any breaking change to the public API.
