# Remove inline detector geometry flags from profile create/set (issue #62)

Branch: `62-remove-inline-detector-geometry-flags-from-profile`

## Context

Follow-up to #54, which removed the nine inline beam phase-space flags from
`pyrite profile create`/`set`. This slice removes the parallel three inline
detector geometry flags (`--observation-angle`, `--polar-acceptance`,
`--solid-angle`): named detectors (`pyrite detector create/set`, attached with
`pyrite profile set NAME --detector NAME`) already cover the surface, and the
inline path duplicated it while letting a profile's detector geometry drift
out from under the `[detectors.NAME]` object it references.

Not a mechanical copy of #54: the detector path also drove
`campaign/profile_edit.apply_detector_updates`, which detached a named
reference on inline edit (copying the named object's resolved geometry
first), and `ACTIVE_DETECTOR_FIELDS`-derived `detector_labels` wiring into
`set_profile`'s overwrite-confirmation text. Both are retired outright; there
is no replacement for the detach-and-copy convenience (migrate via
`pyrite detector create` from the existing values, then `--detector`).

Timing: `DEPRECATED_FLAGS` rows say `remove_in 0.3.0`; nothing is tagged or
released (`pyproject.toml` at 0.2.0), so the window is nominal. Followed #54's
precedent for the twin beam family: removed outright (plain "no such option"
usage errors) rather than deprecated, since named detectors already fully
cover the surface.

## Scope

- `cli/commands/profile.py`: dropped `_detector_cli_options`,
  `_DETECTOR_FLAG_PARAMS`, `_warn_inline_detector_flags`, and the detector
  scalar params on `create`/`set`. `--beam`/`--detector` help text and
  docstrings reworded to match #54's "attach a named object" phrasing. Dropped
  `@click.pass_context`/`ctx` from both commands (their only remaining use).
- `cli/_deprecations.py`: dropped `_DETECTOR_FLAG_NAMES`/`_DETECTOR_FLAG_NOTE`,
  their `DEPRECATED_FLAGS` rows, and `SELF_WARNING_FLAGS` (now empty — no
  producer left after beam's own rows were dropped in #54).
- `campaign/profile_edit.py`: retired `apply_detector_updates` and the
  `detector_updates` kwarg on `create_profile`/`set_profile`. `set_profile`
  now returns just `overwriting` (the `detector_labels` half was always
  driven by `detector_updates`, which no longer exists).
- `cli/commands/_detector_shared.py`: untouched — `pyrite detector` still
  owns it exclusively.
- Tests: ported detector domain-validation coverage
  (`--polar-acceptance`/`--solid-angle` range checks) from
  `tests/cli/test_profile.py` to `tests/cli/test_detector.py` (it was
  previously exercised only through the profile path even though the
  validation lives in the shared `_detector_shared.py` options). Added
  `test_inline_detector_flags_are_gone_from_create_and_set`, mirroring #54's
  beam version. Trimmed the conflict/detach-compatibility assertions out of
  the unknown-detector test since that behavior no longer exists.
- `tests/data/cli_contract.json`: regenerated via
  `scripts/freeze_cli_contract.py --write`; also added the missing
  `profile-inline-beam-flags-removed` row to that script's
  `INTENTIONAL_P0_CORRECTIONS` (it existed only hand-edited into the frozen
  JSON since #54, which `--write` would otherwise have silently dropped) plus
  the new `profile-inline-detector-flags-removed` row.
- Docs: `docs/guides/sweep-profiles.md` and `docs/repo_map.md` updated to
  match the beam-removal wording from #54;
  `docs/repo-design/cli/cli-reference.md` and `cli-deprecations.md`
  regenerated.

## Status

Implemented; verification in progress.
