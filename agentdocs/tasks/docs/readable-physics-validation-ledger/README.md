# Readable physics validation ledger

## Problem and scope

The canonical physics validation ledger renders as very wide Markdown tables in
the Sphinx HTML site. PyData Sphinx Theme compresses the eight columns into the
content width, leaving only a few characters per wrapped line and making rows
extremely tall. The result is not practically readable.

Reformat the detailed ledger as one labeled record per validation ID. Preserve
all IDs, domain grouping, claims, code anchors, sources, statuses, checks,
anchors, notes, links, equations, and validation meaning. Update the generated
view parser and focused tests for the new canonical representation. This task
does not change physics, validation verdicts, sign-off state, or methodology.

## Implementation path and likely owners

- `docs/validation/physics-validation-ledger.md`: replace each eight-cell row
  with a stable per-ID heading and labeled fields suitable for narrow HTML.
- `src/pyrite/devtools/validation_ledger.py`: parse the record representation
  while retaining duplicate-ID, malformed-record, and empty-ledger failures.
- `tests/dev/test_validation_ledger.py`: pin parsing and generated-view
  behavior for the new representation.
- Regenerate `docs/validation/status-summary.md` and
  `docs/validation/domain-inventories.md`; make their links target the matching
  record anchors if MyST's generated anchors are stable.
- Avoid global theme CSS unless a small local presentation adjustment remains
  necessary after the structural rewrite.

## Checklist

- [ ] Record the current parsed inventory (IDs, domains, claims, statuses) as a
      preservation baseline before converting the ledger.
- [ ] Define a concise plain-Markdown record schema with a stable ID heading
      and the seven labeled detail fields.
- [ ] Convert every ledger row mechanically without changing field content or
      validation state.
- [ ] Update the parser and focused tests, including malformed and duplicate
      record coverage.
- [ ] Regenerate compact validation views and deep-link inventory IDs to their
      detailed records.
- [ ] Build the Sphinx site and inspect representative short and long records
      at desktop and narrow content widths.
- [ ] Review the scoped diff for accidental scientific or status changes.

## Decisions and open questions

- Chosen over CSS-only widening/horizontal scrolling: labeled records are
  readable without traversing eight columns and work at narrow widths.
- Chosen over tables split by domain: several `checks` and `notes` fields are
  intrinsically long, so narrower tables retain the same wrapping failure.
- The detailed Markdown ledger remains authoritative and directly editable;
  generated summaries remain derived artifacts.
- Preserve the methodology's plain-Markdown, GitHub-renderable,
  git-diffable design. Do not introduce JavaScript or a custom Sphinx-only
  renderer.
- No material decision remains open. Minor field-label syntax is an
  implementation detail, provided it renders clearly in GitHub and Sphinx.

## Delegation

One self-contained implementation slice: ledger conversion, parser/tests,
generated views, and rendered-doc verification. Required skills:
`implement-task` and `documentation-maintenance`. Serena one-shot is suitable.
No physics skill is required because content and verdict changes are explicit
non-goals; stop and request physics review if conversion exposes ambiguous row
boundaries or requires interpreting scientific content.

## Acceptance checks

- The before/after parsed inventories match exactly for validation ID, domain,
  claim, and status; total entry count is unchanged.
- Every former table field remains present in its corresponding record, with
  equations, links, code anchors, and status unchanged.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/dev/test_validation_ledger.py`
  passes.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev validation-ledger --check`
  passes after regeneration.
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs` passes cleanly.
- Representative long records are readable in built HTML without
  character-by-character wrapping or requiring horizontal table traversal.

