# Improve documentation typesetting and quality gates

## Problem and scope

The Sphinx/MyST site builds successfully, but its maintained prose uses almost
none of Sphinx's semantic document model: the published Markdown currently has
no numbered figure, table, equation, or semantic-reference directives. Authors
therefore rely on visual order, prose references, and ad hoc reference sections
instead of stable labels and cross-references.

The current strict build also hides all `docutils` diagnostics through
`suppress_warnings = ["docutils"]`. An audit with that suppression disabled and
nitpicky mode enabled exposed substantial existing autodoc debt: malformed
plain-text physics expressions, indentation/definition-list diagnostics, and
unresolved Python-domain references. Removing the suppression without a staged
policy would make the documentation gate unusable; retaining it lets real
prose defects escape.

Improve rendered scientific-document quality and make regressions visible
without converting the public documentation corpus from MyST Markdown to rST.
This task covers Sphinx/MyST configuration, authoring rules, a representative
pilot page, developer commands/tests, and a bounded warning-debt policy. It does
not rewrite all documentation, alter scientific claims, or perform a general
public-docstring rewrite.

## Implementation path and likely owners

- `docs/conf.py`: semantic numbering/reference configuration, MyST extensions,
  warning policy, HTML/PDF-neutral settings.
- `docs/README.md`: durable authoring rules and examples for labels,
  cross-references, equations, figures/tables, accessibility, units, and
  citations.
- One maintained scientific or validation page selected for the pilot: apply
  the rules to real equations/tables/figures without changing its physics.
- `src/cxr_mc/_dev.py`: a cross-platform `cxr-dev docs` command that owns the
  canonical strict build invocation and an explicit opt-in external link check
  if it can be made deterministic enough.
- `tests/dev/`: command/config regressions; use built doctrees or compact HTML
  assertions where rendered numbering/reference behavior needs pinning.
- `pyproject.toml`: only dependencies justified by the chosen citation or lint
  design.

Primary skill: `documentation-maintenance`. Use `scientific-library` for any
public-docstring repair and the matching physics skill if a pilot edit touches
equations or claim wording. Use `cli-ui-ux` if the developer-command interface
grows beyond a single non-interactive check command.

## Checklist

1. Capture a reproducible warning inventory with the ignored autosummary tree
   rebuilt from scratch; separate maintained-prose, autodoc parsing, missing
   cross-reference, and optional-dependency diagnostics.
2. Define the supported semantic conventions in `docs/README.md`:
   stable label namespace; numbered/captioned figures and tables; labeled
   display equations and semantic references; alt text and source/provenance;
   heading structure; units and symbols; citation form; permitted raw HTML/rST
   escape hatches.
3. Configure Sphinx/MyST for consistent figure/table/equation numbering and
   cross-reference rendering in both HTML and any future LaTeX/PDF builder.
   Enable only extensions exercised by the rules or pilot.
4. Replace the blanket `docutils` suppression. Prefer fixing malformed
   high-value public docstrings and resolving internal references. If the full
   inherited debt is too broad, install a narrow, documented baseline/filter
   that cannot suppress diagnostics originating in maintained `docs/` prose
   and fails on new autodoc diagnostics.
5. Add `cxr-dev docs` as the canonical clean strict build (`-W`, nitpicky where
   supportable, and keep-going for complete diagnostics). Keep network link
   checking explicit rather than part of the offline default.
6. Apply the conventions to one representative maintained page containing
   multiple semantic objects. Verify labels, numbering, references, captions,
   math, and accessibility in rendered output without changing scientific
   meaning or validation status.
7. Add focused tests for the command contract and for the warning boundary so
   future prose errors cannot disappear behind the autodoc allowance.
8. Run the clean strict build, focused tests, lint, and a scoped diff review.
   Record any intentionally deferred repository-wide docstring cleanup as a
   separate follow-up, not an unbounded extension of this task.

## Decisions and open questions

- **Decided:** retain MyST Markdown and the mixed Sphinx source tree; no bulk rST
  migration.
- **Decided:** semantic numbering and references are opt-in per meaningful
  object, not a mechanical rewrite of every table or equation.
- **Decided:** the offline strict HTML build is the required gate. External
  link checking remains separate because network availability is not
  deterministic.
- **Decided:** warning suppression must distinguish maintained prose from
  inherited autodoc debt; a global `docutils` category suppression is not an
  acceptable end state.
- **Open:** whether native Sphinx/MyST citations suffice or a shared BibTeX
  database plus `sphinxcontrib-bibtex` is justified. Decide from repeated-source
  reuse and desired bibliography output; do not add a dependency only for
  syntax preference.
- **Open:** select the pilot page after the warning inventory. Prefer a stable,
  already-verified document with equations and at least one table/figure; avoid
  changing an unverified physics record merely to demonstrate formatting.
- **Open:** whether nitpicky unresolved-reference checking can become global in
  this slice or needs a checked baseline followed by a dedicated public API
  reference cleanup.

These open choices affect configuration and dependency shape, so this task is
not a Serena one-shot until they are resolved.

## Delegation slices

1. **Warning inventory and proposed boundary** — read-only, self-contained
   investigation; suitable for one-shot review after an explicit command and
   classification format are supplied. Required skill:
   `documentation-maintenance`; add `scientific-library` when interpreting
   docstring ownership.
2. **Typesetting rules/configuration/pilot** — one coherent reviewed slice, but
   not one-shot until the citation and pilot choices are resolved. Required
   skill: `documentation-maintenance`; matching physics review skill if claim
   text changes.
3. **Developer command and regression tests** — self-contained after the
   warning-boundary contract is approved; suitable for one-shot. Required
   skills: `documentation-maintenance`, and `cli-ui-ux` if the interface is more
   than the fixed `cxr-dev docs` command.

## Acceptance checks

- `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev docs` succeeds from a clean
  checkout and performs a fresh warnings-as-errors build.
- Maintained `docs/` parser errors and broken semantic references fail that
  command; inherited exceptions, if any, are narrowly documented and checked
  against regression.
- The representative page renders numbered/captioned semantic objects with
  stable cross-references and accessible image text where images exist.
- `docs/README.md` gives copyable MyST examples and states when plain Markdown,
  MyST directives, and local rST escape hatches are appropriate.
- Autosummary/autodoc still generate the API reference; generated CLI reference
  and deprecation files remain unchanged unless source behavior changed.
- Focused developer-command/config tests pass.
- `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint` passes.
- A direct clean `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run --group docs
  sphinx-build -W -b html docs <temporary-output>` succeeds as an independent
  check of the wrapper.
