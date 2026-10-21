# Documentation authoring and maintenance

This page defines the conventions for maintained project documentation.

Project documentation is written in MyST Markdown and built with Sphinx. Plain Markdown is preferred for headings, paragraphs, lists, simple unnumbered tables, links, and inline mathematics.

Use MyST roles and directives when an object needs a stable label, caption, number, cross-reference, or accessibility metadata. Use raw HTML or embedded reStructuredText only when MyST cannot express the required result; add a short comment explaining the escape hatch and verify both HTML and future LaTeX-neutral semantics.

Agent-operational plans and handoffs do not belong under `docs/`; they belong under `agentdocs/` and are not published as project documentation.

## Structure, labels, and references

Use one `#` document title, then descend through headings without skipping a level.

Heading anchors are conveniences, not stable scientific references. Label meaningful objects with lowercase, hyphenated, globally unique names:

* `fig-<topic>-<object>` for figures,
* `tbl-<topic>-<object>` for tables,
* `eq-<topic>-<quantity>` for equations.

Do not encode an object's rendered number in its label.

Refer to numbered objects semantically:

* use `{numref}` for figures and tables,
* use `{eq}` for equations,
* use `{ref}` for labeled sections and other targets.

Do not manually write references such as “Table 3” or “the equation above” when a semantic cross-reference can be used instead.

Link published pages with relative Markdown links, for example
`[Atomic data sources](../physics/atomic-physics/atomic-data-sources.md)`.
These follow the documentation version in tags and PR previews. Use `{doc}`
for a document reference and `{ref}` for a stable explicit claim label, such
as `validation-atomic-form-factor`.

For documented Python objects, use the matching Python-domain role with its
fully qualified name: `{py:func}`, `{py:class}`, `{py:meth}` or `{py:mod}`.
For example:

```markdown
{py:func}`pyrite.materials.atomic.atomic_form_factor`
{ref}`validation-atomic-form-factor`
```

API pages also provide Sphinx's `[source]` links. Do not add private helpers to
the public API inventory solely to obtain links.

### Current and historical source references

Use the checked `source` role for repository files and private Python owners:

```markdown
{source}`src/pyrite/materials/atomic.py`
{source}`src/pyrite/montecarlo/spectrum/cxr/_formation.py::formation_factor`
{source}`tests/montecarlo/test_coherent_emission.py`
```

Paths start at the repository root. An optional `::` suffix names a definition
or dotted class member. The build checks that the file and definition exist,
using the ledger's AST symbol resolver without importing the owner. It emits
a GitHub link with the familiar path label and the revision in its tooltip.
Definition links include a line anchor pinned to that revision.

`docs/conf.py` owns the repository URL and revision. Hosted builds use
`READTHEDOCS_GIT_COMMIT_HASH`; local checkouts use `git rev-parse HEAD`.
Source archives without Git fall back to the release tag. Commit local changes
before publishing so the linked commit contains the reviewed source.

Keep canonical ledger `Code` values as literal `file::symbol` anchors: the
ledger parser and marker checks consume that format. Add checked source links
in the detailed write-up alongside those anchors. Generated status/domain
views contain no implementation fields and need no source-link rewrite.

Label historical excerpts, paths and line numbers as historical. Link them
with ordinary Markdown URLs pinned to their recorded commit; the current-source
role deliberately checks today's checkout and cannot validate a historical tree.
Do not replace a historical reference with an assertion about current ownership.

The warning-as-error build rejects missing explicit document, claim and
Python-role targets in maintained Markdown, and missing files/definitions in
`source` roles. It does not enable global nitpicky checking of inferred API
types. Literal code spans, code excerpts, unlinked identifiers and historical
external URLs are outside this check; retain them when quoting source or when
no resolvable target exists. These checks establish navigation and current
ownership, not physical validity or external GitHub availability.

Cite external scientific sources with a descriptive author/year/title/DOI link in prose or with a MyST footnote. Add a shared BibTeX dependency only when sources are reused often enough to justify repository-wide bibliography ownership.

### Labeled equations

A labeled equation and semantic reference should use the MyST math directive:

````markdown
```{math}
:label: eq-example-doppler-factor

D = 1 - \boldsymbol{\beta}\cdot\hat{\mathbf n}
```

The spectrum scales with the Doppler factor in
{eq}`eq-example-doppler-factor`.
````

## Tables

Use semantic tables for tabular data.

Tables intended only for layout are not semantic tables; use prose, lists, grids, or another appropriate structure instead.

Data tables should have header cells, and quantities should include units in their headings when applicable.

A numbered table can be written as:

````markdown
```{list-table} Limiting behavior of the example model.
:name: tbl-example-limits
:header-rows: 1

* - Limit
  - Result
* - $\beta \to 0$
  - $D \to 1$
```

The limits are summarized in {numref}`tbl-example-limits`.
````

## Figures

Every numbered figure needs a concise caption stating what is shown.

Figures should also have useful alt text. Describe the information conveyed by the figure rather than its colors, filename, or purely visual appearance.

Record the source or provenance in the caption or adjacent prose, including whether the asset is measured, simulated, or adapted.

A standard figure looks like:

````markdown
```{figure} figures/example-spectrum.svg
:name: fig-example-spectrum
:alt: Simulated intensity peaks at 1.4 keV and falls toward both grid edges.

Simulated spectrum for the stated beam and crystal configuration; generated by
`checks/example.py` from catalog revision `<commit>`.
```
````

## Equations and physical claims

Prefer display-math directives for equations that are referenced elsewhere or that carry a substantive scientific claim.

Define every symbol near first use.

State assumptions and limiting cases when they affect the interpretation or domain of validity of an expression.

Physics functions and claims subject to the project's formal verification process should also follow the conventions defined in the [physics validation methodology](../validation/methodology.md).

## Citations

External sources are cited through a single BibTeX bibliography, `docs/references.bib`, rendered at [References](../references.md). Do **not** write a full reference inline or as a Markdown footnote: the same source is cited from several pages, and duplicated reference text drifts.

Add the entry to `docs/references.bib` with a `firstauthor+year` key, then cite it with the MyST roles:

````markdown
The stopping power is the Joy--Luo modification of the Bethe expression{cite:p}`joyluo1989`.

{cite:t}`feranchuk2000` derive the PXR and CBS amplitudes together.
````

`{cite:p}` is parenthetical (a bare numeric mark, for a claim already stated in the prose) and `{cite:t}` is textual (renders the author names, for when the source is the grammatical subject). Every entry carries a DOI or a stable URL — a reference that cannot be resolved is not a citation.

Cite the source for any equation reproduced from the literature, any tabulated dataset, and any number a model is compared against. When a page states a source's *equation number*, that number must be traceable to something the repository actually holds; if it comes from a ledger `Source` field rather than from the paper itself, say so rather than implying the transcription was checked.

Per-material provenance is not literature: individual catalog crystals record their refinements in the [crystal structure provenance ledger](../validation/ledger-crystal-structure-provenance.md).

## Units

Use unambiguous unit spellings and place a space between the value and unit:

* `30 keV`
* `2.5 µm`
* `0.4 rad`

Keep mathematical symbols italic and unit text upright where ambiguity matters.

Preserve the repository's stated natural-unit convention rather than silently mixing SI and $\hbar=c=1$ expressions.

## Architecture and design records

Long-lived descriptions of how the current system works belong in the appropriate reference or repository-design documentation.

Durable records of **why** an architectural choice was made belong in the [architecture decision records](../adr/index.md).

Long-form RFC/design documents may contain detailed rationale and alternatives while a design is active. Once accepted, the ADR records the durable result; obsolete planning detail may be retired and remains available through Git history.

Agent task plans and handoffs belong under `agentdocs/`, not under `docs/`.

## Building the documentation

Before review, run:

```bash
pyrite-dev docs
```

This removes generated doctrees and autosummary stubs and performs the canonical offline warnings-as-errors documentation build.

To additionally check external links, run:

```bash
pyrite-dev docs --linkcheck
```

External link checking requires network access and is therefore not part of the offline gate.

## Browser results demo

The overview page embeds `docs/demo/results_app.py`, a marimo notebook that
imports only `marimo`, `numpy`, and `plotly` so it runs under Pyodide. The
`_marimo_demo` Sphinx extension exports it with `marimo export html-wasm` into
`demo/` of every HTML build (about 27 MB of marimo frontend assets) and copies
`docs/demo/public/` beside it; Pyodide and packages load from a CDN at view
time. Keep the directory export: a `--single-file` page runs Pyodide from a
`blob:` worker, so `mo.notebook_location()` cannot find the bundled data. The
page needs HTTP; preview a local build with
`python -m http.server -d docs/_build/html`, not `file://`. The notebook reads only the committed
`docs/demo/public/results_demo.json`, which carries its own provenance
(version, commit, model markers, ledger statuses read at export time).

Regenerate the data and the README preview image after populating the Zhai
cache:

```bash
pyrite run --preset zhai          # or --remote, then pull the cache back
uv run python scripts/export_docs_demo.py
```

The export refuses a missing or stale cache and never reruns the reproduction.
Commit the JSON and `docs/_static/demo/results-demo-preview.png` together.

## Warning policy

The checked autodoc warning baseline is intentionally empty: source docstrings must render without parser warnings. The fingerprint mechanism remains so any new source-docstring warning fails the build for review.

Warnings produced by maintained prose under `docs/` are never baselined. Any change to the zero-warning source-docstring fingerprint fails the build.

To inspect a source-docstring warning that changed the fingerprint, run the same build with:

```bash
PYRITE_DOCS_SHOW_AUTODOC_WARNINGS=1
```

The build intentionally fails while printing every baseline record.

## Navigation and discovery checks

Keep the homepage concise and route users to installation, physics, and API
workflows. Global navigation shows major sections and the active branch;
validation evidence, API objects, and ADR inventories remain reachable from
their section pages. Preserve existing page URLs when reorganizing toctrees.

After navigation changes, inspect the rendered homepage, Getting started,
physics, validation, and API at desktop and narrow widths. Check sidebar
scrolling, focus/tab order, mobile opening and closing, and deep links. Generated
references must be regenerated at their source, including family CLI pages and
validation status/domain summaries.

The homepage metadata is maintained in `index.md`; Sphinx's `html_baseurl` uses
`READTHEDOCS_CANONICAL_URL` with a local-build fallback to `en/latest/`. Read the
Docs supplies the preferred domain/version as described in its
[canonical URL documentation](https://docs.readthedocs.com/platform/stable/canonical-urls.html).
Inspect the built homepage and an API page for distinct per-page canonical
links. Repeat on the deployed site: local output cannot prove the hosting
version policy or a search engine's selected canonical.

On October 17, 2026, public HTTP checks returned a sitemap containing
`en/latest/` and `en/stable/`; robots.txt allowed crawling and linked that
sitemap. The then-deployed homepage lacked a canonical link. After deployment,
recheck [the sitemap](https://pyrite.readthedocs.io/sitemap.xml), homepage
metadata, and an API page. A project owner must use Google Search Console to
inspect homepage indexability, last crawl, selected canonical, and sitemap
submission. Repository-only automation has no authenticated access to that
property. Metadata and indexing requests do not guarantee search ranking;
retain useful API and validation pages in search.
