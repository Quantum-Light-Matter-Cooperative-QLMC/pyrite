# Documentation authoring and maintenance

This page defines the conventions for maintained project documentation.

Project documentation is written in MyST Markdown and built with Sphinx. Plain Markdown is preferred for headings, paragraphs, lists, simple unnumbered tables, links, and inline mathematics.

Use MyST roles and directives when an object needs a stable label, caption, number, cross-reference, or accessibility metadata. Use raw HTML or embedded reStructuredText only when MyST cannot express the required result; add a short comment explaining the escape hatch and verify both HTML and future LaTeX-neutral semantics.

Agent-operational plans and handoffs do not belong under `docs/`; they belong under `agentdocs/` and are not published as project documentation.

Published documentation and attached evidence use generic usernames, hostnames,
home paths and example network addresses. Redact operator-identifying metadata
before adding logs or reports. Preserve numerical measurements, scientific
configuration, source revisions and content digests; indicate when published
metadata contains placeholders rather than original operator paths.

## Choose the reader's documentation need

PyRITE uses [Diátaxis](https://diataxis.fr/) as an editorial guide, not a rule
that the sidebar must have exactly four sections. Choose one primary mode
before adding a page, and link to other modes rather than repeating their content.

| Reader's question | Mode and primary location | Example |
| --- | --- | --- |
| Teach me to do this | **Tutorial**, `docs/notebooks/` | Simulate and plot a first PXR spectrum |
| How do I accomplish this task? | **How-to**, `docs/guides/` | Configure a detector acquisition |
| What does this interface accept? | **Reference**, `docs/reference/` and public docstrings | `Observation` parameters |
| Why does this work? | **Explanation**, `docs/physics/` | Detector acceptance and broadening |

The existing `docs/api.md` is a compatibility-preserving inventory while
namespace pages grow under `docs/reference/python-api/`. CLI help and settings
remain generated from their canonical owners. Do not manually synchronize API
signatures or introduce unsupported/private objects to make a reference complete.

**Tutorials** guide a novice through one scientific objective, with minimal
setup, visible results, short interpretation, and suggested variations. The
Marimo/MoLab notebook source and static export workflow are owned by #758;
use the `tutorial-authoring` skill, not the application `notebook-workflow`.
**How-tos** assume relevant background and give a concise, tested recipe for a
specific outcome, with required alternatives or cautions but no introductory
lesson. A short API example need not become a separate guide. **Reference**
describes complete and authoritative contracts without a long narrative;
**explanation** develops physical or numerical reasoning without duplicating
signature tables.

Additional sections serve distinct audiences or evidence needs:

- `docs/validation/` owns applicability, independent evidence and sign-off
  status; verification is not automatically experimental validation.
- `docs/development/`, `docs/repo-design/` and `docs/adr/` own contributor
  procedures, current internals, and durable architectural decisions.
- `docs/research/` marks proposals and unsupported explorations explicitly.
- `README.md` and Getting Started provide brief entry points, not fifth modes.

Find the canonical existing owner before creating a page. Choose task-based
titles, short sections, and one primary toctree parent; group long inventories
behind section index pages. Preserve published URLs and stable scientific
anchors when relocating content; add a redirect/compatibility page if necessary.
Review inbound links and move the explanation rather than copying it. Every
scientific claim still needs applicable units, assumptions, provenance,
limitations, and links to its validation evidence. Agent plans stay outside
published documentation.

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

API pages also provide Sphinx's `[source]` links. For functions that preserve
a historical `__module__` after relocation, the documentation extension resolves
the implementation from the function's code filename through Sphinx's
`viewcode-follow-imported` hook. Runtime import and pickle identities stay intact.
Alias pages link to their documented class; data exports use `autodata` without
expanded values. Do not add private helpers to the public API inventory solely
to obtain links.

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

(static-executable-tutorials)=
## Existing static examples

New guided learning content belongs in the Marimo/MoLab curriculum under
`docs/notebooks/`, following `tutorial-authoring` and #758. Do not extend the
legacy static example system into a parallel tutorial curriculum. Retain its
checked outputs while examples still support published pages; when a notebook
supersedes a walkthrough, link to that notebook and preserve useful task recipes
and reference material at their canonical owners.

Existing static examples use maintained Python sources plus Matplotlib's
[`plot` directive](https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html)
inside MyST pages. A `literalinclude` displays the executable source and the
directive places its generated figure directly below it in the PyData theme.
Wrap `.. plot::` in an `eval-rst` fence: the directive inserts generated
reStructuredText, which MyST's direct directive adapter cannot handle.
Give each inline plot on a page a distinct `:filename-prefix:`. Separate
`eval-rst` blocks can reset the automatic plot counter and otherwise reuse the
first image even when their code differs.
Matplotlib is declared in the docs dependency group. Interactive workflows keep
the marimo export below.

The infrastructure prototype compared these approaches:

| Approach | Strength | Cost for the current static guides |
| --- | --- | --- |
| [MyST-NB](https://myst-nb.readthedocs.io/en/latest/computation/execute.html) | Native notebook cells/outputs and `jupyter-cache` execution | Adds a notebook parser/kernel/cache stack alongside current MyST pages; cache keys still need care for external data and package changes. |
| [Sphinx-Gallery](https://sphinx-gallery.github.io/stable/configuration.html) | Executable scripts, captured outputs, figures and generated galleries | Adds gallery generation and script narrative conventions for a workflow that needs one figure in an existing page. |
| [marimo export](https://docs.marimo.io/guides/exporting/) | Maintained reactive sources export to HTML or notebook formats | Best for interactive analysis; native notebook rendering adds an export and parser stage. Static HTML is a separate embedded page. |
| Matplotlib `plot` directive (selected) | Existing MyST, copyable Python source, native figures, no new rendering package | Explicitly manage small precomputed outputs and their provenance. |

Maintenance conventions for existing examples:

- `docs/examples/NAME.py` owns an executable example. `compute()` returns a
  JSON-serializable mapping (including resolved seed and result provenance);
  `plot(data)` returns its Matplotlib figure. A main guard runs the workflow and
  `plt.show()`. Analysis examples may accept a saved input path in `compute`;
  pass it through the regeneration command's `--input PATH` option. Use
  canonical public imports. Use supported PyRITE plotting
  functions when they accept the result; direct Matplotlib is appropriate for
  native-grid component comparisons, including loaded checkpoint records,
  without manufacturing campaign or `Result` adapters.
- `docs/examples/data/NAME.json` is a small, committed output record. Scientific
  tutorials store **simulation output**, never invented or experimental data;
  configuration and material-authoring examples store actual inspection output.
  The wrapper records source SHA-256,
  package/Python versions and generating checkout revision. Preserve result
  provenance, table/model identity, units and actual resolved seeds. A dirty
  generating checkout is recorded by result provenance; the source hash pins
  the tutorial itself. Do not claim cross-version/backend bitwise identity.
- `docs/_tutorials.py` registers examples in `NAMES`, regenerates their outputs
  only on explicit request, and supplies a named rendering entry point for each
  `plot` directive. It rejects missing outputs or changed source hashes at
  Sphinx builder initialization. Rendering uses the same `plot(data)` function
  as the runnable simulation. No Monte Carlo or table downloads run in Sphinx.
- CLI tutorials may return `display_text` instead of a figure. Regeneration
  writes that captured output to `docs/examples/data/NAME.txt` for
  `literalinclude`; freshness checks verify it matches the JSON record.
  Keep machine-readable completion metadata alongside the transcript. A CLI
  exercise may own a downloadable `docs/examples/NAME.sh`, invoked by its
  Python capture helper. Store its SHA-256 as `script_sha256`; freshness checks
  then reject edits to either source. Isolate user configuration, writable
  catalog and workspace before any example writes.
- Figures live only in `docs/_build/` (PNG at 150 dpi plus SVG). Do not commit
  hand-edited screenshots. Plot native component grids separately; label units,
  normalization and sample/convergence limitations in the tutorial.
- Inputs requiring external fixtures belong in `docs/examples/data/` with
  source/license/provenance metadata. Record their filenames and SHA-256 hashes
  in the output's `input_sha256` mapping; freshness checks reject changed inputs.
  Existing small repository evidence can instead be pinned by repository-relative
  path in `repository_input_sha256`. Analysis-only examples preserve the original
  simulation revision/version separately from their rendering-wrapper metadata;
  reducing historical evidence does not constitute a fresh simulation.
  Do not silently fetch them. Large or
  GPU-dependent runs require explicit regeneration outside the normal build;
  publish only bounded reduced outputs. Optional interactive entry points use
  maintained marimo sources and lightweight/precomputed inputs.

For the first example, explicitly regenerate with installed reference tables:

```bash
MPLCONFIGDIR=/tmp/pyrite-matplotlib PYRITE_MC_BACKEND=cpu \
  uv run python docs/_tutorials.py --regenerate python_api_workflow
```

The results tutorial loads an existing single-case quickstart checkpoint:

```bash
uv run python docs/_tutorials.py --regenerate working_with_results \
  --input pyrite-output/checkpoints/hopg@quickstart-cdf8eed9af74
```

Use the stem reported by your run when its resolved inputs differ. This
regeneration reads stored components and preserves their hashes and dataset
identity; it does not run transport.

Review source and JSON changes together. For a deterministic check in the same
environment, regenerate twice and compare spectral arrays and resolved identity;
the metadata can change independently. Source-hash checking detects tutorial
edits, not scientific drift in the library: rerun affected examples explicitly
after relevant model/API/data changes. Expensive examples belong in scheduled or
explicit checks, never normal RTD execution.

`pyrite-dev docs` checks output freshness and renders every registered tutorial
under warnings-as-errors, with no table dependency. It clears `_build`, so plot
directive caches cannot hide changed output data. Incremental Sphinx builds can
reuse figure caches; use the canonical clean command for verification. Focused
freshness/rendering checks live in `tests/dev/test_docs_tutorials.py`.

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


## Interactive notebook captures

Canonical tutorial notebooks live in `docs/notebooks/`, independently of
`pyrite.apps`. Their documentation pages embed normal marimo static HTML and
include the same `.py` source. Follow this page's authoring policy and the
repository's tutorial-authoring guidance when available. Keep conceptual
explanation, task recipes, interface contracts and validation evidence at their
existing canonical owners; link to them from the learning sequence.

Install the docs dependency group and fetch/verify reference tables once before
executing a transport notebook. Run each notebook locally with
`uv run marimo edit docs/notebooks/01-first-spectrum.py`, or open its
notebook-specific Molab badge on its RTD page. Place the badge immediately after
the RTD page title; keep it out of the notebook source. Inline dependency metadata
pins the PyRITE release for standalone/Molab execution; checkout captures use the current project.
The explicit `--regenerate-notebook NAME` command first synchronizes that
notebook's inline `pyrite-mc` pin and any installation examples with the version
in `pyproject.toml`. Source, session, figure and manifest must be committed
together. Normal docs builds and `--check-notebooks` never change sources.
The first cell of each notebook calls `pyrite.fetch_tables()` immediately after
import, installing missing pinned tables before simulation. A complete install
returns without network access. This runs only during explicit capture, local
marimo or Molab execution, **not** during Sphinx builds; a fresh hosted session
may need network access to acquire tables.

Regenerate every registered tutorial on CPU with `pyrite-dev notebooks`. The
command provisions the docs dependency group, synchronizes version pins and
executes each notebook sequentially, stopping on the first failure. `verify`
runs this regeneration before its Sphinx build, so it refreshes captured files
and needs the CPU runtime/reference tables. `pyrite-dev docs` itself remains
capture-only. Use `--check` for read-only freshness/import/badge validation, or
the lower-level single-notebook command for a targeted refresh:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run --group docs python docs/_tutorials.py \
  --regenerate-notebook 01-first-spectrum
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev notebooks
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev notebooks --check
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run marimo check docs/notebooks/*.py
```

The exporter first runs marimo's structure check, then its session and normal
HTML exports (both execute the same notebook). Both exports and the Sphinx
browser-demo export pass `--no-sandbox`, using the current project environment
without inline-dependency sandbox prompts. The WASM demo retains its inline
browser dependencies and Pyodide runtime; `--no-sandbox` only disables the local
export-time environment prompt. Its frontend assets, worker directory and
`public/` data remain part of the RTD export. The exporter includes source code,
extracts PNG fallback figures from the captured session, and records source/exporter and
artifact digests, Git revision/dirty state, Python/PyRITE versions, backend,
resolved seeds and table identities. Commit source, HTML, session, figure and
manifest together. Review the figures and provenance; freshness hashes establish
source/artifact correspondence, not scientific validity. A dirty capture records
the pre-commit revision together with the exact notebook digest.

Sphinx only validates and copies these committed captures. It never executes the
scientific notebooks or fetches their reference tables. CI's documentation build
therefore rejects stale source/artifact digests and unsupported imports without
requiring a GPU. Focused tutorial tests additionally check capture contents and a
small deterministic CPU notebook. Transport smoke execution belongs in explicit
capture regeneration, not the documentation build. Large/GPU examples need an
explicit remote regeneration process; do not add them to this CPU exporter.

Check the notebook-specific Molab path and inline dependencies before publishing.
A badge targeting `main` becomes launchable after its source is merged; do not
claim that a live Molab session was checked from an unpublished branch.
