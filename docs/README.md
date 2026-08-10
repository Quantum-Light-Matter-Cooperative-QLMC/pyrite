# Documentation map

Guides, references, and validation records. They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../docs/repo_map.md`](repo_map.md) — canonical package ownership and dependency map.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.
- [`../agentdocs/`](../agentdocs/) — tracked agent task plans and handoffs;
  never published project documentation.

| Document | Topic | Status |
|---|---|---|
| [cli-reference.md](cli-reference.md) | Generated reference for every current `pyrite` command | authoritative |
| [cli-deprecations.md](cli-deprecations.md) | Generated compatibility and removal-window registry | authoritative |
| [api.md](api.md) | Generated Python API reference | authoritative |
| [development-workspace.md](development-workspace.md) | Single-project contributor environment and focused verification | guide |
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `pyrite run` under SLURM (`sbatch` + job-array templates) | guide |
| [performance-profile-analysis.md](performance-profile-analysis.md) | Analyze performance-profile NDJSON, classify bottlenecks, and design controlled tuning runs | guide |
| [sweep-profiles.md](sweep-profiles.md) | Named full/survey fidelity policies, resolved provenance, and variant checkpoint identity | guide |
| [checkpoint-case-store.md](checkpoint-case-store.md) | Cross-profile per-case CAS, manifests, cache modes, and compatibility | implemented |
| [physics-validation-ledger.md](physics-validation-ledger.md) | Physics claim status and evidence | living ledger |
| [coherent-emission.md](coherent-emission.md) | Optional phased segment/electron sum and validation boundary | experimental, unverified |
| [coherent-streaming-rawkernel.md](coherent-streaming-rawkernel.md) | Streaming coherent GPU reduction design and evidence | implemented |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic broadening and exact orientation averaging | implemented |
| [detector-solid-angle.md](detector-solid-angle.md) | Default single-direction treatment and opt-in face integral | opt-in integral implemented |
| [external-bremsstrahlung-validation.md](external-bremsstrahlung-validation.md) | Versioned external-background fixtures, comparison, fitting, and subtraction | implemented |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: absorption, radiation, transport | implemented |
| [atomic-data-sources.md](atomic-data-sources.md) | Atomic scattering data supplied by xraydb | adopted |
| [crystal-db-comparison.md](crystal-db-comparison.md) | Offline external-database lattice cross-check | implemented |
| [debye-waller-audit.md](debye-waller-audit.md) | Thermal-displacement provenance and scalar/tensor model scope | in progress |

## Decisions

These are dev-facing and excluded from the published site.

- [`adr/`](adr/) — numbered architecture decision records (MADR-lite); the
  durable, greppable "why we decided X" log.
- `*-rfc.md` — long-form design rationale (surface, artifact model, structure);
  each accepted RFC gets an ADR stub.

Agent-operational plans and handoffs belong only in
[`../agentdocs/`](../agentdocs/), not under `docs/`.

## Authoring conventions

Write maintained prose in MyST Markdown. Plain Markdown is preferred for
headings, paragraphs, lists, simple unnumbered tables, links, and inline math.
Use MyST roles and directives when an object needs a stable label, caption,
number, cross-reference, or accessibility metadata. Use raw HTML or embedded
reStructuredText only when MyST cannot express the required result; add a short
comment explaining the escape hatch and verify both HTML and future
LaTeX-neutral semantics.

### Structure, labels, and references

- Use one `#` document title, then descend through headings without skipping a
  level. Heading anchors are conveniences, not stable scientific references.
- Label meaningful objects with lowercase, hyphenated, globally unique names:
  `fig-<topic>-<object>`, `tbl-<topic>-<object>`, and
  `eq-<topic>-<quantity>`. Do not encode an object number in its label.
- Refer to numbered objects semantically: `{numref}` for figures/tables and
  `{eq}` for equations. Use `{ref}` for labeled sections or other targets.
  Never write a number such as “the equation above” or “Table 3” by hand.
- Link files with relative Markdown links. Cite external scientific sources
  with a descriptive author/year/title/DOI link in prose or a MyST footnote.
  Add a shared BibTeX dependency only when sources are reused enough to justify
  repository-wide bibliography ownership.

A labeled equation and reference:

````markdown
```{math}
:label: eq-example-doppler-factor

D = 1 - \boldsymbol{\beta}\cdot\hat{\mathbf n}
```

The spectrum scales with the Doppler factor in
{eq}`eq-example-doppler-factor`.
````

A numbered table:

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

### Figures, equations, units, and sources

- Every numbered figure and table needs a concise caption that states what is
  shown. A figure also needs useful alt text; describe the information, not its
  colors or filename. Record the source/provenance in the caption or adjacent
  prose, including whether the asset is measured, simulated, or adapted.
- Prefer display-math directives for equations that are referenced or carry a
  claim. Define every symbol near first use. State assumptions and limiting
  cases where they affect interpretation.
- Use unambiguous unit spellings and a space between value and unit (`30 keV`,
  `2.5 µm`, `0.4 rad`). Keep symbols italic in math and unit text upright where
  ambiguity matters. Preserve the repository's stated natural-unit convention
  rather than silently mixing SI and $\hbar=c=1$ expressions.
- Tables intended only for layout are not semantic tables. Use prose or a list
  instead. Data tables need header cells and units in their headings.

Copyable figure form:

````markdown
```{figure} figures/example-spectrum.svg
:name: fig-example-spectrum
:alt: Simulated intensity peaks at 1.4 keV and falls toward both grid edges.

Simulated spectrum for the stated beam and crystal configuration; generated by
`checks/example.py` from catalog revision `<commit>`.
```
````

Run `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs` before review. It
removes generated doctrees/autosummary stubs and performs the canonical offline
warnings-as-errors build. `pyrite-dev docs --linkcheck` additionally checks
external links and therefore requires network access; it is not an offline
gate. The checked autodoc warning baseline covers inherited source-docstring
parser debt only. Warnings from maintained `docs/` prose are never baselined,
and any change to the inherited warning fingerprint fails the build for review.
To inspect that debt, set `CXR_DOCS_SHOW_AUTODOC_WARNINGS=1` for the same
command; the build intentionally fails while printing every baseline record.
