# Rederivation LaTeX/MyST style rules

Formatting-only rules for `docs/validation/<domain>/<id>.md` write-ups. These
rules make derivations human-readable and consistent; they never license a
content, sign, factor, or wording change. A worker applying this ruleset to an
existing write-up touches markup only — if a fix looks like it also changes
what the document claims, stop and flag it instead of editing.

The Sphinx build (`docs/conf.py`) enables MyST `dollarmath` and `amsmath`.
Only `$...$`, `$$...$$`, and the ```` ```{math} ```` directive render.
`\(...\)` and `\[...\]` do **not**: CommonMark eats the backslash escape, MyST
tags the page `mathjax_ignore`, and the LaTeX source lands in the page as
literal text. Never use them.

## Three content types, three markups

A write-up mixes three different things. Keep them visually distinct:

1. **Abstract physics** (the independent derivation, the source equation, any
   symbolic expression you'd write on paper) → **LaTeX math**, per the rules
   below. Never put a physics expression in a plain code span or code fence —
   `` `r_e = e^2/(mc^2)` `` reads as code, not as the equation it is.
2. **Literal source excerpts** (pseudocode reproducing the actual
   implementation, with its exact variable spellings, `**`, `@`, `np.exp`,
   line continuations) → plain fenced code block (```` ``` ```` or ` ```text `).
   Do not "clean up" these into LaTeX — they are evidence about what the code
   says, and their value is being literal.
3. **Individual code identifiers** referenced in prose (`chi_g`, `photon_E_eV`,
   `mc_spectrum`, `B_ang2`) → inline code span with the exact spelling from
   source. Do not convert these to a math symbol even if they name a physical
   quantity — the point of the backtick is "this is what the code calls it."

When prose needs both — the code identifier and the physics symbol it
represents — give both: "the docstring's `B_ang2` is the Debye–Waller
parameter $B$."

## LaTeX conventions for content type 1

- **Inline math:** `$...$`, on one line, with no space just inside either
  delimiter. Example: `$\chi_{\mathbf g}$`.
- **Display math:** `$$` on its own line, above and below one unnumbered
  equation or a short aligned block. Example:

  ```text
  $$
  \chi_{\mathbf g}
  =-\frac{4\pi r_e}{k^2V_{\rm cell}}S_{\mathbf g}.
  $$
  ```

- **Numbered/cross-referenced equations only:** the ```` ```{math} ```` +
  `:label:` directive. Most write-ups never need a cross-reference; default to
  `$$`.
- **Never** use `\(...\)` / `\[...\]` — they are silently dropped to literal
  text in the HTML build. A LaTeX line break inside display math stays
  `\\[2pt]`; it is not a delimiter.
- Math inside a `{list-table}` or other directive body is still parsed as
  markdown, so it takes `$...$` too.
- **Do not** write bare-ASCII pseudo-math (`k=omega/c=2 pi/lambda`,
  `beta_i^2`, `S(g)`) inside inline code spans as a substitute for LaTeX. If
  it is an equation, it gets LaTeX delimiters and macros (`\omega`, `\pi`,
  `\lambda`, `\beta_i^2`, `S(\mathbf g)`).

### Symbol conventions

- Vectors: `\mathbf{g}`, `\mathbf{r}`, `\mathbf{v}`. Unit vectors get a hat on
  the bold symbol: `\hat{\mathbf{n}}`.
- Greek letters and named constants always use their macro inside math mode:
  `\alpha`, `\beta`, `\pi`, `\omega`, `\hbar`, `\gamma` — never the spelled-out
  ASCII form (`alpha`, `pi`, `hbar`) once inside `$...$`/`$$`.
  Spelled-out ASCII is fine in code fences/spans (content types 2–3) because
  that is what the source literally says.
- Subscripts/superscripts: `\beta_i`, `\beta_i^2`, `T_f`, not `beta_i`,
  `beta_i^2`, `T_f` inside math mode.
- Absolute value / norm: `\lvert \cdot \rvert` or `\|\cdot\|` consistent with
  the quantity (amplitude modulus vs. field norm) — match whichever the
  ledger row already uses for that claim rather than introducing a new
  convention within one write-up.
- Operators from the derivation (`\operatorname{sinc}`, `\exp`, `\ln`) use the
  matching LaTeX macro, not upright/italic ambiguity from plain text.
- Boxed key results (the sign/factor the write-up is built around) may use
  `\boxed{...}` inside display math, matching existing usage in
  `radiation-physics/coherent-emission.md`.

### Tables

Inline math (`$...$`) renders inside GFM table cells as long as the cell
contains no literal `|`. Prefer this over ASCII pseudo-math in comparison
tables (units/limits/term-by-term diff tables are common in these write-ups).

## Worked example

Before (mixed styles, from a pre-cleanup write-up):

````text
Using `r_e=e^2/(mc^2)`, `k=omega/c=2 pi/lambda`,

```{math}
\chi_{\mathbf g}
=-\frac{4\pi r_e}{k^2V_{cell}}S_{\mathbf g}
=-\frac{r_e\lambda^2}{\pi V_{cell}}S_{\mathbf g}.
```
````

After:

```text
Using $r_e = e^2/(mc^2)$ and $k = \omega/c = 2\pi/\lambda$,

$$
\chi_{\mathbf g}
=-\frac{4\pi r_e}{k^2 V_{\rm cell}}S_{\mathbf g}
=-\frac{r_e\lambda^2}{\pi V_{\rm cell}}S_{\mathbf g}.
$$
```

Only the markup changed — same symbols, same factors, same sign.

## Checklist for a cleanup pass

- [ ] Every abstract-physics expression uses `$...$`/`$$`, not a code span or
      ASCII-only code fence.
- [ ] No `\(`/`\[` delimiters remain.
- [ ] Greek letters, vectors, and operators use LaTeX macros inside math mode.
- [ ] Literal source pseudocode stays in a fenced code block, untouched.
- [ ] Code identifiers stay as exact-spelling inline code, not converted to
      symbols.
- [ ] Diff contains no changed number, sign, exponent, unit, word of
      adjudication, or verdict — `git diff` should read as pure markup
      substitution.
- [ ] `uv run pyrite-dev docs` builds without new Sphinx/MyST warnings for the
      touched file.
- [ ] The rendered page proves it: every expression in the built
      `docs/_build/html/validation/<domain>/<id>.html` sits inside a
      `class="math notranslate"` element, and no `$` survives in the article
      body. An unrendered equation raises no warning, so this positive check is
      the only one that catches it.
