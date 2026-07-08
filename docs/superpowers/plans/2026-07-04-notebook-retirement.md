# Notebook Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete every `.ipynb` in the repo (`notebooks/scan.ipynb`, `notebooks/analysis.ipynb`, `checks/cxr_analysis_feranchuk.ipynb`, `checks/zhai_fig1c_check.ipynb`, `checks/zhai_fig1c_validation.ipynb`) and their jupytext `.md` pairs, converting the three `checks/` notebooks to plain Python first (matching the checks/ directory's existing convention of standalone scripts), then remove the now-dead Jupyter-notebook tooling (docs mentions, `_compile_nb.py`, `dev.py` nbqa/nbstrip commands, pre-commit hooks, jupyter-only pyproject dependencies).

**Architecture:** `notebooks/scan.ipynb`/`analysis.ipynb` are fully superseded by the already-complete marimo apps `notebooks/scan_app.py`/`analysis_app.py` (2026-06-27 migration) — pure deletion, no new file. The three `checks/` notebooks have no marimo equivalent and checks/ has an established convention of plain, non-notebook validation scripts (`feranchuk_check_script.py`, `detector_solid_angle_check.py`, `kinematic_validity_check.py`, ...) — each converts to THAT shape, not to marimo, since none of them have genuine reactive/UI needs. `checks/zhai_fig1c_validation.ipynb` turns out to be fully redundant with `checks/anchor_figures.py::main()` (already a headless equivalent) — it needs no replacement file at all, just deletion.

**Tech Stack:** Python, matplotlib (`Agg` backend for headless figure output), jupytext (read-only, for extracting notebook content — not needed after this plan lands), uv/pyproject.

## Global Constraints

- **Verbatim conversion only** — no physics/logic changes. The three `checks/` notebooks call already-validated library functions (`feranchuk_spence`, `cxr_mc.crystallography`, `cxr_mc.montecarlo`, `checks/anchor_figures.py`); converting their container format must not change a single computed value.
- `docs/physics-validation-ledger.md` does not reference any of the three `checks/` notebooks by name (only `checks/anchor_figures.py` is named, in the `closed-form-flux` row) — confirmed via `grep -in "feranchuk\|zhai_fig1c\|cxr_analysis" docs/physics-validation-ledger.md`. No ledger edits are needed by this plan.
- `notebooks/*.ipynb`, `checks/*.ipynb`, and their jupytext `.md` pairs are IN SYNC at the current commit (verified: `git log -1` on each `.ipynb`/`.md` pair returns the same commit hash, and `git status` is clean for both directories) — safe to read the `.md` files as the accurate source of notebook content instead of parsing the JSON `.ipynb` directly.
- Verify every step with `uv run pytest`, `uv run ruff check .`, `uv run pyright` (repo baseline: 201 tests / 0 ruff / 0 pyright, or whatever the M6 plan left it at if run first).
- Commit convention on this branch: stage files **by name**, never `git add -A` — the pre-existing unstaged `D analysis.py` (root, out of scope) must stay untouched throughout.
- This plan is independent of the M6 frame-builders plan (`2026-07-04-m6-frame-builders.md`) — no file overlap. Order between the two doesn't matter; do M6 first if you want the branch's numbered `TODO.md` backlog cleared before starting this larger one.

---

## File Structure

- **Create** `checks/zhai_fig1c_check.py` — converted from `checks/zhai_fig1c_check.ipynb` (verbatim script, `plt.show()` → `fig.savefig("zhai_fig1c_analog.png", ...)` per the notebook's OWN docstring, which already claims this file writes a PNG).
- **Create** `docs/pxr-cbr-derivation.md` — the PXR/CBR derivation prose (with LaTeX) from `checks/cxr_analysis_feranchuk.ipynb`, moved into `docs/` (design notes) verbatim.
- **Create** `checks/feranchuk_figures_check.py` — the 8 code cells from `checks/cxr_analysis_feranchuk.ipynb`, verbatim, `plt.show()` → `fig.savefig(...)` into a `figures/` output dir (matching `checks/anchor_figures.py::main`'s `outdir` convention), referencing `docs/pxr-cbr-derivation.md` for the physics background in its module docstring.
- **Delete** `notebooks/scan.ipynb`, `notebooks/scan.md`, `notebooks/analysis.ipynb`, `notebooks/analysis.md`, `checks/cxr_analysis_feranchuk.ipynb`, `checks/cxr_analysis_feranchuk.md`, `checks/zhai_fig1c_check.ipynb`, `checks/zhai_fig1c_check.md`, `checks/zhai_fig1c_validation.ipynb`, `checks/zhai_fig1c_validation.md`.
- **Delete** `src/cxr_mc/_compile_nb.py` (only ever compiled the 5 notebooks above; dead once they're gone — not wired into CI, confirmed via grep).
- **Modify** `scripts/dev.py` — remove `iter_notebooks`/`cmd_nbqa`/`cmd_nbstrip` and their subparser entries (no notebooks left to lint/strip); `cmd_repo_map`'s command list drops the two `nbqa`/`nbstrip` lines.
- **Modify** `.pre-commit-config.yaml` — remove the `nbQA` and `nbstripout` hook blocks.
- **Modify** `pyproject.toml` — remove `jupyter-core`, `jupyter-mcp-server`, `jupyterlab`, `jupytext`, `nbconvert`, `nbqa`, `nbstripout` from `dependencies`; remove `[tool.jupytext]`; remove the `"*.ipynb" = ["F821"]` per-file-ignore. Leave `ipykernel`/`ipympl` in place (still reachable transitively via `IPython.display`, used directly by `notebooks/scan_app.py`/`analysis_app.py` and `src/cxr_mc/results/tables.py`, `src/cxr_mc/results/scoring.py`, `src/cxr_mc/plots/interactive.py` — removing them is a separate, unverified risk this plan does not take on).
- **Modify** `.gitattributes` — remove the two `*.ipynb` lines (`filter=nbstripout`, `diff=ipynb`).
- **Modify** `README.md`, `docs/repo_map.md`, `docs/running-on-a-cluster.md`, `.claude/skills/repo-orientation/SKILL.md`, `.agents/skills/repo-orientation/SKILL.md`, `.claude/skills/notebook/SKILL.md`, `.agents/skills/notebook/SKILL.md`, `scripts/export_pdf.py` — replace every `notebooks/scan.ipynb` / `notebooks/analysis.ipynb` / `checks/*.ipynb` mention with the marimo app / plain-script equivalent.
- **Modify** `TODO.md`, `docs/dedup-inventory.md` — record the retirement (this wasn't a pre-existing dedup-inventory item; add a short new paragraph, not a fabricated `M*` id).

---

## Task 1: Convert `checks/zhai_fig1c_check.ipynb` → `checks/zhai_fig1c_check.py`

**Files:**
- Create: `checks/zhai_fig1c_check.py`
- Delete: `checks/zhai_fig1c_check.ipynb`, `checks/zhai_fig1c_check.md`

**Interfaces:**
- Consumes: `cxr_mc.crystallography` (`HBARC_EV_ANG`, `ALPHA_FS`, `CRYSTALS`, `reciprocal_g_vector`), `feranchuk_spence` (`amplitudes_PXR_CBS_both`, `bremsstrahlung_background`), `cxr_mc.montecarlo` (`simulate_trajectories`, `mc_spectrum`, `mc_brem_spectrum`, `beta_from_keV`, `eds_fwhm_eV`, `aperture_fwhm_eV`, `convolve_detector`) — all pre-existing, unchanged.
- Produces: a standalone script; running it prints the validation lines this notebook already prints and writes `zhai_fig1c_analog.png` to the current working directory (repo root, matching how the sibling `checks/*.py` scripts and this notebook's OWN docstring assume `python checks/zhai_fig1c_check.py` is run from repo root).

- [ ] **Step 1: Write `checks/zhai_fig1c_check.py`**

Copy the SINGLE code cell from `checks/zhai_fig1c_check.md` (currently lines 22-238, i.e. everything from `"""` `Reproduce Zhai et al....` through `print(f"\nTotal time: {time.perf_counter() - t0:.1f} s")`) verbatim into the new file, with exactly these changes:
1. Drop the leading bootstrap cell (`sys.path.insert(0, "src")`, `.md` lines 15-20) — replace with the sibling-script convention (`os.path.dirname` relative insert), since this file lives in `checks/` and must resolve `cxr_mc` regardless of CWD, matching `checks/detector_solid_angle_check.py`'s bootstrap.
2. Remove the `%matplotlib inline` line (Jupyter magic, meaningless in a script).
3. Replace the trailing `plt.show()` with a `savefig`, matching the docstring's own claim ("Writes zhai_fig1c_analog.png").

```python
"""
checks/zhai_fig1c_check.py

Reproduce Zhai et al., Nat. Commun. 16, 11218 (2025), Fig. 1c:
tunable X-ray (PXR+CBS) spectra from a 1 mm HOPG bulk crystal under
17.5 / 20 / 22.5 / 25 keV electrons, observed at theta_obs = 119 deg
(EDS take-off geometry) into 0.066 sr, plus the bulk vs 29 nm thin-film
enhancement at 25 keV.

Run:  uv run python checks/zhai_fig1c_check.py   (from the repo root)
Writes zhai_fig1c_analog.png and prints validation + peak tables.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cxr_mc.crystallography import (
    HBARC_EV_ANG,
    ALPHA_FS,
    CRYSTALS,
    reciprocal_g_vector,
)
from feranchuk_spence import amplitudes_PXR_CBS_both, bremsstrahlung_background
from cxr_mc.montecarlo import (
    simulate_trajectories,
    mc_spectrum,
    mc_brem_spectrum,
    beta_from_keV,
    eds_fwhm_eV,
    aperture_fwhm_eV,
    convolve_detector,
)

# < ... copy the rest of the cell body verbatim from checks/zhai_fig1c_check.md
#     lines 57-238 (from "# ---- experimental conditions" through the r10/r_eds
#     print in the E0 loop, the bulk-vs-film block, and the plotting block) --
#     UNCHANGED except the final two lines. ... >

fig.suptitle(
    r"PXR+CBS from HOPG, $\theta_{obs}$=119$\degree$, 0.066 sr "
    f"(Ne = {NE} electrons/energy)"
)
fig.tight_layout()
fig.savefig("zhai_fig1c_analog.png", dpi=150, bbox_inches="tight")
print(f"wrote zhai_fig1c_analog.png")
print(f"\nTotal time: {time.perf_counter() - t0:.1f} s")
```

(`checks/feranchuk_spence.py` is importable without a path hack because `checks/` is already on `sys.path` when running `python checks/zhai_fig1c_check.py` — same as every sibling `checks/*.py` script that does `from feranchuk_spence import ...` with only the `src/` path inserted.)

- [ ] **Step 2: Delete the notebook and its jupytext pair**

```bash
git rm checks/zhai_fig1c_check.ipynb checks/zhai_fig1c_check.md
```

- [ ] **Step 3: Run the new script and sanity-check its output**

Run: `uv run python checks/zhai_fig1c_check.py`
Expected: prints the `single-segment check: ...` line, four `E0 = ... keV: ...` lines, the BS/Eq.(17) comparison line, the 25 keV bulk/film enhancement line, the transmitted-count line, `wrote zhai_fig1c_analog.png`, and a total-time line — no exceptions. Confirm `zhai_fig1c_analog.png` was written to the repo root, then delete it (it's a gitignored generated artifact, matching every other `*.png` in this repo).

- [ ] **Step 4: Diff the printed numbers against the notebook's last-known output**

Since the notebook is committed WITH stripped outputs (`nbstripout` on commit), there is no baseline output to diff against from git — instead, run the script twice (it's deterministic: every `simulate_trajectories` call passes an explicit `seed=`) and confirm the two runs print byte-identical numbers, confirming no accidental nondeterminism was introduced by the conversion.

Run: `uv run python checks/zhai_fig1c_check.py > /tmp_run1.txt 2>&1; uv run python checks/zhai_fig1c_check.py > /tmp_run2.txt 2>&1; diff /tmp_run1.txt /tmp_run2.txt`
Expected: no diff (the `Total time` line will legitimately differ — if `diff` only flags that line, this passes; if any physics number differs, investigate before proceeding).

- [ ] **Step 5: Verify lint/type-check**

Run: `uv run ruff check checks/zhai_fig1c_check.py && uv run pyright` (pyright's `include = ["src"]` per `pyproject.toml`, so this file isn't pyright-checked — ruff is the only gate)
Expected: 0 ruff errors.

- [ ] **Step 6: Commit**

```bash
git add checks/zhai_fig1c_check.py checks/zhai_fig1c_check.ipynb checks/zhai_fig1c_check.md
git commit -m "$(cat <<'EOF'
refactor(checks): retire zhai_fig1c_check.ipynb -> plain script

Converts to checks/zhai_fig1c_check.py, matching the plain-script
convention every other checks/ validation file already uses (this
notebook had no genuine interactivity -- it's a linear reproduction).
plt.show() -> fig.savefig("zhai_fig1c_analog.png"), matching the file's
own docstring claim. Verbatim physics -- verified by running the script
twice and confirming identical printed numbers (every MC call is
explicitly seeded).
EOF
)"
```

---

## Task 2: Retire `checks/zhai_fig1c_validation.ipynb` (no replacement needed)

**Files:**
- Delete: `checks/zhai_fig1c_validation.ipynb`, `checks/zhai_fig1c_validation.md`

**Interfaces:** none — this notebook's entire body (build a `ZhaiAnchor`, call `model_spectra`/`reference_curve`/`validation_table`/`figure_spectra`/`figure_flux_anchor`/`figure_enhancement`) is already fully duplicated by `checks/anchor_figures.py::main()` (`checks/anchor_figures.py:501-552`), which additionally saves all three figures to disk (`outdir="figures"` by default) instead of `plt.show()`-ing them inline. There is nothing in the notebook not already covered headlessly.

- [ ] **Step 1: Confirm `anchor_figures.main()` is a superset of the notebook's behavior**

Run: `uv run python -c "import sys; sys.path.insert(0, 'checks'); sys.path.insert(0, 'src'); import anchor_figures as af; af.main(ne=50, ne_brem=20)"`
Expected: prints the "Running MC model spectra..." line, the validation table (via `tabulate`), the enhancement line, and three `wrote figures/zhai_...png` lines — exercising every code path the notebook's cells did (anchor construction, `theory_line_energies`, `model_spectra`, `reference_curve`, `validation_table`, all three figure functions). Delete the generated `figures/` dir afterward (gitignored generated output, not part of this change).

- [ ] **Step 2: Delete the notebook and its jupytext pair**

```bash
git rm checks/zhai_fig1c_validation.ipynb checks/zhai_fig1c_validation.md
```

- [ ] **Step 3: Verify nothing else referenced this notebook**

Run: `grep -rn "zhai_fig1c_validation" --include="*.py" --include="*.md" --include="*.toml" --include="*.yaml" .` (exclude `.git`)
Expected: only doc mentions remain (README.md's `checks/` validation list, handled in Task 5) — no source-code import references it (it's a notebook, nothing `import`s it).

- [ ] **Step 4: Commit**

```bash
git add checks/zhai_fig1c_validation.ipynb checks/zhai_fig1c_validation.md
git commit -m "$(cat <<'EOF'
refactor(checks): retire zhai_fig1c_validation.ipynb (redundant with anchor_figures.main())

The notebook was a thin, cell-by-cell wrapper over anchor_figures.py's
ZhaiAnchor/model_spectra/validation_table/figure_* functions -- exactly
what anchor_figures.py::main() already does headlessly (prints the same
validation table, saves all three figures to figures/). No replacement
file needed; `uv run python checks/anchor_figures.py` (or `af.main()`)
is the direct equivalent. README's checks/ section updated in a later
commit of this same retirement (Task 5).
EOF
)"
```

---

## Task 3: Split `checks/cxr_analysis_feranchuk.ipynb` → `docs/pxr-cbr-derivation.md` + `checks/feranchuk_figures_check.py`

**Files:**
- Create: `docs/pxr-cbr-derivation.md`
- Create: `checks/feranchuk_figures_check.py`
- Delete: `checks/cxr_analysis_feranchuk.ipynb`, `checks/cxr_analysis_feranchuk.md`

**Interfaces:**
- Consumes: `cxr_mc.crystallography` (`CRYSTALS`, `reciprocal_g_vector`, `beta_from_Ee`, `absorption_length_ang`, `Z_TABLE`, `ALPHA_FS`, `HC_EV_ANG`), `cxr_mc.atomic_form_factors.atomic_form_factor`, `feranchuk_spence` (`omega_n`, `amplitudes_PXR_CBS_sweep`, `delta_g`, `cxr_to_bremsstrahlung`, `cxr_lines_fixed`) — all pre-existing, unchanged.
- Produces: `docs/pxr-cbr-derivation.md` is a documentation-only artifact (no code executes from it); `checks/feranchuk_figures_check.py` is a standalone script writing PNGs under `figures/`.

- [ ] **Step 1: Write `docs/pxr-cbr-derivation.md`**

Copy the markdown PROSE ONLY from `checks/cxr_analysis_feranchuk.md` — everything from `# Parametric X-ray Radiation (PXR)` (currently line 15) through the symbol table ending `...contributes <0.1% here) |` (currently line 343) — verbatim, dropping:
- the jupytext YAML front-matter (lines 1-13; this is a plain doc now, not a paired notebook),
- the `![{5AE...}.png](attachment:...)` broken attachment-image reference (line 21; it never resolved outside the original `.ipynb`'s embedded binary and has no committed file to point at — replace with a one-line note: `*(Coordinate-system figure: see the original notebook's embedded image, not carried over -- TODO redraw as a committed diagram if this derivation is revisited.)*`).

Add a one-line header (not present in the original) directly under the H1, before "The coordinate system...":

```markdown
# Parametric X-ray Radiation (PXR) and Coherent Bremsstrahlung Radiation (CBR)

*Design note, moved from `checks/cxr_analysis_feranchuk.ipynb` (retired --
see `checks/feranchuk_figures_check.py` for the paper-figure reproductions
this derivation supports).*

<!-- (rest of the derivation prose, verbatim from checks/cxr_analysis_feranchuk.md
     lines 19-343, with the attachment-image line replaced as noted above) -->
```

- [ ] **Step 2: Write `checks/feranchuk_figures_check.py`**

Copy the 8 code cells from `checks/cxr_analysis_feranchuk.md` (currently lines 345-1021) verbatim, in order, into the new file, with exactly these changes:
1. A new module docstring (below) replacing the bare first bootstrap cell's one-line comment.
2. Remove every `%matplotlib inline` line (2 occurrences, in cells 3 and unchanged elsewhere).
3. Add `import matplotlib; matplotlib.use("Agg")` right after the `sys.path` bootstrap, before any other matplotlib import, matching `checks/anchor_figures.py::main`'s headless-safety pattern (this script has multiple `for crystal in ...:` / `for Ee_overlay_eV in ...:` loops that each produce a figure — `plt.show()` would block once per iteration under a headless backend).
4. Replace every `plt.show()` with a descriptive `fig.savefig(...)` (or, for the two cells that reuse a bare `fig`/`ax` name across a loop without reassigning `fig` per iteration, capture the current figure via `plt.gcf()` before saving) into a `figures/` directory created if missing, matching `checks/anchor_figures.py`'s `outpath.mkdir(exist_ok=True)` convention. Concretely:
   - Cell "CXR Energy vs. Emission Angle" (one fig per crystal) → `fig.savefig(figures_dir / f"feranchuk_energy_vs_angle_{crystal}.png", dpi=150, bbox_inches="tight")`
   - Cell "PXR vs. CBS" 2x3 grid (one fig per crystal) → `fig.savefig(figures_dir / f"feranchuk_pxr_vs_cbs_{crystal}.png", dpi=150, bbox_inches="tight")`
   - Cell "cross-crystal overlay" (one fig per `Ee_overlay_eV`) → `fig.savefig(figures_dir / f"feranchuk_crosscrystal_{int(Ee_overlay_eV/1e3)}kev.png", dpi=150, bbox_inches="tight")`
   - Cell "FULL energy spectrum through the detector aperture" (one fig) → `fig.savefig(figures_dir / "feranchuk_aperture_spectrum.png", dpi=150, bbox_inches="tight")`
   - Cell "Replication of paper Fig. 2 (a),(b)" (one fig) → `fig.savefig(figures_dir / "feranchuk_paper_fig2_si.png", dpi=150, bbox_inches="tight")`

```python
"""
checks/feranchuk_figures_check.py

Paper-figure reproductions for the PXR/CBR derivation in
docs/pxr-cbr-derivation.md: CXR line energy vs. emission angle, PXR-vs-CBS
flux and polarization balance, a cross-crystal overlay, the full aperture
spectrum (coherent lines vs. bremsstrahlung background), and a replication
of Feranchuk-Spence's paper Fig. 2 (a),(b) for silicon along <111>/<100>.
See the module docstring in docs/pxr-cbr-derivation.md for the full
dispersion-relation derivation these figures are built on.

Run:  uv run python checks/feranchuk_figures_check.py   (from the repo root)
Writes PNGs to figures/feranchuk_*.png.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib

matplotlib.use("Agg")

# < ... copy the rest of cell 1 (checks/cxr_analysis_feranchuk.md lines 356-379,
#     the `reflections`/`crystals` dict build) verbatim ... >

# < ... copy cell 2 (lines 382-522: imports, plot_photon_energy,
#     calculate_photon_energy, flux_and_ratio_vs_Omega) verbatim, dropping
#     the "%matplotlib inline" line ... >

figures_dir = Path(__file__).resolve().parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)

# < ... copy cell 3 (lines 528-585: first per-crystal energy-vs-angle plot)
#     verbatim, replacing the implicit end-of-cell display with: >
    fig.savefig(figures_dir / f"feranchuk_energy_vs_angle_{crystal}.png", dpi=150, bbox_inches="tight")

# < ... copy cell 4 (lines 588-691: per-crystal 2x3 PXR-vs-CBS grid) verbatim,
#     replacing `plt.show()` with: >
    fig.savefig(figures_dir / f"feranchuk_pxr_vs_cbs_{crystal}.png", dpi=150, bbox_inches="tight")

# < ... copy cell 5 (lines 697-742: cross-crystal overlay) verbatim, replacing
#     `plt.show()` with: >
    fig.savefig(figures_dir / f"feranchuk_crosscrystal_{int(Ee_overlay_eV / 1e3)}kev.png", dpi=150, bbox_inches="tight")

# < ... copy cell 6 (lines 744-904: full aperture spectrum) verbatim, replacing
#     `plt.show()` with: >
fig.savefig(figures_dir / "feranchuk_aperture_spectrum.png", dpi=150, bbox_inches="tight")

# < ... copy cell 7 (lines 906-1021: paper Fig. 2 (a),(b) replication) verbatim,
#     replacing `plt.show()` with: >
fig.savefig(figures_dir / "feranchuk_paper_fig2_si.png", dpi=150, bbox_inches="tight")

print(f"wrote {len(list(figures_dir.glob('feranchuk_*.png')))} figures to {figures_dir}")
```

(Cell ordering note: the `.md`'s 8th `{code-cell}` block, lines 345-350, is just the `sys.path.insert(0, "src")` bootstrap — already folded into this file's header, not a separate copy step. That accounts for "8 code cells" total: bootstrap + 7 content cells enumerated above.)

- [ ] **Step 3: Run the new script**

Run: `uv run python checks/feranchuk_figures_check.py`
Expected: no exceptions; prints `wrote 8 figures to .../figures` (one energy-vs-angle + one PXR-vs-CBS per crystal in `crystals_to_plot` — 3 crystals in `CRYSTALS` by default (graphite, silicon, diamond) = 6, plus 3 cross-crystal overlays (one per `Ee_overlay_eV`), plus 1 aperture spectrum, plus 1 paper-fig2 = varies with how many crystals have basis data; don't hardcode the count in an assertion, just confirm the run completes and the dir is non-empty). Delete the generated `figures/` dir afterward (gitignored, not part of this change).

- [ ] **Step 4: Verify lint**

Run: `uv run ruff check checks/feranchuk_figures_check.py`
Expected: 0 errors. (`E501`/`E402` are already globally ignored per `pyproject.toml`, which matters here since this file's derivation-adjacent code carries the same long-line/import-after-comment style as its notebook origin.)

- [ ] **Step 5: Delete the notebook and its jupytext pair**

```bash
git rm checks/cxr_analysis_feranchuk.ipynb checks/cxr_analysis_feranchuk.md
```

- [ ] **Step 6: Commit**

```bash
git add docs/pxr-cbr-derivation.md checks/feranchuk_figures_check.py checks/cxr_analysis_feranchuk.ipynb checks/cxr_analysis_feranchuk.md
git commit -m "$(cat <<'EOF'
refactor(checks,docs): retire cxr_analysis_feranchuk.ipynb -> doc + script

The notebook was a ~250-line LaTeX PXR/CBR derivation followed by 7 code
cells reproducing paper figures. Split rather than force into checks/'s
plain-script convention wholesale: the derivation prose moves verbatim
into docs/pxr-cbr-derivation.md (a design note, matching CLAUDE.md's
"docs/ has the design notes"); the figure-reproduction code moves
verbatim into checks/feranchuk_figures_check.py (matching every other
checks/*.py validation script), with plt.show() -> fig.savefig(...) into
figures/ (headless-safe, mirrors anchor_figures.py::main's outdir
pattern) since the original's per-crystal/per-energy loops would block
repeatedly under a non-interactive backend. Verbatim physics -- verified
by running the new script and confirming it completes and writes the
expected figure set.
EOF
)"
```

---

## Task 4: Delete `notebooks/scan.ipynb` and `notebooks/analysis.ipynb` (marimo apps are the full replacement)

**Files:**
- Delete: `notebooks/scan.ipynb`, `notebooks/scan.md`, `notebooks/analysis.ipynb`, `notebooks/analysis.md`

**Interfaces:** none — `notebooks/scan_app.py` and `notebooks/analysis_app.py` already exist and were built specifically to replace these two notebooks (2026-06-27 marimo/Altair migration; see `CC-Session-Logs/2026-06-27_session-marimo-altair-migration-complete.md`). No new file needed.

- [ ] **Step 1: Confirm the marimo apps cover the same workflow**

Read `notebooks/scan_app.py` and `notebooks/analysis_app.py` side-by-side against `notebooks/scan.md`/`analysis.md` (already done during planning — `scan_app.py` runs `material_sweep`→`build_cases`→`run_sweep` with a live `geometry_table` display, `analysis_app.py` has a material/tilt dropdown driving `browse`-equivalent tabs for every figure the old notebook drew). No gaps found. If you want to re-confirm before deleting: `uv run python -c "import marimo"` succeeds and `uv run marimo edit notebooks/scan_app.py` opens without error (Ctrl-C out; this is a manual spot-check, not part of the automated verification below).

- [ ] **Step 2: Delete the notebooks and their jupytext pairs**

```bash
git rm notebooks/scan.ipynb notebooks/scan.md notebooks/analysis.ipynb notebooks/analysis.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(notebooks): retire scan.ipynb/analysis.ipynb (marimo apps supersede them)

notebooks/scan_app.py and notebooks/analysis_app.py (2026-06-27 marimo/
Altair migration) are already complete, verified replacements -- same
config.py-shared grids, same checkpoint contract, full figure coverage
via analysis_app.py's tabbed UI. No new file needed, pure deletion.
Doc/tooling touchpoints (README, repo_map, skills, dev.py, pre-commit,
pyproject) updated in later commits of this same retirement.
EOF
)"
```

---

## Task 5: Update documentation touchpoints

**Files:**
- Modify: `README.md`, `docs/repo_map.md`, `docs/running-on-a-cluster.md`, `.claude/skills/repo-orientation/SKILL.md`, `.agents/skills/repo-orientation/SKILL.md`, `.claude/skills/notebook/SKILL.md`, `.agents/skills/notebook/SKILL.md`, `scripts/export_pdf.py`

- [ ] **Step 1: `README.md` — repository layout block (currently lines 73-85)**

Replace:
```
notebooks/scan.ipynb      RUNNER:  pick MATERIAL → Sweep → run_sweep → checkpoints/<material>.pkl
notebooks/analysis.ipynb  VIZ:     load that checkpoint → all figures (no sweeps here)
scan.py            root shim → cxr_mc.scan (guarded; python scan.py, or cxr scan)
scripts/export_pdf.py  shim → cxr_mc.export (notebooks/analysis.ipynb → PDF, or cxr export)
```
with:
```
notebooks/scan_app.py      RUNNER (marimo):  pick MATERIAL → Sweep → run_sweep → checkpoints/<material>.pkl
notebooks/analysis_app.py  VIZ (marimo):     load that checkpoint → all figures (no sweeps here)
scan.py            root shim → cxr_mc.scan (guarded; python scan.py, or cxr scan)
scripts/export_pdf.py  shim → cxr_mc.export (checkpoint → PDF, or cxr export)
```
and just below (currently line 89-90), replace:
```
`*.png` images are gitignored; notebooks are output-stripped on commit by
`nbstripout` via `.gitattributes`.
```
with:
```
`*.png` images are gitignored. Notebooks are plain marimo `.py` apps (no
Jupyter `.ipynb` in this repo), so there's no output-stripping step.
```

- [ ] **Step 2: `README.md` — Installation section (currently lines 132-136)**

Replace:
```
Launch the notebooks with:

```bash
uv run jupyter lab
```
```
with:
```
Launch the marimo apps with:

```bash
uv run marimo edit notebooks/scan_app.py
uv run marimo edit notebooks/analysis_app.py
```
```

- [ ] **Step 3: `README.md` — Quickstart section (currently lines 155-166)**

Replace:
```
The workflow is **two notebooks that share the grids in `config.py`** — edit a
material's thickness / energies / tilts / energy-grids there once and both
notebooks pick it up.

1. **`notebooks/scan.ipynb`** (the runner): set `MATERIAL`, then
   `material_sweep(MATERIAL)` → `build_cases` → `run_sweep`, which writes
   `checkpoints/<material>.pkl` and streams the per-tilt statistics tables live.
2. **`notebooks/analysis.ipynb`** (the viz): set the same `MATERIAL`, `load_checkpoint`,
   `cases_from_results`, then `browse` / heatmaps / Eagle XO / Timepix /
   penetration figures. No sweeps run here.
```
with:
```
The workflow is **two marimo apps that share the grids in `config.py`** — edit
a material's thickness / energies / tilts / energy-grids there once and both
apps pick it up.

1. **`notebooks/scan_app.py`** (the runner): set `MATERIAL`, then
   `material_sweep(MATERIAL)` → `build_cases` → `run_sweep`, which writes
   `checkpoints/<material>.pkl` and streams the per-tilt statistics tables live.
2. **`notebooks/analysis_app.py`** (the viz): pick the same `MATERIAL` from its
   dropdown, then browse rankings / spectra / scans / detectors / penetration /
   cross-material tabs. No sweeps run here.
```

- [ ] **Step 4: `README.md` — Validation (`checks/`) section (currently lines 293-307)**

Replace:
```
- `zhai_fig1c_check.ipynb`, `cxr_analysis_feranchuk.ipynb` — figure reproductions.
- `anchor_figures.py` + `zhai_fig1c_validation.ipynb` — **model-vs-theory anchor
  figures**: the MC spectra against the Eq.(10) dispersion-relation line energies,
  the Eq.(12) closed-form flux (single-segment MC/closed ratio ≈ 1), and the
  bulk-vs-film enhancement. Drop a digitized Fig 1c curve in
  `checks/reference_data/zhai_fig1c.csv` (schema in that dir's README) and the
  spectra figure overlays the measured data automatically.
- `src/cxr_mc/_compile_nb.py` — compiles every notebook's code cells (syntax smoke test).
```
with:
```
- `zhai_fig1c_check.py`, `feranchuk_figures_check.py` — figure reproductions
  (`feranchuk_figures_check.py`'s derivation background lives in
  `docs/pxr-cbr-derivation.md`).
- `anchor_figures.py` (run directly, or via `af.main()`) — **model-vs-theory
  anchor figures**: the MC spectra against the Eq.(10) dispersion-relation line
  energies, the Eq.(12) closed-form flux (single-segment MC/closed ratio ≈ 1),
  and the bulk-vs-film enhancement. Drop a digitized Fig 1c curve in
  `checks/reference_data/zhai_fig1c.csv` (schema in that dir's README) and the
  spectra figure overlays the measured data automatically.
```
(the `_compile_nb.py` line is simply deleted — nothing survives it once every `.ipynb` is gone.)

- [ ] **Step 5: `docs/repo_map.md` — Entry points section (currently lines 30-42)**

Replace the `**Notebooks**` bullet:
```
- **Notebooks**: `notebooks/scan.ipynb` (sweep) → `notebooks/analysis.ipynb` (viz); both read the
  per-material grids in `config.py`.
```
with:
```
- **Notebooks**: `notebooks/scan_app.py` (sweep, marimo) → `notebooks/analysis_app.py`
  (viz, marimo); both read the per-material grids in `config.py`.
```

- [ ] **Step 6: `docs/repo_map.md` — `_compile_nb.py` entry (currently lines 202-204)**

Delete the `### _compile_nb.py` subsection entirely (the file no longer exists).

- [ ] **Step 7: `docs/running-on-a-cluster.md`**

Replace (currently line 84):
```
Then open `notebooks/analysis.ipynb` (set the same `MATERIAL`) or run `cxr export` locally —
```
with:
```
Then open `notebooks/analysis_app.py` (`uv run marimo edit notebooks/analysis_app.py`, pick the
same `MATERIAL`) or run `cxr export` locally —
```

- [ ] **Step 8: `.claude/skills/repo-orientation/SKILL.md` and `.agents/skills/repo-orientation/SKILL.md`**

In `.claude/skills/repo-orientation/SKILL.md`, replace (line 15):
```
- Treat `notebooks/scan.ipynb` as the sweep runner and `notebooks/analysis.ipynb` as the viz notebook.
```
with:
```
- Treat `notebooks/scan_app.py` as the sweep runner and `notebooks/analysis_app.py` as the viz app (both marimo).
```
In `.agents/skills/repo-orientation/SKILL.md`, apply the equivalent edit to its own line 13 (`Treat \`scan.ipynb\` as the sweep runner and \`analysis.ipynb\` as the viz notebook.` → `Treat \`scan_app.py\` as the sweep runner and \`analysis_app.py\` as the viz app (both marimo).`), preserving that file's already-diverged heading style (`## Ground Rules` etc.) — do not resync the two files' formatting beyond this one line, that's a pre-existing, out-of-scope divergence.

- [ ] **Step 9: `.claude/skills/notebook/SKILL.md` and `.agents/skills/notebook/SKILL.md`**

In `.claude/skills/notebook/SKILL.md`, replace the "Use this skill when" line:
```
Use this skill when changing `notebooks/scan.ipynb`, `notebooks/analysis.ipynb`, or any notebook under `checks/`.
```
with:
```
Use this skill when changing `notebooks/scan_app.py`, `notebooks/analysis_app.py` (marimo apps), or any script under `checks/` that reproduces a figure or validates model output.
```
and its "Canonical notebook commands" section — since `scripts/dev.py nbqa`/`nbstrip` are removed in Task 6, replace:
```
## Canonical notebook commands

- `uv run python scripts/dev.py nbqa`
- `uv run python scripts/dev.py nbstrip`
- `uv run python scripts/dev.py verify`
```
with:
```
## Canonical commands

- `uv run ruff check .`
- `uv run python scripts/dev.py verify`
```
Apply the equivalent content edit to `.agents/skills/notebook/SKILL.md`, preserving its own already-diverged header/description text (only change the substantive "use when" sentence and the command list, matching Step 8's precedent of not resyncing pre-existing formatting divergence).

- [ ] **Step 10: `scripts/export_pdf.py`**

Replace its module docstring:
```python
"""analysis.ipynb -> results/<material>_cxr_<date>.pdf -- thin shim to cxr_mc.export.

Kept in scripts/ for muscle-memory ``python scripts/export_pdf.py [stem]``; prefer
the installed CLI ``cxr export [stem]``. The real logic lives in cxr_mc/export.py.
"""
```
with:
```python
"""checkpoint -> results/<material>_cxr_<date>.pdf -- thin shim to cxr_mc.export.

Kept in scripts/ for muscle-memory ``python scripts/export_pdf.py [stem]``; prefer
the installed CLI ``cxr export [stem]``. The real logic lives in cxr_mc/export.py.
"""
```
(Check `src/cxr_mc/export.py` and `src/cxr_mc/cli.py` too, per the earlier grep match — read them before editing; if their docstrings/help text mention `analysis.ipynb` specifically, apply the same "checkpoint" rewording there. If they only mention "a checkpoint" generically already, no change needed.)

- [ ] **Step 11: Grep-sweep for anything missed**

Run: `grep -rn "scan\.ipynb\|analysis\.ipynb\|cxr_analysis_feranchuk\.ipynb\|zhai_fig1c_check\.ipynb\|zhai_fig1c_validation\.ipynb" --include="*.py" --include="*.md" --include="*.toml" --include="*.yaml" .` (from repo root, excluding `.git`)
Expected: no results (everything above was the exhaustive list from the earlier repo-wide grep during planning).

- [ ] **Step 12: Full verification**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: 0 ruff, 0 pyright, 201/201 (docs/skill-only changes, no test impact).

- [ ] **Step 13: Commit**

```bash
git add README.md docs/repo_map.md docs/running-on-a-cluster.md .claude/skills/repo-orientation/SKILL.md .agents/skills/repo-orientation/SKILL.md .claude/skills/notebook/SKILL.md .agents/skills/notebook/SKILL.md scripts/export_pdf.py
git commit -m "$(cat <<'EOF'
docs: point every notebook reference at the marimo apps / plain scripts

README, repo_map, running-on-a-cluster, and both repo-orientation/notebook
skill mirrors (.claude + .agents) updated to reference scan_app.py/
analysis_app.py and the checks/*.py scripts instead of the retired
.ipynb files. _compile_nb.py's repo_map entry removed (file deleted in
an earlier commit of this retirement).
EOF
)"
```

---

## Task 6: Remove dead Jupyter-notebook tooling

**Files:**
- Delete: `src/cxr_mc/_compile_nb.py`
- Modify: `scripts/dev.py`, `.pre-commit-config.yaml`, `pyproject.toml`, `.gitattributes`

**Interfaces:** none removed that anything else calls — confirmed via Task 5 Step 11's grep sweep plus the earlier research grep showing `_compile_nb.py` is referenced only in prose docs (already fixed in Task 5), never imported or invoked from CI/`dev.py`.

- [ ] **Step 1: Delete `_compile_nb.py`**

```bash
git rm src/cxr_mc/_compile_nb.py
```

- [ ] **Step 2: Trim `scripts/dev.py`**

Remove `iter_notebooks` (currently lines 41-48), `cmd_nbqa` (lines 103-109), `cmd_nbstrip` (lines 111-117), the `NOTEBOOK_SKIP_PARTS` constant (lines 27-34, now unused), and their two `sub.add_parser` registrations (in `build_parser`, currently the `("nbqa", cmd_nbqa)` and `("nbstrip", cmd_nbstrip)` tuple entries). Also drop the two matching lines from `cmd_repo_map`'s printed command list (currently lines 89-90: `"uv run python scripts/dev.py nbqa"`, `"uv run python scripts/dev.py nbstrip"`) and update the module docstring's `Commands:` list (currently lines 9-16) to drop the `nbqa`/`nbstrip` lines.

- [ ] **Step 3: Run `dev.py`'s own commands to confirm it still works**

Run: `uv run python scripts/dev.py repo-map && uv run python scripts/dev.py lint && uv run python scripts/dev.py test`
Expected: all three run clean (same as before — `lint`/`test` never touched notebooks anyway; `repo-map` just prints a shorter command list now).

- [ ] **Step 4: Trim `.pre-commit-config.yaml`**

Remove the two hook blocks:
```yaml
  # Jupyter notebook linting via nbqa
  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.9.1  # pin a stable version
    hooks:
      - id: nbqa-ruff
        args: [--fix, "--extend-ignore=F821"]  # F821 is always a false positive across notebook cells

  # Strip notebook outputs
  - repo: https://github.com/kynan/nbstripout
    rev: 0.9.1
    hooks:
      - id: nbstripout
```
leaving only the Ruff hook block.

- [ ] **Step 5: Trim `pyproject.toml`**

Remove from `dependencies` (7 entries): `"jupyter-core>=5.9.1"`, `"jupyter-mcp-server>=1.0.2"`, `"jupyterlab==4.4.1"`, `"jupytext>=1.19.4"`, `"nbconvert>=7.17.1"`, `"nbqa>=1.9.1"`, `"nbstripout>=0.9.1"`. Leave `ipykernel`, `ipympl` (still used — see Global Constraints). Remove the `[tool.jupytext]` section (`formats = "ipynb,md:myst"`). Remove the `"*.ipynb" = ["F821"]` line from `[tool.ruff.lint.per-file-ignores]` (leave the `"notebooks/*_app.py" = ["B018"]` line — marimo apps are unaffected by this retirement).

- [ ] **Step 6: Regenerate the lockfile and verify nothing breaks**

Run: `uv sync`
Expected: succeeds, `uv.lock` updates (fewer packages). If `uv sync` or the next step fails because something unexpectedly depended on a removed package, re-add ONLY that specific package back to `dependencies` rather than reverting the whole trim.

Run: `uv run ruff check . && uv run pyright && uv run pytest -q`
Expected: 0 ruff, 0 pyright, 201/201.

- [ ] **Step 7: Trim `.gitattributes`**

Remove the two lines:
```
*.ipynb filter=nbstripout
*.ipynb diff=ipynb
```

- [ ] **Step 8: Commit**

```bash
git add -- src/cxr_mc/_compile_nb.py scripts/dev.py .pre-commit-config.yaml pyproject.toml uv.lock .gitattributes
git commit -m "$(cat <<'EOF'
chore: remove dead Jupyter-notebook tooling

No .ipynb remain in the repo after this retirement (see the preceding
commits), so _compile_nb.py, dev.py's nbqa/nbstrip commands, the
nbQA/nbstripout pre-commit hooks, the jupyter-core/jupyter-mcp-server/
jupyterlab/jupytext/nbconvert/nbqa/nbstripout pyproject dependencies,
[tool.jupytext], the "*.ipynb" ruff per-file-ignore, and the .gitattributes
ipynb filters are all confirmed-dead. ipykernel/ipympl are kept -- still
reachable via IPython.display, used directly by the marimo apps and by
src/cxr_mc/results/tables.py, results/scoring.py, plots/interactive.py.
Verified via `uv sync` (lockfile regenerates clean) + full suite.
EOF
)"
```

Note: `git add --` with a leading `--` avoids ambiguity with the previous `git rm src/cxr_mc/_compile_nb.py` in Step 1 already having staged that deletion — a plain `git status` before this commit should show exactly the files listed above as staged, nothing else (in particular, `analysis.py`'s pre-existing unstaged deletion must still show as unstaged).

---

## Task 7: Update `TODO.md` / `docs/dedup-inventory.md` and final full-repo verification

**Files:**
- Modify: `TODO.md`, `docs/dedup-inventory.md`

- [ ] **Step 1: Add a short paragraph to `TODO.md`**

This retirement wasn't one of the original 7 duplication clusters `dedup-inventory.md` tracks — don't invent a fake `M*` id for it. Add a new paragraph after the M6 summary (or after whatever M6's final wording ends up as, if the M6 plan ran first), in the same "Also done (this branch)" style:

```
**Also done** (this branch): retired every Jupyter notebook in the repo.
`notebooks/scan.ipynb`/`analysis.ipynb` were fully superseded by the
already-existing marimo apps `scan_app.py`/`analysis_app.py` (deleted, no
replacement needed). `checks/zhai_fig1c_check.ipynb` and
`checks/cxr_analysis_feranchuk.ipynb` converted to plain scripts
(`checks/zhai_fig1c_check.py`, `checks/feranchuk_figures_check.py` +
`docs/pxr-cbr-derivation.md` for the derivation prose), matching the
plain-script convention every other `checks/` validation file already
uses. `checks/zhai_fig1c_validation.ipynb` needed no replacement --
`checks/anchor_figures.py::main()` was already a complete headless
equivalent. Dead tooling removed alongside: `_compile_nb.py`, `dev.py`'s
`nbqa`/`nbstrip` commands, the nbQA/nbstripout pre-commit hooks, and the
Jupyter-only pyproject dependencies (`jupyter-core`, `jupyter-mcp-server`,
`jupyterlab`, `jupytext`, `nbconvert`, `nbqa`, `nbstripout`).
```

- [ ] **Step 2: `docs/dedup-inventory.md` — no edit needed**

This file tracks the 7 original duplication clusters (M1-M7) plus the HIGH/LOW findings from the `/sc:analyze` inventory; the notebook retirement is out of that inventory's scope by design (confirmed in the plan's Global Constraints — no ledger or dedup-inventory row references any of the retired notebooks). Leave it untouched; do not add a fabricated finding for symmetry.

- [ ] **Step 3: Final full-repo verification**

Run: `uv run ruff check . && uv run pyright && uv run pytest -q && git status`
Expected: 0 ruff, 0 pyright, 201/201 tests; `git status` shows a clean tree except the pre-existing unstaged `D analysis.py` (untouched throughout this entire plan) and (if this commit hasn't happened yet) the staged `TODO.md` change from Step 1.

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(TODO): record the Jupyter-notebook retirement

Not one of the original 7 dedup-inventory.md clusters -- tracked here
only, as a standalone "also done (this branch)" paragraph.
EOF
)"
```

---

## Self-Review Notes (for the executor)

- **Spec coverage:** both scoping answers from the user are covered — "retire both, but convert the checks as well first" (Tasks 1-4: checks/ converted-or-confirmed-redundant BEFORE deletion, notebooks/ deleted directly since its replacement already existed) and "Delete outright" (every `git rm`, no archive directory). The Feranchuk derivation's "split: docs/ note + checks/ script" answer is Task 3.
- **Risk containment:** the one meaningfully risky step (pyproject dependency trim + `uv sync`, Task 6 Step 5-6) is isolated to its own task with its own lockfile-regeneration verification gate, separable from the (zero-risk) file deletions and (low-risk) doc edits — if the lockfile trim causes an unexpected break, only Task 6 needs to be revisited, not the whole plan.
- **`ipykernel`/`ipympl`:** deliberately NOT removed — flagged as a conservative choice in Global Constraints and Task 6's commit message, not a silent gap.
