# TODO: crystallography adapters — merge diffpy + dans-diffraction (TODO P2 #1)

Implemented the `docs/crystallography-adapters-review.md` MERGE-AFTER-FIXES verdict for
`origin/feature/diffpy` (`67f27e0`, diffpy.structure CIF importer) and
`origin/feature/dans-diffraction` (`d4dbeff`, optional Dans_Diffraction validation oracle;
already contained diffpy's commits, so a single merge brought in both).

Done:

1. Merged, resolving the three trivial conflicts the review predicted: `TODO.md` (this file),
   the validation-ledger progress summary line (recomputed to **0/28 signed-off · 5 rederived
   · 1 anchored · 1 filtered · 1 blocked** — main's own pre-merge line was already stale at
   0/24 against 26 actual rows, now correct at 28), and the `tests/test_crystallography.py`
   append point (kept both blocks).
2. Dropped the dangling reference in `claudedocs/research_dans_diffraction_20260702_0956.md`
   to `claudedocs/research_crystals_alternatives_20260702.md`, which was never committed to
   this repo.
3. Installed `diffpy-structure` for real via a worktree-local `uv run` (builds its own isolated
   venv per this worktree's `pyproject.toml`, per the shared-venv trap noted in
   `docs/repo_map.md` — did not `uv sync` the shared `C:/dev/cxr-mc/.venv`) and confirmed all
   3 diffpy-requiring tests genuinely pass (not skipped): `pytest -k diffpy` → 3 passed. Full
   suite: 22/22 in the two adapter test files, whole repo green. `ruff check` and `pyright`
   both clean on the merged tree.
4. Left the `crystals`-package redundancy question open per the review (symmetry-expansion
   gap: does `diffpy.structure`'s CIF parser expand space-group + Wyckoff CIFs to full P1, or
   does that still need `crystals`/spglib?) — out of scope for this merge, needs the
   ~1-hour empirical spike the review recommends before scoping any `crystals` integration.

Next: land this on `main` (whoever merges should run the shared venv's `uv sync` once,
deliberately and not concurrently with other agents, so `diffpy-structure` installs there too).
