# TODO: crystallography adapters — merge diffpy + dans-diffraction (TODO P2 #1)

Implements the `docs/crystallography-adapters-review.md` MERGE-AFTER-FIXES verdict for
`origin/feature/diffpy` (`67f27e0`, diffpy.structure CIF importer) and
`origin/feature/dans-diffraction` (`d4dbeff`, optional Dans_Diffraction validation oracle).

Scope for this branch:

1. Merge both branches (dans-diffraction already contains diffpy's commits), resolving the
   three trivial conflicts the review predicted: `TODO.md`, the validation-ledger progress
   summary line, and the `tests/test_crystallography.py` append point.
2. Drop the dangling reference in `claudedocs/research_dans_diffraction_*.md` to
   `claudedocs/research_crystals_alternatives_20260702.md`, which was never committed.
3. Actually install `diffpy-structure` (via a worktree-local `uv run`, per
   `docs/repo_map.md`'s shared-venv warning — do not `uv sync` the shared `C:/dev/cxr-mc/.venv`
   from here) and confirm the two diffpy-requiring tests pass for real before the ledger row
   is called `anchored`.
4. Leave the `crystals`-package redundancy question (symmetry-expansion gap) open per the
   review — out of scope for this merge.
