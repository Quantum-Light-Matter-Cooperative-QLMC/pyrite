Verify ledger claim `coherent-line-hkl-batch` in the worktree /home/alexa/dev/cxr-mc/worktrees/coherent-line-batching (branch feature/coherent-line-batching). Follow docs/validation/README.md.

Claim row: docs/physics-validation-ledger.md, "Core coherent physics" section, id `coherent-line-hkl-batch`, status `filtered`.
Code: src/cxr_mc/montecarlo/spectrum.py::mc_spectrum, the batched (n_seg, N_g) branch, steps 5c/6c/7c (search for "Validation: coherent-line-hkl-batch" and "-- 5c." / "-- 7c.").

This is an EVALUATION-ORDER claim, not a new equation. What you must independently check:
1. The complex amplitude built in step 5c on the (n_block, N_g) grid is term-for-term the same expression the per-hkl coherent path builds in the `_accumulate` loop (the legacy path, reached when groove/layers/sinc_cutoff are set). Compare the two expression trees directly.
2. `|Sum_j|^2` is taken PER (reflection, mosaic orientation) row in step 7c, so reflections and mosaic orientations remain INCOHERENT while the segment sum inside a row keeps its phase. Confirm no g-mixing occurs before the square. Pay attention to the g-major flatten/compaction (`gm`, `coh_counts`, `starts`) — confirm each row's slice really contains only that row's segments.
3. The phase, escape/absorption factor, sinc width, and mosaic weight applied in 7c match the legacy path's.
4. Limiting cases: single segment coherent == incoherent self-term; N^2 in-phase build-up; on-resonance phase cancellation. tests/montecarlo/test_coherent_emission.py.

Do NOT modify the implementation. Do not mark the row signed-off (human only). Report: whether the claim holds, any discrepancy, and the status you think it warrants.

Env note: use `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test <paths>` from the worktree; never bare pytest. tests/notebooks and tests/plots have pre-existing failures; ignore. tests/montecarlo/test_imports.py (3) and test_spectrum_escape_helpers.py::test_finite_side_exit_layered_absorption_stays_in_emission_layer also fail pre-existing in this worktree's venv; ignore.