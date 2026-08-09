Verify ledger claim `coherent-line-hkl-batch` from current `main`. The
implementation branch has landed and was deleted; do not recreate it unless a
review finding requires a tracked fix. Follow `docs/validation/README.md`.

Claim row: docs/physics-validation-ledger.md, "Core coherent physics" section, id `coherent-line-hkl-batch`, status `filtered`.
Code: src/cxr_mc/montecarlo/spectrum.py::mc_spectrum, the batched (n_seg, N_g) branch, steps 5c/6c/7c (search for "Validation: coherent-line-hkl-batch" and "-- 5c." / "-- 7c.").

This is an EVALUATION-ORDER claim, not a new equation. What you must independently check:
1. The complex amplitude built in step 5c on the (n_block, N_g) grid is term-for-term the same expression the per-hkl coherent path builds in the `_accumulate` loop (the legacy path, reached when groove/layers/sinc_cutoff are set). Compare the two expression trees directly.
2. `|Sum_j|^2` is taken PER (reflection, mosaic orientation) row in step 7c, so reflections and mosaic orientations remain INCOHERENT while the segment sum inside a row keeps its phase. Confirm no g-mixing occurs before the square. Pay attention to the g-major flatten/compaction (`gm`, `coh_counts`, `starts`) — confirm each row's slice really contains only that row's segments.
3. The phase, escape/absorption factor, sinc width, and mosaic weight applied in 7c match the legacy path's.
4. Limiting cases: single segment coherent == incoherent self-term; N^2 in-phase build-up; on-resonance phase cancellation. tests/montecarlo/test_coherent_emission.py.

Do NOT modify the implementation. Do not mark the row signed-off (human only). Report: whether the claim holds, any discrepancy, and the status you think it warrants.

Use `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test <paths>`; never bare
pytest. Reproduce and report any unrelated failure rather than relying on this
dated handoff's historical failure list.
