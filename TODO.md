# TODO — refactor/dedup-m4-m7

This branch carries one backlog item; the full triaged backlog lives on `main`.

## De-duplication follow-through — M4 + M7 (P2)

`refactor/dedup-followthrough` is merged; it deliberately parked two clusters, tracked in
[`docs/dedup-inventory.md`](docs/dedup-inventory.md). Close them here.

**M4 — wide-brem overlay physics (x4).** `inc_b = brem_wide * scale; det_b = inc_b * qe(...)`
(plus the `* Eb / W_EHP_EV` charge variant) is repeated across `_draw_eaglexo_detected`,
`_draw_eaglexo_charge`, `eaglexo_detected_frame`, and `eaglexo_charge_frame`. Extract
`_eag_wide_brem` / `_eag_wide_charge` next to the existing `_eag_detected`.

**M7a — `line_fwhm_eV(case, E_pk, mosaic_rad)`.** The EDS² + aperture² + capped-mosaic
quadrature is duplicated between `results/store.py::store_result` and
`plots/spectra.py::plot_mosaic_comparison`.

**M7b — escape helpers.** The shared `n_hat`-default and `L_esc` logic between `mc_spectrum`
and `mc_brem_spectrum`.

**Constraints.** Anything under `montecarlo/` is **verbatim moves only** — a behavior change
there triggers the full physics-validation ceremony (derivation docstring, `Validation:` marker,
ledger row, limiting-case test). The export-freeze tests (`test_plots_exports.py`,
`test_montecarlo_exports.py`, `test_results_exports.py`) assert
`set(pkg.__all__) == FROZEN_EXPORTS`: adding names is fine, removing or failing to re-export is
not. Note `results/store.py` is owned by this branch — `feature/material-filters` is instructed
not to touch it.
