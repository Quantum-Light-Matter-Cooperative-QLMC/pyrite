# feature/energy-grid-semantics — issue #98

Pin energy-grid semantics; audit uniform-only consumers. Blocks #99 and #100.
Parent epic #97. Source: `docs/research/beam-transport/energy-grid-recommendations.md`.

## Problem

Grid semantics (evaluation nodes vs bin edges, densities in photons/eV,
integration in physical energy with local widths) are stated only in a research
note. Multiple consumers silently assume uniform spacing: they read one `dE`,
key identity on size+endpoints, or rebin with a single input width. Nothing
fails loudly when a nonuniform grid arrives.

## Scope

Audit + documentation + guards/tests only. No default flips, no LUT change, no
continuum change. Existing uniform-grid results stay bit-for-bit identical.

## Checklist

- [x] Maintained-doc page for grid semantics (nodes vs edges, photons/eV
      density, physical-energy integration with local widths, log-energy
      Jacobian, ban on interchanging node-centered trapezoid integrals with
      bin-integrated line masses). Place under `docs/physics/` or
      `docs/repo-design/` per `docs/repo-design/documentation.md`.
- [x] Per-owner audit record (nonuniform-safe | guarded) for:
      `montecarlo/spectrum/lines/_per_hkl.py`,
      `montecarlo/spectrum/lines/_kernels.py::_line_tabulation_grid`,
      `detectors/_si_sensor.py::poisson_core`, `::grid_key`,
      `detectors/timepix_response.py`, `detectors/response.py`,
      `detectors/eaglexo_response.py`, `results/metrics.py`,
      `energy_grid/derive.py`, `bounds.py`, `apply.py`.
- [x] Same audit over CUDA and CPU-fallback routes.
- [x] Regression tests that fail loudly (or assert an explicit unsupported
      error) when a nonuniform grid reaches a uniform-only consumer.
- [x] Identify hard diagnostic ceilings that do not expand with requested range.

## Decisions

- Audit-and-guard only; behavior changes for uniform grids are out of scope.
- `_catalog_decode.py` accepting `logspace` and `_energy_grid_encoding.py`
  preserving nonuniform arrays do not establish end-to-end support.

## Landed

- Doc: `docs/physics/radiation-physics/energy-grid-semantics.md`, added to the
  `docs/physics/index.md` toctree and cross-linked from
  `radiation-physics/spectral-observables.md`.
- Guard helper: `src/pyrite/_grid_semantics.py` (package-root leaf, so no
  `detectors -> energy_grid` edge; `pyrite-dev repo-map` confirmed the layer
  graph is unchanged) with the facade `src/pyrite/energy_grid/semantics.py`.
- Audit record: `audit.md` in this directory.
- Tests: `tests/energy-grid/test_grid_semantics.py` and one case added to
  `tests/montecarlo/test_spectrum_phases.py`.
