# Uniform-only consumer audit — issue #98

Branch `feature/energy-grid-semantics`. Audit + docs + guards only; no default
flips, no LUT change, no continuum change. Line numbers are post-change unless
marked "(pre)".

Guard helper: `src/pyrite/_grid_semantics.py` (package-root leaf, facade at
`src/pyrite/energy_grid/semantics.py`). It exposes
`require_uniform_grid(E, consumer=..., remedy=...)`, which returns exactly
`float(E[1] - E[0])` for a uniform grid — the value every guarded caller already
used, so uniform results are bit-for-bit unchanged — and raises
`NonuniformEnergyGridError` otherwise. It also exposes `grid_identity`, a cache
key over the complete node coordinates.

`spacing_spread` credits the spread `16` ulp of the largest node in the input's
own floating dtype before dividing by the median spacing. Without that floor the
real fine line-window grid `linspace(1599.0, 1601.5, 100_001)` (step 2.5e-5 eV at
1600 eV ≈ 1e8 float64 ulp) reports ~9e-9 relative jitter and a naive 1e-9
tolerance rejects it — observed as a failure of
`tests/montecarlo/test_xray_dispersion.py::test_interference_phase_matches_the_in_medium_closed_form[per_hkl]`
during development, now pinned by
`tests/energy-grid/test_grid_semantics.py::test_fine_step_at_high_energy_stays_uniform`.

## Verdicts

| Owner | Evidence | Assumption | Verdict | Guard / test |
| --- | --- | --- | --- | --- |
| `montecarlo/spectrum/lines/_per_hkl.py` (coherent sinc window) | `_per_hkl.py:287,298-301` | `dE = E_grid[1] - E_grid[0]`, then `i0/i1 = (E - E_grid[0]) // dE` — an energy half-width converted to a node index by one global spacing | **guarded** | guard hoisted to `_setup.py:243-258`, gated on `request.sinc_cutoff is not None`; `tests/montecarlo/test_spectrum_phases.py::test_prepare_refuses_sinc_windowing_on_a_nonuniform_grid` |
| `montecarlo/spectrum/lines/_per_hkl.py` (incoherent sinc² window) | `_per_hkl.py:593,604-607` | identical index arithmetic on the incoherent route | **guarded** | same guard (one chokepoint covers both routes); same test |
| `montecarlo/spectrum/lines/_per_hkl.py` (unwindowed routes) | `_per_hkl.py:273-285,583-591`, `_kernels.py:51-63` | none — `sinc(a_width*(E_grid - E_r)/pi)` evaluates the profile at the nodes themselves | **nonuniform-safe** | positive assertion in the same test (log grid with `sinc_cutoff=None` prepares fine) |
| `montecarlo/spectrum/lines/_kernels.py::_line_tabulation_grid` | `_kernels.py:262-278` | builds `np.arange(lo, hi+1.0, 1.0)` unioned with native Henke nodes; the *result* is already nonuniform and is consumed by `_interp_index` (`_kernels.py:189-212`, `searchsorted`) | **nonuniform-safe** (correctness) | none needed; the defect is a scaling ceiling, recorded below |
| `detectors/_si_sensor.py::poisson_core` | `_si_sensor.py:57-76` (pre: `dE = E[1] - E[0]` at `:58`) | expected counts `rate * dE * live_time` spend the first spacing on every bin | **guarded** | `require_uniform_grid` at `:66`; `test_grid_semantics.py::test_poisson_core_refuses_a_log_grid` + `::test_poisson_core_uniform_expected_counts_unchanged` |
| `detectors/_si_sensor.py::grid_key` | `_si_sensor.py:27-39` (pre: `return (E.size, round(E[0],6), round(E[-1],6))`) | **cache-correctness bug**: (size, first, last) is not a grid identity — a linear and a log grid over the same interval with the same node count collide, and the second caller silently receives the first grid's response matrix | **fixed** (was unsafe-unguarded) | now `grid_identity` (blake2b over the exact float64 bytes); `test_grid_semantics.py::test_grid_identity_separates_grids_that_share_size_and_endpoints` asserts the legacy key *did* collide and the new one does not, plus `::test_grid_key_is_stable_for_the_same_coordinates` |
| `detectors/timepix_response.py::TimepixResponse.__init__/.apply` | `timepix_response.py:349-358` (pre: `self.dE_fine = float(E[1]-E[0])`), used at `:419-422` | `bincount(idx_in, weights=spec * dE_fine)` spends one input-bin width on every fine sample; `idx_in` (`:367`) and the coarse input edges (`:362`) are likewise built from one constant `dE_mc` offset by half a fine width | **guarded** | `require_uniform_grid` at `:349`; `test_grid_semantics.py::test_timepix_response_refuses_a_log_grid` |
| `detectors/response.py::convolve_detector` | `response.py:91-105` (pre: `dE = E_grid_eV[1]-E_grid_eV[0]` at `:87`) | `gaussian_filter1d` works in **sample** units, so `sigma_bins = fwhm/dE` is a fixed energy width only on a uniform grid | **guarded** | `require_uniform_grid` at `:91`; `test_grid_semantics.py::test_convolve_detector_refuses_a_log_grid` + `::test_convolve_detector_still_runs_on_a_uniform_grid` |
| `detectors/eaglexo_response.py` (energy resolution) | `eaglexo_response.py:322-326` (`apply`) — `convolve_detector(self.E, det, median(energy_fwhm_eV(...)))` | inherits the sample-unit Gaussian; docstring at `:286-288` already claimed "resolve_energy=True requires uniform spacing" but nothing enforced it | **guarded** (indirectly, through `convolve_detector`) | covered by the `convolve_detector` tests; the EagleXO `resolve_energy=True` path raises through it |
| `detectors/eaglexo_response.py` (QE-only path) | `eaglexo_response.py:217-243` (`qe`, log-energy `np.interp`), `:322-323` (`apply` is a per-bin multiply) | none — pointwise in `E` | **nonuniform-safe** | none |
| `detectors/eaglexo_response.py::get_response` cache | `eaglexo_response.py:391` | keys on `grid_key(E) + (...)` | **fixed** via `grid_identity` | same grid-identity test |
| `detectors/eaglexo_response.py::integrated_charge` | `eaglexo_response.py:363` | `np.trapezoid(charge_density, self.E)` — passes the coordinate | **nonuniform-safe** | none |
| `results/metrics.py::line_metrics` | `metrics.py:191-200` (post-#110: `_sample_energy`/`_integrate_energy_window` interpolate on `E` directly) | superseded — #110 replaced the sample-space `w_samp * dE` conversion with physical-energy interpolation; `peak_widths`' fractional sample crossings are mapped through `E` via `np.interp`, so the FWHM window and its integral are exact on any grid | **nonuniform-safe** (was guarded) | `test_grid_semantics.py::test_line_metrics_is_correct_on_a_log_grid` |
| `results/metrics.py` integrals | `metrics.py:172-175` | `np.trapezoid(spec[lo:hi], E[lo:hi])` — passes the coordinate | **nonuniform-safe** | — |
| `energy_grid/derive.py` | `derive.py:50-66,707-708` | diagnostic grids are `np.arange` only; `WIDE_BREM_EV` starts at `0.0` (a log grid cannot contain zero); ceilings are constants | **unsafe-unguarded, by design** — descriptor/policy surface, not a runtime consumer | none (scope: no default flips). Ceilings recorded below |
| `energy_grid/bounds.py::coverage_energy` | `bounds.py:55` | `np.diff(E_grid) * (spec[:-1]+spec[1:]) / 2` — local widths; the physical-energy trapezoid | **nonuniform-safe** | none |
| `energy_grid/bounds.py::spacing_num` | `bounds.py:122-126` | emits `num` for a `linspace` at a target *uniform* spacing — linear descriptor only | **unsafe-unguarded, by design** (policy surface) | none |
| `energy_grid/apply.py` | `apply.py:95-113` (`_line_item` writes `grid.linspace`), `:123-136` (`_existing_line_rows` reads only `["grid"]["linspace"]`), `:204-216` (`_merge_brem` writes `arange`), `:242-250` (`effective_brem` returns `brem.get("arange")`), `:877,929` | catalog round-trip is **linear-descriptor only**: a `logspace` or `values` row decodes (`materials/_catalog_decode.py:14`) but `apply`/`show` would `KeyError` on it | **unsafe-unguarded, by design** (no nonuniform row can be written today, so nothing reaches it) | none; recorded as the #101 blocker |

### CUDA and fallback routes

| Route | Evidence | Finding |
| --- | --- | --- |
| Fused CUDA line reduction | `_batched.py:963-970` | gated `sinc_cutoff is None`; `line_jit_kernel.py:35-46,98-118,193-221,312-346` gather `E_grid[k]` by index — node evaluation, **nonuniform-safe**, and unreachable with windowing |
| Coherent CUDA stream | `_batched.py:990-997`, `coherent_stream_jit_kernel.py:338-361,442-474,594-627` | same: gated `sinc_cutoff is None`, indexes `E_grid` directly — **nonuniform-safe** |
| CuPy/NumPy batched fallback | `_batched.py` steps 1-6 | evaluates at nodes; no `E[1]-E[0]` anywhere in the file — **nonuniform-safe** |
| Per-hkl CPU/GPU compatibility route | `_per_hkl.py:287,593` | the only spacing-dependent line code, on both CPU and GPU (`_to_cpu(dE)` is taken on either backend) — **guarded** at the shared `_setup` chokepoint, so CPU and CUDA fail identically |
| Brem kernels | `brem.py:863-875,1047-1064` | `mu` and `dsig` evaluated at `E_grid` nodes; `brem_jit_kernel` takes the array — **nonuniform-safe** |
| Characteristic kernel | `characteristic.py:532-542`, `:545-...` | `_energy_bin_edges_and_widths` builds midpoint edges with one-sided outer half-widths and returns `np.diff(edges)`; `_lorentzian_bin_weights` integrates the profile analytically per bin — **nonuniform-safe**, and the reference implementation for {eq}`eq-grid-midpoint-edges` |
| `diagnostics.py` brem quadrature | `diagnostics.py:374-377` | builds full local trapezoid weights `0.5*(E[2:]-E[:-2])` with one-sided ends — **nonuniform-safe** |
| Grating | `detectors/grating.py:356-358` | cumulative trapezoid against `np.diff(x)`, then differences at pixel edges — conservative rebinning, **nonuniform-safe** |
| Backend precision floor | `_setup.py:221-241` | pre-existing guard: refuses a grid whose nodes collapse after the cast to `REAL`. Complements the uniformity guard; a six-decade float32 grid would hit this first |

## Hard diagnostic ceilings (recorded, not changed)

None of these expand with the requested energy range.

| Ceiling | Evidence | Behavior |
| --- | --- | --- |
| Line coverage grid stop, 20 keV | `derive.py:54-57` (`WIDE_GRID_STOP_EV`), overridable only via `--grid-stop` (`derive.py:636-642`) | fixed constant; the comment records that the 2026-07-16 full scan already hit it at 150–300 keV beams. `coverage_energy` refuses (`CoverageGridTooNarrow`, `bounds.py:14-28,63-65`) rather than pinning silently, so the failure is loud but the operator must widen it by hand for every high-energy sweep |
| Brem coverage grid stop, 40 keV | `derive.py:63-66` (`WIDE_BREM_STOP_EV`), `--brem-grid-stop` at `derive.py:650-656` | same shape |
| Brem diagnostic **step**, 25 eV | `derive.py:65` (`WIDE_BREM_STEP_EV`), hard-wired at `derive.py:708` | **not overridable at all**: `--brem-step` (`derive.py:680-685`) sets the downstream catalog `E_grid_brem` step, not the diagnostic grid. Widening `--brem-grid-stop` therefore grows the node count linearly with no way to coarsen |
| Line coverage grid start, 10 eV | `derive.py:51` (`WIDE_GRID_START_EV`); no CLI flag | fixed floor |
| Brem diagnostic grid starts at 0.0 eV | `derive.py:66,708` (`np.arange(0.0, ...)`) | a zero first node is representable on a linear grid and not on a log one; converting this grid needs an explicit positive floor decision, not an epsilon |
| Job defaults mirror the same numbers | `job.py:27,30` (`DEFAULT_GRID_STOP = 20_000.0`, `DEFAULT_BREM_GRID_STOP = 40_000.0`), `job.py:280-281` | two independent copies of the ceiling; widening one does not widen the other |
| Catalog line-grid `start_eV` policy | `bounds.py:129-132` | 10 eV at ≤60 keV beam, 50 eV above — a two-step function of beam energy, not a derived floor |
| Line tabulation mesh, 1 eV over the whole padded interval | `_kernels.py:269` | correct but unbounded: the padded interval scales with the requested grid, so a 0.1 eV–100 MeV request would build ~10⁸ nodes before the Henke union |
| Transport LUT point cap | `montecarlo/transport/lut.py` (read-only here; owned by #99) | not re-audited; the research note records that above the cap it silently coarsens the spacing |

## Out of scope / not done

* No change to `montecarlo/transport/**` (owned by #99) and no test pinning the
  LUT's current linear indexing.
* No policy default flipped; no ceiling widened.
* `energy_grid/apply.py` still writes linear descriptors only. Making it
  round-trip a `logspace`/`values` row is the substance of #101 and would change
  catalog text, so it is deliberately untouched.
* No GPU execution: the environment has no CuPy, so the CUDA verdicts above are
  by code inspection of the gating conditions and the kernels' indexing, not by
  running them.

## Follow-ups for #100 / #101

1. **#100 — sinc windowing on actual coordinates.** Replace the
   `(E - E_grid[0]) // dE` bracket in `_per_hkl.py:298-301,604-607` with
   `searchsorted` on `E_grid` (the same primitive `_interp_index` already uses),
   then drop the `_setup.py` guard. Smallest real unlock, and it is local.
2. **#100 — conservative detector rebinning.** `TimepixResponse` needs explicit
   input-bin masses rather than `spec * dE_fine`; `_energy_bin_edges_and_widths`
   is the existing helper to reuse. `poisson_core` falls out of the same change.
3. **#100 — physical-energy peak metrics. Done in #110.** `line_metrics` now
   converts `peak_widths`' interpolated sample crossings to energies by
   interpolating `E`, and cuts the `n_fwhm` window in eV; see the updated
   verdict above.
4. **#101 — grid identity in artifact/case/checkpoint keys.** `grid_identity`
   now protects the in-memory response caches only. The same collision exists
   wherever a grid is summarized by `(start, stop, num)`; the compact codec in
   `_energy_grid_encoding.py` is the place to check next.
5. **#101 — ceilings that track the request.** Derive `WIDE_GRID_STOP_EV` /
   `WIDE_BREM_STOP_EV` from the beam energies actually being scanned, expose the
   brem diagnostic step, and collapse the `derive.py`/`job.py` duplicate
   constants to one owner.
6. **Detector channels as their own coordinate.** The doc states the policy
   (explicit edges, zero-based channel as an explicit edge); `EnergyBins`
   (`detectors/spec.py:39-60`) still carries node arrays only. A separate
   `edges` field is the prerequisite for scoring a nonuniform source spectrum.
