# Issue 57 slice: analytic inter-electron decoherence for the coherent path

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/57

Work branch: `57-settable-coherent-bunch-length`

## Problem

Originally filed as "coherent bunch length is set by the radiation line
wavelength" — that symptom is **already fixed** on `main`
(`bunch_length_fs`/the resolved `longitudinal` policy is emission-agnostic,
locked in by `tests/scan/test_sweep.py::test_build_cases_preserves_beam_bunch_length_for_coherent_emission`,
commits `326204a3`/`85cf1719`). Live-verified during triage: an explicit
`bunch_length_fs` lands unchanged in both coherent and incoherent case dicts.

Follow-up testing found the real bug behind the observed symptom: running the
`hopg_hbn` default profile (`beams.gaussian_200fs`: `envelope_rms_fs=200.0`,
`transverse_fwhm_mm=0.1`) under `emission="coherent"` still shows large,
non-vanishing "coherent enhancement" even though a 200 fs bunch is many
optical periods and should fully decohere the inter-electron cross terms.

Root cause is the already-ledgered `transverse-bunch-form-factor` discrepancy
(`docs/validation/ledger-core-coherent-physics.md`,
`docs/validation/radiation-physics/transverse-bunch-form-factor.md`,
2026-08-19), and it is not actually transverse-specific — it applies to the
longitudinal offset by the identical mechanism:

- `mc_spectrum(coherent=True)` gives each electron a per-electron offset
  (`t0_ang` from `bunch_length_fs`/`longitudinal`, and/or a rigid transverse
  translation from `beam_fwhm_mm`/`transverse_distribution`), bakes both into
  one flat complex phase sum over every segment of every electron, and squares
  **once**. That is a single Monte Carlo realization of a coherent (amplitude)
  sum, not an average of independent intensities.
- The physically correct observable is the ensemble average over those
  offsets, which has a closed form: `exp[-(ω σ_z)²]` (longitudinal) and
  `exp[-(q⊥ σ⊥)²]` (transverse) — the standard Gaussian/superradiant form
  factor. The code never takes that average.
- For a squared coherent sum this is speckle statistics, not shot noise: a
  single realization of `|Σ random-phase terms|²` has ~100% relative
  fluctuation around its mean **independent of N** (Rayleigh/exponential
  statistics — unlike an incoherent average, which shrinks as `1/√N`). The
  ledger measured this directly: 2.83–3.25× spurious enhancement at `Nₑ=300`
  for the transverse case, flat vs. electron count, confirmed non-shrinking
  even at large tile counts (speckle angular scale ~2×10⁻⁴ rad needs
  `n_side≈180` / ~32000 directions/case to resolve by brute-force averaging,
  which is not affordable).
- Because `hopg_hbn`'s default beam bundles a long bunch *and* a finite spot
  in one profile, the transverse offset's un-averaged speckle noise rides
  along on any coherent run of it regardless of bunch length, which is why
  the enhancement looked bunch-length-related.

## Decisions

- **Fix approach: analytic ensemble average, not more sampling/tiling.** Per
  segment grouping already carries `elec_id`/`electron_id`
  (`src/pyrite/montecarlo/spectrum/lines.py`), so per-electron coherent
  segment sums `S_e = Σ_{j∈e} E_j` (using only the intrinsic geometric/position
  phase — **no** `t0_ang`/transverse offset baked in) are already
  constructible without new bookkeeping. Because the per-electron offsets are
  independent identically-distributed Gaussians, the ensemble-averaged total
  factorizes exactly to

  ```
  Total(E) = F(E) * |Σ_e S_e(E)|²  +  (1 - F(E)) * Σ_e |S_e(E)|²
  ```

  where `F(E) = exp[-(ω(E) σ_z)² - (q⊥(E) σ⊥)²]` is a single scalar per
  (energy, reflection, orientation) row — no RNG draw needed for the
  coherent-spectrum computation at all. `F→1` recovers the fully coherent `N²`
  limit exactly; `F→0` recovers the intra-electron-only floor
  `Σ_e|S_e|²`. This needs a fresh-context re-derivation before it is trusted
  (see below), but the shape follows directly from the existing derivations in
  `coherent-emission.md`/`transverse-bunch-form-factor.md`.
- `F` is a function of photon energy `E` (via `ω(E)`) and of the
  (reflection, orientation) row (via `q⊥ = (ω n̂ + g)⊥`), matching the scope
  the existing sum already runs at — it is **not** a single line-center
  scalar.
- Stochastic `t0_ang`/transverse-offset sampling stays as-is for transport,
  trajectories, and the incoherent path; this task changes only how the
  **coherent line spectrum** consumes those distributions (closed-form `F`
  from `bunch_length_fs`/`beam_fwhm_mm`/resolved `longitudinal`/
  `transverse_distribution`, instead of the sampled realization's raw phase).
  Do not remove or alter `_sample_bunch_offsets`/transverse sampling itself.
- Out of scope for this slice (flag explicitly in the ledger as follow-on,
  do not silently half-handle):
  - Elliptical `beam_fwhm_y_mm` / Courant–Snyder `transverse_distribution`
    covariance and any transverse–longitudinal correlation — the doc notes
    these don't factorize into the simple product `F` above and need their
    own derivation.
  - The finite-footprint branch (`crystal_width_mm`/`crystal_height_mm`),
    where the transverse offset also perturbs escape attenuation (an
    amplitude effect, not just phase) — the form factor covers phase only.
- Existing coherent checkpoints/output are **not** a compatibility surface
  here: `coherent-emission` is explicitly unsigned-off and blocked by this
  exact discrepancy, so numeric changes are a correctness fix, not a break.
  Confirm during implementation whether `parameter_sha256`/
  `CURRENT_IDENTITY_VERSION` (`src/pyrite/campaign/profiles.py`) need touching
  — current expectation is no, since nothing about the *inputs* changes, only
  how the already-`coherent_emission`-gated output is computed.
- GPU/CPU parity: `coherent_jit_kernel.py`/`coherent_stream_jit_kernel.py`
  reduce an already-built per-segment `phase` array (cos/sin + accumulate);
  they don't compute the offset injection themselves. The expectation is that
  restructuring happens upstream (in `lines.py`, where `d_all`/`t0_ang` are
  assembled today) and the two kernels end up summing `S_e`-shaped inputs with
  `F` applied outside — confirm this during implementation rather than
  assuming no kernel changes are needed.

## Plan

- [x] Fresh-context re-derive the combined closed form above (independent
      Gaussian `σ_z`, `σ⊥`; factorization; both limiting cases) as its own
      validation write-up, extending
      `docs/validation/radiation-physics/transverse-bunch-form-factor.md` (or
      a new paired longitudinal doc) — do not just restate the existing
      per-axis notes as the derivation. Landed as its own doc,
      `docs/validation/radiation-physics/coherent-inter-electron-decoherence.md`.
- [x] Implement per-electron `S_e` construction (offset-free geometric phase
      only) reusing `elec_id`/`electron_id` grouping, and the closed-form `F`
      from `bunch_length_fs`/resolved `longitudinal` and `beam_fwhm_mm` (CPU
      path, `src/pyrite/montecarlo/spectrum/lines.py`). Implemented `F` as the
      *empirical* characteristic function of the actual sampled
      `initial_t0_ang`/`initial_r_ang` population rather than a closed-form
      parametrized `σ` — converges to the same boxed result, needs no
      per-policy `σ`-resolution logic, and extends for free to elliptical/
      Courant–Snyder transverse spots (see below).
- [x] Confirm/port the GPU reduction path
      (`coherent_jit_kernel.py`/`coherent_stream_jit_kernel.py`) to the same
      structure; add/extend a CPU/GPU reproducibility check. **Ported, with
      no new device code.** Both kernel entry points already ACCUMULATE into
      the caller's buffer, so `Σ_e|S_e|²` is the same kernel called once per
      electron over that electron's own lines, at unit mosaic weight, into a
      zeroed buffer (`_coherent_jit_grouped_row`, shared by `_accumulate`'s
      reduction and the batched per-row fallback); the streaming path uses a
      fused segmented reducer over whole-electron segment blocks and blends
      per row in plain CuPy array math rather than
      `finalize_coherent_fields`, whose fused collapse would sum rows before
      `F` could multiply them. The stream prologue now takes the geometric
      (offset-free) position/phase slope like the other two paths — a no-op
      swap when no offset is configured, so the inactive dispatch stays
      bit-for-bit. Verified on the lab RTX 5080 via `pyrite remote sync` +
      ssh: 8 device tests pass (closed-form reference on the streaming and
      per-row-reduction routes; all three routes in lockstep against the
      generic CuPy path they used to fall back to, across a 9-row mosaic
      cone so each row carries its own `F`; offset-free dispatch bit-for-bit)
      plus a real `simulate_trajectories` (300 electrons, 200 fs bunch,
      0.1 mm spot) → `mc_spectrum(coherent=True)` smoke agreeing with the
      generic path to 5.4e-4. The first production-scale `hopg_hbn` run exposed
      the original grouped streaming floor as launch-bound at ~0.8 ms/electron.
      The fused fix removes the `Nₑ` launch count: an uncached 20,000-electron
      HOPG run on the lab RTX 5080 sustained 431 cases in 38 s (~11.3 cases/s),
      versus ~0.05 cases/s for the regressed per-electron path, while GPU
      utilization rose from ~10% to 56–62%. The two non-streaming per-row
      routes retain their `Nₑ`-launch implementation because `hopg_hbn` does
      not use them.
- [x] Add limiting-case regression tests: point source/point bunch (`F→1`,
      recovers today's `N²` degenerate behavior bit-for-bit), large-σ floor
      (`F→0`, recovers `Σ_e|S_e|²` and matches the incoherent-per-electron
      scaling described in `coherent-emission.md`), and a mid-regime check
      against the ledger's measured `hopg_hbn`/300-electron numbers.
      `test_coherent_decoherence_blend_matches_reference_formula` pins the
      general blend against an independently-built reference (not a
      self-consistency check); `test_coherent_decoherence_inactive_by_default`
      pins the `F→1`/no-offset degenerate case bit-for-bit.
- [x] Explicitly reject/error (not silently ignore) the out-of-scope cases:
      elliptical/Courant–Snyder transverse distributions and the
      finite-footprint branch, under `emission="coherent"`/`"both"`, until
      their own derivations land — or confirm they already error and just
      document it. **Revised during implementation**: elliptical/Courant–
      Snyder spots are *not* rejected — the empirical-`F` implementation
      choice measures whatever offset distribution was actually sampled, so
      it covers them for free without a separate derivation (documented as a
      byproduct of the implementation choice, not a new physics result). Only
      the finite-footprint branch's general partially coherent case
      (amplitude/phase coupling breaks the derivation's first algebraic step)
      and `sinc_cutoff` (implementation gap in the grouped-floor reduction)
      are rejected. Follow-up implementation now supports the exact fully
      longitudinally decohered Gaussian limit: the analytic longitudinal
      factor vanishes, so the grouped CPU/CUDA-JIT floor keeps each sampled
      electron's finite-prism attenuation. This covers `hopg_hbn`'s regular
      200 fs Gaussian beam and is pinned by
      `test_long_gaussian_bunch_supports_finite_footprint` plus the CUDA device
      twin. Partially coherent finite-footprint grids still raise.
- [x] Add `Validation: <id>` marker(s) in code and update
      `docs/validation/ledger-core-coherent-physics.md` (promote
      `transverse-bunch-form-factor` past `discrepancy`, add a row for the
      longitudinal/combined result) and
      `docs/physics/radiation-physics/coherent-emission.md`. Human sign-off
      stays pending per repo physics rules even after this lands.
      `transverse-bunch-form-factor` promoted `discrepancy` → `rederived`;
      new `coherent-inter-electron-decoherence` row added, `rederived`; both
      docs and `status-summary.md`/`domain-inventories.md`/`index.md`
      regenerated consistently. Human sign-off still pending on all of them.
- [x] Run focused + coherent-path regression suites, lint, typecheck; confirm
      no unrelated checkpoint-identity behavior changed. `lint`/`typecheck`
      clean; `tests/montecarlo/test_coherent_emission.py` 16/16 passed.
      `parameter_sha256`/`CURRENT_IDENTITY_VERSION` confirmed untouched —
      this changes only how the already-`coherent_emission`-gated output is
      computed, not any profile input.
      Follow-up finite-footprint slice: coherent CPU suite 18/18. After the
      production-scale launch regression fix, CUDA device decoherence suite
      12/12 on the lab GPU; focused fused-kernel tests 4/4; lint clean.
      Core suite reached 1821 passed/74 skipped with only the sandbox-blocked
      forkserver end-to-end test failing (`PermissionError: [Errno 1] Operation
      not permitted`). Current full typecheck is blocked by 21 unrelated
      pre-existing Altair diagnostics under `src/pyrite/plots/altair/`; changed
      Monte Carlo files have no Serena diagnostics.

## Delegation

- Slice 1 (derivation + ledger/doc scaffolding): needs the fresh-context
  derivation before anything downstream can be trusted — do this first, not
  in parallel with implementation.
- Slice 2 (CPU implementation in `lines.py` + limiting-case tests): depends on
  slice 1's closed form. Needs `monte-carlo` skill (stochastic/RNG-adjacent
  correctness) and `physics:implement`.
- Slice 3 (GPU kernel parity): depends on slice 2's restructuring shape.
  Needs `monte-carlo` skill's CPU/GPU reproducibility scope; may need
  `performance` if the restructuring changes kernel shape/cost.
- Slice 4 (docs + ledger promotion + regression lock-in): depends on 1–3
  landing. Needs `documentation-maintenance` + `regression-testing`.

No slice is `one-shot`: the factorization/out-of-scope boundaries (elliptical
spot, finite-footprint amplitude coupling, identity-version question) are
material open decisions that should be confirmed, not assumed, before a
worker starts. Given the cross-subsystem scope (CPU + GPU kernels, physics
derivation, validation ledger) and correctness stakes, this reads as
`lead-task` scope rather than independent lite/normal slices.

## Acceptance checks

- Fresh-context derivation doc for the combined form factor, cross-checked
  against the existing per-axis derivations with no divergent sign/factor.
- New limiting-case tests pass and pin both endpoints plus the measured
  mid-regime `hopg_hbn` numbers from the ledger.
- CPU and GPU coherent paths agree on the new construction (existing
  CPU/GPU-parity test pattern extended, not a new one-off check).
- `docs/validation/ledger-core-coherent-physics.md` and
  `docs/physics/radiation-physics/coherent-emission.md` updated; discrepancy
  row promoted or explicitly superseded, not left stale.
- `uv run pyrite-dev lint`, `typecheck`, and the coherent/longitudinal focused
  suites pass.
