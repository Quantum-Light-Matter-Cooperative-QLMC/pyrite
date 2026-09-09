# Issue #66 — collapse the four CPU transport kernels into one templated body

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/66
Branch: `66-collapse-the-four-cpu-transport-kernels-into-one-t` (issue body names
`66-collapse-transport-kernels`; the actual linked branch is this one).
Worktree: `wt/pyrite/66-collapse-the-four-cpu-transport-kernels-into-one-t`.

Authority (direct user dispatch): commits checkpoint, push no, issue-writer no,
pr no, delegate no.

Dependencies verified landed before start: #64 finding 13 (design record lifted,
9005d13b), #64 finding 16 / phase 3 (`_jit_kernel.py` relocated and split,
9d5dfc34), #65 mc_spectrum decomposition (closed).

## Golden contract (M1)

`checks/transport_core_goldens.py` — capture/check/bench over 5 CPU kernels x
{frozen, midpoint} x {straggling off, on} = 20 runs (Ne=300 carbon, seed 66;
midpoint runs add max_dE_frac=0.05). Bit-for-bit compare, exact equality.
Golden data: `.golden66/` (untracked working artifact, captured from pre-change
code at 74f3bf39). Self-check passes.

    uv run python checks/transport_core_goldens.py check --dir .golden66
    uv run python checks/transport_core_goldens.py bench --dir .golden66

## Baselines (M0, recorded at 74f3bf39)

Cold numba cache (transport caches purged), first-call time per kernel
(compile + run, Ne=300): lockstep 6.06s, lockstep_lut 3.14s, grooved 5.44s,
perelectron 2.45s, perelectron_lut 2.25s. Sum ~19.3s.

Warm bench (Ne=4000, `.golden66/timings_bench.json`):
lockstep frozen/nostraggle 0.26s, midpoint/straggle 0.39s;
lockstep_lut 0.09s / 0.27s; grooved 0.19s / 0.30s;
perelectron 0.23s / 0.46s; perelectron_lut 0.14s / 0.34s.

## Where the five kernels differ (M3)

Common skeleton in all five: (1) draw elastic-collision distance from the total
rate; (2) truncate at geometry; (3) close energy/clock (straggled or
deterministic; frozen or midpoint); (4) record segment, advance, decrement tau;
(5) exit / internal boundary / collision (element pick + angle draw + rotate).

Axis 1 — lockstep vs per-electron:
- Loop: step-major `while lockstep_step < max_steps and n_alive > 0` over all
  electrons vs run-to-completion `for _step in range(max_steps)` per electron
  over `[e_start, e_start+e_count)`.
- RNG: shared `rng.random()` (order: step-major across electrons) vs
  counter-addressed `_stream_uniform_scalar(key, draw)` per electron.
  Straggling keys: lockstep reads `stream_keys_arr[e]`; per-electron reuses its
  own `stream_key[e]`. Same array, same salted rehash downstream.
- State: per-electron arrays `tau_left[e]`/`flight_of[e]`/`substep_of[e]` vs
  per-thread scalars.
- Tables: ragged typed lists `L_Js[L]` vs padded `(n_layers, max_el)` + `L_nel`;
  `_dEds_spliced_compound_scalar(row...)` vs
  `_dEds_spliced_packed_scalar(tables, L, n_el, ...)`.
- Element pick: lockstep buffers rates in `rate_arr` and reuses them;
  per-electron recomputes rates inside the pick loop (no per-thread array).
- Layer lookup: `np.searchsorted(internal_bounds, z, "right")` vs
  `_searchsorted_right_scalar(bounds, z, n_layers-1)` — equivalent because
  `internal_bounds = L_bot[:-1]` has exactly n_layers-1 entries (api.py:502).
- Output: shared `nseg` counter + raise at `max_segments` vs slot
  `i*cap + local_nseg` with `if local_nseg < cap` guard, `seg_count[i]`,
  `exit_code[i]`; no raise (caller replays).
- Exit: shared counters + `alive[e] = False` vs exit_code priority chain
  (back, trans, side, cutoff; initial STEP_LIMITED).

Axis 2 — exact vs LUT:
- Rate: per-element loop vs `_lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)`.
- Stopping: `_dEds_spliced_compound_*` vs `_lut_lerp_2d(lut_dEds, ...)`.
- Clock: `step / beta_from_keV_scalar(E)` vs `step * _lut_lerp_1d(lut_inv_beta)`.
- Element pick: cumulative-rate vs `_lut_lerp_3d(lut_cdf)`; LUT draws
  `u = rng.random()` unscaled.
- Alpha: Mott table interp / SR-Joy at post-flight energy vs
  `_lut_lerp_3d(lut_alpha)` at post-flight energy.
- Straggling ALWAYS uses the exact per-element tables (LUT cores take
  `L_Js..L_E_cross` appended at the end of the signature just for this).

Axis 3 — ungrooved vs grooved (lockstep only):
- Grooved adds step 2b facet truncation (`surface_first`, included in
  `geometry_event`, cleared by cutoff/cap), vacuum-leg follow-through (vac_*
  rows, clock advance, surface_eps nudge, surface/zero-event guards), `nvac`
  return, `max_vac`.
- Loop control: `n_eligible` + per-electron `material_steps` (re-entry does not
  consume a step) vs global `lockstep_step`.
- Collision gated on `full_j` (not limited/crossed/surface).
- Flight bookkeeping: identical semantics (non-limited closes the flight),
  hoisted before surface handling in the grooved core.

## Merge design notes (M4)

- Factory `make_cpu_transport_core(grooved, per_electron, lut)` returning
  `@njit(cache=True)` closures; axis flags as frozen freevars so numba prunes
  dead branches per specialization. VERIFIED on numba 0.67.0: closures cache
  (`<factory>.locals.core` .nbc/.nbi entries per flag specialization) and a
  second process loads them with the correct freevar values. Fallback if this
  breaks on the real body: four module-level wrappers over one shared body.
- One body text: runtime draws behind a small stream tuple so lockstep and
  counter-addressed modes share the row logic.
- Straggling invariants (cores.py:168-201 and
  docs/repo-design/compute/straggled-transport-integration.md) are preserved
  verbatim in the merged body.

## Handoff state (M2 and M5 done, paused before M4)

Verified: all 20 goldens reproduce bit-for-bit, and the focused transport
tests pass (165 passed, 21 CUDA-hardware skipped: test_groove,
test_transport_per_electron, test_transport_energy_model,
test_straggling_remaining_cores, test_straggling_transport_integration,
test_transport_cutoff, test_segment_staging, test_transport_diagnostics).

M2 grouped-parameter refactor is applied everywhere:

- `cores.py`: all five kernels take grouped tuples and unpack them at the top
  of the body; bodies below the unpack are unchanged.
- Layout (the call-site contract):
  `control = (max_steps, max_segments, elastic_model_code, energy_model_code,
  max_dE_frac)` (per-electron cores carry but ignore `max_segments`);
  `geometry = (n_layers, internal_bounds, z_total, finite_footprint,
  width_ang, height_ang, L_nel, L_top, L_bot)` (exact lockstep cores never
  read `L_nel`);
  `materials` = 11 exact tables `(L_Js, L_Zs, L_ks, L_coeffs, L_E_cross,
  L_ncm3, L_sr_rate_numer, L_mott_numer, L_mott_denom1, L_mott_denom2,
  L_sr_joy_numer)`, or only the first five (straggling sampler tables) on LUT
  cores;
  `mott = (mott_has_table, mott_start, mott_len, mott_logE_flat,
  mott_logA_flat)`;
  `lut = (E_min_keV, inv_dE_keV, n_energy, total_rate, dEds, inv_beta, cdf,
  alpha)`;
  `state = (alive, clock, pos, dirs, E_keV, E_cut_by_electrons)`;
  `segments` = the 11 seg_* arrays in `_alloc_scratch` order;
  `straggling = (straggle_on, stream_keys_arr, stragg_dE)` on lockstep cores,
  `(straggle_on, stragg_dE)` on per-electron cores (their `stream_key` group
  feeds the salted rehash);
  lockstep cores take `(Ne, rng, control, ...)`; grooved inserts `groove =
  (spacing, depth, st, ct, max_vac, vac_start, vac_end, vac_E, vac_t0,
  vac_id)` after `geometry`; per-electron cores take `(run, control, ...)`
  with `run = (e_start, e_count, cap, stream_key)` and `pe_out = (seg_count,
  exit_code)` after `segments`.
- `batching.py`: both per-electron drivers build the tuples once after upload
  and pass them through; `scratch` IS the `segments` tuple.
- `_jit_launch.py`: `run_transport_kernel` / `run_transport_lut_kernel` take
  the same grouped signature positionally and unpack into the flat cupyx
  launch; `make_cuda_transport_lut_core` drops the CPU-only `materials` +
  `straggling` groups by name. UNTESTED on hardware (no CUDA here) — the
  CUDA-only parity test
  `test_transport_per_electron.py::test_launcher_signature_tracks_the_reference_core`
  must pass on a GPU box before landing.
- `api.py`: shared `control`/`state`/`segments`/`materials_ragged`/
  `mott_group`/`straggling_lockstep`/`geometry` tuples built once before the
  core dispatch; LUT geometry swaps in `transport_lut.n_el`.
- `tests/montecarlo/test_groove.py`: direct `_transport_core_grooved` call
  regrouped.
- `batching.py`: exact and LUT entry points retain their distinct upload/table
  preparation, then pass prebuilt kernel groups through one
  `_drive_per_electron_batches` implementation for capacity replay, compaction,
  exit-code accounting, and optional device joining. Focused tests and all 20
  goldens pass bit-for-bit after the hoist.

Remaining: M4 (factory), M6 (full checks incl.
`test-suite core`, `--numba`, `pyrite-dev verify`, duplicate-window scan
139 -> <20, physics-ledger-auditor). See issue #66 acceptance for thresholds.

Verify commands:
    uv run python checks/transport_core_goldens.py check --dir .golden66
    uv run python checks/transport_core_goldens.py bench --dir .golden66
    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/montecarlo

## Milestones

- [x] M0 baselines, M1 goldens
- [x] M2 grouped parameter tuples on all 5 kernels; goldens hold
- [ ] M4 factory generates all 4 (+5th) kernels; goldens hold
- [x] M5 batching.py per-electron driver hoist
- [ ] M6 full checks: core suite, --numba, verify, perf, dup scan, ledger audit
