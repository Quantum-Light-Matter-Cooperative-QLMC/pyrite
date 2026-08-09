# Electron transport correctness

## Problem and scope

The August 2026 transport review identifies three definite correctness defects
in the current 1--300 keV electron path:

1. coherent CXR uses the segment midpoint position with the segment-start
   transport age;
2. a flight that crosses `E_cut_keV` is retained at its full sampled or
   boundary-limited length; and
3. electrons that exhaust `max_steps` are included in `n_stopped` rather than
   reported as computationally incomplete.

Source: [`docs/electron_transport_physics_recommendations.docx`](../../../../docs/electron_transport_physics_recommendations.docx), Stage 0.

This task owns the correctness fixes and their migration/validation surface.
It does not introduce energy-loss substeps or replace elastic/stopping models.
Those are separate dependent tasks.

The review referred to `transport(4).py`; the current owner is
`src/cxr_mc/montecarlo/transport.py`. The same algorithm now exists in four
paths, all in scope:

- lockstep ungrooved CPU: `_transport_core_ungrooved`;
- grooved/layered CPU: `_transport_core_grooved`;
- per-electron CPU: `_transport_core_ungrooved_perelectron`; and
- CUDA: `montecarlo/transport_jit_kernel.py`.

`montecarlo/spectrum.py::mc_spectrum` and `::mc_brem_spectrum` consume the
segment schema. `simulate_trajectories` owns validation and termination
diagnostics.

## Implementation path and likely owners

- Pair coherent midpoint position with midpoint transport age. Preserve
  `t_ang` as segment-start age for compatibility unless the migration decision
  below explicitly changes the schema; the smallest correction can derive
  midpoint age from `t_ang`, `L_ang`, and the current constant segment beta.
- Truncate a cutoff-crossing material flight at the first point where the
  current stopping rule reaches `E_cut_keV`, then recompute midpoint, length,
  endpoint energy, and clock advance consistently. Do this before recording in
  every transport core, including grooved and CUDA paths.
- Distinguish physical cutoff stops from step-budget exhaustion. Raise on
  incomplete histories by default and provide an explicit, visible permissive
  path only if a caller needs partial trajectories.
- Validate finite positive initial/cutoff energies with
  `0 < E_cut_by_electrons[i] < initial_E_keV[i]`, reject non-finite/zero beam
  directions before normalization, and confirm existing layer-continuity
  validation covers the review's requirement.
- Update the `electron-transport` and coherent-emission validation records.
  The existing coherent validation rederives the phase convention but does not
  test that position and time refer to the same point on a physical flight.

Likely tests: `tests/montecarlo/test_montecarlo.py`,
`test_transport_per_electron.py`, `test_transport_core_default.py`,
`test_groove.py`, and `test_coherent_emission.py`. CUDA parity requires the
remote GPU workflow; do not run a heavy/GPU sweep locally.

## Checklist

- [x] A -- Specify segment-time compatibility and incomplete-history API;
      identify every result/count consumer before changing keys or exceptions.
- [x] B -- Add the constant-velocity one-flight versus two-subsegment coherent
      field invariance regression, covering both coherent reduction routes.
- [x] C -- Correct coherent midpoint time in all spectrum paths and add a new
      `Validation: <id>` marker, ledger row, assumptions, and limiting case.
- [x] D -- Add a constructed cutoff-crossing regression; truncate the terminal
      flight consistently in lockstep, grooved, per-electron, and CUDA cores.
- [x] E -- Add explicit cutoff-stop and step-limited termination states/counts;
      raise by default when any history is incomplete.
- [x] F -- Add the review's input validation without changing valid runs.
- [ ] G -- Prove CPU-core parity with focused statistical/deterministic tests;
      verify CUDA behavior on the configured remote GPU.
- [ ] H -- Update physics documentation/ledger and regenerate affected golden
      data only after the implementation and fresh-context review agree.

## Decisions and open questions

- **Decided:** a physical segment's stored representative position and time
  must describe the same point.
- **Decided:** no below-cutoff portion may contribute to line or bremsstrahlung
  radiation.
- **Decided:** `max_steps` exhaustion is not a physical stop and must be visible.
- **Decided (A): preserve `t_ang` as segment-start age; do not add
  `t_mid_ang`.** `r_mid` remains the representative position and coherent
  radiation derives its matching time once, before either reduction route, as
  `t_ang + 0.5 * L_ang / beta_from_keV(E_keV)`. `t0_ang` remains an additive
  per-electron absolute-time offset. This keeps trajectory ordering/reveal
  semantics and the eight-array host/device staging contract unchanged. For
  synthetic coherent inputs, absent `t_ang` retains the existing zero-start-age
  compatibility default, after which the same midpoint correction applies.
- **Decided (A): incomplete histories are errors, with no partial-result API.**
  `simulate_trajectories` raises `RuntimeError` by default whenever any entered
  electron exhausts `max_steps`; the stable message must include
  `n_step_limited`, `Ne`, and `max_steps`. No current internal, app, check, or
  checkpoint caller consumes partial trajectories, so do not add
  `allow_incomplete` or attach recoverable segments to the exception. Internally
  every core must distinguish cutoff-stop from step-limited termination. A
  successful result adds `n_cutoff_stopped` and `n_step_limited == 0`, while
  legacy `n_stopped` remains as an alias of `n_cutoff_stopped` rather than its
  former residual count. Successful-count invariant:
  `n_backscattered + n_transmitted + n_side_exited + n_missed + n_cutoff_stopped
  == Ne`.
- **Decided (A): Stage 0 uses the present left-endpoint stopping rule exactly.**
  On a candidate material flight, hold `dEds(E_start)` and `beta(E_start)`
  constant. If `E_start + dEds * L_candidate <= E_cut`, set
  `L_cut = (E_cut - E_start) / dEds`, retain only `min(L_candidate, L_cut)`,
  and recompute midpoint, endpoint energy, and clock from that length. A
  geometry boundary wins an exact distance tie; otherwise the cutoff event
  clears collision/boundary exit handling. This is an algebraic crossing under
  the current rule, not the local solve, midpoint energy, or hazard substepping
  owned by controlled propagation.
- **Decided (A): population-specific cutoffs clip in the spectrum adapter.** A
  shared line/bremsstrahlung transport uses the lower per-electron cutoff, so
  the current start-energy masks can retain the below-cutoff tail of a segment
  crossing the higher population cutoff. Before line or brem reduction,
  `spectrum.py` must apply the same linear crossing rule using the segment's
  layer composition, shorten `L_ang`, and move `r_mid` from the original
  midpoint to the retained midpoint. The coherent midpoint time is derived
  after this clipping. This adapter is a no-op when the segment does not cross
  that consumer's cutoff and prevents either radiation path from recovering a
  below-cutoff contribution.

## Slice A consumer inventory and migration contract

### Producers and internal adapters

- `montecarlo/transport.py`: `_transport_core_ungrooved`,
  `_transport_core_grooved`, `_transport_core_ungrooved_perelectron`,
  `_run_per_electron_transport`, and `simulate_trajectories`. The lockstep and
  grooved cores can classify remaining `alive` entered electrons as
  step-limited; the per-electron driver must replace the collapsed
  `EXIT_ALIVE_OR_STOPPED` code with distinct cutoff and step-limit codes.
- `montecarlo/transport_jit_kernel.py`: `_transport_kernel`,
  `run_transport_kernel`, and `make_cuda_transport_core` mirror the
  per-electron exit-code and segment-buffer contract. CUDA parity remains G and
  must use the remote GPU workflow.
- `montecarlo/spectrum.py`: `_segments_in_layer` and `_segments_on_device`
  preserve additive keys; `mc_spectrum` (incoherent, batched coherent, streamed
  coherent, and per-reflection coherent paths) consumes `r_mid`, `v_hat`,
  `L_ang`, `E_keV`, `t_ang`, `t0_ang`, `elec_id`, and `layer`;
  `mc_brem_spectrum` consumes the same geometry/energy/length/identity subset.
  Population cutoff clipping belongs here so live and repair paths cannot
  diverge.
- `montecarlo/runner.py`: `_transport_case`, `_brem_for_case`, and
  `_transport_lines_for_case` call transport; `_lines_for_segments` and
  `_brem_wide_from_segments` select population/layer segments; and
  `_spectrum_case_impl` reduces transport diagnostics to `eta`, `hit_frac`, and
  `n_segments`. All production/repair calls keep fail-closed defaults.

### App, plotting, and validation/check consumers

- `campaign/config.py::gate_cases_by_penetration` reads `n_transmitted`; its
  watchdog must fail rather than treating an incomplete run as low survival.
- `plots/trajectories.py::_trajectory_data` reconstructs segment starts and
  endpoints, orders by start-age `t_ang`, and reads backscatter/transmission
  counts. `plots/altair_trajectories.py` and
  `plots/plotly_trajectories.py` consume that projected dataset indirectly.
  Keeping `t_ang` start-based preserves track and reveal behavior.
- `apps/anchor_figures.py::model_spectra` and `model_coherent_spectra` call
  transport and read backscatter/transmission; `single_segment_anchor` supplies
  a synthetic incoherent segment and needs no new time key.
- Direct developer-check callers are `checks/detector_solid_angle_check.py`,
  `checks/mosaic_mc_check.py`, `checks/multilayer_check.py`, and
  `checks/multilayer_validation_check.py`; `multilayer_slice3_check.py`
  consumes segment dictionaries passed by its caller. They remain fail-closed.

### Results, checkpoints, and public compatibility

- Raw segment dictionaries and transport termination counts do not cross the
  result/checkpoint boundary. `runner._spectrum_case_impl` emits only spectra,
  `eta`, `hit_frac`, and `n_segments`; `results/store.py::store_result` persists
  spectra, `eta`, and `hit_frac`. `runs/run.py` CAS payloads likewise exclude
  segments and termination counts. Therefore the new successful-result count
  keys are additive only at the direct `simulate_trajectories` API.
- `montecarlo/__init__.py` re-exports `simulate_trajectories`; no new public
  exception or configuration symbol is required. Existing callers catching
  `RuntimeError` remain compatible. `n_stopped` keeps its key and narrows to its
  intended physical meaning.
- Test helpers constructing segment dictionaries (`tests/helpers/segments.py`
  plus local fixtures in coherent, staging, surface-orientation, multilayer,
  GPU-retry, and anchor tests) need no `t_mid_ang`; coherent fixtures keep or
  add explicit `t_ang` where phase matters.

### Owner and test impact for B--F

- B--C: `montecarlo/spectrum.py` and
  `tests/montecarlo/test_coherent_emission.py`; cover both coherent reduction
  routes with one-flight/two-half-flight complex-field and spectrum invariance.
- D: all four transport cores plus the spectrum cutoff adapter;
  `test_montecarlo.py`, `test_transport_per_electron.py`, `test_groove.py`, and
  focused staging/CUDA-source parity assertions. Include a shared-transport case
  whose line cutoff exceeds its transport cutoff.
- E: transport cores/wrapper and count fixtures in `test_montecarlo.py`,
  `test_transport_per_electron.py`, `test_transport_core_default.py`, and
  `test_groove.py`. Force a one-step incomplete run and assert failure text;
  separately prove cutoff stops preserve the successful-count invariant.
- F: `simulate_trajectories` validation and `test_montecarlo.py`; finite positive
  initial/cutoff energies and finite nonzero inward beam directions only.

## Next dispatchable slice

G may proceed: prove aggregate CPU-core agreement across seeds and verify the
per-electron/CUDA cutoff and termination behavior through the configured remote
GPU workflow. The CUDA source now mirrors the CPU reference, but this checkpoint
contains no runtime GPU evidence. Keep H's fresh-context physics validation and
documentation/ledger closure separate.

## Slice B--C outcome

- `mc_spectrum` derives `seg_t_mid = seg_t + 0.5 * seg_L / beta_all` once;
  both coherent reduction routes consume the resulting `d_all`. `t_ang`
  remains segment-start age and the segment/device schema is unchanged.
- The deterministic 30 keV HOPG regression independently evaluates the
  centered complex field. A fixed reference emitter makes the formerly global
  subdivision phase observable in the public spectrum. Before the fix, both
  routes failed all 700 bins with up to about 96% relative error; afterward
  both pass at backend-scaled reduction tolerances.
- `Validation: coherent-segment-midpoint-time` owns the source equation, units,
  assumptions, `L -> 0`, single-self-term, and one-flight/two-half limiting
  cases. Its ledger status remains `unverified`; fresh-context validation is H.
- Implementation-context physics review: `L/beta` is an Angstrom time-like
  length, so `0.5 L/beta` matches `t_ang`; the sample-frame midpoint and time
  use the same constant velocity; phase remains dimensionless; final `/Ne`
  normalization and all incoherent paths are unchanged.

## Slice D--F outcome

- All four material cores now compute the exact left-endpoint crossing distance
  before recording a terminal flight. The retained endpoint is set to the
  electron cutoff, the shortened length drives midpoint and clock advance, and
  an exact geometry-distance tie remains a geometry event. Grooved cutoff wins
  only when strictly earlier than a layer/prism/facet event.
- `spectrum._clip_segments_to_cutoff` applies the same algebra with the emitting
  layer's composition before both line and bremsstrahlung reduction. It retains
  segment-start energy/time and moves the midpoint along the retained prefix;
  the coherent midpoint-time derivation therefore sees the clipped length.
- Lockstep and grooved cores return cutoff and surviving-alive counts. The
  per-electron CPU/CUDA contract uses distinct cutoff, step-limited, and
  not-entered exit codes. `simulate_trajectories` raises the stable incomplete
  history error before exposing segments, and successful dictionaries report
  `n_cutoff_stopped`, `n_step_limited == 0`, and `n_stopped` as an alias.
- Validation now rejects non-finite/non-positive initial or cutoff energies,
  cutoffs not strictly below each sampled initial energy, and malformed,
  non-finite, zero, or outward beam directions before transport.
- Deterministic regressions cover exact terminal clipping in both ungrooved CPU
  cores and the grooved/layered core, both spectrum entry points, explicit
  incomplete-history failures, successful count semantics, validation, and
  CUDA-source rule/code parity. CUDA runtime validation remains G.

## Delegation slices and required skills

- A is `lead-task` + `repo-orientation`; not Serena `one-shot` because it sets
  public/internal result semantics.
- B--C are `implement-task` + `monte-carlo` + `regression-testing` +
  `physics-review`; not `one-shot` until the time-schema decision is closed.
- D--F are `implement-task` + `monte-carlo` + `regression-testing`; D is not
  `one-shot` because four execution cores must remain equivalent.
- G uses `remote-gpu-jobs`, `performance`, and `run-cxr-mc`; no local GPU sweep.
- H uses `physics-validation` with fresh context, `regen-golden`, and
  `documentation-maintenance`. Only a human may mark physics `signed-off`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_coherent_emission.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_montecarlo.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_transport_per_electron.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_groove.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite core
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
```

- One physical constant-velocity flight and two numerical halves yield the
  same complex coherent field and spectrum within the stated numerical bound.
- A constructed cutoff crossing retains exactly the above-cutoff path; line and
  bremsstrahlung consumers cannot recover a below-cutoff contribution.
- Forced `max_steps` exhaustion raises by default; permissive mode, if retained,
  reports `n_step_limited` separately and never increments `n_stopped`.
- All valid historical CPU cases remain deterministic under their pinned core;
  CUDA and CPU per-electron implementations preserve their documented
  distribution/parity contract.
- New/changed physics has source equation, assumptions, limiting case,
  `Validation: <id>`, ledger coverage, and fresh-context validation.
