# Groove JIT transport: restore stale test_groove.py coverage

Branch: `fix/groove-jit-test-coverage`
TODO scope: direct input, triaged 2026-08-05.

## Problem

`tests/montecarlo/test_groove.py` has 8 failing tests that monkeypatch
`_transport_module.first_surface_event` / `first_prism_exit`. Those module
attributes no longer exist: commit `e338f3c` ("feat(JIT transport): full
support for JIT transport") deleted the pure-Python
`_transport_core_grooved_legacy` core (which imported and called
`first_surface_event`/`first_prism_exit` directly, matching these tests'
monkeypatch signatures) and replaced it with `@njit(cache=True)
_transport_core_grooved` in `src/cxr_mc/montecarlo/transport.py`, which calls
private scalar numba helpers (`_first_surface_event_scalar_numba`,
`_first_prism_exit_scalar`) instead. The tests were never updated; they
currently fail with `AttributeError`.

Confirmed empirically (throwaway script) that patching the scalar numba
helpers directly is *also* not viable: numba bakes the compiled dispatcher in
at JIT-compile time, so post-hoc Python monkeypatching of the referenced
global is silently ignored. There is no cheap shim back to injectable-mock
testing of this inner loop under the current njit design.

## Scope and owners

- `tests/montecarlo/test_groove.py`: rewrite the 8 failing tests (listed below) as
  black-box tests against `simulate_trajectories()` output, using real
  `groove_spec` / `n_atoms_per_ang3` / `E_cut_keV` / `layers` / beam-geometry
  combinations (seed=9, deterministic) that trigger each scenario, instead of
  monkeypatching internals.
- `src/cxr_mc/montecarlo/transport.py` (`_transport_core_grooved`,
  `_first_surface_event_scalar_numba`, `_first_prism_exit_scalar`): read-only
  reference for the current event semantics; not expected to change unless
  black-box triggering proves genuinely infeasible for some scenario, in which
  case flag rather than restructure production code.

Out of scope: reintroducing `_transport_core_grooved_legacy` or any other
non-jit fallback path purely for testability (considered and rejected — see
Decisions). No changes to `src/cxr_mc/montecarlo/groove.py` or `geometry.py`
expected.

## Tests to rewrite (`tests/montecarlo/test_groove.py`)

Each must preserve its original intent, now via real triggering geometry
instead of a mock:

- [ ] `test_surface_cutoff_stops_before_vacuum_reentry` — electron whose
      energy drops below `E_cut` exactly at a groove surface-exit event must
      not search for vacuum re-entry; stops instead
      (`n_stopped == 1`, `vacuum_start_ang` empty, `n_backscattered == 0`).
- [ ] `test_permanent_surface_exit_counts_backscatter` — surface exit with no
      valid re-entry (permanent escape through the entrance facet) counts as
      backscatter, no vacuum leg recorded.
- [ ] `test_surface_reentry_does_not_consume_material_step_budget` — a
      material-exit/re-entry pair through vacuum does not consume
      `max_steps`; with `max_steps=1` the electron still transmits after
      reentry.
- [ ] `test_surface_event_exhaustion_raises_instead_of_classifying_survivor_stopped`
      — repeated re-entries exceeding the per-electron surface-event guard
      raise `RuntimeError` matching `"grooved surface event limit exhausted"`
      rather than silently classifying the electron stopped.
- [ ] `test_reentry_resumes_material_stopping_and_elastic_scattering` — after
      re-entry, continuous energy loss and elastic scattering (direction
      change) resume normally in material.
- [ ] `test_repeated_zero_length_surface_events_raise` — back-to-back
      near-zero-length surface events raise `RuntimeError` matching
      `"repeated zero-length grooved surface events"`.
- [ ] `test_layer_and_back_face_events_precede_far_surface` — a layer
      boundary / back-face prism-exit event takes precedence over a distant
      groove surface event; multi-layer transmission still works correctly.
- [ ] `test_finite_side_exit_before_reentry_records_no_vacuum_leg` — a finite
      transverse crystal footprint causing a side exit during the vacuum leg
      (before material re-entry) is classified `n_side_exited`, with no
      vacuum leg recorded.

## Implementation checklist

- [ ] Re-derive, per test, the real `groove_spec` (spacing/depth/tilt),
      density, thickness, `E_cut_keV`, `max_steps`, and beam geometry that
      deterministically (`seed=9`) reproduces the event sequence the original
      mock forced. A scratch probe
      (`${HOME}/dev/cxr-mc/tests/montecarlo/test_groove.py`-adjacent, see prior
      session's `probe_groove.py`) showed the existing `_one_electron_transport`
      default kwargs (straight-vertical beam, shallow spec) pass through with
      zero groove interaction — non-default geometry/kwargs are needed per
      scenario.
- [ ] Rewrite each test body; keep the existing `_one_electron_transport`
      helper if still useful, otherwise adjust it.
- [ ] Run `tests/montecarlo/test_groove.py` in full (not just the 8) to confirm no
      regression in the other 25 already-passing tests.
- [ ] Run project lint/typecheck on the touched file.
- [ ] Scoped diff review: only `tests/montecarlo/test_groove.py` should change unless a
      stop condition below is hit.

## Decisions and open questions

- Rejected reintroducing a non-jit legacy core purely for test injectability
  — fights the intent of the JIT rework and reintroduces dead production
  code. Black-box testing against the real shipped path is more honest
  coverage.
- Open: whether every one of the 8 scenarios (especially the exact-tie /
  zero-length-event edge cases) is reachable via real geometry within a
  reasonable `max_steps`/`max_segments` budget, or whether one or two need a
  documented "closest reachable approximation" instead of an exact
  reproduction. If a scenario turns out to be unreachable black-box, stop and
  report rather than guessing at intent.
