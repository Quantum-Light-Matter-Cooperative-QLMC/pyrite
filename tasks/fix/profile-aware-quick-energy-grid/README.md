# Profile-aware quick energy grids

Branch: `fix/profile-aware-quick-energy-grid`

## Problem and scope

`cxr run PROFILE --quick` currently replaces every profile's beam-energy grid
with `[30, 50]` keV. Named profiles may own an
`E_grid_line_by_energy` mapping without 50 keV. Case construction then fails
before compute with:

```text
ValueError: no E_grid_line configured for beam energy 50 keV
```

The failure blocks ordinary quick runs and the bounded CPU cProfile phase used
by `cxr remote run PROFILE --cpu` / `--cpu-only`, because that phase always
adds `--quick`. It was reproduced remotely with `hopg_test --quick --cpu` and
`compute_test_300keV --cpu`; a `sub_100keV` profile containing 30 and 50 keV
completed both GPU and CPU phases.

Fix quick-grid resolution, not physical line-grid validation. Keep CPU
profiling bounded and uncached. Do not add missing photon grids silently,
weaken `sweep._line_grid_for_energy`, change full/survey grids, or change
physics, seeds, detector settings, and checkpoint identity outside the resolved
quick-energy subset.

## Implementation path and likely owners

- `src/cxr_mc/scan.py`: make `_resolved_run` choose a small beam-energy subset
  compatible with the effective material/profile line grids. Preserve the
  standard profile's current `[30, 50]` quick behavior where both energies are
  valid.
- `src/cxr_mc/sweep.py`: retain the existing missing-grid error as the final
  invariant; no fallback grid belongs here.
- `src/cxr_mc/_remote/scripts.py`: keep the CPU cProfile phase's `--quick`,
  `--workers 0`, isolated checkpoint, no sampler, and 10-minute cap. Change
  only if command construction must expose the resolved quick selection.
- `tests/test_local_click_cli.py` or a narrowly owned scan test: cover quick
  resolution for profiles with and without 50 keV.
- `tests/test_remote.py`: preserve generated-script CPU profiling contracts and
  add coverage proving a profile without 50 keV can reach a valid bounded CPU
  command path.
- `docs/cli-reference.md` and `tests/data/cli_contract.json`: regenerate only if
  user-facing CLI text or options change.

## Stepwise checklist

- [ ] Encode the desired quick-energy selection policy in focused tests before
      changing resolution.
- [ ] Resolve quick energies from the effective catalog profile and material,
      selecting only energies with valid line grids.
- [ ] Preserve deterministic ordering, bounded case count, dataset identity,
      and the standard profile's valid `[30, 50]` behavior.
- [ ] Keep `build_cases` fail-closed for genuinely inconsistent energy/grid
      inputs.
- [ ] Verify local quick resolution for `hopg_test`,
      `compute_test_300keV`, `sub_100keV`, and `standard` without running heavy
      Monte Carlo locally.
- [ ] Run focused scan/profile/remote script and CLI-contract tests.
- [ ] Exercise one authorized remote combined quick GPU/CPU profile that lacks
      50 keV in its normal grid; confirm terminal success, pull, and analysis.
- [ ] Document any intentional quick-selection semantic change and regenerate
      generated CLI artifacts if required.

## Decisions and open questions

Decided:

- Quick mode remains a bounded representative workload; CPU profiling must not
  fall back to a full serial campaign.
- Compatibility belongs in profile-aware quick resolution. Missing-grid
  validation remains strict downstream.
- One fix covers local `--quick`, combined `--cpu`, and `--cpu-only`; do not add
  a CPU-only special-case photon grid.

Open for implementation review:

- When nominal 30/50 keV quick energies are unavailable, select the first two
  configured energies, the nearest valid energies to 30/50, or another stable
  bounded policy. The choice must be deterministic, documented, and retain
  enough energy diversity for representative profiling.
- Whether a profile with only one valid energy should run that one energy or
  fail with a clearer profile-validation error.
- Whether quick selection should be a reusable profile helper or stay local to
  `scan._resolved_run` until another caller needs it.

## Delegation and required skills

- Worker: `implement-task`; bounded cross-cutting behavior with runtime and
  generated-script regressions.
- Required: `cli-ui-ux`, `performance`, `remote-gpu-jobs`,
  `regression-testing`, `run-cxr-mc`.
- Add `documentation-maintenance` only if help/reference semantics change.
- No physics-review or Monte Carlo-kernel work expected; stop if the proposed
  fix changes physical grids or kernels.

## Acceptance checks

- `cxr run hopg_test -m hopg --quick` and
  `cxr run compute_test_300keV -m mos2 --quick` resolve valid bounded case sets
  instead of requesting an absent 50 keV line grid.
- `cxr run standard -m hopg --quick` retains valid 30/50 keV behavior.
- Generated remote `--cpu` and `--cpu-only` scripts remain serial, uncached,
  bounded, phase-aware, and fail terminally on real CPU-profiler errors.
- An authorized remote combined quick GPU/CPU run for a formerly failing
  profile completes both phases and produces GPU NDJSON plus `.cpu.prof` and
  `.cpu.txt` artifacts.
- `sweep._line_grid_for_energy` still rejects genuinely missing per-energy
  grids with the exact selected energy in the error.
- Focused scan/profile/remote tests, shell-script syntax checks, and any touched
  generated CLI contracts pass.

## Stop conditions

Stop on incompatible profile semantics, non-deterministic energy selection,
changed full/survey physics grids, checkpoint identity collision, a need to run
heavy Monte Carlo locally, or unrelated dirty work.
