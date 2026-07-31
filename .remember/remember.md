# Write next handoff to: /home/alexa/dev/cxr-mc/.remember/remember.md

## 13:30 PDT | feature/compute-performance-optimization

**Landed + pushed: 3e6e944** perf(spectrum): fuse batched line launch-storm.
On origin. Box (qlmc) NOT yet synced — still pre-fusion (grep _line_kin_core = 0).

### What 3e6e944 does (spectrum.py batched incoherent line path only)
- `_line_kin_core` (xp.fuse): steps 1+4 resonance freq + photon kinematics,
  ~15 elementwise launches -> 1. `x**2` spelled `x*x`, exact op order. BIT-FOR-BIT.
- `_interp_index` + `_interp_gather2d`/`_interp_gather1d`: chi/U re+im + mu all
  sample same E_res on same E_tab_g grid; bracket ONCE, gather per table off shared
  index (GCOL hoisted). Kills 4x redundant searchsorted/clip. Bit-for-bit vs old
  `_batch_interp`/`_interp1` blend.
- `_line_weight_core` (xp.fuse): step 6+7 escape factor + PXR prefactor, T_abs
  folded in. `/_PREF_C1` division + `t_L*t_L` preserved. BIT-FOR-BIT.
- Removed orphaned `_batch_interp`; `_interp1` kept (per-hkl path).
- NO new validation debt (pure kernel-merge). Existing `Validation: line-hkl-batch`
  marker intact at spectrum.py:~907.
- Verify done: 122/122 line goldens (chunk-invariance, montecarlo, mosaic,
  multilayer, coherent, faceting), lint + typecheck clean.

### To MEASURE the fusion (user drives qlmc submission)
push already done -> sync box to 3e6e944 -> rerun nsys profile -> pull, compare
launch count vs baseline below.

### Baseline = performance-profiles/hopg_test-2/ (PRE-fusion, hopg 189 cases)
- wall 29.66s, transport 66.4s agg (/6 workers ~11s effective), spectrum 10.25s,
  feed-wait 15.6%.
- cuLaunchKernel TOTAL = 82,457. Storm confirmed: cupy_multiply 10,474 +
  cupy_add 8,443 + take 7,448 + where 4,914 + prepare_array_indexing 4,580 +
  searchsorted 1,195 + clip 1,195. Fusion collapses the multiply/add/subtract/
  divide + 4x redundant bracket. Post-fusion expect launch count to drop sharply.
- NVTX: cxr.lines 4.43s (tab 0.59 + accum 0.44 + setup ~3.4s launch-bound=target),
  cxr.brem 5.38s (COMPUTE-bound, leave), cxr.interpolate small.
- HOPG has few reflections -> cxr.lines small here (mos2 was ~18s). Fusion wall
  impact modest on hopg; bigger on mos2/high-reflection materials.

### SURPRISE next lever (from hopg_test-2 baseline)
**checkpoint = 16.42s on a 29.66s wall (~55%).** Serial I/O, not GPU. Bigger single
win than line fusion on this profile. Not yet investigated. (checkpoint_count 45,
checkpoint_seconds_total 16.42 in hopg.latest.json.) Transport 66.4s agg also large.

### User WIP restored in working tree (uncommitted, THEIRS, do not touch)
git stash was popped (user had stashed pre-rebase, aborted, forgot):
- src/cxr_mc/data/materials.toml: adds [profiles.hopg_test] + [profiles.mos2_test]
  (energy_keV=[30,60,100,200,300], tilt [15,45,75], azim [105,135,165], 5x5mm).
- src/cxr_mc/montecarlo/runner.py: NVTX cxr.transport.line / cxr.transport.brem
  around the two simulate_trajectories calls in _transport_case. NOTE: earlier I
  wrongly called this a no-op; the WIP comment is right — nsys traces the
  exec-based workers, so the ranges DO fire. Legit.
- src/cxr_mc/scan.py: --quick grid tweaks.

### Deferred physics debt on this branch (unchanged, needs ledger+regen+human signoff)
line-absorption-tabulation, line-hkl-batch, line-amplitude-fusion,
line-gemv-elementwise. The 3e6e944 fusions add NOTHING new (bit-for-bit).
