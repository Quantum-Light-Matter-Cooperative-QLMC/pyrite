# Slow CPU test audit

The October 2026 audit for [#646](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/646)
separates dense-eigensolve oversubscription from test-memory pressure. Neither
requires weaker scientific assertions or a physics-model change.

## Baseline and data identity

The current baseline is `72087d7784c64d4855ba11f170724495b0c522b0`
(0.20.0), on `qlmc` / `DESKTOP-QNIHO3D`: AMD Ryzen 9 9950X3D,
32 logical CPUs, Python 3.14.6, NumPy 2.5.4 and pthreads OpenBLAS
0.3.34.106.0. Job 2493 requests 16 CPUs and 30 GiB, uses six xdist
workers with work stealing, and pins OpenBLAS to one thread per process.
Its recorded CPU affinity exposes all 32 logical CPUs; the scheduler CPU request
alone does not constrain that affinity.

Both the checkout workspace and Linux user-data stores were refreshed to
`tables-sbethe-4`, `tables-sbethe-positron-4`, `tables-elsepa-4`,
`tables-elsepa-positron-4` and `tables-bremslib-3`. `pyrite tables verify`
passes all 185 default pins. Fetch also installs the 84 positron SBETHE and
55 positron ELSEPA tables. Pytest clears the workspace/catalog environment and
isolates configuration, so workspace-only verification is insufficient: refresh
the user-data store before a remote test audit. Sync all tracked test and
repository assets as well as the runtime code.

The earlier runs 2345 and 2491 failed because of missing tables in the tier
pytest actually selected (and stale repository/configuration assets in 2345).
Their suite timings are excluded from the corrected baseline. The box's normal
config still names an old catalog; audit jobs use an empty `XDG_CONFIG_HOME`
and explicit catalog/workspace settings without editing that personal config.

## Corrected full-run results

Job 2493 passed: **10,009 passed, 271 skipped, 10 xfailed in 200.32 s**.
All setup/call/teardown durations and total-duration JUnit records were saved.
Per-test sampled protocol wall-time percentiles were p50 0.0052 s,
p90 0.0816 s, p95 0.3015 s, p99 2.0581 s; maximum 67.3034 s.
One full repetition was needed to replace the contaminated baseline; this is
not a repeated suite-level performance comparison. It follows exploratory full
runs, so filesystem and Numba caches are warm/retained, not reset.

| Largest call contributors | Call (s) |
|---|---:|
| `test_doc_blocks.py::test_python_api_workflow_executes_as_one_accumulating_namespace` | 67.30 |
| `test_coherent_windowed_line_grid.py::test_automatic_full_axis_yield_audit_uses_the_production_capture` | 45.37 |
| Axial main line `[110-0-cue_et_al]` | 19.65 |
| `test_continuum_floor.py` built-profile band, mose2 / wse2 | 16.31 / 15.89 |
| `test_mev_full_track.py` Geant4 gate, si_30mev | 15.15 |
| `test_angular_heatmap_app.py` default refinement/history normalization | 14.07 |

Worker busy times range from 168.78 to 191.10 s; the first worker finishes
21.16 s before the last. The capped-thread run has no multi-minute axial tail.
The two-second process-tree sampler peaks at 12.51 GiB summed RSS.

## Axial tail

Recovered job 2366 ran the two named axial tests (three parametrized cases)
with unchanged inputs, `max_order=16` (1089 basis waves), and fixed CPU affinity
`0-15`. Each run starts fresh Python processes; the serial process can reuse its
line cache across cases, while xdist caches are private to each worker. OS and
Numba caches were not flushed. These are whole-command wall times including
interpreter/pytest/worker startup and the RSS probe:

| Execution | BLAS threads/process | Repetition 1 (s) | Repetition 2 (s) |
|---|---:|---:|---:|
| Serial | 1 | 21.79 | 21.67 |
| Serial | 2 | 13.19 | 13.18 |
| Serial | 4 | 11.16 | 9.11 |
| Serial | 8 | 7.00 | 6.98 |
| Serial | 16 | 8.28 | 8.34 |
| Two workers | 1 | 24.71 | 23.91 |
| Two workers | 8 | 8.44 | 8.39 |
| Two workers | 16 | 649.51 | interrupted |
| Three workers | 1 | 25.86 | not run |
| Three workers | 16 | timeout at 1500 | not run |

The old benchmark checkout was `803995fd`; the named test, axial-state solver
and line-catalog implementation are unchanged between it and the current
baseline. It is retained as a separate experiment, not pooled with current-code
repetitions. The environment was not fully captured for that older run.

With two workers and 16 threads, the two reference calls took 645.65 and
641.96 seconds, versus a 2.01-second ordering case. This reproduces a long
end-of-suite tail under excess runnable BLAS threads, consistent with the
laptop's `numpy.linalg.eigh` stack samples. It does not prove that every minute
of the interrupted laptop run had the same cause. Cumulative worker CPU totals
are not individual test timings.

Current-code job 2494 repeats the identical three cases in fresh processes
on the same node (32 CPUs visible; 16 requested). All eight runs pass:

| Execution | BLAS threads/process | Repetition 1 / 2 wall (s) |
|---|---:|---:|
| Serial | 1 | 21.11 / 20.85 |
| Serial | 8 | 6.86 / 6.82 |
| Two workers | 1 | 24.11 / 24.10 |
| Two workers | 8 | 9.74 / 8.99 |

This confirms the serial/multiworker tradeoff without pooling the different
affinity conditions with job 2366. The current single-catalog diagnostic
(Si axis 110, 2.78 MeV, order 16) measures:

| Phase | One BLAS thread (s) | Eight threads (s) |
|---|---:|---:|
| Profile / extrema | 0.048 / 0.002 | 0.085 / 0.002 |
| One axial-state construction + solve | 0.467 | 0.122 |
| Independent 1089-square random symmetric eigensolve | 0.087 | 0.030 |
| Line-catalog assembly, including its own solves | 1.392 | 0.383 |
| Line yields | 0.003 | 0.003 |

The axis-111 catalog is 1.384 / 0.361 s. Diagnostic peaks are 429–435 MiB.
Separate cProfile runs find five `eigh` calls taking 1.875 / 0.476 s of own
time; four `axial_states` calls take 1.832 / 0.496 s cumulative, and the catalog
1.379 / 0.383 s cumulative. These overlapping diagnostic totals include the
extra standalone state solve and random eigensolve; do not sum them or treat
them as a partition of the reference tests. Eigensolves dominate this profile;
profile/extrema construction and yield accumulation are small. No solver
rewrite or reduced convergence order is warranted by these measurements.

For parallel CPU audits, start with `OPENBLAS_NUM_THREADS=1`, then measure larger
counts within the actual CPU budget. Eight threads improved this isolated
workload at two workers; it is not a safe blanket choice for six workers.
No default runner thread policy or eigensolver has been changed by this audit.

## Memory change

`test_channeling_states.py::test_presets_are_normalized_in_momentum_space[airy-2d]`
materialized the full complex amplitude tensor, then its density. Its fine
phase-resolution quadrature caused a multi-GiB transient even though the test
only returns a scalar norm. The test helper now evaluates 64 columns at a time,
contracts the first axis and retains only the marginal for the second axis.
Every original node, weight, preset and normalization tolerance is retained;
only floating-point reduction grouping changes. Production state amplitudes
and their solvers are untouched.

Fresh isolated processes, one BLAS thread, two repetitions each:

| Version | Test protocol wall (s) | Sampled process peak (GiB) |
|---|---:|---:|
| Original full tensor | 2.1089 / 0.6112 | 4.743 / 4.721 |
| Blocked integration | 0.4096 / 0.4213 | 0.3104 / 0.3103 |

Peak RSS falls about **93.4%**. The first baseline repetition has extra cold-use
cost; the measurements do not justify claiming a fivefold steady-state speedup.
All 47 state tests pass unchanged in 3.34 s, including the 12 normalization
presets and their original tolerances.

### Other memory candidates

These are worker peaks from the passing full run, not independent isolated
allocation measurements. The starting RSS shows inherited worker memory.
Keep these nodes remote, or cap any necessary isolated local reproduction.

| Node (under `tests/`) | Start / peak (GiB) |
|---|---:|
| `montecarlo/test_channeling_states.py` normalization `[airy-2d]`, before fix | 1.18 / 5.62 |
| `energy-grid/test_coherent_windowed_line_grid.py::test_windowed_auto_matches_a_fine_explicit_grid_on_identical_trajectories` | 1.25 / 4.60 |
| `dev/test_doc_blocks.py::test_python_api_workflow_executes_as_one_accumulating_namespace` | 0.53 / 3.14 |
| `energy-grid/test_coherent_windowed_line_grid.py::test_the_production_reducer_leaks_less_than_the_window_bound[hkl1]` | 1.18 / 2.84 |
| `montecarlo/test_channeling_directional.py::test_basal_kernel_depends_on_which_axis_carries_c` | 1.53 / 2.53 |

Airy is a demonstrated memory-pressure candidate and is fixed here. Historical
laptop OOM attribution cannot be proved without its OOM/cgroup logs; no such
logs were supplied. The remaining ranked cases are documented for safe routing,
not changed or weakened.

## Reproduction and interpretation

[The probe README](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/tree/main/checks/slow_test_audit) describes environment
setup, the full durations/JUnit/RSS command, isolated `/usr/bin/time -v` runs,
controlled BLAS/xdist comparisons and the axial decomposition/profile commands.
Run them in remote SLURM allocations, sequentially, with the same inputs and
cache policy. Preserve failed runs rather than presenting them as clean timings.

Raw evidence lives at `qlmc:~/pyrite-646/out/<job-id>/`: 2493 is the corrected
full baseline, and 2494 holds memory-change validation, current axial repeats
and profiles. Job 2492 was cancelled while pending to correct the data preflight.
The [checked-in summary](data/slow-test-audit-2026-10-24.json) includes the environment, distribution, phase timings, memory ranking and worker totals. RSS sampling includes retained
worker memory and can miss transient allocations; process-tree RSS double-counts
shared pages and is not a cgroup memory measurement. A high ranked worker peak
names a candidate for isolation, not proof of historical laptop OOM causation.
