# Slow-test audit

Linux-only instrumentation for #646. Run heavy work in a bounded SLURM allocation,
from a clean, synced checkout, with `uv sync --all-groups --locked`. The probes
use `psutil` and `threadpoolctl` from that environment. Do not run the full audit
on the laptop.

`rssprobe.py` samples the current test worker every 10 ms, including setup and
teardown, and its child tree every 100 ms. `peak_rss` includes retained memory
from earlier tests; compare `start_rss` before attributing an allocation. Sampling
can miss short-lived peaks. `ru_maxrss` is a process lifetime high-water mark,
not a per-test measurement. Fresh isolated runs under `/usr/bin/time -v` provide
an independent peak measurement. Summing process RSS counts shared pages more
than once; the tree total is not unique physical memory or SLURM's cgroup peak.

Before testing, sync **all tracked files**, not only `pyrite remote sync`'s
runtime subset: the suite reads tests, documentation, workflow and tooling files.
The root conftest removes `PYRITE_HOME` and `PYRITE_CATALOG`, and isolates the
config store. Fetch into the user-data tier it actually reads as well as the
synced workspace. On the current Linux box:

```bash
export PATH="$HOME/.local/bin:$PATH"
export XDG_CONFIG_HOME="$PWD/.audit-config"
mkdir -p "$XDG_CONFIG_HOME"
PYRITE_HOME="$HOME/.local/share/pyrite" \
  PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" uv run pyrite tables fetch
env -u PYRITE_HOME -u PYRITE_CATALOG uv run pyrite tables verify
```

Save the sync stamp, environment (Python, NumPy, `threadpool_info()`, CPU
model/affinity), table verification JSON, scheduler job ID and allocation size.
For an unmodified full baseline:

```bash
export OPENBLAS_NUM_THREADS=1
export RSSPROBE_DIR="$PWD/audit-output/rss"
mkdir -p "$RSSPROBE_DIR"
/usr/bin/time -v uv run pyrite-dev test --slow \
  -p checks.slow_test_audit.rssprobe --durations=0 --durations-min=0 \
  --junitxml=audit-output/junit.xml -o junit_duration_report=total \
  > audit-output/pytest.log 2> audit-output/time.err &
test_pid=$!
uv run python checks/slow_test_audit/treemon.py "$test_pid" audit-output/tree.csv &
monitor_pid=$!
wait "$test_pid"
rc=$?
wait "$monitor_pid"
# Preserve rc in the job's exit status.
exit "$rc"
```

Repeat an isolated node in fresh processes with `/usr/bin/time -v`. For the axial
comparison, use identical node selectors with `-p no:xdist`, then `-n 2 --dist
worksteal`, and compare `OPENBLAS_NUM_THREADS=1`, `8`, and `16`. Keep CPU affinity
and all inputs fixed. The high-thread arm may need a timeout; report it as
censored, never as a completed timing. Each process/worker starts with an empty
Python line cache; later tests assigned to it may reuse that cache. OS and Numba
caches are not flushed.

Decompose one catalog in a fresh process (axis, energy in MeV, optional order):

```bash
OPENBLAS_NUM_THREADS=1 uv run python checks/slow_test_audit/axial_probe.py 110 2.78 16
OPENBLAS_NUM_THREADS=1 uv run python -m cProfile -o axial.prof \
  checks/slow_test_audit/axial_probe.py 110 2.78 16
uv run python -m pstats axial.prof
```

`axial_probe.py` times a profile, extrema, one axial-state solve, an independent
random symmetric eigensolve (seed 0), a line catalog, and its line yields in that
order. These diagnostics do not partition a test's runtime: the catalog performs
its own solves, and the earlier calls can warm internal caches. Keep profiling
runs separate from timing repetitions.

`summarize.py AUDIT_DIR` produces duration percentiles, top tests, worker busy
intervals and sampled process-tree peak. It expects `environment.json`,
`sync-stamp`, `tables.json`, `junit.xml`, `pytest.log`, `tree.csv` and `rss/*.jsonl`.
Retain the raw artifacts; a failed environment run is not a passing baseline.
