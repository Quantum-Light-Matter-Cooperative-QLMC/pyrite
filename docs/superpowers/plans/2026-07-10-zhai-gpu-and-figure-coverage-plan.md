# Zhai GPU Reproduction + Supplementary Figure Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Zhai reproduction + WSe₂/MoSe₂/h-BN supplementary Monte
Carlos run on the GPU lab box instead of the laptop (`cxr remote check`),
render the whole publication figure set from the pulled caches in one command
(`cxr check --export`), add a cross-material overview figure, and document
the supplementary studies' provenance (including the open azimuth = 0
question) — closing out the spec at
`docs/superpowers/specs/2026-07-10-zhai-gpu-and-figure-coverage-design.md`.

**Architecture:** `checks/anchor_figures.py` gains two pure-orchestration
functions (`reproduce_all` — populate every cache; `export_all_figures` —
render every figure from cache) plus one new figure (`figure_supplementary_overview`).
A new root shim `reproduce_zhai.py` (mirrors `scan.py`) is the box-invokable
entry point. `src/cxr_mc/remote.py` gains a `check` subcommand that reuses the
existing sync/queue/pull machinery. `src/cxr_mc/check.py` gains `--export`.
No new physics: everything here calls existing ledgered functions
(`cached_model_spectra`, `cached_coherent_spectra`, `mc_spectrum`, etc.)
through existing entry points — no ledger rows change.

**Tech Stack:** Python, pytest, argparse, matplotlib (`Agg` backend in tests),
ssh/scp (Windows OpenSSH via Git Bash), existing `_checkpoint_io` pickle
format.

## Global Constraints

- Work happens on branch `feature/zhai-gpu-reproduction` (created in Task 1,
  step 1) — never commit this work directly to `main`.
- Test command (per repo convention — `-q` in `addopts` hides the pass/fail
  summary): `uv run pytest -p no:cacheprovider path/to/test.py -v`
- Lint: `uv run ruff check .` — Format: `uv run ruff format .` — Type check:
  `uv run pyright`. Run all three before each task's commit.
- **Import-then-use in one edit**: this repo's ruff pre-commit hook deletes
  *new* imports it considers unused if added in a separate commit from their
  first use — always add an import in the same edit as the code that uses it.
- No new physics lands here: every Monte Carlo call in this plan goes through
  functions the ledger already covers (`docs/physics-validation-ledger.md` is
  untouched). No `Validation: <id>` markers needed.
- `notebooks/validation_app.py` is **not modified** by this plan — the new
  `--export` path is a separate CLI entry, not a notebook cell.
- If executed inside an isolated git worktree, `uv` can re-point the shared
  `.venv` — use `uv run --active --no-sync` (see repo's own `uv` gotcha notes)
  rather than plain `uv run` when working from a worktree.
- Windows CRLF: this repo's `.gitattributes` pins `* text=auto eol=lf`; new
  files should be created with LF line endings (the editor tools used here
  write LF by default — do not run any CRLF-normalizing step over `src/` or
  `checks/`).

---

### Task 1: Create branch + `reproduce_all()` cache-populator

**Files:**
- Create: (branch) `feature/zhai-gpu-reproduction`
- Modify: `checks/anchor_figures.py` (add `reproduce_all`, after
  `cached_coherent_spectra` at line 524, before the `# ---- optional digitized
  reference` comment at line 527)
- Modify: `TODO.md` (branch-scoped detail; see step 8)
- Test: `tests/test_anchor_figures.py` (append)

**Interfaces:**
- Consumes: `ZhaiAnchor`, `ZHAI_SUPPLEMENTARY_STUDIES`, `cached_model_spectra(anchor, ne=500, ne_brem=200, *, cache_dir=None, refresh=False) -> tuple[dict, bool, Path]`, `cached_coherent_spectra(study, thickness_nm, ne=500, *, cache_dir=None, refresh=False) -> tuple[dict[float, np.ndarray], bool, Path]` (all already in `checks/anchor_figures.py`)
- Produces: `reproduce_all(ne=20_000, ne_brem=200, ne_supp=200, *, cache_dir=None, refresh=False) -> list[tuple[str, Path, bool]]` — labels are `"zhai-fig1c"` then `f"{crystal}-{thickness_nm:g}nm"` for each of the 7 supplementary `(study, thickness)` pairs (8 entries total). Later tasks (2, 5) call this and its `export_all_figures` sibling.

- [ ] **Step 1: Create and check out the branch, scope TODO.md**

```bash
git checkout -b feature/zhai-gpu-reproduction
```

Edit `TODO.md`: replace the `# USER ADDED:` Zhai bullet (the one starting
"Zhai supplementary coherent-emission figures...") with the branch-scoped
detail below, leaving the "Implement automated ACP server startups..." bullet
untouched:

```markdown
- Zhai supplementary coherent-emission figures (WSe₂ 42/55/75 nm, MoSe₂
  47/112/147 nm, h-BN 921 nm, all 200 keV, at polar tilts -10/-15/-17.5/-20 deg)
  are implemented in `checks/anchor_figures.py` + the validation app. This
  branch (`feature/zhai-gpu-reproduction`) adds: a `cxr remote check` command
  to run the Zhai + supplementary Monte Carlos on the GPU lab box (`ne=20_000`
  Fig.1c anchor, `ne=200` per supplementary tilt) and pull the caches back; a
  `cxr check --export` command to render the complete figure set from cache
  in one shot; a cross-material overview figure; and a provenance write-up
  covering the still-open azimuth = 0 assumption. Design:
  [`docs/superpowers/specs/2026-07-10-zhai-gpu-and-figure-coverage-design.md`](docs/superpowers/specs/2026-07-10-zhai-gpu-and-figure-coverage-design.md).
```

```bash
git add TODO.md
git commit -m "docs(todo): scope feature/zhai-gpu-reproduction branch"
```

- [ ] **Step 2: Write the failing test for `reproduce_all`**

Append to `tests/test_anchor_figures.py`:

```python
def test_reproduce_all_populates_every_cache_and_reuses_it(tmp_path, monkeypatch):
    calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        calls.append(("fig1c", ne, ne_brem))
        return {"peak": True}

    def fake_coherent(study, thickness_nm, ne):
        calls.append((study.crystal, thickness_nm, ne))
        return {tilt: np.zeros(4) for tilt in study.polar_tilts_deg}

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    monkeypatch.setattr(af, "model_coherent_spectra", fake_coherent)

    results = af.reproduce_all(ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path)

    assert len(results) == 8
    labels = [label for label, _path, _hit in results]
    assert labels[0] == "zhai-fig1c"
    assert "wse2-42nm" in labels and "mose2-147nm" in labels and "hbn-921nm" in labels
    assert all(path.exists() for _label, path, _hit in results)
    assert all(hit is False for _label, _path, hit in results)  # first run: no cache hits
    assert len(calls) == 8

    calls.clear()
    results2 = af.reproduce_all(ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path)

    assert calls == []  # fully cache-hit; no MC re-run
    assert [label for label, _p, _h in results2] == labels
    assert all(hit for _label, _path, hit in results2)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py::test_reproduce_all_populates_every_cache_and_reuses_it -v`
Expected: FAIL with `AttributeError: module 'anchor_figures' has no attribute 'reproduce_all'`

- [ ] **Step 4: Implement `reproduce_all`**

Insert into `checks/anchor_figures.py` after `cached_coherent_spectra` (after
line 524, before the `# ---- optional digitized reference` section comment at
line 527):

```python
def reproduce_all(
    ne: int = 20_000,
    ne_brem: int = 200,
    ne_supp: int = 200,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> list[tuple[str, Path, bool]]:
    """Force-populate every Zhai cache the validation app can hit, at the
    app's own default sample counts so a pulled cache is a guaranteed hit
    locally.

    No figures -- this only leaves correct, hash-addressed .pkl files on disk
    under ``cache_dir`` (default checkpoints/zhai_reproduction/). This is the
    GPU-box-runnable unit behind ``reproduce_zhai.py`` / ``cxr remote check``.

    Returns [(label, path, cache_hit)] for the Fig.1c anchor plus every
    supplementary (study, thickness) pair -- 8 entries total.
    """
    results: list[tuple[str, Path, bool]] = []
    anchor = ZhaiAnchor()
    _, hit, path = cached_model_spectra(
        anchor, ne=ne, ne_brem=ne_brem, cache_dir=cache_dir, refresh=refresh
    )
    results.append(("zhai-fig1c", path, hit))
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        for thickness_nm in study.thicknesses_nm:
            _, hit, path = cached_coherent_spectra(
                study, thickness_nm, ne=ne_supp, cache_dir=cache_dir, refresh=refresh
            )
            results.append((f"{crystal}-{thickness_nm:g}nm", path, hit))
    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py::test_reproduce_all_populates_every_cache_and_reuses_it -v`
Expected: PASS

- [ ] **Step 6: Lint, format, type check**

```bash
uv run ruff check checks/anchor_figures.py tests/test_anchor_figures.py
uv run ruff format checks/anchor_figures.py tests/test_anchor_figures.py
uv run pyright checks/anchor_figures.py
```

- [ ] **Step 7: Commit**

```bash
git add checks/anchor_figures.py tests/test_anchor_figures.py
git commit -m "feat(checks): add reproduce_all cache-populator for Zhai + supplementary studies"
```

---

### Task 2: Root shim `reproduce_zhai.py`

**Files:**
- Create: `reproduce_zhai.py` (repo root, mirrors `scan.py`)
- Test: Create `tests/test_reproduce_zhai.py`

**Interfaces:**
- Consumes: `reproduce_all(ne, ne_brem, ne_supp, *, cache_dir, refresh) -> list[tuple[str, Path, bool]]` (Task 1)
- Produces: `reproduce_zhai.main(argv=None)` — CLI entry the box runs as `python reproduce_zhai.py [--ne N] [--ne-brem N] [--ne-supp N] [--refresh] [--cache-dir DIR]`. Consumed by `remote.py`'s `remote_check`/`_zhai_queue_script` (Task 3), which shell out to this file by name.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reproduce_zhai.py`:

```python
"""Tests for the reproduce_zhai.py root shim (the box-invokable entry point
for cxr remote check) -- argument parsing and CLI wiring only; the actual MC
work is reproduce_all, tested in tests/test_anchor_figures.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import reproduce_zhai  # noqa: E402


def test_cli_defaults_match_app_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(
        reproduce_zhai, "reproduce_all", lambda **kw: calls.append(kw) or []
    )

    reproduce_zhai.main([])

    assert calls == [
        {"ne": 20_000, "ne_brem": 200, "ne_supp": 200, "cache_dir": None, "refresh": False}
    ]


def test_cli_forwards_overrides(monkeypatch, capsys):
    calls = []

    def fake_reproduce_all(**kw):
        calls.append(kw)
        return [("zhai-fig1c", Path("checkpoints/zhai_reproduction/zhai-abc.pkl"), True)]

    monkeypatch.setattr(reproduce_zhai, "reproduce_all", fake_reproduce_all)

    reproduce_zhai.main(
        ["--ne", "11", "--ne-brem", "3", "--ne-supp", "5", "--refresh", "--cache-dir", "/tmp/x"]
    )

    assert calls == [
        {"ne": 11, "ne_brem": 3, "ne_supp": 5, "cache_dir": "/tmp/x", "refresh": True}
    ]
    out = capsys.readouterr().out
    assert "zhai-fig1c" in out and "cached" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_reproduce_zhai.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reproduce_zhai'`

- [ ] **Step 3: Write `reproduce_zhai.py`**

Create at repo root:

```python
"""Headless Zhai/supplementary Monte-Carlo cache populator -- thin shim to
checks/anchor_figures.py::reproduce_all.

Kept at the repo root so cxr_mc.remote (which runs ``python
reproduce_zhai.py`` on the GPU box) and muscle-memory ``python
reproduce_zhai.py`` keep working from a checkout with no install. Mirrors
scan.py's shape: sys.path shim + thin __main__ guard, real logic lives in
checks/. Populates checkpoints/zhai_reproduction/ with every cache the
validation app's Zhai sections can hit -- no figures, no display -- so a
later ``cxr remote check --pull`` (or a plain local ``cxr check``) sees an
instant cache hit.

Run (defaults match the validation app's own UI defaults, so a pulled cache
is guaranteed to hit locally):

    python reproduce_zhai.py
    python reproduce_zhai.py --ne 20000 --ne-brem 200 --ne-supp 200 --refresh
"""

import argparse
import os
import sys

_CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks")
if _CHECKS not in sys.path:
    sys.path.insert(0, _CHECKS)

from anchor_figures import reproduce_all  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="populate the Zhai reproduction cache")
    ap.add_argument(
        "--ne", type=int, default=20_000, help="Fig.1c anchor line electrons per energy"
    )
    ap.add_argument(
        "--ne-brem",
        type=int,
        default=200,
        help="Fig.1c anchor bremsstrahlung electrons per energy",
    )
    ap.add_argument(
        "--ne-supp", type=int, default=200, help="supplementary electrons per polar-tilt spectrum"
    )
    ap.add_argument(
        "--refresh", action="store_true", help="recompute even if a matching cache already exists"
    )
    ap.add_argument("--cache-dir", default=None, help="override the cache directory (testing)")
    args = ap.parse_args(argv)

    results = reproduce_all(
        ne=args.ne,
        ne_brem=args.ne_brem,
        ne_supp=args.ne_supp,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    for label, path, cache_hit in results:
        status = "cached" if cache_hit else "computed"
        print(f"{label:16s} {status:9s} {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_reproduce_zhai.py -v`
Expected: PASS

- [ ] **Step 5: Lint, format, type check**

```bash
uv run ruff check reproduce_zhai.py tests/test_reproduce_zhai.py
uv run ruff format reproduce_zhai.py tests/test_reproduce_zhai.py
uv run pyright reproduce_zhai.py
```

- [ ] **Step 6: Commit**

```bash
git add reproduce_zhai.py tests/test_reproduce_zhai.py
git commit -m "feat: add reproduce_zhai.py root shim for the box-runnable Zhai cache populator"
```

---

### Task 3: `cxr remote check` — sync, run on box, pull, detached queue

**Files:**
- Modify: `src/cxr_mc/remote.py`
- Modify: `tests/test_remote.py` (append)

**Interfaces:**
- Consumes: `reproduce_zhai.py` (Task 2, invoked by name over ssh — not
  imported), existing `remote.py` internals: `sync_code()`, `_run(cmd, **kw)`,
  `_ssh_capture(cmd) -> str`, `_refuse_if_busy(materials, quick)`,
  `_live_jobs() -> list[tuple[str, bool, list[str]]]`,
  `_launch_queue_command(jobid) -> str`, `HOST`, `REMOTE_DIR`, `REMOTE_UV`,
  `LOCAL_ROOT`, `JOBS_SUBDIR`, `SYNC_PATHS`
- Produces: `ZHAI_STEM = "zhai"` (module constant), `remote_check(ne=20_000,
  ne_brem=200, ne_supp=200, refresh=False, no_sync=False) -> None`,
  `pull_zhai_cache() -> None`, `_zhai_queue_script(jobid, ne, ne_brem,
  ne_supp, refresh) -> str`, `start_zhai_queue(ne=20_000, ne_brem=200,
  ne_supp=200, refresh=False, no_sync=False, dry_run=False) -> str` (job id).
  CLI: `cxr remote check [--ne N] [--ne-brem N] [--ne-supp N] [--refresh]
  [--no-sync] [--detached/-d [--follow/-f]] [--pull]`.

- [ ] **Step 1: Write the failing tests for the zhai queue script + start**

Append to `tests/test_remote.py`:

```python
# ---- cxr remote check (Zhai GPU reproduction) ------------------------------
def test_zhai_queue_script_has_ne_flags_and_meta():
    s = remote._zhai_queue_script("20260101-000000", ne=11, ne_brem=3, ne_supp=5, refresh=True)
    assert "reproduce_zhai.py" in s
    assert "--ne 11" in s and "--ne-brem 3" in s and "--ne-supp 5" in s and "--refresh" in s
    assert "materials: zhai" in s and "quick: False" in s
    assert "20260101-000000" in s


def test_zhai_queue_script_no_refresh_flag_when_unset():
    s = remote._zhai_queue_script("j", ne=1, ne_brem=1, ne_supp=1, refresh=False)
    assert "--refresh" not in s


def test_zhai_start_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.start_zhai_queue()


def test_zhai_start_dry_run_prints_without_ssh_or_sync(monkeypatch, capsys):
    monkeypatch.setattr(remote, "_live_jobs", lambda: pytest.fail("dry-run must not check busy"))
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        remote.subprocess, "run", lambda *a, **kw: pytest.fail("dry-run must not ssh")
    )

    jobid = remote.start_zhai_queue(dry_run=True)

    out = capsys.readouterr().out
    assert jobid in out and "reproduce_zhai.py" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -k zhai_queue_script_has_ne_flags -v`
Expected: FAIL with `AttributeError: module 'cxr_mc.remote' has no attribute '_zhai_queue_script'`

- [ ] **Step 3: Implement `ZHAI_STEM`, `_zhai_flags`, `_zhai_queue_script`, `start_zhai_queue`**

In `src/cxr_mc/remote.py`, insert after line 76 (`JOBS_SUBDIR = "jobs"`), before
the `_MATERIAL_RE` comment at line 78:

```python

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"
```

Insert after `_queue_script` (after line 276, before `def
_launch_queue_command` at line 279):

```python
def _zhai_flags(ne, ne_brem, ne_supp, refresh):
    flags = f" --ne {ne} --ne-brem {ne_brem} --ne-supp {ne_supp}"
    if refresh:
        flags += " --refresh"
    return flags


def _zhai_queue_script(jobid, ne, ne_brem, ne_supp, refresh):
    """The bash runner for a detached Zhai-reproduction job: same
    pid/meta/state bookkeeping as _queue_script, but runs reproduce_zhai.py
    once instead of looping scan.py over materials. The meta's `materials:
    zhai` / `quick: False` lines are what let _live_jobs/_refuse_if_busy/
    stop_jobs treat this job like any material-keyed one, keyed on ZHAI_STEM."""
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    flags = _zhai_flags(ne, ne_brem, ne_supp, refresh)
    return f"""#!/usr/bin/env bash
set -u
JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
echo $$ > "$JOBDIR/pid"
{{ echo "job: {jobid}"; echo "materials: {ZHAI_STEM}"; echo "quick: False"; \
echo "ne: {ne}"; echo "ne_brem: {ne_brem}"; echo "ne_supp: {ne_supp}"; \
echo "started: $(date -Is)"; echo "pid: $$"; }} > "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
echo "running zhai reproduction since $(date -Is)" > "$JOBDIR/state"
if ! {REMOTE_UV} run --no-sync python reproduce_zhai.py{flags} >> "$JOBDIR/log" 2>&1
then
  echo "FAILED $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
echo "done $(date -Is)" > "$JOBDIR/state"
"""
```

Insert after `start_queue` (after its closing `return jobid`, currently ending
around line 425, before `def list_jobs` at line 428):

```python
def start_zhai_queue(ne=20_000, ne_brem=200, ne_supp=200, refresh=False, no_sync=False, dry_run=False):
    """Launch a detached Zhai-reproduction job on the box (mirrors
    start_queue): sync code, write the runner, nohup setsid it. Returns the
    job id; pull results with `cxr remote check --pull` once state is 'done'."""
    if not dry_run:
        _refuse_if_busy([ZHAI_STEM], False)
    jobid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    script = _zhai_queue_script(jobid, ne, ne_brem, ne_supp, refresh)
    launch = _launch_queue_command(jobid)

    if dry_run:
        print(f"# zhai job {jobid}: ne={ne} ne_brem={ne_brem} ne_supp={ne_supp}")
        print(f"# --- ssh {HOST}: mkdir -p {jobdir} && cat > {jobdir}/run.sh <<\n")
        print(script)
        print(f"# --- ssh {HOST}: {launch}")
        return jobid

    if not no_sync:
        sync_code()
    subprocess.run(
        ["ssh", HOST, f"mkdir -p '{jobdir}' && cat > '{jobdir}/run.sh'"],
        input=script.replace("\r\n", "\n").encode(),
        check=True,
    )
    _run(["ssh", "-n", HOST, launch])

    print(
        f"\nstarted zhai job {jobid} on {HOST}\n"
        f"  watch:  cxr remote status {jobid}\n"
        f"  logs:   cxr remote logs {jobid} --follow\n"
        f"  pull:   cxr remote check --pull   (when state is 'done')"
    )
    return jobid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -k zhai -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing tests for foreground run + pull**

Append to `tests/test_remote.py`:

```python
def test_remote_check_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(remote, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    monkeypatch.setattr(remote, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote.remote_check()


def test_remote_check_syncs_runs_and_pulls(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])
    monkeypatch.setattr(remote, "sync_code", lambda: calls.append("sync"))
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: calls.append(("run", cmd)))
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))

    remote.remote_check(ne=11, ne_brem=3, ne_supp=5, refresh=True)

    assert calls[0] == "sync"
    assert calls[1][0] == "run"
    ssh_cmd = calls[1][1]
    assert ssh_cmd[:3] == ["ssh", "-n", remote.HOST]
    assert "reproduce_zhai.py" in ssh_cmd[3]
    assert "--ne 11" in ssh_cmd[3] and "--refresh" in ssh_cmd[3]
    assert calls[2] == "pull"


def test_remote_check_no_sync_skips_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "_live_jobs", lambda: [])
    monkeypatch.setattr(remote, "sync_code", lambda: calls.append("sync"))
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: calls.append("run"))
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))

    remote.remote_check(no_sync=True)

    assert calls == ["run", "pull"]


def test_pull_zhai_cache_fetches_every_listed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(
        remote, "_ssh_capture", lambda *a: "/r/checkpoints/zhai_reproduction/zhai-a.pkl\n"
        "/r/checkpoints/zhai_reproduction/zhai-b.pkl\n"
    )
    runs = []
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.pull_zhai_cache()

    assert len(runs) == 2
    assert runs[0][0] == "scp" and runs[0][1].endswith("zhai-a.pkl")
    assert (tmp_path / "checkpoints" / "zhai_reproduction").is_dir()


def test_pull_zhai_cache_reports_when_empty(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(remote, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(remote, "_ssh_capture", lambda *a: "")
    monkeypatch.setattr(remote, "_run", lambda cmd, **kw: pytest.fail("nothing to pull"))

    remote.pull_zhai_cache()

    assert "no zhai cache files" in capsys.readouterr().out
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -k "remote_check or pull_zhai" -v`
Expected: FAIL with `AttributeError: module 'cxr_mc.remote' has no attribute 'remote_check'`

- [ ] **Step 7: Implement `remote_check` and `pull_zhai_cache`**

Insert after `remote_scan` (after line 188, before `def pull` at line 191):

```python
def remote_check(ne=20_000, ne_brem=200, ne_supp=200, refresh=False, no_sync=False):
    """Sync code, run reproduce_zhai.py on the box (populating
    checkpoints/zhai_reproduction/ there), then pull every cache file back.
    Foreground: holds the ssh session open until the run finishes."""
    _refuse_if_busy([ZHAI_STEM], False)
    if not no_sync:
        sync_code()
    cmd = (
        f"cd {REMOTE_DIR} && {REMOTE_UV} run --no-sync python reproduce_zhai.py"
        f"{_zhai_flags(ne, ne_brem, ne_supp, refresh)}"
    )
    _run(["ssh", "-n", HOST, cmd])
    pull_zhai_cache()
```

Insert after `pull` (after its closing, currently ending at line 231, before
the `# ---- detached job queue` comment at line 234):

```python
def pull_zhai_cache():
    """Fetch every cache file under checkpoints/zhai_reproduction/ from the
    box.

    Lists remote filenames first (like clear_remote's listing step) rather
    than `scp -r`, which double-nests the directory when the local
    destination already exists -- listing + per-file scp is unambiguous
    either way."""
    remote_dir = f"{REMOTE_DIR}/checkpoints/zhai_reproduction"
    listing = f'[ -d "{remote_dir}" ] || exit 0; ls -1 "{remote_dir}"/*.pkl 2>/dev/null'
    names = [Path(p).name for p in _ssh_capture(listing).split()]
    if not names:
        print("(no zhai cache files on the box -- run `cxr remote check` first)")
        return
    dest = LOCAL_ROOT / "checkpoints" / "zhai_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        _run(["scp", f"{HOST}:{remote_dir}/{name}", str(dest / name)])
    print(f"pulled -> checkpoints/zhai_reproduction/ ({len(names)} cache files)")
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -k "remote_check or pull_zhai" -v`
Expected: PASS (5 tests)

- [ ] **Step 9: Write the failing tests for CLI wiring + SYNC_PATHS**

Append to `tests/test_remote.py`:

```python
def test_check_cli_pull_flag_skips_run(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "pull_zhai_cache", lambda: calls.append("pull"))
    monkeypatch.setattr(
        remote, "remote_check", lambda **kw: pytest.fail("--pull must not run the reproduction")
    )

    remote.main(["check", "--pull"])

    assert calls == ["pull"]


def test_check_cli_detached_starts_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: calls.append(kw) or "jid")
    monkeypatch.setattr(remote, "attach", lambda jobid: pytest.fail("no --follow: must not attach"))

    remote.main(["check", "--detached", "--ne", "11"])

    assert calls[0]["ne"] == 11


def test_check_cli_detached_follow_attaches(monkeypatch):
    monkeypatch.setattr(remote, "start_zhai_queue", lambda **kw: "jid")
    attached = []
    monkeypatch.setattr(remote, "attach", attached.append)

    remote.main(["check", "--detached", "--follow"])

    assert attached == ["jid"]


def test_check_cli_foreground_calls_remote_check(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "remote_check", lambda **kw: calls.append(kw))

    remote.main(["check", "--ne", "11", "--refresh"])

    assert calls == [
        {"ne": 11, "ne_brem": 200, "ne_supp": 200, "refresh": True, "no_sync": False}
    ]


def test_sync_paths_ship_checks_and_zhai_shim():
    assert "checks" in remote.SYNC_PATHS
    assert "reproduce_zhai.py" in remote.SYNC_PATHS
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -k check_cli -v`
Expected: FAIL — `remote.main(["check", ...])` raises `SystemExit` (argparse:
"invalid choice: 'check'")

- [ ] **Step 11: Wire the CLI and update SYNC_PATHS**

In `src/cxr_mc/remote.py`, replace line 86:

```python
SYNC_PATHS = ["src", "scan.py", "pyproject.toml", "uv.lock", "README.md"]
```

with:

```python
SYNC_PATHS = [
    "src",
    "scan.py",
    "reproduce_zhai.py",
    "checks",
    "pyproject.toml",
    "uv.lock",
    "README.md",
]
```

Insert after `_cli_sync` (after line 692, before `def _build_remote_parser` at
line 695):

```python
def _cli_check(args):
    if args.pull:
        pull_zhai_cache()
        return
    if args.detached:
        jobid = start_zhai_queue(
            ne=args.ne,
            ne_brem=args.ne_brem,
            ne_supp=args.ne_supp,
            refresh=args.refresh,
            no_sync=args.no_sync,
        )
        if args.follow:
            attach(jobid)
        return
    remote_check(
        ne=args.ne,
        ne_brem=args.ne_brem,
        ne_supp=args.ne_supp,
        refresh=args.refresh,
        no_sync=args.no_sync,
    )
```

In `_build_remote_parser`, insert after the `sy = sub.add_parser("sync", ...)`
block (after line 784, before `return ap` at line 786):

```python
    ck = sub.add_parser(
        "check",
        help="run the Zhai reproduction + supplementary MC on the box, pull caches back",
    )
    ck.add_argument("--ne", type=int, default=20_000, help="Fig.1c anchor line electrons per energy")
    ck.add_argument(
        "--ne-brem", type=int, default=200, help="Fig.1c anchor bremsstrahlung electrons per energy"
    )
    ck.add_argument(
        "--ne-supp", type=int, default=200, help="supplementary electrons per polar-tilt spectrum"
    )
    ck.add_argument("--refresh", action="store_true", help="recompute even if a matching cache exists")
    ck.add_argument("--no-sync", action="store_true", help="skip the code upload")
    ck.add_argument(
        "--detached", "-d", action="store_true", help="launch as a DETACHED job (survives disconnect)"
    )
    ck.add_argument(
        "--follow", "-f", action="store_true", help="with --detached: track the job live after launching"
    )
    ck.add_argument(
        "--pull", action="store_true", help="skip the run; just fetch existing zhai cache files"
    )
    ck.set_defaults(func=_dispatch(_cli_check))
```

Update the module docstring's command list (in the `"""..."""` block, lines
15-46) by adding this line after the `cxr remote sync` example (line 22):

```
    cxr remote check [--ne N] [--ne-brem N] [--ne-supp N] [--refresh]
                      [--detached [--follow]] [--pull]
                                     # run the Zhai + supplementary MC on the
                                     # box, or pull its cache back
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_remote.py -v`
Expected: PASS (all tests, including the pre-existing ones — full-file run to
confirm no regression)

- [ ] **Step 13: Lint, format, type check**

```bash
uv run ruff check src/cxr_mc/remote.py tests/test_remote.py
uv run ruff format src/cxr_mc/remote.py tests/test_remote.py
uv run pyright src/cxr_mc/remote.py
```

- [ ] **Step 14: Commit**

```bash
git add src/cxr_mc/remote.py tests/test_remote.py
git commit -m "feat(remote): add cxr remote check to run Zhai reproduction on the GPU box"
```

---

### Task 4: `figure_supplementary_overview`

**Files:**
- Modify: `checks/anchor_figures.py` (add after `figure_supplementary_hbn`,
  currently ending at line 763, before the `# ---- validation table + CLI`
  comment at line 766)
- Test: `tests/test_anchor_figures.py` (append)

**Interfaces:**
- Consumes: `supplementary_study(crystal) -> SupplementaryCoherentStudy`,
  `ZHAI_SUPPLEMENTARY_STUDIES` (both existing)
- Produces: `figure_supplementary_overview(spectra: dict[str, np.ndarray]) ->
  matplotlib.figure.Figure` — `spectra` keys must be exactly `{"wse2",
  "mose2", "hbn"}`; each value is that material's intrinsic coherent spectrum
  at its steepest requested tilt (`study.polar_tilts_deg[-1]`, i.e. -20°) and
  thinnest listed thickness (`study.thicknesses_nm[0]`), on that study's own
  `E_grid`. Consumed by `export_all_figures` (Task 5).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anchor_figures.py`:

```python
def test_supplementary_overview_figure_smoke():
    from matplotlib.figure import Figure

    spectra = {
        crystal: _synthetic_supplementary_spectra(af.supplementary_study(crystal))[-20.0]
        for crystal in ("wse2", "mose2", "hbn")
    }

    fig = af.figure_supplementary_overview(spectra)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "WSe_2" in titles and "MoSe_2" in titles and "h-BN" in titles
    fig.canvas.draw()


def test_supplementary_overview_rejects_missing_material():
    with pytest.raises(ValueError, match="wse2"):
        af.figure_supplementary_overview({"wse2": np.zeros(4), "mose2": np.zeros(4)})


def test_supplementary_overview_rejects_unknown_key():
    spectra = {
        crystal: _synthetic_supplementary_spectra(af.supplementary_study(crystal))[-20.0]
        for crystal in ("wse2", "mose2", "hbn")
    }
    spectra["hopg"] = spectra.pop("hbn")
    with pytest.raises(ValueError, match="hbn"):
        af.figure_supplementary_overview(spectra)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py -k supplementary_overview -v`
Expected: FAIL with `AttributeError: module 'anchor_figures' has no attribute
'figure_supplementary_overview'`

- [ ] **Step 3: Implement `figure_supplementary_overview`**

Insert into `checks/anchor_figures.py` after `figure_supplementary_hbn` (after
line 763, before the `# ---- validation table + CLI` comment at line 766):

```python
def figure_supplementary_overview(spectra: dict[str, np.ndarray]):
    """One row of three panels -- WSe2, MoSe2, h-BN side by side -- each
    showing that material's steepest requested polar tilt at its thinnest
    listed thickness: a single representative slice per material, for the
    paper's validation appendix. The full tilt x thickness grid is the
    per-material figure_supplementary_tmd/hbn panels above."""
    import matplotlib.pyplot as plt

    expected = {"wse2", "mose2", "hbn"}
    if set(spectra) != expected:
        raise ValueError(f"spectra must contain exactly {sorted(expected)}, got {sorted(spectra)}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, crystal in zip(axes, ("wse2", "mose2", "hbn"), strict=True):
        study = supplementary_study(crystal)
        thickness_nm = study.thicknesses_nm[0]
        tilt_deg = study.polar_tilts_deg[-1]  # steepest requested tilt
        ax.plot(study.E_grid, spectra[crystal], color="C0")
        ax.set_title(f"{study.label}\n{thickness_nm:g} nm, tilt {tilt_deg:g}°")
        ax.set_xlabel("Photon energy (eV)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Coherent emission $d^2N/(dE\,d\Omega\,e^-)$")
    fig.suptitle(
        f"Zhai supplementary studies overview, "
        f"{ZHAI_SUPPLEMENTARY_STUDIES['wse2'].energy_keV:g} keV"
    )
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py -k supplementary_overview -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, type check**

```bash
uv run ruff check checks/anchor_figures.py tests/test_anchor_figures.py
uv run ruff format checks/anchor_figures.py tests/test_anchor_figures.py
uv run pyright checks/anchor_figures.py
```

- [ ] **Step 6: Commit**

```bash
git add checks/anchor_figures.py tests/test_anchor_figures.py
git commit -m "feat(checks): add figure_supplementary_overview cross-material figure"
```

---

### Task 5: `export_all_figures` + `cxr check --export`

**Files:**
- Modify: `checks/anchor_figures.py` (add `export_all_figures`, after
  `figure_supplementary_overview` from Task 4, before `# ---- validation
  table + CLI`)
- Modify: `src/cxr_mc/check.py`
- Test: `tests/test_anchor_figures.py` (append), create `tests/test_check.py`

**Interfaces:**
- Consumes: `ZhaiAnchor`, `cached_model_spectra`, `reference_curve`,
  `figure_spectra`, `figure_flux_anchor`, `figure_enhancement`,
  `ZHAI_SUPPLEMENTARY_STUDIES`, `cached_coherent_spectra`,
  `figure_supplementary_tmd`, `figure_supplementary_hbn`,
  `figure_supplementary_overview` (Task 4) — all in `checks/anchor_figures.py`
- Produces: `export_all_figures(outdir="figures", ne=20_000, ne_brem=200,
  ne_supp=200, *, cache_dir=None) -> list[Path]` (PNG paths written; a PDF is
  written alongside each). `cxr check --export [--outdir DIR] [--ne N]
  [--ne-brem N] [--ne-supp N]` in `check.py`.

- [ ] **Step 1: Write the failing test for `export_all_figures`**

Append to `tests/test_anchor_figures.py`:

```python
def test_export_all_figures_writes_expected_files(tmp_path, monkeypatch):
    anchor = af.ZhaiAnchor()

    def fake_model_spectra(received_anchor, ne, ne_brem):
        return _synthetic_model(anchor)

    def fake_coherent(study, thickness_nm, ne):
        return _synthetic_supplementary_spectra(study)

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    monkeypatch.setattr(af, "model_coherent_spectra", fake_coherent)

    outdir = tmp_path / "figures"
    written = af.export_all_figures(
        outdir=outdir, ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path / "cache"
    )

    names = {p.name for p in written}
    assert "zhai_fig1c_spectra_vs_theory.png" in names
    assert "zhai_flux_anchor.png" in names
    assert "zhai_bulk_vs_film_enhancement.png" in names
    assert "zhai_supplementary_wse2_42nm.png" in names
    assert "zhai_supplementary_mose2_147nm.png" in names
    assert "zhai_supplementary_hbn_921nm.png" in names
    assert "zhai_supplementary_overview.png" in names
    for path in written:
        assert path.exists()
        assert path.with_suffix(".pdf").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py::test_export_all_figures_writes_expected_files -v`
Expected: FAIL with `AttributeError: module 'anchor_figures' has no attribute
'export_all_figures'`

- [ ] **Step 3: Implement `export_all_figures`**

Insert into `checks/anchor_figures.py` after `figure_supplementary_overview`
(Task 4), before `# ---- validation table + CLI`:

```python
def export_all_figures(
    outdir: str | Path = "figures",
    ne: int = 20_000,
    ne_brem: int = 200,
    ne_supp: int = 200,
    *,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    """Render the complete publication figure set from whatever is already
    cached under checkpoints/zhai_reproduction/ (a cache miss recomputes
    locally rather than failing) -- the Fig.1c trio, every supplementary
    panel, and the cross-material overview -- to `outdir`. One command turns
    a `cxr remote check` pull into the full figure set with no per-study
    clicking in the app. Returns the list of PNG paths written (a PDF is
    written alongside each)."""
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(name, fig):
        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        written.append(outpath / f"{name}.png")

    anchor = ZhaiAnchor()
    model, _hit, _path = cached_model_spectra(anchor, ne=ne, ne_brem=ne_brem, cache_dir=cache_dir)
    reference = reference_curve(anchor)
    _save("zhai_fig1c_spectra_vs_theory", figure_spectra(anchor, model, reference))
    _save("zhai_flux_anchor", figure_flux_anchor(anchor, model))
    _save("zhai_bulk_vs_film_enhancement", figure_enhancement(anchor, model))

    overview_spectra: dict[str, np.ndarray] = {}
    for crystal, study in ZHAI_SUPPLEMENTARY_STUDIES.items():
        for thickness_nm in study.thicknesses_nm:
            spectra, _hit, _path = cached_coherent_spectra(
                study, thickness_nm, ne=ne_supp, cache_dir=cache_dir
            )
            fig = (
                figure_supplementary_hbn(study, thickness_nm, spectra)
                if crystal == "hbn"
                else figure_supplementary_tmd(study, thickness_nm, spectra)
            )
            _save(f"zhai_supplementary_{crystal}_{thickness_nm:g}nm", fig)
            if thickness_nm == study.thicknesses_nm[0]:
                overview_spectra[crystal] = spectra[study.polar_tilts_deg[-1]]

    _save("zhai_supplementary_overview", figure_supplementary_overview(overview_spectra))
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py::test_export_all_figures_writes_expected_files -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `cxr check --export`**

Create `tests/test_check.py`:

```python
"""Tests for cxr_mc.check: marimo launch args + the --export figure batch."""

import sys
from pathlib import Path

import pytest

from cxr_mc import check


def test_command_uses_run_by_default():
    cmd = check._command()
    assert cmd[2:5] == ["-m", "marimo", "run"]
    assert check.NOTEBOOK in cmd
    assert "--watch" not in cmd


def test_command_edit_and_watch_flags():
    cmd = check._command(edit=True, watch=True)
    assert "edit" in cmd and "run" not in cmd
    assert "--watch" in cmd


def test_export_cli_calls_export_all_figures_and_skips_marimo(monkeypatch, tmp_path):
    calls = []

    class _FakeAF:
        @staticmethod
        def export_all_figures(outdir, ne, ne_brem, ne_supp):
            calls.append((outdir, ne, ne_brem, ne_supp))
            return [Path(outdir) / "a.png"]

    monkeypatch.setitem(sys.modules, "anchor_figures", _FakeAF())
    monkeypatch.setattr(check, "_launch", lambda **kw: pytest.fail("--export must not launch marimo"))

    check.main(["check", "--export", "--outdir", str(tmp_path), "--ne", "11"])

    assert calls == [(str(tmp_path), 11, 200, 200)]


def test_default_cli_launches_marimo_not_export(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kw: calls.append(kw))

    check.main(["check"])

    assert calls == [{"edit": False, "watch": False}]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest -p no:cacheprovider tests/test_check.py -v`
Expected: `test_command_*` PASS (unchanged behavior); `test_export_cli_*` FAIL
with `argparse` error (`unrecognized arguments: --export`)

- [ ] **Step 7: Implement `cxr check --export`**

Replace the full contents of `src/cxr_mc/check.py`:

```python
"""``cxr check`` -- launch the marimo validation app (``notebooks/validation_app.py``),
or render its figures in batch from cache.

Unlike ``cxr analyze``, this command takes no material argument -- the
validation app reproduces fixed literature figures (e.g. Zhai et al.) rather
than sweeping a chosen material, so there's no initial-material selection to
resolve or persist.

    cxr check                      # `marimo run` the validation app
    cxr check --watch              # add marimo's --watch
    cxr check --edit               # `marimo edit` instead of `marimo run`
    cxr check --export             # skip marimo; render the full Zhai figure
                                    # set from checkpoints/zhai_reproduction/
                                    # (see `cxr remote check`) to figures/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

NOTEBOOK = "notebooks/validation_app.py"


def _command(*, edit=False, watch=False):
    """The marimo argv for one launch (module-run through the current
    interpreter so the venv's marimo is the one that runs). Marimo's own flags
    go before the notebook path; app args go after ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        NOTEBOOK,
        "--",
    ]


def _launch(*, edit=False, watch=False):
    cmd = _command(edit=edit, watch=watch)
    env = {**os.environ}
    subprocess.run(cmd, check=True, env=env)


def _export(outdir="figures", ne=20_000, ne_brem=200, ne_supp=200):
    checks_dir = Path(__file__).resolve().parents[2] / "checks"
    if str(checks_dir) not in sys.path:
        sys.path.insert(0, str(checks_dir))
    import anchor_figures as af

    written = af.export_all_figures(outdir, ne=ne, ne_brem=ne_brem, ne_supp=ne_supp)
    for path in written:
        print(f"wrote {path}")


def _cli(args):
    if args.export:
        _export(args.outdir, ne=args.ne, ne_brem=args.ne_brem, ne_supp=args.ne_supp)
        return
    _launch(edit=args.edit, watch=args.watch)


def add_subparser(sub):
    """Register the ``check`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser("check", help=f"launch {NOTEBOOK} (marimo run/edit), or --export its figures")
    ap.add_argument("--watch", action="store_true", help="pass marimo's --watch")
    ap.add_argument("--edit", action="store_true", help="use `marimo edit` instead of `marimo run`")
    ap.add_argument(
        "--export",
        action="store_true",
        help="skip marimo; render the full Zhai figure set from cache to --outdir",
    )
    ap.add_argument("--outdir", default="figures", help="with --export: output directory")
    ap.add_argument("--ne", type=int, default=20_000, help="with --export: Fig.1c line electrons")
    ap.add_argument("--ne-brem", type=int, default=200, help="with --export: Fig.1c brem electrons")
    ap.add_argument("--ne-supp", type=int, default=200, help="with --export: supplementary electrons")
    ap.set_defaults(func=_cli)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-check", description="launch the validation app")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest -p no:cacheprovider tests/test_check.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Lint, format, type check**

```bash
uv run ruff check checks/anchor_figures.py src/cxr_mc/check.py tests/test_anchor_figures.py tests/test_check.py
uv run ruff format checks/anchor_figures.py src/cxr_mc/check.py tests/test_anchor_figures.py tests/test_check.py
uv run pyright src/cxr_mc/check.py checks/anchor_figures.py
```

- [ ] **Step 10: Commit**

```bash
git add checks/anchor_figures.py src/cxr_mc/check.py tests/test_anchor_figures.py tests/test_check.py
git commit -m "feat(check): add cxr check --export to batch-render the Zhai figure set from cache"
```

---

### Task 6: Guard-clause coverage + cache-key portability regression test

**Files:**
- Modify: `tests/test_anchor_figures.py` (append)

**Interfaces:**
- Consumes: `figure_supplementary_tmd`, `figure_supplementary_hbn` (existing,
  unmodified), `_CHECKS` (existing fixture, module-level `Path` in this test
  file)
- Produces: no new production code; test-only task

- [ ] **Step 1: Write the guard-clause tests**

Append to `tests/test_anchor_figures.py`:

```python
def test_supplementary_tmd_figure_rejects_hbn_crystal():
    study = af.supplementary_study("hbn")
    with pytest.raises(ValueError, match="only for the WSe2 and MoSe2"):
        af.figure_supplementary_tmd(
            study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
        )


def test_supplementary_tmd_figure_rejects_incomplete_tilt_set():
    study = af.supplementary_study("wse2")
    spectra = _synthetic_supplementary_spectra(study)
    del spectra[study.polar_tilts_deg[0]]
    with pytest.raises(ValueError, match="exactly the study's four polar tilts"):
        af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)


def test_supplementary_hbn_figure_rejects_non_hbn_crystal():
    study = af.supplementary_study("wse2")
    with pytest.raises(ValueError, match="only for the h-BN study"):
        af.figure_supplementary_hbn(
            study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
        )


def test_supplementary_hbn_figure_rejects_incomplete_tilt_set():
    study = af.supplementary_study("hbn")
    spectra = _synthetic_supplementary_spectra(study)
    del spectra[study.polar_tilts_deg[0]]
    with pytest.raises(ValueError, match="exactly the study's four polar tilts"):
        af.figure_supplementary_hbn(study, study.thicknesses_nm[0], spectra)


def test_physics_source_tree_is_lf_only():
    """The Zhai cache key hashes raw bytes of every src/cxr_mc/**/*.py file
    (_zhai_cache_key / _supplementary_cache_key); cxr remote check relies on
    the box and the laptop hashing identical bytes for a pulled cache to be a
    hit. The repo's .gitattributes pins `* text=auto eol=lf`, so a CRLF file
    slipping into src/ would silently produce a different hash per platform
    and break that invariant with no visible error -- catch it here instead."""
    repo_root = _CHECKS.parent
    offenders = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "src" / "cxr_mc").glob("**/*.py"))
        if b"\r\n" in path.read_bytes()
    ]
    assert offenders == [], f"CRLF line endings found (breaks box<->laptop cache hash): {offenders}"
```

- [ ] **Step 2: Run tests to verify current status**

Run: `uv run pytest -p no:cacheprovider tests/test_anchor_figures.py -k "rejects or lf_only" -v`
Expected: All 5 PASS immediately — these test *existing* guard clauses in
`figure_supplementary_tmd`/`figure_supplementary_hbn` (unchanged by this
plan) and an *existing* repo invariant (`.gitattributes`), so no production
code changes are needed; this step locks both in as regression guards.

- [ ] **Step 3: Lint, format**

```bash
uv run ruff check tests/test_anchor_figures.py
uv run ruff format tests/test_anchor_figures.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_anchor_figures.py
git commit -m "test(anchor_figures): cover supplementary figure guard clauses + CRLF cache-key invariant"
```

---

### Task 7: Supplementary provenance write-up

**Files:**
- Create: `docs/validation/zhai-supplementary.md`
- Modify: `TODO.md`

**Interfaces:**
- Consumes: `checks/anchor_figures.py::ZHAI_SUPPLEMENTARY_STUDIES` (values
  transcribed into the doc, read-only — no code change), `src/cxr_mc/montecarlo/geometry.py::tilted_geometry`
  (referenced, not modified)
- Produces: a documentation artifact only; no ledger status change
  (`docs/physics-validation-ledger.md` untouched)

- [ ] **Step 1: Write `docs/validation/zhai-supplementary.md`**

```markdown
# Validation status: Zhai supplementary coherent-emission studies

**Scope.** `checks/anchor_figures.py::ZHAI_SUPPLEMENTARY_STUDIES` — the
WSe₂/MoSe₂/h-BN coherent-only reproductions (`figure_supplementary_tmd`,
`figure_supplementary_hbn`), driven by `model_coherent_spectra` and rendered
by the validation app's "Zhai supplementary" section and `cxr check --export`.

This is a **provenance and open-question record, not a physics
re-derivation write-up** (contrast `docs/validation/hbn-structure.md`) — none
of these study *inputs* carry a ledger `id` of their own; they parameterize
existing ledgered claims (`line-energy-dispersion`, `coherent-line-spectrum`,
`electron-transport`). Promoting anything here past this record requires a
fresh-context re-derivation per `docs/validation/README.md`, and only a human
signs off.

## What's encoded, and where it came from

| material | thicknesses (nm) | energy window (eV) | polar tilts (deg) | beam energy |
|----------|-------------------|---------------------|--------------------|-------------|
| WSe₂ | 42, 55, 75 | 800–1200 | −10, −15, −17.5, −20 | 200 keV |
| MoSe₂ | 47, 112, 147 | 800–1200 | −10, −15, −17.5, −20 | 200 keV |
| h-BN | 921 | 600–1200 | −10, −15, −17.5, −20 | 200 keV |

These match the values transcribed into `TODO.md`'s original user-added note
("Zhai supplementary coherent-emission figures ... all 200 keV, at polar
tilts −10/−15/−17.5/−20 deg") — the code and that note agree, which is as far
as a provenance check without the source SI in-repo can go. Confirming these
numbers directly against the published SI figures/tables is unclaimed here.

## Open question: azimuthal angle

`model_coherent_spectra` (`checks/anchor_figures.py`) builds each tilt's
geometry via:

```python
beam_dir, n_hat = tilted_geometry(study.theta_obs_rad, float(np.deg2rad(tilt_deg)))
```

`tilted_geometry` (`src/cxr_mc/montecarlo/geometry.py`) takes only a polar
tilt — there is no azimuthal parameter in this call at all, so every
supplementary panel is implicitly computed at **azimuth = 0** (the function's
own docstring: "azimuth 0. Tilting the sample so its normal points along...").

**Unconfirmed:** whether the Zhai SI's reported WSe₂/MoSe₂/h-BN polar-tilt
series were themselves measured/computed at azimuth = 0, or at some other
fixed azimuthal setting. If the SI's series varies azimuth (or fixes it at a
nonzero value), the current code reproduces the wrong slice of the parameter
space at every tilt in the table above — this would not show up as a
qualitative failure (the underlying lineshape physics is still correct), only
as a quantitative mismatch against the specific published panel.

**Next step (separate from this write-up):** a fresh-context reviewer with
access to the Zhai et al. SI should locate the exact figure/table describing
the polar-tilt series' azimuthal convention and confirm or refute azimuth = 0
against `docs/validation/README.md`'s re-derivation workflow. Until then this
stays an open flag, not a `discrepancy` (no check has actually failed — the
assumption is merely unverified) and not `rederived` (no independent
confirmation exists yet).

## Status

Provenance: internally consistent (code ≡ transcribed note). Azimuth
assumption: **unconfirmed**, open question above. No ledger row changes as a
result of this write-up.
```

- [ ] **Step 2: Update `TODO.md`'s branch-scoped note to point at the doc**

Edit the branch-scoped Zhai bullet under `# USER ADDED:` (written in Task 1,
step 1) to add a trailing sentence:

```markdown
  Provenance + the azimuth = 0 status: [`docs/validation/zhai-supplementary.md`](docs/validation/zhai-supplementary.md).
```

(Append this line to the existing bullet from Task 1 rather than replacing
it.)

- [ ] **Step 3: Commit**

```bash
git add docs/validation/zhai-supplementary.md TODO.md
git commit -m "docs(validation): write up Zhai supplementary study provenance + open azimuth question"
```

---

### Task 8: Full verification + branch TODO.md sync for merge

**Files:**
- Modify: `TODO.md` (slim the branch-scoped note to a one-line summary, per
  this repo's branch-workflow convention)

**Interfaces:** none — wrap-up/verification only.

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -p no:cacheprovider
```

Expected: all tests pass, including every test added in Tasks 1–6 plus the
full pre-existing suite (no regressions).

- [ ] **Step 2: Run lint, format check, and type check across the whole repo**

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Expected: `ruff check` and `ruff format --check` clean. `pyright` should show
no *new* errors versus `main` — `git stash` (or compare against `main` in a
second worktree) and re-run `uv run pyright` if unsure which errors are
pre-existing versus introduced here; confirm any reported errors are outside
every file this plan touched (`checks/anchor_figures.py`, `reproduce_zhai.py`,
`src/cxr_mc/remote.py`, `src/cxr_mc/check.py`).

- [ ] **Step 3: Slim TODO.md to a one-line summary for main**

Per `TODO.md`'s own "Item generation" convention (step 3: "Move to main,
create 1 sentence summary of new item"), replace the branch-scoped Zhai bullet
(from Tasks 1 and 7) with:

```markdown
- Zhai supplementary coherent-emission figures (WSe₂/MoSe₂/h-BN, `checks/anchor_figures.py`
  + the validation app) now have GPU lab-box reproduction (`cxr remote check`), a
  one-command local figure export (`cxr check --export`), and a provenance write-up
  covering the open azimuth = 0 question: [`docs/validation/zhai-supplementary.md`](docs/validation/zhai-supplementary.md).
```

```bash
git add TODO.md
git commit -m "docs(todo): summarize feature/zhai-gpu-reproduction for main"
```

- [ ] **Step 4: Hand off**

Do not merge or open a PR — per this repo's workflow, that is a deliberate,
separate step the user takes (via `superpowers:finishing-a-development-branch`
or an explicit merge/PR request), not an automatic last step of this plan.
Report the branch name and the commit log for the user's review.
