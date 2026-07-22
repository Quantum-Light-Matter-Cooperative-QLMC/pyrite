"""``cxr remote`` -- schedule heavy CXR scans on the lab GPU box, keep data-vis local.

The split this enables: the laptop holds the project and does all interactive
analysis and static-HTML export, while
the lab box (an RTX 5080, ssh host 'qlmc') only does the GPU-heavy Monte-Carlo
sweep. Every compute-producing subcommand ships the current code up, submits a
one-GPU SLURM batch script there (see :mod:`cxr_mc.scan`), and pulls results into
./checkpoints
-- so you never hand-ssh in or copy files, and the lab box needs no visualization
toolchain.

Optional, dev-only tool: it is only useful if you have an ssh host configured
(default 'qlmc', override via CXR_REMOTE_HOST) to run sweeps on. Every other
``cxr`` command works without it.

One-shot (submit, wait for SLURM, then pull):

    cxr remote scan mose2               # sync code up, submit sweep, pull checkpoint
    cxr remote scan mose2 --quick       # tiny grid smoke test
    cxr remote scan mose2 --no-sync     # skip the code upload (code unchanged)
    cxr remote pull mose2 wse2          # fetch existing checkpoints (grid-filtered)
    cxr remote pull mose2 --full        # fetch the full, un-filtered checkpoint
    cxr remote sync                     # only push the current code
    cxr remote check [--ne N] [--ne-brem N] [--ne-supp N] [--refresh]
                     [--detached [--follow]] [--pull]
                                    # run the Zhai + supplementary MC on the
                                    # box, or pull its cache back

Asynchronous SLURM queue (survives ssh disconnect -- launch, walk away, reconnect later):

    cxr remote start mose2 wse2 mos2    # queue several materials, run detached
    cxr remote start mose2 --follow     # launch, then track it live
    cxr remote start mose2 --quick      # detached quick smoke test
    cxr remote attach [JOBID]           # (re)connect + track live (default: latest)
    cxr remote jobs                     # list jobs on the box + their state
    cxr remote status [JOBID] [-v|-vv]  # job summary; SLURM details; case progress
    cxr remote logs [JOBID] --follow    # tail the remote log (live)
    cxr remote stop mose2 wse2          # cancel live SLURM job(s) by material
    cxr remote stop --all               # cancel every live SLURM job
    cxr remote pull mose2 wse2 mos2     # fetch the finished checkpoints (grid-filtered)

`start` returns immediately: it ships the code, writes a batch script under
<remote>/jobs/<jobid>/, and submits it to SLURM. The batch job processes the
materials with bounded concurrency (two `scan.py` processes by default),
writing meta/state/log and its scheduler ID into the job dir.

The SLURM job is independent of the submission SSH connection, so the SSH link
is only ever a VIEWER. `attach` renders one local case-progress bar per material;
`logs --follow` remains the raw shared diagnostic stream. Both exit when the job
finishes. To DISCONNECT, just Ctrl-C (or close the terminal / drop the link) --
that tears down the viewer only, and the job runs to completion. Reconnect any
time with `attach`/`status`/`logs`, then `pull` once state is `done`.

Then locally: run ``cxr analyze <material>`` or ``cxr export [stem]``.

Transport is ssh/scp only (uses the 'qlmc' host in ~/.ssh/config, cloudflared
ProxyCommand and all) -- no rsync dependency, so it works from Windows Git Bash.
Override the box via env: CXR_REMOTE_HOST / CXR_REMOTE_DIR / CXR_REMOTE_UV.
"""

import argparse
import datetime
import hashlib
import io
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from ._remote import presentation as _presentation
from .materials import CATALOG
from .scan import load_all_materials

_SHELL_TOKEN_RE = _presentation._SHELL_TOKEN_RE
_STATE_COLORS = _presentation._STATE_COLORS
_TQDM_FRAME_RE = _presentation._TQDM_FRAME_RE
_active_work_label = _presentation._active_work_label
_aggregate_progress = _presentation._aggregate_progress
_overall_progress_line = _presentation._overall_progress_line
_clean_recent_log = _presentation._clean_recent_log
_color_enabled = _presentation._color_enabled
_format_case_progress = _presentation._format_case_progress
_format_fields = _presentation._format_fields
_format_job_status = _presentation._format_job_status
_format_table = _presentation._format_table
_legacy_progress = _presentation._legacy_progress
_marked_sections = _presentation._marked_sections
_material_label = _presentation._material_label
_metadata_fields = _presentation._metadata_fields
_metadata_value = _presentation._metadata_value
_mode_summary = _presentation._mode_summary
_paint = _presentation._paint
_parse_progress_records = _presentation._parse_progress_records
_progress_group = _presentation._progress_group
_progress_track = _presentation._progress_track
_scheduler_fields = _presentation._scheduler_fields
_style_states = _presentation._style_states

HOST = os.environ.get("CXR_REMOTE_HOST", "qlmc")
REMOTE_DIR = os.environ.get("CXR_REMOTE_DIR", "/home/aamador/dev/cxr-mc")
REMOTE_UV = os.environ.get("CXR_REMOTE_UV", "/home/aamador/.local/bin/uv")
SLURM_PARTITION = "gpu"
SLURM_GPUS = 1
SLURM_TIME = "UNLIMITED"
DEFAULT_PARALLEL_MATERIALS = 2
MAX_PARALLEL_MATERIALS = 4
# repo root = three levels up from src/cxr_mc/remote.py. remote.py orchestrates
# the *checkout* (it tars the working tree up to the box), so it resolves paths
# against the repo root, not its own package dir.
LOCAL_ROOT = Path(__file__).resolve().parents[2]
MATS_FILE = LOCAL_ROOT / "mats_to_sim.toml"

# detached-job bookkeeping lives under <REMOTE_DIR>/jobs/<jobid>/ on the box
# (gitignored there): run.sh, meta, state, log. One subdir per `start`.
JOBS_SUBDIR = "jobs"
RESERVATIONS_SUBDIR = "reservations"

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too), plus the root scan.py shim the box invokes and
# pyproject.toml. Not checkpoints/ (the output we pull back the other way).
SYNC_PATHS = [
    "src",
    "scan.py",
    "reproduce_zhai.py",
    "checks",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "mats_to_sim.toml",
]

# text extensions whose CRLF is normalized to LF before tarring (see _add_to_tar):
# the laptop is Windows so its working files are CRLF, and shipping those over the
# box's LF checkout dirties `git status` there even though content is identical.
TEXT_EXTS = {".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".csv"}


def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _ssh_capture(remote_cmd):
    """Run a remote command over ssh and return its stdout (text). Prints the
    box's stderr and aborts on a nonzero exit.

    Decode as UTF-8 explicitly: the Linux box emits UTF-8 (job logs carry tqdm's
    block-glyph progress bars, e.g. `████▌`), but `text=True` alone would decode
    with the Windows locale (cp1252), which chokes on those bytes -- so `status`,
    `jobs`, and `logs` would crash mid-read. `errors="replace"` keeps any stray
    non-UTF-8 byte from aborting the whole command."""
    r = subprocess.run(
        ["ssh", "-n", HOST, remote_cmd],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ssh command failed (exit {r.returncode})")
    return r.stdout


def _local_sha256(path: Path) -> str | None:
    """Return a checkpoint's SHA-256 digest, or ``None`` when it is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha256(path: str) -> str:
    """Return the SHA-256 digest of one remote checkpoint artifact."""
    output = _ssh_capture(f"sha256sum {path}").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\S+", output)
    if match is None:
        raise SystemExit(f"invalid sha256sum output for remote artifact {path!r}")
    return match.group(1).lower()


def _check_shell_tokens(tokens):
    """Reject tokens that are unsafe to interpolate into remote shell commands."""
    bad = [token for token in tokens if not _SHELL_TOKEN_RE.fullmatch(token)]
    if bad:
        raise SystemExit(
            f"invalid remote shell token(s) {bad}: expected only letters, digits, "
            "underscore, or hyphen"
        )


def _check_materials(materials):
    """Validate runnable material keys for remote scan/start operations."""
    _check_shell_tokens(materials)
    unknown = [material for material in materials if material not in CATALOG.materials]
    if unknown:
        raise SystemExit(
            f"unknown material(s): {', '.join(unknown)}; valid: {', '.join(CATALOG.material_keys)}"
        )


def _add_to_tar(tar, local, arcname):
    """Add one local file to the tar. Text files (TEXT_EXTS) get CRLF->LF so the
    box receives LF-clean content -- no cosmetic `git status` diff there, so a
    later `git pull` isn't blocked. Other files are added verbatim."""
    if local.suffix.lower() in TEXT_EXTS:
        data = local.read_bytes().replace(b"\r\n", b"\n")
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = int(local.stat().st_mtime)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    else:
        tar.add(local, arcname=arcname)


def sync_code():
    """Tar SYNC_PATHS up (CRLF->LF normalized for text, via _add_to_tar) and
    extract them over the repo on the box.

    Normalizing line endings here keeps the edit-locally / run-remotely loop --
    it ships the current WORKING tree (no commit required) yet stays LF-clean, so
    the box's `git status` doesn't flag every synced .py as modified and a later
    `git pull` there isn't blocked. (The committed-state alternative -- `git push`
    from the laptop + `git fetch && git reset --hard origin/<branch>` on the box --
    would force a commit before every run, which this loop is built to avoid.)"""
    with tempfile.TemporaryDirectory() as td:
        tarpath = os.path.join(td, "cxr_code.tgz")
        with tarfile.open(tarpath, "w:gz") as t:
            for p in SYNC_PATHS:
                local = LOCAL_ROOT / p
                if not local.exists():
                    continue
                if local.is_dir():
                    for f in sorted(local.rglob("*")):
                        if f.is_file():
                            arc = (Path(p) / f.relative_to(local)).as_posix()
                            _add_to_tar(t, f, arc)
                else:
                    _add_to_tar(t, local, p)
        _run(["scp", tarpath, f"{HOST}:/tmp/cxr_code.tgz"])
    # -n: redirect ssh's stdin from null. Without it, ssh.exe inherits the
    # interactive console stdin and its stdin-forwarding thread never sees EOF,
    # so the client hangs after the remote command (tar) has already exited.
    _run(
        [
            "ssh",
            "-n",
            HOST,
            f"mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR} && tar xzf /tmp/cxr_code.tgz && rm -f /tmp/cxr_code.tgz",
        ]
    )


def remote_scan(material, quick=False, workers=None):
    """Submit one material through SLURM and follow it to completion.

    This compatibility helper deliberately does not pull: callers that need a
    checkpoint can apply their own grid/trim policy after it returns.
    """
    _check_materials([material])
    jobid = start_queue([material], quick=quick, workers=workers, chunk_minutes=0)
    attach(jobid)
    return jobid


def remote_check(
    ne=20_000,
    ne_brem=200,
    ne_supp=200,
    tmd_azimuth=0.0,
    refresh=False,
    no_sync=False,
):
    """Submit the Zhai reproduction through SLURM, follow it, then pull cache."""
    jobid = start_zhai_queue(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
    )
    if not attach(jobid):
        print("Zhai job is still running or its viewer disconnected; skipping automatic cache pull")
        return
    if not _job_succeeded(jobid):
        state = _job_state(jobid) or "no terminal state recorded"
        raise SystemExit(
            f"Zhai SLURM job {jobid} did not complete successfully ({state}); cache not pulled"
        )
    pull_zhai_cache()


def pull(stems, grid=False, drop_wide_brem=False, downcast=False, no_sync=False):
    """Fetch checkpoints/<stem>.pkl back from the box for each stem (stem =
    material, or material_quick for a --quick run).

    With ``grid``, filter on the box BEFORE the transfer: slim each checkpoint to
    just the material's current grid (``cxr slim --grid``, plus the optional byte
    trimmers) into a box temp, scp that smaller file into the local active slot,
    and delete the temp. ``sync_code()`` runs first (unless ``no_sync``) so the
    box rebuilds the grid from the same ``config.py`` the laptop has -- closing
    sync drift. Without ``grid`` this is the plain whole-file scp."""
    _check_shell_tokens(stems)
    dest = LOCAL_ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    if grid and not no_sync:
        sync_code()  # box must rebuild the grid from the same config.py
    for stem in stems:
        local = dest / f"{stem}.pkl"
        try:
            if grid:
                flags = " --grid"
                if drop_wide_brem:
                    flags += " --drop-wide-brem"
                if downcast:
                    flags += " --downcast"
                remote_tmp = f"/tmp/{stem}.grid.pkl"
                ckpt = f"{REMOTE_DIR}/checkpoints/{stem}.pkl"
                try:
                    _run(
                        [
                            "ssh",
                            "-n",
                            HOST,
                            f"cd {REMOTE_DIR} && {REMOTE_UV} run --no-sync cxr slim "
                            f"{ckpt}{flags} -o {remote_tmp}",
                        ]
                    )
                    if (digest := _remote_sha256(remote_tmp)) and digest == _local_sha256(local):
                        print(f"already current -> checkpoints/{stem}.pkl")
                    else:
                        _run(["scp", f"{HOST}:{remote_tmp}", str(local)])
                        print(f"pulled (grid) -> checkpoints/{stem}.pkl")
                finally:
                    _run(["ssh", "-n", HOST, f"rm -f {remote_tmp}"])
            else:
                ckpt = f"{REMOTE_DIR}/checkpoints/{stem}.pkl"
                remote_tmp = f"{REMOTE_DIR}/checkpoints/.{stem}.pull.{uuid.uuid4().hex}.pkl"
                try:
                    # Checkpoint writers publish with os.replace; a sibling hard link
                    # freezes the exact inode that both the digest and scp will read.
                    _run(["ssh", "-n", HOST, f"ln {ckpt} {remote_tmp}"])
                    if (digest := _remote_sha256(remote_tmp)) and digest == _local_sha256(local):
                        print(f"already current -> checkpoints/{stem}.pkl")
                    else:
                        _run(["scp", f"{HOST}:{remote_tmp}", str(local)])
                        print(f"pulled -> checkpoints/{stem}.pkl")
                finally:
                    _run(["ssh", "-n", HOST, f"rm -f {remote_tmp}"])
        except (OSError, subprocess.CalledProcessError, SystemExit):
            print(f"warning: could not pull checkpoint {stem!r}; continuing")


def pull_zhai_cache():
    """Fetch every cache file under checkpoints/zhai_reproduction/ from the
    box.

    Lists remote filenames first (like clear_remote's listing step) rather
    than `scp -r`, which double-nests the directory when the local destination
    already exists -- listing + per-file scp is unambiguous either way."""
    remote_dir = f"{REMOTE_DIR}/checkpoints/zhai_reproduction"
    listing = (
        f'[ -d "{remote_dir}" ] || exit 0; '
        f'find "{remote_dir}" -maxdepth 1 -type f -name "*.pkl" -printf "%f\\n"'
    )
    names = [Path(p).name for p in _ssh_capture(listing).split()]
    if not names:
        print("(no zhai cache files on the box -- run `cxr remote check` first)")
        return
    dest = LOCAL_ROOT / "checkpoints" / "zhai_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        _run(["scp", f"{HOST}:{remote_dir}/{name}", str(dest / name)])
    print(f"pulled -> checkpoints/zhai_reproduction/ ({len(names)} cache files)")


# ---- detached job queue -------------------------------------------------------
def _stems(materials, quick):
    """Checkpoint stems a queue produces (scan.py writes <material>_quick.pkl
    for --quick runs)."""
    return [f"{m}_quick" if quick else m for m in materials]


def _new_jobid() -> str:
    """Return a collision-resistant, shell-safe local job directory name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _completed_materials(jobid, materials):
    """Return materials with a successful queue-log completion marker.

    Each concurrent child appends one marker only after ``scan.py`` exits
    successfully. Reading those markers after ``attach`` keeps a
    warning-and-continue batch from being reported as though every requested
    checkpoint were available to pull.
    """
    _check_shell_tokens([jobid, *materials])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    completed = set(
        _ssh_capture(f'D="{jobdir}"; sed -n "s/^completed: //p" "$D/log" 2>/dev/null').split()
    )
    return [material for material in materials if material in completed]


def _validate_parallel_materials(parallel_materials):
    """Return a supported in-allocation material-process limit."""
    if (
        not isinstance(parallel_materials, int)
        or not 1 <= parallel_materials <= MAX_PARALLEL_MATERIALS
    ):
        raise SystemExit(f"parallel materials must be between 1 and {MAX_PARALLEL_MATERIALS}")
    return parallel_materials


def _queue_script(
    jobid,
    materials,
    quick,
    workers,
    parallel_materials=DEFAULT_PARALLEL_MATERIALS,
):
    """CXR payload for one bounded-concurrency queue in a SLURM allocation."""
    parallel_materials = _validate_parallel_materials(parallel_materials)
    flags = ""
    if quick:
        flags += " --quick"
    if workers is not None:
        flags += f" --workers {workers}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    return f"""JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
mats=({mats})
total=${{#mats[@]}}
parallel_materials={parallel_materials}
n=0
failures=0
active=0
run_material() {{
  local i="$1"
  local m="$2"
  echo "running $m [$i/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$i" "$total" "$m" "$(date -Is)" \
>> "$JOBDIR/log"
  if ! {REMOTE_UV} run --no-sync python scan.py "$m"{flags} \
    --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1
  then
    echo "WARNING: scan failed for $m; continuing" >> "$JOBDIR/log"
    echo "warning at $m [$i/$total] $(date -Is)" > "$JOBDIR/state"
    return 1
  fi
  echo "completed: $m" >> "$JOBDIR/log"
}}
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  run_material "$n" "$m" &
  active=$((active + 1))
  if [ "$active" -ge "$parallel_materials" ]; then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  fi
done
while [ "$active" -gt 0 ]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done
if [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


def _chunked_queue_script(jobid, materials, quick, workers, chunk_minutes):
    """One SLURM slice of a self-resubmitting chain (spec: chunked remote jobs).

    Reused verbatim by every slice: it resumes from checkpoint, does about
    chunk_minutes of work via scan.py --max-minutes, and either terminates the
    chain (all materials completed:/failed:) or hands off: write the
    'queued slice' state FIRST, then sbatch fail-closed, then append the SID.
    State-first ordering keeps the EXIT trap from releasing reservations while
    the next slice is already pending (spec Component 2 step 4).
    """
    flags = ""
    if quick:
        flags += " --quick"
    if workers is not None:
        flags += f" --workers {workers}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
slice_start=$(date +%s)
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && continue
  now=$(date +%s)
  remaining=$((slice_start + chunk_seconds - now))
  [ "$remaining" -gt 0 ] || break
  remaining_min=$(awk "BEGIN {{ printf \\"%.2f\\", $remaining / 60 }}")
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  {REMOTE_UV} run --no-sync python scan.py "$m"{flags} --max-minutes "$remaining_min" \
    --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
  else
    echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  fi
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""


def _zhai_flags(ne, ne_brem, ne_supp, tmd_azimuth, refresh):
    flags = f" --ne {ne} --ne-brem {ne_brem} --ne-supp {ne_supp} --tmd-azimuth {tmd_azimuth}"
    if refresh:
        flags += " --refresh"
    return flags


def _zhai_queue_script(jobid, ne, ne_brem, ne_supp, tmd_azimuth, refresh):
    """CXR payload for one Zhai reproduction inside a SLURM allocation."""
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    flags = _zhai_flags(ne, ne_brem, ne_supp, tmd_azimuth, refresh)
    return f"""JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
echo "running zhai reproduction since $(date -Is)" > "$JOBDIR/state"
if ! {REMOTE_UV} run --no-sync python reproduce_zhai.py{flags} >> "$JOBDIR/log" 2>&1
then
  echo "FAILED $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
echo "done $(date -Is)" > "$JOBDIR/state"
"""


def _reservation_root() -> str:
    return f"{REMOTE_DIR}/{JOBS_SUBDIR}/{RESERVATIONS_SUBDIR}"


def _release_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Return a remote command that releases only reservations owned by ``jobid``."""
    _check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    releases = "; ".join(
        f'if [ "$(cat "$R/{stem}/jobid" 2>/dev/null)" = "$J" ]; then rm -rf "$R/{stem}"; fi'
        for stem in stems
    )
    command = f'R="{reservations}"; J="{jobid}"'
    return f"{command}; {releases}" if releases else command


def _release_job_reservations_command(jobid: str) -> str:
    """Return a remote command that releases every reservation owned by a job."""
    _check_shell_tokens([jobid])
    return (
        f'R="{_reservation_root()}"; J="{jobid}"; '
        'for d in "$R"/*; do [ -d "$d" ] || continue; '
        '[ "$(cat "$d/jobid" 2>/dev/null)" = "$J" ] && rm -rf "$d"; done'
    )


def _reserve_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Atomically reserve checkpoint stems while a job is being staged.

    Each ``mkdir`` is the cross-client compare-and-set: a second submitter
    cannot pass between the prior ``squeue`` snapshot and ``sbatch``.
    """
    _check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    stem_words = " ".join(stems)
    return f'''R="{reservations}"; J="{jobid}"; mkdir -p "$R"; claimed=""; \
for stem in {stem_words}; do \
  if mkdir "$R/$stem" 2>/dev/null; then \
    printf '%s\\n' "$J" > "$R/$stem/jobid"; claimed="$claimed $stem"; \
  else \
    owner=$(cat "$R/$stem/jobid" 2>/dev/null || true); \
    for held in $claimed; do \
      [ "$(cat "$R/$held/jobid" 2>/dev/null)" = "$J" ] && rm -rf "$R/$held"; \
    done; \
    echo "refusing to stage job $J: checkpoint $stem is reserved by ${{owner:-another staging job}}" >&2; \
    exit 17; \
  fi; \
done'''


def _slurm_batch_script(
    jobid: str,
    payload: str,
    *,
    job_name: str,
    reservation_stems: list[str] | None = None,
    time_limit: str = SLURM_TIME,
) -> str:
    """Wrap a CXR queue payload in the lab box's one-GPU SLURM profile."""
    reservation_stems = reservation_stems or []
    _check_shell_tokens([jobid, *reservation_stems])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    reservations = _reservation_root()
    release_lines = (
        "\n  ".join(
            f'if [ "$(cat "$RESERVATIONS/{stem}/jobid" 2>/dev/null)" = "$JOBID" ]; then rm -rf "$RESERVATIONS/{stem}"; fi;'
            for stem in reservation_stems
        )
        or ":"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={SLURM_GPUS}
#SBATCH --gres=gpu:{SLURM_GPUS}
#SBATCH --time={time_limit}
#SBATCH --output={jobdir}/slurm-%j.out
#SBATCH --error={jobdir}/slurm-%j.err

set -u
module purge 2>/dev/null || true
module load cuda openmpi hdf5 2>/dev/null || true

JOBDIR="{jobdir}"
JOBID="{jobid}"
RESERVATIONS="{reservations}"
release_reservations() {{
  {release_lines}
}}
termination_signal=""
termination_status=0
record_termination() {{
  termination_signal=$1
  termination_status=$2
  echo "FAILED (signal $termination_signal) $(date -Is)" > "$JOBDIR/state"
}}
finish() {{
  status=$?
  if [ -n "$termination_signal" ]; then
    echo "FAILED (signal $termination_signal) $(date -Is)" > "$JOBDIR/state"
    release_reservations
    trap - EXIT
    exit "$termination_status"
  fi
  current=$(cat "$JOBDIR/state" 2>/dev/null || true)
  case "$current" in
    "queued slice"*) ;;
    done*|FAILED*|cancelled*|cancelling*) release_reservations ;;
    *) echo "FAILED (exit $status) $(date -Is)" > "$JOBDIR/state"; release_reservations ;;
  esac
}}
trap 'record_termination TERM 143' TERM
trap 'record_termination INT 130' INT
trap 'record_termination HUP 129' HUP
trap finish EXIT
echo "running $(date -Is)" > "$JOBDIR/state"
{{
  echo "===== SLURM job ${{SLURM_JOB_ID:-unknown}} ====="
  echo "host: $(hostname)"
  echo "gpus: {SLURM_GPUS}"
  echo "partition: {SLURM_PARTITION}"
  echo "started: $(date -Is)"
}} >> "$JOBDIR/log"

{payload}"""


def _submit_slurm_command(
    jobid: str, reservation_stems: list[str] | None = None, *, nice: bool = False
) -> str:
    """Return the remote submission protocol for an already-written batch script."""
    reservation_stems = reservation_stems or []
    _check_shell_tokens([jobid, *reservation_stems])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    release = _release_checkpoint_stems_command(jobid, reservation_stems)
    sbatch = "sbatch --parsable --nice=10000" if nice else "sbatch --parsable"
    return f"""D='{jobdir}'; \
echo "queued $(date -Is)" > "$D/state"; \
SID=$({sbatch} '{jobdir}/run.sh') || {{ \
  echo "FAILED (sbatch submission) $(date -Is)" > "$D/state"; {release}; exit 1; \
}}; \
SID=${{SID%%;*}}; \
case "$SID" in ''|*[!0-9]*) \
  echo "FAILED (sbatch submission) $(date -Is)" > "$D/state"; {release}; exit 1 ;; \
esac; \
printf 'slurm_job_id: %s\\n' "$SID" >> "$D/meta"; \
printf '%s\\n' "$SID"
"""


def _queue_metadata(
    jobid,
    materials,
    quick,
    workers,
    parallel_materials: int | None = DEFAULT_PARALLEL_MATERIALS,
    chunk_minutes: float = 0,
):
    """Static metadata persisted before a queue becomes visible to SLURM."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {' '.join(materials)}",
            f"quick: {bool(quick)}",
            f"workers: {workers}",
            f"parallel_materials: {parallel_materials}",
            f"chunk_minutes: {chunk_minutes}",
            "progress_dashboard: True",
            "",
        ]
    )


def _zhai_queue_metadata(jobid, ne, ne_brem, ne_supp):
    """Static metadata persisted before the Zhai batch job is submitted."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {ZHAI_STEM}",
            "quick: False",
            f"ne: {ne}",
            f"ne_brem: {ne_brem}",
            f"ne_supp: {ne_supp}",
            "",
        ]
    )


def _write_job_script_command(jobdir, metadata):
    """Exclusively create a job directory, then receive its script on stdin."""
    return (
        f"mkdir '{jobdir}' && cat > '{jobdir}/run.sh' && "
        f"printf %s {shlex.quote(metadata)} > '{jobdir}/meta'"
    )


def _stage_job_script(jobid: str, stems: list[str], upload: str, script: str) -> None:
    """Reserve stems and upload a batch script, releasing on upload failure."""
    _run(["ssh", "-n", HOST, _reserve_checkpoint_stems_command(jobid, stems)])
    try:
        subprocess.run(
            ["ssh", HOST, upload],
            input=script.replace("\r\n", "\n").encode(),
            check=True,
        )
    except BaseException:
        _run(["ssh", "-n", HOST, _release_checkpoint_stems_command(jobid, stems)])
        raise


def _submission_outcome(jobid: str) -> str:
    """Classify a submission whose SSH response was lost, conservatively."""
    _check_shell_tokens([jobid])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    remote = (
        f'D="{jobdir}"; '
        '[ -d "$D" ] || { echo missing; exit 0; }; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        "case \"$SID\" in *[!0-9]*|'') ;; *) echo submitted; exit 0 ;; esac; "
        'STATE=$(cat "$D/state" 2>/dev/null || true); '
        'case "$STATE" in '
        '"FAILED (sbatch submission)"*) echo failed ;; '
        "queued*|running*|cancelling*) echo pending ;; "
        "*) echo unknown ;; esac"
    )
    return _ssh_capture(remote).strip()


def _release_if_submission_definitely_failed(jobid: str, stems: list[str]) -> None:
    """Release staging locks only after remote state proves ``sbatch`` failed."""
    try:
        definitely_failed = _submission_outcome(jobid) == "failed"
    except BaseException:
        return
    if definitely_failed:
        _run(["ssh", "-n", HOST, _release_checkpoint_stems_command(jobid, stems)])


def _submit_staged_job(jobid: str, stems: list[str], *, nice: bool = False) -> str:
    """Submit an uploaded script without freeing locks after an ambiguous SSH loss."""
    try:
        scheduler_id = _ssh_capture(_submit_slurm_command(jobid, stems, nice=nice)).strip()
    except BaseException:
        _release_if_submission_definitely_failed(jobid, stems)
        raise
    if not scheduler_id.isdigit():
        _release_if_submission_definitely_failed(jobid, stems)
        raise SystemExit(f"SLURM submission for job {jobid} returned no scheduler ID")
    return scheduler_id


def _squeue_state_command(scheduler_id: str, *, retired: str) -> str:
    """Build fail-closed Bash that stores a live SLURM state in ``STATE``.

    SLURM reports a recently retired ID as a nonzero error on some clusters;
    callers provide the control-flow action that means "not queued" in their
    surrounding shell context. Every other query failure remains fatal.
    """
    return (
        f"STATE=$(squeue -h -j {scheduler_id} -o '%T' 2>&1); STATUS=$?; "
        'if [ "$STATUS" -ne 0 ]; then case "$STATE" in '
        f'*"Invalid job id specified"*) {retired} ;; '
        f'*) echo "could not query SLURM job {scheduler_id}" >&2; exit "$STATUS" ;; esac; fi; '
    )


def _live_jobs():
    """[(jobid, quick, [materials])] for jobs still reported by SLURM.

    Legacy job directories without a recorded scheduler ID are deliberately
    non-live: PID liveness is not a safe fallback for scheduler-managed work.
    """
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; [ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$d/meta" 2>/dev/null | tail -1); '
        "case \"$SID\" in ''|*[!0-9]*) continue ;; esac; "
        + _squeue_state_command("$SID", retired="continue")
        + '[ -n "$STATE" ] || continue; '
        'q=$(sed -n "s/^quick: //p" "$d/meta" 2>/dev/null); '
        'm=$(sed -n "s/^materials: //p" "$d/meta" 2>/dev/null); '
        'printf "%s\\t%s\\t%s\\n" "$(basename "$d")" "$q" "$m"; done'
    )
    jobs = []
    for line in _ssh_capture(remote).splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        jobid, q, mats = parts[0].strip(), parts[1].strip(), parts[2].split()
        jobs.append((jobid, q == "True", mats))
    return jobs


def _reservation_holders(stems: list[str]) -> list[tuple[str, str]]:
    """Return checkpoint reservations that must block destructive operations.

    A reservation is acquired before ``sbatch``.  It is therefore authoritative
    for avoiding checkpoint deletion even if submission was ambiguous and no
    scheduler ID was ever recorded in the job metadata.
    """
    _check_shell_tokens(stems)
    remote = (
        f'R="{_reservation_root()}"; [ -d "$R" ] || exit 0; '
        f"for stem in {' '.join(stems)}; do "
        '[ -d "$R/$stem" ] || continue; '
        'OWNER=$(cat "$R/$stem/jobid" 2>/dev/null) || OWNER="unknown"; '
        'printf "%s\\t%s\\n" "$stem" "$OWNER"; done'
    )
    holders = []
    for line in _ssh_capture(remote).splitlines():
        stem, separator, owner = line.partition("\t")
        if separator and stem in stems:
            holders.append((stem, owner or "unknown"))
    return holders


def _clear_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Atomically reserve and delete checkpoint stems on the remote box.

    The temporary reservation spans the delete itself, closing the interval
    between a clear's liveness check and ``rm`` where a concurrent ``start``
    could otherwise claim the same checkpoint.  The EXIT trap releases only
    reservations owned by this clear, including after a failed deletion.
    """
    _check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    stem_words = " ".join(stems)
    releases = " ".join(
        f'if [ "$(cat "$R/{stem}/jobid" 2>/dev/null)" = "$J" ]; then rm -rf "$R/{stem}"; fi;'
        for stem in stems
    )
    return f'''R="{reservations}"; J="{jobid}"; C="{REMOTE_DIR}/checkpoints"; \
mkdir -p "$R" || exit $?; \
release() {{ {releases} }}; trap release EXIT; \
for stem in {stem_words}; do \
  if mkdir "$R/$stem" 2>/dev/null; then printf '%s\\n' "$J" > "$R/$stem/jobid"; \
  else printf 'RESERVED\\t%s\\n' "$stem"; exit 0; fi; \
done; \
cd "$C" 2>/dev/null || exit 0; \
for stem in {stem_words}; do f="$stem.pkl"; [ -f "$f" ] || continue; rm -f "$f" || exit $?; \
  printf 'CLEARED\\t%s\\n' "$f"; done'''


def _slurm_job_id(jobid: str) -> str | None:
    """Return a queue's recorded numeric scheduler ID, if it has one."""
    _check_shell_tokens([jobid])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    scheduler_id = _ssh_capture(
        f'D="{jobdir}"; sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1'
    ).strip()
    return scheduler_id if scheduler_id.isdigit() else None


def _slurm_state(slurm_job_id: str) -> str | None:
    """Return the live SLURM state, or ``None`` after it leaves ``squeue``."""
    if not slurm_job_id.isdigit():
        return None
    state = _ssh_capture(
        _squeue_state_command(slurm_job_id, retired="exit 0") + 'printf "%s\\n" "$STATE"'
    ).strip()
    return state.splitlines()[0] if state else None


def _job_state(jobid: str) -> str:
    """Return a job's persisted terminal/progress state without inferring liveness."""
    _check_shell_tokens([jobid])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    return _ssh_capture(f'cat "{jobdir}/state" 2>/dev/null').strip()


def _job_metadata(jobid: str) -> str:
    """Return the persisted queue metadata for attach-mode selection."""
    _check_shell_tokens([jobid])
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    return _ssh_capture(
        f'D="{jobdir}"; [ -d "$D" ] || {{ echo "no such job: {jobid}" >&2; exit 1; }}; '
        'cat "$D/meta" 2>/dev/null'
    )


def _job_succeeded(jobid: str) -> bool:
    """Whether the batch script recorded a successful terminal state."""
    return _job_state(jobid).startswith("done")


def _reservation_ledger() -> tuple[int, list[tuple[str, str, int]]]:
    """Return ``(now_epoch, [(stem, jobid, mtime_epoch)])`` for every held reservation.

    One ``ssh`` round trip enumerates the whole reservation root and stamps the
    box's own clock, so reservation age is measured against the box -- never the
    caller -- keeping the staging-race guard immune to client/host clock skew.
    """
    remote = (
        f'R="{_reservation_root()}"; [ -d "$R" ] || exit 0; '
        'printf "NOW\\t%s\\n" "$(date +%s)"; '
        'for d in "$R"/*/; do [ -d "$d" ] || continue; '
        's=$(basename "$d"); '
        'j=$(cat "$d/jobid" 2>/dev/null) || j="unknown"; '
        'm=$(stat -c %Y "$d" 2>/dev/null) || m=0; '
        'printf "%s\\t%s\\t%s\\n" "$s" "$j" "$m"; done'
    )
    now = 0
    rows: list[tuple[str, str, int]] = []
    for line in _ssh_capture(remote).splitlines():
        parts = line.split("\t")
        if parts[0] == "NOW" and len(parts) == 2 and parts[1].isdigit():
            now = int(parts[1])
        elif len(parts) == 3:
            stem, jobid, mtime = parts
            rows.append(
                (stem.strip(), jobid.strip() or "unknown", int(mtime) if mtime.isdigit() else 0)
            )
    return now, rows


def _is_reapable(live_state: str | None, age_seconds: int, min_age_seconds: float) -> bool:
    """Decide whether a reservation-holding job is a safe reap target.

    ``live_state`` is the job's current SLURM state (e.g. ``"RUNNING"``), or
    ``None`` when the scheduler has no allocation recorded for it. ``age_seconds``
    is how long the job's newest reservation has sat untouched on the box;
    ``min_age_seconds`` is the staging-race guard.
    """
    # Reap only when both hold: no live SLURM allocation (dead), and the newest
    # lock is older than the staging-race guard (not a job mid-submit).
    return live_state is None and age_seconds >= min_age_seconds


def _orphaned_reservation_jobs(
    min_age_seconds: float,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Partition reservation holders into reapable orphans and protected jobs.

    Returns ``(orphans, protected)``, each mapping ``jobid -> sorted[stem]``.
    Liveness reuses the same recorded-scheduler-ID + ``squeue`` signal as
    ``_live_jobs``; PID guessing is never a safe fallback for scheduler work.
    """
    now, rows = _reservation_ledger()
    by_job: dict[str, list[tuple[str, int]]] = {}
    for stem, jobid, mtime in rows:
        by_job.setdefault(jobid, []).append((stem, mtime))
    orphans: dict[str, list[str]] = {}
    protected: dict[str, list[str]] = {}
    for jobid, entries in by_job.items():
        stems = sorted(stem for stem, _m in entries)
        scheduler_id = _slurm_job_id(jobid) if jobid != "unknown" else None
        live_state = _slurm_state(scheduler_id) if scheduler_id else None
        newest = max(mtime for _s, mtime in entries)
        age = now - newest if now else 0
        if _is_reapable(live_state, age, min_age_seconds):
            orphans[jobid] = stems
        else:
            protected[jobid] = stems
    return orphans, protected


def _reap_job_command(jobid: str) -> str:
    """Remote command: release a job's reservations, then stamp its state terminal."""
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    release = _release_job_reservations_command(jobid)
    return f'{release}; echo "reaped (orphan reservations released) $(date -Is)" > "{jobdir}/state"'


def _refuse_if_busy(materials, quick):
    """Abort `start` if a live job is already producing any checkpoint this run
    would write. Two runs writing the same `<stem>.pkl` share one `<stem>.pkl.tmp`
    and race on `os.replace` -- the first rename consumes the temp, the second
    dies with FileNotFoundError (see run.py:_checkpoint_save). Comparing *stems*
    (material, or material_quick) not bare materials lets a `--quick` smoke test
    run alongside a full sweep of the same material, since they write different
    files."""
    wanted = set(_stems(materials, quick))
    busy = [
        (jid, sorted(clash))
        for jid, jquick, jmats in _live_jobs()
        if (clash := wanted.intersection(_stems(jmats, jquick)))
    ]
    if busy:
        detail = "\n".join(f"  job {jid} is running -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to start: a live job is already producing the same "
            "checkpoint(s), and two runs writing one <stem>.pkl race on its "
            f".tmp and crash.\n{detail}\n"
            "attach to it (cxr remote attach <jobid>) or stop it "
            "(cxr remote stop <material>) first, or run different materials."
        )


def clear_remote(materials, yes=False):
    """Delete one or more materials' accumulated checkpoints on the box: both
    ``checkpoints/<material>.pkl`` and ``checkpoints/<material>_quick.pkl`` for
    each material.  Accepts a single crystal key or a list.

    Refuses (before touching anything) if a live job or a pre-submission
    reservation protects any stem.  Without ``yes`` this is a safe dry preview:
    it prints exactly which files exist and would be deleted, then stops. With
    ``yes`` it ``rm -f``s them and reports what went."""
    if isinstance(materials, str):
        materials = [materials]
    _check_materials(materials)  # interpolated into a remote shell command
    label = ", ".join(materials)
    wanted = {stem for m in materials for stem in (m, f"{m}_quick")}
    busy = [
        (jid, sorted(clash))
        for jid, jquick, jmats in _live_jobs()
        if (clash := wanted.intersection(_stems(jmats, jquick)))
    ]
    if busy:
        detail = "\n".join(f"  job {jid} is producing -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to clear: a live job is still producing one of these "
            f"checkpoints, and clearing it would race a running sweep.\n{detail}\n"
            "stop it (cxr remote stop <material>) first, or wait for it to finish."
        )
    stems = [stem for m in materials for stem in (m, f"{m}_quick")]
    if yes:
        outcome = _ssh_capture(
            _clear_checkpoint_stems_command(f"clear-{_new_jobid()}", stems)
        ).splitlines()
        reservations = [line.split("\t", 1)[1] for line in outcome if line.startswith("RESERVED\t")]
        if reservations:
            raise SystemExit(
                "refusing to clear: a checkpoint reservation is still active for "
                f"{', '.join(reservations)}. Wait for submission to resolve, or stop "
                "the recorded job before clearing."
            )
        existing = [line.split("\t", 1)[1] for line in outcome if line.startswith("CLEARED\t")]
        if not existing:
            print(f"(nothing to clear for {label})")
            return
        print("cleared on the box:")
        for f in existing:
            print(f"  checkpoints/{f}")
        return
    reservations = _reservation_holders(stems)
    if reservations:
        detail = "\n".join(
            f"  reservation {owner} protects -> {stem}" for stem, owner in reservations
        )
        raise SystemExit(
            "refusing to clear: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, or stop the recorded job before clearing."
        )
    # which stems actually exist on the box; the `|| true` keeps a missing last
    # stem's failed `[ -f ]` from becoming the loop's -- and hence ssh's -- exit
    # status, which would make _ssh_capture abort the whole clear
    pkl_names = " ".join(f"{stem}.pkl" for stem in stems)
    listing = (
        f"cd {REMOTE_DIR}/checkpoints 2>/dev/null || exit 0; "
        f'for f in {pkl_names}; do [ -f "$f" ] && echo "$f" || true; done'
    )
    existing = _ssh_capture(listing).split()
    if not existing:
        print(f"(nothing to clear for {label})")
        return
    if not yes:
        print("would delete on the box (re-run with --yes to delete):")
        for f in existing:
            print(f"  checkpoints/{f}")
        return


def clear_all_remote(yes=False):
    """Empty the box's ``checkpoints/`` directory: delete every ``*.pkl`` file
    under it (recursively, so per-reproduction subdirs are included too).

    Refuses (before touching anything) if any live job is running or any
    checkpoint reservation is held: both signal an in-flight sweep whose output
    a blanket clear would destroy or race. Without ``yes`` this is a safe dry
    preview: it lists the files that would be deleted, then stops. With ``yes``
    it deletes them and reports the count."""
    live = _live_jobs()
    if live:
        detail = "\n".join(
            f"  job {jid} is producing -> {', '.join(sorted(_stems(jmats, jquick)))}"
            for jid, jquick, jmats in live
        )
        raise SystemExit(
            "refusing to clear --all: live job(s) are still producing checkpoints, "
            f"and clearing would race running sweeps.\n{detail}\n"
            "stop them (cxr remote stop --all) first, or wait for them to finish."
        )
    _now, reservations = _reservation_ledger()
    if reservations:
        detail = "\n".join(
            f"  reservation {jobid} protects -> {stem}" for stem, jobid, _mtime in reservations
        )
        raise SystemExit(
            "refusing to clear --all: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, reap orphans (cxr remote reap), or stop the "
            "recorded job before clearing."
        )
    # list first (dry preview), then delete only under --yes. `|| true` keeps a
    # missing checkpoints/ dir or a find failure from becoming ssh's exit status.
    listing = (
        f"cd {REMOTE_DIR}/checkpoints 2>/dev/null || exit 0; "
        r'find . -type f -name "*.pkl" 2>/dev/null | sed "s|^\./||" | sort || true'
    )
    existing = _ssh_capture(listing).split()
    if not existing:
        print("(nothing to clear: checkpoints/ holds no .pkl files)")
        return
    if not yes:
        print(f"would delete on the box (re-run with --yes to delete) -- {len(existing)} file(s):")
        for f in existing:
            print(f"  checkpoints/{f}")
        return
    _ssh_capture(
        f"cd {REMOTE_DIR}/checkpoints 2>/dev/null || exit 0; "
        r'find . -type f -name "*.pkl" -delete'
    )
    print(f"cleared on the box: {len(existing)} checkpoint file(s) under checkpoints/")


def start_queue(
    materials,
    quick=False,
    workers=None,
    no_sync=False,
    dry_run=False,
    parallel_materials=None,
    chunk_minutes=10.0,
):
    """Submit a material queue to SLURM. Returns its job id.

    By default the queue is chunked: each ~chunk_minutes slice resumes from
    checkpoint, does bounded work via ``scan.py --max-minutes``, and either
    terminates the chain or resubmits itself with ``--nice=10000`` so other
    users of the single-GPU box get priority at every slice boundary. Pass
    ``chunk_minutes=0`` for the original monolithic allocation, which is the
    only mode that accepts ``parallel_materials``.
    """
    _check_materials(materials)
    chunked = chunk_minutes > 0
    if chunked and parallel_materials is not None:
        raise SystemExit(
            "--parallel-materials only applies to a monolithic allocation; "
            "pass --chunk-minutes 0 to use it"
        )
    if not dry_run:
        _refuse_if_busy(materials, quick)
    jobid = _new_jobid()
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    stems = _stems(materials, quick)
    if chunked:
        parallel_materials = None
        payload = _chunked_queue_script(jobid, materials, quick, workers, chunk_minutes)
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        parallel_materials = _validate_parallel_materials(
            DEFAULT_PARALLEL_MATERIALS if parallel_materials is None else parallel_materials
        )
        payload = _queue_script(jobid, materials, quick, workers, parallel_materials)
        time_limit = SLURM_TIME
    script = _slurm_batch_script(
        jobid, payload, job_name=f"cxr-{jobid}", reservation_stems=stems, time_limit=time_limit
    )
    upload = _write_job_script_command(
        jobdir,
        _queue_metadata(jobid, materials, quick, workers, parallel_materials, chunk_minutes),
    )
    submit = _submit_slurm_command(jobid, stems, nice=chunked)

    if dry_run:
        print(f"# job {jobid}: {' '.join(materials)}{' (quick)' if quick else ''}")
        print(f"# --- ssh {HOST}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {HOST}: {submit}")
        return jobid

    if not no_sync:
        sync_code()
    # create the job dir and write run.sh (script piped over stdin).
    # Send as LF-only bytes: text=True on Windows translates \n->\r\n,
    # which produces a CRLF run.sh that bash silently refuses to execute.
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems, nice=chunked)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + _format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", HOST),
                ("Materials", ", ".join(materials)),
                (
                    "Mode",
                    _mode_summary(
                        _queue_metadata(
                            jobid, materials, quick, workers, parallel_materials, chunk_minutes
                        )
                    ),
                ),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", f"cxr remote pull {' '.join(stems)}  (after completion)"),
            ]
        )
    )
    return jobid


def start_zhai_queue(
    ne=20_000,
    ne_brem=200,
    ne_supp=200,
    tmd_azimuth=0.0,
    refresh=False,
    no_sync=False,
    dry_run=False,
):
    """Submit a Zhai-reproduction batch job to SLURM. Returns the local job id."""
    if not dry_run:
        _refuse_if_busy([ZHAI_STEM], False)
    jobid = _new_jobid()
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    stems = [ZHAI_STEM]
    payload = _zhai_queue_script(jobid, ne, ne_brem, ne_supp, tmd_azimuth, refresh)
    script = _slurm_batch_script(
        jobid, payload, job_name=f"cxr-zhai-{jobid}", reservation_stems=stems
    )
    upload = _write_job_script_command(jobdir, _zhai_queue_metadata(jobid, ne, ne_brem, ne_supp))
    submit = _submit_slurm_command(jobid, stems)

    if dry_run:
        print(f"# zhai job {jobid}: ne={ne} ne_brem={ne_brem} ne_supp={ne_supp}")
        print(f"# --- ssh {HOST}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {HOST}: {submit}")
        return jobid

    if not no_sync:
        sync_code()
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + _format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", HOST),
                ("Workload", "Zhai reproduction"),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", "cxr remote check --pull  (after completion)"),
            ]
        )
    )
    return jobid


def _recorded_job_dirs_command() -> str:
    """Shell fragment that emits only submitted job directories, in order.

    ``jobs/reservations`` is bookkeeping for checkpoint ownership, not a job.
    A submitted job has its metadata file written before ``sbatch`` runs, so
    that marker also excludes incomplete or unrelated directories safely.
    """
    return (
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; printf "%s\\n" "$(basename "$d")"; done'
    )


def list_jobs():
    """Print every submitted job with identifying metadata, oldest first."""
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; '
        '[ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$d/meta" 2>/dev/null | tail -1); '
        'Q=$(sed -n "s/^quick: //p" "$d/meta" 2>/dev/null | tail -1); '
        'M=$(sed -n "s/^materials: //p" "$d/meta" 2>/dev/null | tail -1); '
        'S=$(cat "$d/state" 2>/dev/null | tr "\\n\\t" "  "); '
        'printf "%s\\t%s\\t%s\\t%s\\t%s\\n" "$(basename "$d")" '
        '"${SID:--}" "${Q:-?}" "${M:--}" "${S:-(no state yet)}"; done'
    )
    rows = []
    for line in _ssh_capture(remote).splitlines():
        fields = line.split("\t", 4)
        if len(fields) != 5:
            continue
        jobid, scheduler_id, quick, materials, state = fields
        mode = "quick" if quick == "True" else "standard" if quick == "False" else "?"
        rows.append((jobid, scheduler_id, mode, materials.replace(" ", ", "), state))
    if not rows:
        print(f"No remote jobs on {HOST}. Start one with `cxr remote start <materials>`.")
        return
    print(_style_states(_format_table(("JOB", "SLURM", "MODE", "MATERIALS", "LAST EVENT"), rows)))


def _job_assign(jobid):
    """Bash that sets JOB to the given id, or the latest job dir if none given."""
    if jobid:
        _check_shell_tokens([jobid])
        return f'JOB="{jobid}"'
    return f"JOB=$({_recorded_job_dirs_command()} | tail -1)"


def _status_remote_command(job_assign, detail):
    """One round-trip that emits the marked sections ``_format_job_status`` reads.

    Progress snapshots are cheap per-material JSON, so they ship at every
    verbosity (the overall bar and CASE PROGRESS block render at 0/1/2 alike);
    only the 32 KiB log tail is gated behind ``-vv``. Shared verbatim by the
    one-shot ``status`` and the live ``attach`` loop so both render identically.
    """
    slurm_detail = (
        'squeue -h -j "$SID" -o '
        "'job_id=%i|state=%T|name=%j|partition=%P|elapsed=%M|left=%L|nodes=%D|reason=%R'; "
        if detail >= 1
        else 'printf "job_id=%s|state=%s\\n" "$SID" "$STATE"; '
    )
    progress = (
        'echo "@@PROGRESS"; for f in "$D"/progress/*.json; do '
        '[ -f "$f" ] || continue; cat "$f" 2>/dev/null || true; printf "\\n"; done; '
    )
    log = 'echo "@@LOG"; tail -c 32768 "$D/log" 2>/dev/null; ' if detail >= 2 else ""
    return (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; {job_assign}; '
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        'echo "@@JOB"; printf "%s\\n" "$JOB"; '
        'echo "@@META"; cat "$D/meta" 2>/dev/null; '
        'echo "@@STATE"; cat "$D/state" 2>/dev/null; '
        'echo "@@SQUEUE"; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        'case "$SID" in \'\'|*[!0-9]*) echo "job_id=-|state=NOT_QUEUED" ;; '
        "*) "
        + _squeue_state_command("$SID", retired="STATE=")
        + 'if [ -n "$STATE" ]; then '
        + slurm_detail
        + 'else printf "job_id=%s|state=NOT_QUEUED\\n" "$SID"; fi ;; esac; '
        + progress
        + log
    )


def job_status(jobid=None, detail=0):
    """Print one structured job report; verbosity adds allocation and log tail."""
    if detail < 0:
        raise ValueError("detail must be non-negative")
    output = _ssh_capture(_status_remote_command(_job_assign(jobid), detail))
    sections = _marked_sections(output)
    if not sections:
        print(output, end="")
        return
    print(_style_states(_format_job_status(sections, detail)))


def tail_logs(jobid=None, follow=False):
    """Tail a job's log. With --follow, stream live (blocks until Ctrl-C)."""
    tail = "tail -f" if follow else "tail -n 60"
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; {_job_assign(jobid)}; '
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        f'printf "LOG %s · {HOST}\\n\\n" "$JOB"; '
        f'{tail} "$D/log"'
    )
    if follow:
        try:
            # Close stdin so Windows OpenSSH cannot hang on console forwarding,
            # while stdout/stderr still inherit for live streaming and Ctrl-C.
            subprocess.run(["ssh", "-n", HOST, remote])
        except KeyboardInterrupt:
            print("\n(stopped following; the job is unaffected)")
    else:
        print(_ssh_capture(remote), end="")


def _latest_jobid():
    """The most recent job id on the box (job dirs are timestamp-named), or None."""
    out = _ssh_capture(
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; {_recorded_job_dirs_command()} | tail -1'
    ).strip()
    return out or None


def _disconnect_hint(jobid):
    print(
        f"\n\nVIEWER DISCONNECTED · job {jobid} keeps running on {HOST}\n"
        f"  Reconnect  cxr remote attach {jobid}\n"
        f"  Status     cxr remote status {jobid} -vv\n"
        "  Stop       cxr remote stop <material>"
    )


# How many consecutive not-live polls the attach viewers tolerate before the
# chain is declared broken: ~30 s at the 2 s poll, covering normal inter-slice
# SLURM latency (a slice exits, the queued next slice has not started yet)
# without masking a chain whose resubmission actually failed.
_POLL_GRACE_POLLS = 15


def _is_terminal_state(state):
    """Chain-terminal persisted states (spec 3c): done / FAILED / cancelled."""
    return state.startswith(("done", "FAILED", "cancelled"))


def _render_frame(frame, *, tty):
    """Repaint one attach frame: in place on a tty, appended when piped."""
    if tty:
        # Home the cursor, clear the screen and scrollback so each poll
        # overwrites the previous frame instead of scrolling -- the
        # "continually updating" status view.
        sys.stdout.write("\x1b[H\x1b[2J\x1b[3J" + frame + "\n")
        sys.stdout.flush()
    else:
        print(frame)
        print("─" * 60)


def _attach_header(refresh):
    """One-line banner above each live frame; the counter proves it's polling."""
    return _paint(
        f"ATTACHED · {HOST} · refresh {refresh} · Ctrl-C detaches (job keeps running)",
        "inactive",
    )


def _live_status(jobid, detail):
    """Re-render ``status <jobid>`` at ``detail`` each poll until the job is terminal.

    The viewer is read-only: Ctrl-C or a dropped SSH link tears down only this
    loop, never the SLURM job. A chunked chain hops scheduler IDs between
    slices, so the shared status command re-reads the latest recorded ID each
    poll; the same ``_POLL_GRACE_POLLS`` watchdog declares a chain broken only
    after ~30 s with no live allocation and a non-terminal recorded state.
    Returns True once the job reaches a terminal state, False on viewer
    disconnect or a stalled chain.
    """
    remote = _status_remote_command(_job_assign(jobid), detail)
    tty = _color_enabled()
    refresh = 0
    missed = 0
    state = ""
    broken = False
    try:
        while True:
            refresh += 1
            output = _ssh_capture(remote)
            sections = _marked_sections(output)
            if not sections:
                print(output, end="")
                return False
            state = sections.get("STATE", "")
            scheduler = _scheduler_fields(sections.get("SQUEUE", ""))
            live = scheduler.get("state", "") not in ("", "NOT_QUEUED")
            frame = (
                _attach_header(refresh)
                + "\n\n"
                + _style_states(_format_job_status(sections, detail))
            )
            _render_frame(frame, tty=tty)
            if _is_terminal_state(state):
                break
            missed = 0 if live else missed + 1
            if missed >= _POLL_GRACE_POLLS:
                broken = True
                break
            time.sleep(2)
    except KeyboardInterrupt:
        _disconnect_hint(jobid)
        return False
    if broken:
        print(
            f"\nJOB {jobid} · CHAIN STALLED\n"
            "  No live SLURM allocation for ~30 s; recorded state is not terminal.\n"
            f"  Inspect  cxr remote status {jobid} -vv\n"
            f"  Logs     cxr remote logs {jobid}"
        )
        return False
    print(f"\nJOB {jobid} · FINISHED\n  State   {state}\n  Inspect cxr remote status {jobid} -vv")
    return True


def attach(jobid=None, detail=0):
    """Live-track a job: re-render its ``status`` report at ``detail`` until terminal.

    Equivalent to ``cxr remote status [-v|-vv]``, but the same report repaints
    in place every ~2 s. Ctrl-C (or a dropped link) detaches the viewer only;
    the SLURM job keeps running. Returns True once the job reaches a terminal
    state, False on disconnect or a stalled chain -- callers (scan/check) key
    their auto-pull off that.
    """
    jobid = jobid or _latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to attach to (start one: cxr remote start <materials>)")
    _check_shell_tokens([jobid])
    return _live_status(jobid, detail)


def _stop_jobid(jobid):
    """Cancel one active scheduler job and record the terminal job state."""
    _check_shell_tokens([jobid])
    scheduler_id = _slurm_job_id(jobid)
    if scheduler_id is None or _slurm_state(scheduler_id) is None:
        raise SystemExit(f"job {jobid} is not an active SLURM job")
    release = _release_job_reservations_command(jobid)
    # Write the STOP sentinel BEFORE scancel (spec 3b): if the cancelled slice
    # was already past its scan loop and about to resubmit, the next slice's
    # STOP check still terminates the chain instead of re-queueing it.
    remote = (
        f'D="{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"; '
        ': > "$D/STOP"; '
        f"scancel {scheduler_id} || exit $?; "
        f'echo "cancelling [{scheduler_id}] $(date -Is)" > "$D/state"; '
        "while :; do "
        + _squeue_state_command(str(scheduler_id), retired="break")
        + '[ -n "$STATE" ] || break; sleep 1; done; '
        f"{release}; "
        f'echo "cancelled [{scheduler_id}] $(date -Is)" > "$D/state"; '
        f'echo "cancelled SLURM job {scheduler_id} for job {jobid}"'
    )
    _run(["ssh", "-n", HOST, remote])


def stop_jobs(materials=None, all_jobs=False):
    """Stop live queue jobs by material name, or every live job with ``all_jobs``.

    Each material can only be owned by one live job because start/scan refuse
    checkpoint-stem collisions, so material names are the useful user-facing
    handle and job ids stay an internal implementation detail.
    """
    if all_jobs:
        if materials:
            raise SystemExit("stop --all does not take material names")
        live = _live_jobs()
        jobids = [jobid for jobid, _quick, _materials in live]
        if not jobids:
            print("(no live jobs to stop)")
            return
    else:
        if not materials:
            raise SystemExit("stop needs material(s), or use --all")
        _check_shell_tokens(materials)
        wanted = set(materials)
        live = _live_jobs()
        matches = [
            (jobid, wanted.intersection(jmats))
            for jobid, _quick, jmats in live
            if wanted.intersection(jmats)
        ]
        found = {material for _jobid, matched in matches for material in matched}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit("no live job found for material(s): " + ", ".join(missing))
        jobids = sorted({jobid for jobid, _matched in matches})

    for jobid in jobids:
        _stop_jobid(jobid)


def reap_reservations(min_age_minutes=5.0, yes=False):
    """Release checkpoint reservations orphaned by hard-killed jobs.

    A reservation is a ``mkdir`` lock released by the batch job's EXIT trap. A
    hard kill (OOM, node reboot, ``kill -9``) skips the trap and strands every
    lock, wedging ``start``/``scan`` with no recovery path -- ``stop`` refuses a
    job the scheduler no longer knows. This reaps locks whose owning job has no
    SLURM allocation and whose newest lock predates the staging-race guard.

    Without ``yes`` this is a safe dry preview. With ``yes`` it releases the
    orphans and stamps each reaped job's state terminal.
    """
    min_age_seconds = max(0.0, min_age_minutes) * 60
    orphans, protected = _orphaned_reservation_jobs(min_age_seconds)
    if not orphans and not protected:
        print("(no active checkpoint reservations on the box)")
        return
    for jobid, stems in sorted(protected.items()):
        print(f"keeping {jobid}: {len(stems)} reservation(s) held by a live or too-recent job")
    if not orphans:
        print("(no orphaned reservations to reap)")
        return
    for jobid, stems in sorted(orphans.items()):
        verb = "releasing" if yes else "would release"
        print(f"{verb} {jobid}: {len(stems)} orphan reservation(s) -> {', '.join(stems)}")
    if not yes:
        print("re-run with --yes to release them")
        return
    for jobid in sorted(orphans):
        _run(["ssh", "-n", HOST, _reap_job_command(jobid)])
    print(f"reaped {len(orphans)} orphaned job(s)")


def _ensure_utf8_stdio():
    """The box's output is UTF-8 (job logs embed tqdm block-glyph progress bars
    like `████▌`). On Windows stdout defaults to cp1252 -- and when it is,
    printing that text raises UnicodeEncodeError, so `status`/`jobs`/`logs`/
    `attach` would crash on the glyphs. Force UTF-8 with replacement so they
    never do. Scoped to remote subcommands (called from their CLI wrappers, not
    at import time) so it doesn't change stdio encoding for unrelated ``cxr``
    commands."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


# ---- CLI wiring ----------------------------------------------------------------
def _dispatch(handler):
    def _run_cli(args):
        _ensure_utf8_stdio()
        return handler(args)

    return _run_cli


def _selected_materials(args, attribute):
    """Resolve explicit material arguments or the shared ``--all`` manifest."""
    explicit = getattr(args, attribute)
    if args.all:
        if explicit:
            raise SystemExit(f"{args.remote_command} --all does not take material names")
        return load_all_materials(MATS_FILE)
    if explicit:
        return explicit if isinstance(explicit, list) else [explicit]
    raise SystemExit(f"{args.remote_command} needs material name(s), or use --all")


def _cli_scan(args):
    if args.quick and args.grid:
        raise SystemExit(
            "scan --quick --grid: quick checkpoints aren't grid-filterable "
            "(their grid isn't reproducible from material_sweep), so the "
            "trailing pull would fail after the whole sweep ran. Drop --grid."
        )
    materials = _selected_materials(args, "material")
    # `start_queue` validates material tokens, refuses checkpoint collisions,
    # syncs once, and submits one bounded-concurrency scheduler job for the materials.
    jobid = start_queue(
        materials,
        quick=args.quick,
        workers=args.workers,
        parallel_materials=getattr(args, "parallel_materials", None),
        chunk_minutes=getattr(args, "chunk_minutes", 10.0),
        no_sync=args.no_sync,
    )
    if not attach(jobid):
        print("scan is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = _completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM scan produced no successful checkpoints; nothing to pull")
        return
    stems = _stems(completed, args.quick)
    # Code is already synced by the queue, so a grid pull skips its own sync;
    # preserve the requested grid/trim policy.
    pull(
        stems,
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}.pkl is local; run `cxr analyze {stem}` "
            f"(or run `cxr export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_pull(args):
    pull(
        _selected_materials(args, "material"),
        grid=not args.full,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        no_sync=args.no_sync,
    )


def _cli_start(args):
    materials = _selected_materials(args, "materials")
    jobid = start_queue(
        materials,
        quick=args.quick,
        workers=args.workers,
        parallel_materials=args.parallel_materials,
        chunk_minutes=args.chunk_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
    )
    if args.follow and not args.dry_run:
        attach(jobid)


def _cli_attach(args):
    attach(args.jobid, args.verbose)


def _cli_jobs(args):
    list_jobs()


def _cli_status(args):
    job_status(args.jobid, args.verbose)


def _cli_logs(args):
    tail_logs(args.jobid, args.follow)


def _cli_stop(args):
    stop_jobs(args.materials, args.all)


def _cli_reap(args):
    reap_reservations(min_age_minutes=args.min_age_minutes, yes=args.yes)


def _cli_clear(args):
    if args.all_checkpoints:
        if args.materials:
            args._clear_parser.error("clear --all takes no material argument")
        clear_all_remote(args.yes)
        return
    if not args.materials:
        args._clear_parser.error("clear needs material(s), or --all")
    clear_remote(args.materials, args.yes)


def _cli_sync(args):
    sync_code()


def _cli_check(args):
    if args.follow and not args.detached:
        args._check_parser.error("--follow requires --detached")
    if args.pull:
        pull_zhai_cache()
        return
    if args.detached:
        jobid = start_zhai_queue(
            ne=args.ne,
            ne_brem=args.ne_brem,
            ne_supp=args.ne_supp,
            tmd_azimuth=args.tmd_azimuth,
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
        tmd_azimuth=args.tmd_azimuth,
        refresh=args.refresh,
        no_sync=args.no_sync,
    )


def _build_remote_parser(ap):
    """Add every ``remote`` subcommand (scan/pull/start/attach/jobs/status/logs/
    stop/clear/sync) to ``ap``'s own subparsers. Nested one level under a
    ``remote`` group -- rather than flat alongside ``cxr scan`` etc -- because
    several of these names (``scan`` in particular) collide with top-level cxr
    subcommands that mean something different (a local sweep vs this
    sync+run-on-the-box+pull)."""
    sub = ap.add_subparsers(dest="remote_command", required=True)

    s = sub.add_parser("scan", help="sync code, submit sweep(s), wait, and pull checkpoints")
    s.add_argument("material", nargs="?")
    s.add_argument(
        "-a", "--all", action="store_true", help="run every material in mats_to_sim.toml"
    )
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument(
        "--parallel-materials",
        type=int,
        choices=range(1, MAX_PARALLEL_MATERIALS + 1),
        default=None,
        metavar="N",
        help="simultaneous material scans in one GPU allocation; only with "
        "--chunk-minutes 0, which defaults it to 2 (max: 4)",
    )
    s.add_argument(
        "--chunk-minutes",
        type=float,
        default=10.0,
        help="length of each self-resubmitting SLURM slice in minutes; "
        "0 = one whole-box monolithic run (default: 10.0)",
    )
    s.add_argument("--no-sync", action="store_true", help="skip the code upload")
    s.add_argument(
        "--grid", action="store_true", help="grid-filter the checkpoint on the box before pulling"
    )
    s.add_argument("--drop-wide-brem", action="store_true", help="with --grid: drop wide-brem too")
    s.add_argument("--downcast", action="store_true", help="with --grid: downcast to float32 too")
    s.set_defaults(func=_dispatch(_cli_scan))

    st = sub.add_parser(
        "start",
        help="sync code and submit a SLURM queue of materials (survives disconnect)",
    )
    st.add_argument("materials", nargs="*", help="one or more crystal keys")
    st.add_argument(
        "-a", "--all", action="store_true", help="queue every material in mats_to_sim.toml"
    )
    st.add_argument("--quick", action="store_true")
    st.add_argument("--workers", type=int, default=None)
    st.add_argument(
        "--parallel-materials",
        type=int,
        choices=range(1, MAX_PARALLEL_MATERIALS + 1),
        default=None,
        metavar="N",
        help="simultaneous material scans in one GPU allocation; only with "
        "--chunk-minutes 0, which defaults it to 2 (max: 4)",
    )
    st.add_argument(
        "--chunk-minutes",
        type=float,
        default=10.0,
        help="length of each self-resubmitting SLURM slice in minutes; "
        "0 = one whole-box monolithic run (default: 10.0)",
    )
    st.add_argument("--no-sync", action="store_true", help="skip the code upload")
    st.add_argument(
        "--dry-run",
        action="store_true",
        help="print the SLURM batch script + submission command, don't ssh",
    )
    st.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="track the job live after launching (Ctrl-C disconnects; job keeps running)",
    )
    st.set_defaults(func=_dispatch(_cli_start))

    at = sub.add_parser(
        "attach",
        help="live-track a job: status report, refreshed in place (Ctrl-C disconnects; default: latest)",
    )
    at.add_argument("jobid", nargs="?", default=None)
    at.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="add SLURM allocation detail; repeat for the recent log tail (mirrors status)",
    )
    at.set_defaults(func=_dispatch(_cli_attach))

    jb = sub.add_parser("jobs", help="list jobs with SLURM IDs, materials, and last events")
    jb.set_defaults(func=_dispatch(_cli_jobs))

    js = sub.add_parser(
        "status", help="show one job; -v allocation, -vv case progress and recent log"
    )
    js.add_argument("jobid", nargs="?", default=None)
    js.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="add SLURM allocation detail; repeat for case progress and recent log",
    )
    js.set_defaults(func=_dispatch(_cli_status))

    lg = sub.add_parser("logs", help="show a job's diagnostic log (default: latest)")
    lg.add_argument("jobid", nargs="?", default=None)
    lg.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="stream live; Ctrl-C disconnects without stopping the job",
    )
    lg.set_defaults(func=_dispatch(_cli_logs))

    sp = sub.add_parser(
        "stop",
        help="cancel active SLURM job(s) by material, or every live job",
        description="cancel active SLURM job(s) by material, or every live job",
    )
    sp.add_argument("materials", nargs="*", help="material name(s) owned by live jobs")
    sp.add_argument("-a", "--all", action="store_true", help="stop every live job")
    sp.set_defaults(func=_dispatch(_cli_stop))

    rp = sub.add_parser(
        "reap",
        help="release checkpoint reservations orphaned by hard-killed jobs (dry preview unless --yes)",
        description="release checkpoint reservations orphaned by hard-killed jobs "
        "whose owning SLURM allocation is gone (dry preview unless --yes)",
    )
    rp.add_argument(
        "--min-age-minutes",
        type=float,
        default=5.0,
        help="only reap reservations whose newest lock is older than this, guarding "
        "the reserve-before-submit window (default: 5.0)",
    )
    rp.add_argument(
        "--yes", action="store_true", help="actually release (default: dry preview only)"
    )
    rp.set_defaults(func=_dispatch(_cli_reap))

    p = sub.add_parser("pull", help="fetch one or more existing checkpoints from the box")
    p.add_argument("material", nargs="*", help="checkpoint stem(s), e.g. mose2 mose2_quick")
    p.add_argument(
        "-a", "--all", action="store_true", help="pull every material in mats_to_sim.toml"
    )
    p.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="pull the full, un-filtered checkpoint instead of the default grid-filtered pull",
    )
    p.add_argument(
        "--drop-wide-brem", action="store_true", help="with grid pull: drop wide-brem too"
    )
    p.add_argument(
        "--downcast", action="store_true", help="with grid pull: downcast to float32 too"
    )
    p.add_argument(
        "--no-sync", action="store_true", help="with grid pull: skip the pre-pull code sync"
    )
    p.set_defaults(func=_dispatch(_cli_pull))

    c = sub.add_parser(
        "clear", help="delete a material's accumulated checkpoints on the box, or --all"
    )
    c.add_argument(
        "materials",
        nargs="*",
        metavar="material",
        help="crystal key(s); clears both <material>.pkl and <material>_quick.pkl for each",
    )
    c.add_argument(
        "--all",
        dest="all_checkpoints",
        action="store_true",
        help="empty the entire checkpoints/ directory (mutually exclusive with a material)",
    )
    c.add_argument("--yes", action="store_true", help="actually delete (default: dry preview only)")
    c.set_defaults(func=_dispatch(_cli_clear), _clear_parser=c)

    sy = sub.add_parser("sync", help="push the current code to the box only")
    sy.set_defaults(func=_dispatch(_cli_sync))

    ck = sub.add_parser(
        "check",
        help="run the Zhai reproduction + supplementary MC on the box, pull caches back",
    )
    ck.add_argument(
        "--ne", type=int, default=20_000, help="Fig.1c anchor line electrons per energy"
    )
    ck.add_argument(
        "--ne-brem", default=200, type=int, help="Fig.1c anchor bremsstrahlung electrons per energy"
    )
    ck.add_argument(
        "--ne-supp", type=int, default=200, help="supplementary electrons per polar-tilt spectrum"
    )
    ck.add_argument(
        "--tmd-azimuth",
        type=float,
        default=0.0,
        help="exploratory azimuth for TMD studies whose azimuth is unreported",
    )
    ck.add_argument(
        "--refresh", action="store_true", help="recompute even if a matching cache exists"
    )
    ck.add_argument("--no-sync", action="store_true", help="skip the code upload")
    mode = ck.add_mutually_exclusive_group()
    mode.add_argument(
        "--detached",
        "-d",
        action="store_true",
        help="launch as a DETACHED job (survives disconnect)",
    )
    ck.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="with --detached: track the job live after launching",
    )
    mode.add_argument(
        "--pull", action="store_true", help="skip the run; just fetch existing zhai cache files"
    )
    ck.set_defaults(func=_dispatch(_cli_check), _check_parser=ck)

    return ap


def add_subparser(sub):
    """Register the ``remote`` subcommand group on an argparse subparsers
    object. Everything under it (``cxr remote scan|pull|start|attach|jobs|
    status|logs|stop|clear|sync``) is optional -- it only works with an ssh host
    configured to run sweeps on (default 'qlmc', see CXR_REMOTE_HOST)."""
    ap = sub.add_parser(
        "remote", help="[dev] push code + run/manage MC sweeps on a remote GPU box over ssh"
    )
    return _build_remote_parser(ap)


def main(argv=None):
    ap = _build_remote_parser(
        argparse.ArgumentParser(
            prog="cxr-remote", description="push code + run/manage MC sweeps on a remote GPU box"
        )
    )
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
