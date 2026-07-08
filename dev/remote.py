"""
remote.py -- run the heavy CXR scan on the GPU box, keep the data-vis local.

The split this enables: the laptop holds the project and does ALL the data-vis +
PDF export (where matplotlib and the xelatex/webpdf toolchain are set up), while
the lab box (an RTX 5080, ssh host 'qlmc') only does the GPU-heavy Monte-Carlo
sweep. This script ships the current code up, runs scan.py there, and pulls the
resulting checkpoint back into ./checkpoints -- so you never hand-ssh in or copy
files, and you never need a PDF toolchain on the lab box.

One-shot (foreground, holds the ssh session open until the sweep finishes):

    python dev/remote.py scan mose2               # sync code up, run sweep, pull checkpoint
    python dev/remote.py scan mose2 --quick       # tiny grid smoke test
    python dev/remote.py scan mose2 --no-sync     # skip the code upload (code unchanged)
    python dev/remote.py pull mose2 wse2          # fetch existing checkpoints (grid-filtered)
    python dev/remote.py pull mose2 --full        # fetch the full, un-filtered checkpoint
    python dev/remote.py sync                     # only push the current code

Detached QUEUE (survives ssh disconnect -- launch, walk away, reconnect later):

    python dev/remote.py start mose2 wse2 mos2    # queue several materials, run detached
    python dev/remote.py start mose2 --follow     # launch, then track it live
    python dev/remote.py start mose2 --quick      # detached quick smoke test
    python dev/remote.py attach [JOBID]           # (re)connect + track live (default: latest)
    python dev/remote.py jobs                     # list jobs on the box + their state
    python dev/remote.py status [JOBID]           # one job: meta + state + log tail (default: latest)
    python dev/remote.py logs [JOBID] --follow    # tail the remote log (live)
    python dev/remote.py stop JOBID               # SIGTERM a running job's process group
    python dev/remote.py pull mose2 wse2 mos2     # fetch the finished checkpoints (grid-filtered)

`start` returns immediately: it ships the code, writes a small runner under
<remote>/jobs/<jobid>/ and launches it with `nohup setsid` so it keeps running
after you disconnect. The runner processes the materials sequentially (one
`scan.py` per material), writing meta/state/log/pid into the job dir.

The job is detached on the box from the moment it starts, so the ssh connection
is only ever a VIEWER. `--follow` (or `attach`) streams the log live and exits
when the job finishes; to DISCONNECT, just Ctrl-C (or close the terminal / drop
the link) -- that tears down the viewer only, and the job runs to completion.
Reconnect any time with `attach`/`status`/`logs`, then `pull` once state is `done`.

Then locally: open analysis.ipynb (same MATERIAL) or run scripts/export_pdf.py.

Transport is ssh/scp only (uses the 'qlmc' host in ~/.ssh/config, cloudflared
ProxyCommand and all) -- no rsync dependency, so it works from Windows Git Bash.
Override the box via env: CXR_REMOTE_HOST / CXR_REMOTE_DIR / CXR_REMOTE_UV.
"""

import argparse
import datetime
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HOST = os.environ.get("CXR_REMOTE_HOST", "qlmc")
REMOTE_DIR = os.environ.get("CXR_REMOTE_DIR", "/home/aamador/dev/cxr-mc")
REMOTE_UV = os.environ.get("CXR_REMOTE_UV", "/home/aamador/.local/bin/uv")
# repo root = two levels up from dev/remote.py. remote.py orchestrates the
# *checkout* (it tars the working tree up to the box), so it resolves paths
# against the repo root, not its own dev/ dir.
LOCAL_ROOT = Path(__file__).resolve().parents[1]

# detached-job bookkeeping lives under <REMOTE_DIR>/jobs/<jobid>/ on the box
# (gitignored there): run.sh, meta, pid, state, log. One subdir per `start`.
JOBS_SUBDIR = "jobs"

# material keys are embedded into a remote shell command, so constrain them to
# the crystal-key alphabet -- this both rejects typos early and blocks shell
# injection through the material argument.
_MATERIAL_RE = re.compile(r"^[A-Za-z0-9_]+$")

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too), plus the root scan.py shim the box invokes and
# pyproject.toml. Not checkpoints/ (the output we pull back the other way).
SYNC_PATHS = ["src", "scan.py", "pyproject.toml", "uv.lock", "README.md"]

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


def _check_materials(materials):
    bad = [m for m in materials if not _MATERIAL_RE.match(m)]
    if bad:
        raise SystemExit(
            f"invalid material name(s) {bad}: expected crystal keys like "
            f"mose2 / hopg / silicon (letters, digits, underscore only)."
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
    cmd = f"cd {REMOTE_DIR} && {REMOTE_UV} run --no-sync python scan.py {material}"
    if quick:
        cmd += " --quick"
    if workers is not None:
        cmd += f" --workers {workers}"
    _run(["ssh", "-n", HOST, cmd])


def pull(stems, grid=False, drop_wide_brem=False, downcast=False, no_sync=False):
    """Fetch checkpoints/<stem>.pkl back from the box for each stem (stem =
    material, or material_quick for a --quick run).

    With ``grid``, filter on the box BEFORE the transfer: slim each checkpoint to
    just the material's current grid (``cxr slim --grid``, plus the optional byte
    trimmers) into a box temp, scp that smaller file into the local active slot,
    and delete the temp. ``sync_code()`` runs first (unless ``no_sync``) so the
    box rebuilds the grid from the same ``config.py`` the laptop has -- closing
    sync drift. Without ``grid`` this is the plain whole-file scp."""
    dest = LOCAL_ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    if grid and not no_sync:
        sync_code()  # box must rebuild the grid from the same config.py
    for stem in stems:
        local = dest / f"{stem}.pkl"
        if grid:
            flags = " --grid"
            if drop_wide_brem:
                flags += " --drop-wide-brem"
            if downcast:
                flags += " --downcast"
            remote_tmp = f"/tmp/{stem}.grid.pkl"
            ckpt = f"{REMOTE_DIR}/checkpoints/{stem}.pkl"
            # slim on the box; a missing checkpoint makes this exit nonzero (via
            # _run's check=True), so the scp below is never reached for that stem.
            _run(
                [
                    "ssh",
                    "-n",
                    HOST,
                    f"cd {REMOTE_DIR} && {REMOTE_UV} run --no-sync cxr slim "
                    f"{ckpt}{flags} -o {remote_tmp}",
                ]
            )
            _run(["scp", f"{HOST}:{remote_tmp}", str(local)])
            _run(["ssh", "-n", HOST, f"rm -f {remote_tmp}"])
            print(f"pulled (grid) -> checkpoints/{stem}.pkl")
        else:
            _run(["scp", f"{HOST}:{REMOTE_DIR}/checkpoints/{stem}.pkl", str(local)])
            print(f"pulled -> checkpoints/{stem}.pkl")


# ---- detached job queue -------------------------------------------------------
def _stems(materials, quick):
    """Checkpoint stems a queue produces (scan.py writes <material>_quick.pkl
    for --quick runs)."""
    return [f"{m}_quick" if quick else m for m in materials]


def _queue_script(jobid, materials, quick, workers):
    """The bash runner shipped to the box and launched detached. It records
    pid/meta/state, then runs `scan.py` once per material in sequence, appending
    all output to the job log and marking the state at each transition."""
    flags = ""
    if quick:
        flags += " --quick"
    if workers is not None:
        flags += f" --workers {workers}"
    mats = " ".join(materials)  # safe: each token matched _MATERIAL_RE
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    return f"""#!/usr/bin/env bash
set -u
JOBDIR="{jobdir}"
cd "{REMOTE_DIR}" || exit 1
echo $$ > "$JOBDIR/pid"
{{ echo "job: {jobid}"; echo "materials: {mats}"; echo "quick: {bool(quick)}"; \
echo "workers: {workers}"; echo "started: $(date -Is)"; echo "pid: $$"; \
}} > "$JOBDIR/meta"
{REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
mats=({mats})
total=${{#mats[@]}}
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" \
>> "$JOBDIR/log"
  if ! {REMOTE_UV} run --no-sync python scan.py "$m"{flags} >> "$JOBDIR/log" 2>&1
  then
    echo "FAILED at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    exit 1
  fi
done
echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
"""


def _live_jobs():
    """[(jobid, quick, [materials])] for every job on the box whose runner
    process is still alive (pid file present and `kill -0` succeeds). Reads the
    material list + quick flag straight from each job's meta."""
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; [ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        'p=$(cat "$d/pid" 2>/dev/null) || continue; '
        'kill -0 "$p" 2>/dev/null || continue; '
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
            "attach to it (python dev/remote.py attach <jobid>) or stop it "
            "(python dev/remote.py stop <jobid>) first, or run different materials."
        )


def clear_remote(material, yes=False):
    """Delete a material's accumulated checkpoints on the box: both
    ``checkpoints/<material>.pkl`` and ``checkpoints/<material>_quick.pkl``.

    Refuses (before touching anything) if a live job is producing either stem,
    reusing the same ``_live_jobs`` guard as ``start``/``scan`` so a clear can't
    yank a checkpoint out from under a running sweep. Without ``yes`` this is a
    safe dry preview: it prints exactly which of the two files exist and would be
    deleted, then stops. With ``yes`` it ``rm -f``s them and reports what went."""
    _check_materials([material])  # interpolated into a remote shell command
    wanted = {material, f"{material}_quick"}
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
            "stop it (python dev/remote.py stop <jobid>) first, or wait for it to finish."
        )
    # which of the two stems actually exist on the box; the `|| true` keeps a
    # missing last stem's failed `[ -f ]` from becoming the loop's -- and hence
    # ssh's -- exit status, which would make _ssh_capture abort the whole clear
    listing = (
        f"cd {REMOTE_DIR}/checkpoints 2>/dev/null || exit 0; "
        f'for f in {material}.pkl {material}_quick.pkl; do [ -f "$f" ] && echo "$f" || true; done'
    )
    existing = _ssh_capture(listing).split()
    if not existing:
        print(f"(nothing to clear for {material})")
        return
    if not yes:
        print("would delete on the box (re-run with --yes to delete):")
        for f in existing:
            print(f"  checkpoints/{f}")
        return
    _run(["ssh", "-n", HOST, f"cd {REMOTE_DIR}/checkpoints && rm -f {' '.join(existing)}"])
    print("cleared on the box:")
    for f in existing:
        print(f"  checkpoints/{f}")


def start_queue(materials, quick=False, workers=None, no_sync=False, dry_run=False):
    """Launch a detached queue on the box: sync code, write the per-job runner,
    and `nohup setsid` it so it survives ssh disconnect. Returns the job id."""
    _check_materials(materials)
    if not dry_run:
        _refuse_if_busy(materials, quick)
    jobid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    script = _queue_script(jobid, materials, quick, workers)
    # detach: redirect all three std streams off the ssh channel and setsid into
    # a new session, so the ssh command returns immediately and the job keeps
    # running after you disconnect.
    launch = (
        f"cd '{REMOTE_DIR}' && : > '{jobdir}/log' && "
        f"nohup setsid bash '{jobdir}/run.sh' >/dev/null 2>&1 </dev/null & "
        f"echo 'launched {jobid}'"
    )

    if dry_run:
        print(f"# job {jobid}: {' '.join(materials)}{' (quick)' if quick else ''}")
        print(f"# --- ssh {HOST}: mkdir -p {jobdir} && cat > {jobdir}/run.sh <<\n")
        print(script)
        print(f"# --- ssh {HOST}: {launch}")
        return jobid

    if not no_sync:
        sync_code()
    # create the job dir and write run.sh (script piped over stdin).
    # Send as LF-only bytes: text=True on Windows translates \n->\r\n,
    # which produces a CRLF run.sh that bash silently refuses to execute.
    subprocess.run(
        ["ssh", HOST, f"mkdir -p '{jobdir}' && cat > '{jobdir}/run.sh'"],
        input=script.replace("\r\n", "\n").encode(),
        check=True,
    )
    _run(["ssh", "-n", HOST, launch])

    stems = _stems(materials, quick)
    print(
        f"\nstarted job {jobid} on {HOST}: {' '.join(materials)}"
        f"{' (quick)' if quick else ''}\n"
        f"  watch:  python dev/remote.py status {jobid}\n"
        f"  logs:   python dev/remote.py logs {jobid} --follow\n"
        f"  pull:   python dev/remote.py pull {' '.join(stems)}   (when state is 'done')"
    )
    return jobid


def list_jobs():
    """Print every job dir on the box with its current state line, oldest first
    (the dirs are timestamp-named, so this is chronological)."""
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; '
        '[ -d "$JOBS" ] || { echo "(no jobs)"; exit 0; }; '
        'found=; for d in "$JOBS"/*/; do [ -d "$d" ] || continue; found=1; '
        'printf "%s  %s\\n" "$(basename "$d")" '
        '"$(cat "$d/state" 2>/dev/null || echo "?")"; done; '
        '[ -n "$found" ] || echo "(no jobs)"'
    )
    print(_ssh_capture(remote), end="")


def _job_assign(jobid):
    """Bash that sets JOB to the given id, or the latest job dir if none given."""
    if jobid:
        _check_materials([jobid.replace("-", "")])  # reject odd chars in the id
        return f'JOB="{jobid}"'
    return 'JOB=$(ls -1 "$JOBS" 2>/dev/null | tail -1)'


def job_status(jobid=None):
    """Print one job's meta, current state, whether its process is still alive,
    and the tail of its log. Defaults to the most recent job."""
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; {_job_assign(jobid)}; '
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        'echo "== job $JOB =="; cat "$D/meta" 2>/dev/null; '
        'echo "-- state --"; cat "$D/state" 2>/dev/null || echo "(no state yet)"; '
        'if [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; '
        'then echo "process: ALIVE"; else echo "process: not running"; fi; '
        'echo "-- log tail --"; tail -n 20 "$D/log" 2>/dev/null'
    )
    print(_ssh_capture(remote), end="")


def tail_logs(jobid=None, follow=False):
    """Tail a job's log. With --follow, stream live (blocks until Ctrl-C)."""
    tail = "tail -f" if follow else "tail -n 60"
    remote = (
        f'JOBS="{REMOTE_DIR}/{JOBS_SUBDIR}"; {_job_assign(jobid)}; '
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        f'{tail} "$D/log"'
    )
    if follow:
        try:
            subprocess.run(["ssh", HOST, remote])  # inherit stdio -> live stream
        except KeyboardInterrupt:
            print("\n(stopped following; the job is unaffected)")
    else:
        print(_ssh_capture(remote), end="")


def _latest_jobid():
    """The most recent job id on the box (job dirs are timestamp-named), or None."""
    out = _ssh_capture(f'ls -1 "{REMOTE_DIR}/{JOBS_SUBDIR}" 2>/dev/null | tail -1').strip()
    return out or None


def _disconnect_hint(jobid):
    print(
        f"\n\ndisconnected from job {jobid} -- it keeps running on {HOST}.\n"
        f"  reconnect: python dev/remote.py attach {jobid}\n"
        f"  status:    python dev/remote.py status {jobid}\n"
        f"  stop:      python dev/remote.py stop   {jobid}"
    )


def attach(jobid=None):
    """Live-track a job: stream its log until it finishes, then print the final
    state. Disconnecting -- Ctrl-C, closing the terminal, or a dropped ssh --
    tears down the VIEWER only; the job is detached server-side (nohup setsid)
    and runs to completion regardless. Defaults to the most recent job."""
    jobid = jobid or _latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to attach to (start one: remote.py start <materials>)")
    jobdir = f"{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"
    # tail the log live, but self-terminate once the job process exits, so a
    # finished job doesn't leave you stuck in tail -f. The tail is NOT nohup'd, so
    # a local Ctrl-C / dropped ssh tears it (and the wait loop) down while the
    # setsid'd job keeps going.
    remote = (
        f'D="{jobdir}"; '
        f'[ -d "$D" ] || {{ echo "no such job: {jobid}"; exit 1; }}; '
        'tail -n 50 -f "$D/log" 2>/dev/null & TP=$!; '
        # `start --follow` attaches the instant after launch, which can beat the
        # detached runner writing its pid (the runner does it as its first act,
        # but bash/setsid startup is not instantaneous). Poll for the pid instead
        # of testing once -- otherwise we find it missing, skip the wait below,
        # and tear the tail down right away, wrongly reporting "job finished".
        "P=; for _ in $(seq 1 30); do "
        'if [ -s "$D/pid" ]; then P=$(cat "$D/pid"); break; fi; sleep 1; done; '
        # Then hold (keeping the tail live) until the runner process exits. A job
        # that already finished still has its pid file, so kill -0 fails at once
        # and we fall straight through to the teardown -- correct for that case.
        'if [ -n "$P" ]; then '
        'while kill -0 "$P" 2>/dev/null; do sleep 2; done; fi; '
        'sleep 1; kill "$TP" 2>/dev/null; '
        'printf "\\n--- job finished ---\\n"; cat "$D/state" 2>/dev/null'
    )
    print(f"attached to job {jobid} on {HOST} -- Ctrl-C to disconnect (the job keeps running).\n")
    try:
        subprocess.run(["ssh", HOST, remote])
    except KeyboardInterrupt:
        _disconnect_hint(jobid)


def stop_job(jobid):
    """SIGTERM a running job's whole process group (the runner + scan.py + its
    transport worker pool), then mark the job stopped."""
    _check_materials([jobid.replace("-", "")])
    remote = (
        f'D="{REMOTE_DIR}/{JOBS_SUBDIR}/{jobid}"; '
        'if [ ! -f "$D/pid" ]; then echo "no pid for job {0}"; exit 1; fi; '
        'P=$(cat "$D/pid"); '
        'kill -TERM -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null; '
        'echo "stopped [{0}] $(date -Is)" > "$D/state"; '
        'echo "sent SIGTERM to job {0} (pgid $P)"'
    ).format(jobid)
    _run(["ssh", "-n", HOST, remote])


def main(argv=None):
    # The box's output is UTF-8 (job logs embed tqdm block-glyph progress bars
    # like `████▌`). On Windows stdout defaults to cp1252 -- and when it is,
    # printing that text raises UnicodeEncodeError, so `status`/`jobs`/`logs`
    # would crash on the glyphs. Force UTF-8 with replacement so they never do.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(prog="remote.py", description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="sync code, run ONE sweep in the foreground, pull checkpoint")
    s.add_argument("material")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--no-sync", action="store_true", help="skip the code upload")
    s.add_argument(
        "--grid", action="store_true", help="grid-filter the checkpoint on the box before pulling"
    )
    s.add_argument("--drop-wide-brem", action="store_true", help="with --grid: drop wide-brem too")
    s.add_argument("--downcast", action="store_true", help="with --grid: downcast to float32 too")

    st = sub.add_parser(
        "start",
        help="sync code, launch a DETACHED queue of materials (survives disconnect)",
    )
    st.add_argument("materials", nargs="+", help="one or more crystal keys")
    st.add_argument("--quick", action="store_true")
    st.add_argument("--workers", type=int, default=None)
    st.add_argument("--no-sync", action="store_true", help="skip the code upload")
    st.add_argument("--dry-run", action="store_true", help="print the runner + commands, don't ssh")
    st.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="track the job live after launching (Ctrl-C disconnects; job keeps running)",
    )

    at = sub.add_parser(
        "attach",
        help="live-track a job until it finishes (Ctrl-C disconnects; default: latest)",
    )
    at.add_argument("jobid", nargs="?", default=None)

    sub.add_parser("jobs", help="list jobs on the box and their state")

    js = sub.add_parser("status", help="show one job (default: latest)")
    js.add_argument("jobid", nargs="?", default=None)

    lg = sub.add_parser("logs", help="tail a job's log (default: latest)")
    lg.add_argument("jobid", nargs="?", default=None)
    lg.add_argument("--follow", "-f", action="store_true", help="stream live")

    sp = sub.add_parser("stop", help="SIGTERM a running job")
    sp.add_argument("jobid")

    p = sub.add_parser("pull", help="fetch one or more existing checkpoints from the box")
    p.add_argument("material", nargs="+", help="checkpoint stem(s), e.g. mose2 mose2_quick")
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

    c = sub.add_parser("clear", help="delete a material's accumulated checkpoints on the box")
    c.add_argument(
        "material", help="crystal key; clears both <material>.pkl and <material>_quick.pkl"
    )
    c.add_argument("--yes", action="store_true", help="actually delete (default: dry preview only)")

    sub.add_parser("sync", help="push the current code to the box only")

    args = ap.parse_args(argv)
    match args.cmd:
        case "sync":
            sync_code()
        case "pull":
            pull(
                args.material,
                grid=not args.full,
                drop_wide_brem=args.drop_wide_brem,
                downcast=args.downcast,
                no_sync=args.no_sync,
            )
        case "clear":
            clear_remote(args.material, args.yes)
        case "scan":
            if args.quick and args.grid:
                raise SystemExit(
                    "scan --quick --grid: quick checkpoints aren't grid-filterable "
                    "(their grid isn't reproducible from material_sweep), so the "
                    "trailing pull would fail after the whole sweep ran. Drop --grid."
                )
            # same checkpoint-collision guard as `start`: a foreground scan and a
            # detached job writing the same <stem>.pkl would race on its .tmp.
            _refuse_if_busy([args.material], args.quick)
            if not args.no_sync:
                sync_code()
            remote_scan(args.material, args.quick, args.workers)
            stem = f"{args.material}_quick" if args.quick else args.material
            # code is already synced above, so the trailing grid-pull skips its
            # own sync (no_sync=True); forward the same grid/trim flags.
            pull(
                [stem],
                grid=args.grid,
                drop_wide_brem=args.drop_wide_brem,
                downcast=args.downcast,
                no_sync=True,
            )
            print(
                f"\ndone. checkpoints/{stem}.pkl is local; open notebooks/analysis.ipynb "
                f"with MATERIAL='{stem}' (or run scripts/export_pdf.py) -- all viz/PDF "
                "stays local."
            )
        case "start":
            jobid = start_queue(
                args.materials, args.quick, args.workers, args.no_sync, args.dry_run
            )
            if args.follow and not args.dry_run:
                attach(jobid)
        case "attach":
            attach(args.jobid)
        case "jobs":
            list_jobs()
        case "status":
            job_status(args.jobid)
        case "logs":
            tail_logs(args.jobid, args.follow)
        case "stop":
            stop_job(args.jobid)


if __name__ == "__main__":
    main()
