"""Live job viewers: attach, status, list, logs."""

import subprocess
import sys
import time

import tqdm  # noqa: F401 -- kept importable at module level for test monkeypatching

from . import config, presentation, scripts, state, transport


def list_jobs():
    """Print every submitted job with identifying metadata, oldest first."""
    remote = (
        f'JOBS="{config.REMOTE_DIR}/{config.JOBS_SUBDIR}"; '
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
    for line in transport._ssh_capture(remote).splitlines():
        fields = line.split("\t", 4)
        if len(fields) != 5:
            continue
        jobid, scheduler_id, quick, materials, state_ = fields
        mode = "quick" if quick == "True" else "standard" if quick == "False" else "?"
        rows.append((jobid, scheduler_id, mode, materials.replace(" ", ", "), state_))
    if not rows:
        print(f"No remote jobs on {config.HOST}. Start one with `cxr remote start <materials>`.")
        return
    print(
        presentation._style_states(
            presentation._format_table(("JOB", "SLURM", "MODE", "MATERIALS", "LAST EVENT"), rows)
        )
    )


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
        f'JOBS="{config.REMOTE_DIR}/{config.JOBS_SUBDIR}"; {job_assign}; '
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
        + scripts._squeue_state_command("$SID", retired="STATE=")
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
    output = transport._ssh_capture(_status_remote_command(scripts._job_assign(jobid), detail))
    sections = presentation._marked_sections(output)
    if not sections:
        print(output, end="")
        return
    print(presentation._style_states(presentation._format_job_status(sections, detail)))


def tail_logs(jobid=None, follow=False):
    """Tail a job's log. With --follow, stream live (blocks until Ctrl-C)."""
    tail = "tail -f" if follow else "tail -n 60"
    remote = (
        f'JOBS="{config.REMOTE_DIR}/{config.JOBS_SUBDIR}"; {scripts._job_assign(jobid)}; '
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        f'printf "LOG %s · {config.HOST}\\n\\n" "$JOB"; '
        f'{tail} "$D/log"'
    )
    if follow:
        try:
            # Close stdin so Windows OpenSSH cannot hang on console forwarding,
            # while stdout/stderr still inherit for live streaming and Ctrl-C.
            subprocess.run(["ssh", "-n", config.HOST, remote])
        except KeyboardInterrupt:
            print("\n(stopped following; the job is unaffected)")
    else:
        print(transport._ssh_capture(remote), end="")


def _disconnect_hint(jobid):
    print(
        f"\n\nVIEWER DISCONNECTED · job {jobid} keeps running on {config.HOST}\n"
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
    return presentation._paint(
        f"ATTACHED · {config.HOST} · refresh {refresh} · Ctrl-C detaches (job keeps running)",
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
    remote = _status_remote_command(scripts._job_assign(jobid), detail)
    tty = presentation._color_enabled()
    refresh = 0
    missed = 0
    state_ = ""
    broken = False
    try:
        while True:
            refresh += 1
            output = transport._ssh_capture(remote)
            sections = presentation._marked_sections(output)
            if not sections:
                print(output, end="")
                return False
            state_ = sections.get("STATE", "")
            scheduler = presentation._scheduler_fields(sections.get("SQUEUE", ""))
            live = scheduler.get("state", "") not in ("", "NOT_QUEUED")
            frame = (
                _attach_header(refresh)
                + "\n\n"
                + presentation._style_states(presentation._format_job_status(sections, detail))
            )
            _render_frame(frame, tty=tty)
            if _is_terminal_state(state_):
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
    print(f"\nJOB {jobid} · FINISHED\n  State   {state_}\n  Inspect cxr remote status {jobid} -vv")
    return True


def attach(jobid=None, detail=0):
    """Live-track a job: re-render its ``status`` report at ``detail`` until terminal.

    Equivalent to ``cxr remote status [-v|-vv]``, but the same report repaints
    in place every ~2 s. Ctrl-C (or a dropped link) detaches the viewer only;
    the SLURM job keeps running. Returns True once the job reaches a terminal
    state, False on disconnect or a stalled chain -- callers (scan/check) key
    their auto-pull off that.
    """
    jobid = jobid or state._latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to attach to (start one: cxr remote start <materials>)")
    transport._check_shell_tokens([jobid])
    return _live_status(jobid, detail)
