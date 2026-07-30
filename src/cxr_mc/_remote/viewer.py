"""Live job viewers: attach, status, list, logs."""

import queue
import select
import subprocess
import sys
import threading
import time

import tqdm  # noqa: F401 -- kept importable at module level for test monkeypatching

from . import config, lifecycle, presentation, scripts, state, transport


def _jobs_remote_command():
    return (
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$d/meta" 2>/dev/null | tail -1); '
        'Q=$(sed -n "s/^quick: //p" "$d/meta" 2>/dev/null | tail -1); '
        'K=$(sed -n "s/^kind: //p" "$d/meta" 2>/dev/null | tail -1); '
        '[ -n "$K" ] || { grep -q "^ne: " "$d/meta" 2>/dev/null && K=check || K=scan; }; '
        'M=$(sed -n "s/^materials: //p" "$d/meta" 2>/dev/null | tail -1); '
        'S=$(cat "$d/state" 2>/dev/null | tr "\\n\\t" "  "); '
        'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "$(basename "$d")" '
        '"${SID:--}" "${Q:-?}" "$K" "${M:--}" "${S:-(no state yet)}"; done'
    )


def jobs_raw():
    """Return machine-oriented job records without printing."""
    return transport._ssh_capture(_jobs_remote_command())


def list_jobs():
    """Print every submitted job with identifying metadata, oldest first."""
    rows = []
    for line in jobs_raw().splitlines():
        fields = line.split("\t", 5)
        if len(fields) == 6:
            jobid, scheduler_id, quick, _command, materials, state_ = fields
        elif len(fields) == 5:
            jobid, scheduler_id, quick, materials, state_ = fields
        else:
            continue
        jobid, scheduler_id, quick, materials, state_ = (
            presentation._sanitize_terminal(value)
            for value in (jobid, scheduler_id, quick, materials, state_)
        )
        mode = "quick" if quick == "True" else "standard" if quick == "False" else "?"
        rows.append((jobid, scheduler_id, mode, materials.replace(" ", ", "), state_))
    if not rows:
        print(
            f"No remote jobs on {config.remote_host()}. Start one with `cxr remote run [PROFILE]`."
        )
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
    progress = (
        '{ for f in "$D"/progress/*.json; do '
        '[ -f "$f" ] || continue; cat "$f" 2>/dev/null || true; printf "\\n"; done; '
        "} | emit PROGRESS; "
    )
    performance = (
        '{ for f in "$D"/performance/*/*.latest.json; do '
        '[ -f "$f" ] || continue; cat "$f" 2>/dev/null || true; printf "\\n"; done; '
        "} | emit PERFORMANCE; "
    )
    resources = (
        "{ "
        "if command -v top >/dev/null 2>&1; then "
        "LC_ALL=C top -bn1 2>/dev/null | "
        "awk '/Cpu\\(s\\)|^%Cpu/ {for (i=1;i<=NF;i++) if ($i ~ /^id/) "
        '{printf "cpu_percent=%.1f|", 100-$(i-1); exit}}\'; fi; '
        "if command -v free >/dev/null 2>&1; then "
        "free -b 2>/dev/null | awk '/^Mem:/ {printf "
        '"memory_used_bytes=%s|memory_total_bytes=%s|memory_percent=%.1f|", '
        "$3, $2, 100*$3/$2}'; fi; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null | "
        "awk -F, 'NR==1 {printf \"gpu_percent=%.1f|vram_used_mib=%.0f|"
        "vram_total_mib=%.0f|vram_percent=%.1f|\", $1, $2, $3, 100*$2/$3}'; fi; "
        "} | emit RESOURCES; "
        if detail >= 2
        else ""
    )
    log = '{ tail -c 32768 "$D/log" 2>/dev/null; } | emit LOG; ' if detail >= 2 else ""
    return (
        "set -o pipefail; "
        "emit() { printf 'CXR_REMOTE_V1\\t%s\\t' \"$1\"; "
        "base64 | tr -d '\\n'; printf '\\n'; }; "
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; {job_assign}; "
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        '{ printf "%s\\n" "$JOB"; } | emit JOB; '
        '{ cat "$D/meta" 2>/dev/null; } | emit META; '
        '{ cat "$D/state" 2>/dev/null; } | emit STATE; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        "QUEUE_RAW=; "
        "case \"$SID\" in ''|*[!0-9]*) ;; *) "
        f"QUEUE_RAW=$(squeue -h --partition={config.shell_word(config.SLURM_PARTITION)} "
        "--states=PENDING,RUNNING --sort=-p,i "
        "-o 'job_id=%i|state=%T|name=%j|partition=%P|elapsed=%M|left=%L|"
        "nodes=%D|reason=%R|priority=%Q|user=%u' 2>&1); "
        'QUEUE_RC=$?; if [ "$QUEUE_RC" -ne 0 ]; then '
        'printf "%s\\n" "$QUEUE_RAW" >&2; exit "$QUEUE_RC"; fi; '
        ";; esac; "
        "{ "
        'case "$SID" in \'\'|*[!0-9]*) echo "job_id=-|state=NOT_QUEUED" ;; *) '
        'TARGET=$(printf "%s\\n" "$QUEUE_RAW" | '
        'awk -v sid="$SID" \'index($0, "job_id=" sid "|") == 1 { print; exit }\'); '
        'if [ -n "$TARGET" ]; then printf "%s\\n" "$TARGET"; '
        'else printf "job_id=%s|state=NOT_QUEUED\\n" "$SID"; fi ;; esac; '
        "} | emit SQUEUE || exit $?; "
        "{ "
        f"printf 'cohort_partition={config.SLURM_PARTITION}|"
        "order=priority_desc_job_id_asc\\n'; "
        'if [ -n "${QUEUE_RAW+x}" ]; then printf "%s\\n" "$QUEUE_RAW"; fi; '
        "} | emit QUEUE; " + progress + performance + resources + log
    )


def status_sections(jobid=None, detail=0):
    """Fetch and decode one status report without rendering it."""
    if detail < 0:
        raise ValueError("detail must be non-negative")
    output = transport._ssh_capture(_status_remote_command(scripts._job_assign(jobid), detail))
    return presentation._marked_sections(output), output


def job_status(jobid=None, detail=0):
    """Print one structured job report; verbosity adds allocation and log tail."""
    sections, output = status_sections(jobid, detail)
    if not sections:
        print(presentation._sanitize_terminal(output, multiline=True), end="")
        return
    print(presentation._style_states(presentation._format_job_status(sections, detail)))


def tail_logs(jobid=None, follow=False):
    """Tail a job's log. With --follow, stream live (blocks until Ctrl-C)."""
    tail = "tail -f" if follow else "tail -n 60"
    remote = (
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        f"{scripts._job_assign(jobid)}; "
        'D="$JOBS/$JOB"; '
        'if [ -z "$JOB" ] || [ ! -d "$D" ]; then echo "no such job: ${JOB:-<none>}"; '
        "exit 1; fi; "
        f'printf "LOG %s · {config.remote_host()}\\n\\n" "$JOB"; '
        f'{tail} "$D/log"'
    )
    if follow:
        try:
            # Close stdin so Windows OpenSSH cannot hang on console forwarding,
            # while stdout/stderr still inherit for live streaming and Ctrl-C.
            result = subprocess.run(["ssh", "-n", config.remote_host(), remote])
        except KeyboardInterrupt:
            print("\n(stopped following; the job is unaffected)", file=sys.stderr)
            return 130
        return 0 if result.returncode == 0 else 1
    else:
        print(transport._ssh_capture(remote), end="")
        return 0


def _disconnect_hint(jobid):
    print(
        f"\n\nVIEWER DISCONNECTED · job {jobid} keeps running on {config.remote_host()}\n"
        f"  Reconnect  cxr remote status {jobid} -a\n"
        f"  Status     cxr remote status {jobid} -vv\n"
        "  Stop       cxr remote stop <material>"
    )


_STATUS_FRAME_END = "CXR_REMOTE_FRAME_END"


def _status_stream_command(remote: str) -> str:
    """Wrap one status snapshot in a framed two-second remote polling loop."""
    snapshot = remote.rstrip().removesuffix(";")
    return f"while :; do {snapshot}; printf '{_STATUS_FRAME_END}\\n'; sleep 2; done"


def _status_stream(remote: str):
    """Yield snapshots, reconnecting twice after an SSH stream disconnect."""
    stream_command = _status_stream_command(remote)
    reconnects = 2
    last_error = "unknown error"
    for attempt in range(reconnects + 1):
        process = None
        lines: list[str] = []
        try:
            process = subprocess.Popen(
                ["ssh", "-n", config.remote_host(), stream_command],
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("ssh status stream has no stdout")
            for line in process.stdout:
                if line.rstrip("\r\n") == _STATUS_FRAME_END:
                    yield "".join(lines)
                    lines.clear()
                else:
                    lines.append(line)
            last_error = f"exit {process.wait()}"
        except OSError as error:
            last_error = str(error)
        finally:
            if process is not None:
                if process.stdout is not None:
                    process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if attempt < reconnects:
            print(
                f"ssh status stream disconnected ({last_error}); "
                f"reconnecting {attempt + 1}/{reconnects}",
                file=sys.stderr,
            )
    raise SystemExit(
        f"ssh status stream disconnected after {reconnects} reconnect attempts ({last_error}); "
        "remote job is unaffected"
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


# Cancel-keybinding sequence (item 5): a bare 'q'/single keystroke must never
# tear down a live SLURM allocation, so cancelling is two DIFFERENT keys --
# 'x' arms, 'y' confirms within the window below; anything else (including a
# repeated 'x') disarms silently. Checked once per ~2 s poll (see _KeyListener).
_CANCEL_ARM_KEY = "x"
_PULL_ARM_KEY = "p"
_CANCEL_CONFIRM_KEY = "y"
_CANCEL_ARM_SECONDS = 6.0


class _KeyListener:
    """Background nonblocking single-keypress capture for the attach loop.

    A no-op (``active`` False, ``poll()`` always empty) unless stdin is an
    interactive terminal -- piped/non-tty attach then behaves exactly as
    before, with no cancel-keybinding hint shown. Cross-platform: POSIX uses
    ``termios``/``tty`` cbreak mode plus ``select``; Windows uses ``msvcrt``
    (no raw-mode setup needed there). Reads happen on a background daemon
    thread so the main loop's blocking SSH read is never held up; ``poll()``
    drains and returns every key buffered since the last call, in order, so a
    fast 'x' then 'y' typed within one ~2 s window is never collapsed to just
    the latest keystroke.
    """

    def __init__(self):
        self.active = False
        self._keys = queue.Queue()
        self._stop_event = threading.Event()
        self._restore = None
        self._thread = None
        if not sys.stdin.isatty():
            return
        try:
            import msvcrt  # noqa: F401 -- Windows only; ImportError picks the POSIX branch below
        except ImportError:
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except (OSError, ValueError, termios.error):
                return
            self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._thread = threading.Thread(target=self._poll_posix, args=(fd,), daemon=True)
        else:
            self._thread = threading.Thread(target=self._poll_windows, daemon=True)
        self._thread.start()
        self.active = True

    def _poll_windows(self):
        import msvcrt

        while not self._stop_event.is_set():
            if msvcrt.kbhit():  # ty: ignore[unresolved-attribute]
                self._keys.put(msvcrt.getwch())  # ty: ignore[unresolved-attribute]
            else:
                self._stop_event.wait(0.1)

    def _poll_posix(self, fd):
        while not self._stop_event.is_set():
            ready, _write, _exceptional = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            char = sys.stdin.read(1)
            if char == "":
                break  # EOF (e.g. stdin redirected from a closed pipe)
            self._keys.put(char)

    def poll(self):
        """Every key buffered since the last call, oldest first."""
        keys = []
        while True:
            try:
                keys.append(self._keys.get_nowait())
            except queue.Empty:
                break
        return keys

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._restore is not None:
            self._restore()


def _attach_header(refresh, *, armed=False, cancel_hint=True):
    """One-line banner above each live frame; the counter proves it's polling."""
    if armed:
        action = "PULL" if armed == "pull" else "CANCEL"
        return presentation._paint(
            f"ATTACHED · {config.remote_host()} · refresh {refresh} · "
            f"{action} ARMED -- press {_CANCEL_CONFIRM_KEY} to confirm, any other key aborts",
            "warning",
        )
    hint = (
        f" · {_CANCEL_ARM_KEY} cancels job · {_PULL_ARM_KEY} pulls progress (confirm)"
        if cancel_hint
        else ""
    )
    return presentation._paint(
        f"ATTACHED · {config.remote_host()} · refresh {refresh} · "
        f"Ctrl-C detaches (job keeps running){hint}",
        "inactive",
    )


def _pull_attached_progress(jobid, sections):
    """Pull checkpoint stems that have emitted progress for attached job."""
    fields = presentation._metadata_fields(sections.get("META", ""))
    records = presentation._parse_progress_records(sections.get("PROGRESS", ""))
    materials = list(records)
    if not materials:
        raise SystemExit("no partial checkpoints have reported progress yet")
    stems = scripts._stems(
        materials,
        fields.get("quick") == "True",
        fields.get("fidelity", "full"),
        high_energy_min_kev=(
            float(fields["high_energy_min_kev"])
            if fields.get("high_energy_min_kev") not in (None, "None")
            else None
        ),
        catalog_profile=fields.get("catalog_profile", "standard"),
    )
    lifecycle.pull(stems, no_sync=True)
    print(f"\nPARTIAL PULL COMPLETE · job {jobid} · {len(stems)} checkpoint(s)")


def _live_status(jobid, detail):
    """Re-render ``status <jobid>`` at ``detail`` each poll until the job is terminal.

    The viewer is read-only by default: Ctrl-C or a dropped SSH link tears
    down only this loop, never the SLURM job. The one exception is the
    cancel keybinding (item 5) -- 'x' then 'y' within a few seconds scancels
    the attached job via :func:`lifecycle._stop_jobid`; see ``_KeyListener``
    and ``_CANCEL_ARM_SECONDS``. A chunked chain hops scheduler IDs between
    slices, so the shared status command re-reads the latest recorded ID each
    poll; the same ``_POLL_GRACE_POLLS`` watchdog declares a chain broken only
    after ~30 s with no live allocation and a non-terminal recorded state.
    Returns True once the job reaches a terminal state, False on viewer
    disconnect, a stalled chain, or a confirmed user cancel.
    """
    remote = _status_remote_command(scripts._job_assign(jobid), detail)
    tty = presentation._color_enabled()
    refresh = 0
    missed = 0
    state_ = ""
    broken = False
    cancelled = False
    pull_requested = False
    stream = None
    keys = _KeyListener()
    armed_until = None
    armed_action = None
    try:
        stream = _status_stream(remote)
        for output in stream:
            refresh += 1
            sections = presentation._marked_sections(output)
            if not sections:
                print(presentation._sanitize_terminal(output, multiline=True), end="")
                return False
            state_ = presentation._sanitize_terminal(sections.get("STATE", ""), multiline=True)
            scheduler = presentation._scheduler_fields(sections.get("SQUEUE", ""))
            live = scheduler.get("state", "") not in ("", "NOT_QUEUED")

            if armed_until is not None and time.monotonic() >= armed_until:
                armed_until = None  # confirm window lapsed; disarm silently
                armed_action = None
            for key in keys.poll():
                if armed_until is not None:
                    confirmed = key.lower() == _CANCEL_CONFIRM_KEY
                    armed_until = None
                    action = armed_action
                    armed_action = None
                    if confirmed:
                        if action == "cancel":
                            cancelled = True
                            break
                        if action == "pull":
                            pull_requested = True
                elif key.lower() == _CANCEL_ARM_KEY:
                    armed_until = time.monotonic() + _CANCEL_ARM_SECONDS
                    armed_action = "cancel"
                elif key.lower() == _PULL_ARM_KEY:
                    armed_until = time.monotonic() + _CANCEL_ARM_SECONDS
                    armed_action = "pull"

            frame = (
                _attach_header(
                    refresh,
                    armed=armed_action if armed_until is not None else False,
                    cancel_hint=keys.active,
                )
                + "\n\n"
                + presentation._style_states(presentation._format_job_status(sections, detail))
            )
            _render_frame(frame, tty=tty)
            if pull_requested:
                pull_requested = False
                try:
                    _pull_attached_progress(jobid, sections)
                except (SystemExit, ValueError) as error:
                    print(f"\nPARTIAL PULL FAILED · job {jobid}\n  {error}")
            if cancelled or _is_terminal_state(state_):
                break
            missed = 0 if live else missed + 1
            if missed >= _POLL_GRACE_POLLS:
                broken = True
                break
    except KeyboardInterrupt:
        _disconnect_hint(jobid)
        return False
    finally:
        keys.stop()
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if cancelled:
        try:
            lifecycle._stop_jobid(jobid)
        except SystemExit as error:
            print(f"\nJOB {jobid} · CANCEL REQUEST FAILED\n  {error}")
        else:
            print(f"\nJOB {jobid} · CANCELLED BY USER\n  Verify  cxr remote status {jobid} -vv")
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
    the SLURM job keeps running unless cancelled via the 'x'/'y' keybinding
    (see :func:`_live_status`). Returns True once the job reaches a terminal
    state, False on disconnect, a stalled chain, or a confirmed cancel --
    callers (scan/check) key their auto-pull off that.
    """
    defaulted = jobid is None
    jobid = jobid or state._latest_jobid()
    if not jobid:
        raise SystemExit("no jobs to attach to (start one: cxr remote run [PROFILE])")
    transport._check_shell_tokens([jobid])
    if defaulted:
        current = state._job_state(jobid)
        if _is_terminal_state(current):
            print(
                f"warning: no job id given; defaulting to {jobid}, which is not running "
                f"(state: {current}).\n"
                "  Live jobs   cxr remote jobs\n"
                "  Monitor one cxr remote status <job-id> -a",
                file=sys.stderr,
            )
    return _live_status(jobid, detail)
