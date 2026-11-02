"""Submit and reattach authenticated marimo apps inside remote SLURM allocations.

The job protocol intentionally uses the same ``jobs/<id>/{run.sh,meta,state,log}``
layout as the Monte Carlo queues. The token is private to the submitting user;
never send it through SLURM directives, process arguments, or job logs.
"""

import re
import shlex
import socket
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..console.outputs import output_dir
from . import config, jobs, scripts, state, transport

_APPS = ("analysis", "viewer", "validation")
_TIME_RE = re.compile(r"\d{1,3}:[0-5]\d:[0-5]\d")
_READY_RE = re.compile(r"[A-Za-z0-9._-]+")
_STARTUP_POLL_SECONDS = 2
_STARTUP_TIMEOUT_SECONDS = 600


def _app_command(argv: list[str], notebook: str, *, batch: bool) -> str:
    """Render the regular local marimo builder's argv against the synced checkout.

    The CLI owns app-specific command construction; the remote layer only maps
    machine-specific paths and inserts mandatory, non-overridable auth flags.
    """
    index = argv.index(notebook)
    args = [*argv[3:index], f"src/pyrite/apps/{Path(notebook).name}", *argv[index + 1 :]]
    if batch:
        index = args.index("--output")
        args[index + 1] = "__PYRITE_OUTPUT__"
    else:
        # Quiet mode avoids logging marimo's tokenized startup URL.
        args = [
            "-q",
            args[0],
            "--headless",
            "--host",
            "127.0.0.1",
            "--port",
            "__PYRITE_PORT__",
            "--token",
            "--token-password-file",
            "__PYRITE_TOKEN__",
            *args[1:],
        ]

    def quote(arg: str) -> str:
        if arg == "__PYRITE_PORT__":
            return '"$PORT"'
        if arg == "__PYRITE_TOKEN__":
            return '"$JOBDIR/token"'
        if arg == "__PYRITE_OUTPUT__":
            return '"$JOBDIR/app.html"'
        return shlex.quote(arg)

    return " ".join(quote(arg) for arg in args)


def _script(
    jobid: str,
    app: str,
    material: str | None,
    *,
    argv: list[str],
    notebook: str,
    base_port: int,
    gpu: bool,
    walltime: str,
    batch: bool,
) -> str:
    """Build a CPU-only-by-default batch script on the selected SLURM target."""
    if not _TIME_RE.fullmatch(walltime) or not any(map(int, walltime.split(":"))):
        raise ValueError("--time must be HH:MM:SS with positive walltime")
    transport._check_shell_tokens([jobid])
    if material is not None:
        transport._check_materials([material])
    if app not in _APPS or (app == "validation" and material is not None):
        raise ValueError("invalid app or material combination")
    command = _app_command(argv, notebook, batch=batch)
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    sbatch_jobdir = config.sbatch_remote_path(config.JOBS_SUBDIR, jobid)
    nodelist = config.slurm_nodelist()
    nodes = f"#SBATCH --nodelist={nodelist}\n" if nodelist else ""
    gpu_line = f"#SBATCH --gres={config.slurm_gres()}\n" if gpu else ""
    vendor_prelude = scripts._VENDOR_PRELUDES[config.remote_gpu_vendor()]() if gpu else ""
    # Reuse the synced checkout's existing venv. uv --no-sync MUST NOT mutate a
    # shared venv while another SLURM allocation is executing in that checkout.
    uv_cmd = f"{config.shell_remote_uv()} run --no-sync --package pyrite-mc python -m marimo"
    run = f"{uv_cmd} {command}"
    body = (
        f"cd {config.shell_word(config.remote_dir())}\n"
        f"{vendor_prelude}\n"
        "export PYRITE_HOME=" + config.shell_word(config.remote_dir()) + "\n"
        "export PYRITE_CATALOG=" + config.shell_word(config.remote_catalog_path()) + "\n"
        "export PYRITE_USER_CATALOG=" + config.shell_word(config.remote_user_catalog_path()) + "\n"
    )
    if app in {"analysis", "viewer"}:
        variable = "PYRITE_ANALYZE_INITIAL" if app == "analysis" else "PYRITE_VIEWER_INITIAL"
        body += f"export {variable}={config.shell_word(material or 'hopg')}\n"
    if batch:
        body += (
            f'{run} >> "$JOBDIR/log" 2>&1\n'
            'test -s "$JOBDIR/app.html"\n'
            'echo "done $(date -Is)" > "$JOBDIR/state"\n'
        )
    else:
        # Pick a free loopback port on the allocated node, not on the login host.
        # A random ephemeral assignment would work too, but probing from the
        # per-app TUNNEL_PORT keeps the first instance predictable.
        body += f"""PY="$PYRITE_HOME/.venv/bin/python"
if [ ! -x "$PY" ]; then echo 'missing remote .venv; install pyrite-mc[apps] first' >&2; exit 1; fi
PORT=$("$PY" -c 'import socket,sys
for port in range(int(sys.argv[1]), 65536):
    with socket.socket() as sock:
        try: sock.bind(("127.0.0.1", port))
        except OSError: continue
        print(port); break
    ' {base_port})
test -n "$PORT"
"$PY" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$JOBDIR/token"
{run} >> "$JOBDIR/log" 2>&1 &
APP_PID=$!
ready=0
for _attempt in $(seq 1 120); do
  if ! kill -0 "$APP_PID" 2>/dev/null; then break; fi
  if "$PY" -c 'import socket,sys
with socket.socket() as sock:
    sock.settimeout(0.2)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))))' "$PORT"; then
    ready=1; break
  fi
  sleep 0.5
done
if [ "$ready" != 1 ]; then
  echo 'marimo did not start; see job log' >> "$JOBDIR/log"
  exit 1
fi
NODE=$(hostname -f 2>/dev/null || hostname)
# Token and URL are only readable by the job owner (umask 077).
{{
  printf 'node: %s\\n' "$NODE"
  printf 'port: %s\\n' "$PORT"
  printf 'url: http://127.0.0.1:%s/?access_token=%s\\n' "$PORT" "$(cat "$JOBDIR/token")"
  printf 'ready: yes\\n'
}} >> "$JOBDIR/meta"
wait "$APP_PID"
echo "done $(date -Is)" > "$JOBDIR/state"
"""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=pyrite-app-{app}
#SBATCH --partition={config.slurm_partition()}
{nodes}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
{gpu_line}#SBATCH --time={walltime}
#SBATCH --output={sbatch_jobdir}/slurm-%j.out
#SBATCH --error={sbatch_jobdir}/slurm-%j.err

set -euo pipefail
umask 077
JOBDIR={config.shell_word(jobdir)}
chmod 700 "$JOBDIR"
chmod 600 "$JOBDIR/meta"
finish() {{
  code=$?
  if [ "$code" != 0 ]; then echo "FAILED (exit $code) $(date -Is)" > "$JOBDIR/state"; fi
}}
trap finish EXIT
# Avoid importing from a checkout while a sync is replacing its contents.
if command -v flock >/dev/null 2>&1; then
  flock -s -w {config.SYNC_LOCK_WAIT_SECONDS} \
    {config.shell_word(config.remote_sync_lock_path())} true || exit 1
fi
echo "running $(date -Is)" > "$JOBDIR/state"
{body}"""


def _metadata(app: str, material: str | None, batch: bool) -> str:
    return f"kind: app\napp: {app}\nmaterials: {material or '-'}\nbatch: {str(batch).lower()}\n"


def submit(
    app: str,
    material: str | None = None,
    *,
    argv: list[str],
    notebook: str,
    base_port: int,
    edit: bool = False,
    gpu: bool = False,
    walltime: str = "04:00:00",
    batch: bool = False,
) -> str:
    """Submit an app, returning its job ID; leaves it running on local exit."""
    if batch and edit:
        raise ValueError("--batch cannot be used with --edit")
    jobid = scripts._new_jobid()
    script = _script(
        jobid,
        app,
        material,
        argv=argv,
        notebook=notebook,
        base_port=base_port,
        gpu=gpu,
        walltime=walltime,
        batch=batch,
    )
    transport.sync_code()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    upload = scripts._write_job_script_command(jobdir, _metadata(app, material, batch))
    jobs._stage_job_script(jobid, [], upload, script)
    scheduler_id = jobs._submit_staged_job(jobid, [])
    print(f"Submitted app {app} as job {jobid} (SLURM {scheduler_id}).")
    print(f"Reattach: pyrite remote app attach {jobid}")
    print(f"Cancel: pyrite remote job stop {jobid} --yes")
    return jobid


def _fields(meta: str) -> dict[str, str]:
    return dict(line.split(": ", 1) for line in meta.splitlines() if ": " in line)


def _read_job(jobid: str) -> dict[str, str]:
    transport._check_shell_tokens([jobid])
    raw = state._job_metadata(jobid)
    fields = _fields(raw)
    if fields.get("kind") != "app" or fields.get("app") not in _APPS:
        raise ValueError(f"job {jobid} is not a PyRITE app job")
    return fields


def _local_port(preferred: int) -> int:
    for port in range(preferred, 65536):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("no local loopback port available")


def _tunnel_argv(node: str, local_port: int, remote_port: int) -> list[str]:
    host = config.remote_host()
    # Hostnames come from SLURM/hostname, never user-supplied shell fragments.
    config.validate_remote_target(node)
    destination = host if node == host or node.split(".")[0] == host.split(".")[0] else node
    jump = [] if destination == host else ["-J", host]
    return config.ssh_argv(
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        *jump,
        destination,
    )


def _ready(fields: dict[str, str]) -> tuple[str, int, str] | None:
    if fields.get("ready") != "yes":
        return None
    node, port, url = (fields.get(key, "") for key in ("node", "port", "url"))
    if not _READY_RE.fullmatch(node) or not port.isascii() or not port.isdigit():
        raise ValueError("invalid app node/port in remote job metadata")
    remote_port = int(port)
    parsed = urlsplit(url)
    if (
        not 1 <= remote_port <= 65535
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != remote_port
        or not parse_qs(parsed.query).get("access_token")
    ):
        raise ValueError("invalid authenticated URL in remote job metadata")
    return node, remote_port, url


def _wait(jobid: str, *, batch: bool, timeout: int = _STARTUP_TIMEOUT_SECONDS) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        fields = _read_job(jobid)
        state_text = state._job_state(jobid)
        if batch and state_text.startswith("done"):
            return fields
        if not batch and _ready(fields) is not None:
            if state_text.startswith(("FAILED", "cancelled", "done")):
                raise RuntimeError(f"app {jobid} already terminated: {state_text}")
            return fields
        if state_text.startswith(("FAILED", "cancelled")):
            raise RuntimeError(
                f"app job {jobid} terminated: {state_text}; inspect pyrite remote job logs {jobid}"
            )
        time.sleep(_STARTUP_POLL_SECONDS)
    raise TimeoutError(
        f"job {jobid} is still queued/starting; retry pyrite remote app attach {jobid}"
    )


def _pull_html(jobid: str, app: str) -> Path:
    destination = output_dir("results") / f"remote-app-{app}-{jobid}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = config.remote_path(config.JOBS_SUBDIR, jobid, "app.html")
    transport._ssh_download(f"cat {config.shell_arg(path)}", destination, label="Pulling app HTML")
    return destination


def attach(jobid: str) -> None:
    """Reopen the tunnel, or pull HTML when a batch app has finished."""
    fields = _wait(jobid, batch=_read_job(jobid).get("batch") == "true")
    app = fields["app"]
    if fields.get("batch") == "true":
        print(f"HTML written to {_pull_html(jobid, app)}")
        return
    node, port, remote_url = _ready(fields) or ("", 0, "")
    if not node:
        raise RuntimeError("remote marimo server is not ready")
    local_port = _local_port(port)
    url_parts = urlsplit(remote_url)
    local_url = f"http://127.0.0.1:{local_port}{url_parts.path}?{url_parts.query}"
    print(f"App {app} is available at {local_url}")
    print(f"Ctrl-C disconnects the tunnel; SLURM job {jobid} keeps running.")
    command = _tunnel_argv(node, local_port, port)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("SSH forwarding failed; check compute-node SSH policy")
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", local_port)) == 0:
                    break
            time.sleep(0.2)
        else:
            raise TimeoutError("SSH tunnel did not become ready within 20 seconds")
        webbrowser.open(local_url)
        process.wait()
    except KeyboardInterrupt:
        print(f"\nTunnel disconnected. Reattach: pyrite remote app attach {jobid}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
