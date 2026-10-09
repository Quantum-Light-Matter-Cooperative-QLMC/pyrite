"""Repeat isolated remote-server/browser trials over a localhost SSH tunnel.

Stage the same source snapshot, optional dependencies and named captures into
an isolated remote directory first. Example::

    uv run --extra trajectory-viewer --with playwright python \
      checks/saved_trajectory_remote_trials.py --host qlmc --root /tmp/RUNTIME \
      --captures /tmp/CAPTURES --output /tmp/TRIALS

The rendering host's uv must be available at ~/.local/bin/uv. Only fresh child
servers created by this invocation are stopped. Remote copies and sidecars are
retrieved and compared with the actual browser download hashes.
"""

import argparse
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--adapter", help="Mesa D3D12 adapter substring; verify the recorded renderer."
    )
    parser.add_argument("--scales", nargs="+", default=["x4", "x34"])
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Use captures staged only on the rendering host; hashes remain recorded.",
    )
    parser.add_argument("--max-segments", type=int, default=2000000)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument(
        "--run-prefix",
        default="measured",
        help="Fresh remote output prefix when reusing a staged runtime.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def ssh(command, **kwargs):
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host, command], check=True, **kwargs
        )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        local_port = sock.getsockname()[1]
    remote_port = 38719
    tunnel = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{local_port}:127.0.0.1:{remote_port}",
            args.host,
        ]
    )
    processes = []
    try:
        time.sleep(1)
        if tunnel.poll() is not None:
            raise RuntimeError("SSH tunnel failed")
        for scale in args.scales:
            capture = args.captures / f"{scale}.h5"
            remote_hash = ssh(
                f"sha256sum {shlex.quote(args.root + '/' + capture.name)}",
                capture_output=True,
                text=True,
            ).stdout.split()[0]
            if not args.remote_only:
                with capture.open("rb") as stream:
                    expected_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if remote_hash != expected_hash:
                    raise RuntimeError("local and remote inputs differ")
            for trial in range(1, args.repeats + 1):
                name = f"{scale}-{trial}"
                suffix = "-" + args.adapter.lower() if args.adapter else ""
                remote_run = f"{args.root}/{args.run_prefix}{suffix}-{name}"
                ssh(f"test ! -e {shlex.quote(remote_run)}")
                # Do not accidentally attach to a server left by an interrupted
                # invocation while the new child prepares its scene.
                ssh("command -v ss >/dev/null && test -z \"$(ss -H -lnt 'sport = :38719')\"")
                invocation = [
                    "python",
                    "checks/saved_trajectory_remote_worker.py",
                    remote_run,
                    str(Path(args.root) / capture.name),
                    "--port",
                    str(remote_port),
                    "--max-segments",
                    str(args.max_segments),
                    "--memory-mib",
                    str(args.memory_mib),
                    "--output-dir",
                    remote_run + "/exports",
                ]
                command = f"cd {shlex.quote(args.root)} && ~/.local/bin/uv run --extra trajectory-viewer {shlex.join(invocation)}"
                if args.adapter:
                    command = f"cd {shlex.quote(args.root)} && MESA_D3D12_DEFAULT_ADAPTER_NAME={shlex.quote(args.adapter)} ~/.local/bin/uv run --extra trajectory-viewer {shlex.join(invocation)}"
                start = time.perf_counter()
                log = (args.output / f"{name}-ssh.log").open("w")
                process = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", args.host, command],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                processes.append(process)
                report_path = args.output / f"{name}.json"
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("saved_trajectory_browser_probe.py")),
                            str(capture),
                            "--server-url",
                            f"http://127.0.0.1:{local_port}",
                            "--output",
                            str(report_path),
                            *(["--capture-sha256", remote_hash] if args.remote_only else []),
                        ],
                        check=True,
                        timeout=300,
                    )
                finally:
                    resource_path = remote_run + "/resources.json"
                    resource = json.loads(
                        ssh(
                            f"cat {shlex.quote(resource_path)}", capture_output=True, text=True
                        ).stdout
                    )
                    # The PID is read from this run's exclusively created report.
                    ssh(f"kill -TERM {int(resource['supervisor_pid'])}")
                    process.wait(timeout=15)
                    log.close()
                    subprocess.run(
                        [
                            "scp",
                            "-q",
                            "-r",
                            f"{args.host}:{remote_run}",
                            str(args.output / f"{name}-server"),
                        ],
                        check=True,
                    )
                record = json.loads(report_path.read_text())
                server_dir = args.output / f"{name}-server"
                record["remote_resources"] = json.loads((server_dir / "resources.json").read_text())
                record["server_preparation"] = json.loads((server_dir / "phases.json").read_text())
                if (
                    record["steps"]["camera_orbit_0"]["displayed_cells"]
                    != record["server_preparation"]["segments"]
                ):
                    raise RuntimeError("browser selection does not match the owned server's input")
                if (
                    args.adapter
                    and args.adapter.lower() not in record["server_preparation"]["renderer"].lower()
                ):
                    record["failures"].append(f"requested adapter {args.adapter} was not used")
                    report_path.write_text(json.dumps(record, indent=2) + "\n")
                    raise RuntimeError(
                        "requested rendering adapter unavailable; see raw renderer report"
                    )
                record["startup"]["ssh_launch_to_first_image_s"] = (
                    record["probe_tick_monotonic"]
                    + record["startup"]["probe_start_to_first_image_s"]
                    - start
                )
                for download in record["downloads"].values():
                    saved = server_dir / "exports" / download["server_filename"]
                    with saved.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != download["sha256"]:
                            raise RuntimeError("remote export and browser download differ")
                    meta = json.loads(saved.with_suffix(saved.suffix + ".json").read_text())
                    if meta["image_size"] != download["size"] or meta["scale"] != "closeup":
                        raise RuntimeError("remote export provenance differs from browser view")
                record["remote_export_verification"] = (
                    "download hashes match both remote outputs; sidecar size/scale checked"
                )
                record["adapter_request"] = args.adapter
                record["network"] = (
                    "actual SSH tunnel between client workstation and configured lab host; no emulated throttling"
                )
                report_path.write_text(json.dumps(record, indent=2) + "\n")
                print(f"remote trial {name} passed", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        tunnel.terminate()
        tunnel.wait(timeout=10)


if __name__ == "__main__":
    main()
