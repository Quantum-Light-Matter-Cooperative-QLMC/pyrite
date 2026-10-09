"""Isolated remote viewer supervisor: sampled server RSS and device-wide VRAM.

Run with the project runner on the rendering host. First argument is an output
directory; remaining arguments are the real trajectory launch options. SIGTERM
stops only this supervisor's child server. No capture/transport is generated.
"""

import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        supervisor_pid=os.getpid(),
        host=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,
        rendering_environment={
            name: os.environ.get(name)
            for name in ("LIBGL_ALWAYS_SOFTWARE", "MESA_D3D12_DEFAULT_ADAPTER_NAME")
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ("vtk", "pyvista", "trame", "trame-vtk")
        },
        peak_server_tree_rss_mib=0.0,
        target_rss_interval_s=0.05,
        rss_samples=0,
        gpu_samples=[],
        gpu_scope="device-wide used VRAM, not per-process; other users and desktop allocations included",
    )

    def gpu():
        tick = time.perf_counter()
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        report["gpu_samples"].append(
            dict(
                elapsed_s=time.perf_counter() - start,
                query_s=time.perf_counter() - tick,
                value=result.stdout.strip(),
                error=result.stderr.strip(),
                exit_code=result.returncode,
            )
        )

    start = time.perf_counter()

    def save():
        staged = output / "resources.json.tmp"
        staged.write_text(json.dumps(report, indent=2) + "\n")
        staged.replace(output / "resources.json")

    try:
        gpu()
    except (OSError, subprocess.TimeoutExpired) as error:
        report["gpu_unavailable"] = str(error)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (output / "server.log").open("w") as log:
        launch = sys.argv[2:]
        native = launch[:1] == ["--native-probe"]
        if native:
            launch = [
                *launch[1:],
                "--output",
                str(output / "phases.json"),
                "--image",
                str(output / "frame.png"),
            ]
        child = subprocess.Popen(
            [
                sys.executable,
                str(
                    Path(__file__).with_name(
                        "saved_trajectory_viewer_probe.py"
                        if native
                        else "saved_trajectory_browser_worker.py"
                    )
                ),
                *([] if native else [str(output / "phases.json")]),
                *launch,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        report["server_pid"] = child.pid
        last_save = last_gpu = last_sample = time.perf_counter()
        report["max_rss_sample_gap_s"] = 0.0
        try:
            while not stopped and child.poll() is None:
                now = time.perf_counter()
                report["max_rss_sample_gap_s"] = max(
                    report["max_rss_sample_gap_s"], now - last_sample
                )
                last_sample = now
                try:
                    process = psutil.Process(child.pid)
                    total = sum(
                        p.memory_info().rss for p in [process, *process.children(recursive=True)]
                    )
                    report["peak_server_tree_rss_mib"] = max(
                        report["peak_server_tree_rss_mib"], total / 1024**2
                    )
                    report["rss_samples"] += 1
                except psutil.NoSuchProcess:
                    pass
                if now - last_gpu >= 1 and "gpu_unavailable" not in report:
                    gpu()
                    last_gpu = now
                if now - last_save >= 0.5:
                    save()
                    last_save = now
                time.sleep(0.05)
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            report["server_exit_code"] = child.returncode
            save()


if __name__ == "__main__":
    main()
