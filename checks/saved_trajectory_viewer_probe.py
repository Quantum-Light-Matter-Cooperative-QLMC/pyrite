"""Fresh-process native/headless saved-viewer costs; never browser FPS.

uv run --extra trajectory-viewer python checks/saved_trajectory_viewer_probe.py \
  capture.h5 --output trial.json --image frame.png

Make synthetic whole-shower copies with --replicate N --scaled-output capture.h5.
This generates no new transport and proves no production shower statistics.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from pyrite.montecarlo.trajectories import write_trajectory_artifact
from pyrite.trajectory_viewer.render import CaptureViewer
from pyrite.trajectory_viewer.selection import load_selection, plan_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--scaled-output", type=Path)
    args = parser.parse_args()
    if args.scaled_output:
        plan = plan_selection(args.artifact)
        source = load_selection(plan)
        count = plan.segments * args.replicate
        estimate = 1024**3 + (plan.estimated_peak_bytes - 1024**3) * args.replicate
        if args.replicate < 1 or estimate * 2 > psutil.virtual_memory().available:
            raise RuntimeError("insufficient memory for requested synthetic scaling")
        data = {}
        for name, values in source.items():
            if name == "segment_id":
                continue
            data[name] = np.tile(values, (args.replicate,) + (1,) * (values.ndim - 1))
        stride = int(max(source["electron_id"].max(), source["track_id"].max())) + 1
        for name in ("electron_id", "track_id", "parent_id"):
            for i in range(args.replicate):
                block = data[name][i * plan.segments : (i + 1) * plan.segments]
                block[block >= 0] += i * stride
        write_trajectory_artifact(
            args.scaled_output,
            data,
            case=plan.case,
            scene=plan.scene,
            provenance=dict(
                synthetic_whole_shower_replication=args.replicate,
                source=str(args.artifact),
                source_sha256=hashlib.file_digest(args.artifact.open("rb"), "sha256").hexdigest(),
            ),
        )
        print(json.dumps(dict(segments=count, estimated_peak_bytes=estimate)))
        return
    if not args.output or not args.image:
        parser.error("--output and --image are required for a measured trial")
    stop = threading.Event()
    process = psutil.Process()
    peak = 0
    samples = 0

    def sample():
        nonlocal peak, samples
        while not stop.wait(0.02):
            total = 0
            for member in [process, *process.children(recursive=True)]:
                try:
                    total += member.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, total)
            samples += 1

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    times = {}
    tick = perf_counter()
    plan = plan_selection(args.artifact, max_segments=10_000_000)
    times["selection_scan_s"] = perf_counter() - tick
    if plan.estimated_peak_bytes * 1.5 > psutil.virtual_memory().available:
        raise RuntimeError("insufficient memory for measured trial")
    tick = perf_counter()
    data = load_selection(plan, memory_budget_bytes=plan.estimated_peak_bytes + 1)
    times["load_selected_arrays_s"] = perf_counter() - tick
    tick = perf_counter()
    viewer = CaptureViewer(plan, data, off_screen=True)
    times["scene_build_s"] = perf_counter() - tick
    tick = perf_counter()
    viewer.screenshot(args.image, overwrite=True)
    times["first_headless_png_s"] = perf_counter() - tick
    times["camera_render_s"] = []
    for _ in range(3):
        tick = perf_counter()
        viewer.plotter.camera.azimuth += 15
        viewer.plotter.render()
        viewer.plotter.screenshot()
        times["camera_render_s"].append(perf_counter() - tick)
    tick = perf_counter()
    energy = float(data["E_start_keV"].max()) * 0.1
    filtered = viewer.apply_filters(energy_min=energy, generation_max=0)
    viewer.plotter.screenshot()
    times["energy_generation_filter_and_image_s"] = perf_counter() - tick
    tick = perf_counter()
    time_max = float(viewer.mesh.cell_data["time_fs"].max()) * 0.5
    timed = viewer.apply_filters(time_max=time_max)
    viewer.plotter.screenshot()
    times["time_filter_and_image_s"] = perf_counter() - tick
    tick = perf_counter()
    clipped = viewer.apply_filters(clip_x=float(np.mean(viewer.mesh.bounds[:2])))
    viewer.plotter.screenshot()
    times["clipping_and_image_s"] = perf_counter() - tick
    tick = perf_counter()
    info = viewer.inspect_cell(0)
    times["inspect_cell_s"] = perf_counter() - tick
    renderer = viewer.plotter.render_window.ReportCapabilities()
    viewer.close()
    stop.set()
    monitor.join()
    result = dict(
        schema="pyrite.saved-viewer-trial.v1",
        artifact=args.artifact.name,
        input_sha256=hashlib.file_digest(args.artifact.open("rb"), "sha256").hexdigest(),
        input_bytes=args.artifact.stat().st_size,
        segments=plan.segments,
        tracks=int(len(np.unique(data["track_id"]))),
        selected_attributes=plan.fields,
        estimated_peak_bytes=plan.estimated_peak_bytes,
        peak_process_tree_rss_bytes=peak,
        rss_sampling_interval_s=0.02,
        rss_samples=samples,
        gpu_memory_bytes=None,
        gpu_memory_reason="not sampled by this probe; actual renderer capabilities recorded below",
        client_connected=False,
        browser_payload_bytes=None,
        browser_latency_s=None,
        backend="PyVista native offscreen",
        window_size=[1000, 700],
        cold_process=True,
        times=times,
        filtered_cells=filtered,
        timed_cells=timed,
        clipped_cells=clipped,
        picked_identity=info,
        image_bytes=args.image.stat().st_size,
        renderer=renderer,
        environment=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            cpu_count=os.cpu_count(),
            total_memory_bytes=psutil.virtual_memory().total,
            rendering_environment={
                name: os.environ.get(name)
                for name in (
                    "LIBGL_ALWAYS_SOFTWARE",
                    "MESA_D3D12_DEFAULT_ADAPTER_NAME",
                    "MPLCONFIGDIR",
                )
            },
            versions={
                name: importlib.metadata.version(name)
                for name in ("pyvista", "vtk", "trame", "numpy", "h5py")
            },
        ),
        case=plan.case,
        provenance=plan.provenance,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(segments=plan.segments, times=times, peak_mib=peak / 1024**2)))


if __name__ == "__main__":
    main()
