"""Capture a production CXR batch, then compare unchanged CUDA reducers (#200)."""

import argparse
import hashlib
import json
import platform
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(path):
    import cupy as cp

    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.montecarlo.runner import run_case
    from pyrite.montecarlo.spectrum.cxr._batched import _CxrBatch

    if args.artifact:
        from pyrite.montecarlo.runner.artifacts import spectrum_from_artifact
        from pyrite.montecarlo.trajectories import read_trajectory_artifact

        artifact = read_trajectory_artifact(args.artifact)
        case = artifact.case
        if "spectrum_kwargs" in artifact.spectrum_inputs:
            from pyrite.montecarlo.spectrum.cxr import mc_cxr_spectrum

            spectrum_kwargs = dict(artifact.spectrum_inputs["spectrum_kwargs"])
            if args.incoherent_artifact:
                spectrum_kwargs["coherent"] = False
            if args.quadrature:
                spectrum_kwargs["line_quadrature"] = args.quadrature
            if spectrum_kwargs.get("coherent", False):
                raise ValueError(
                    "saved spectrum is coherent; select --incoherent-artifact explicitly"
                )
    else:
        cases = build_cases(material_sweep(args.material, catalog_profile=args.profile))
        case = max(cases, key=lambda c: (c.E0_keV, c.thickness_ang, -c.tilt_deg))
        case = replace(case, Ne=args.electrons, Ne_brem=100)

    class Captured(Exception):
        pass

    original = _CxrBatch.flush

    def save(batch):
        if not batch.n_cxr_rows:
            return
        arrays = {
            name: cp.asnumpy(cp.concatenate(getattr(batch, name))) for name in ("E_r", "aw", "w")
        }
        arrays["grid"] = cp.asnumpy(batch.E_grid)
        if batch.bin_edges is not None:
            arrays["edges"] = cp.asnumpy(batch.bin_edges)
            arrays["inv_width"] = cp.asnumpy(batch.bin_inv_width)
        np.savez(path, **arrays)
        metadata = {
            "case": case.to_dict() if hasattr(case, "to_dict") else case,
            "source_revision": args.revision,
            "fixture_sha256": digest(path),
            "route": "bin-mean" if batch.bin_edges is not None else "points",
            "lines": batch.n_cxr_rows,
            "nodes": int(batch.E_grid.size),
            "capture": "first production flush; reduction intentionally aborted",
            "transport_artifact_sha256": digest(args.artifact) if args.artifact else None,
            "spectrum_kwargs": spectrum_kwargs
            if args.artifact and "spectrum_kwargs" in artifact.spectrum_inputs
            else None,
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, default=lambda v: np.asarray(v).tolist())
        )
        print(json.dumps(metadata, default=lambda v: np.asarray(v).tolist()), flush=True)
        raise Captured

    _CxrBatch.flush = save
    try:
        if args.artifact:
            if "spectrum_kwargs" in artifact.spectrum_inputs:
                mc_cxr_spectrum(
                    artifact.transport, artifact.spectrum_inputs["energy_eV"], **spectrum_kwargs
                )
            else:
                spectrum_from_artifact(artifact, record_timing=True)
        else:
            run_case(case, record_timing=True)
    except Captured:
        return
    finally:
        _CxrBatch.flush = original
    raise RuntimeError("case did not reach the batched CUDA reducer")


def replay(path):
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import run_bin_mean_reduction_kernel
    from pyrite.montecarlo.spectrum.cxr_jit_kernel import SpectrumKernelConfig, run_reduction_kernel

    arrays = dict(np.load(path))
    props = cp.cuda.runtime.getDeviceProperties(0)
    report = {
        "revision": args.revision,
        "fixture_sha256": digest(path),
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "gpu": props["name"].decode(),
        "gpu_memory_bytes": props["totalGlobalMem"],
        "sm_count": props["multiProcessorCount"],
        "repeats": args.repeats,
        "protocol": "one warmup per arm; point arms alternate forward/reverse; bin methods grouped; CUDA events and synchronized wall; reset excluded",
        "point_tolerance": "100 float32 eps times baseline peak, fixed before comparing timings; diagnostic only",
        "workloads": [],
    }
    for label, line_count, stride in (
        ("small", min(4097, len(arrays["w"])), max(1, len(arrays["grid"]) // 1025)),
        ("full-batch", len(arrays["w"]), 1),
    ):
        e, a, w = [cp.asarray(arrays[name][:line_count]) for name in ("E_r", "aw", "w")]
        grid = cp.asarray(arrays["grid"][::stride], dtype=cp.float32)
        out = cp.zeros(len(grid), dtype=cp.float32)
        configs = [(t, k) for t in (128, 256, 512, 1024) for k in (1, 2, 3, 4)]
        times = {config: [] for config in configs}
        walls = {config: [] for config in configs}
        spectra = {}
        for t, k in configs:
            out.fill(0)
            run_reduction_kernel(e, a, w, grid, out=out, config=SpectrumKernelConfig(t, k))
        cp.cuda.get_current_stream().synchronize()
        for repeat in range(args.repeats):
            for t, k in configs if repeat % 2 == 0 else list(reversed(configs)):
                out.fill(0)
                cp.cuda.get_current_stream().synchronize()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                wall_start = time.perf_counter()
                start.record()
                run_reduction_kernel(e, a, w, grid, out=out, config=SpectrumKernelConfig(t, k))
                end.record()
                end.synchronize()
                walls[t, k].append(1000 * (time.perf_counter() - wall_start))
                times[t, k].append(cp.cuda.get_elapsed_time(start, end))
                if repeat == 0:
                    spectra[t, k] = cp.asnumpy(out)
        baseline = spectra[512, 3].astype(np.float64)
        peak = float(np.max(abs(baseline)))
        floor = max(peak * 1e-6, np.finfo(np.float32).tiny)
        tolerance = 100 * np.finfo(np.float32).eps * peak
        rows = []
        for t, k in configs:
            diff = abs(spectra[t, k].astype(np.float64) - baseline)
            rows.append(
                {
                    "threads": t,
                    "energies": k,
                    "gpu_ms": times[t, k],
                    "wall_ms": walls[t, k],
                    "median_gpu_ms": float(np.median(times[t, k])),
                    "pairs_per_s": line_count * len(grid) / (np.median(times[t, k]) * 1e-3),
                    "max_abs": float(diff.max()),
                    "max_rel_with_floor": float(np.max(diff / np.maximum(abs(baseline), floor))),
                    "peak_scaled_abs": float(diff.max() / peak),
                    "within_peak_roundoff_allowance": bool(np.all(diff <= tolerance)),
                    "relative_discrete_yield_change": float(
                        (
                            np.trapezoid(
                                spectra[t, k].astype(np.float64),
                                arrays["grid"][::stride].astype(np.float64),
                            )
                            - np.trapezoid(baseline, arrays["grid"][::stride].astype(np.float64))
                        )
                        / np.trapezoid(baseline, arrays["grid"][::stride].astype(np.float64))
                    ),
                }
            )
        workload = {"name": label, "lines": line_count, "nodes": len(grid), "points": rows}
        workload["bin_mean_pending"] = "edges" in arrays and not args.skip_bin_mean
        workload["bin_mean_skipped"] = bool("edges" in arrays and args.skip_bin_mean)
        report["workloads"].append(workload)
        args.output.write_text(json.dumps(report, indent=2))
        if "edges" in arrays and not args.skip_bin_mean:
            # Adjacent original bins only: preserve the production bin arithmetic.
            nbins = min(1025, len(grid)) if label == "small" else len(grid)
            edges = cp.asarray(arrays["edges"][: nbins + 1])
            inv = cp.asarray(arrays["inv_width"][:nbins])
            bout = cp.zeros(nbins, dtype=cp.float32)
            bin_rows = []
            methods = (
                ("pairs", "tree", "auto")
                if line_count * nbins <= args.max_bin_pairs
                else ("auto", "tree")
            )
            for method in methods:
                bout.fill(0)
                run_bin_mean_reduction_kernel(e, a, w, edges, inv, out=bout, method=method)
                cp.cuda.get_current_stream().synchronize()
                samples = []
                results = None
                for _ in range(args.repeats):
                    bout.fill(0)
                    cp.cuda.get_current_stream().synchronize()
                    t0 = time.perf_counter()
                    run_bin_mean_reduction_kernel(e, a, w, edges, inv, out=bout, method=method)
                    cp.cuda.get_current_stream().synchronize()
                    samples.append(1000 * (time.perf_counter() - t0))
                    results = cp.asnumpy(bout)
                if method == methods[0]:
                    bref = results.astype(np.float64)
                delta = abs(results.astype(np.float64) - bref)
                bin_rows.append(
                    {
                        "method": method,
                        "wall_ms": samples,
                        "max_abs_vs_reference": float(delta.max()),
                        "peak_scaled_abs_vs_reference": float(
                            delta.max() / max(abs(bref).max(), np.finfo(float).tiny)
                        ),
                    }
                )
            workload["bin_mean"] = {
                "bins": nbins,
                "results": bin_rows,
                "reference_method": methods[0],
                "forced_pairs_skipped": "pairs" not in methods,
                "max_bin_pairs": args.max_bin_pairs,
            }
        workload["bin_mean_pending"] = False
        report["gpu_pool_reserved_bytes_at_end"] = cp.get_default_memory_pool().total_bytes()
        report["host_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(workload), flush=True)
    # Pool reservation is recorded, not misrepresented as peak device memory.


def derive(path):
    """Create a labeled size stress fixture from saved real lines."""
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    arrays = dict(np.load(path))
    output = {key: np.tile(arrays[key], args.line_repeats) for key in ("E_r", "aw", "w")}
    output["w"] /= np.float32(args.line_repeats)
    output["grid"] = np.linspace(
        arrays["grid"][0], arrays["grid"][-1], args.grid_nodes, dtype=np.float32
    )
    output["edges"], widths = _energy_bin_edges_and_widths(output["grid"])
    output["inv_width"] = 1.0 / widths
    np.savez(args.output, **output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_sha256": digest(path),
                "fixture_sha256": digest(args.output),
                "lines_repeated": args.line_repeats,
                "weights_divided_by": args.line_repeats,
                "nodes": args.grid_nodes,
                "classification": "derived stress fixture; not an independent production case",
            },
            indent=2,
        )
    )


def profile(path):
    """Launch the default kernel twice, so Nsight can skip its warmup."""
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr_jit_kernel import (
        DEFAULT_SPECTRUM_KERNEL_CONFIG,
        run_reduction_kernel,
    )

    arrays = dict(np.load(path))
    e, a, w = (cp.asarray(arrays[key]) for key in ("E_r", "aw", "w"))
    grid = cp.asarray(arrays["grid"], dtype=cp.float32)
    out = cp.zeros(len(grid), dtype=cp.float32)
    for _ in range(2):
        out.fill(0)
        run_reduction_kernel(e, a, w, grid, out=out, config=DEFAULT_SPECTRUM_KERNEL_CONFIG)
        cp.cuda.get_current_stream().synchronize()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("capture", "replay", "profile", "derive"))
parser.add_argument("fixture", type=Path)
parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
parser.add_argument("--revision", required=True)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--material", default="hbn")
parser.add_argument("--profile", default="high_energy_demo")
parser.add_argument("--electrons", type=int, default=8000)
parser.add_argument("--artifact", type=Path, help="Replay saved transport instead of simulating")
parser.add_argument(
    "--incoherent-artifact",
    action="store_true",
    help="Select incoherent spectrum on a saved spectrum-only transport",
)
parser.add_argument(
    "--quadrature",
    choices=("node", "bin-mean"),
    help="Quadrature for a saved spectrum-only transport",
)
parser.add_argument(
    "--max-bin-pairs",
    type=int,
    default=5_000_000_000,
    help="Skip forced all-pairs bin reductions above this pair budget; auto/tree still run",
)
parser.add_argument(
    "--skip-bin-mean", action="store_true", help="Measure only point-sampled launch configurations"
)
parser.add_argument("--line-repeats", type=int, default=32)
parser.add_argument("--grid-nodes", type=int, default=100003)
args = parser.parse_args()
if args.repeats < 5:
    parser.error("--repeats must be at least 5")
if args.mode == "derive" and (
    args.output.suffix != ".npz" or args.line_repeats < 1 or args.grid_nodes < 2
):
    parser.error(
        "derive requires --output NAME.npz, positive --line-repeats, and --grid-nodes >= 2"
    )
if args.mode == "capture":
    capture(args.fixture)
elif args.mode == "replay":
    replay(args.fixture)
elif args.mode == "profile":
    profile(args.fixture)
else:
    derive(args.fixture)
