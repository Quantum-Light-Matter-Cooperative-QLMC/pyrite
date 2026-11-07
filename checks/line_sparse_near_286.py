"""Bounded sparse-near / existing hierarchical-far prototype for #286.

Remote GPU measurement only. Replays committed-kernel flush inputs; production
code/defaults are unchanged. Conservatively partition the existing tree leaves
with its guarded near radius, evaluate compact pairs in GPU-wide parallelism,
and sum sorted per-bin segments without floating-point atomics. Report repeat
agreement rather than assuming determinism. Sorting and mapping are timed.
"""

import argparse
import json
import time
from functools import cache
from pathlib import Path

import numpy as np

from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as quad
from pyrite.montecarlo.spectrum.cxr import _bin_tree as tree


def near_intervals(e, a, edges, exact_widths, xp=np):
    """Half-open bin intervals using the production tree's radius guard."""
    e = xp.asarray(e, dtype=xp.float64)
    physical = float(exact_widths) * np.pi / xp.asarray(a, dtype=xp.float64)
    radius = physical * (1 + 2e-6) + 1e-12 * xp.maximum(xp.abs(e), physical)
    n = len(edges) - 1
    first = xp.clip(xp.searchsorted(edges, e - radius, side="right") - 1, 0, n)
    stop = xp.clip(xp.searchsorted(edges, e + radius, side="left"), 0, n)
    return first.astype(xp.int64), xp.maximum(first, stop).astype(xp.int64)


def pair_keys(first, offsets, start, stop, xp=np, *, sort=True):
    """Map a bounded line-major ordinal tile to (bin, line) keys.

    ``sort=False`` keeps line-major order: each warp then holds one line across
    consecutive bins, so its lanes stay on the same ``Si`` branch (#1003).
    """
    ordinal = xp.arange(start, stop, dtype=xp.int64)
    line = xp.searchsorted(offsets, ordinal, side="right") - 1
    bins = first[line] + ordinal - offsets[line]
    keys = bins * len(first) + line
    return xp.sort(keys) if sort else keys


def far_source():
    """Keep production traversal/moments, skip only the compact leaf pairs."""
    defines = """
#define PYRITE_TREE_NEAR_ARGS const long long *near_first, const long long *near_stop,
#define PYRITE_TREE_SKIP_NEAR(line, bin) ((bin) >= near_first[line] && (bin) < near_stop[line])
"""
    source = tree._TREE_SOURCE.replace(
        "void pyrite_bin_tree_reduce(", "void pyrite_sparse_far_reduce("
    )
    return quad.SINCSQ_BIN_PREAMBLE + defines + source


NEAR_SOURCE = r"""
extern "C" __global__ void sparse_near(
    const long long *keys, long long count, long long n_lines,
    const float *e, const float *a, const float *w,
    const double *edges, const double *iw, double *out, float exact_widths
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long bin = keys[i] / n_lines, line = keys[i] % n_lines;
    double lo = edges[bin], hi = edges[bin + 1];
    float lf = (float)lo, hf = (float)hi, mean;
    double value = pyrite_sincsq_far_mean(
        a[line], e[line], 0.0f, lf, (float)(lo - lf), hf, (float)(hi - hf),
        exact_widths, &mean) ? (double)mean :
        pyrite_sincsq_bin_mean(a[line], e[line], lo, hi, iw[bin]);
    out[i] = (double)w[line] * value;
}
"""


@cache
def kernels(cp):
    return (
        cp.RawKernel(quad.SINCSQ_BIN_PREAMBLE + NEAR_SOURCE, "sparse_near"),
        cp.RawKernel(far_source(), "pyrite_sparse_far_reduce"),
    )


def sparse_reduce(cp, inputs, *, exact_widths, tile_pairs, line_major=False):
    """Return a complete per-flush candidate, construction and work metrics."""
    sync = cp.cuda.Device().synchronize
    stages = {}

    def stamp(name, start):
        sync()
        stages[name] = time.perf_counter() - start

    started = time.perf_counter()
    host_tree = tree.build_cxr_tree(*(cp.asnumpy(v) for v in inputs[:3]), exact_widths=exact_widths)
    stamp("host_tree_s", started)
    started = time.perf_counter()
    device_tree = tuple(cp.asarray(v) for v in host_tree)
    e, a, w = device_tree[:3]
    edges, inv_width = inputs[3:]
    first, stop = near_intervals(e, a, edges, exact_widths, cp)
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(stop - first)))
    count = int(offsets[-1].item())
    n_bins, n_lines = int(inv_width.size), int(e.size)
    near = cp.zeros(n_bins, dtype=cp.float64)
    stamp("transfer_intervals_scan_s", started)
    near_kernel, far_kernel = kernels(cp)
    mapping = evaluate = reduction = 0.0
    peak_used = cp.get_default_memory_pool().used_bytes()
    for start in range(0, count, tile_pairs):
        started = time.perf_counter()
        keys = pair_keys(
            first, offsets, start, min(start + tile_pairs, count), cp, sort=not line_major
        )
        sync()
        mapping += time.perf_counter() - started
        started = time.perf_counter()
        values = cp.empty(keys.size, dtype=cp.float64)
        near_kernel(
            ((keys.size + 255) // 256,),
            (256,),
            (
                keys,
                np.int64(keys.size),
                np.int64(n_lines),
                e,
                a,
                w,
                edges,
                inv_width,
                values,
                np.float32(exact_widths),
            ),
        )
        sync()
        evaluate += time.perf_counter() - started
        started = time.perf_counter()
        if line_major:
            # Same (bin, line) sum order as the bin-major route; keys are unique.
            order = cp.argsort(keys)
            keys, values = keys[order], values[order]
        bins = keys // n_lines
        begins = cp.concatenate(
            (cp.zeros(1, dtype=cp.int64), cp.flatnonzero(bins[1:] != bins[:-1]) + 1)
        )
        sums = cp.add.reduceat(values, begins)
        # Each segment has one distinct destination; fixed tile launch order.
        near[bins[begins]] += sums
        sync()
        reduction += time.perf_counter() - started
        peak_used = max(peak_used, cp.get_default_memory_pool().used_bytes())
    stages.update(
        mapping_sort_s=mapping, near_evaluation_s=evaluate, segmented_reduction_s=reduction
    )
    started = time.perf_counter()
    far = cp.empty(n_bins, dtype=cp.float64)
    far_kernel(
        ((n_bins + 127) // 128,),
        (128,),
        (
            *device_tree,
            edges,
            inv_width,
            first,
            stop,
            far,
            np.int32(n_bins),
            np.int32(len(host_tree[3])),
            np.int32(len(host_tree[3]) // 2),
            np.float32(exact_widths),
        ),
    )
    result = near + far
    stamp("hierarchical_far_s", started)
    return result, {
        **stages,
        "near_pairs": count,
        "max_tile_pairs": min(count, tile_pairs),
        "live_pool_sample_max_mib": peak_used / 2**20,
        "pool_reserved_mib": cp.get_default_memory_pool().total_bytes() / 2**20,
    }


def replay(cp, artifact, repeats, tile_pairs):
    with np.load(artifact) as data:
        arrays = tuple(cp.asarray(data[k]) for k in ["E_r", "aw", "w", "edges", "inv_width"])
        exact = float(data["exact_widths"])
        out_dtype = str(data["out_dtype"].item())
    records = {"auto": [], "sparse": []}
    outputs = {}
    first_sparse = None
    deterministic = True
    for repeat in range(repeats + 1):
        methods = ["auto", "sparse"] if repeat % 2 == 0 else ["sparse", "auto"]
        for method in methods:
            cp.cuda.Device().synchronize()
            started = time.perf_counter()
            if method == "auto":
                output = cp.zeros(arrays[-1].size, dtype=out_dtype)
                quad.run_bin_mean_reduction_kernel(*arrays, out=output, exact_widths=exact)
                metrics = {}
            else:
                output, metrics = sparse_reduce(
                    cp, arrays, exact_widths=exact, tile_pairs=tile_pairs
                )
                output = output.astype(out_dtype)
            cp.cuda.Device().synchronize()
            wall = time.perf_counter() - started
            host = cp.asnumpy(output)
            records[method].append({"wall_s": wall, **metrics})
            outputs[method] = host
            if method == "sparse":
                if first_sparse is None:
                    first_sparse = host
                else:
                    deterministic &= np.array_equal(first_sparse, host)
    baseline, candidate = outputs["auto"], outputs["sparse"]
    peak = float(np.max(np.abs(baseline), initial=0))
    error = float(np.max(np.abs(candidate - baseline), initial=0))
    widths = cp.asnumpy(1 / arrays[-1])
    mass = float(np.dot(baseline, widths))
    return {
        "artifact": str(artifact),
        "lines": int(arrays[0].size),
        "bins": int(arrays[-1].size),
        "measurements": records,
        "max_error_over_peak": error / peak if peak else None,
        "relative_integral_error": float(np.dot(candidate - baseline, widths) / mass)
        if mass
        else None,
        "finite": bool(np.isfinite(candidate).all()),
        "repeat_bitwise_equal": bool(deterministic),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["population_bin-mean", "adaptive_3", "common_adaptive_12_uniform_12"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tile-pairs", type=int, default=1_000_000)
    args = parser.parse_args()
    if min(args.repeats, args.tile_pairs) < 1:
        parser.error("positive repeats and tile-pairs required")
    from pyrite._backend import xp

    if getattr(xp, "__name__", "") != "cupy":
        parser.error("remote CUDA required")
    pilot = json.loads(args.pilot_json.read_text())
    report = {
        "pilot": str(args.pilot_json),
        "tile_pairs": args.tile_pairs,
        "repeats": args.repeats,
        "workloads": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in pilot["cases"]:
        if case["seed"] != args.seed:
            continue
        for arm in args.arms:
            flushes = case["arms"][arm]["reducer_profiles"][1]["flushes"]
            for index, flush in enumerate(flushes):
                result = replay(
                    xp,
                    Path(flush["forced_routes"]["input_artifact"]),
                    args.repeats,
                    args.tile_pairs,
                )
                result.update(
                    arm=arm, flush=index, seed=case["seed"], energy_keV=case["energy_keV"]
                )
                report["workloads"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)
    if not report["workloads"]:
        parser.error("no matching captured workloads")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
