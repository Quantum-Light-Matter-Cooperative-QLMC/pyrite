"""Capture and profile MeV incoherent bin-mean line reductions (#1003).

``capture`` transports one production case on a frozen line grid and saves
every incoherent bin-mean line row ``(E_r, aw, w)`` instead of reducing it. On
CPU it records the rows the dense lineshape would have reduced; on CUDA it
records each ``_CxrBatch`` flush. ``stats`` reports the per-bin near-line
counts that drive the reducer's pair/tree choice and its cost model.
``replay`` times every reducer route on the captured flushes on a CUDA device
and compares them against the current production result. ``grid`` freezes the
line grid a case's transport builds; ``wall`` times warm production runs on a
frozen grid and saves their spectra, and ``compare`` diffs two ``wall``
outputs, so the same script run in two checkouts gives a before/after pair.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _case(args, grid):
    from pyrite._energy_grid_encoding import encode_energy_grid
    from pyrite.energy_grid.bandwidth_check import build_case

    built = build_case(
        args.material, args.energy, _config(args), seed=args.seed, resolution="local"
    )
    case = {key: value for key, value in built.items() if key != "line_grid_policy"}
    encoded = encode_energy_grid(grid)
    case.update(E_grid=encoded, E_grid_line=encoded, Ne=args.electrons, Ne_brem=16)
    case["line_quadrature"] = "bin-mean"
    return case


class _Recorded:
    """Stand-in lineshape: ``w @ S`` records the weights and adds nothing."""

    __array_ufunc__ = None

    def __init__(self, sink, aw, e_r, n_bins):
        self.sink, self.aw, self.e_r, self.n_bins = sink, aw, e_r, n_bins

    def __rmatmul__(self, w):
        from pyrite._backend import _to_cpu

        self.sink.append(
            (
                _to_cpu(self.e_r).astype(np.float32),
                _to_cpu(self.aw).astype(np.float32),
                _to_cpu(w).astype(np.float32),
            )
        )
        return 0.0


def capture(args):
    from pyrite.montecarlo.runner import run_case
    from pyrite.montecarlo.spectrum.cxr import _batched, _per_hkl

    grid = np.load(args.grid)
    case = _case(args, grid)
    rows = []

    def lineshape(aw, e_r, edges, inv_width, **_):
        return _Recorded(rows, aw, e_r, int(inv_width.shape[0]))

    def flush(batch):
        import cupy as cp

        if batch.n_cxr_rows:
            rows.append(
                tuple(
                    cp.asnumpy(cp.concatenate(getattr(batch, k)).astype(cp.float32))
                    for k in ("E_r", "aw", "w")
                )
            )
        batch.E_r.clear(), batch.aw.clear(), batch.w.clear()
        batch.n_cxr_rows = 0

    saved = (_batched.sincsq_bin_lineshape, _per_hkl.sincsq_bin_lineshape, _batched._CxrBatch.flush)
    _batched.sincsq_bin_lineshape = _per_hkl.sincsq_bin_lineshape = lineshape
    _batched._CxrBatch.flush = flush
    start = time.perf_counter()
    try:
        run_case(case, transport_core=args.core)
    finally:
        _batched.sincsq_bin_lineshape, _per_hkl.sincsq_bin_lineshape, _batched._CxrBatch.flush = (
            saved
        )
    wall = time.perf_counter() - start
    e_r, aw, w = (np.concatenate([r[k] for r in rows]) for k in range(3))
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_axis

    edges, inv_width = (_to_cpu(v) for v in bin_axis(grid))
    np.savez(args.out, E_r=e_r, aw=aw, w=w, edges=edges, inv_width=inv_width, grid=grid)
    meta = {
        "case": {k: v for k, v in case.items() if k not in ("E_grid", "E_grid_line")},
        "grid_sha256": hashlib.sha256(Path(args.grid).read_bytes()).hexdigest(),
        "rows": int(e_r.size),
        "bins": int(inv_width.size),
        "core": args.core,
        "wall_s": wall,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("rows", "bins", "wall_s")}))


def _flushes(n_rows, target):
    return [slice(s, min(s + target, n_rows)) for s in range(0, n_rows, target)]


def stats(args):
    from pyrite.montecarlo.spectrum.cxr import _bin_tree
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import BIN_MEAN_EXACT_WIDTHS
    from pyrite.montecarlo.spectrum.cxr._policy import _JIT_CXR_BATCH_TARGET

    data = np.load(args.fixture)
    e, a, w, edges = data["E_r"], data["aw"], data["w"], data["edges"]
    keep = w != 0
    print(f"rows {e.size}, nonzero-weight {int(keep.sum())}, bins {edges.size - 1}")
    widths = np.diff(edges)
    radius = BIN_MEAN_EXACT_WIDTHS * np.pi / a.astype(np.float64)
    q = [0, 0.05, 0.5, 0.95, 1]
    print("near radius eV quantiles", np.quantile(radius, q))
    print("bin width eV quantiles", np.quantile(widths, q))
    print("E_r eV quantiles", np.quantile(e, q))
    for sl in _flushes(e.size, _JIT_CXR_BATCH_TARGET):
        counts = _bin_tree.near_line_counts(e[sl], a[sl], edges, BIN_MEAN_EXACT_WIDTHS)
        tree_s = _bin_tree._TREE_SECONDS_PER_BUSIEST_LINE * counts.max()
        pairs_s = _bin_tree._PAIRS_SECONDS_PER_NEAR_PAIR * counts.sum()
        n = sl.stop - sl.start
        print(
            f"flush {sl.start}:{sl.stop} lines {n} pairs {n * (edges.size - 1):.3g} "
            f"near pairs {counts.sum():.3g} busiest {counts.max()} "
            f"bins>0 {(counts > 0).sum()} model tree {tree_s:.2f}s pairs {pairs_s:.2f}s"
        )


def _sync_time(fn):
    import cupy as cp

    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    fn()
    cp.cuda.Device().synchronize()
    return time.perf_counter() - start


_SERIES_DEFINE = "#define PYRITE_BIN_MEAN_SERIES {}\n"


def _set_series(on):
    """Rebuild every bin-mean kernel with the midpoint series on or off."""
    from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as quad
    from pyrite.montecarlo.spectrum.cxr import _bin_sparse, _bin_tree

    base = quad.SINCSQ_BIN_PREAMBLE
    for flag in (0, 1):
        base = base.removeprefix(_SERIES_DEFINE.format(flag))
    preamble = _SERIES_DEFINE.format(int(on)) + base
    quad.SINCSQ_BIN_PREAMBLE = _bin_tree.SINCSQ_BIN_PREAMBLE = preamble
    _bin_sparse.SINCSQ_BIN_PREAMBLE = preamble
    _bin_sparse._NEAR_KERNEL = _bin_sparse._FAR_KERNEL = None
    quad._BIN_REDUCE_KERNEL = quad._BIN_MEAN_KERNEL = None
    _bin_tree._TREE_KERNEL = None
    _sparse_module().kernels.cache_clear()


def _sparse_module():
    import importlib.util
    import sys

    name = "line_sparse_near_286"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("line_sparse_near_286.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _fused_stages(inputs, out):
    """The fused route with a synchronized clock between its stages."""
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr import _bin_sparse as fused
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import BIN_MEAN_EXACT_WIDTHS as widths

    e, a, w, edges, inv_width = inputs
    stages = {}

    def stage(name, work):
        start = time.perf_counter()
        result = work()
        cp.cuda.Device().synchronize()
        stages[name] = time.perf_counter() - start
        return result

    tree, first, stop, offsets, n_jobs, scale = stage(
        "host_setup_s", lambda: fused.prepare_sparse_reduction(e, a, w, edges)
    )
    near = stage(
        "near_s",
        lambda: fused.run_sparse_near(
            tree, first, stop, offsets, n_jobs, scale, edges, inv_width, widths
        ),
    )
    far = stage("far_s", lambda: fused.run_sparse_far(tree, first, stop, edges, inv_width, widths))
    out += near + far
    print("fused-stages", json.dumps({**stages, "warp_jobs": n_jobs}), flush=True)


def _arm(name, inputs, tile_pairs, nodes=None):
    """``run(out)`` adding the arm's reduction of ``inputs`` into ``out``."""
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import (
        BIN_MEAN_EXACT_WIDTHS,
        run_bin_mean_reduction_kernel,
    )
    from pyrite.montecarlo.spectrum.cxr._bin_tree import run_tree_reduction

    kind = name.removesuffix("+series")
    if kind == "node":
        # Point sampling, a different quadrature: reported for #944, not a
        # same-accuracy candidate for #1003.
        from pyrite.montecarlo.spectrum.cxr_jit_kernel import (
            DEFAULT_SPECTRUM_KERNEL_CONFIG,
            run_reduction_kernel,
        )

        def run_node(out):
            sampled = cp.zeros(out.size, dtype=cp.float32)
            run_reduction_kernel(
                *inputs[:3], nodes, out=sampled, config=DEFAULT_SPECTRUM_KERNEL_CONFIG
            )
            out += sampled

        return run_node
    if kind in ("auto", "pairs"):
        return lambda out: run_bin_mean_reduction_kernel(*inputs, out=out, method=kind)
    if kind == "tree":
        return lambda out: run_tree_reduction(*inputs, out=out)
    if kind == "fused":
        from pyrite.montecarlo.spectrum.cxr._bin_sparse import run_sparse_reduction

        return lambda out: run_sparse_reduction(*inputs, out=out)
    if kind == "fused-stages":
        return lambda out: _fused_stages(inputs, out)
    if kind in ("sparse", "sparse-line"):
        sparse = _sparse_module()

        def run(out):
            result, stages = sparse.sparse_reduce(
                cp,
                inputs,
                exact_widths=BIN_MEAN_EXACT_WIDTHS,
                tile_pairs=tile_pairs,
                line_major=kind == "sparse-line",
            )
            out += result
            print(name, json.dumps(stages), flush=True)

        return run
    raise ValueError(f"unknown arm {name!r}")


def _accuracy(candidate, ref, widths):
    peak = float(np.max(np.abs(ref))) or 1.0
    diff = np.abs(candidate - ref)
    floor = np.maximum(np.abs(ref), 1e-6 * peak)
    return {
        "max_dev_over_peak": float(diff.max() / peak),
        "max_rel_dev_floor1e-6peak": float(np.max(diff / floor)),
        "integral_rel_dev": float(np.dot(candidate - ref, widths) / np.dot(ref, widths)),
        "l1_rel_dev": float(np.dot(diff, widths) / np.dot(np.abs(ref), widths)),
        "bin_rel_dev_q50_q95_q99": [float(v) for v in np.quantile(diff / floor, [0.5, 0.95, 0.99])],
        "finite": bool(np.isfinite(candidate).all()),
    }


def replay(args):
    """Time every arm per flush; compare FP64 per-bin sums with the first arm.

    Arms rotate order each repeat after one untimed warmup (JIT). The first
    arm is the reference: the series-free ``tree``, the route before #1003.
    """
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr._policy import _JIT_CXR_BATCH_TARGET

    data = np.load(args.fixture)
    edges = cp.asarray(data["edges"])
    inv_width = cp.asarray(data["inv_width"])
    widths = 1.0 / data["inv_width"]
    e, a, w = (cp.asarray(data[k]) for k in ("E_r", "aw", "w"))
    # Captures without the grid (the #286 flushes) sample at bin centres.
    has_grid = "grid" in data.files
    nodes = data["grid"] if has_grid else 0.5 * (data["edges"][:-1] + data["edges"][1:])
    nodes = cp.asarray(nodes, dtype=cp.float32)
    arms = args.arms.split(",")
    report = {
        "fixture": str(args.fixture),
        "arms": arms,
        "node_coordinates": "grid" if has_grid else "bin centres",
        "flushes": [],
    }
    for index, sl in enumerate(_flushes(int(e.size), _JIT_CXR_BATCH_TARGET)):
        if args.max_flushes and index >= args.max_flushes:
            break
        if args.slice_lines:
            sl = slice(sl.start, min(sl.stop, sl.start + args.slice_lines))
        inputs = (e[sl], a[sl], w[sl], edges, inv_width)
        row = {"lines": sl.stop - sl.start, "arms": {}}
        outs = {}
        for repeat in range(args.repeats + 1):
            order = arms if repeat % 2 == 0 else arms[::-1]
            for name in order:
                _set_series(name.endswith("+series"))
                run = _arm(name, inputs, args.tile_pairs, nodes)
                out = cp.zeros(edges.size - 1, dtype=cp.float64)
                elapsed = _sync_time(lambda run=run, out=out: run(out))
                host = cp.asnumpy(out)
                entry = row["arms"].setdefault(name, {"s": [], "repeat_bitwise_equal": True})
                if repeat:
                    entry["s"].append(elapsed)
                if name in outs:
                    entry["repeat_bitwise_equal"] &= bool(np.array_equal(outs[name], host))
                outs[name] = host
                print(index, repeat, name, f"{elapsed:.3f}s", flush=True)
        for name in arms[1:]:
            row["arms"][name].update(_accuracy(outs[name], outs[arms[0]], widths))
        for name in arms:
            row["arms"][name]["median_s"] = float(np.median(row["arms"][name]["s"]))
        report["flushes"].append(row)
        args.out.write_text(json.dumps(report, indent=2))
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["flushes"], indent=1))


def grid(args):
    """Save the line grid this case's transport builds at ``--electrons``."""
    from pyrite._backend import _to_cpu
    from pyrite.energy_grid.bandwidth_check import build_case
    from pyrite.montecarlo import runner

    built = build_case(
        args.material, args.energy, _config(args), seed=args.seed, resolution="local"
    )
    transport = runner._transport_case(built, transport_core=args.core)
    E_grid = np.asarray(_to_cpu(transport["E_grid"]), dtype=float)
    np.save(args.out, E_grid)
    print(json.dumps({"bins": int(E_grid.size), "sha256": _sha(args.out)}))


def _config(args):
    return dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=args.electrons,
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays(result, prefix=""):
    """Numeric arrays of a run result, flattened one ``/`` level per dict."""
    from pyrite._backend import _to_cpu

    out = {}
    for key, value in result.items():
        if isinstance(value, dict):
            out.update(_arrays(value, f"{prefix}{key}/"))
            continue
        try:
            array = np.asarray(_to_cpu(value))
        except Exception:
            continue
        if array.ndim and array.dtype.kind in "fc":
            out[f"{prefix}{key}"] = array
    return out


def wall(args):
    """Warm wall per primary of ``--repeats`` runs after one warmup run."""
    from pyrite.montecarlo.runner import run_case

    case = _case(args, np.load(args.grid))
    run_case(dict(case, Ne=args.warmup, Ne_brem=args.warmup), transport_core=args.core)
    times, result = [], None
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run_case(case, transport_core=args.core, record_timing=True)
        times.append(time.perf_counter() - start)
    arrays = _arrays(result)
    np.savez(args.out, **arrays)
    meta = {
        "material": args.material,
        "energy_keV": args.energy,
        "electrons": args.electrons,
        "core": args.core,
        "grid_sha256": _sha(args.grid),
        "wall_s": times,
        "wall_per_primary_s": float(np.median(times)) / args.electrons,
        "scalars": {k: v for k, v in result.items() if np.isscalar(v)},
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("wall_s", "wall_per_primary_s")}))


def compare(args):
    """Per-array agreement of ``candidate`` with ``reference`` (``wall`` outputs)."""
    ref, cand = np.load(args.reference), np.load(args.candidate)
    widths = None
    if args.grid is not None:
        from pyrite._backend import _to_cpu
        from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_axis

        widths = 1.0 / _to_cpu(bin_axis(np.load(args.grid))[1])
    report = {}
    for key in sorted(set(ref.files) & set(cand.files)):
        r, c = ref[key].astype(float), cand[key].astype(float)
        if r.shape != c.shape or r.ndim != 1:
            continue
        w = widths if widths is not None and widths.size == r.size else np.ones_like(r)
        report[key] = {"bitwise_equal": bool(np.array_equal(r, c))}
        if np.any(r):
            report[key].update(_accuracy(c, r, w))
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


def _case_args(p):
    p.add_argument("--material", default="hbn")
    p.add_argument("--energy", type=float, default=5000.0)
    p.add_argument("--thickness", type=float, default=1e7)
    p.add_argument("--tilt", type=float, default=10.0)
    p.add_argument("--azimuth", type=float, default=100.0)
    p.add_argument("--electrons", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--core", default="auto")


parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest="mode", required=True)
cap = sub.add_parser("capture")
cap.add_argument("--grid", type=Path, required=True)
cap.add_argument("--out", type=Path, required=True)
_case_args(cap)
st = sub.add_parser("stats")
st.add_argument("fixture", type=Path)
rep = sub.add_parser("replay")
rep.add_argument("fixture", type=Path)
rep.add_argument("--out", type=Path, required=True)
rep.add_argument(
    "--arms",
    default="auto,pairs,tree,sparse,auto+series,pairs+series,tree+series,sparse+series",
)
rep.add_argument("--repeats", type=int, default=2)
rep.add_argument("--tile-pairs", type=int, default=50_000_000)
rep.add_argument("--slice-lines", type=int, default=0, help="first N lines per flush")
rep.add_argument("--max-flushes", type=int, default=0)
gr = sub.add_parser("grid")
gr.add_argument("--out", type=Path, required=True)
_case_args(gr)
wl = sub.add_parser("wall")
wl.add_argument("--grid", type=Path, required=True)
wl.add_argument("--out", type=Path, required=True)
wl.add_argument("--warmup", type=int, default=8)
wl.add_argument("--repeats", type=int, default=2)
_case_args(wl)
cm = sub.add_parser("compare")
cm.add_argument("reference", type=Path)
cm.add_argument("candidate", type=Path)
cm.add_argument("--grid", type=Path)
cm.add_argument("--out", type=Path)
args = parser.parse_args()
modes = {"capture": capture, "stats": stats, "replay": replay, "grid": grid, "wall": wall}
modes["compare"] = compare
modes[args.mode](args)
