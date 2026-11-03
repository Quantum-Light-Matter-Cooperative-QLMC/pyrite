"""Capture a bounded #821 production population and replay identical band inputs.

All output must be gitignored. Example (CUDA allocation, configured catalogs):
uv run python checks/brem_primary_band_benchmark.py capture si-3MeV-1mm \
    --histories 128 --output scratch/issue938/si3
uv run python checks/brem_primary_band_benchmark.py replay scratch/issue938/si3 \
    --output scratch/issue938/si3-replay --repeats 5

Capture reuses the pinned in-tree #821 production setup, aborting after its first
transport call. Replay alternates warmed baseline/candidate scoring on that same
saved population; compilation is excluded. No Geant4 arm is run. Arrays, row
counts, source/table pins, sums/squares, mean/RSE, wall/device time and memory
remain in scratch for subsequent analyses. Node-count sweeps measure this band
only; they do not establish angular/energy quadrature convergence.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def ignored(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "check-ignore", "--quiet", str(path)], check=True)


def metadata_encode(value):
    """Encode nested transport metadata without discarding array/tuple types."""
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [metadata_encode(item) for item in value]}
    if isinstance(value, list):
        return [metadata_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: metadata_encode(item) for key, item in value.items()}
    return value


def metadata_decode(value):
    """Restore arrays and coupled table identity tuples on captured inputs."""
    if isinstance(value, dict):
        if "__array__" in value:
            return np.asarray(value["__array__"], dtype=value["dtype"])
        if "__tuple__" in value:
            return tuple(metadata_decode(item) for item in value["__tuple__"])
        return {key: metadata_decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [metadata_decode(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(metadata_encode(value), indent=2) + "\n")


def source_pin():
    stamp = Path(".pyrite-sync")
    if stamp.exists():
        value = dict(line.split(": ", 1) for line in stamp.read_text().splitlines() if ": " in line)
        if value["code_dirty"] != "False":
            raise RuntimeError("benchmark requires a clean committed sync")
        return value["code_revision"]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def capture(args):
    import cupy as cp

    from pyrite.montecarlo import transport

    filename = Path(__file__).parent / "geant4_performance/pyrite_bench.py"
    spec = importlib.util.spec_from_file_location("benchmark821", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = transport.simulate_trajectories

    class Captured(Exception):
        pass

    def save(**kwargs):
        result = original(**kwargs)
        arrays, metadata = {}, {}
        for name, value in result.items():
            if hasattr(value, "shape"):
                arrays[name] = cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
            else:
                metadata[name] = value
        np.savez(args.output.with_suffix(".npz"), **arrays)
        metadata.update(
            scene=args.scene,
            source_sha=source_pin(),
            seed=kwargs["seed"],
            histories=kwargs["Ne"],
            rows=int(arrays["L_ang"].size),
            capture_sha256=hashlib.sha256(args.output.with_suffix(".npz").read_bytes()).hexdigest(),
        )
        write_json(args.output.with_suffix(".json"), metadata)
        raise Captured

    transport.simulate_trajectories = save
    sys.argv = [
        str(filename),
        args.scene,
        "--histories",
        str(args.histories),
        "--warmup-histories",
        str(args.histories),
        "--seed",
        str(args.seed),
        "--rows-per-round",
        "1024",
        "--device-staging",
        "--output",
        str(args.output),
    ]
    try:
        module.main()
    except Captured:
        pass
    finally:
        transport.simulate_trajectories = original


def moments(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = scores.mean()
    return dict(
        sum=float(scores.sum()),
        sum_squares=float(scores @ scores),
        mean=float(mean),
        rse=float(scores.std(ddof=1) / np.sqrt(scores.size) / mean),
    )


def replay(args):
    import cupy as cp

    from pyrite.montecarlo.spectrum import brem, brem_band_jit
    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    metadata = metadata_decode(json.loads(args.capture.with_suffix(".json").read_text()))
    data_path = args.capture.with_suffix(".npz")
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == metadata["capture_sha256"]
    segments = dict(np.load(data_path))
    for name, value in metadata.items():
        if name not in {"scene", "source_sha", "seed", "histories", "rows", "capture_sha256"}:
            segments[name] = value
    segments["radiative"]["bremslib_tables"] = tuple(
        tuple(item) for item in segments["radiative"]["bremslib_tables"]
    )
    scenes = json.loads((Path(__file__).parent / "geant4_performance/scenes.json").read_text())
    scene = next(s for s in scenes["scenes"] if s["id"] == metadata["scene"])
    element = scene["material"]
    comp = [(element, scenes["materials"][element]["number_density_ang3"])]
    tables = load_bremsstrahlung_tables([element])
    report = dict(
        source_sha=source_pin(),
        capture={
            key: value
            for key, value in metadata.items()
            if key in {"scene", "source_sha", "seed", "histories", "rows", "capture_sha256"}
        },
        cupy=cp.__version__,
        gpu=str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        repeats=args.repeats,
        precision_arms={
            "baseline": "portable scoring",
            "candidate": "FP64 photon partials / FP64 primary sums (reference control)",
            "row32": "FP32 photon partials / FP64 primary sums (production)",
            "primary32": "FP32 photon partials / FP32 primary sums (diagnostic)",
        },
        protocol="one warmup each, alternating arm order; synchronized wall and CUDA events",
        table_pins=segments["radiative"],
        workloads=[],
    )
    scores_to_save = {
        "primary_row_counts": np.bincount(segments["elec_id"], minlength=int(segments["Ne"]))
    }
    pool = cp.get_default_memory_pool()
    pool.set_limit(fraction=0.8)
    arms = ("baseline", "candidate", "row32", "primary32")

    def select(arm):
        brem._USE_JIT_BREM_REDUCTION = arm != "baseline"
        brem_band_jit.ROW_ACCUMULATION_DTYPE = (
            np.float32 if arm in {"row32", "primary32"} else np.float64
        )
        brem._BAND_PRIMARY_DTYPE = np.float32 if arm == "primary32" else np.float64

    from pyrite.perf.comparison import cone_solid_angle
    from pyrite.perf.detector_scoring import absorption_edges_keV, band_panels, band_rule

    production_rules = [
        band_rule(band_panels(lo, min(hi, scene["energy_keV"]), absorption_edges_keV(comp)), 16)
        for lo, hi in ((10.0, 50.0), (50.0, 100.0), (100.0, float("inf")))
    ]
    workloads = [
        (
            f"stress-{n}",
            [
                (
                    np.geomspace(1e4, scene["energy_keV"] * 1e3, n),
                    np.gradient(np.geomspace(1e4, scene["energy_keV"] * 1e3, n)),
                )
            ],
        )
        for n in args.nodes
    ]
    workloads.append(("production-point-order8", production_rules))
    for nodes, rules in workloads:
        grid, weights = rules[0]
        band_scores = {}
        latest_bands = []
        kwargs = dict(
            cutoff_eV=1000,
            composition=comp,
            bremslib_tables=tables,
            n_hat=np.array([1.0, 0.0, 0.0]),
            electron_band_weights=weights,
            layers=[(0.0, segments["thickness_ang"], [(element, 0.0)])],
            chunk=1000000,
        )

        def score(latest_bands=latest_bands, rules=rules, kwargs=kwargs, nodes=nodes):
            latest_bands.clear()
            for energy, quadrature in rules:
                options = dict(kwargs, electron_band_weights=quadrature)
                latest_bands.append(mc_coupled_brem_spectrum(segments, energy, **options))
            if nodes == "production-point-order8":
                solid_angle = cone_solid_angle(np.deg2rad(scenes["cone"]["half_angle_deg"]))
                return np.sum(latest_bands, axis=0, dtype=np.float64) * solid_angle
            return latest_bands[0]

        scores, times, warmup, memory = {}, {arm: [] for arm in arms}, {}, {}
        for arm in arms:
            select(arm)
            start = time.perf_counter()
            scores[arm] = score()
            band_scores[arm] = np.asarray(latest_bands, dtype=np.float64).copy()
            cp.cuda.get_current_stream().synchronize()
            warmup[arm] = time.perf_counter() - start
        for repeat in range(args.repeats):
            for arm in arms if repeat % 2 == 0 else arms[::-1]:
                select(arm)
                pool.free_all_blocks()
                cp.cuda.get_current_stream().synchronize()
                start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
                start = time.perf_counter()
                start_event.record()
                scores[arm] = score()
                band_scores[arm] = np.asarray(latest_bands, dtype=np.float64).copy()
                stop_event.record()
                stop_event.synchronize()
                times[arm].append(
                    dict(
                        wall_s=time.perf_counter() - start,
                        device_s=cp.cuda.get_elapsed_time(start_event, stop_event) / 1000,
                    )
                )
                scores_to_save[f"{arm}_{nodes}_repeat{repeat}"] = scores[arm].copy()
                memory[arm] = dict(
                    pool_reserved_bytes=pool.total_bytes(), pool_used_bytes=pool.used_bytes()
                )
        if args.profile and nodes == "production-point-order8":
            for arm in arms:
                select(arm)
                profile = cProfile.Profile()
                profile.runcall(score)
                profile.dump_stats(str(args.output) + f"-{arm}.prof")
        a, b = scores["baseline"], scores["candidate"]
        reference_moments = moments(a)
        errors = {}
        for arm, values in scores.items():
            diff = np.asarray(values, dtype=np.float64) - a
            relative = np.abs(diff) / np.maximum(np.abs(a), 1e-300)
            stats = moments(values)
            repeated = np.stack(
                [scores_to_save[f"{arm}_{nodes}_repeat{r}"] for r in range(args.repeats)]
            ).astype(np.float64)
            spread = np.ptp(repeated, axis=0)
            errors[arm] = dict(
                repeat_max_absolute=float(spread.max()),
                repeat_max_relative=float((spread / np.maximum(np.abs(a), 1e-300)).max()),
                max_absolute=float(np.abs(diff).max()),
                max_relative=float(relative.max()),
                relative_quantiles=np.quantile(relative, [0.5, 0.95, 0.99]).tolist(),
                moment_relative={
                    key: float(abs(stats[key] - value) / max(abs(value), 1e-300))
                    for key, value in reference_moments.items()
                },
                target_2pct_pass=bool(stats["rse"] <= 0.02),
                target_2pct_matches=bool(
                    (stats["rse"] <= 0.02) == (reference_moments["rse"] <= 0.02)
                ),
            )
        np.testing.assert_allclose(b, a, rtol=128 * np.finfo(np.float32).eps)
        for arm, values in scores.items():
            scores_to_save[f"{arm}_{nodes}"] = values
            scores_to_save[f"{arm}_{nodes}_bands"] = band_scores[arm]
        scores_to_save[f"grid_{nodes}"] = grid
        scores_to_save[f"weights_{nodes}"] = weights
        report["workloads"].append(
            dict(
                nodes=nodes,
                precision_errors=errors,
                band_nodes=[len(rule[0]) for rule in rules],
                band_moments={
                    arm: [moments(band) for band in bands] for arm, bands in band_scores.items()
                },
                band_max_relative={
                    arm: float(
                        np.max(
                            np.abs(values - band_scores["baseline"])
                            / np.maximum(np.abs(band_scores["baseline"]), 1e-300)
                        )
                    )
                    for arm, values in band_scores.items()
                },
                timing=times,
                warmup_s=warmup,
                memory=memory,
                chunk=1000000,
                moments={arm: moments(values) for arm, values in scores.items()},
                max_relative_error=float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300))),
            )
        )
    report["peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    np.savez(args.output.with_suffix(".npz"), **scores_to_save)
    write_json(args.output.with_suffix(".json"), report)
    print(json.dumps(metadata_encode(report)), flush=True)


def stress(args):
    """Isolate original-ID accumulation over long histories with known FP64 sums."""
    import cupy as cp

    from pyrite.montecarlo.spectrum.brem_band_jit import _group_rows

    n = 100003
    ids = np.repeat(np.array([0, 2, 4], dtype=np.int64), n)
    values = np.tile(np.geomspace(1e-12, 1e-3, n).astype(np.float32), 3)
    reference = np.bincount(ids, weights=values.astype(np.float64), minlength=6)
    arrays = dict(primary_ids=ids, partials=values, reference=reference)
    rng = np.random.default_rng(938)
    report = dict(
        source_sha=source_pin(),
        rows_per_primary=n,
        rows=int(ids.size),
        description="synthetic positive scoring partials; numerical stress, not independent physical histories",
        experiments=[],
    )
    for order, indices in (
        ("forward", np.arange(ids.size)),
        ("reverse", np.arange(ids.size)[::-1]),
        ("shuffled", rng.permutation(ids.size)),
    ):
        device_ids = cp.asarray(ids[indices])
        for dtype in (np.float64, np.float32):
            device_values = cp.asarray(values[indices], dtype=dtype)
            output = cp.zeros(6, dtype=dtype)

            def launch(device_values=device_values, device_ids=device_ids, output=output):
                _group_rows(
                    ((ids.size + 255) // 256,),
                    (256,),
                    (device_values, device_ids, output, np.uint32(ids.size)),
                )

            launch()
            cp.cuda.get_current_stream().synchronize()
            times, scores = [], []
            for _repeat in range(5):
                output.fill(0)
                cp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                launch()
                cp.cuda.get_current_stream().synchronize()
                times.append(time.perf_counter() - start)
                scores.append(cp.asnumpy(output).astype(np.float64))
            scores = np.asarray(scores)
            label = f"{order}-{np.dtype(dtype).name}"
            arrays[label] = scores
            report["experiments"].append(
                dict(
                    order=order,
                    dtype=np.dtype(dtype).name,
                    wall_s=times,
                    max_relative=float(
                        (np.abs(scores - reference) / np.maximum(reference, 1e-300)).max()
                    ),
                    repeat_max_relative=float(
                        (np.ptp(scores, axis=0) / np.maximum(reference, 1e-300)).max()
                    ),
                    values_bytes=int(device_values.nbytes + output.nbytes),
                )
            )
            if dtype == np.float64:
                np.testing.assert_allclose(
                    scores, np.broadcast_to(reference, scores.shape), rtol=1e-11
                )
            np.testing.assert_array_equal(scores[:, [1, 3, 5]], 0)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    write_json(args.output.with_suffix(".json"), report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("scene")
    cap.add_argument("--histories", type=int, default=128)
    cap.add_argument("--seed", type=int, default=938)
    rep = sub.add_parser("replay")
    rep.add_argument("capture", type=Path)
    rep.add_argument("--repeats", type=int, default=5)
    rep.add_argument("--profile", action="store_true")
    rep.add_argument("--nodes", nargs="+", type=int, default=[64, 176, 496])
    accumulation = sub.add_parser("stress")
    for child in (cap, rep, accumulation):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ignored(args.output.with_suffix(".json"))
    ignored(args.output.with_suffix(".npz"))
    {"capture": capture, "replay": replay, "stress": stress}[args.mode](args)


if __name__ == "__main__":
    main()
