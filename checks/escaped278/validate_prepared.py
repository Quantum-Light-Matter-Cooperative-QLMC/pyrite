"""Saved-track CPU/CUDA parity and warmed reversed-order single-detector timings."""

import argparse
import json
import resource
import time
from functools import partial
from pathlib import Path

import numpy as np

from checks.brem_primary_band_benchmark import metadata_decode, source_pin, write_json
from checks.escaped278.run import COMP, rules
from checks.escaped278.workload import angular, load_tables, recipe, transport


def reference(segments, tables, energy, center, order):
    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum

    out = np.zeros((int(segments["Ne"]), 6))
    bands = rules(np.geomspace(1e4, energy * 1000.0, 7), 16)
    directions, weights = angular(center, order)
    for direction, omega in zip(directions, weights, strict=True):
        for band, (grid, q) in enumerate(bands):
            out[:, band] += omega * mc_coupled_brem_spectrum(
                segments,
                grid,
                cutoff_eV=1000.0,
                composition=COMP,
                bremslib_tables=tables[2],
                n_hat=direction,
                layers=[(0.0, 1e6, COMP)],
                electron_band_weights=q,
            )
    return out


def capture(path, segments):
    from pyrite._backend import _to_cpu

    arrays, metadata = {}, {}
    for name, value in segments.items():
        if hasattr(value, "shape"):
            arrays[name] = _to_cpu(value)
        else:
            metadata[name] = value
    np.savez(path.with_suffix(".npz"), **arrays)
    write_json(path.with_suffix(".json"), metadata)


def load(path):
    segments = metadata_decode(json.loads(path.with_suffix(".json").read_text()))
    with np.load(path.with_suffix(".npz")) as arrays:
        segments.update(dict(arrays))
    return segments


def compare(a, b):
    # FP32 cell arithmetic and independent positive reduction orders. Avoid a
    # max relative error at zero; require zero scores to stay exactly zero.
    np.testing.assert_allclose(a, b, rtol=128 * np.finfo(np.float32).eps, atol=0)
    np.testing.assert_allclose(
        (a * a).sum(0), (b * b).sum(0), rtol=256 * np.finfo(np.float32).eps, atol=0
    )
    return dict(
        max_relative_primary_difference=float(
            np.max(np.divide(np.abs(a - b), np.abs(b), out=np.zeros_like(b), where=b != 0))
        )
    )


def prepared_score(scorer, segments, directions, weights):
    return scorer.bind(segments).integrate(directions, weights)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--histories", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--reuse-tracks", action="store_true", help="Replay existing saved populations")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

        tables = None, None, load_bremsstrahlung_tables(["W"])
    else:
        import cupy as cp

        cp.get_default_memory_pool().set_limit(fraction=0.7)
        tables = load_tables()
        warm = transport(tables, 3000, 1, seed=980278)
        recipe(tables, 3000, 16).bind(warm).integrate(*angular(0, 2))
        del warm
    from pyrite._bremslib_table import BREMSSTRAHLUNG_BREMSLIB_MODEL

    report = dict(
        source_sha=source_pin(),
        model=BREMSSTRAHLUNG_BREMSLIB_MODEL,
        backend="cpu" if args.cpu else "cuda",
        complete=False,
        reused_tracks=args.cpu or args.reuse_tracks,
        cases=[],
    )
    for energy, order in ((3000, 2), (30000, 4), (100000, 16)):
        for center in (0, 15):
            # CPU checks the same real tracks/nodes at the central direction;
            # full-cone CUDA parity below covers every production angular node.
            selected_order = 0 if args.cpu else order
            path = args.output / f"tracks-{energy}-{center}"
            if args.cpu or args.reuse_tracks:
                segments = load(path)
            else:
                segments = transport(tables, energy, args.histories, seed=971278 + energy + center)
                capture(path, segments)
            first = time.perf_counter()
            scorer = recipe(tables, energy, 16)
            prepared_seconds = time.perf_counter() - first
            directions, weights = angular(center, selected_order)
            operations = dict(
                reference=partial(reference, segments, tables, energy, center, selected_order),
                prepared=partial(prepared_score, scorer, segments, directions, weights),
            )
            baseline, candidate = operations["reference"](), operations["prepared"]()
            parity = compare(candidate, baseline)
            if args.cpu:
                cuda = np.load(args.output / f"scores-{energy}-{center}.npz")
                parity["cpu_cuda_central_direction"] = compare(candidate, cuda["central_prepared"])
                repeats = 0
            else:
                np.savez(
                    args.output / f"scores-{energy}-{center}.npz",
                    reference=baseline,
                    prepared=candidate,
                    central_prepared=scorer.bind(segments).integrate(*angular(center, 0)),
                )
                repeats = args.repeats
            times = dict(reference=[], prepared=[])
            for repeat in range(repeats):
                for name in (
                    ("reference", "prepared") if repeat % 2 == 0 else ("prepared", "reference")
                ):
                    started = time.perf_counter()
                    value = operations[name]()
                    times[name].append(time.perf_counter() - started)
                    compare(value, baseline)
            item = dict(
                energy_keV=energy,
                center_deg=center,
                angular_order=selected_order,
                histories=int(segments["Ne"]),
                rows=len(segments["L_ang"]),
                parity=parity,
                static_preparation_seconds=prepared_seconds,
                seconds=times,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            if not args.cpu:
                item["gpu_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
                item["median_speedup"] = float(
                    np.median(times["reference"]) / np.median(times["prepared"])
                )
            report["cases"].append(item)
            write_json(args.output / ("cpu.json" if args.cpu else "cuda.json"), report)
            print(json.dumps(item), flush=True)
            del segments, scorer, operations
    report["complete"] = True
    write_json(args.output / ("cpu.json" if args.cpu else "cuda.json"), report)


if __name__ == "__main__":
    main()
