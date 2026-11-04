"""One energy/detector per process, independent pilots and primary streams.

Run from the checkout: uv run python -m checks.escaped278.prepared --energy 3000
--center 0 --output evidence/prepared278. Warmup/pilots are outside production
timers but included in the per-process wall budget and reported separately.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from checks.brem_primary_band_benchmark import source_pin
from checks.escaped278.run import diff, save
from checks.escaped278.workload import angular, load_tables, recipe, statistics, transport


def calibrate(tables, args):
    """Select per-detector quadrature; verify on a larger independent sample."""
    started = time.perf_counter()
    evidence = []
    r = transport(tables, args.energy, 16, seed=args.seed + 100000)
    order, energy_order = None, 8
    coarse = recipe(tables, args.energy, energy_order).bind(r).integrate(*angular(args.center, 0))
    for fine_order in (2, 4, 8, 16, 32):
        fine = (
            recipe(tables, args.energy, energy_order)
            .bind(r)
            .integrate(*angular(args.center, fine_order))
        )
        error = diff(coarse, fine)
        evidence.append(dict(stage="angular-pilot", order=fine_order, change=error))
        if error <= 0.002:
            order = 0 if fine_order == 2 else fine_order // 2
            break
        coarse = fine
    if order is None:
        raise ValueError("angular pilot failed through order 32")
    coarse = (
        recipe(tables, args.energy, energy_order).bind(r).integrate(*angular(args.center, order))
    )
    for energy_order in (16, 32, 64, 128):
        fine = (
            recipe(tables, args.energy, energy_order)
            .bind(r)
            .integrate(*angular(args.center, order))
        )
        error = diff(coarse, fine)
        evidence.append(dict(stage="energy-pilot", order=energy_order, change=error))
        if error <= 0.002:
            break
        coarse = fine
    if error > 0.002:
        raise ValueError("energy pilot failed through order 128")
    del r
    r = transport(tables, args.energy, args.validation_histories, seed=args.seed + 200000)
    while True:
        coarse = (
            recipe(tables, args.energy, energy_order)
            .bind(r)
            .integrate(*angular(args.center, order))
        )
        fine = (
            recipe(tables, args.energy, energy_order)
            .bind(r)
            .integrate(*angular(args.center, max(2, 2 * order)))
        )
        error = diff(coarse, fine)
        evidence.append(dict(stage="heldout-angular", order=order, change=error))
        if error <= 0.002:
            break
        order = max(2, 2 * order)
        if order > 16:
            raise ValueError("held-out angular check failed at order 16 versus 32")
    while True:
        coarse = (
            recipe(tables, args.energy, energy_order)
            .bind(r)
            .integrate(*angular(args.center, order))
        )
        fine = (
            recipe(tables, args.energy, energy_order * 2)
            .bind(r)
            .integrate(*angular(args.center, order))
        )
        error = diff(coarse, fine)
        evidence.append(dict(stage="heldout-energy", order=energy_order, change=error))
        if error <= 0.002:
            break
        energy_order *= 2
        if energy_order > 64:
            raise ValueError("held-out energy check failed at order 64 versus 128")
    block = min(1024, max(16, int(2000000 / max(1, len(r["L_ang"]) / args.validation_histories))))
    return dict(
        angular_order=order,
        energy_order=energy_order,
        block_histories=block,
        numerical_evidence=evidence,
        calibration_seconds=time.perf_counter() - started,
    )


def run(args):
    import cupy as cp

    cp.get_default_memory_pool().set_limit(fraction=0.7)
    started = time.perf_counter()
    tables = load_tables()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.energy}-{args.center}.json"
    identity = dict(
        source_sha=source_pin(),
        energy_keV=args.energy,
        center_deg=args.center,
        half_angle_deg=1,
        seed=args.seed,
        material="W",
        thickness_um=100,
        independent_detector_run=True,
        validation_histories=args.validation_histories,
        tables={el: t.digest for el, t in tables[2].items()},
    )
    state = (
        json.loads(path.read_text())
        if path.exists()
        else dict(
            identity=identity,
            histories=0,
            total=[0.0] * 6,
            squares=[0.0] * 6,
            transport_seconds=0.0,
            scoring_seconds=0.0,
            preparation_seconds=0.0,
            production_seconds=0.0,
            invocation_wall_seconds=0.0,
            converged=False,
        )
    )
    if state["identity"] != identity:
        raise ValueError("resume identity differs; use a fresh output directory")
    if state["converged"]:
        return
    t0 = time.perf_counter()
    warm = transport(tables, args.energy, 1, seed=args.seed + 300000)
    recipe(tables, args.energy, 8).bind(warm).integrate(*angular(args.center, 0))
    del warm
    state["warmup_seconds"] = time.perf_counter() - t0
    if "angular_order" not in state:
        try:
            state.update(calibrate(tables, args))
        except ValueError as exc:
            state.update(status="numerical_failure", numerical_failure=str(exc))
            save(path, state)
            raise
    t0 = time.perf_counter()
    scorer = recipe(tables, args.energy, state["energy_order"])
    directions, weights = angular(args.center, state["angular_order"])
    state["preparation_seconds"] += time.perf_counter() - t0
    while time.perf_counter() - started < args.max_minutes * 60:
        first = time.perf_counter()
        r = transport(
            tables, args.energy, state["block_histories"], seed=args.seed, offset=state["histories"]
        )
        t1 = time.perf_counter()
        bound = scorer.bind(r)
        t2 = time.perf_counter()
        scores = bound.integrate(directions, weights)
        t3 = time.perf_counter()
        state["transport_seconds"] += t1 - first
        state["preparation_seconds"] += t2 - t1
        state["scoring_seconds"] += t3 - t2
        state["production_seconds"] += t3 - first
        state["histories"] += len(scores)
        total = np.asarray(state["total"]) + scores.sum(0)
        squares = np.asarray(state["squares"]) + (scores * scores).sum(0)
        state.update(
            total=total.tolist(),
            squares=squares.tolist(),
            **statistics(total, squares, state["histories"]),
        )
        state.update(
            status="converged" if state["converged"] else "running",
            gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            current_invocation_wall_seconds=time.perf_counter() - started,
        )
        save(path, state)
        print(path, state["histories"], state["rse"], state["status"], flush=True)
        del r, bound
        if state["converged"]:
            break
    state["invocation_wall_seconds"] += time.perf_counter() - started
    state["status"] = "converged" if state["converged"] else "budget_exhausted"
    state["summary_sha256"] = hashlib.sha256(np.asarray(state["total"]).tobytes()).hexdigest()
    save(path, state)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--energy", type=int, choices=(3000, 30000, 100000), required=True)
    ap.add_argument("--center", type=int, choices=(0, 15), required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--validation-histories", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.seed is None:
        args.seed = 141000 + (3000, 30000, 100000).index(args.energy) * 100 + args.center
    run(args)


if __name__ == "__main__":
    main()
