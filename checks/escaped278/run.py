"""Escaped finite-cone benchmark, resumable per-primary moments (#278)."""

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

COMP = [("W", 0.06322199557658834)]
DTYPE = np.dtype([("event", "<i4"), ("counts", "<f8", (2, 6))])


def save(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False))
    tmp.replace(path)


def rules(edges, order):
    from pyrite.perf.detector_scoring import absorption_edges_keV

    absorption_edges = np.asarray(absorption_edges_keV(COMP)) * 1000
    x, w = np.polynomial.legendre.leggauss(order)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        cuts = np.unique(
            np.r_[
                np.geomspace(lo, hi, math.ceil(math.log(hi / lo) / 0.25) + 1),
                absorption_edges[(absorption_edges > lo) & (absorption_edges < hi)],
            ]
        )
        nodes = np.concatenate(
            [(a + b) / 2 + (b - a) / 2 * x for a, b in zip(cuts[:-1], cuts[1:], strict=False)]
        )
        weights = np.concatenate(
            [(b - a) / 2 * w for a, b in zip(cuts[:-1], cuts[1:], strict=False)]
        )
        out.append((nodes, weights))
    return out


def score(r, edges, brem, angular, energy_order):
    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
    from pyrite.perf.comparison import cone_rule
    from pyrite.perf.detector_scoring import detector_axis

    out = np.zeros((int(r["Ne"]), 2, 6))
    for view, center in enumerate((0.0, 15.0)):
        if angular == 0:
            dirs = [detector_axis(center)]
            omega = [2 * math.pi * (1 - math.cos(math.radians(1)))]
        else:
            dirs, omega = cone_rule(detector_axis(center), math.radians(1), angular, 2 * angular)
        for direction, weight in zip(dirs, omega, strict=False):
            for b, (nodes, q) in enumerate(rules(edges, energy_order)):
                out[:, view, b] += weight * np.asarray(
                    mc_coupled_brem_spectrum(
                        r,
                        nodes,
                        cutoff_eV=1000.0,
                        bremslib_tables=brem,
                        composition=COMP,
                        n_hat=direction,
                        layers=[(0.0, float(r["thickness_ang"]), COMP)],
                        electron_band_weights=q,
                        chunk=1000000,
                    )
                )
    return out


def diff(a, b):
    return max(
        float(np.max(np.abs((a**p).sum(0) - (b**p).sum(0)) / np.maximum((b**p).sum(0), 1e-100)))
        for p in (1, 2)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["pyrite", "geant4", "mu"])
    ap.add_argument("--output", type=Path, default=Path("evidence/escaped278"))
    ap.add_argument("--executable", default="g4escaped-build/g4bench")
    ap.add_argument("--max-hours", type=float, default=3.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--msc", type=int, default=0)
    ap.add_argument("--splitting", type=int, default=1)
    ap.add_argument("--split-plan", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split_plan = json.loads(args.split_plan.read_text()) if args.split_plan else {}
    if args.engine == "mu":
        from pyrite.materials.attenuation import _mu_total_inv_ang

        e = np.geomspace(1e4, 1e8, 20001)
        mu = np.asarray(_mu_total_inv_ang(COMP, e)) * 1e4
        assert np.all(np.isfinite(mu)) and np.all(mu > 0)
        np.savetxt("evidence/escaped278-mu.txt", np.c_[e / 1000, mu])
        return
    started = time.monotonic()
    if args.engine == "pyrite":
        import cupy as cp

        from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
        from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
        from pyrite.montecarlo.transport.continuation import TransportContinuationConfig
        from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
        from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
        from pyrite.xsgen.sbethe.catalog import resolve_composition_table

        cp.get_default_memory_pool().set_limit(fraction=0.7)
        elastic = resolve_layer_tables(COMP)
        stopping = resolve_composition_table("W", COMP)
        brem = load_bremsstrahlung_tables(["W"])

        def transport(energy, thickness, n, offset, seed):
            return simulate_trajectories(
                E0_keV=energy,
                composition=COMP,
                thickness_ang=thickness * 1e4,
                crystal_width_mm=10.0,
                crystal_height_mm=10.0,
                beam_dir=np.array([0.0, 0.0, 1.0]),
                E_cut_keV=10.0,
                energy_model="midpoint",
                max_dE_frac=0.05,
                max_steps=2000000,
                elastic_model="elsepa",
                elastic_tables=[[dict(e.arrays) for e in elastic]],
                stopping_tables=[stopping.arrays()],
                atomic_electron_deflection="kawrakow",
                straggling=False,
                radiative_model="bremslib-soft-hard",
                radiative_cutoff_eV=1000.0,
                bremslib_tables=brem,
                radiative_moments="table",
                transport_core="cuda",
                transport_lut_config=TransportLUTConfig(enabled=False),
                per_electron_config=PerElectronTransportConfig(
                    continuation=TransportContinuationConfig(
                        rows_per_round=1024, device_staging=True
                    )
                ),
                Ne=n,
                seed=seed,
                _electron_start=offset,
            )

    cases = [(e, 100) for e in (3000, 30000, 100000)]
    if args.smoke:
        cases = cases[:1]
    for _sweep in range(100000):
        pending = False
        for index, (energy, thickness) in enumerate(cases):
            if time.monotonic() - started > args.max_hours * 3600:
                return
            path = args.output / f"{energy}-{thickness}.json"
            state = (
                json.loads(path.read_text())
                if path.exists()
                else dict(
                    histories=0,
                    total=np.zeros((2, 6)).tolist(),
                    squares=np.zeros((2, 6)).tolist(),
                    seconds=0.0,
                    transport_seconds=0.0,
                    scoring_seconds=0.0,
                    converged=False,
                    engine=args.engine,
                    centers_deg=[0, 15],
                    half_angle_deg=1,
                    seed_base=41000 + index,
                    material="W",
                    thickness_um=100,
                    g4_msc=args.msc,
                    attenuation=(
                        "EPDL2025 Beer-Lambert"
                        if args.engine == "pyrite"
                        else "native Geant4 photon transport; uncollided target exit"
                    ),
                )
            )
            if state["converged"] or state.get("numerical_failure"):
                continue
            pending = True
            state["g4_splitting"] = split_plan.get(str(energy), args.splitting)
            edges = np.geomspace(1e4, energy * 1000.0, 7)
            t0 = time.perf_counter()
            if args.engine == "pyrite":
                if "angular_order" not in state:
                    warm = transport(energy, thickness, 1, 0, 91000 + index)
                    score(warm, edges, brem, 0, 8)
                    del warm
                    pilot = transport(energy, thickness, 16, 0, 81000 + index)
                    previous = score(pilot, edges, brem, 0, 8)
                    evidence = []
                    chosen = None
                    for order in (2, 4, 8, 16, 32):
                        current = score(pilot, edges, brem, order, 8)
                        error = diff(previous, current)
                        evidence.append(dict(order=order, relative_moment_change=error))
                        if error <= 0.002:
                            chosen = 0 if order == 2 else order // 2
                            break
                        previous = current
                    state["quadrature"] = evidence
                    if chosen is None:
                        state["numerical_failure"] = "angular pilot failed through order 32"
                        save(path, state)
                        continue
                    coarse = score(pilot, edges, brem, chosen, 8)
                    energy_evidence = []
                    for energy_order in (16, 32, 64, 128):
                        fine = score(pilot, edges, brem, chosen, energy_order)
                        error = diff(coarse, fine)
                        energy_evidence.append(
                            dict(order=energy_order, relative_moment_change=error)
                        )
                        if error <= 0.002:
                            break
                        coarse = fine
                    state.update(
                        angular_order=chosen,
                        energy_order=energy_order,
                        energy_quadrature=energy_evidence,
                        energy_relative_moment_change=error,
                        block_histories=min(
                            20000, max(16, int(2000000 / max(1, len(pilot["L_ang"]) / 16)))
                        ),
                    )
                    del pilot
                    if error > 0.002:
                        state["numerical_failure"] = "energy pilot failed 0.2% moment tolerance"
                    save(path, state)
                    print("pilot", path, state, flush=True)
                    if state.get("numerical_failure"):
                        continue
                    t0 = time.perf_counter()
                n = 16 if args.smoke else min(1024, state["block_histories"])
                r = transport(energy, thickness, n, state["histories"], state["seed_base"])
                t1 = time.perf_counter()
                x = score(r, edges, brem, state["angular_order"], state["energy_order"])
                t2 = time.perf_counter()
                state["transport_seconds"] += t1 - t0
                state["scoring_seconds"] += t2 - t1
                del r
                state["gpu_pool_bytes"] = cp.get_default_memory_pool().total_bytes()
            else:
                n = 100 if args.smoke else 250000
                prefix = args.output / f"{energy}-{thickness}-batch{state['histories']}"
                command = [
                    args.executable,
                    "--material",
                    "W",
                    "--energy-kev",
                    str(energy),
                    "--thickness-um",
                    str(thickness),
                    "--histories",
                    str(n),
                    "--warmup",
                    "1000",
                    "--seed",
                    str(state["seed_base"] + state["histories"] // n),
                    "--msc",
                    str(args.msc),
                    "--splitting",
                    str(split_plan.get(str(energy), args.splitting)),
                    "--threads",
                    "16",
                    "--spectral",
                    "1",
                    "--output",
                    str(prefix),
                ]
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                rows = np.concatenate(
                    [
                        np.fromfile(f, dtype=DTYPE)
                        for f in sorted(args.output.glob(prefix.name + ".t*.bin.spectrum"))
                    ]
                )
                if not np.array_equal(np.sort(rows["event"]), np.arange(n)):
                    raise ValueError("incomplete events")
                x = rows["counts"]
                timing = re.search(
                    r"G4BENCH_TIMING init_s=([\d.]+) warmup_s=([\d.]+) measured_s=([\d.]+)",
                    prefix.with_suffix(".log").read_text(),
                )
                if timing is None:
                    raise ValueError("missing Geant4 timing")
                state["measured_runner_seconds"] = state.get(
                    "measured_runner_seconds", 0.0
                ) + float(timing[3])
            if not np.all(np.isfinite(x)) or np.any(x < 0):
                raise ValueError("invalid primary scores")
            total = np.asarray(state["total"]) + x.sum(0)
            squares = np.asarray(state["squares"]) + (x * x).sum(0)
            count = state["histories"] + len(x)
            mean = total / count
            variance = np.maximum(0, (squares - total * total / count) / max(1, count - 1))
            rse = np.divide(
                np.sqrt(variance / count), mean, out=np.full_like(mean, np.inf), where=mean > 0
            )
            state.update(
                histories=count,
                total=total.tolist(),
                squares=squares.tolist(),
                seconds=state["seconds"] + time.perf_counter() - t0,
                edges_keV=(edges / 1000).tolist(),
                mean_per_primary=mean.tolist(),
                rse=[[float(v) if np.isfinite(v) else None for v in row] for row in rse],
                converged=count >= 10000 and bool(np.all(rse <= 0.02)),
            )
            save(path, state)
            print(path, count, state["rse"], state["converged"], flush=True)
        if args.smoke or not pending:
            return


if __name__ == "__main__":
    main()
