"""Research-only axial source profile and bound-population comparator (#1154).

Run in an isolated, synced checkout through ``pyrite remote``. CPU jobs pin
to qlmc-queen. Example: ``uv run python checks/axial_source_performance.py
--case B --count 16 --repeats 3 --profile --out runs/B.json``.
No production hooks, RNG changes, admission changes or validation verdicts.
Scaling overrides are research probes outside the admitted numerical settings.
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import hashlib
import json
import os
import platform
import pstats
import resource
import runpy
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pyrite.montecarlo.channeling.axial_fiber_case import axial_fiber_inputs
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES
from pyrite.validation.check_records import current_revision


def inputs(case_key, overrides):
    """Use the production input builder; label all overridden probes explicitly."""
    case = ADMITTED_AXIAL_CASES[f"si-fibre/{case_key}"]
    changes = dict(overrides)
    physical = {
        name: changes.pop(name)
        for name in ("thickness_ang", "axis_uvw", "kinetic_energy_eV", "tilt_deg")
        if name in changes
    }
    if "axis_uvw" in physical:
        physical["axis_uvw"] = tuple(physical["axis_uvw"])
    case = dataclasses.replace(
        case, **physical, numerics=dataclasses.replace(case.numerics, **changes)
    )
    return case, axial_fiber_inputs(case)


def source(case, data, ids, seed):
    from pyrite.materials.attenuation import linear_attenuation_inv_mm
    from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_on_axis_source

    energies = np.linspace(*case.band_eV, 185)
    result = fiber_on_axis_source(
        data.channels,
        data.entry_orders,
        kinetic_energy_eV=case.kinetic_energy_eV,
        hbar_v_eV_ang=data.hbar_v_eV_ang,
        depths_ang=data.depths_ang,
        photon_energy_eV=energies,
        incident_ids=ids,
        seed=seed,
        carrier_inv_ang=data.carrier_inv_ang,
        boundary_eV=data.boundary_eV,
        absorption_inv_ang=linear_attenuation_inv_mm(case.crystal, energies) * 1e-7,
        exit_path_ang=data.depths_ang[-1] - data.depths_ang,
        search_step_ang=data.search_step_ang,
    )
    arrays = [result.spectrum, result.sem, result.correlation.values, result.correlation.trace]
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    events = [dataclasses.asdict(e) for e in result.events]
    digest.update(json.dumps(events, sort_keys=True).encode())
    return {
        "sha256": digest.hexdigest(),
        "spectrum_per_eV_sr": result.spectrum.tolist(),
        "sem_per_eV_sr": result.sem.tolist(),
        "photon_energy_eV": energies.tolist(),
        "integrated_per_sr": float(np.trapezoid(result.spectrum, energies)),
        "events": events,
    }


def fiber_bound(case, data, ids, seed):
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import unravel_fiber_outgoing

    depths = np.linspace(0, case.thickness_ang, 7)
    values = np.zeros((len(ids), len(depths)))
    lookup = {float(z): i for i, z in enumerate(depths)}

    def observe(incident, depth, carrier, energies, state):
        values[incident - ids.start, lookup[depth]] = np.sum(
            np.abs(state[energies < data.boundary_eV]) ** 2
        )

    box = (2 * case.numerics.outgoing_order_max + 1,) * 2
    window = np.zeros(box, complex)
    window[0, 0] = 1
    events = unravel_fiber_outgoing(
        data.channels,
        data.entry_orders,
        hbar_v_eV_ang=data.hbar_v_eV_ang,
        window=window,
        max_depth_ang=case.thickness_ang,
        incident_ids=ids,
        seed=seed,
        carrier_inv_ang=data.carrier_inv_ang,
        search_step_ang=data.search_step_ang,
        observe=observe,
        checkpoints_ang=depths,
    )
    values *= np.array([e.weight for e in events])[:, None]
    return {
        "depths_ang": depths.tolist(),
        "bound_weight": values.mean(0).tolist(),
        "bound_weight_sem": (values.std(0, ddof=1) / np.sqrt(len(ids))).tolist(),
        "bound_exposure_ang": float(np.trapezoid(values.mean(0), depths)),
    }


class GridBound:
    """Adapt the existing full grid-Hamiltonian projector; no truncated levels.

    CPU eigensolves and device transfers are included in comparator wall time.
    This deliberately favors accuracy over observable throughput.
    """

    cuda_compatible = True

    def __init__(self, solver, potential, cells, boundary):
        from pyrite.materials.crystal import HBAR2_2M_EV_ANG2

        projector = runpy.run_path(str(Path(__file__).with_name("axial_mcwf_return_flux.py")))
        self.projector = projector["GridProjector"](
            potential, solver.grid.kx, solver.grid.ky, cells, HBAR2_2M_EV_ANG2 / solver.gamma
        )
        self.boundary = boundary
        self.seconds = 0.0

    def __call__(self, phi, tilt):
        start = time.perf_counter()
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as cp

            value = self.projector.split(
                cp.asnumpy(phi), tuple(float(t) for t in tilt), self.boundary
            )[0]
            out = cp.asarray([value])
        else:
            out = np.array([self.projector.split(phi, tilt, self.boundary)[0]])
        self.seconds += time.perf_counter() - start
        return out


def grid_bound(case, count, seed, cells, points, step, backend):
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
    from pyrite.montecarlo.channeling.states import gaussian

    profile = axial_profile(case.crystal, case.axis_uvw, temperature_K=case.temperature_K)
    geometry = (cells, cells)
    shape = (cells * points,) * 2
    dissipator = axial_dissipator(profile, case.kinetic_energy_eV, geometry, shape)
    potential = profile.on_grid(geometry, shape)
    solver = AxialMCWF(dissipator, case.kinetic_energy_eV, potential, backend=backend)
    observable = GridBound(solver, potential, geometry, profile.extrema()[1])
    depths = np.linspace(0, case.thickness_ang, 7)
    result = solver.run(
        gaussian(waist_ang=20.0),
        depths,
        count,
        seed=seed,
        step_ang=step,
        observable=observable,
    )
    device = platform.processor()
    if backend == "cuda":
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        cp.cuda.get_current_stream().synchronize()
    return {
        "depths_ang": depths.tolist(),
        "bound_weight": result.observables[:, 0].tolist(),
        "bound_exposure_ang": float(np.trapezoid(result.observables[:, 0], depths)),
        "observable_seconds": observable.seconds,
        "jumps": result.jumps,
        "device": device,
    }


def profile_rows(profiler):
    """Exclusive and cumulative times; cumulative rows must not be added."""
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        rows.append(
            dict(
                file=filename.split("/site-packages/")[-1].split("/src/")[-1],
                line=line,
                function=name,
                calls=calls,
                primitive_calls=primitive,
                exclusive_s=own,
                cumulative_s=cumulative,
            )
        )
    return sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[:80]


def worker(args):
    case_key, overrides, mode, start, count, seed, grid = args
    before = time.perf_counter()
    case, data = inputs(case_key, overrides)
    setup = time.perf_counter() - before
    before = time.perf_counter()
    if mode == "source":
        result = source(case, data, range(start, start + count), seed)
    elif mode == "fiber-bound":
        result = fiber_bound(case, data, range(start, start + count), seed)
    else:
        if case_key != "B" or overrides or start:
            raise ValueError("grid comparator requires aligned B, no overrides or ID offset")
        result = grid_bound(case, count, seed, **grid)
    return {
        "setup_s": setup,
        "work_s": time.perf_counter() - before,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "start": start,
        "count": count,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=["A", "B"], default="B")
    parser.add_argument("--mode", choices=["source", "fiber-bound", "grid-bound"], default="source")
    parser.add_argument("--overrides", default="{}", help="JSON numerical overrides; research only")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1154)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--step", type=float, default=5)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()
    if ns.count < 2 or ns.chunk < 2 or ns.count % ns.chunk or ns.repeats < 1 or ns.workers < 1:
        parser.error("count must be a multiple of chunk; both >=2; repeats/workers >=1")
    if ns.mode == "grid-bound" and ns.workers != 1:
        parser.error("grid comparator uses one device process")
    grid = dict(cells=ns.cells, points=ns.points, step=ns.step, backend=ns.backend)
    jobs = [
        (ns.case, json.loads(ns.overrides), ns.mode, i, ns.chunk, ns.seed, grid)
        for i in range(0, ns.count, ns.chunk)
    ]
    if ns.mode == "grid-bound":
        jobs = [(ns.case, {}, ns.mode, 0, ns.count, ns.seed, grid)]
    samples = []
    # A fresh worker pool on every repeat keeps process/setup policy identical.
    for _ in range(ns.repeats):
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=ns.workers) as pool:
            rows = list(pool.map(worker, jobs))
        samples.append({"wall_s": time.perf_counter() - start, "chunks": rows})
    out = {
        "status": "research measurement; no claim/admission changes",
        "revision": current_revision(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "settings": {**vars(ns), "out": str(ns.out)},
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "samples": samples,
        "cache_policy": "fresh input engine per chunk; fresh pool per repeat; first sample cold, later filesystem/import caches warm",
    }
    if ns.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        row = worker(jobs[0])
        profiler.disable()
        out["profile"] = {"chunk": row, "functions": profile_rows(profiler)}
        if ns.mode == "source":
            out["profile"]["bitwise_equal"] = (
                row["result"]["sha256"] == samples[0]["chunks"][0]["result"]["sha256"]
            )
            if not out["profile"]["bitwise_equal"]:
                raise RuntimeError("profiling changed source output")
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
