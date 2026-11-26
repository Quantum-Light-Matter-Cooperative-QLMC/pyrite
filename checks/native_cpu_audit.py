"""Bounded CPU-route audit (#1432); diagnostic only, no production overrides.

Run from a clean synced checkout in a remote allocation; see
checks/native_cpu_audit/README.md. Reuses production runners and JIT recorders.
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import resource
import runpy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "thin30",
    "thin100",
    "crystal300",
    "mos2_5mev",
    "layered",
    "photon",
    "pair",
    "axial",
    "axial-prod",
    "axial-prod-cache8",
    "eedl",
    "bh",
    "char-multi",
    "char-layered",
    "photon-case",
    "crystal300-coherent",
    "crystal300-coherent-hr",
    "crystal300-binmean",
    "transport-si300",
    "channeling-demo",
)


def digest(array):
    a = np.ascontiguousarray(array)
    return {
        "shape": list(a.shape),
        "dtype": a.dtype.str,
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def environment():
    import numba
    from threadpoolctl import threadpool_info

    from pyrite._backend import BACKEND, REAL
    from pyrite.validation.check_records import current_revision

    try:
        threading_layer = numba.threading_layer()
    except ValueError:
        threading_layer = None

    versions = {}
    for package in ("numpy", "numba", "scipy", "llvmlite", "dask", "distributed", "pyfftw"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "revision": current_revision(),
        "python": platform.python_version(),
        "versions": versions,
        "machine": platform.machine(),
        "cpu": next(
            (
                s.split(":", 1)[1].strip()
                for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")
            ),
            None,
        ),
        "affinity": sorted(os.sched_getaffinity(0)),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "numba_threads": numba.get_num_threads(),
        "numba_threading_layer": threading_layer,
        "backend": BACKEND,
        "precision": str(np.dtype(REAL)),
        "native_pools": threadpool_info(),
        "threads": {
            k: os.environ.get(k)
            for k in (
                "NUMBA_NUM_THREADS",
                "NUMBA_THREADING_LAYER",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYRITE_FFT_THREADS",
            )
        },
        "cache": {k: os.environ.get(k) for k in ("NUMBA_CACHE_DIR", "CUPY_CACHE_DIR")},
        "numpy_config": np.show_config(mode="dicts"),
        "allocation": {
            k: os.environ.get(k)
            for k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")
        },
    }


def dispatchers():
    from numba.core.dispatcher import Dispatcher

    from pyrite._jit_cache import numba_dispatcher_stats

    active = []
    for obj in gc.get_objects():
        if not issubclass(type(obj), Dispatcher) or not obj.overloads:
            continue
        f = obj.py_func
        if not str(f.__module__).startswith("pyrite"):
            continue
        active.append(
            {
                "kernel": f"{f.__module__}.{f.__qualname__}",
                "signatures": list(map(str, obj.signatures)),
                "nopython_signatures": list(map(str, obj.nopython_signatures)),
                "targetoptions": obj.targetoptions,
                "parallel_metadata_present": [
                    bool(getattr(c, "metadata", None) and c.metadata.get("parfor_diagnostics"))
                    for c in obj.overloads.values()
                ],
            }
        )
    return {
        "cache_stats": numba_dispatcher_stats(),
        "active": sorted(active, key=lambda s: s["kernel"]),
    }


def coherent_case(energy_keV, thickness_ang, histories, cxr_nodes=400):
    """#611 physics/grids with coherent emission on a laterally infinite slab.

    Coherent emission refuses incoherent-only reflection selection and needs an
    infinite slab (or Gaussian bunch) for its analytic bunch average.
    """
    import pyrite as pr
    from pyrite import api
    from pyrite.config import FixedSampling
    from pyrite.detectors import EnergyBins

    scene = pr.Scene(
        pr.Beam(
            energy_keV=energy_keV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
        ),
        pr.geometry.Slab(
            "hopg", thickness_ang=thickness_ang, tilt_deg=30.0, width=None, height=None
        ),
        pr.Detector(),
        physics=pr.Physics(
            emission="coherent",
            bremsstrahlung_model="bremslib",
            radiative_model="bremslib-soft-hard",
            radiative_cutoff_eV=1000.0,
        ),
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=histories, brem_histories=histories),
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.linspace(1000.0, 20000.0, cxr_nodes),
            brem=np.arange(1000.0, energy_keV * 1e3, 1000.0),
        ),
    )
    case = api.build_case(scene, numerics)
    return dataclasses.replace(case, E_cut_lines_keV=5.0, E_cut_brem_keV=5.0, seed=1432)


def case_workload(name):
    from pyrite.montecarlo.runner import run_case
    from pyrite.montecarlo.runner.case_tables import case_table_markers

    # Exactly the maintained #611 case builder: same geometry, grids and physics.
    build = runpy.run_path(str(ROOT / "checks/streamed_scoring_memory.py"))["_case"]
    variant = None
    if name.startswith("crystal300-"):
        name, variant = name.split("-", 1)
    material, energy, thickness, count = {
        "thin30": ("hopg", 30, 1e4, 16),
        "thin100": ("hopg", 100, 1e4, 16),
        "crystal300": ("hopg", 300, 1e4, 16),
        "mos2_5mev": ("mos2", 5000, 1e7, 2),
        "layered": ("mos2", 100, 1e4, 16),
    }[name]
    case = build(material, count, energy, thickness, 1432)
    if variant == "coherent":
        case = coherent_case(energy, thickness, count)
    elif variant == "coherent-hr":
        case = coherent_case(energy, thickness, count, cxr_nodes=19001)
    elif variant == "binmean":
        case = dataclasses.replace(case, line_quadrature="bin-mean")
    if name == "layered":
        import pyrite as pr
        from pyrite import api
        from pyrite.detectors import EnergyBins

        case = api.build_case(
            pr.Scene(
                pr.Beam(
                    energy_keV=energy,
                    transverse=pr.beams.Spot(),
                    longitudinal=pr.beams.PointBunch(),
                ),
                pr.geometry.Stack.on_substrate(
                    "mos2", 1e4, "silicon", 1e5, tilt_deg=30, footprint=pr.geometry.Footprint(5, 5)
                ),
                pr.Detector(),
                physics=pr.Physics(
                    bremsstrahlung_model="bremslib",
                    radiative_model="bremslib-soft-hard",
                    radiative_cutoff_eV=1000,
                ),
            ),
            pr.Numerics(
                sampling=pr.config.FixedSampling(count, count),
                energy_model="midpoint",
                energy_bins=EnergyBins(
                    cxr=np.linspace(1000, 20000, 400), brem=np.arange(1000, 1e5, 1000)
                ),
            ),
        )
        case = dataclasses.replace(case, seed=1432)

    def run():
        out = run_case(case, record_timing=True, transport_core="per-electron")
        arrays = {k: digest(v) for k, v in out.items() if isinstance(v, np.ndarray)}
        return {
            "arrays": arrays,
            "phases": {k: v for k, v in out.items() if k.startswith("_t_")},
            "n_segments": out.get("n_segments"),
            "fallbacks": {k: v for k, v in out.items() if "fallback" in k},
        }

    return run, {
        "kind": "complete-case",
        "case": case.to_dict(),
        "table_markers": case_table_markers(case),
    }


def photon_case_workload():
    import pyrite as pr
    from pyrite import api
    from pyrite.config import FixedSampling
    from pyrite.montecarlo.runner import run_case
    from pyrite.montecarlo.runner.case_tables import case_table_markers

    # Documented opt-in usage: 10 um Cu, 30 keV, brem-sourced photon aperture.
    histories, samples = 16, 1000
    scene = pr.Scene(
        pr.Beam(
            energy_keV=30.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()
        ),
        pr.geometry.Slab("cu", thickness_ang=1e5, tilt_deg=5.0, width=None, height=None),
        pr.detectors.PlanarDetector(pr.detectors.PlanarPose((0, 0, 10)), size_mm=(1.0, 1.0)),
        physics=pr.Physics(
            inelastic_model="continuous",
            straggling=True,
            bremsstrahlung_model="bremslib",
            radiative_model="bremslib-soft-hard",
            photon_transport=pr.PhotonTransport(
                (1000.0, 3000.0, 10000.0, 30000.0), source="brem", n_samples=samples, seed=341
            ),
        ),
        numerics=pr.Numerics(
            sampling=FixedSampling(cxr_histories=histories, brem_histories=histories),
            energy_bins=pr.detectors.EnergyBins(
                cxr=np.linspace(1000, 20000, 400), brem=np.arange(1000, 3e4, 500)
            ),
        ),
    )
    case = api.build_case(scene)

    def run():
        out = run_case(case, record_timing=True, transport_core="per-electron")
        arrays = {k: digest(v) for k, v in out.items() if isinstance(v, np.ndarray)}
        return {
            "arrays": arrays,
            "phases": {k: v for k, v in out.items() if k.startswith("_t_")},
            "fallbacks": {k: v for k, v in out.items() if "fallback" in k},
        }

    return run, {
        "kind": "complete-case-photon-aperture",
        "case": case.to_dict(),
        "table_markers": case_table_markers(case),
        "photon_samples": samples,
    }


def transport_workload():
    import pyrite as pr
    from pyrite.config import FixedSampling
    from pyrite.transport import simulate_trajectories

    # Transport-heavy (#1469): default physics (shell-soft-hard, ELSEPA single
    # scattering), 2048 primaries through 200 um Si at 300 keV. The explicit
    # 5 keV cutoff calls the engine directly (no case-level transport reuse),
    # so repeats re-transport; the result is not scorable.
    histories = 2048
    beam = pr.Beam(
        energy_keV=300.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()
    )
    slab = pr.geometry.Slab("silicon", thickness_ang=2e6, tilt_deg=5.0, width=None, height=None)
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=histories, brem_histories=histories), seed=1469
    )

    def run():
        out = simulate_trajectories(beam, slab, numerics=numerics, E_cut_keV=5.0)
        segs = out.segments
        return {
            "arrays": {k: digest(v) for k, v in segs.items() if isinstance(v, np.ndarray)},
            "phases": {},
            "n_segments": out.segment_count,
            "fallbacks": {},
        }

    return run, {
        "kind": "transport-only-engine",
        "histories": histories,
        "material": "silicon",
        "energy_keV": 300.0,
        "thickness_ang": 2e6,
        "E_cut_keV": 5.0,
        "seed": 1469,
    }


def _math_record(result):
    try:
        return dict(result.provenance["math"])
    except (KeyError, TypeError):
        return None


def channeling_workload():
    import pyrite as pr

    # The admitted planar channeling_demo case of the experimental planar
    # channeling guide (17 MeV, 2 um Si, plane (1-10), 200 histories) (#1469).
    scene = dict(
        beam=pr.Beam(
            energy_keV=17_000, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
        ),
        target=pr.geometry.Slab(
            "silicon",
            thickness_ang=20_000,
            tilt_deg=3,
            tilt_azim_deg=45,
            footprint=pr.geometry.Footprint(5, 5),
        ),
        detector=pr.Detector(
            0,
            acceptance=pr.detectors.SensorAcceptance(distance_mm=1_000, width_mm=3, height_mm=3),
        ),
    )
    options = dict(
        physics=pr.Physics(radiative_model="uncoupled"),
        channeling=pr.quantum.Channeling("planar", plane=(1, -1, 0)),
        numerics=pr.Numerics(
            sampling=pr.config.FixedSampling(cxr_histories=200, brem_histories=200),
            energy_bins=pr.detectors.EnergyBins(
                cxr=np.arange(1_000, 24_000, 50), brem=np.arange(1_000, 17_000_000, 500)
            ),
        ),
    )

    def run():
        result = pr.simulate(scene["beam"], scene["target"], scene["detector"], **options)
        densities = {
            "cxr_incoherent": result.cxr.incoherent,
            "cxr_coherent": result.cxr.coherent,
            "brem": result.brem,
            "characteristic": result.characteristic,
            "channeling": result.channeling,
        }
        arrays = {k: digest(v.density) for k, v in densities.items() if v is not None}
        return {
            "arrays": arrays,
            "phases": {},
            "fallbacks": {},
            "math": _math_record(result),
        }

    return run, {"kind": "complete-case-channeling", "guide": "experimental-planar-channeling"}


def photon_workload(name):
    from pyrite.montecarlo.transport.photon_histories import (
        ledger_residual_eV,
        track_photon_histories,
    )

    n, energy, element, density = (
        (128, 2e4, "Cu", 0.0849) if name == "photon" else (128, 2e7, "W", 0.063)
    )
    args = (
        np.tile([0.0, 0.0, -1.0], (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.full(n, energy),
        np.ones(n),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
        [(0.0, 1e9, [(element, density)])],
    )

    def run():
        result = track_photon_histories(
            *args, seed=341, photon_cutoff_eV=1e3, electron_cutoff_eV=1e3
        )
        residual = np.max(np.abs(ledger_residual_eV(result.ledger)))
        assert residual <= 1e-12 * energy
        return {
            "arrays": {
                f"{group}.{k}": digest(v)
                for group in ("ledger", "interactions", "charged", "exits")
                for k, v in getattr(result, group).items()
            },
            "ledger_max_residual_eV": float(residual),
        }

    return run, {
        "kind": "photon-queue-only",
        "fixture": "tests/montecarlo/transport/test_photon_histories.py",
        "histories": n,
        "energy_eV": energy,
        "element": element,
        "density_ang3": density,
        "thickness_ang": 1e9,
        "seed": 341,
        "cutoffs_eV": [1000, 1000],
    }


def axial_workload():
    from pyrite.montecarlo.channeling import (
        AxialEnergyMCWFOptions,
        IncidentEnergyEnsemble,
        axial_energy_mcwf,
    )
    from pyrite.montecarlo.channeling.axial_propagation import AxialPropagationConfig

    config = AxialPropagationConfig(
        mode="mcwf",
        kinetic_energy_eV=1e7,
        mechanisms=("tds",),
        shape=(12, 12),
        depths_ang=(0.0, 2.0),
        step_ang=1.0,
    )
    beam = IncidentEnergyEnsemble(1e7, energy_nodes_eV=(8e6, 12e6), node_weights=(1.0, 1.0))
    options = AxialEnergyMCWFOptions(histories=8, seed=627)

    def run():
        result = axial_energy_mcwf(beam, config, options=options)
        return {
            "arrays": {k: digest(v) for k, v in vars(result).items() if isinstance(v, np.ndarray)}
        }

    return run, {
        "kind": "solver-only-small-grid",
        "fixture": "tests/montecarlo/quantum/test_channeling_axial_energy_mcwf_physics.py",
        "config": dataclasses.asdict(config),
        "options": dataclasses.asdict(options),
        "energy_nodes_eV": [8e6, 12e6],
        "node_weights": [1.0, 1.0],
    }


AXIAL_PROD_HISTORIES = 1000


def axial_production_workload(histories=AXIAL_PROD_HISTORIES, max_cached_solvers=2):
    """Documented energy-resolved example at production depth (1000 A, 5 A steps)."""
    from pyrite.montecarlo.channeling import (
        AxialEnergyMCWFOptions,
        IncidentEnergyEnsemble,
        axial_energy_mcwf,
    )
    from pyrite.montecarlo.channeling.axial_propagation import AxialPropagationConfig

    beam = IncidentEnergyEnsemble(5e6, energy_spread_frac=0.002, order=9)
    config = AxialPropagationConfig(
        kinetic_energy_eV=5e6,
        mode="mcwf",
        shape=(32, 32),
        depths_ang=(0.0, 250.0, 500.0, 1000.0),
        step_ang=5.0,
        oam_string=0,
    )
    options = AxialEnergyMCWFOptions(
        histories=histories,
        seed=627,
        chunk_size=25,
        work_limit=10**9,
        max_cached_solvers=max_cached_solvers,
    )
    edges = np.linspace(100, 10_000, 301)

    def run():
        result = axial_energy_mcwf(
            beam,
            config,
            options=options,
            coherence_pairs=((0, 0), (0, 1)),
            photon_energy_edges_eV=edges,
        )
        return {
            "arrays": {k: digest(v) for k, v in vars(result).items() if isinstance(v, np.ndarray)}
        }

    return run, {
        "kind": "production-depth-energy-resolved-mcwf",
        "fixture": "docs/physics/radiation-physics/channeling-energy-resolved-mcwf.md example; depth 1000 A, step 5 A",
        "config": dataclasses.asdict(config),
        "options": dataclasses.asdict(options),
        "beam": {"energy_eV": 5e6, "energy_spread_frac": 0.002, "order": 9},
        "coherence_pairs": [[0, 0], [0, 1]],
        "photon_energy_edges_eV": digest(edges),
        "split_steps": 200,
    }


def synthetic_segments():
    rng = np.random.default_rng(819)
    n = 4096
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z = rng.uniform(0.05, 0.95, n) * 2000
    return {
        "r_mid": np.column_stack((rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), z)),
        "v_hat": v,
        "L_ang": rng.uniform(5, 40, n),
        "E_keV": rng.uniform(12, 190, n),
        "elec_id": rng.integers(0, 16, n),
        "Ne": 16,
        "thickness_ang": 2000.0,
    }


def continuum_workload(name):
    from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum

    segments = synthetic_segments()
    grid = np.linspace(800.0, 150000.0, 256)
    model = "eedl" if name == "eedl" else "bethe-heitler"

    def run():
        out = mc_brem_spectrum(
            segments, grid, composition=[("Si", 0.05)], cross_section_model=model, chunk=512
        )
        return {"arrays": {"spectrum": digest(out)}}

    return run, {
        "kind": "synthetic-scorer-only",
        "fixture": "tests/montecarlo/brem/test_brem_photon_blocking.py::_segments",
        "model": model,
        "rows": int(segments["L_ang"].size),
        "bins": 256,
        "seed": 819,
        "inputs": {k: digest(v) for k, v in segments.items() if isinstance(v, np.ndarray)},
        "grid": digest(grid),
        "composition": [["Si", 0.05]],
        "chunk": 512,
    }


def characteristic_workload(name):
    from pyrite.montecarlo.spectrum.characteristic import (
        load_characteristic_cross_sections,
        mc_characteristic_spectrum,
    )

    segments = synthetic_segments()
    grid = np.linspace(100.0, 80000.0, 512)
    segments["layer"] = np.zeros(segments["L_ang"].size, dtype=np.int64)
    compositions = [[("Mo", 0.0188), ("S", 0.0376)]]
    layers = None
    populations = [segments]
    if name == "char-layered":
        compositions.append([("Si", 0.05)])
        layers = [(0.0, 400.0, compositions[0]), (400.0, 2000.0, compositions[1])]
        segments["layer"] = (segments["r_mid"][:, 2] >= 400.0).astype(np.int64)
        populations = [
            {
                k: v[segments["layer"] == layer] if isinstance(v, np.ndarray) else v
                for k, v in segments.items()
            }
            for layer in range(2)
        ]
    caches = [{} for _ in compositions]
    tables = {el: load_characteristic_cross_sections(el) for comp in compositions for el, _ in comp}
    pins = {
        el: {
            field.name: digest(getattr(table, field.name))
            for field in dataclasses.fields(table)
            if isinstance(getattr(table, field.name), np.ndarray)
        }
        for el, table in tables.items()
    }

    def run():
        values = [
            mc_characteristic_spectrum(
                population,
                grid,
                composition=comp,
                layers=layers,
                chunk=512,
                E_cut_keV=1.0,
                _table_cache=cache,
            )
            for population, comp, cache in zip(populations, compositions, caches, strict=True)
        ]
        total = np.sum(values, axis=0)
        if not np.isfinite(total).all() or not np.any(total > 0):
            raise RuntimeError("characteristic fixture produced no finite positive signal")
        return {"arrays": {"spectrum": digest(total)}}

    return run, {
        "kind": "synthetic-characteristic-scorer-only",
        "rows": 4096,
        "fixture": "existing photon-blocking segment recipe; per-layer caller ownership follows test_characteristic.py",
        "seed": 819,
        "compositions": compositions,
        "layers": layers,
        "inputs": {k: digest(v) for k, v in segments.items() if isinstance(v, np.ndarray)},
        "table_array_pins": pins,
        "grid": digest(grid),
        "chunk": 512,
        "prepared_cache": "one cache per layer, reused across calls",
    }


def profile_rows(profiler):
    rows = []
    for (filename, line, name), (primitive, calls, exclusive, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        if "pyrite/" in filename:
            filename = "src/pyrite/" + filename.split("pyrite/", 1)[1]
        else:
            filename = Path(filename).name
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "calls": calls,
                "primitive_calls": primitive,
                "exclusive_s": exclusive,
                "cumulative_s": cumulative,
            }
        )
    return sorted(rows, key=lambda r: r["cumulative_s"], reverse=True)[:100]


def measure(args):
    from pyrite._jit_cache import NumbaCompileRecorder

    expected_cpus = os.environ.get("AUDIT1432_CPUS")
    if expected_cpus is not None and len(os.sched_getaffinity(0)) != int(expected_cpus):
        raise RuntimeError("actual OS affinity does not match the frozen CPU allocation")
    start = time.perf_counter()
    if args.case in ("photon", "pair"):
        run, parameters = photon_workload(args.case)
    elif args.case == "photon-case":
        run, parameters = photon_case_workload()
    elif args.case == "transport-si300":
        run, parameters = transport_workload()
    elif args.case == "channeling-demo":
        run, parameters = channeling_workload()
    elif args.case == "axial":
        run, parameters = axial_workload()
    elif args.case == "axial-prod":
        run, parameters = axial_production_workload()
    elif args.case == "axial-prod-cache8":
        run, parameters = axial_production_workload(max_cached_solvers=8)
    elif args.case in ("eedl", "bh"):
        run, parameters = continuum_workload(args.case)
    elif args.case in ("char-multi", "char-layered"):
        run, parameters = characteristic_workload(args.case)
    else:
        run, parameters = case_workload(args.case)
    setup_s = time.perf_counter() - start
    samples = []
    for index in range(args.repeats + 1):
        start = time.perf_counter()
        cpu_start = time.process_time()
        with NumbaCompileRecorder() as recorder:
            result = run()
        samples.append(
            {
                "wall_s": time.perf_counter() - start,
                "process_cpu_s": time.process_time() - cpu_start,
                "first": index == 0,
                "compile_s": recorder.total_s,
                "compile_by_kernel": recorder.by_kernel(),
                **result,
            }
        )
        if index and result["arrays"] != samples[0]["arrays"]:
            raise RuntimeError("repeated input produced different arrays")
    profiler = cProfile.Profile()
    start = time.perf_counter()
    with profiler:
        profiled = run()
    profile_wall_s = time.perf_counter() - start
    if profiled["arrays"] != samples[0]["arrays"]:
        raise RuntimeError("profiling changed output")
    import numba

    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = None
    return {
        "schema": "pyrite.native-cpu-audit.v1",
        "case": args.case,
        "environment": environment(),
        "parameters": parameters,
        "setup_s": setup_s,
        "samples": samples,
        "profile_wall_s": profile_wall_s,
        "profile": profile_rows(profiler),
        "dispatchers": dispatchers(),
        "threading_layer": layer,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 5:
        parser.error("repeats must be 1..5; larger budgets need a reviewed protocol")
    report = measure(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"case": args.case, "wall_s": [s["wall_s"] for s in report["samples"]]}))


if __name__ == "__main__":
    main()
