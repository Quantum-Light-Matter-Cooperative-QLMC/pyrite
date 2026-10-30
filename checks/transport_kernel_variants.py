"""Exact CUDA transport kernel: specialized variants versus the monolithic kernel (#827).

Two modes, each one JSON record on stdout (or ``--output``):

``attributes``
    Runs each named feature configuration once on a small population with the
    monolithic kernel and then with its specialized variant, and reports, per
    compiled kernel, registers per thread, local (stack) bytes and the
    theoretical occupancy from the CUDA occupancy API at 32-512 threads per
    block, plus the first-launch (transpile + NVRTC compile or CuPy disk-cache
    load) and warm launch wall times. Point ``CUPY_CACHE_DIR`` at an empty
    directory to measure cold compiles; a second process then measures the
    disk-cache hit.

``timing``
    Warm transport timing of the #821 benchmark physics (ELSEPA, midpoint,
    SBETHE, coupled BremsLib at 1 keV, 10 keV cutoff, 10 x 10 mm slab) on the
    Si 300 keV / 100 um and W 300 keV / 10 um anchors. For every (scene,
    schedule, seed) the two arms run back to back in alternating order after an
    untimed warmup of both, and every returned array is hashed; arms must agree
    bitwise. On a checkout without #827 the single arm is ``main``, so the
    digests can be compared with a run of this branch.

Needs a CUDA device and the catalog (ELSEPA, SBETHE and BremsLib tables); run
it through ``pyrite remote`` in a bounded GPU allocation. No validation claim.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import numpy as np

THREADS = (32, 64, 128, 256, 512)
SCENES = {
    "si-300keV-100um": ("Si", "silicon", 0.049942993373787194, 300.0, 100.0),
    "w-300keV-10um": ("W", "W", 0.06322199557658834, 300.0, 10.0),
}


def _launch_module():
    from pyrite.montecarlo.transport import _jit_launch

    return _jit_launch


def _arms():
    return ["monolithic", "specialized"] if _has_variants() else ["main"]


def _has_variants():
    return hasattr(_launch_module(), "_SPECIALIZE_TRANSPORT_KERNEL")


def _select(arm):
    if arm != "main":
        _launch_module()._SPECIALIZE_TRANSPORT_KERNEL = arm == "specialized"


def _sync():
    import cupy

    cupy.cuda.Device().synchronize()


def _digest(result):
    """sha256 over every array, in key order; device arrays are downloaded."""
    import cupy

    digest = hashlib.sha256()
    for key in sorted(result):
        value = result[key]
        if isinstance(value, cupy.ndarray):
            value = cupy.asnumpy(value)
        if isinstance(value, np.ndarray):
            digest.update(key.encode())
            digest.update(str(value.dtype).encode() + str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).view(np.uint8).tobytes())
        elif isinstance(value, (int, float, str, bool, np.generic)):
            digest.update(f"{key}={value!r}".encode())
    return digest.hexdigest()


def _scene_kwargs(scene):
    from pyrite.montecarlo.transport import TransportLUTConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    element, catalog_key, density, energy_keV, thickness_um = SCENES[scene]
    composition = [(element, density)]
    return dict(
        E0_keV=energy_keV,
        composition=composition,
        thickness_ang=thickness_um * 1e4,
        crystal_width_mm=10.0,
        crystal_height_mm=10.0,
        beam_dir=np.array([0.0, 0.0, 1.0]),
        E_cut_keV=10.0,
        energy_model="midpoint",
        max_dE_frac=0.05,
        max_steps=200_000,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in resolve_layer_tables(composition)]],
        stopping_tables=[resolve_composition_table(catalog_key, composition).arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables=load_bremsstrahlung_tables([element]),
        radiative_moments="table",
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=False),
    )


def _schedule_config(schedule):
    from pyrite.montecarlo.transport.batching import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG

    return replace(DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG, schedule=schedule)


def timing(args):
    from pyrite.montecarlo.transport import simulate_trajectories

    arms = _arms()
    rows = []
    for scene in args.scenes:
        kwargs = _scene_kwargs(scene)
        for schedule in args.schedules:
            config = _schedule_config(schedule)

            def run(seed, n, kwargs=kwargs, config=config):
                _sync()
                start = time.perf_counter()
                result = simulate_trajectories(
                    **kwargs, Ne=n, seed=seed, per_electron_config=config
                )
                _sync()
                return time.perf_counter() - start, result

            for arm in arms:  # warm: transpile, compile, tables
                _select(arm)
                run(args.seeds[0] + 1_000_000, min(args.histories, 2000))
            for index, seed in enumerate(args.seeds):
                order = arms if index % 2 == 0 else arms[::-1]
                record = dict(scene=scene, schedule=schedule, seed=seed, histories=args.histories)
                for arm in order:
                    _select(arm)
                    seconds, result = run(seed, args.histories)
                    record[f"{arm}_s"] = seconds
                    record[f"{arm}_digest"] = _digest(result)
                    record["rows"] = int(np.asarray(result["electron_id"]).size)
                    del result
                if len(arms) == 2:
                    record["identical"] = (
                        record["monolithic_digest"] == record["specialized_digest"]
                    )
                    record["speedup"] = record["monolithic_s"] / record["specialized_s"]
                rows.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    summary = {}
    for scene in args.scenes:
        for schedule in args.schedules:
            group = [r for r in rows if r["scene"] == scene and r["schedule"] == schedule]
            summary[f"{scene}/{schedule}"] = {
                f"{arm}_s": _spread([r[f"{arm}_s"] for r in group]) for arm in arms
            } | (
                {
                    "speedup": _spread([r["speedup"] for r in group]),
                    "all_identical": all(r["identical"] for r in group),
                }
                if len(arms) == 2
                else {}
            )
    return dict(mode="timing", arms=arms, rows=rows, summary=summary)


def _spread(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def _configurations():
    """Small populations exercising the variants the issue names."""
    from pyrite.materials import CATALOG
    from pyrite.montecarlo.transport import TransportLUTConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    silicon = CATALOG.crystal("silicon").composition
    base = dict(
        E0_keV=30.0,
        thickness_ang=2.0e4,
        composition=silicon,
        E_cut_keV=10.0,
        seed=11,
        energy_model="midpoint",
        max_dE_frac=0.05,
        elastic_model="elsepa",
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    radiative = dict(
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables=load_bremsstrahlung_tables(["Si"]),
        radiative_moments="table",
    )
    shell = dict(
        inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0, inelastic_materials=["silicon"]
    )
    return {
        "bench821": base | radiative | {"crystal_width_mm": 10.0, "crystal_height_mm": 10.0},
        "bench821+straggling": base | radiative | {"straggling": True},
        "shell": base | shell,
        "shell+straggling": base | shell | {"straggling": True},
        "shell+secondaries": base | shell | {"secondary_threshold_eV": 10_000.0},
        "all": base
        | shell
        | radiative
        | {
            "straggling": True,
            "secondary_threshold_eV": 10_000.0,
            "crystal_width_mm": 2e-3,
            "crystal_height_mm": 2e-3,
        },
    }


def _kernel_attributes(kernel):
    from cupy_backends.cuda.api import driver

    out = []
    for function, _ in kernel._cache.values():
        registers = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS, function.ptr)
        local = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function.ptr)
        occupancy = {}
        for threads in THREADS:
            blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, threads, 0)
            occupancy[threads] = blocks * threads / _max_threads_per_sm()
        out.append(dict(registers=registers, local_bytes=local, occupancy=occupancy))
    return out


def _max_threads_per_sm():
    import cupy

    return cupy.cuda.Device().attributes["MaxThreadsPerMultiProcessor"]


def attributes(args):
    from pyrite.montecarlo.transport import _jit_kernel_variants, simulate_trajectories
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig

    launch = _launch_module()
    selected = []
    original = launch.transport_kernel_variant

    def recording(switches):
        selected.append(switches)
        return original(switches)

    launch.transport_kernel_variant = recording
    records = {}
    for name, kwargs in _configurations().items():
        # The all-features variant includes the continuation switch.
        config = PerElectronTransportConfig(
            schedule="continuation" if name == "all" else "full-history"
        )
        record = {}
        for arm in ("monolithic", "specialized"):
            _select(arm)
            selected.clear()
            times = []
            for _ in range(2):
                _sync()
                start = time.perf_counter()
                simulate_trajectories(**kwargs, Ne=args.histories, per_electron_config=config)
                _sync()
                times.append(time.perf_counter() - start)
            kernel = (
                _jit_kernel_variants.monolithic_transport_kernel
                if arm == "monolithic"
                else original(selected[0])
            )
            record[arm] = dict(
                first_s=times[0],
                warm_s=times[1],
                kernels=_kernel_attributes(kernel),
                switches=selected[0]._asdict() if selected else None,
                passes=len({tuple(s) for s in selected}),
            )
        records[name] = record
        print(json.dumps({name: record}), file=sys.stderr, flush=True)
    return dict(
        mode="attributes",
        histories=args.histories,
        cupy_cache_dir=os.environ.get("CUPY_CACHE_DIR"),
        variants_compiled=original.cache_info().currsize,
        configurations=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["attributes", "timing"])
    parser.add_argument("--histories", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    parser.add_argument("--schedules", nargs="+", default=["full-history", "continuation"])
    parser.add_argument("--output")
    args = parser.parse_args()
    import cupy

    record = timing(args) if args.mode == "timing" else attributes(args)
    record["device"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    record["cupy"] = cupy.__version__
    text = json.dumps(record, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
