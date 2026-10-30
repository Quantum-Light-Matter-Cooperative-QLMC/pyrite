"""Exact CUDA transport kernel: compile options and read-only loads (#796).

Compares the production kernel (the #827 specialized variant compiled by
``cupyx.jit``) with the same generated source compiled through
:mod:`pyrite.montecarlo.transport._jit_rawmodule` under a
:class:`~pyrite.montecarlo.transport._jit_rawmodule.KernelBuild`: a register
cap (``--maxrregcount``), ``__launch_bounds__`` and/or ``__ldg`` loads of the
arrays the kernel never writes. Builds are named on the command line, e.g.
``raw``, ``ro``, ``r168``, ``ro+r128``, ``lb128x2``.

``attributes``
    Runs the #821 benchmark physics once per build on a small population and
    reports per compiled kernel: registers, local (stack) bytes, theoretical
    occupancy from the CUDA occupancy API at 64/128/256 threads, and, from
    ``cuobjdump -sass`` of the cubin the build added to ``CUPY_CACHE_DIR``,
    the count of ``LDG.E.CONSTANT``-class loads, local loads/stores
    (``LDL``/``STL``, i.e. stack and spill traffic) and instructions. A build
    that adds no cubin compiled to a cached one (``raw`` reuses the JIT's).
    Point ``CUPY_CACHE_DIR`` at an empty directory.

``timing``
    Warm transport time of the #821 benchmark physics on its Si/W anchors.
    For every (scene, schedule, seed) all arms run back to back in rotating
    order after an untimed warmup of each, and every returned array is
    hashed; every arm must match ``jit`` bitwise.

Needs a CUDA device and the catalog (ELSEPA, SBETHE and BremsLib tables); run
it through ``pyrite remote`` in a bounded GPU allocation. No validation claim.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from transport_kernel_variants import (  # noqa: E402
    _configurations,
    _digest,
    _max_threads_per_sm,
    _stopping_table,
    _sync,
)

THREADS = (64, 128, 256)
# (element, catalog key, number density / A^-3, energy / keV, thickness / um, histories)
SCENES = {
    "si-300keV-100um": ("Si", "silicon", 0.049942993373787194, 300.0, 100.0, 10_000),
    "w-300keV-10um": ("W", "W", 0.06322199557658834, 300.0, 10.0, 10_000),
    "w-1MeV-100um": ("W", "W", 0.06322199557658834, 1000.0, 100.0, 1_250),
    "si-3MeV-1mm": ("Si", "silicon", 0.049942993373787194, 3000.0, 1000.0, 1_000),
}
_BUILD = re.compile(r"raw|ro|r(\d+)|lb(\d+)x(\d+)")


def parse_build(name):
    """``KernelBuild`` for a name such as ``ro+r128`` (``jit``: None)."""
    from pyrite.montecarlo.transport._jit_rawmodule import TRANSPORT_READONLY, KernelBuild

    if name == "jit":
        return None
    fields = {}
    for part in name.split("+"):
        match = _BUILD.fullmatch(part)
        if match is None:
            raise ValueError(f"unknown build component {part!r}")
        if part == "ro":
            fields["readonly"] = TRANSPORT_READONLY
        elif match[1]:
            fields["maxrregcount"] = int(match[1])
        elif match[2]:
            fields["launch_bounds"] = (int(match[2]), int(match[3]))
    return KernelBuild(**fields)


class Arms:
    """Swap the launcher's variant factory per arm; builds are cached per arm."""

    def __init__(self, names):
        from pyrite.montecarlo.transport import _jit_launch
        from pyrite.montecarlo.transport._jit_rawmodule import RawModuleKernel

        self.launch = _jit_launch
        self.original = _jit_launch.transport_kernel_variant
        self.builds = {name: parse_build(name) for name in names}
        self.kernels = {name: {} for name in names}
        self._wrap = RawModuleKernel

    def select(self, name):
        build, kernels = self.builds[name], self.kernels[name]

        def variant(switches):
            jit_kernel = self.original(switches)
            if build is None:
                kernels[switches] = jit_kernel
            elif switches not in kernels:
                kernels[switches] = self._wrap(jit_kernel, build)
            return kernels[switches]

        self.launch._SPECIALIZE_TRANSPORT_KERNEL = True
        self.launch.transport_kernel_variant = variant


def _functions(kernel):
    """``(Function, source)`` pairs of a JIT or RawModule kernel."""
    if hasattr(kernel, "_compiled"):
        return [(entry[0], entry[1]) for entry in kernel._compiled.values()]
    return [
        (function, kernel._cached_codes[key[0]].code)
        for key, (function, _) in kernel._cache.items()
    ]


def _function_attributes(function):
    from cupy_backends.cuda.api import driver

    registers = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS, function.ptr)
    local = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function.ptr)
    occupancy = {}
    for threads in THREADS:
        blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, threads, 0)
        occupancy[threads] = blocks * threads / _max_threads_per_sm()
    return dict(registers=registers, local_bytes=local, occupancy=occupancy)


def _sass_counts(cached, scratch):
    """SASS statistics of a CuPy cache entry (a hash prefix, then the cubin).

    The stored cubin can end short of its program headers; it is padded so
    the disassembler accepts it. ``CUOBJDUMP`` names a ``cuobjdump`` at least
    as new as the NVRTC that compiled it.
    """
    payload = cached.read_bytes()
    cubin = bytearray(payload[payload.index(b"\x7fELF") :])
    phoff = int.from_bytes(cubin[32:40], "little")
    phentsize = int.from_bytes(cubin[54:56], "little")
    phnum = int.from_bytes(cubin[56:58], "little")
    cubin.extend(bytes(max(0, phoff + phentsize * phnum - len(cubin))))
    elf = scratch / (cached.stem + ".elf.cubin")
    elf.write_bytes(bytes(cubin))
    tool = os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump") or "cuobjdump"
    sass = subprocess.run([tool, "-sass", str(elf)], capture_output=True, text=True).stdout
    elf.unlink()
    lines = [line for line in sass.splitlines() if re.search(r"/\*[0-9a-f]{4,}\*/", line)]

    def count(pattern):
        return sum(1 for line in lines if re.search(pattern, line))

    return dict(
        instructions=len(lines),
        ldg=count(r"\bLDG\."),
        ldg_constant=count(r"\bLDG\.E[.\w]*CONSTANT"),
        ldl=count(r"\bLDL\b"),
        stl=count(r"\bSTL\b"),
    )


def attributes(args):
    from pyrite.montecarlo.transport import simulate_trajectories

    cache = Path(os.environ["CUPY_CACHE_DIR"])
    kwargs = _configurations()["bench821"]
    arms = Arms(args.builds)
    records = {}
    for name in args.builds:
        before = set(cache.glob("*.cubin"))
        arms.select(name)
        simulate_trajectories(**kwargs, Ne=args.histories)
        _sync()
        added = sorted(set(cache.glob("*.cubin")) - before)
        kernels = []
        for kernel in arms.kernels[name].values():
            for function, source in _functions(kernel):
                kernels.append(
                    _function_attributes(function)
                    | dict(source_sha256=hashlib.sha256(source.encode()).hexdigest())
                )
                if args.dump:
                    Path(args.dump).mkdir(parents=True, exist_ok=True)
                    Path(args.dump, f"{name}.cu").write_text(source)
        records[name] = dict(
            kernels=kernels,
            new_cubins=[path.name for path in added],
            sass=[_sass_counts(path, Path(args.dump or ".")) for path in added],
        )
        print(json.dumps({name: records[name]}), file=sys.stderr, flush=True)
    return dict(mode="attributes", histories=args.histories, builds=records)


def _scene_kwargs(scene):
    from pyrite.montecarlo.transport import TransportLUTConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables

    element, catalog_key, density, energy_keV, thickness_um, _ = SCENES[scene]
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
        stopping_tables=[_stopping_table(catalog_key, composition).arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables=load_bremsstrahlung_tables([element]),
        radiative_moments="table",
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=False),
    )


def timing(args):
    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.montecarlo.transport.batching import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG

    names = args.builds
    arms = Arms(names)
    rows = []
    for scene in args.scenes:
        kwargs = _scene_kwargs(scene)
        histories = args.histories or SCENES[scene][-1]
        for schedule in args.schedules:
            config = replace(DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG, schedule=schedule)

            def run(seed, n, kwargs=kwargs, config=config):
                _sync()
                start = time.perf_counter()
                result = simulate_trajectories(
                    **kwargs, Ne=n, seed=seed, per_electron_config=config
                )
                _sync()
                return time.perf_counter() - start, result

            for name in names:  # warm: transpile, compile, tables
                arms.select(name)
                run(args.seeds[0] + 1_000_000, min(histories, 2000))
            for index, seed in enumerate(args.seeds):
                shift = index % len(names)
                order = names[shift:] + names[:shift]
                record = dict(scene=scene, schedule=schedule, seed=seed, histories=histories)
                for name in order:
                    arms.select(name)
                    seconds, result = run(seed, histories)
                    record[f"{name}_s"] = seconds
                    record[f"{name}_digest"] = _digest(result)
                    record["rows"] = int(np.asarray(result["electron_id"]).size)
                    del result
                record["identical"] = {
                    name: record[f"{name}_digest"] == record["jit_digest"] for name in names
                }
                record["speedup"] = {name: record["jit_s"] / record[f"{name}_s"] for name in names}
                rows.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    summary = {}
    for scene in args.scenes:
        for schedule in args.schedules:
            group = [r for r in rows if r["scene"] == scene and r["schedule"] == schedule]
            summary[f"{scene}/{schedule}"] = {
                name: dict(
                    seconds=_spread([r[f"{name}_s"] for r in group]),
                    speedup=_spread([r["speedup"][name] for r in group]),
                    all_identical=all(r["identical"][name] for r in group),
                )
                for name in names
            }
    return dict(mode="timing", builds=names, rows=rows, summary=summary)


def _spread(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["attributes", "timing"])
    parser.add_argument("--builds", nargs="+", default=["jit", "raw"])
    parser.add_argument("--histories", type=int, default=0, help="0: the scene's (timing)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    parser.add_argument("--schedules", nargs="+", default=["full-history", "continuation"])
    parser.add_argument("--dump", help="attributes: write each build's source here")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.builds[0] != "jit":
        args.builds = ["jit", *[b for b in args.builds if b != "jit"]]
    if args.mode == "attributes" and not args.histories:
        args.histories = 500
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
