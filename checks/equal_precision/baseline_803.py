"""One pinned #803 baseline run: a scene under fully named transport controls.

Every control that can change cost or output is an explicit argument: core,
elastic model, transport LUT, radiative coupling and moment policy, inelastic
mode and CUDA schedule. Transport is free to bypass a request (the combined LUT
excludes ELSEPA and coupled radiation; ``auto`` may keep full-history), so the
run also records what was *resolved*, by wrapping the host-level LUT builder,
schedule resolver and radiative preparation. The wrappers time and observe;
they never change arguments or results. No production module imports this.

A fresh process times one cold request (JIT, table preparation, upload) and
then ``--repeats`` warm requests of the measured seed. Warm repeats must share
one trajectory digest; a schedule comparison needs digests to match across
processes. Per-primary scores are kept for later ensemble equivalence.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import runpy
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

START = time.perf_counter()
HERE = Path(__file__).parent
SPEC = HERE / "baseline-803.json"
CORES = ("per-electron", "cuda")
SCHEDULES = ("full-history", "continuation", "auto")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--core", choices=CORES, required=True)
    parser.add_argument("--elastic", choices=["elsepa", "sr"], required=True)
    parser.add_argument("--radiative", choices=["uncoupled", "bremslib-soft-hard"], required=True)
    parser.add_argument("--inelastic", choices=["continuous", "shell-soft-hard"], required=True)
    parser.add_argument("--lut", choices=["off", "on"], required=True)
    parser.add_argument("--moments", choices=["table", "exact"], required=True)
    parser.add_argument("--schedule", choices=SCHEDULES, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.histories < 2 or args.repeats < 1:
        parser.error("need at least two histories and one warm repeat")
    if args.schedule != "full-history" and args.core != "cuda":
        parser.error("continuation and auto schedules are CUDA comparisons")
    return args


def load_scene(scene_id, spec_path=SPEC):
    """Return ``(protocol, scene)`` for one ``baseline-803.json`` scene id."""
    protocol = json.loads(spec_path.read_text())
    scene = next((s for s in protocol["scenes"] if s["id"] == scene_id), None)
    if scene is None:
        raise SystemExit(f"unknown scene {scene_id!r} in {spec_path.name}")
    return protocol, scene


def layer_stack(scene, composition_of):
    """``(layers, thickness_ang, keys)``: contiguous ``(z_top, z_bot, comp)`` rows."""
    layers, z = [], 0.0
    for key, thickness_um in scene["layers"]:
        top, z = z, z + float(thickness_um) * 1e4
        layers.append((top, z, composition_of(key)))
    return layers, z, [key for key, _ in scene["layers"]]


def trajectory_digest(result):
    """SHA-256 over every top-level array, in key order (bitwise identity)."""
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(result):
        value = result[key]
        if isinstance(value, np.ndarray):
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


class Observed:
    """Host-level wrappers recording resolved controls and preparation time."""

    def __init__(self):
        self.calls = []

    def wrap(self, owner, name, describe):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            start = time.perf_counter()
            out = original(*args, **kwargs)
            self.calls.append({"site": name, "wall_s": time.perf_counter() - start} | describe(out))
            return out

        setattr(owner, name, observed)

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def describe_lut(lut):
    if lut is None:
        return {"built": False}
    return {
        "built": True,
        "points": int(lut.n_energy),
        "intervals_per_decade": float(lut.intervals_per_decade),
        "max_rel_interp_error": float(lut.max_rel_interp_error),
        "worst_energy_keV": float(lut.worst_energy_keV),
        "tolerance_met": bool(lut.tolerance_met),
    }


def describe_schedule(config):
    return {
        "schedule": config.schedule,
        "continuation": None if config.continuation is None else asdict(config.continuation),
        "max_batch": config.max_batch,
    }


def describe_radiative(args):
    packed = args[-1]
    return {"packed_float64": int(packed.size), "layers": int(args[2].size)}


def install_observers():
    from pyrite.montecarlo.transport import hard_radiative, host_pass

    observed = Observed()
    observed.wrap(host_pass, "_transport_lut", describe_lut)
    observed.wrap(host_pass, "resolve_schedule", describe_schedule)
    observed.wrap(hard_radiative, "radiative_core_args", describe_radiative)
    return observed


def head_from_git_dir(root):
    """HEAD commit read from ``root/.git`` for nodes without a git binary."""
    git_dir = root / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.exists():
        return loose.read_text().strip()
    for line in (git_dir / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve {ref} in {git_dir}")


def git_identity():
    def git(*cmd):
        return subprocess.run(
            ["git", *cmd], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "sha": git("rev-parse", "HEAD"),
            "dirty_paths": git("status", "--porcelain").splitlines(),
        }
    except FileNotFoundError:
        # Dirty state needs git; record it as unverified rather than clean.
        return {"sha": head_from_git_dir(HERE.parents[1]), "dirty_paths": None}


def hardware(core):
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        info["cpu"] = next(
            line.split(":", 1)[1].strip()
            for line in cpu.splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    import numba
    import numpy as np

    info["numpy"], info["numba"] = np.__version__, numba.__version__
    if core == "cuda":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        info["gpu"] = props["name"].decode()
        info["cupy"] = cp.__version__
        info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        info["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
    return info


def transport_arguments(args, protocol, scene, adapter):
    """Complete ``simulate_trajectories`` keywords and their table identities."""
    import numpy as np

    from pyrite.montecarlo.transport import DEFAULT_TRANSPORT_LUT_CONFIG, TransportLUTConfig
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    common = protocol["common"]
    layers, thickness_ang, keys = layer_stack(scene, adapter["catalog_composition"])
    elements = sorted({el for _, _, comp in layers for el, _ in comp})
    stopping = [
        resolve_composition_table(key, comp) for key, (_, _, comp) in zip(keys, layers, strict=True)
    ]
    lut = DEFAULT_TRANSPORT_LUT_CONFIG if args.lut == "on" else TransportLUTConfig(enabled=False)
    kw = dict(
        E0_keV=scene["energy_keV"],
        thickness_ang=thickness_ang,
        composition=layers[0][2],
        layers=layers if len(layers) > 1 else None,
        crystal_width_mm=common["width_mm"],
        crystal_height_mm=common["width_mm"],
        beam_dir=np.array(
            [
                math.sin(math.radians(common["angle_deg"])),
                0,
                math.cos(math.radians(common["angle_deg"])),
            ]
        ),
        E_cut_keV=common["electron_cutoff_keV"],
        energy_model=common["energy_model"],
        max_dE_frac=common["max_dE_frac"],
        max_steps=common["max_steps"],
        elastic_model=args.elastic,
        stopping_tables=[s.arrays() for s in stopping],
        atomic_electron_deflection=common["atomic_electron_deflection"],
        straggling=common["straggling"],
        transport_core=args.core,
        transport_lut_config=lut,
        radiative_moments=args.moments,
        # Explicit schedule: a separately built config otherwise means
        # full-history implicitly, while the process default is ``auto``.
        per_electron_config=PerElectronTransportConfig(schedule=args.schedule),
    )
    tables = {
        "sbethe": [
            {
                "key": s.key,
                "manifest_sha256": s.digest,
                "arrays_sha256": s.manifest["arrays_sha256"],
            }
            for s in stopping
        ]
    }
    if args.elastic == "elsepa":
        entries = [resolve_layer_tables(comp) for _, _, comp in layers]
        kw["elastic_tables"] = [[dict(e.arrays) for e in entry] for entry in entries]
        tables["elsepa"] = [
            [{"key": t.key, "manifest_sha256": t.digest} for e in entry for t in e.tables]
            for entry in entries
        ]
    if args.radiative == "bremslib-soft-hard":
        brem = load_bremsstrahlung_tables(elements)
        kw |= dict(
            radiative_model="bremslib-soft-hard",
            radiative_cutoff_eV=common["hard_photon_cutoff_eV"],
            bremslib_tables=brem,
        )
        tables["bremslib"] = [
            {"key": brem[el].key, "manifest_sha256": brem[el].digest} for el in elements
        ]
    else:
        kw |= dict(radiative_model="uncoupled", radiative_cutoff_eV=None, bremslib_tables=None)
    if args.inelastic == "shell-soft-hard":
        kw |= dict(
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=float(common["inelastic_cutoff_eV"]),
            inelastic_materials=keys,
        )
    controls = {
        "core": args.core,
        "elastic": args.elastic,
        "radiative": args.radiative,
        "inelastic": args.inelastic,
        "lut_requested": args.lut,
        "lut_config": asdict(lut),
        "moments_requested": args.moments,
        "schedule_requested": args.schedule,
    } | {k: common[k] for k in sorted(common)}
    return kw, tables, controls


def device_pool_bytes(core):
    if core != "cuda":
        return None
    import cupy as cp

    return int(cp.get_default_memory_pool().total_bytes())


def write_unsupported(args, scene, controls, error):
    record = {
        "schema": "pyrite.lut-baseline-803.sample.v1",
        "status": "unsupported",
        "scene": scene,
        "controls": controls,
        "histories": args.histories,
        "seed": args.seed,
        "refusal": f"{type(error).__name__}: {error}",
        "source": git_identity(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": "unsupported"}))


def main(argv=None):
    args = parse_args(argv)
    protocol, scene = load_scene(args.scene)
    adapter = runpy.run_path(str(HERE / "pyrite_adapter.py"))
    setup = time.perf_counter()
    import numpy as np

    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.perf.precision import primary_moments

    observed = install_observers()
    kw, tables, controls = transport_arguments(args, protocol, scene, adapter)

    device = None
    if args.core == "cuda":
        import cupy as cp

        device = cp.cuda.Device()

    def sync():
        if device is not None:
            device.synchronize()

    sync()
    tables_s = time.perf_counter() - setup

    def request(seed):
        sync()
        start = time.perf_counter()
        result = simulate_trajectories(Ne=args.histories, seed=seed, **kw)
        sync()
        return result, time.perf_counter() - start

    # A documented refusal of a control combination is a result, not a crash.
    try:
        cold, cold_s = request(args.seed + 1_000_000)
    except (ValueError, NotImplementedError) as error:
        write_unsupported(args, scene, controls, error)
        return
    cold_calls = observed.take()
    del cold
    cold_pool = device_pool_bytes(args.core)
    warm, digests, scores = [], [], None
    for _ in range(args.repeats):
        result, wall = request(args.seed)
        warm.append({"wall_s": wall, "observed": observed.take()})
        digests.append(trajectory_digest(result))
        if scores is None:
            coupled = args.radiative == "bremslib-soft-hard"
            scorer = adapter["score_segments" if coupled else "transport_scores"]
            scores = scorer(result, args.histories)
            radiative = result.get("radiative") or {}
            identity = {
                k: radiative.get(k)
                for k in ("moment_model", "moment_table_sha256")
                if k in radiative
            }
            segments = int(np.asarray(result["L_ang"]).size)
        del result
    if len(set(digests)) != 1:
        raise RuntimeError(f"warm repeats of one seed diverged: {sorted(set(digests))}")
    resolved = {
        "lut_built": any(c.get("built") for c in cold_calls if c["site"] == "_transport_lut"),
        "schedules": sorted({c["schedule"] for c in cold_calls if c["site"] == "resolve_schedule"}),
        "radiative_moments": identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".npz")
    np.savez_compressed(raw, **scores)
    walls = np.array([w["wall_s"] for w in warm])
    record = {
        "schema": "pyrite.lut-baseline-803.sample.v1",
        "status": "ok",
        "scene": scene,
        "controls": controls,
        "resolved": resolved,
        "histories": args.histories,
        "seed": args.seed,
        "cold_seed": args.seed + 1_000_000,
        "segments": segments,
        "trajectory_sha256": digests[0],
        "timings_s": {
            "harness_startup": setup - START,
            "imports_and_tables": tables_s,
            "cold_request": cold_s,
            "warm_requests": walls.tolist(),
            "warm_median": float(np.median(walls)),
            "warm_per_history": float(np.median(walls)) / args.histories,
            "process_total": time.perf_counter() - START,
        },
        "observed_cold": cold_calls,
        "observed_warm": [w["observed"] for w in warm],
        "memory": {
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "device_pool_reserved_bytes_after_cold": cold_pool,
            "device_pool_reserved_bytes_after_warm": device_pool_bytes(args.core),
        },
        "observables": {k: primary_moments(v) for k, v in scores.items()},
        "raw_scores_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "tables": tables,
        "source": git_identity(),
        "hardware": hardware(args.core),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "PYRITE_MC_BACKEND",
                "PYRITE_MC_TRANSPORT_SCHEDULE",
                "PYRITE_MC_ELECTRON_BATCH_MAX",
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "PYRITE_CATALOG",
            )
        },
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "warm_median_s": record["timings_s"]["warm_median"]}
        )
    )


if __name__ == "__main__":
    main()
