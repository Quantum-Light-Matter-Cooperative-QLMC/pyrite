"""Paired spectrum timing on one immutable transport realization (#362).

Run thin anchors locally; thick/GPU cases through pyrite remote sync and SLURM.
This measures omission on a fixed axis, not the axis's quadrature accuracy.
Validation: coherent-flat-term-omission
"""

import argparse
import hashlib
import json
import platform
import resource
from time import perf_counter

import numpy as np

from pyrite._backend import REAL, _to_cpu, xp
from pyrite.materials.crystal import CRYSTALS
from pyrite.montecarlo import mc_spectrum, simulate_trajectories
from pyrite.montecarlo.spectrum.lines import _policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--electrons", type=int, default=12)
    parser.add_argument("--physical-electrons", type=float, default=None)
    parser.add_argument("--thickness-ang", type=float, default=1000.0)
    parser.add_argument("--energy-kev", type=float, default=60.0)
    parser.add_argument("--step-ev", type=float, default=2.0)
    parser.add_argument("--stop-ev", type=float, default=12000.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=362)
    parser.add_argument("--route", choices=("auto", "eager", "jit"), default="auto")
    args = parser.parse_args()
    if args.repeats < 1 or args.step_ev <= 0 or args.stop_ev <= 500:
        parser.error("need positive repeats/step and stop above 500 eV")
    if args.route != "auto":
        _policy._USE_JIT_COHERENT_STREAM = False
        _policy._USE_JIT_COHERENT_REDUCTION = args.route == "jit"

    info = CRYSTALS["hopg"]
    started = perf_counter()
    segments = simulate_trajectories(
        args.energy_kev,
        args.electrons,
        args.thickness_ang,
        element="C",
        n_atoms_per_ang3=len(info["basis"]) / info["V_cell"],
        E_cut_keV=5.0,
        seed=args.seed,
        elastic_model="sr",
        transport_core="lockstep",
        bunch_length_fs=1e-3,
        crystal_width_mm=1.0,
        crystal_height_mm=1.0,
    )
    transport_s = perf_counter() - started
    digest = hashlib.sha256()
    for key in sorted(segments):
        value = segments[key]
        if hasattr(value, "dtype"):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(_to_cpu(value)).tobytes())
    energy = np.arange(500.0, args.stop_ev, args.step_ev)
    common = dict(
        crystal="hopg",
        hkl_list=[(0, 0, 2), (0, 0, -2)],
        B_ang2=0.8,
        theta_obs_rad=np.deg2rad(119.0),
        coherent=True,
        longitudinal_rms_fs=1e-3,
        physical_electrons=args.physical_electrons,
    )

    def run(limit):
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()
        started = perf_counter()
        spectrum = mc_spectrum(
            segments,
            energy,
            coherent_flat_omission_limit=limit,
            **common,
        )
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()
        elapsed = perf_counter() - started
        return elapsed, np.asarray(_to_cpu(spectrum), dtype=float)

    # Equal warm-up for both arms; alternate the timed order across repeats.
    run(0.0)
    run(1e-4)
    times = {"full": [], "masked": []}
    spectra = {}
    for repeat in range(args.repeats):
        arms = [("full", 0.0), ("masked", 1e-4)]
        for name, limit in arms[:: 1 if repeat % 2 == 0 else -1]:
            elapsed, spectra[name] = run(limit)
            times[name].append(elapsed)
    full, masked = spectra["full"], spectra["masked"]
    if not np.all(np.isfinite([full, masked])):
        raise RuntimeError("non-finite paired spectrum; no measurement is accepted")
    _, floor = run(1e300)
    roundoff = float(100 * np.finfo(REAL).eps * float(np.max(np.abs(full))))
    error = np.abs(masked - full)
    if np.any(error > 1e-4 * floor + roundoff):
        raise RuntimeError("paired spectrum exceeds the omission bound plus reduction roundoff")
    yields = [float(np.trapezoid(s, energy)) for s in (full, masked)]
    if min(yields) <= 0:
        raise RuntimeError("nonpositive paired yield; no relative comparison is defined")
    centroids = [
        float(np.trapezoid(energy * s, energy)) / y
        for s, y in zip((full, masked), yields, strict=True)
    ]
    changes = {
        "yield": abs(yields[1] / yields[0] - 1),
        "centroid": abs(centroids[1] / centroids[0] - 1),
    }
    result = {
        "workload": vars(args),
        "backend": xp.__name__,
        "dtype": np.dtype(REAL).name,
        "host": platform.node(),
        "python": platform.python_version(),
        "segments": int(segments["r_mid"].shape[0]),
        "segment_sha256": digest.hexdigest(),
        "points": energy.size,
        "transport_s": transport_s,
        "warmups_per_arm": 1,
        "spectrum_wall_s": times,
        "speedup_median": float(np.median(times["full"]) / np.median(times["masked"])),
        "yield": yields,
        "centroid_eV": centroids,
        "relative_change": changes,
        "max_relative_observable_change": max(changes.values()),
        "max_absolute_spectrum_change": float(np.max(np.abs(masked - full))),
        "omission_bound_plus_roundoff_passed": True,
        "comparison_roundoff_absolute": roundoff,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if xp.__name__ == "cupy":
        result["gpu_pool_used_bytes_at_end"] = xp.get_default_memory_pool().used_bytes()
        result["gpu"] = str(xp.cuda.runtime.getDeviceProperties(0)["name"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
