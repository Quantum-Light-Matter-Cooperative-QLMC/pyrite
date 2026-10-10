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
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrite._backend import REAL, _to_cpu, xp
from pyrite.materials.crystal import CRYSTALS
from pyrite.montecarlo import mc_spectrum, simulate_trajectories
from pyrite.montecarlo.spectrum.lines import _policy
from pyrite.montecarlo.spectrum.phase_retention import PhaseRetentionCost, PhaseRetentionPolicy


def _benchmark_transport(args, energy, common):
    """Generate once or replay saved benchmark transport and spectrum inputs."""
    from pyrite.montecarlo.trajectories import (
        case_digest,
        read_trajectory_artifact,
        write_trajectory_artifact,
    )

    info = CRYSTALS["hopg"]
    settings = dict(
        energy_kev=args.energy_kev,
        electrons=args.electrons,
        thickness_ang=args.thickness_ang,
        element="C",
        n_atoms_per_ang3=len(info["basis"]) / info["V_cell"],
        E_cut_keV=5.0,
        seed=args.seed,
        elastic_model="sr",
        transport_core="lockstep",
        bunch_length_fs=args.bunch_rms_fs,
        crystal_width_mm=1.0,
        crystal_height_mm=1.0,
    )
    inputs = {"energy_eV": energy, "spectrum_kwargs": common}
    fingerprint = case_digest({"transport": settings, "spectrum": inputs})
    path = args.trajectory
    if path is not None and path.exists():
        started = perf_counter()
        artifact = read_trajectory_artifact(path)
        if artifact.provenance.get("benchmark_inputs_sha256") != fingerprint:
            raise ValueError("saved benchmark trajectory does not match requested transport/axes")
        if case_digest(artifact.spectrum_inputs) != case_digest(inputs):
            raise ValueError("saved benchmark spectrum inputs changed")
        return artifact.transport, 0.0, perf_counter() - started, True
    started = perf_counter()
    segments = simulate_trajectories(
        settings["energy_kev"],
        settings["electrons"],
        settings["thickness_ang"],
        **{
            k: v
            for k, v in settings.items()
            if k not in {"energy_kev", "electrons", "thickness_ang"}
        },
    )
    transport_s = perf_counter() - started
    io_s = 0.0
    if path is not None:
        started = perf_counter()
        write_trajectory_artifact(
            path,
            segments,
            case={"name": "phase-retention-benchmark", "E0_keV": args.energy_kev, **settings},
            settings=settings,
            provenance={"benchmark_inputs_sha256": fingerprint},
            spectrum_inputs=inputs,
        )
        segments = read_trajectory_artifact(path).transport
        io_s = perf_counter() - started
    return segments, transport_s, io_s, False


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
    parser.add_argument("--bunch-rms-fs", type=float, default=1e-3)
    parser.add_argument("--observation-deg", type=float, default=119.0)
    parser.add_argument(
        "--trajectory", type=Path, help="save/replay benchmark trajectories and axes"
    )
    parser.add_argument(
        "--single-reflection", action="store_true", help="evaluate the (002) row alone"
    )
    parser.add_argument(
        "--calibrate-policy", action="store_true", help="fit and check the gate on one reflection"
    )
    parser.add_argument(
        "--phase-policy", action="store_true", help="measure conservative cost-gate fallback"
    )
    parser.add_argument("--route", choices=("auto", "eager", "jit"), default="auto")
    args = parser.parse_args()
    if args.calibrate_policy and (not args.single_reflection or not args.phase_policy):
        parser.error("calibration requires --single-reflection and --phase-policy")
    if (
        args.repeats < 1
        or args.step_ev <= 0
        or args.stop_ev <= 500
        or not np.isfinite(args.bunch_rms_fs)
        or args.bunch_rms_fs <= 0
        or not np.isfinite(args.observation_deg)
        or not 0 <= args.observation_deg <= 180
    ):
        parser.error(
            "need positive repeats/step/bunch RMS, stop above 500 eV and angle in [0, 180]"
        )
    if args.route != "auto":
        _policy._USE_JIT_COHERENT_STREAM = False
        _policy._USE_JIT_COHERENT_REDUCTION = args.route == "jit"

    energy = np.arange(500.0, args.stop_ev, args.step_ev)
    common = dict(
        crystal="hopg",
        hkl_list=[(0, 0, 2)] if args.single_reflection else [(0, 0, 2), (0, 0, -2)],
        B_ang2=0.8,
        theta_obs_rad=np.deg2rad(args.observation_deg),
        coherent=True,
        longitudinal_rms_fs=args.bunch_rms_fs,
        physical_electrons=args.physical_electrons,
    )
    segments, transport_s, artifact_io_s, replayed = _benchmark_transport(args, energy, common)
    digest = hashlib.sha256()
    for key in sorted(segments):
        value = segments[key]
        if hasattr(value, "dtype"):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(_to_cpu(value)).tobytes())

    decisions = []
    phase_policy = PhaseRetentionPolicy(report=decisions.append)

    def run(limit, policy=None):
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()
        started = perf_counter()
        spectrum = mc_spectrum(
            segments,
            energy,
            coherent_flat_omission_limit=limit,
            phase_retention=policy,
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
    if args.phase_policy:
        run(1e-4, phase_policy)
        times["gated_fallback"] = []
        decisions.clear()
    spectra = {}
    for repeat in range(args.repeats):
        arms = [("full", 0.0, None), ("masked", 1e-4, None)]
        if args.phase_policy:
            arms.append(("gated_fallback", 1e-4, phase_policy))
        for name, limit, policy in arms[:: 1 if repeat % 2 == 0 else -1]:
            elapsed, spectra[name] = run(limit, policy)
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
        "workload": {**vars(args), "trajectory": str(args.trajectory) if args.trajectory else None},
        "backend": xp.__name__,
        "dtype": np.dtype(REAL).name,
        "host": platform.node(),
        "python": platform.python_version(),
        "segments": int(segments["r_mid"].shape[0]),
        "segment_sha256": digest.hexdigest(),
        "energy_sha256": hashlib.sha256(np.ascontiguousarray(energy).tobytes()).hexdigest(),
        "points": energy.size,
        "transport_s": transport_s,
        "artifact_io_s": artifact_io_s,
        "replayed_trajectory": replayed,
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
    if args.phase_policy:
        from dataclasses import asdict

        if not np.array_equal(spectra["gated_fallback"], full):
            raise RuntimeError("cost-gate fallback changed the full spectrum")
        result["phase_policy"] = {
            "mode": "missing-cost-evidence",
            "full_equivalence": True,
            "decisions": [
                {**asdict(d), "retained": d.retained, "skipped": d.skipped} for d in decisions
            ],
            # End-to-end difference includes mask certification and sync, not
            # just Python lookup. Noisy extrema are an empirical estimate,
            # not a rigorous upper bound or a production calibration rule.
            "overhead_empirical_upper_s": max(
                0.0, max(times["gated_fallback"]) - min(times["full"])
            ),
        }
    if args.calibrate_policy:
        # One row is one decision scope. Whole-request paired timings must not
        # be applied separately to multiple eager/JIT reflection rows.
        calibration_decisions = []
        full_lower = min(times["full"])
        reduced_upper = max(times["masked"])
        overhead_upper = result["phase_policy"]["overhead_empirical_upper_s"]

        def estimate(scope):
            if scope.rows != 1 or scope.physical_electrons != args.physical_electrons:
                return None
            return PhaseRetentionCost(
                scope,
                full_lower,
                reduced_upper,
                overhead_upper,
                evidence=f"paired calibration on {digest.hexdigest()}; same process/axis/row",
            )

        calibrated = PhaseRetentionPolicy(estimate=estimate, report=calibration_decisions.append)
        run(0.0)
        run(1e-4, calibrated)
        calibration_decisions.clear()
        validation_times = {"full": [], "calibrated": []}
        validation_spectra = {}
        for repeat in range(args.repeats):
            arms = [("full", 0.0, None), ("calibrated", 1e-4, calibrated)]
            for name, limit, policy in arms[:: 1 if repeat % 2 == 0 else -1]:
                elapsed, validation_spectra[name] = run(limit, policy)
                validation_times[name].append(elapsed)
        accepted = any(d.simplify for d in calibration_decisions)
        expected = masked if accepted else full
        if not np.array_equal(validation_spectra["calibrated"], expected):
            raise RuntimeError("calibrated policy did not evaluate its reported reduction")
        result["calibrated_policy"] = {
            "calibration": {
                "full_lower_s": full_lower,
                "reduced_upper_s": reduced_upper,
                "overhead_upper_s": overhead_upper,
                "interpretation": "empirical estimates; not certified runtime bounds",
            },
            "accepted": accepted,
            "spectrum_wall_s": validation_times,
            "speedup_median": float(
                np.median(validation_times["full"]) / np.median(validation_times["calibrated"])
            ),
            "reported_reduction_equivalence": True,
            "decisions": [
                {**asdict(d), "retained": d.retained, "skipped": d.skipped}
                for d in calibration_decisions
            ],
        }
    if xp.__name__ == "cupy":
        result["gpu_pool_used_bytes_at_end"] = xp.get_default_memory_pool().used_bytes()
        result["gpu"] = str(xp.cuda.runtime.getDeviceProperties(0)["name"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
