"""Issue 350 charge-weighted CUDA validation; run through pyrite remote jobs.

One persisted transport serves every ladder rung, including scheduler slices.
These are numerical comparisons, not production window or sampling certificates.
"""

import argparse
import hashlib
import json
import pickle
import resource
import time
import warnings
from pathlib import Path

import numpy as np


def _atomic(path, value, *, binary=False):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb" if binary else "w") as stream:
        if binary:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def _host(value):
    from pyrite._backend import _to_cpu

    if isinstance(value, dict):
        return {key: _host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_host(item) for item in value)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return np.asarray(_to_cpu(value))
    return value


def _stamp():
    path = Path(".pyrite-sync")
    if not path.exists():
        raise RuntimeError("run this check in a stamped pyrite remote checkout")
    return dict(line.split(": ", 1) for line in path.read_text().splitlines() if ": " in line)


def _require_gpu():
    from pyrite._backend import REAL, xp

    if xp.__name__ != "cupy" or np.dtype(REAL) != np.dtype(np.float64):
        raise RuntimeError("this remote check requires CUDA float64")


def _gpu_limits():
    from pyrite.montecarlo import mc_spectrum
    from pyrite.montecarlo.spectrum.coherent_population import CoherentSamplingError

    energy = np.arange(700.0, 1500.0, 2.0)

    def tracks(count):
        return {
            "r_mid": np.tile([0.0, 0.0, 5.0], (count, 1)),
            "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
            "L_ang": np.full(count, 10.0),
            "E_keV": np.full(count, 30.0),
            "t_ang": np.zeros(count),
            "t0_ang": np.zeros(count),
            "elec_id": np.arange(count),
            "Ne": count,
            "thickness_ang": 10.0,
            "crystal_width_ang": None,
            "crystal_height_ang": None,
        }

    def spectrum(segments, population):
        return mc_spectrum(
            segments,
            energy,
            "hopg",
            [(0, 0, 2)],
            B_ang2=0.8,
            n_hat=np.array([1.0, 0.0, 0.1]),
            coherent=True,
            physical_electrons=population,
        )

    single = spectrum(tracks(1), 1)
    if np.any(~np.isfinite(single)) or np.max(single) <= 0:
        raise AssertionError("invalid CUDA single-electron reference")
    for count in (2, 3, 7):
        np.testing.assert_allclose(
            spectrum(tracks(count), 12), 12 * single, rtol=1e-10, equal_nan=False
        )
    unresolved = tracks(2)
    unresolved["t_ang"][1] = 5.0
    try:
        spectrum(unresolved, 12)
    except CoherentSamplingError:
        pass
    else:
        raise AssertionError("CUDA negative pair estimate was not refused")
    return {"aligned_samples": [2, 3, 7], "physical_electrons": 12, "negative_refusal": True}


def _case(args):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    smoke = args.mode == "smoke"
    samples = 200 if smoke else 40
    sweep = material_sweep(
        "hopg",
        catalog_profile="hopg_short" if smoke else "standard",
        energy_keV=60.0 if smoke else args.energy,
        thickness_ang=1e5 if smoke else 1e7,
        tilt_deg=45.0 if smoke else 5.0,
        tilt_azim_deg=135.0 if smoke else 0.0,
        **({"transverse_fwhm_mm": 0.1} if smoke else {}),
        line_grid_policy={"windows": True, "max_points": args.max_points},
        n_electrons=samples,
        n_electrons_brem=samples,
    )
    case = dict(build_cases(sweep, samples, samples, coherent_emission=True)[0])
    case["bunch_charge_pc"] = args.charge_pc
    if not smoke:
        case.update(seed=7, bunch_length_fs=100.0)
    return case


def _replay_transport(record_path, config, stamp):
    """Import trusted prior-run inputs, validating them against their evidence."""
    from pyrite.energy_grid.convergence import require_identical_segments, segment_fingerprint

    record_path = Path(record_path)
    snapshot = record_path.with_suffix(".transport.pkl")
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    with snapshot.open("rb") as stream:
        snapshot_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = pickle.load(stream)
    source_stamp = record["stamp"]
    if not source_stamp.get("code_digest") or not source_stamp.get("code_tables_digest"):
        raise RuntimeError("source record lacks code/table provenance")
    if saved["config"] != config or record["config"] != config:
        raise RuntimeError("replay inputs differ from the source record/snapshot")
    if (
        saved["digest"] != source_stamp["code_digest"]
        or saved.get("tables_digest") != source_stamp["code_tables_digest"]
    ):
        raise RuntimeError("source snapshot belongs to different code/tables")
    if source_stamp["code_tables_digest"] != stamp.get("code_tables_digest"):
        raise RuntimeError("replay requires the original table digest")
    case, transport = saved["case"], saved["transport"]
    require_identical_segments(record["fingerprint"], segment_fingerprint(transport["segs"]))
    if (
        case["Ne"] != record["incident_samples"]
        or transport["Ne_lines"] != record["incident_samples"]
        or case["bunch_charge_pc"] != config["charge_pc"]
    ):
        raise RuntimeError("replay population differs from its source evidence")
    energy = np.asarray(transport["E_grid"], dtype=float)
    axis = record["auto_record"]
    if (
        energy.ndim != 1
        or energy.size < 2
        or np.any(~np.isfinite(energy))
        or np.any(np.diff(energy) <= 0)
        or energy.size != axis["num"]
        or energy[0] != axis["start_eV"]
        or energy[-1] != axis["stop_eV"]
    ):
        raise RuntimeError("replay axis differs from its source evidence")
    provenance = {
        "record": str(record_path.resolve()),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "snapshot_sha256": snapshot_digest,
        "stamp": source_stamp,
        "fingerprint": record["fingerprint"],
    }
    return case, transport, provenance


def run(args):
    from pyrite._backend import BACKEND, REAL, _to_cpu
    from pyrite._line_windows import FeatureSeed, build_window_plan
    from pyrite.energy_grid.convergence import require_identical_segments, spectrum_observables
    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.coherent_population import (
        CoherentSamplingError,
        physical_bunch_electrons,
    )

    started = time.perf_counter()
    output = Path(args.out)
    snapshot = output.with_suffix(".transport.pkl")
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"out", "max_minutes", "transport_record"}
    }
    stamp = _stamp()
    result = (
        json.loads(output.read_text())
        if output.exists()
        else {
            "config": config,
            "stamp": stamp,
            "evals": {},
            "state": "prepared",
        }
    )
    if result["config"] != config or any(
        result["stamp"].get(key) != stamp.get(key) for key in ("code_digest", "code_tables_digest")
    ):
        raise RuntimeError("refusing to resume with changed inputs, code or table digest")
    replay = None
    if getattr(args, "transport_record", None):
        source_path = Path(args.transport_record)
        if (
            source_path.resolve() == output.resolve()
            or source_path.with_suffix(".transport.pkl").resolve() == snapshot.resolve()
        ):
            raise RuntimeError("replay output must differ from its source record and snapshot")
        replay = _replay_transport(source_path, config, stamp)
        if "transport_source" in result and result["transport_source"] != replay[2]:
            raise RuntimeError("replay source changed since the previous slice")
        result["transport_source"] = replay[2]
    elif "transport_source" in result:
        raise RuntimeError("resuming an imported transport requires --transport-record")
    if result["state"] == "done":
        return 0
    caught = []

    def persist():
        # Every resumable checkpoint carries warnings, even before a slice exits.
        result["warnings"] = list(
            dict.fromkeys(result.get("warnings", []) + [str(item.message) for item in caught])
        )
        _atomic(output, result)

    persist()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _require_gpu()
            if "gpu_limits" not in result:
                result["gpu_limits"] = _gpu_limits()
            if snapshot.exists():
                with snapshot.open("rb") as stream:
                    saved = pickle.load(stream)
                if (
                    saved["config"] != config
                    or saved["digest"] != stamp["code_digest"]
                    or saved.get("tables_digest") != stamp.get("code_tables_digest")
                ):
                    raise RuntimeError("transport snapshot belongs to different inputs/code")
                case, transport = saved["case"], saved["transport"]
                ladder = CaseLadder(case, transport=transport, coherent=True)
            elif replay is not None:
                case, transport, _ = replay
                ladder = CaseLadder(case, transport=transport, coherent=True)
                _atomic(
                    snapshot,
                    {
                        "config": config,
                        "digest": stamp["code_digest"],
                        "tables_digest": stamp.get("code_tables_digest"),
                        "case": case,
                        "transport": transport,
                    },
                    binary=True,
                )
            else:
                case = _case(args)
                t0 = time.perf_counter()
                result["physical_electrons"] = physical_bunch_electrons(case)
                result["incident_samples"] = case["Ne"]
                persist()
                print(
                    "transport",
                    args.mode,
                    "incident",
                    case["Ne"],
                    "physical",
                    result["physical_electrons"],
                    flush=True,
                )
                transport = runner._transport_case(case, transport_core="auto")
                ladder = CaseLadder(case, transport=transport, coherent=True)
                result["transport_and_grid_wall_s"] = time.perf_counter() - t0
                _atomic(
                    snapshot,
                    {
                        "config": config,
                        "digest": stamp["code_digest"],
                        "tables_digest": stamp.get("code_tables_digest"),
                        "case": case,
                        "transport": _host(transport),
                    },
                    binary=True,
                )
            if "fingerprint" in result:
                require_identical_segments(result["fingerprint"], ladder.fingerprint)
            result.update(
                fingerprint=ladder.fingerprint,
                dtype=np.dtype(REAL).name,
                physical_electrons=physical_bunch_electrons(case),
                incident_samples=transport["Ne_lines"],
                case={
                    key: case.get(key)
                    for key in (
                        "crystal",
                        "hkl_list",
                        "E0_keV",
                        "thickness_ang",
                        "seed",
                        "bunch_charge_pc",
                        "bunch_length_fs",
                        "longitudinal_distribution",
                        "beam_fwhm_mm",
                        "crystal_width_mm",
                        "crystal_height_mm",
                    )
                },
            )
            record = transport.get("diagnostic_grid") or {}
            result["auto_record"] = {
                key: record.get(key)
                for key in (
                    "num",
                    "start_eV",
                    "stop_eV",
                    "coherent_windows",
                    "min_spacing_eV",
                )
            }
            persist()
            automatic = np.asarray(transport["E_grid"], dtype=float)
            if args.mode == "smoke":
                t0 = time.perf_counter()
                spectra = runner._spectrum_case(case, transport)
                energy = np.asarray(spectra["E_grid"], dtype=float)
                for name in ("spec", "spec_coherent"):
                    density = np.asarray(_to_cpu(spectra[name]), dtype=float)
                    if np.any(~np.isfinite(density)) or np.any(density < 0):
                        raise AssertionError(f"invalid {name}")
                    result[name + "_yield"] = float(np.trapezoid(density, energy))
                result["spectrum_wall_s"] = time.perf_counter() - t0
            else:
                plan = record["window_plan"]
                rows = [row for row in record["coherent_windows"]["rows"] if row.get("points")]
                # Rows take the all-electron step only below their switch energy.
                finest = min(
                    min(row["step_electron_eV"], row["step_all_eV"])
                    if row.get("electron_step_from_eV") is None
                    or row["electron_step_from_eV"] > row["window_eV"][0]
                    else row["step_electron_eV"]
                    for row in rows
                )
                reference_step = finest / args.reference_divisor
                for name in ("auto", "finer", "reference"):
                    if name in result["evals"]:
                        continue
                    if name == "auto":
                        energy = automatic
                    elif name == "finer":
                        seeds = [FeatureSeed(**seed) for seed in plan["seeds"]]
                        seeds = [
                            FeatureSeed(**{**seed.payload(), "spacing_eV": seed.spacing_eV / 2})
                            if seed.source == "pxr-coherent"
                            else seed
                            for seed in seeds
                        ]
                        refined_plan = build_window_plan(
                            plan["start_eV"],
                            plan["stop_eV"],
                            plan["backbone_spacing_eV"],
                            seeds,
                        )
                        if refined_plan.num > args.reference_points:
                            raise RuntimeError(
                                f"finer axis needs {refined_plan.num} points above its explicit budget"
                            )
                        energy = refined_plan.coordinates()
                    else:
                        points = int(np.ceil((automatic[-1] - automatic[0]) / reference_step)) + 1
                        if points > args.reference_points:
                            raise RuntimeError(
                                f"reference needs {points} points above its explicit budget"
                            )
                        energy = np.linspace(automatic[0], automatic[-1], points)
                    t0 = time.perf_counter()
                    print("evaluating", name, int(energy.size), "coordinates", flush=True)
                    density = ladder.lines(energy)
                    observables = spectrum_observables(
                        energy, density, np.zeros_like(energy), detectors={}
                    )
                    measured = {
                        key: float(observables[key]) for key in ("yield", "centroid_eV", "fwhm_eV")
                    }
                    if any(not np.isfinite(value) or value <= 0 for value in measured.values()):
                        raise ArithmeticError(f"unresolved {name} observables: {measured}")
                    result["evals"][name] = {
                        "num": int(energy.size),
                        "wall_s": time.perf_counter() - t0,
                        **measured,
                        "step_eV": float((automatic[-1] - automatic[0]) / (energy.size - 1))
                        if name == "reference"
                        else None,
                    }
                    persist()
                    print(name, result["evals"][name], flush=True)
                    BACKEND.release_memory()
                    if (
                        name != "reference"
                        and time.perf_counter() - started >= args.max_minutes * 60
                    ):
                        result["state"] = "between-rungs"
                        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        persist()
                        return 75
                reference = result["evals"]["reference"]
                for name in ("auto", "finer"):
                    evaluated = result["evals"][name]
                    for key in ("yield", "centroid_eV", "fwhm_eV"):
                        error = abs(evaluated[key] - reference[key]) / abs(reference[key])
                        evaluated[key + "_rel"] = error
                        if error > 1e-3:
                            raise AssertionError(
                                f"{name} {key} relative error {error} exceeds 1e-3"
                            )
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.pop("error", None)
        result.pop("sampling_diagnostics", None)
        result["state"] = "done"
        persist()
        print("done", output, flush=True)
        return 0
    except Exception as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        result.pop("sampling_diagnostics", None)
        if isinstance(error, CoherentSamplingError) and error.diagnostics is not None:
            result["sampling_diagnostics"] = error.diagnostics
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ladder", "smoke"), required=True)
    parser.add_argument(
        "--energy",
        type=float,
        choices=(30.0, 60.0),
        default=30.0,
        help="ladder mode only; smoke mode always runs 60 keV (the value is still logged in config)",
    )
    parser.add_argument("--charge-pc", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=20000000)
    parser.add_argument("--reference-points", type=int, default=40000000)
    parser.add_argument("--reference-divisor", type=float, default=3.0)
    parser.add_argument("--max-minutes", type=float, default=10.0)
    parser.add_argument(
        "--transport-record",
        help="Replay a trusted prior JSON record and adjacent .transport.pkl without new transport",
    )
    parser.add_argument("--out", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
