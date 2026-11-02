"""Bounded FULL production-spectrum roulette gate (#879); remote GPU only.

One fixed population, one production-resolved grid, original primary IDs and
normalization in EVERY call. Per-primary calls capture a reference instrument;
only batched production calls are used for timing comparisons. A post-hoc
spectral reference policy is a diagnostic, never a deployable predictor or a
universal opportunity bound. The cheap regression is fitted
on a disjoint pilot prefix and frozen before held-out prediction. Its pilot
capture cost is reported separately and charged to a one-use cost comparison.

Artifacts, arrays, timing logs and reports must live in ignored scratch paths.
No production estimator, RNG, physics or policy implementation is changed.
Example (lab allocation, after ship checkpoint):
    uv run python checks/line_roulette_spectral_gate_879.py --out scratch/roulette879/gate
"""

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from roulette879_analysis import (
    angle_features,
    band_weights,
    conditional_errors,
    fit_predictor,
    positive_probabilities,
    predict,
    spectral_importance,
    tail_statistics,
)


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _ignored_output(path):
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    path = path.resolve()
    relative = path.relative_to(root)
    if "scratch" not in relative.parts:
        raise ValueError("measurement output must be under an ignored scratch directory")
    subprocess.run(["git", "check-ignore", "-q", "--", str(relative / "probe")], check=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timed(fn):
    from pyrite._backend import BACKEND

    BACKEND.synchronize()
    start = time.perf_counter()
    value = fn()
    BACKEND.synchronize()
    return value, time.perf_counter() - start


def primary_subset(segments, selected):
    """Whole-primary row gather; scalar Ne and ORIGINAL IDs stay unchanged."""
    from pyrite.montecarlo.trajectories import row_fields

    ids = np.asarray(segments.get("electron_id", segments.get("elec_id")))
    rows = np.asarray(selected, dtype=bool)[ids]
    return {key: value[rows] if key in row_fields() else value for key, value in segments.items()}


def _lines(case, tp, segments):
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo import runner

    device = runner._segments_on_device(segments)
    return np.asarray(
        _to_cpu(
            runner._cxr_for_segments(
                device,
                tp["E_grid"],
                case,
                tp["n_hat"],
                case.get("abs_layers"),
                tp.get("groove"),
                coherent=False,
                Ne=tp["Ne_lines"],
                table_cache={},
            )
        ),
        dtype=float,
    )


def stratum_score(segments, probabilities, selected, score):
    """Batched production scoring of each constant-p stratum, then 1/p sum."""
    result = None
    for p in np.unique(probabilities[selected]):
        keep = selected & (probabilities == p)
        part = score(primary_subset(segments, keep)) / p
        result = part if result is None else result + part
    return result


def _difference(actual, expected, widths, tolerance):
    delta = np.abs(actual - expected)
    mass = float(np.abs(expected) @ widths)
    peak = float(np.max(np.abs(expected), initial=0))
    result = {
        "l1_relative": float(delta @ widths) / mass if mass else float(delta.max(initial=0)),
        "max_bin_peak_normalized": float(delta.max(initial=0)) / peak if peak else 0.0,
        "tolerance": tolerance,
    }
    if max(result["l1_relative"], result["max_bin_peak_normalized"]) > tolerance:
        raise ValueError(f"production / per-primary reference parity failed: {result}")
    return result


def _transport(args, out, state):
    from pyrite.energy_grid.bandwidth_check import build_case
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver
    from pyrite.montecarlo.trajectories import (
        TrajectoryCapture,
        artifact_spectrum_inputs,
        read_trajectory_artifact,
    )

    if state.get("artifact"):
        artifact = read_trajectory_artifact(state["artifact"])
        return artifact.case, dict(artifact_spectrum_inputs(artifact), segs=artifact.transport)
    config = dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=args.ne,
    )
    case = build_case(args.material, args.energy, config, seed=args.seed, resolution="local")
    if case.get("coherent_emission") or case.get("secondary_transport"):
        raise ValueError("gate supports the specified incoherent primary-population case only")
    capture = TrajectoryCapture(str(out / "capture"), provenance={"code_sha": state["code_sha"]})
    real_resolve = case_driver.resolve_cxr_grid
    grid_seconds, capture_seconds = [], []
    original_write = TrajectoryCapture.write

    def resolve(*a, **kw):
        value, seconds = _timed(lambda: real_resolve(*a, **kw))
        grid_seconds.append(seconds)
        return value

    def write(self, *a, **kw):
        value, seconds = _timed(lambda: original_write(self, *a, **kw))
        capture_seconds.append(seconds)
        return value

    with (
        patch.object(case_driver, "resolve_cxr_grid", resolve),
        patch.object(TrajectoryCapture, "write", write),
    ):
        tp, seconds = _timed(
            lambda: runner._transport_case(
                case,
                transport_core=args.core,
                trajectory_capture=capture,
            )
        )
    state.update(
        artifact=str(capture.path_for(case)),
        transport_grid_capture_s=seconds,
        grid_resolution_s=sum(grid_seconds),
        artifact_capture_s=sum(capture_seconds),
        transport_excluding_grid_and_capture_s=seconds - sum(grid_seconds) - sum(capture_seconds),
    )
    _json(out / "state.json", state)
    return case, tp


def _full_phase(case, tp):
    """Synchronized inclusive estimator times, avoiding nested slot double-counting."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.emission_adapters import EmissionAdapter

    original = EmissionAdapter.evaluate
    active, seconds = set(), {}

    def evaluate(self, batch, slot, **kw):
        if slot in active:
            return original(self, batch, slot, **kw)
        active.add(slot)
        try:
            result, elapsed = _timed(lambda: original(self, batch, slot, **kw))
            seconds[slot] = seconds.get(slot, 0) + elapsed
            return result
        finally:
            active.remove(slot)

    with patch.object(EmissionAdapter, "evaluate", evaluate):
        result, elapsed = _timed(lambda: runner._spectrum_case(case, tp))
    return result, {"spectrum_s": elapsed, "estimator_s": seconds}


def _capture_primaries(args, out, state, case, tp, widths):
    grid = np.asarray(tp["E_grid"])
    size = args.ne * grid.size * 8
    if grid.size > args.max_nodes or size > args.max_array_gib * 2**30:
        raise ValueError(
            f"full production grid exceeds declared budget: nodes={grid.size}, bytes={size}"
        )
    path = out / "per_primary.npy"
    done = int(state.get("completed_primaries", 0))
    f = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype=np.float64, shape=(args.ne, grid.size)
    )
    elapsed = state.setdefault("per_primary_capture_s", [])
    for primary in range(done, args.ne):
        if time.monotonic() >= args.deadline:
            raise TimeoutError(
                "capture budget reached; state and completed reference rows retained"
            )
        keep = np.arange(args.ne) == primary
        subset = primary_subset(tp["segs"], keep)
        value, seconds = _timed(lambda subset=subset: _lines(case, tp, subset))
        f[primary] = value
        f.flush()
        elapsed.append(seconds)
        state["completed_primaries"] = primary + 1
        _json(out / "state.json", state)
        if primary % 32 == 0:
            print(
                json.dumps(
                    {"primary": primary, "capture_s": sum(elapsed), "mass": float(value @ widths)}
                ),
                flush=True,
            )
    return f


def _measure_policy(args, out, name, case, tp, f, p, rows, widths, screen=None, start=0):
    """Warm and counterbalance all-primary/retained BATCH calls on identical rows."""
    population = np.arange(args.ne) >= start
    analog = np.ones(args.ne)
    saved, measurements = {}, []
    rng = np.random.default_rng(args.resample_seed)

    def score(s):
        return _lines(case, tp, s)

    def evaluate(candidate):
        if candidate:
            probabilities = p if screen is None else screen()
            selected = (rng.random(args.ne) < probabilities) & population
            value = stratum_score(tp["segs"], probabilities, selected, score)
        else:
            probabilities, selected = analog, population
            value = score(tp["segs"] if start == 0 else primary_subset(tp["segs"], population))
        if value is None:
            value = np.zeros(f.shape[1])
        saved.update(value=value, probabilities=probabilities.copy(), selected=selected)
        return value

    for candidate in (False, True):
        evaluate(candidate)  # in-process warmup, same population/grid/policy
    for rep in range(args.timing_repeats):
        for candidate in (False, True) if rep % 2 == 0 else (True, False):
            value, seconds = _timed(lambda candidate=candidate: evaluate(candidate))
            weights = saved["selected"] / saved["probabilities"]
            expected = np.asarray(weights @ f)
            parity = _difference(value, expected, widths, args.parity_tolerance)
            row = {
                "rep": rep,
                "arm": "retained" if candidate else "all",
                "wall_s_including_screen_selection_gather_upload_setup": seconds,
                "retained_primaries": int(saved["selected"].sum()),
                "retained_segment_rows": int(rows[saved["selected"]].sum()),
                "reference_parity": parity,
            }
            measurements.append(row)
            np.save(out / f"{name}-{rep}-{row['arm']}.npy", value)
            np.save(out / f"{name}-{rep}-{row['arm']}-selected.npy", saved["selected"])
            print(json.dumps({"policy": name, **row}), flush=True)
            _json(out / f"{name}-timings.json", measurements)
    return measurements


def run(args):
    from pyrite._backend import BACKEND, REAL, _to_cpu
    from pyrite.montecarlo.runner.oom import _ensure_pool_limit
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_axis

    out = _ignored_output(args.out)
    if BACKEND.name != "cuda":
        raise ValueError(f"remote CUDA gate required; resolved {BACKEND.name}")
    _ensure_pool_limit()
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "deadline")}
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    state_path = out / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "schema": 1,
            "code_sha": sha,
            "settings": settings,
            "backend": BACKEND.name,
            "device": BACKEND.device.name,
            "dtype": np.dtype(REAL).name,
        }
    )
    if (
        state["code_sha"] != sha
        or state["settings"] != settings
        or state["dtype"] != np.dtype(REAL).name
    ):
        raise ValueError(
            "resume requires identical code, settings and dtype; choose new output root"
        )
    print(json.dumps(state), flush=True)
    case, tp = _transport(args, out, state)
    grid = np.asarray(tp["E_grid"], dtype=float)
    edges = np.asarray(_to_cpu(bin_axis(grid)[0]), dtype=float)
    widths = np.diff(edges)
    np.save(out / "grid_eV.npy", grid)
    state.update(
        grid_nodes=grid.size,
        grid_cell_band_eV=[float(edges[0]), float(edges[-1])],
        grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
        line_grid_resolved=tp.get("diagnostic_grid"),
        n_segments=int(tp["segs"]["L_ang"].size),
    )
    _json(state_path, state)
    f = _capture_primaries(args, out, state, case, tp, widths)
    mass = np.asarray(f @ widths)
    representative = (
        int(np.argmax(mass)) == args.dominant_primary and mass.max() / mass.sum() >= 0.5
    )
    state.update(
        reference_tail=tail_statistics(mass),
        dominant_primary=int(np.argmax(mass)),
        known_rare_population_captured=bool(representative),
    )
    _json(state_path, state)
    if not representative:
        raise ValueError("known rare contribution not captured: retain evidence; redesign capture")
    # Full production spectrum warmup and repeats: separate line/continuum/characteristic cost.
    _full_phase(case, tp)
    phases = []
    for _rep in range(args.phase_repeats):
        spectrum, timing = _full_phase(case, tp)
        timing["reference_parity"] = _difference(
            spectrum["spec"], f.sum(axis=0), widths, args.parity_tolerance
        )
        phases.append(timing)
        _json(out / "production-phases.json", phases)
    features, feature_seconds = _timed(lambda: angle_features(tp["segs"], tp["n_hat"], args.ne))
    x, rows = features
    pilot = args.pilot
    model, fit_seconds = _timed(lambda: fit_predictor(x[:pilot], mass[:pilot]))
    predicted = predict(x, model)
    _, threshold = positive_probabilities(predicted[:pilot], args.keep_fraction, args.floor)
    frozen = dict(
        model=model, threshold=threshold, pilot_ids=[0, pilot], heldout_ids=[pilot, args.ne]
    )
    _json(out / "frozen-predictor.json", frozen)

    def screen():
        current, _ = angle_features(tp["segs"], tp["n_hat"], args.ne)
        return positive_probabilities(
            predict(current, model), args.keep_fraction, args.floor, threshold=threshold
        )[0]

    reference_scores = spectral_importance(f, widths)
    policies = {
        "posthoc-spectral-reference": (
            positive_probabilities(reference_scores, args.keep_fraction, args.floor)[0],
            0,
        ),
        "frozen-cheap-heldout": (screen(), pilot),
    }
    boundaries = sorted(
        set(
            [
                float(edges[0]),
                float(edges[-1]),
                *[
                    value
                    for value in (1000.0, 5000.0, 20000.0, 100000.0, 1000000.0)
                    if edges[0] < value < edges[-1]
                ],
            ]
        )
    )
    bands = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    weights = band_weights(edges, bands)
    reports = {}
    for name, (p, start) in policies.items():
        report, variance = conditional_errors(
            f[start:], p[start:], widths, weights, repeats=args.resamples, seed=args.resample_seed
        )
        report.update(
            band_edges_eV=bands,
            sample_primary_ids=[start, args.ne],
            expected_segment_rows=float(p[start:] @ rows[start:]),
            measured_screen_features_s=feature_seconds,
            predictor_fit_s=fit_seconds if start else None,
            pilot_reference_capture_s=sum(state["per_primary_capture_s"][:pilot])
            if start
            else None,
            policy_definition=(
                "Pilot log-total-line-mass regression on angular/path features; "
                "pilot-frozen top-score threshold, positive floor otherwise."
                if start
                else "Post-hoc score sqrt(sum_b width_b * "
                "(f_ib / max(sum_i f_ib, 1e-6 * spectrum_peak))**2); "
                "top-score threshold, positive floor otherwise. "
                "Heuristic reference, not optimized for a spectral error/cost objective."
            ),
            negative_result_scope="This policy and declared conditional spectral metrics only.",
            deployment_claim=False,
        )
        np.save(out / f"{name}-conditional-variance.npy", variance)
        np.save(out / f"{name}-probabilities.npy", p)
        report["timings"] = _measure_policy(
            args,
            out,
            name,
            case,
            tp,
            f,
            p,
            rows,
            widths,
            screen=screen if start else None,
            start=start,
        )
        all_times = [
            r["wall_s_including_screen_selection_gather_upload_setup"]
            for r in report["timings"]
            if r["arm"] == "all"
        ]
        kept_times = [
            r["wall_s_including_screen_selection_gather_upload_setup"]
            for r in report["timings"]
            if r["arm"] == "retained"
        ]
        report["median_batched_line_time_ratio"] = float(
            np.median(all_times) / np.median(kept_times)
        )
        if start:
            report["one_use_time_ratio_charging_pilot_capture"] = float(
                np.median(all_times)
                / (np.median(kept_times) + report["pilot_reference_capture_s"] + fit_seconds)
            )
        reports[name] = report
        _json(out / "conditional-errors.json", reports)
    state.update(
        complete=True,
        production_phases=phases,
        reports=reports,
        host_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        allocator=BACKEND.allocator_stats(),
        limitation=(
            "Conditional on one captured population; rare-event convergence and deployable "
            "FOM unproven. Neither policy provides a universal opportunity bound; "
            "a negative result does not reject roulette generally."
        ),
    )
    _json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--ne", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--core", choices=("per-electron", "cuda"), default="per-electron")
    parser.add_argument("--dominant-primary", type=int, default=159)
    parser.add_argument("--pilot", type=int, default=128)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--keep-fraction", type=float, default=0.1)
    parser.add_argument("--resamples", type=int, default=256)
    parser.add_argument("--resample-seed", type=int, default=879)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--phase-repeats", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=600000)
    parser.add_argument("--max-array-gib", type=float, default=3.0)
    parser.add_argument("--capture-minutes", type=float, default=40.0)
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=5e-5,
        help="maximum L1/peak-normalized error, below the 1%% spectral gate margin",
    )
    args = parser.parse_args()
    if (
        not 5 <= args.pilot < args.ne
        or args.resamples < 2
        or min(args.timing_repeats, args.phase_repeats) < 2
    ):
        parser.error("need disjoint pilot/heldout populations and repeated measurements")
    args.deadline = time.monotonic() + args.capture_minutes * 60
    try:
        run(args)
    except Exception as error:
        out = _ignored_output(args.out)
        _json(out / "stopped.json", {"error": repr(error), "conditional_gate_complete": False})
        raise


if __name__ == "__main__":
    main()
