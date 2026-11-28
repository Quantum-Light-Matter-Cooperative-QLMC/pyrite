"""Phase IIc: fixed-grid history-fiber retained/live/HDF5 replay parity.

Run from the checkout with ``python checks/fiber_history_replay.py``. A
separate artifact is written under --output; no catalog/profile is changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from fiber_randomized_rolls import _case, _sync

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
COUNTS = ("eta", "hit_frac", "n_segments")


def comparison(reference, result, *, peak_tol, mass_tol):
    """Full spectra and scalar records; no smoothing or scale fitting."""
    for key in ("E_grid", "E_grid_brem"):
        np.testing.assert_array_equal(result[key], reference[key], err_msg=key)
    metrics = {}
    for key in SPECTRA:
        a = np.asarray(reference[key], dtype=np.float64)
        b = np.asarray(result[key], dtype=np.float64)
        assert a.shape == b.shape, key
        assert np.isfinite(a).all() and np.isfinite(b).all(), key
        delta = b - a
        peak = float(np.max(np.abs(a)))
        mass = float(np.sum(np.abs(a)))
        if mass == 0.0:
            np.testing.assert_array_equal(b, a, err_msg=key)
            metrics[key] = {"mass_l1": 0.0, "peak_linf": 0.0, "relative_yield_error": 0.0}
            continue
        metrics[key] = {
            "mass_l1": float(np.sum(np.abs(delta)) / mass),
            "peak_linf": float(np.max(np.abs(delta)) / peak),
            "relative_yield_error": float(np.sum(delta) / mass),
        }
        assert metrics[key]["mass_l1"] <= mass_tol, (key, metrics[key])
        assert metrics[key]["peak_linf"] <= peak_tol, (key, metrics[key])
    assert np.max(reference["spec"]) > 0.0, "empty CXR parity is not evidence"
    for key in COUNTS:
        assert result[key] == reference[key], key
    assert result["brem_lattice_share"] == reference["brem_lattice_share"]
    return metrics


def timed(fn):
    _sync()
    start = perf_counter()
    value = fn()
    _sync()
    return value, perf_counter() - start


def main():
    from pyrite._backend import BACKEND, REAL
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL
    from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel
    from pyrite.montecarlo.trajectories import (
        TrajectoryArtifactError,
        case_digest,
        read_trajectory_artifact,
        write_trajectory_artifact,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--histories", type=int, default=64)
    p.add_argument("--brem-histories", type=int, default=96)
    p.add_argument("--blocks", type=int, nargs="+", default=[17, 41])
    p.add_argument("--replay-rows", type=int, nargs="+", default=[51, 119])
    p.add_argument("--energy", type=float, default=30)
    p.add_argument("--thick-nm", type=float, default=500)
    p.add_argument("--tilt", type=float, default=15)
    p.add_argument("--tilt-azim", type=float, default=180)
    p.add_argument("--theta", type=float, default=119)
    p.add_argument("--cutoff", type=float, default=4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--transport-seed", type=int, default=101)
    p.add_argument("--mosaic", action="store_true")
    p.add_argument("--joint-mosaic", action="store_true", help="sample tilt and roll per primary")
    p.add_argument("--mosaic-nodes", type=int, default=3)
    p.add_argument("--energy-start", type=float, default=100)
    p.add_argument("--energy-max", type=float, default=4000)
    p.add_argument("--step", type=float, default=20)
    p.add_argument("--peak-tol", type=float)
    p.add_argument("--mass-tol", type=float)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if min(a.histories, a.brem_histories, a.batch, a.mosaic_nodes, *a.blocks, *a.replay_rows) < 1:
        p.error("positive counts required")
    if not 10 <= a.energy_start < a.energy_max <= 1000 * a.energy or a.step <= 0:
        p.error("invalid energy grid")
    a.reference = 2  # the legacy quadrature count is immaterial to history mode
    a.mosaic = a.mosaic or a.joint_mosaic
    a.mosaic_route = "mc" if a.mosaic else "auto"
    grid = np.arange(a.energy_start, a.energy_max, a.step)
    if grid.size < 2:
        p.error("need at least two photon bins")
    case = _case(a, grid)
    old = FiberCXRModel.from_case_record(case["cxr_fiber"])
    model = replace(old, numerics=replace(old.numerics, roll_sampling="history"))
    case = dict(case, cxr_fiber=model.case_record(), Ne_brem=a.brem_histories)
    if a.joint_mosaic:
        from pyrite.montecarlo.runner.fiber_emission import validate_fiber_case

        case["mosaic_mc_sampling"] = "history"
        validate_fiber_case(case)
    for live in (False, True):
        assert streaming_refusal(case, live=live) is None

    # Different GPU reductions need not be bitwise identical. Keep these
    # explicit engineering gates separate from the exact CPU regression tests.
    fp32 = np.dtype(REAL) == np.dtype(np.float32)
    peak_tol = (
        a.peak_tol if a.peak_tol is not None else max(FOURIER_RESIDUAL_TOL, 3e-4 if fp32 else 1e-8)
    )
    mass_tol = a.mass_tol if a.mass_tol is not None else (3e-4 if fp32 else 1e-5)
    if min(peak_tol, mass_tol) <= 0 or not np.isfinite([peak_tol, mass_tol]).all():
        p.error("positive finite tolerances required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(str(a.output) + ".h5")
    if not a.overwrite and artifact_path.exists():
        p.error(f"artifact already exists: {artifact_path}; pass --overwrite")
    record = {
        "settings": vars(a),
        "case_digest": case_digest(case),
        "fiber": case["cxr_fiber"],
        "backend": type(BACKEND).__name__,
        "dtype": np.dtype(REAL).name,
        "peak_tol": peak_tol,
        "mass_tol": mass_tol,
        "arms": {},
    }
    spectra = {}
    transport, record["transport_s"] = timed(
        lambda: runner._transport_case(case, transport_core="per-electron")
    )
    reference, record["retained_s"] = timed(partial(runner._spectrum_case, case, transport))
    assert int(transport["Ne_lines"]) == a.histories
    assert int(transport["Ne_brem"]) == a.brem_histories
    for key in ("E_grid", "E_grid_brem", *SPECTRA):
        spectra[key + "_retained"] = np.asarray(reference[key])
    inputs = {key: value for key, value in transport.items() if key != "segs"}
    write_trajectory_artifact(
        artifact_path,
        transport["segs"],
        case=case,
        spectrum_inputs=inputs,
        overwrite=a.overwrite,
    )
    del transport

    def arm(name, fn):
        value, seconds = timed(fn)
        metrics = comparison(reference, value, peak_tol=peak_tol, mass_tol=mass_tol)
        record["arms"][name] = {"seconds": seconds, "metrics": metrics}
        for key in SPECTRA:
            spectra[key + "_" + name] = np.asarray(value[key])
        print(json.dumps({"arm": name, "seconds": seconds, "metrics": metrics}), flush=True)

    for block in a.blocks:
        arm(f"live_{block}", partial(run_case_streamed, case, block, transport_core="per-electron"))
    artifact = read_trajectory_artifact(artifact_path)
    assert artifact.case["cxr_fiber"] == case["cxr_fiber"]
    assert artifact.case["seed"] == case["seed"]
    arm("replay_retained", partial(runner.spectrum_from_artifact, artifact))
    del artifact
    for rows in a.replay_rows:
        arm(
            f"replay_{rows}",
            partial(runner.stream_spectrum_from_artifact, artifact_path, max_segments=rows),
        )
    for other in (dict(case, seed=case["seed"] + 1), dict(case, cxr_fiber=old.case_record())):
        try:
            runner.stream_spectrum_from_artifact(artifact_path, case=other)
        except TrajectoryArtifactError as error:
            if "different case" not in str(error):
                raise RuntimeError("Unexpected artifact refusal") from error
        else:
            raise AssertionError("artifact accepted a different roll seed or estimator")
    record["identity_refusals"] = True
    record["passed"] = True
    Path(str(a.output) + ".json").write_text(
        json.dumps(record, indent=2, default=str, allow_nan=False) + "\n"
    )
    np.savez_compressed(str(a.output) + ".npz", **spectra)


if __name__ == "__main__":
    main()
