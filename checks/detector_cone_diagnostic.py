"""Replay #203's dominant h-BN history with copy and line attribution.

Remote NVIDIA only. Measurement instrument; no estimator or transport changes.
Timings exclude attribution but remain diagnostic, not a counterbalanced FOM.
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from detector_cone_variance_reduction import _host, _settings, _stats, build


def line_geometry(lines, n_hat, half_angle):
    """Directions of captured lines against fixed and local 1/gamma cones."""
    if not lines:
        return {}
    velocity = np.column_stack([lines[k] for k in ("vx", "vy", "vz")])
    norm = np.linalg.norm(velocity, axis=1)
    theta = np.arccos(np.clip((velocity @ n_hat) / norm, -1.0, 1.0))
    gamma = np.asarray(lines["gamma"])
    return {
        "theta_rad": theta,
        "theta_over_fixed_cone": theta / half_angle,
        "theta_times_local_gamma": theta * gamma,
        "electron_energy_keV": (gamma - 1.0) * 511.0,
    }


def attribute(case, tp, ne, half_angle, top):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.cxr._attribution import merge_cxr_attribution

    band = case["line_grid_policy"]["bandwidth"]
    start, stop = band["start_eV"], band["stop_eV"]
    audit = {
        "start_eV": start,
        "stop_eV": stop,
        "n_electrons": ne,
        "attribute": {"top": top, "stop_eV": stop},
    }
    runner._cxr_for_segments(
        tp["segs"],
        np.array([start, stop]),
        case,
        tp["n_hat"],
        case.get("abs_layers"),
        tp.get("groove"),
        coherent=False,
        Ne=ne,
        truncation_audit=audit,
    )
    merged = merge_cxr_attribution(audit.get("attribution", []), top)
    lines = merged["lines_by_mass"]
    lines.update(line_geometry(lines, tp["n_hat"], half_angle))
    return {"lines": lines, "electron_mass": _host(audit.get("electron_mass", np.zeros(ne)))}


def primary_trace(segs, primary, n_hat, half_angle):
    ids = np.asarray(segs["elec_id"])
    select = ids == primary
    fields = (
        "r_mid",
        "v_hat",
        "E_keV",
        "E_repr_keV",
        "E_end_keV",
        "L_ang",
        "flight_id",
        "substep_id",
        "event_kind",
        "t_mid_ang",
    )
    trace = {k: np.asarray(segs[k])[select] for k in fields if segs.get(k) is not None}
    direction = trace["v_hat"]
    theta = np.arccos(np.clip(direction @ n_hat, -1.0, 1.0))
    energy = trace.get("E_repr_keV", trace["E_keV"])
    return {
        "rows": trace,
        "n_rows": int(select.sum()),
        "min_angle_rad": float(theta.min()),
        "rows_inside_fixed_cone": int(np.count_nonzero(theta <= half_angle)),
        "rows_inside_local_cone": int(np.count_nonzero(theta * (1.0 + energy / 511.0) <= 1.0)),
    }


def run(args):
    from pyrite._backend import BACKEND
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver
    from pyrite.montecarlo.runner.adaptive import case_measure
    from pyrite.montecarlo.transport.cone_split import ConeSplit, history_sums

    case = build(5000, 1e7, 10, 100, args.ne, args.seed)
    n_hat = runner._case_geometry(case)[3]
    half_angle = args.cone_factor / (1.0 + 5000 / 511.0)
    cone = ConeSplit(tuple(n_hat), half_angle, args.target_weight)
    original = case_driver.simulate_trajectories
    report = {
        "args": vars(args),
        "half_angle_rad": half_angle,
        "n_hat": n_hat,
        "device": BACKEND.device.name,
        "arms": {},
    }
    for arm in args.order.split(","):

        def transport(*a, arm=arm, **kw):
            return original(*a, **kw, **({"_cone_split": cone} if arm == "split" else {}))

        started = time.perf_counter()
        with (
            patch.object(case_driver, "simulate_trajectories", transport),
            patch.object(
                case_driver, "resolve_cxr_grid", lambda c, s, n, ne, grid, *a: (grid, None)
            ),
        ):
            tp = runner._transport_case(case, transport_core="per-electron")
        transport_s = time.perf_counter() - started
        measure = case_measure(case, _settings(args.ne))
        started = time.perf_counter()
        primary = {k: _host(v) for k, v in measure(0, args.ne, tp["segs"]).items()}
        values = {k: v.copy() for k, v in primary.items()}
        record = tp["segs"].get("cone_split")
        copies = 0 if record is None else record["parent"].size
        copy_values = {}
        if copies:
            copy_values = {k: _host(v) for k, v in measure(0, copies, record["copies"]).items()}
            assert np.all(record["weight"] > 0.0)
            assert np.all(record["v_hat"] @ n_hat >= np.cos(half_angle) - 1e-12)
            assert np.unique(record["stream_keys"]).size == copies
            for key, scores in copy_values.items():
                values[key] = history_sums(
                    primary[key], scores, record["parent"], record["weight"], args.ne
                )
                reference = primary[key].copy()
                np.add.at(reference, record["parent"], record["weight"] * scores)
                np.testing.assert_allclose(values[key], reference, rtol=1e-13, atol=0)
        measure_s = time.perf_counter() - started
        row = {
            "transport_s": transport_s,
            "measure_s": measure_s,
            "copies": int(copies),
            "primary_values": primary,
            "copy_values": copy_values,
            "history_values": values,
            "statistics": {k: _stats(v) for k, v in values.items()},
            "trace": primary_trace(tp["segs"], args.primary, n_hat, half_angle),
        }
        if record is not None:
            row["copy_record"] = {k: v for k, v in record.items() if k != "copies"}
        started = time.perf_counter()
        row["attribution"] = attribute(case, tp, args.ne, half_angle, args.top)
        row["attribution_s"] = time.perf_counter() - started
        np.testing.assert_allclose(
            row["attribution"]["electron_mass"], primary["cxr"], rtol=2e-6, atol=0
        )
        if copies:
            copy_tp = {**tp, "segs": record["copies"]}
            row["copy_attribution"] = attribute(case, copy_tp, copies, half_angle, args.top)
            np.testing.assert_allclose(
                row["copy_attribution"]["electron_mass"], copy_values["cxr"], rtol=2e-6, atol=0
            )
        report["arms"][arm] = row
        # Preserve each complete arm even if a later diagnostic check fails.
        Path(args.out).write_text(json.dumps(report, default=lambda x: x.tolist(), indent=2))
        print(
            json.dumps(
                {
                    "arm": arm,
                    "copies": copies,
                    "statistics": row["statistics"],
                    "trace": {k: v for k, v in row["trace"].items() if k != "rows"},
                }
            ),
            flush=True,
        )
        del tp, record
        BACKEND.release_memory()
    a = report["arms"]["analog"]["trace"]["rows"]
    b = report["arms"]["split"]["trace"]["rows"]
    report["dominant_primary_prefix_equal"] = all(
        a[k].shape[0] >= v.shape[0] and np.array_equal(a[k][: v.shape[0]], v) for k, v in b.items()
    )
    Path(args.out).write_text(json.dumps(report, default=lambda x: x.tolist(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ne", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary", type=int, default=159)
    parser.add_argument("--cone-factor", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=0.001)
    parser.add_argument("--top", type=int, default=256)
    parser.add_argument("--order", choices=("analog,split", "split,analog"), default="analog,split")
    parser.add_argument("--out", required=True)
    run(parser.parse_args())
