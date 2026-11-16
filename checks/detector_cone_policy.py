"""Bounded #203 cone-policy diagnostic; run on a remote GPU box only.

Compare admitted cone radii, never change transport/scoring physics or RNG.
Each primary plus all copies is ONE sample. Frozen physical bins use the
production incoherent sinc-bin integral; this instrument bypasses production
spectral reduction. Its attribution cost is reported separately and is not a
production-spectrum FOM. Small pilots cannot establish population efficiency,
unbiasedness, convergence, or the required analog count.
"""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np


class LineBins:
    """Consume bounded line chunks, retain only per-parent physical-bin sums."""

    def __init__(self, count, edges):
        self.edges = np.asarray(edges, dtype=float)
        if not np.all(np.isfinite(self.edges)) or np.any(np.diff(self.edges) <= 0):
            raise ValueError("bin edges must be finite and increasing")
        if self.edges.size < 2:
            raise ValueError("need at least two bin edges")
        self.mass = np.zeros(count)
        self.bins = np.zeros((count, self.edges.size - 1))
        self.lines = 0

    def consume(self, audit, resonance, width, weight, electron=None):
        from detector_cone_variance_reduction import _host

        from pyrite.montecarlo.spectrum.cxr._bin_quadrature import sincsq_bin_masses

        if electron is None:
            raise ValueError("line observer requires emitting electron ids")
        # Host transfer and quadrature working sets stay bounded, even when a
        # production launch group contains many lines. No line list is retained.
        for start in range(0, resonance.size, 4096):
            sl = slice(start, start + 4096)
            ids = _host(electron[sl]).astype(np.int64)
            e, a, w = (_host(v[sl]).astype(float) for v in (resonance, width, weight))
            if np.any(ids < 0) or np.any(ids >= self.mass.size):
                raise ValueError("line id outside the declared population")
            np.add.at(self.mass, ids, w * np.pi / a)
            np.add.at(self.bins, ids, w[:, None] * sincsq_bin_masses(a, e, self.edges))
            self.lines += ids.size


def line_bins(case, tp, count, edges):
    """Observe production line factors without retaining or reducing a spectrum."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.cxr import _batched, _per_hkl

    result = LineBins(count, edges)
    # Existing collect mode skips spectral reduction. Replace only its consumer;
    # factors, masks, absorption and emission policy remain production-owned.
    audit = {"collect": [], "start_eV": edges[0], "stop_eV": edges[-1]}
    with ExitStack() as stack:
        for owner in (_batched, _per_hkl):
            stack.enter_context(patch.object(owner, "_accumulate_edge_truncation", result.consume))
        runner._cxr_for_segments(
            tp["segs"],
            np.asarray([edges[0], edges[-1]]),
            case,
            tp["n_hat"],
            case.get("abs_layers"),
            tp.get("groove"),
            coherent=False,
            Ne=count,
            truncation_audit=audit,
        )
    return result


def targeting(analog, primary, copies, parent, weight, ordinary_share):
    """Attribute added mass to parents below a declared analog mass share."""
    increment = np.zeros_like(analog)
    np.add.at(increment, parent, weight * copies)
    total = analog.sum()
    ordinary = analog <= ordinary_share * total
    added = increment.sum()
    return {
        "ordinary_definition": f"analog parent mass <= {ordinary_share} of analog total",
        "ordinary_parent_ids": np.flatnonzero(ordinary),
        "parents_with_copies": np.unique(parent),
        "productive_copy_parents": np.flatnonzero(increment > 0),
        "productive_ordinary_parents": np.flatnonzero(ordinary & (increment > 0)),
        "ordinary_copy_mass_share": float(increment[ordinary].sum() / added) if added else None,
        "copy_mass_by_parent": increment,
        "primary_change_by_parent": primary - analog,
        "net_change_by_parent": primary + increment - analog,
        "multiplicity_by_parent": np.bincount(parent, minlength=analog.size),
    }


def covered(observed, monitor):
    """Observer mass is the in-window subset of the monitor's full line mass.

    Reflection selection follows the window, so Doppler-boosted lines above its
    top edge are absent from the observer but present in the monitor. Require
    observer <= monitor and report the covered fraction.
    """
    observed, monitor = np.asarray(observed, float), np.asarray(monitor, float)
    if np.any(observed > monitor * (1 + 2e-6) + np.finfo(float).tiny):
        raise AssertionError("line observer exceeds the monitor's full line mass")
    fraction = np.divide(observed, monitor, out=np.ones_like(monitor), where=monitor > 0)
    return {"fraction": fraction, "mass_fraction": float(observed.sum() / monitor.sum())}


def min_local_u(segs, n_hat, count):
    """Smallest gamma*theta to the detector direction along each primary."""
    from detector_cone_variance_reduction import _host

    ids = _host(segs["elec_id"]).astype(np.int64)
    direction = _host(segs["v_hat"]).astype(float)
    energy = _host(segs.get("E_repr_keV", segs["E_keV"])).astype(float)
    theta = np.arccos(np.clip(direction @ np.asarray(n_hat, dtype=float), -1.0, 1.0))
    u = np.full(count, np.inf)
    np.minimum.at(u, ids, theta * (1.0 + energy / 511.0))
    return u


def arm(case, args, factor):
    from detector_cone_variance_reduction import _host, _settings, _stats

    from pyrite._backend import BACKEND
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver
    from pyrite.montecarlo.runner.adaptive import case_measure
    from pyrite.montecarlo.transport.cone_split import ConeSplit, _ConeSplitPass, history_sums

    n_hat = runner._case_geometry(case)[3]
    half_angle = factor / (1.0 + 5000 / 511.0)
    cone = None
    if factor:
        cone = ConeSplit(
            tuple(n_hat),
            half_angle,
            args.target_weight,
            args.max_copies_per_electron,
            min_energy_keV=args.min_energy_keV,
            angle_scale=args.angle_scale,
        )
    original = case_driver.simulate_trajectories
    collect = _ConeSplitPass.collect

    def capped_collect(self, *a, **kw):
        record = collect(self, *a, **kw)
        if record["parent"].size > args.max_copies:
            raise RuntimeError("copy budget exceeded before launching copy generation")
        return record

    def transport(*a, **kw):
        return original(*a, **kw, **({"_cone_split": cone} if factor else {}))

    started = time.perf_counter()
    with (
        patch.object(case_driver, "simulate_trajectories", transport),
        patch.object(_ConeSplitPass, "collect", capped_collect),
        patch.object(case_driver, "resolve_cxr_grid", lambda c, s, n, ne, grid, *a: (grid, None)),
    ):
        tp = runner._transport_case(case, transport_core="per-electron")
    transport_s = time.perf_counter() - started
    try:
        record = tp["segs"].get("cone_split")
        parent = np.zeros(0, dtype=int) if record is None else record["parent"]
        weight = np.zeros(0) if record is None else record["weight"]
        count = parent.size
        rows = tp["segs"]["L_ang"].size
        if count:
            rows += record["copies"]["L_ang"].size
            assert np.all(weight > 0)
            assert np.all(record["v_hat"] @ n_hat >= np.cos(half_angle) - 1e-12)
            assert np.unique(record["stream_keys"]).size == count
        if rows > args.max_rows:
            raise RuntimeError("segment budget exceeded before scoring")
        measure = case_measure(case, _settings(args.ne))
        started = time.perf_counter()
        primary = {k: _host(v) for k, v in measure(0, args.ne, tp["segs"]).items()}
        copy = {k: np.zeros(0) for k in primary}
        if count:
            copy = {k: _host(v) for k, v in measure(0, count, record["copies"]).items()}
        values = {k: history_sums(v, copy[k], parent, weight, args.ne) for k, v in primary.items()}
        monitor_s = time.perf_counter() - started
        started = time.perf_counter()
        bins = line_bins(case, tp, args.ne, args.edges_eV)
        observed_coverage = {"primary": covered(bins.mass, primary["cxr"])}
        copy_bins = np.zeros((0, bins.bins.shape[1]))
        if count:
            observed = line_bins(case, {**tp, "segs": record["copies"]}, count, args.edges_eV)
            observed_coverage["copy"] = covered(observed.mass, copy["cxr"])
            copy_bins = observed.bins
        total_bins = bins.bins.copy()
        np.add.at(total_bins, parent, weight[:, None] * copy_bins)
        bin_s = time.perf_counter() - started
        # The finite window deliberately excludes tails; record their mass.
        tails = values["cxr"] - total_bins.sum(axis=1)
        if np.any(tails < -2e-6 * np.maximum(values["cxr"], np.finfo(float).tiny)):
            raise AssertionError("physical bins exceed full analytic line mass")
        return {
            "factor": factor,
            "half_angle_rad": half_angle if factor else None,
            "transport_s": transport_s,
            "monitor_s": monitor_s,
            "transport_monitor_s": transport_s + monitor_s,
            "bin_attribution_s": bin_s,
            "rows": int(rows),
            "copies": int(count),
            "trajectory_dtype": str(tp["segs"]["r_mid"].dtype),
            "primary_min_u": min_local_u(tp["segs"], n_hat, args.ne),
            "primary_values": primary,
            "copy_values": copy,
            "history_values": values,
            "statistics": {k: _stats(v) for k, v in values.items()},
            "primary_bin_values": bins.bins,
            "copy_bin_values": copy_bins,
            "history_bin_values": total_bins,
            "bin_statistics": [_stats(v) for v in total_bins.T],
            "band_statistics": _stats(total_bins.sum(axis=1)),
            "outside_window_statistics": _stats(tails),
            "observer_coverage": observed_coverage,
            "copy_record": {}
            if record is None
            else {k: v for k, v in record.items() if k != "copies"},
        }
    finally:
        del tp
        BACKEND.release_memory()


def summarize_cohort(rows, ordinary_share):
    from detector_cone_variance_reduction import _stats

    analog = next(row for row in rows if row["factor"] == 0)
    for row in rows:
        record = row["copy_record"]
        parent = np.asarray(record.get("parent", []), dtype=int)
        weight = np.asarray(record.get("weight", []))
        row["targeting"] = targeting(
            analog["primary_values"]["cxr"],
            row["primary_values"]["cxr"],
            row["copy_values"]["cxr"],
            parent,
            weight,
            ordinary_share,
        )
        difference = row["history_bin_values"] - analog["history_bin_values"]
        row["paired_bin_difference_statistics"] = [_stats(v) for v in difference.T]
        row["paired_bin_difference_se"] = difference.std(axis=0, ddof=1) / np.sqrt(
            difference.shape[0]
        )


def run(args):
    from detector_cone_variance_reduction import build

    from pyrite._backend import BACKEND, REAL

    report = {
        "args": vars(args),
        "device": BACKEND.device.name,
        "scoring_dtype": np.dtype(REAL).name,
        "sample_unit": "one primary plus every weighted copy",
        "limits": "diagnostic physical-bin integrals; no production scoring timing or population FOM",
        "orders": [],
        "cohorts": [],
    }
    out = Path(args.out)
    if not out.parent.is_dir():
        out.parent.mkdir(parents=True)

    def save():
        out.write_text(json.dumps(report, default=lambda x: x.tolist(), indent=2))

    try:
        # Warm every distinct arm, including the line observer, in this process.
        warm = argparse.Namespace(**{**vars(args), "ne": 8})
        for factor in args.factors:
            arm(build(5000, 1e7, 10, 100, warm.ne, args.seeds[0]), warm, factor)
        for index, seed in enumerate(args.seeds):
            case = build(5000, 1e7, 10, 100, args.ne, seed)
            for repeat in range(args.repeats):
                # Alternate arm order across seeds and repeats (counterbalanced timing).
                forward = (index + repeat) % 2 == 0
                order = args.factors if forward else list(reversed(args.factors))
                report["orders"].append({"seed": seed, "repeat": repeat, "factors": order})
                rows = []
                report["cohorts"].append({"seed": seed, "repeat": repeat, "arms": rows})
                for factor in order:
                    row = arm(case, args, factor)
                    rows.append(row)
                    save()  # Preserve completed arms if a later cap/error stops the run.
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "repeat": repeat,
                                "factor": factor,
                                "copies": row["copies"],
                                "statistics": row["statistics"],
                            }
                        ),
                        flush=True,
                    )
                summarize_cohort(rows, args.ordinary_share)
                save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ne", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--factors", type=float, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--target-weight", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--edges-eV", type=float, nargs="+", default=[1000, 2000, 4000, 6000, 10000, 20000, 50000]
    )
    parser.add_argument("--ordinary-share", type=float, default=0.01)
    parser.add_argument("--max-copies", type=int, default=8192)
    parser.add_argument("--max-copies-per-electron", type=int, default=1024)
    parser.add_argument(
        "--min-energy-keV", type=float, default=0.0, help="split only at or above this energy"
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=float("inf"),
        help="u_ref of the angle-scaled roulette (inf: off)",
    )
    parser.add_argument("--max-rows", type=int, default=8000000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.ne < 2 or args.repeats < 1 or args.max_rows < 1:
        parser.error("population, repeats and budgets must be positive; ne >= 2")
    if min(args.max_copies, args.max_copies_per_electron) < 1:
        parser.error("copy budgets must be positive")
    if 0 not in args.factors or len(set(args.factors)) != len(args.factors):
        parser.error("factors must be unique and include analog (0)")
    if not all(np.isfinite(f) and 0 <= f < 16 for f in args.factors):
        parser.error("factors must be finite in [0,16)")
    if not 0 < args.ordinary_share < 1:
        parser.error("ordinary-share must lie in (0,1)")
    LineBins(0, args.edges_eV)
    return args


if __name__ == "__main__":
    run(parse_args())
