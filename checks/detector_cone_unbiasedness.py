"""Detector-cone splitting unbiasedness gate (#203); run on a remote GPU box only.

Analog and split per-electron CPU transports of one case on identical seeds.
Each primary plus its weighted copies is ONE sample. Per history it scores the
full analytic CXR line mass and bremsstrahlung band (production monitor) and
the characteristic yield integrated over a fixed photon window (production
characteristic scorer on that history's rows). The gate compares the
split and analog means of each tally: pooled over all histories, paired by
seed, and with a seed-level SE that is robust to within-seed correlation.
Choose a case where analog sampling resolves the cone population, otherwise
the analog SE omits an unsampled tail. Measurement only; no ledger verdict.
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from detector_cone_variance_reduction import _host, _settings, _stats, build


def transport(case, cone):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver

    original = case_driver.simulate_trajectories

    def simulate(*a, **kw):
        return original(*a, **kw, **({"_cone_split": cone} if cone is not None else {}))

    with (
        patch.object(case_driver, "simulate_trajectories", simulate),
        patch.object(case_driver, "resolve_cxr_grid", lambda c, s, n, ne, grid, *a: (grid, None)),
    ):
        return runner._transport_case(case, transport_core="per-electron")


def characteristic(case, segs, count, n_hat, grid, cache):
    """Per-history characteristic yield (photons/sr per history) in ``grid``'s window."""
    from pyrite.montecarlo import runner

    if int(segs.get("n_layers", 1)) != 1:
        raise ValueError("per-history characteristic supports single-layer cases only")
    ids = _host(segs["elec_id"]).astype(np.int64)
    rows = ids.size
    host = {
        k: _host(v)
        for k, v in segs.items()
        if getattr(v, "ndim", 0) >= 1 and v.shape[0] == rows and k != "copies"
    }
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(count + 1))
    width = np.diff(grid).mean()

    def score(sub, limit):
        spec = runner.mc_characteristic_spectrum(
            sub,
            grid,
            composition=case["composition"],
            n_hat=n_hat,
            layers=case.get("abs_layers"),
            electron_limit=limit,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
            ionization_model=case.get("shell_ionization_model", "eedl"),
            _table_cache=cache,
        )
        return float(np.sum(spec) * width)

    values = np.zeros(count)
    for i in range(count):
        select = order[bounds[i] : bounds[i + 1]]
        if select.size == 0:
            continue
        sub = {**segs, **{k: v[select] for k, v in host.items()}}
        sub["elec_id"] = np.zeros(select.size, dtype=ids.dtype)
        values[i] = score(sub, 1)
    # Linearity check: the population score is the mean of the history scores
    # (rtol covers reduced-precision accumulation in the scorer).
    total = score({**segs, **host}, count) * count
    if not np.isclose(values.sum(), total, rtol=1e-5, atol=1e-300):
        raise AssertionError(f"per-history characteristic {values.sum()} != population {total}")
    return values


def arm(case, args, split):
    from pyrite._backend import BACKEND
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.adaptive import case_measure
    from pyrite.montecarlo.transport.cone_split import ConeSplit, history_sums

    n_hat = runner._case_geometry(case)[3]
    gamma = 1.0 + args.energy_keV / 511.0
    cone = None
    if split:
        cone = ConeSplit(
            tuple(n_hat),
            args.cone_factor / gamma,
            args.target_weight,
            args.max_copies_per_electron,
            min_energy_keV=args.min_energy_fraction * args.energy_keV,
            angle_scale=args.angle_scale,
        )
    grid = np.arange(args.char_window_eV[0], args.char_window_eV[1], args.char_step_eV)
    started = time.perf_counter()
    tp = transport(case, cone)
    transport_s = time.perf_counter() - started
    try:
        record = tp["segs"].get("cone_split")
        parent = np.zeros(0, dtype=int) if record is None else record["parent"]
        weight = np.zeros(0) if record is None else record["weight"]
        count = parent.size
        measure = case_measure(case, _settings(args.ne))
        started = time.perf_counter()
        primary = {k: _host(v) for k, v in measure(0, args.ne, tp["segs"]).items()}
        copy = {k: np.zeros(0) for k in primary}
        if count:
            copy = {k: _host(v) for k, v in measure(0, count, record["copies"]).items()}
        cache = {}
        primary["char"] = characteristic(case, tp["segs"], args.ne, n_hat, grid, cache)
        copy["char"] = np.zeros(0)
        if count:
            copy["char"] = characteristic(case, record["copies"], count, n_hat, grid, cache)
        score_s = time.perf_counter() - started
        values = {k: history_sums(v, copy[k], parent, weight, args.ne) for k, v in primary.items()}
        return {
            "split": split,
            "transport_s": transport_s,
            "score_s": score_s,
            "copies": int(count),
            "killed": 0 if record is None else int(np.count_nonzero(record["killed"])),
            "copy_share": {
                k: float(np.sum(weight * copy[k]) / values[k].sum()) if values[k].sum() else 0.0
                for k in values
            },
            "history_values": values,
            "statistics": {k: _stats(v) for k, v in values.items()},
        }
    finally:
        del tp
        BACKEND.release_memory()


def summarize(report):
    rows = report["seeds"]
    out = {}
    for key in ("cxr", "brem", "char"):
        a = np.concatenate([r["analog"]["history_values"][key] for r in rows])
        b = np.concatenate([r["split"]["history_values"][key] for r in rows])
        d = b - a
        seed_a = np.array([np.mean(r["analog"]["history_values"][key]) for r in rows])
        seed_b = np.array([np.mean(r["split"]["history_values"][key]) for r in rows])
        seed_d = seed_b - seed_a
        se_paired = d.std(ddof=1) / np.sqrt(d.size)
        se_seed = seed_d.std(ddof=1) / np.sqrt(seed_d.size)
        out[key] = {
            "analog_mean": float(a.mean()),
            "split_mean": float(b.mean()),
            "ratio": float(b.mean() / a.mean()) if a.mean() else None,
            "paired_z": float(d.mean() / se_paired) if se_paired else None,
            "seed_z": float(seed_d.mean() / se_seed) if se_seed else None,
            "analog": _stats(a),
            "split": _stats(b),
        }
    report["summary"] = out


def run(args):
    from pyrite._backend import BACKEND

    report = {"args": vars(args), "device": BACKEND.device.name, "seeds": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, default=lambda x: x.tolist(), indent=2))

    def case(ne, seed):
        return build(
            args.energy_keV, args.thickness_ang, args.tilt, args.azimuth, ne, seed, args.material
        )

    try:
        warm = argparse.Namespace(**{**vars(args), "ne": 4})
        for split in (False, True):
            arm(case(4, args.seeds[0]), warm, split)
        for index, seed in enumerate(args.seeds):
            order = (False, True) if index % 2 == 0 else (True, False)
            row = {"seed": seed, "order": ["split" if s else "analog" for s in order]}
            for split in order:
                row["split" if split else "analog"] = arm(case(args.ne, seed), args, split)
            report["seeds"].append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "copies": row["split"]["copies"],
                        "means": {
                            k: [row[a]["statistics"][k]["mean"] for a in ("analog", "split")]
                            for k in ("cxr", "brem", "char")
                        },
                    }
                ),
                flush=True,
            )
            save()
        summarize(report)
        save()
        print(json.dumps(report["summary"], default=float), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy-keV", type=float, default=1000.0)
    parser.add_argument("--thickness-ang", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--ne", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cone-factor", type=float, default=2.0)
    parser.add_argument("--target-weight", type=float, default=0.01)
    parser.add_argument("--min-energy-fraction", type=float, default=0.4)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--max-copies-per-electron", type=int, default=1024)
    parser.add_argument("--char-window-eV", type=float, nargs=2, default=[100.0, 600.0])
    parser.add_argument("--char-step-eV", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.ne < 2 or len(args.seeds) < 5:
        parser.error("need ne >= 2 and at least five seeds")
    if not 0 < args.cone_factor / (1.0 + args.energy_keV / 511.0) < np.pi / 2:
        parser.error("cone half-angle must lie in (0, pi/2)")
    return args


if __name__ == "__main__":
    run(parse_args())
