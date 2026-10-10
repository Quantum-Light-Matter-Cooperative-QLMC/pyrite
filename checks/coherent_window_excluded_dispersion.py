"""Issue 372: window-excluded dispersive power against the certificate.

One transport serves every reflection. Each reflection runs alone with one
mosaic orientation, so its case has one coherent row. The production
reducer is evaluated on a uniform axis over each band the row's windows
omit, at half the row's per-electron window step (plus the far axis end,
so the reducer keeps the same resonating pieces), and ``N_e`` times its
integral is compared with the audit's certified absolute bound for that
side (sampled electrons, no physical charge: the reducer then returns
``[(1 - F) sum_e |S_e|^2 + F |sum_e S_e|^2] / N_e``).
Run ``--case anchor`` on CPU; ``--case thick`` belongs on ``pyrite remote``.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


def _case(name):
    from pyrite.energy_grid import convergence_case as cc

    if name == "anchor":
        case = dict(
            cc.build_ladder_case("hopg", 30.0, 5.0, 45.0, thickness_ang=1e4, n_electrons=3, seed=0)
        )
    else:
        case = dict(
            cc.build_ladder_case("hopg", 30.0, 5.0, 0.0, thickness_ang=1e7, n_electrons=40, seed=7)
        )
    case.update(bunch_length_fs=100.0, mosaic_mc_fwhm_rad=None, mosaic_mc_nodes=1)
    case.pop("bunch_charge_pc", None)
    return case


def run(args):
    from pyrite._line_grid_policy import resolve_line_grid_policy
    from pyrite.energy_grid import convergence_case as cc
    from pyrite.montecarlo.runner.line_grid import resolve_line_grid

    output = Path(args.out)
    snapshot = output.with_suffix(".transport.pkl")
    case = _case(args.case)
    if snapshot.exists():
        with snapshot.open("rb") as stream:
            transport = pickle.load(stream)
        ladder = cc.CaseLadder(case, transport=transport)
    else:
        ladder = cc.CaseLadder(case, transport_core="auto" if args.case == "thick" else "lockstep")
        with snapshot.open("wb") as stream:
            pickle.dump(ladder.transport, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tp = ladder.transport
    bandwidth = case["line_grid_policy"]["bandwidth"]
    start, stop = float(bandwidth["start_eV"]), float(bandwidth["stop_eV"])
    result = {"case": args.case, "start_eV": start, "stop_eV": stop, "rows": []}
    for hkl in case["hkl_list"]:
        row_case = {**case, "hkl_list": [hkl], "coherent_emission": True}
        payload = resolve_line_grid_policy(
            start_eV=start,
            stop_eV=stop,
            per_call={"windows": True, "max_points": args.max_points},
        ).payload()
        row_case["line_grid_policy"] = payload
        _, record = resolve_line_grid(row_case, ladder.segments, tp["n_hat"], tp["Ne_lines"], None)
        summary = record["coherent_windows"]
        rows = [row for row in summary["rows"] if row.get("points")]
        if not rows:
            # The reflection radiates nothing toward this detector.
            result["rows"].append({"hkl": list(hkl), "pieces": 0})
            continue
        if len(rows) != 1:
            raise RuntimeError(f"{hkl}: expected one coherent row, found {len(rows)}")
        (row,) = rows
        (audit,) = summary["dispersion"]["rows"]
        coherent = cc.CaseLadder(row_case, transport=tp, coherent=True)
        lower, upper = row["window_eV"]
        entry = {
            "hkl": list(hkl),
            "window_eV": [lower, upper],
            "band_eV": row["band_eV"],
            "dispersion_widened": row.get("dispersion_widened"),
            "frozen_leak_bound": row["leak_bound"],
            "reference_eV": audit["reference_eV"],
            "sides": {},
        }
        step = 0.5 * row["step_electron_eV"]
        for side, (lo, hi) in (("lower", (start, lower)), ("upper", (upper, stop))):
            if hi <= lo:
                continue
            energy = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
            # The reducer keeps pieces resonating inside the evaluated range;
            # spanning the whole axis keeps the production set.
            other = stop if side == "lower" else start
            grid = np.sort(np.r_[energy, other])
            inside = (grid >= lo) & (grid <= hi)
            t0 = time.perf_counter()
            density = coherent.lines(grid)[inside]
            measured = tp["Ne_lines"] * float(np.trapezoid(density, energy))
            bound = audit["leak_bound"][side] * audit["reference_eV"]
            entry["sides"][side] = {
                "band_eV": [lo, hi],
                "points": int(energy.size),
                "measured_eV": measured,
                "bound_eV": bound,
                "measured_fraction": measured / audit["reference_eV"],
                "bound_fraction": audit["leak_bound"][side],
                "ratio": measured / bound if bound > 0 else None,
                "wall_s": time.perf_counter() - t0,
            }
            print(hkl, side, entry["sides"][side], flush=True)
            if measured > bound:
                raise AssertionError(f"{hkl} {side}: measured {measured} exceeds bound {bound}")
        result["rows"].append(entry)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
    result["state"] = "done"
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("anchor", "thick"), required=True)
    parser.add_argument("--max-points", type=int, default=20000000)
    parser.add_argument("--out", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
