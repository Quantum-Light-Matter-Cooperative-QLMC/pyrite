"""End-to-end runtime of one HOPG case through the real runner (#278 item 6).

Runs ``runner._transport_case`` + ``runner._spectrum_with_recovery`` (the
``run_case`` path: electron blocks, pool cap, OOM recovery) once, with either
the automatic refined continuum grid (``--grid automatic``) or the earlier
uniform 25 eV axis (``--grid uniform25``). Lines use the same explicit uniform
grid in both arms (the automatic line policy refuses 29,999 keV), so the
difference isolates the continuum. One arm per process, one JSON out.
"""

import argparse
import json
import time

import numpy as np
from electron_count_accuracy import explicit_line_grid

from pyrite.energy_grid import convergence_case as cc
from pyrite.montecarlo import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("automatic", "uniform25"), required=True)
    ap.add_argument("--energy-kev", type=float, default=29999.0)
    ap.add_argument("--electrons", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--line-points", type=int, default=600000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--thickness-ang", type=float, default=1e7)
    ap.add_argument("--warmup", action="store_true")
    a = ap.parse_args()

    case = cc.build_ladder_case(
        "hopg",
        a.energy_kev,
        45.0,
        180.0,
        thickness_ang=a.thickness_ang,
        n_electrons=a.electrons,
        seed=a.seed,
    )
    case["E_grid"] = np.array([100.0, 100000.0])
    case = explicit_line_grid(case, a.line_points)
    if a.grid == "uniform25":
        case["E_grid_brem"] = np.arange(25.0, a.energy_kev * 1e3 + 25.0, 25.0)
    if a.warmup:
        warm = dict(case, Ne=1, Ne_brem=1)
        warm_tp = runner._transport_case(warm, True)
        runner._spectrum_with_recovery(warm, warm_tp, True)
    t0 = time.perf_counter()
    tp = runner._transport_case(case, True)
    t1 = time.perf_counter()
    out = runner._spectrum_with_recovery(case, tp, True)
    t2 = time.perf_counter()
    Eb = np.asarray(out.get("E_brem", tp["E_brem"]), float)
    np.savez_compressed(a.output + ".npz", energy_eV=Eb, density=np.asarray(out["brem_wide"]))
    rec = dict(
        grid=a.grid,
        thickness_ang=a.thickness_ang,
        warmup=a.warmup,
        energy_keV=a.energy_kev,
        electrons=a.electrons,
        seed=a.seed,
        brem_nodes=int(Eb.size),
        line_nodes=int(np.asarray(tp["E_grid"]).size),
        segments=int(tp["segs"]["L_ang"].shape[0]),
        seconds=dict(transport=t1 - t0, spectrum=t2 - t1, total=t2 - t0),
        output_keys=sorted(out),
        timing={
            k: v
            for k, v in out.items()
            if (k.startswith("_t_") or any(word in k for word in ("mib", "retries", "recovery")))
            and isinstance(v, (dict, int, float))
        },
    )
    for k, v in out.items():
        if k == "brem_wide" and isinstance(v, np.ndarray) and v.shape == Eb.shape:
            rec.setdefault("brem_integral", {})[k] = float(np.trapezoid(v, Eb))
    with open(a.output, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(json.dumps(rec, default=str)[:2000])


if __name__ == "__main__":
    main()
