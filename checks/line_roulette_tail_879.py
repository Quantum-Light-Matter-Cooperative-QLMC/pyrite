"""How the line-mass tail converges with more primaries (#879); remote GPU only.

Transports whole-electron blocks at one seed and keeps only each primary's line
mass and row count, so memory follows the block size. No spectrum is scored. The
report gives yield RSE, ESS and largest share at each doubling of N, and each
block's measured transport and mass-pass time. Output stays in ignored scratch;
no physics-validation claim.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def _statistics(mass, rows):
    total = float(mass.sum())
    if total <= 0.0 or mass.size < 2:
        return None
    return {
        "n": int(mass.size),
        "mean": float(mass.mean()),
        "rse": float(mass.std(ddof=1) / (np.sqrt(mass.size) * mass.mean())),
        "ess": total**2 / float(mass @ mass),
        "max_share": float(mass.max() / total),
        "rows_per_primary": float(rows.mean()),
    }


def _write(out, args, mass, rows, seen, seconds, band):
    growth = []
    n = args.first_report
    while n <= seen:
        growth.append(_statistics(mass[:n], rows[:n]))
        n *= 2
    if seen and (not growth or growth[-1] is None or growth[-1]["n"] != seen):
        growth.append(_statistics(mass[:seen], rows[:seen]))
    report = {
        "code_sha": args.code_sha,
        "settings": {k: str(v) for k, v in vars(args).items()},
        "band_eV": band,
        "primaries_done": seen,
        "seconds": seconds,
        "growth": growth,
    }
    (out / "tail.json").write_text(json.dumps(report, indent=2))
    np.savez(out / "tail.npz", mass=mass[:seen], rows=rows[:seen])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ne", type=int, default=2**20)
    parser.add_argument("--block", type=int, default=4096)
    parser.add_argument("--first-report", type=int, default=512)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--core", default="per-electron")
    args = parser.parse_args()

    from pyrite._backend import _to_cpu
    from pyrite.energy_grid.bandwidth_check import build_case
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.adaptive import _cxr_band
    from pyrite.montecarlo.runner.artifacts import _case_geometry

    subprocess.run(["git", "check-ignore", "-q", str(args.out)], check=True)
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=args.ne,
    )
    built = dict(build_case(args.material, args.energy, config, seed=args.seed))
    case = {k: v for k, v in built.items() if k != "line_grid_policy"}
    case.update(Ne=args.ne, Ne_brem=args.ne)
    E_grid, _brem, _beam, n_hat, groove = _case_geometry(case)
    band = _cxr_band(built, E_grid)
    axis = np.array(band, dtype=float)
    tables: dict = {}
    mass = np.zeros(args.ne)
    rows = np.zeros(args.ne, dtype=np.int64)
    seconds = {"mass": 0.0, "first_block_mass": None, "transport_blocks": []}
    state = {"seen": 0, "blocks": 0, "last": time.perf_counter()}

    def consume(block):
        started = time.perf_counter()
        seconds["transport_blocks"].append(started - state["last"])
        ids = np.asarray(_to_cpu(block.get("electron_id", block["elec_id"])))
        audit = {
            "start_eV": band[0],
            "stop_eV": band[1],
            "n_electrons": args.ne,
            "mass_only": True,
        }
        with np.errstate(over="ignore", divide="ignore"):
            runner._cxr_for_segments(
                runner._segments_on_device(block),
                axis,
                case,
                n_hat,
                case.get("abs_layers"),
                groove,
                coherent=False,
                Ne=args.ne,
                table_cache=tables,
                truncation_audit=audit,
            )
        present, counts = np.unique(ids, return_counts=True)
        rows[present] = counts
        if "electron_mass" in audit:
            mass[present] = np.asarray(_to_cpu(audit["electron_mass"]), dtype=float)[present]
        elapsed = time.perf_counter() - started
        seconds["mass"] += elapsed
        if seconds["first_block_mass"] is None:
            seconds["first_block_mass"] = elapsed
        state["seen"] = int(present.max()) + 1 if present.size else state["seen"]
        state["blocks"] += 1
        if state["blocks"] % 8 == 0:
            _write(args.out, args, mass, rows, state["seen"], seconds, band)
        state["last"] = time.perf_counter()

    runner._transport_case(
        case,
        transport_core=args.core,
        block_electrons=args.block,
        block_consumer=consume,
        keep_segments_on_device=True,
    )
    _write(args.out, args, mass, rows, args.ne, seconds, band)
    print(json.dumps({"complete": True, "seconds_mass": seconds["mass"]}))


if __name__ == "__main__":
    main()
