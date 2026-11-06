"""Directly timed whole-request line-yield FOM of line roulette (#879); remote GPU only.

Each seed runs ``run_case`` once per arm (analog, then every floor), in an order
rotated per seed, after one discarded warmup. A primary's sample is its exact
line mass for analog and ``kept * mass / p`` for roulette, so the sample variance
holds the population and roulette terms. FOM = 1 / (RSE^2 T), pooled over seeds:
RSE from all primaries, T the mean wall time per arm. The screen pass already
returns the analog yield exactly, which is reported as a separate arm cost.
Measurements stay in ignored scratch; no physics-validation claim.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def _ignored(path):
    subprocess.run(["git", "check-ignore", "-q", str(path)], check=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case(args, seed, floor):
    from pyrite._line_roulette import LineRoulette
    from pyrite.energy_grid.bandwidth_check import build_case

    config = dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=args.ne,
    )
    case = dict(build_case(args.material, args.energy, config, seed=seed, resolution="local"))
    if floor is not None:
        case["line_roulette"] = LineRoulette(
            floor=floor, keep_fraction=args.keep_fraction, pilot_electrons=args.pilot
        ).to_dict()
    return case


def _run(args, seed, floor):
    """One whole request; returns wall time, per-primary samples and the yield."""
    from unittest.mock import patch

    from pyrite._backend import _to_cpu
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import cxr_grid
    from pyrite.montecarlo.runner.line_roulette import LineScreen

    screens, masses = [], []
    real_init = LineScreen.__init__
    real_check = cxr_grid.check_cxr_truncation

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        screens.append(self)

    def check(case, audit):
        if "electron_mass" in audit:
            masses.append(np.asarray(_to_cpu(audit["electron_mass"]), dtype=float))
        return real_check(case, audit)

    case = _case(args, seed, floor)
    runner.BACKEND.release_memory()
    start = time.perf_counter()
    with (
        patch.object(LineScreen, "__init__", init),
        patch("pyrite.montecarlo.runner.case_driver.check_cxr_truncation", check),
    ):
        out = runner.run_case(case, record_timing=True, transport_core=args.core)
    wall = time.perf_counter() - start
    energy = np.asarray(out["E_grid"], dtype=float)
    spec = np.asarray(out["spec"], dtype=float)
    if screens:
        screen = screens[-1]
        sample = screen.kept * screen.mass / screen.probability
        record = out["line_roulette"]
    else:
        sample, record = masses[-1], None
    return {
        "wall_s": wall,
        "transport_s": out.get("_t_transport"),
        "spectrum_s": out.get("_t_spectrum"),
        "sample": sample,
        "spec_yield": float(np.trapezoid(spec, energy)),
        "spec": spec,
        "energy": energy,
        "record": record,
    }


def _rse(sample):
    return float(sample.std(ddof=1) / (np.sqrt(sample.size) * abs(sample.mean())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.75, 0.25, 0.1])
    parser.add_argument("--keep-fraction", type=float, default=0.1)
    parser.add_argument("--pilot", type=int, default=128)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--ne", type=int, default=512)
    parser.add_argument("--core", default="per-electron")
    args = parser.parse_args()
    out = _ignored(args.out)
    arms = [None, *args.floors]
    _run(args, args.seeds[0], None)  # discarded warmup: JIT and table population
    report = {"code_sha": args.code_sha, "settings": vars(args) | {"out": str(args.out)}}
    report["runs"] = []
    samples = {arm: [] for arm in arms}
    walls = {arm: [] for arm in arms}
    for index, seed in enumerate(args.seeds):
        order = arms[index % len(arms) :] + arms[: index % len(arms)]
        analog_spec = None
        results = {}
        for arm in order:
            results[arm] = _run(args, seed, arm)
            samples[arm].append(results[arm]["sample"])
            walls[arm].append(results[arm]["wall_s"])
        analog_spec = results[None]["spec"]
        peak = float(analog_spec.max())
        for arm in order:
            run = results[arm]
            report["runs"].append(
                {
                    "seed": seed,
                    "floor": arm,
                    "wall_s": run["wall_s"],
                    "transport_s": run["transport_s"],
                    "spectrum_s": run["spectrum_s"],
                    "rse": _rse(run["sample"]),
                    "spec_yield": run["spec_yield"],
                    "spec_yield_ratio_to_analog": run["spec_yield"] / results[None]["spec_yield"],
                    "max_bin_error_peak": float(np.abs(run["spec"] - analog_spec).max() / peak),
                    "record": run["record"],
                }
            )
        np.savez(
            out / f"seed-{seed}.npz",
            **{f"sample_{arm}": results[arm]["sample"] for arm in arms},
        )
        _summarize(report, arms, samples, walls)
        (out / "fom.json").write_text(json.dumps(report, indent=2, default=str))
    report["complete"] = True
    (out / "fom.json").write_text(json.dumps(report, indent=2, default=str))


def _summarize(report, arms, samples, walls):
    base = None
    summary = {}
    for arm in arms:
        pooled = np.concatenate(samples[arm])
        rse, wall = _rse(pooled), float(np.mean(walls[arm]))
        fom = 1.0 / (rse**2 * wall)
        base = fom if arm is None else base
        summary[str(arm)] = {
            "pooled_rse": rse,
            "mean_wall_s": wall,
            "wall_range_s": [float(min(walls[arm])), float(max(walls[arm]))],
            "fom": fom,
            "fom_gain": fom / base,
            "max_share": float(pooled.max() / pooled.sum()),
            "ess": float(pooled.sum() ** 2 / (pooled @ pooled)),
        }
    report["summary"] = summary


if __name__ == "__main__":
    main()
