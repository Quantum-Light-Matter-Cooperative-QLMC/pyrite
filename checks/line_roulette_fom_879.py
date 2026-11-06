"""Per-bin RSE of the resolved line spectrum at equal wall time (#879); remote GPU only.

Declared before running. Metric: median and q95 over bins at or above 1e-3 of the
analog peak of the per-bin RSE, across independent seeds, against analog at equal
wall time. Every arm scores one frozen grid, so bins align across seeds and no
arm pays a grid pass. Every arm scores the same continuum population, so roulette
arms gain only from line scoring. Each roulette arm's primaries are fitted from two
timed calibration runs to match the analog batch wall time.

Each seed's rare histories dominate its batch, so K batches give only a coarse RSE.
Measurements stay in ignored scratch; no physics-validation claim.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from roulette879_analysis import bin_rse_summary, equal_wall_electrons


def _case(args, grid, seed, electrons, arm):
    from pyrite._energy_grid_encoding import encode_energy_grid
    from pyrite._line_roulette import LineRoulette
    from pyrite.energy_grid.bandwidth_check import build_case

    config = dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=electrons,
    )
    built = build_case(args.material, args.energy, config, seed=seed, resolution="local")
    case = {key: value for key, value in built.items() if key != "line_grid_policy"}
    encoded = encode_energy_grid(grid)
    case.update(E_grid=encoded, E_grid_line=encoded, Ne=electrons, Ne_brem=args.brem_electrons)
    if arm is not None:
        keep, floor = arm
        case["line_roulette"] = LineRoulette(
            floor=floor, keep_fraction=keep, pilot_electrons=args.pilot
        ).to_dict()
    return case


def _run(args, grid, seed, electrons, arm):
    from pyrite.montecarlo import runner

    case = _case(args, grid, seed, electrons, arm)
    runner.BACKEND.release_memory()
    start = time.perf_counter()
    out = runner.run_case(case, record_timing=True, transport_core=args.core)
    wall = time.perf_counter() - start
    return {
        "seed": seed,
        "electrons": electrons,
        "wall_s": wall,
        "transport_s": out.get("_t_transport"),
        "spectrum_s": out.get("_t_spectrum"),
        "record": out.get("line_roulette"),
        "spec": np.asarray(out["spec"], dtype=float),
    }


def _label(arm):
    return "analog" if arm is None else f"keep{arm[0]:g}-floor{arm[1]:g}"


def _public(run):
    return {key: value for key, value in run.items() if key != "spec"}


def _summarize(report, arms, runs, grid_size):
    finished = [arm for arm in arms if len(runs[_label(arm)]) >= 2]
    if None not in finished:
        return
    analog = np.array([run["spec"] for run in runs["analog"]])
    reference = analog.mean(axis=0)
    wall = float(np.mean([run["wall_s"] for run in runs["analog"]]))
    baseline = None
    summary = {}
    for arm in finished:
        label = _label(arm)
        batches = np.array([run["spec"] for run in runs[label]])
        arm_wall = float(np.mean([run["wall_s"] for run in runs[label]]))
        row, scaled = bin_rse_summary(batches, reference, arm_wall, wall)
        baseline = scaled if arm is None else baseline
        ratio = scaled / np.where(baseline > 0, baseline, np.inf)
        finite = ratio[ratio > 0]
        row.update(
            electrons=int(runs[label][0]["electrons"]),
            bin_fraction_improved=float(np.mean(scaled < baseline)),
            rse_ratio_to_analog_q5_q50_q95=np.quantile(finite, [0.05, 0.5, 0.95]).tolist(),
            mean_integral_ratio_to_analog=float(batches.mean(axis=0).sum() / reference.sum()),
        )
        summary[label] = row
    report["summary"] = summary
    report["grid_nodes"] = grid_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--grid", type=Path, required=True, help="frozen line grid, .npy, eV")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--calibration-seed", type=int, default=100)
    parser.add_argument("--analog-electrons", type=int, default=2048)
    parser.add_argument("--brem-electrons", type=int, default=512)
    parser.add_argument("--arms", nargs="+", default=["0.1:0.75", "0.1:0.25", "0.1:0.1"])
    parser.add_argument("--pilot", type=int, default=128)
    parser.add_argument("--max-electrons", type=int, default=32768)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--core", default="cuda")
    args = parser.parse_args()
    subprocess.run(["git", "check-ignore", "-q", str(args.out)], check=True)
    args.out.mkdir(parents=True, exist_ok=True)
    grid = np.load(args.grid)
    arms = [None, *(tuple(float(v) for v in text.split(":")) for text in args.arms)]
    report = {
        "code_sha": args.code_sha,
        "settings": {key: str(value) for key, value in vars(args).items()},
        "metric": "per-bin RSE, q50 and q95, bins >= 1e-3 of analog peak, equal wall time",
    }

    def save():
        (args.out / "fom.json").write_text(json.dumps(report, indent=2, default=str))

    n_a = args.analog_electrons
    _run(args, grid, args.calibration_seed, args.pilot, None)  # discarded warmup
    target = _run(args, grid, args.calibration_seed, n_a, None)["wall_s"]
    electrons = {"analog": n_a}
    calibration = {"analog": [[n_a, target]]}
    low = max(args.pilot * 2, n_a // 4)
    for arm in arms[1:]:
        label = _label(arm)
        points = [
            (n, _run(args, grid, args.calibration_seed, n, arm)["wall_s"]) for n in (low, n_a)
        ]
        calibration[label] = points
        electrons[label] = equal_wall_electrons(
            points, target, minimum=n_a, maximum=args.max_electrons
        )
    report.update(calibration=calibration, electrons=electrons, runs={})
    save()
    runs = {_label(arm): [] for arm in arms}
    for index, seed in enumerate(args.seeds):
        order = arms[index % len(arms) :] + arms[: index % len(arms)]
        for arm in order:
            label = _label(arm)
            run = _run(args, grid, seed, electrons[label], arm)
            runs[label].append(run)
            report["runs"].setdefault(label, []).append(_public(run))
        np.savez(
            args.out / f"seed-{seed}.npz", **{_label(a): runs[_label(a)][-1]["spec"] for a in arms}
        )
        _summarize(report, arms, runs, grid.size)
        save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
