"""Per-bin RSE from per-block batch means, roulette vs analog at equal wall (#879).

Remote GPU only. Declared before running. One large roulette run and one analog
run, both on the frozen seed-0 grid, streamed in equal electron blocks. Each
block's line spectrum is saved; ``M`` blocks of a run give ``M`` estimates
``M * b_j`` of its spectrum, whose standard error is the batch-means error. The
analog run's primaries are fitted from two timed calibration runs to match the
roulette run's wall time. Metric: median and q95 over bins at or above 1e-3 of
the analog peak of the per-bin RSE, scaled to equal wall time.

Caveats declared in advance: the roulette pilot fills the first block(s) with
analog scoring, so blocks are not identically distributed (each is still
unbiased); a heavy tail makes batch-means RSE coarse at tens of batches. Only
the first ``--brem-electrons`` primaries score continuum, equally in both runs.
Measurements stay in ignored scratch; no physics-validation claim.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from line_roulette_fom_879 import _case
from roulette879_analysis import bin_rse_summary, equal_wall_electrons


def _run(args, grid, electrons, arm, label=None):
    """Stream one case; with ``label``, save each block's line spectrum."""
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import streamed_scoring

    case = _case(args, grid, args.seed, electrons, arm)
    runner.BACKEND.release_memory()
    scorer = streamed_scoring.BlockSpectrumScorer
    add, scoring, blocks = scorer.add, [0.0], []
    previous = [np.zeros(grid.size)]

    def timed_add(self, *a, **kw):
        started = time.perf_counter()
        try:
            return add(self, *a, **kw)
        finally:
            scoring[0] += time.perf_counter() - started
            if label is not None and self.totals is not None:
                total = np.asarray(_to_cpu(self.totals[0]), dtype=float)
                blocks.append(total - previous[0])
                previous[0] = total

    scorer.add = timed_add
    try:
        start = time.perf_counter()
        out = streamed_scoring.run_case_streamed(case, args.block, transport_core=args.core)
        wall = time.perf_counter() - start
    finally:
        scorer.add = add
    run = {
        "electrons": electrons,
        "wall_s": wall,
        "spectrum_s": scoring[0],
        "transport_s": wall - scoring[0],
        "record": out.get("line_roulette"),
    }
    if label is not None:
        stack = np.array(blocks)
        np.save(args.out / f"{label}-blocks.npy", stack)
        np.save(args.out / f"{label}-spec.npy", np.asarray(out["spec"], dtype=float))
        run["blocks"] = int(stack.shape[0])
        run["block_sum_vs_spec_max_rel"] = float(
            np.max(np.abs(stack.sum(axis=0) - out["spec"])) / max(np.max(out["spec"]), 1e-300)
        )
    return run, stack if label is not None else None


def _summarize(report, analog, roulette):
    # M blocks of one run: M * b_j are M unbiased estimates of its spectrum.
    estimates = {"analog": analog.shape[0] * analog, "roulette": roulette.shape[0] * roulette}
    reference = analog.sum(axis=0)
    walls = {label: report["runs"][label]["wall_s"] for label in estimates}
    rows, scaled = {}, {}
    for label, values in estimates.items():
        rows[label], scaled[label] = bin_rse_summary(
            values, reference, walls[label], walls["analog"]
        )
    ratio = scaled["roulette"] / scaled["analog"]
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    rows["roulette"].update(
        bin_fraction_improved=float(np.mean(scaled["roulette"] < scaled["analog"])),
        rse_ratio_to_analog_q5_q50_q95=np.quantile(finite, [0.05, 0.5, 0.95]).tolist(),
        integral_ratio_to_analog=float(roulette.sum() / analog.sum()),
        fom_gain_q50=float(np.quantile(finite, 0.5) ** -2),
    )
    keep = reference >= 1e-3 * reference.max()
    for label, values in (("analog", analog), ("roulette", roulette)):
        mass = values.sum(axis=1)
        rows[label]["block_mass_max_share"] = float(mass.max() / mass.sum())
        rows[label]["block_mass_ess"] = float(mass.sum() ** 2 / (mass @ mass))
        rows[label]["support_bins"] = int(keep.sum())
    report["summary"] = rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--grid", type=Path, required=True, help="frozen line grid, .npy, eV")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--roulette-electrons", type=int, default=1 << 20)
    parser.add_argument("--keep", type=float, default=0.05)
    parser.add_argument("--floor", type=float, default=0.02)
    parser.add_argument("--pilot", type=int, default=4096)
    parser.add_argument("--calibration", type=int, nargs=2, default=[4096, 16384])
    parser.add_argument("--max-analog-electrons", type=int, default=1 << 20)
    parser.add_argument("--brem-electrons", type=int, default=512)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--core", default="cuda")
    parser.add_argument("--block", type=int, default=4096)
    args = parser.parse_args()
    if args.roulette_electrons % args.block or any(n % args.block for n in args.calibration):
        raise SystemExit("electron counts must be whole blocks")
    subprocess.run(["git", "check-ignore", "-q", str(args.out)], check=True)
    args.out.mkdir(parents=True, exist_ok=True)
    grid = np.load(args.grid)
    arm = (args.keep, args.floor)
    report = {
        "code_sha": args.code_sha,
        "settings": {key: str(value) for key, value in vars(args).items()},
        "metric": "per-bin batch-means RSE, q50/q95, bins >= 1e-3 of analog peak, equal wall",
        "runs": {},
    }

    def save():
        (args.out / "blocks.json").write_text(json.dumps(report, indent=2, default=str))

    _run(args, grid, args.block, None)  # discarded warmup
    report["runs"]["roulette"], roulette = _run(
        args, grid, args.roulette_electrons, arm, "roulette"
    )
    save()
    target = report["runs"]["roulette"]["wall_s"]
    points = [(n, _run(args, grid, n, None)[0]["wall_s"]) for n in args.calibration]
    fitted = equal_wall_electrons(
        points, target, minimum=2 * args.block, maximum=args.max_analog_electrons
    )
    analog_n = max(2, round(fitted / args.block)) * args.block
    report["calibration"] = {"points": points, "fitted": fitted, "analog_electrons": analog_n}
    save()
    report["runs"]["analog"], analog = _run(args, grid, analog_n, None, "analog")
    _summarize(report, analog, roulette)
    report["complete"] = True
    save()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
