"""Per-primary continuum statistics at one electron count (#278 tail diagnosis).

Replica ladders cannot separate a heavy tail from a noisy spread. This transports
one population through the real runner and scores each primary's in-band
continuum integral on the automatic grid, on the case's observation direction.
The mean is the band integral of the production density. Reports the single-run
relative SE, a bootstrap interval for it, higher moments, the share held by the
largest primaries and how the RSE estimate moves with the sample. Moments are
reported, not a stopping rule: a tail can hide beyond the largest sampled primary.
"""

import argparse
import json

import numpy as np
from electron_count_accuracy import explicit_line_grid

from pyrite._backend import _to_cpu
from pyrite.energy_grid import convergence_case as cc
from pyrite.montecarlo import runner

BANDS = {"all": (0.0, np.inf), "le_100keV": (0.0, 100e3), "ge_1MeV": (1e6, np.inf)}


def trapezoid_weights(energy, keep):
    """Trapezoid weights in eV over the contiguous kept nodes."""
    nodes = np.flatnonzero(keep)
    weights = np.zeros(energy.size)
    if nodes.size > 1:
        widths = np.diff(energy[nodes])
        weights[nodes[:-1]] += 0.5 * widths
        weights[nodes[1:]] += 0.5 * widths
    return weights


def rse(values):
    return float(values.std(ddof=1) / np.sqrt(values.size) / values.mean())


def tail_statistics(values, rng, *, resamples=2000):
    """Moments and tail shares of per-primary scores, projected at this sample size."""
    n = values.size
    mean, std = float(values.mean()), float(values.std(ddof=1))
    centered = (values - mean) / std
    ordered = np.sort(values)[::-1]
    boot = [rse(values[rng.integers(0, n, n)]) for _ in range(resamples)]
    subsamples = {
        str(m): rse(values[:m]) for m in (1600, 5000, 10000, 20000) if m <= n and values[:m].any()
    }
    return {
        "mean": mean,
        "std_over_mean": std / mean,
        "rse": rse(values),
        "rse_bootstrap_95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "n_for_rse_1e-3": (std / mean / 1e-3) ** 2,
        "skewness": float(np.mean(centered**3)),
        "excess_kurtosis": float(np.mean(centered**4) - 3.0),
        "zero_fraction": float(np.mean(values == 0.0)),
        "largest_over_mean": float(ordered[0] / mean),
        "top_1_share": float(ordered[0] / values.sum()),
        "top_10_share": float(ordered[:10].sum() / values.sum()),
        "top_1_percent_share": float(ordered[: max(1, n // 100)].sum() / values.sum()),
        "rse_by_sample": subsamples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--energy-kev", type=float, default=29999.0)
    ap.add_argument("--electrons", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--thickness-ang", type=float, default=1e7)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--output", required=True)
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
    # Lines are unused here; the automatic line policy refuses high-energy cases.
    case["E_grid"] = np.array([100.0, 100000.0])
    case = explicit_line_grid(case, 2000)
    tp = runner._transport_case(case, True)
    segments = tp["segs"]
    energy = np.asarray(tp["E_brem"], float)
    electron = np.asarray(_to_cpu(segments["elec_id"]))
    rows = np.bincount(electron, minlength=a.electrons)
    result = {
        "energy_keV": a.energy_kev,
        "electrons": a.electrons,
        "seed": a.seed,
        "brem_nodes": int(energy.size),
        "segments": int(electron.size),
        "rows_per_primary": tail_statistics(rows.astype(float), np.random.default_rng(a.seed)),
        "bands": {},
    }
    arrays = {"rows": rows}
    for band, (low, high) in BANDS.items():
        weights = trapezoid_weights(energy, (energy >= low) & (energy <= high))
        if not weights.any():
            continue
        values = np.zeros(a.electrons)
        for block in runner.iter_electron_blocks(segments, a.blocks):
            values += np.asarray(
                _to_cpu(
                    runner._brem_wide_from_segments(
                        block,
                        energy,
                        case,
                        tp["n_hat"],
                        case.get("abs_layers"),
                        groove=tp.get("groove"),
                        Ne=a.electrons,
                        electron_band_weights=weights,
                    )
                ),
                float,
            )
        stats = tail_statistics(values, np.random.default_rng(a.seed + 1))
        stats["correlation_with_rows"] = float(np.corrcoef(values, rows)[0, 1])
        result["bands"][band] = stats
        arrays[band] = values
    np.savez_compressed(a.output + ".npz", **arrays)
    with open(a.output, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result)[:1500])


if __name__ == "__main__":
    main()
