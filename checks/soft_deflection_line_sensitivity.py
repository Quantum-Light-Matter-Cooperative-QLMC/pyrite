"""Line-spectrum sensitivity to the omitted soft inelastic angular deflection.

``checks/soft_inelastic_deflection.py`` sizes the soft inelastic transport
rate ``1/lambda_in,1^(s)`` the shell mode leaves out. This check asks whether
that rate moves the PXR/CBS lines. It transports a Si catalog case once per
seed with ``inelastic_model="shell-soft-hard"`` (``W_c = 50`` eV), then
evaluates the line spectrum on the same segments with three direction sets:

* ``baseline``: the transported ``v_hat``;
* ``row``: each segment tilted by a random small angle with
  ``<theta^2> = k 2 L / (3 lambda_in,1^(s)(E))``, the mean-square deviation
  of a segment's average direction from its start direction under a small-
  angle random walk at the soft rate. This emulates the direction wander
  inside a row that folding the deflection into vertices would not capture;
* ``vertex``: a per-electron cumulative random walk, ``<theta^2> = k 2 L /
  lambda_in,1^(s)`` per row, applied to that row's successors. This emulates
  the extra angular diffusion itself (positions are not re-integrated, so
  it slightly overstates the effect on geometry-sensitive terms).

``k`` scales the soft rate (``k = 1`` is the model; larger ``k`` probes the
response and bounds the conduction-band rate uncertainty). Per seed the
variant-minus-baseline differences are paired: same transport, same grid.
They are compared with the seed-to-seed standard deviation of the baseline,
the Monte Carlo error of one run at this electron count.

Metrics per line of the baseline spectrum (peaks with prominence above 5% of
the maximum): integrated yield, centroid, and rms width in a window around
the peak, plus total line yield. The coherent segment sum (``spec_coherent``)
is evaluated too with ``--coherent``.

Validation: shell-soft-hard-transport

Run (GPU; use the remote box):
  uv run python checks/soft_deflection_line_sensitivity.py --quick
  uv run python checks/soft_deflection_line_sensitivity.py --output REPORT.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from soft_inelastic_deflection import inelastic_transport_rates  # noqa: E402

from pyrite.campaign.config import material_sweep  # noqa: E402
from pyrite.campaign.sweep import build_cases  # noqa: E402
from pyrite.montecarlo import runner  # noqa: E402

MATERIAL = "silicon"
CUTOFF_EV = 50.0
CASES = ((30.0, 1000.0), (30.0, 10000.0), (100.0, 1000.0), (100.0, 10000.0))  # (E0 keV, t A)
TILT_DEG = 30.0
FULL = dict(Ne=20000, seeds=(1, 2, 3, 4, 5, 6), scales=(1.0, 4.0))
QUICK = dict(Ne=300, seeds=(1, 2), scales=(1.0, 4.0))


def _soft_rate_interpolant(e_min_keV, e_max_keV):
    """``1/lambda_in,1^(s)(E)`` [1/Angstrom], log-log interpolated."""
    grid = np.geomspace(max(e_min_keV, 1.0), e_max_keV, 12)
    rates = np.array(
        [inelastic_transport_rates(MATERIAL, e * 1e3, CUTOFF_EV)["soft"][0] for e in grid]
    )
    log_e, log_r = np.log(grid), np.log(rates)
    return lambda e_keV: np.exp(np.interp(np.log(e_keV), log_e, log_r))


def _tilt(v, theta2, rng):
    """Rotate unit rows ``v`` by small angles with mean square ``theta2``."""
    helper = np.where(np.abs(v[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = np.cross(v, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(v, u)
    sigma = np.sqrt(0.5 * theta2)[:, None]
    a, b = rng.normal(size=(2, v.shape[0], 1)) * sigma
    theta = np.hypot(a, b)
    safe = np.where(theta > 0.0, theta, 1.0)
    axis_u, axis_w = a / safe, b / safe
    out = np.cos(theta) * v + np.sin(theta) * (axis_u * u + axis_w * w)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _rodrigues(axis, angle):
    """Rotation matrices about unit ``axis`` rows by ``angle``."""
    k = np.zeros((axis.shape[0], 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * k + (1.0 - c) * (k @ k)


def row_variant(segs, soft_rate, scale, rng):
    theta2 = scale * 2.0 * segs["L_ang"] * soft_rate(segs["E_keV"]) / 3.0
    return _tilt(np.asarray(segs["v_hat"], float), theta2, rng)


def vertex_variant(segs, soft_rate, scale, rng):
    v = np.asarray(segs["v_hat"], float).copy()
    order = np.lexsort((segs["substep_id"], segs["flight_id"], segs["electron_id"]))
    eid = np.asarray(segs["electron_id"])[order]
    start = np.r_[True, eid[1:] != eid[:-1]]
    rank = np.arange(eid.size) - np.maximum.accumulate(np.where(start, np.arange(eid.size), 0))
    _, electron_index = np.unique(eid, return_inverse=True)
    rotation = np.repeat(np.eye(3)[None], electron_index.max() + 1, axis=0)
    increment = scale * 2.0 * segs["L_ang"][order] * soft_rate(segs["E_keV"][order])
    for r in range(int(rank.max()) + 1):
        rows = np.flatnonzero(rank == r)
        e = electron_index[rows]
        new = np.einsum("nij,nj->ni", rotation[e], v[order[rows]])
        v[order[rows]] = new / np.linalg.norm(new, axis=1, keepdims=True)
        # Random axis perpendicular to the realized direction, angle with
        # mean square equal to this row's increment.
        axis = np.cross(v[order[rows]], rng.normal(size=(rows.size, 3)))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle = np.sqrt(increment[rows]) * np.sqrt(rng.chisquare(2, rows.size) / 2.0)
        rotation[e] = _rodrigues(axis, angle) @ rotation[e]
    return v


def line_metrics(E, spec, peaks):
    """Yield, centroid and rms width per baseline peak window, and total."""
    out = {"total": float(np.trapezoid(spec, E))}
    for i, (lo, hi) in enumerate(peaks):
        e, s = E[lo:hi], np.clip(spec[lo:hi], 0.0, None)
        y = float(np.trapezoid(s, e))
        centroid = float(np.trapezoid(s * e, e) / y) if y > 0 else np.nan
        width = float(np.sqrt(np.trapezoid(s * (e - centroid) ** 2, e) / y)) if y > 0 else np.nan
        out[f"peak{i}"] = {
            "E_eV": float(E[lo + np.argmax(s)]),
            "yield": y,
            "centroid": centroid,
            "rms_width": width,
        }
    return out


def peak_windows(E, spec, max_peaks=6):
    idx, props = find_peaks(spec, prominence=0.05 * float(np.max(spec)))
    keep = idx[np.argsort(props["prominences"])[::-1][:max_peaks]]
    keep = np.sort(keep)
    windows = []
    for j, p in enumerate(keep):
        left = (
            (keep[j - 1] + p) // 2 if j else max(0, p - 3 * (keep[1] - p if len(keep) > 1 else 50))
        )
        right = (p + keep[j + 1]) // 2 if j + 1 < len(keep) else min(E.size, p + (p - left))
        windows.append((int(left), int(right)))
    return windows


def run(cfg, coherent):
    report = []
    for E0, thickness in CASES:
        sweep = material_sweep(
            MATERIAL, energy_keV=[E0], thickness_ang=[thickness], tilt_deg=[TILT_DEG]
        )
        extra = {"coherent_emission": True} if coherent else {}
        base_case = build_cases(
            sweep,
            cfg["Ne"],
            cfg["Ne"],
            energy_model="midpoint",
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=CUTOFF_EV,
            **extra,
        )[0]
        per_seed = []
        grid = None
        for seed in cfg["seeds"]:
            case = dict(base_case)
            case["seed"] = seed
            t0 = time.perf_counter()
            tp = runner._transport_case(case)
            # The line grid is resolved from each transport; pin the first
            # seed's grid so every spectrum shares one energy axis.
            if grid is None:
                grid = tp["E_grid"]
            tp["E_grid"] = grid
            segs = tp["segs"]
            soft_rate = _soft_rate_interpolant(float(np.min(segs["E_keV"])), E0)
            rng = np.random.default_rng(1000 + seed)
            spectra = {"baseline": runner._spectrum_case(case, tp)}
            for scale in cfg["scales"]:
                for name, fn in (("row", row_variant), ("vertex", vertex_variant)):
                    v = fn(segs, soft_rate, scale, rng)
                    tp_v = dict(tp)
                    tp_v["segs"] = {**segs, "v_hat": v}
                    spectra[f"{name}x{scale:g}"] = runner._spectrum_case(case, tp_v)
            per_seed.append(spectra)
            print(
                f"E0={E0:g} keV t={thickness:g} A seed={seed}: "
                f"{time.perf_counter() - t0:.1f} s, {segs['L_ang'].size} rows",
                flush=True,
            )
        for key in ("spec", "spec_coherent"):
            if key not in per_seed[0]["baseline"]:
                continue
            E = np.asarray(per_seed[0]["baseline"]["E_grid"])
            mean_base = np.mean([np.asarray(s["baseline"][key]) for s in per_seed], axis=0)
            windows = peak_windows(E, mean_base)
            metrics = {
                variant: [line_metrics(E, np.asarray(s[variant][key]), windows) for s in per_seed]
                for variant in per_seed[0]
            }
            report.append(_summarize(E0, thickness, key, metrics))
    return report


def _metric(metrics, name, field):
    return metrics[name] if field is None else metrics[name][field]


def _summarize(E0, thickness, key, metrics):
    base = metrics["baseline"]
    entry = {"E0_keV": E0, "thickness_ang": thickness, "spectrum": key, "variants": {}}
    names = ["total"] + [k for k in base[0] if k.startswith("peak")]
    fields = {"total": [None]}
    for variant, runs in metrics.items():
        if variant == "baseline":
            continue
        rows = {}
        for name in names:
            for field in fields.get(name) or ["yield", "centroid", "rms_width"]:
                b = np.array([_metric(m, name, field) for m in base])
                d = np.array([_metric(m, name, field) for m in runs]) - b
                label = name if field is None else f"{name}.{field}"
                rows[label] = {
                    "baseline_mean": float(np.mean(b)),
                    "baseline_seed_sd": float(np.std(b, ddof=1)) if b.size > 1 else float("nan"),
                    "paired_diff_mean": float(np.mean(d)),
                    "paired_diff_se": float(np.std(d, ddof=1) / np.sqrt(d.size))
                    if d.size > 1
                    else float("nan"),
                }
        entry["variants"][variant] = rows
    _print(entry)
    return entry


def _print(entry):
    print(f"\n{entry['spectrum']}  E0={entry['E0_keV']:g} keV  t={entry['thickness_ang']:g} A")
    print(
        f"{'variant':10} {'metric':22} {'baseline':>12} {'seed sd':>10} {'diff':>11} {'diff se':>10} {'diff/sd':>8}"
    )
    for variant, rows in entry["variants"].items():
        for label, r in rows.items():
            ratio = (
                r["paired_diff_mean"] / r["baseline_seed_sd"] if r["baseline_seed_sd"] else np.nan
            )
            print(
                f"{variant:10} {label:22} {r['baseline_mean']:12.5g} {r['baseline_seed_sd']:10.3g} "
                f"{r['paired_diff_mean']:11.3g} {r['paired_diff_se']:10.3g} {ratio:8.2f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(QUICK if args.quick else FULL, args.coherent)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
