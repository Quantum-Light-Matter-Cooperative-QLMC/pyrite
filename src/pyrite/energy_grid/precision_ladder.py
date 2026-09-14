"""Float32 versus FP64 line-spectrum distortion as a function of spacing/ulp.

Evidence for the #101 tolerance-derived window floor ``spacing >= ulp(E_max)/rtol``.
Everything but the backend floating-point width is held fixed: one transport is
pickled once (``transport``), the production line reduction
(:meth:`pyrite.energy_grid.convergence_case.CaseLadder.lines`) is evaluated on
the same uniform windows in an FP64 process and in a float32 process
(``evaluate``), and ``compare`` reports the deviation of yield, centroid,
dominant-line FWHM (#110 physical-energy crossings), and the pointwise
lineshape against ``spacing / ulp``.

``REAL`` is fixed when :mod:`pyrite._backend` is imported, so each precision is
its own process. Float32 exists only on a GPU backend, which makes this remote
work (``python -m pyrite.energy_grid.convergence_job start-precision``).

Float32 ulp is constant on each binade ``[2**k, 2**(k+1))`` eV: every node of a
window inside ``[8192, 16384)`` has the 10 keV ulp (``2**-10`` eV) and every node
inside ``[16384, 32768)`` the 20 keV ulp (``2**-9`` eV). Windows therefore sit on
the strongest line inside each binade rather than at a literal 10 or 20 keV,
where a case may carry no line at all.

A measurement instrument only: no kernel, default, or floor changes here.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from .convergence import lineshape_deviation, segment_fingerprint

PAYLOAD_SCHEMA = 1
DEFAULT_SPACING_OVER_ULP = (8.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
DEFAULT_WINDOW_EV = 200.0
DEFAULT_BANDS = ((8192.0, 16384.0), (16384.0, 32768.0))
#: Locator grid for the strongest in-band line; it only places the window.
LOCATOR_STEP_EV = 2.0


def float32_ulp(energy_eV: float) -> float:
    """Float32 unit in the last place at ``energy_eV``, in eV."""
    return float(np.spacing(np.float32(energy_eV)))


def ulp_window_grid(top_eV: float, spacing_over_ulp: float, window_eV: float) -> np.ndarray:
    """Production-style ``linspace`` ending exactly at ``top_eV``.

    The spacing is ``spacing_over_ulp`` float32 ulps of the top node and the
    interval count is the nearest integer to ``window_eV / spacing`` (at least 2).
    """
    step = float(spacing_over_ulp) * float32_ulp(top_eV)
    intervals = max(int(round(float(window_eV) / step)), 2)
    return np.linspace(float(top_eV) - intervals * step, float(top_eV), intervals + 1)


def cast_statistics(E_grid_eV: object, dtype=np.float32) -> dict[str, float]:
    """How a float64 grid's nodes and steps move when cast to ``dtype``."""
    E = np.asarray(E_grid_eV, dtype=float)
    resolved = np.dtype(dtype)
    cast = E.astype(resolved).astype(float)
    spacing = float(np.diff(E).max())
    steps = np.diff(cast)
    return {
        "spacing_eV": spacing,
        "spacing_over_ulp": spacing / float(np.spacing(resolved.type(E.max()))),
        "collapsed_intervals": int(np.count_nonzero(steps <= 0.0)),
        "min_cast_step_over_spacing": float(steps.min() / spacing),
        "max_cast_step_over_spacing": float(steps.max() / spacing),
        "max_node_shift_over_spacing": float(np.abs(cast - E).max() / spacing),
    }


def locate_windows(
    lines_fn, bands: Sequence[tuple[float, float]], window_eV: float
) -> list[dict[str, float]]:
    """Top energy of a ``window_eV`` window on the strongest line of each binade."""
    windows = []
    half = 0.5 * float(window_eV)
    for low, high in bands:
        # Keep every node of the widest-spacing window inside the binade.
        centres = np.arange(low + half + 10.0, high - half - 10.0, LOCATOR_STEP_EV)
        if centres.size == 0:
            raise ValueError(f"window {window_eV:g} eV does not fit in band [{low:g}, {high:g})")
        density = np.asarray(lines_fn(centres), dtype=float)
        if not np.any(density > 0.0):
            raise SystemExit(
                f"no CXR line density in [{low:g}, {high:g}) eV for this case; choose a case "
                "whose lines reach this float32 binade"
            )
        peak = float(centres[int(np.argmax(density))])
        windows.append(
            {
                "band_low_eV": float(low),
                "band_high_eV": float(high),
                "peak_eV": peak,
                "top_eV": peak + half,
                "ulp_eV": float32_ulp(peak + half),
                "band_integral": float(np.trapezoid(density, centres)),
            }
        )
    return windows


def cmd_transport(args: argparse.Namespace) -> int:
    from .convergence_case import CaseLadder, build_ladder_case

    case = build_ladder_case(
        args.material,
        args.energy,
        args.tilt,
        args.azimuth,
        thickness_ang=args.thickness,
        n_electrons=args.ne,
        seed=args.seed,
    )
    ladder = CaseLadder(case)
    bands: list[tuple[float, float]] = []
    for item in (value for value in args.bands.split(",") if value):
        low, high = item.split(":")
        bands.append((float(low), float(high)))
    windows = locate_windows(ladder.lines, bands, args.window)
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "case": case,
        "transport": ladder.transport,
        "fingerprint": ladder.fingerprint,
        "transport_wall_s": ladder.transport_wall_s,
        "window_eV": float(args.window),
        "windows": windows,
    }
    with Path(args.out).open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"fingerprint": ladder.fingerprint, "windows": windows}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .._backend import BACKEND, REAL
    from .convergence_case import CaseLadder

    real = np.dtype(REAL).name
    if args.expect_dtype and real != args.expect_dtype:
        raise SystemExit(
            f"backend REAL is {real} on {BACKEND.name}, expected {args.expect_dtype}; "
            "float32 needs a GPU backend without PYRITE_FP64=1"
        )
    with Path(args.payload).open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - our own measurement artifact
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise SystemExit(f"{args.payload} is not a schema-{PAYLOAD_SCHEMA} precision payload")
    ladder = CaseLadder(payload["case"], transport=payload["transport"])
    if segment_fingerprint(ladder.segments) != payload["fingerprint"]:
        raise SystemExit("pickled segments do not match their stored fingerprint")
    ratios = [float(value) for value in args.ratios.split(",") if value]
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for w, window in enumerate(payload["windows"]):
        for k, ratio in enumerate(ratios):
            grid = ulp_window_grid(window["top_eV"], ratio, payload["window_eV"])
            started = time.perf_counter()
            line = ladder.lines(grid)
            wall = time.perf_counter() - started
            arrays[f"E_{w}_{k}"] = grid
            arrays[f"line_{w}_{k}"] = line
            rows.append({"window": w, "ratio": ratio, "n_points": int(grid.size), "wall_s": wall})
    meta = {
        "real": real,
        "backend": BACKEND.name,
        "device": BACKEND.device.name,
        "fingerprint": payload["fingerprint"],
        "windows": payload["windows"],
        "ratios": ratios,
        "rows": rows,
        "device_peak_mib": BACKEND.allocator_stats().get("peak_mib"),
    }
    # Named arrays only; ``allow_pickle`` stays at its keyword default and the
    # loader reads with ``allow_pickle=False``.
    arrays["meta"] = np.array(json.dumps(meta))
    cast(Any, np.savez)(args.out, **arrays)
    print(json.dumps({"real": real, "backend": BACKEND.name, "rows": rows}, indent=2))
    return 0


def _load_evaluation(path: str | Path) -> tuple[dict[str, Any], Mapping[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    return json.loads(str(data["meta"])), data


def _loglog_slope(ratios: Sequence[float], values: Sequence[float]) -> float:
    points = [
        (math.log(r), math.log(abs(v)))
        for r, v in zip(ratios, values, strict=True)
        if math.isfinite(v) and v != 0.0
    ]
    if len(points) < 2:
        return float("nan")
    x, y = np.array(points).T
    return float(np.polyfit(x, y, 1)[0])


def compare(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    require_precision_pair: bool = True,
) -> dict[str, Any]:
    """Tabulate candidate-versus-reference deviation per window and spacing/ulp."""
    ref_meta, ref = _load_evaluation(reference_path)
    cand_meta, cand = _load_evaluation(candidate_path)
    if require_precision_pair and (ref_meta["real"], cand_meta["real"]) != ("float64", "float32"):
        raise SystemExit(
            f"expected float64 reference and float32 candidate, got "
            f"{ref_meta['real']} and {cand_meta['real']}"
        )
    if ref_meta["fingerprint"] != cand_meta["fingerprint"]:
        raise SystemExit("reference and candidate were evaluated on different transports")
    if ref_meta["ratios"] != cand_meta["ratios"] or ref_meta["windows"] != cand_meta["windows"]:
        raise SystemExit("reference and candidate used different windows or ratios")
    rows = []
    for w, window in enumerate(ref_meta["windows"]):
        for k, ratio in enumerate(ref_meta["ratios"]):
            grid = np.asarray(ref[f"E_{w}_{k}"], dtype=float)
            if not np.array_equal(grid, cand[f"E_{w}_{k}"]):
                raise SystemExit(f"window {w} ratio {ratio:g}: grids differ")
            reference, candidate = ref[f"line_{w}_{k}"], cand[f"line_{w}_{k}"]
            nominal = lineshape_deviation(grid, reference, candidate)
            cast_grid = grid.astype(np.float32).astype(float)
            on_cast = lineshape_deviation(grid, reference, candidate, candidate_E_grid_eV=cast_grid)
            rows.append(
                {
                    "window": w,
                    "top_eV": window["top_eV"],
                    "ulp_eV": window["ulp_eV"],
                    "ratio": ratio,
                    **cast_statistics(grid),
                    **nominal,
                    "centroid_shift_over_ulp": nominal["centroid_shift_eV"] / window["ulp_eV"],
                    "yield_rel_cast_coordinates": on_cast["yield_rel"],
                    "fwhm_rel_cast_coordinates": on_cast["fwhm_rel"],
                    "reference_wall_s": ref_meta["rows"][w * len(ref_meta["ratios"]) + k]["wall_s"],
                    "candidate_wall_s": cand_meta["rows"][w * len(ref_meta["ratios"]) + k][
                        "wall_s"
                    ],
                }
            )
    slopes = {}
    for w in range(len(ref_meta["windows"])):
        window_rows = [row for row in rows if row["window"] == w]
        ratios = [row["ratio"] for row in window_rows]
        slopes[str(w)] = {
            name: _loglog_slope(ratios, [row[name] for row in window_rows])
            for name in ("yield_rel", "fwhm_rel", "max_pointwise_rel", "centroid_shift_eV")
        }
    return {
        "reference": {
            key: ref_meta[key] for key in ("real", "backend", "device", "device_peak_mib")
        },
        "candidate": {
            key: cand_meta[key] for key in ("real", "backend", "device", "device_peak_mib")
        },
        "fingerprint": ref_meta["fingerprint"],
        "windows": ref_meta["windows"],
        "rows": rows,
        "loglog_slope_vs_spacing_over_ulp": slopes,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    report = compare(args.reference, args.candidate)
    Path(args.json_out).write_text(json.dumps(report, indent=2))
    header = "window  top_eV   h/ulp  yield_rel  cent_shift/ulp  fwhm_rel  max_pointwise  collapsed"
    print(header)
    for row in report["rows"]:
        print(
            f"{row['window']:>6} {row['top_eV']:>8.1f} {row['ratio']:>7g} "
            f"{row['yield_rel']:>10.3e} {row['centroid_shift_over_ulp']:>15.3e} "
            f"{row['fwhm_rel']:>9.3e} {row['max_pointwise_rel']:>14.3e} "
            f"{row['collapsed_intervals']:>10d}"
        )
    print(json.dumps(report["loglog_slope_vs_spacing_over_ulp"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    transport = commands.add_parser("transport", help="transport once and pickle the segments")
    transport.add_argument("--material", default="wse2")
    transport.add_argument("--energy", type=float, default=300.0)
    transport.add_argument("--tilt", type=float, default=5.0)
    transport.add_argument("--azimuth", type=float, default=95.0)
    transport.add_argument("--thickness", type=float, default=1.0e5)
    transport.add_argument("--ne", type=int, default=100)
    transport.add_argument("--seed", type=int, default=0)
    transport.add_argument(
        "--bands", default=",".join(f"{low:g}:{high:g}" for low, high in DEFAULT_BANDS)
    )
    transport.add_argument("--window", type=float, default=DEFAULT_WINDOW_EV)
    transport.add_argument("--out", required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate lines on every window and ratio")
    evaluate.add_argument("--payload", required=True)
    evaluate.add_argument(
        "--ratios", default=",".join(f"{value:g}" for value in DEFAULT_SPACING_OVER_ULP)
    )
    evaluate.add_argument("--expect-dtype", choices=("float32", "float64"), default=None)
    evaluate.add_argument("--out", required=True)
    comparison = commands.add_parser("compare", help="float32 versus FP64 deviation table")
    comparison.add_argument("--reference", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--json-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"transport": cmd_transport, "evaluate": cmd_evaluate, "compare": cmd_compare}
    return handler[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
