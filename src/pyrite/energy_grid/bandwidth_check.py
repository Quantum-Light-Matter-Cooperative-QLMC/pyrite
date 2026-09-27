"""Measured line bandwidth against the kinematic-ceiling axis (issue #192).

Evidence for the ``resonance-population`` bandwidth policy. One transport per
configuration; the axis the policy measured and a reference axis on the same
nodes out to the closed-form ceiling are evaluated with the production line and
characteristic reductions (:class:`pyrite.energy_grid.convergence_case.CaseLadder`).

Under ``node`` quadrature a node's density does not depend on how far the axis
extends, so one reference evaluation gives the true truncated yield at any
``stop``: the measured edge, the edge with twice its margin, and the ceiling.
The production audit bound is reported beside the true fraction.

``reference`` runs under FP64 and pickles its transport; ``candidate`` reloads
it in a float32 process and evaluates only the measured axis, so ``compare``
separates bandwidth truncation from backend precision on identical segments.
Heavy: remote only (``python -m pyrite.energy_grid.convergence_job
start-bandwidth``).

A measurement instrument only: no kernel, default, or policy changes here.
"""

import argparse
import dataclasses
import json
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._line_grid_policy import RESONANCE_BANDWIDTH_POLICY

PAYLOAD_SCHEMA = 1
#: Largest axis piece evaluated at once; an 11 GiB device refuses 1.4 M nodes.
PIECE_POINTS = 200_000


def parse_configs(text: str) -> list[dict[str, float]]:
    """``tilt:azimuth:thickness_ang:ne`` items, comma separated."""
    configs = []
    for item in (value for value in text.split(",") if value):
        tilt, azimuth, thickness, ne = item.split(":")
        configs.append(
            {
                "tilt_deg": float(tilt),
                "tilt_azim_deg": float(azimuth),
                "thickness_ang": float(thickness),
                "n_electrons": int(ne),
            }
        )
    if not configs:
        raise ValueError("at least one configuration is required")
    return configs


def build_case(
    material: str, energy_keV: float, config, *, seed: int, resolution: str = "uniform"
) -> dict[str, Any]:
    """One production case under the ``resonance-population`` bandwidth."""
    from ..campaign.config import material_sweep
    from ..campaign.sweep import build_cases

    sweep = material_sweep(
        material,
        thickness_ang=config["thickness_ang"],
        energy_keV=energy_keV,
        tilt_deg=config["tilt_deg"],
        tilt_azim_deg=config["tilt_azim_deg"],
    )
    policy = {"bandwidth": RESONANCE_BANDWIDTH_POLICY}
    if resolution == "local":
        policy["resolution"] = "resonance-local"
    elif resolution != "uniform":
        raise ValueError(f"unknown bandwidth resolution {resolution!r}")
    sweep = dataclasses.replace(sweep, line_grid_policy=policy)
    ne = int(config["n_electrons"])
    case = dict(build_cases(sweep, n_electrons=ne, n_electrons_brem=ne)[0])
    case["seed"] = int(seed)
    return case


def reference_axis(
    measured: np.ndarray, ceiling_eV: float, *, spacing_eV: float | None = None
) -> np.ndarray:
    """Uniform reference through the ceiling at the case's sinc spacing."""
    start, stop = float(measured[0]), float(measured[-1])
    step = float(spacing_eV) if spacing_eV is not None else (stop - start) / (measured.size - 1)
    count = int(np.floor((float(ceiling_eV) - start) / step)) + 1
    return start + step * np.arange(max(count, measured.size))


def truncated_fraction(E_grid: np.ndarray, density: np.ndarray, stop_eV: float) -> float:
    """Share of the trapezoid integral over ``E_grid`` lying above ``stop_eV``."""
    total = float(np.trapezoid(density, E_grid))
    if total <= 0.0:
        return 0.0
    kept = E_grid <= float(stop_eV)
    return 1.0 - float(np.trapezoid(density[kept], E_grid[kept])) / total


def _piecewise(evaluate, grid: np.ndarray, piece: int = PIECE_POINTS) -> np.ndarray:
    """``evaluate`` over ``grid`` in consecutive pieces, concatenated.

    A node's line density does not depend on the axis extent under ``node``
    quadrature, so pieces reproduce one evaluation while the device admits each.
    Characteristic bin masses use each node's bin, so the node next to a piece
    boundary takes a one-sided bin; that perturbs only densities at boundaries.
    """
    return np.concatenate(
        [np.asarray(evaluate(grid[i : i + piece]), dtype=float) for i in range(0, grid.size, piece)]
    )


def _yield_and_centroid(E_grid: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    total = float(np.trapezoid(density, E_grid))
    return total, float(np.trapezoid(density * E_grid, E_grid)) / total if total else float("nan")


def _peak_mib() -> float | None:
    from .._backend import BACKEND

    peak = BACKEND.allocator_stats().get("peak_mib")
    return None if peak is None else float(peak)


def reference(args: argparse.Namespace) -> dict[str, Any]:
    """FP64: transport, evaluate measured and reference axes, pickle segments."""
    from .._backend import REAL
    from .convergence import segment_fingerprint
    from .convergence_case import CaseLadder

    rows, payload = [], []
    for config in parse_configs(args.configs):
        case = build_case(
            args.material, args.energy, config, seed=args.seed, resolution=args.resolution
        )
        ladder = CaseLadder(case)
        measured = np.asarray(ladder.transport["E_grid"], dtype=float)
        record = ladder.transport["diagnostic_grid"]
        ceiling = float(case["line_grid_policy"]["bandwidth"]["stop_eV"])
        axis = reference_axis(
            measured, ceiling, spacing_eV=record.get("feature_width_eV")
        )
        timings = {}
        spectra = {}
        for name, grid in (("measured", measured), ("reference", axis)):
            started = time.perf_counter()
            # Lines and characteristic only: the continuum is not on this axis.
            spectra[name] = {
                "lines": _piecewise(ladder.lines, grid),
                "characteristic": _piecewise(ladder._characteristic, grid),
            }
            timings[name] = time.perf_counter() - started
        stop = float(measured[-1])
        doubled = float(measured[0]) + 2.0 * (stop - float(measured[0]))
        row = {
            **config,
            "real": np.dtype(REAL).name,
            "n_segments": int(np.asarray(ladder.segments["L_ang"]).size),
            "transport_wall_s": ladder.transport_wall_s,
            "measured_points": int(measured.size),
            "reference_points": int(axis.size),
            "minimum_spacing_eV": float(np.diff(measured).min()),
            "maximum_spacing_eV": float(np.diff(measured).max()),
            "reference_spacing_eV": float(axis[1] - axis[0]),
            "stop_eV": stop,
            "ceiling_eV": ceiling,
            "measured_bandwidth": record.get("measured_bandwidth"),
            "measured_wall_s": timings["measured"],
            "reference_wall_s": timings["reference"],
            "device_peak_mib": _peak_mib(),
        }
        for component in ("lines", "characteristic"):
            density = np.asarray(spectra["reference"][component], dtype=float)
            total, centroid = _yield_and_centroid(axis, density)
            kept, kept_centroid = _yield_and_centroid(
                measured, np.asarray(spectra["measured"][component], dtype=float)
            )
            row[component] = {
                "reference_yield": total,
                "measured_yield": kept,
                "yield_rel": (total - kept) / total if total else 0.0,
                "centroid_shift_eV": kept_centroid - centroid,
                "true_fraction_above_stop": truncated_fraction(axis, density, stop),
                "true_fraction_above_2x": truncated_fraction(axis, density, doubled),
            }
        rows.append(row)
        payload.append(
            {
                "config": config,
                "case": case,
                "transport": ladder.transport,
                "fingerprint": segment_fingerprint(ladder.segments),
                "measured": measured,
                "lines_fp64": np.asarray(spectra["measured"]["lines"], dtype=float),
            }
        )
        print(json.dumps(row, default=str), flush=True)
    with Path(args.payload).open("wb") as stream:
        pickle.dump({"schema": PAYLOAD_SCHEMA, "items": payload}, stream, protocol=5)
    report = {"material": args.material, "energy_keV": args.energy, "rows": rows}
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def production(args: argparse.Namespace) -> dict[str, Any]:
    """Production precision at production electron counts: axis, audit, cost.

    No reference axis and no pickle, so it scales to counts whose segments a
    full-ceiling evaluation could not afford. The audit is the one the spectrum
    phase gates on; a refusal is recorded, not raised.
    """
    from .._backend import BACKEND, REAL, _to_cpu
    from .._line_grid_policy import LineGridToleranceError
    from ..montecarlo import runner
    from ..montecarlo.runner.line_grid import check_line_truncation, line_truncation_audit

    rows = []
    for config in parse_configs(args.configs):
        case = build_case(
            args.material, args.energy, config, seed=args.seed, resolution=args.resolution
        )
        row: dict[str, Any] = {**config, "real": np.dtype(REAL).name}
        started = time.perf_counter()
        try:
            transport = runner._transport_case(case, keep_segments_on_device=True)
        except LineGridToleranceError as error:
            row["refused"] = str(error)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            continue
        row["transport_wall_s"] = time.perf_counter() - started
        grid = np.asarray(transport["E_grid"], dtype=float)
        audit = line_truncation_audit(case, grid)
        started = time.perf_counter()
        lines = runner._lines_for_segments(
            transport["segs"],
            grid,
            case,
            transport["n_hat"],
            case.get("abs_layers"),
            transport.get("groove"),
            coherent=False,
            Ne=transport["Ne_lines"],
            truncation_audit=audit,
        )
        row["lines_wall_s"] = time.perf_counter() - started
        try:
            row["truncation_audit"] = check_line_truncation(case, audit)
        except LineGridToleranceError as error:
            row["refused"] = str(error)
        record = transport["diagnostic_grid"]
        density = np.asarray(lines, dtype=float)
        total, centroid = _yield_and_centroid(grid, density)
        row.update(
            n_segments=int(np.asarray(_to_cpu(transport["segs"]["L_ang"])).size),
            points=int(grid.size),
            minimum_spacing_eV=float(np.diff(grid).min()),
            maximum_spacing_eV=float(np.diff(grid).max()),
            stop_eV=float(grid[-1]),
            measured_bandwidth=record.get("measured_bandwidth"),
            line_yield=total,
            line_centroid_eV=centroid,
            device=BACKEND.device.name,
            device_peak_mib=_peak_mib(),
        )
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        del transport, lines
        BACKEND.release_memory()
    report = {"material": args.material, "energy_keV": args.energy, "rows": rows}
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def candidate(args: argparse.Namespace) -> dict[str, Any]:
    """float32: the measured axis on the pickled segments, against FP64."""
    from .._backend import BACKEND, REAL
    from .convergence import lineshape_deviation, segment_fingerprint
    from .convergence_case import CaseLadder

    if np.dtype(REAL) != np.dtype(np.float32):
        raise SystemExit(f"backend REAL is {np.dtype(REAL).name} on {BACKEND.name}, need float32")
    with Path(args.payload).open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - our own measurement artifact
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise SystemExit(f"{args.payload} is not a schema-{PAYLOAD_SCHEMA} bandwidth payload")
    report = json.loads(Path(args.json_out).read_text())
    for row, item in zip(report["rows"], payload["items"], strict=True):
        ladder = CaseLadder(item["case"], transport=item["transport"])
        if segment_fingerprint(ladder.segments) != item["fingerprint"]:
            raise SystemExit("pickled segments do not match their stored fingerprint")
        started = time.perf_counter()
        lines = np.asarray(ladder.lines(item["measured"]), dtype=float)
        wall = time.perf_counter() - started
        grid, reference = item["measured"], item["lines_fp64"]
        fp64_yield, fp64_centroid = _yield_and_centroid(grid, reference)
        f32_yield, f32_centroid = _yield_and_centroid(grid, lines)
        # Where the centroid moves: its per-node contribution difference.
        moment = (lines - reference) * (grid - fp64_centroid)
        worst = int(np.argmax(np.abs(moment)))
        row["float32"] = {
            "yield_rel": (f32_yield - fp64_yield) / fp64_yield if fp64_yield else 0.0,
            "centroid_shift_eV": f32_centroid - fp64_centroid,
            "fp64_centroid_eV": fp64_centroid,
            "deviation": lineshape_deviation(grid, reference, lines),
            "worst_moment_node_eV": float(grid[worst]),
            "moment_share_above_20keV": float(
                np.trapezoid(np.where(grid > 20_000.0, moment, 0.0), grid)
                / (np.trapezoid(moment, grid) or 1.0)
            ),
            "wall_s": wall,
            "device_peak_mib": _peak_mib(),
        }
        print(json.dumps(row["float32"]), flush=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference", help="FP64 transport and reference evaluation")
    ref.add_argument("--material", default="hbn")
    ref.add_argument("--energy", type=float, default=5000.0)
    ref.add_argument("--configs", required=True, help="tilt:azimuth:thickness_ang:ne,...")
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--resolution", choices=("uniform", "local"), default="uniform")
    ref.add_argument("--payload", required=True)
    ref.add_argument("--json-out", required=True)
    prod = commands.add_parser("production", help="production-precision axis, audit and cost")
    prod.add_argument("--material", default="hbn")
    prod.add_argument("--energy", type=float, default=5000.0)
    prod.add_argument("--configs", required=True, help="tilt:azimuth:thickness_ang:ne,...")
    prod.add_argument("--seed", type=int, default=0)
    prod.add_argument("--resolution", choices=("uniform", "local"), default="uniform")
    prod.add_argument("--json-out", required=True)
    cand = commands.add_parser("candidate", help="float32 measured-axis evaluation")
    cand.add_argument("--payload", required=True)
    cand.add_argument("--json-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reference":
        reference(args)
    elif args.command == "production":
        production(args)
    else:
        candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
