"""CUDA versus CPU-fallback agreement of the line route on a windowed axis.

Issue #101 acceptance: the two line routes must agree on a production window
plan, in float32 and in FP64. The measurement contract is the one the
refinement ladder uses -- one transport, pickled once, and nothing but the
backend changing between evaluations -- so a difference here is route or
precision, never Monte Carlo.

``REAL`` is fixed when :mod:`pyrite._backend` is imported, so each backend and
precision is its own process:

    uv run python checks/line_window_backend_agreement.py transport --out p.pkl
    PYRITE_MC_BACKEND=cpu  uv run python checks/line_window_backend_agreement.py \\
        evaluate --payload p.pkl --out cpu64.npz --expect-dtype float64
    PYRITE_MC_BACKEND=cuda uv run python checks/line_window_backend_agreement.py \\
        evaluate --payload p.pkl --out gpu32.npz --expect-dtype float32
    PYRITE_FP64=1 PYRITE_MC_BACKEND=cuda uv run python \\
        checks/line_window_backend_agreement.py evaluate --payload p.pkl \\
        --out gpu64.npz --expect-dtype float64
    uv run python checks/line_window_backend_agreement.py compare \\
        --reference cpu64.npz --candidate gpu64.npz --candidate gpu32.npz

The grid is the real thing: seeds from the registered providers, merged by
``build_window_plan`` on a backbone at the policy's maximum spacing. A
measurement instrument only -- no kernel, default, or policy changes here.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PAYLOAD_SCHEMA = 1
DEFAULT_MATERIAL = "hopg"
DEFAULT_ENERGY_KEV = 30.0
DEFAULT_TILT_DEG = 5.0
DEFAULT_AZIMUTH_DEG = 95.0
DEFAULT_THICKNESS_ANG = 1.0e7
DEFAULT_NE = 200
DEFAULT_BACKBONE_EV = 3.0
DEFAULT_SAMPLES = 8


def cmd_transport(args: argparse.Namespace) -> int:
    from pyrite._line_windows import build_window_plan
    from pyrite.energy_grid.convergence_case import CaseLadder, build_ladder_case
    from pyrite.montecarlo.spectrum.diagnostics import sinc_feature_spacing
    from pyrite.montecarlo.spectrum.line_seeds import (
        DEFAULT_SEED_PROVIDERS,
        SeedContext,
        collect_feature_seeds,
    )

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
    start, stop = ladder.bandwidth_eV
    n_hat = np.asarray(ladder.transport["n_hat"], dtype=float)
    width, _aliased, _n = sinc_feature_spacing(
        ladder.segments, n_hat, electron_limit=ladder.transport["Ne_lines"]
    )
    context = SeedContext(
        case=ladder.case,
        segments=ladder.segments,
        n_hat=n_hat,
        electron_limit=ladder.transport["Ne_lines"],
        start_eV=start,
        stop_eV=stop,
        feature_width_eV=float(width),
        samples_per_feature=int(args.samples),
    )
    seeds, summaries = collect_feature_seeds(context, DEFAULT_SEED_PROVIDERS)
    plan = build_window_plan(start, stop, float(args.backbone), seeds)
    grid = plan.coordinates()
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "case": case,
        "transport": ladder.transport,
        "fingerprint": ladder.fingerprint,
        "E_grid": grid,
        "plan": plan.payload(),
        "seed_summaries": summaries,
        "feature_width_eV": float(width),
    }
    with Path(args.out).open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        json.dumps(
            {
                "fingerprint": ladder.fingerprint,
                "bandwidth_eV": [start, stop],
                "feature_width_eV": float(width),
                "n_points": int(grid.size),
                "n_pieces": len(plan.pieces),
                "min_spacing_eV": float(np.diff(grid).min()),
                "max_spacing_eV": float(np.diff(grid).max()),
                "top_eV": float(grid[-1]),
            },
            indent=2,
        )
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pyrite._backend import BACKEND, REAL
    from pyrite.energy_grid.convergence import segment_fingerprint, spectrum_observables
    from pyrite.energy_grid.convergence_case import CaseLadder

    real = np.dtype(REAL).name
    if args.expect_dtype and real != args.expect_dtype:
        raise SystemExit(
            f"backend REAL is {real} on {BACKEND.name}, expected {args.expect_dtype}; "
            "float32 needs a GPU backend without PYRITE_FP64=1"
        )
    with Path(args.payload).open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - our own measurement artifact
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise SystemExit(f"{args.payload} is not a schema-{PAYLOAD_SCHEMA} agreement payload")
    ladder = CaseLadder(payload["case"], transport=payload["transport"])
    if segment_fingerprint(ladder.segments) != payload["fingerprint"]:
        raise SystemExit("pickled segments do not match their stored fingerprint")

    grid = np.asarray(payload["E_grid"], dtype=float)
    started = time.perf_counter()
    sample = ladder.evaluate(grid)
    wall = time.perf_counter() - started
    line = np.asarray(_to_host(sample.line), dtype=float)
    background = np.asarray(_to_host(sample.background), dtype=float)
    observables = spectrum_observables(grid, line, background)
    np.savez(
        args.out,
        E_grid=grid,
        line=line,
        background=background,
        meta=np.array(
            json.dumps(
                {
                    "backend": BACKEND.name,
                    "real": real,
                    "wall_s": wall,
                    "observables": observables,
                }
            )
        ),
    )
    print(json.dumps({"backend": BACKEND.name, "real": real, **observables}, indent=2))
    return 0


def _to_host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def _load(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["E_grid"], data["line"], meta


def cmd_compare(args: argparse.Namespace) -> int:
    from pyrite.energy_grid.convergence import lineshape_deviation

    E_reference, reference, reference_meta = _load(args.reference)
    rows = []
    for path in args.candidate:
        E_candidate, candidate, meta = _load(path)
        if candidate.shape != reference.shape:
            raise SystemExit(f"{path} has {candidate.size} nodes, reference has {reference.size}")
        deviation = lineshape_deviation(
            E_reference, reference, candidate, candidate_E_grid_eV=E_candidate
        )
        rows.append(
            {
                "candidate": path,
                "backend": meta["backend"],
                "real": meta["real"],
                "max_node_shift_eV": float(np.abs(E_candidate - E_reference).max()),
                **deviation,
            }
        )
    report = {
        "reference": {
            "path": args.reference,
            "backend": reference_meta["backend"],
            "real": reference_meta["real"],
        },
        "candidates": rows,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    worst = max((abs(row.get("yield_rel", 0.0) or 0.0) for row in rows), default=0.0)
    if args.rtol is not None and worst > args.rtol:
        print(f"FAIL: worst yield deviation {worst:.3g} exceeds {args.rtol:g}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    transport = commands.add_parser("transport", help="transport once and pickle the window plan")
    transport.add_argument("--material", default=DEFAULT_MATERIAL)
    transport.add_argument("--energy", type=float, default=DEFAULT_ENERGY_KEV)
    transport.add_argument("--tilt", type=float, default=DEFAULT_TILT_DEG)
    transport.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH_DEG)
    transport.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS_ANG)
    transport.add_argument("--ne", type=int, default=DEFAULT_NE)
    transport.add_argument("--seed", type=int, default=0)
    transport.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    transport.add_argument("--backbone", type=float, default=DEFAULT_BACKBONE_EV)
    transport.add_argument("--out", required=True)

    evaluate = commands.add_parser("evaluate", help="evaluate the line route on one backend")
    evaluate.add_argument("--payload", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--expect-dtype", default=None)

    comparison = commands.add_parser("compare", help="deviation of each candidate from reference")
    comparison.add_argument("--reference", required=True)
    comparison.add_argument("--candidate", action="append", required=True)
    comparison.add_argument("--json-out", default=None)
    comparison.add_argument("--rtol", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "transport":
        return cmd_transport(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
