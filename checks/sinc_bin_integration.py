"""Bin-mean line quadrature on real trajectories (issue #116 acceptance).

Two measurements, both on ONE pickled transport so a difference is quadrature,
route, or precision and never Monte Carlo:

1. Integrated incoherent line yield. Bin-mean on uniform 3 eV and 0.375 eV
   axes (whose bins tile the same interval) and on a #101 window plan, against
   node sampling on a uniform reference whose spacing is below the narrowest
   feature ``pi / a_width`` of every line. ``sinc^2`` is band-limited, so the
   node sum there is the exact integral (the ``line-grid-sinc-convergence``
   argument), not merely a finer approximation. Node sampling on the coarse
   axes is reported alongside to show the aliasing bin-mean removes.
2. Route agreement: the same evaluation in one process per backend and
   precision (CPU FP64, CUDA FP64 -> CuPy fallback, CUDA float32 -> fused
   kernel), compared pointwise and in yield.

``REAL`` is fixed at import, so each backend is its own process::

    uv run python checks/sinc_bin_integration.py transport --out p.pkl
    PYRITE_MC_BACKEND=cpu uv run python checks/sinc_bin_integration.py \\
        evaluate --payload p.pkl --out cpu64.npz --expect-dtype float64
    PYRITE_MC_BACKEND=cuda uv run python checks/sinc_bin_integration.py \\
        evaluate --payload p.pkl --out gpu32.npz --expect-dtype float32
    uv run python checks/sinc_bin_integration.py yield --evaluation gpu32.npz --rtol 1e-3
    uv run python checks/sinc_bin_integration.py agree \\
        --reference cpu64.npz --candidate gpu32.npz --rtol 1e-5

Heavy at 300 keV: run through ``pyrite remote`` (lab box), never locally.
A measurement instrument only -- no kernel, default, or policy changes here.

Validation: sinc-bin-integration
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PAYLOAD_SCHEMA = 1
UNIFORM_SPACINGS_EV = (3.0, 0.375)


def _uniform(start, stop, spacing):
    count = int(round((stop - start) / spacing))
    return start + (np.arange(count) + 0.5) * spacing


def cmd_transport(args):
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
    # Uniform bins tile [start, stop'] with stop' a whole number of 3 eV bins,
    # so both spacings integrate exactly the same interval.
    stop = start + 3.0 * np.floor((stop - start) / 3.0)
    n_hat = np.asarray(ladder.transport["n_hat"], dtype=float)
    width, _aliased, _n = sinc_feature_spacing(
        ladder.segments, n_hat, electron_limit=ladder.transport["Ne_lines"]
    )
    grids = {f"uniform_{s:g}": _uniform(start, stop, s) for s in UNIFORM_SPACINGS_EV}
    narrowest = _narrowest_feature_eV(ladder, n_hat)
    spacing = UNIFORM_SPACINGS_EV[-1]
    while spacing > narrowest:
        spacing /= 2.0
    reference = _uniform(start, stop, spacing)
    if reference.size > args.max_reference_points:
        raise SystemExit(
            f"exact node reference needs {reference.size} points at {spacing:g} eV "
            f"(narrowest feature {narrowest:.4g} eV), above --max-reference-points"
        )
    grids["reference"] = reference
    for samples in (args.window_samples,):
        context = SeedContext(
            case=ladder.case,
            segments=ladder.segments,
            n_hat=n_hat,
            electron_limit=ladder.transport["Ne_lines"],
            start_eV=start,
            stop_eV=stop,
            feature_width_eV=float(width),
            samples_per_feature=int(samples),
        )
        seeds, _summaries = collect_feature_seeds(context, DEFAULT_SEED_PROVIDERS)
        grids[f"windows_{samples}"] = build_window_plan(start, stop, 3.0, seeds).coordinates()
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "case": case,
        "transport": ladder.transport,
        "fingerprint": ladder.fingerprint,
        "grids": grids,
        "reference": "reference",
        "feature_width_eV": float(width),
        "narrowest_feature_eV": narrowest,
    }
    with Path(args.out).open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        json.dumps(
            {
                "fingerprint": ladder.fingerprint,
                "bandwidth_eV": [start, stop],
                "feature_width_eV": float(width),
                "narrowest_feature_eV": narrowest,
                "reference_spacing_eV": spacing,
                "points": {name: int(grid.size) for name, grid in grids.items()},
            },
            indent=2,
        )
    )
    return 0


def _narrowest_feature_eV(ladder, n_hat):
    """``min pi / a_width = 2 pi hbar c / ((1 - beta v.n) t_L)`` over line rows.

    Vacuum denominator: the in-medium correction to ``1 - v.n`` is of order the
    refractive decrement, and a spacing at most a hair above ``pi / a_width``
    aliases only the vanishing edge of the ``sinc^2`` spectrum.
    """
    from pyrite.materials.crystal import HBARC_EV_ANG
    from pyrite.montecarlo.transport import beta_from_keV

    segs = ladder.segments
    rows = np.asarray(_to_host(segs["elec_id"])) < int(ladder.transport["Ne_lines"])
    energy = segs["E_repr_keV"] if segs.get("E_repr_keV") is not None else segs["E_keV"]
    beta = np.asarray(beta_from_keV(_to_host(energy)[rows]), dtype=float)
    v_hat = _to_host(segs["v_hat"])[rows]
    t_L = _to_host(segs["L_ang"])[rows] / beta
    denominator = 1.0 - beta * (v_hat @ n_hat)
    return float(np.min(2.0 * np.pi * HBARC_EV_ANG / (denominator * t_L)))


def _to_host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array, dtype=float)


def _bin_yield(grid, density):
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    return float(np.sum(density * _energy_bin_edges_and_widths(grid)[1]))


def cmd_evaluate(args):
    from pyrite._backend import BACKEND, REAL
    from pyrite.energy_grid.convergence import segment_fingerprint
    from pyrite.energy_grid.convergence_case import CaseLadder

    real = np.dtype(REAL).name
    if args.expect_dtype and real != args.expect_dtype:
        raise SystemExit(f"backend REAL is {real} on {BACKEND.name}, expected {args.expect_dtype}")
    with Path(args.payload).open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - our own measurement artifact
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise SystemExit(f"{args.payload} is not a schema-{PAYLOAD_SCHEMA} payload")
    wanted = set(args.grids.split(",")) if args.grids else set(payload["grids"])
    node = CaseLadder(payload["case"], transport=payload["transport"])
    if segment_fingerprint(node.segments) != payload["fingerprint"]:
        raise SystemExit("pickled segments do not match their stored fingerprint")
    binned = CaseLadder(
        {**payload["case"], "line_quadrature": "bin-mean"}, transport=payload["transport"]
    )
    arrays, rows = {}, {}
    for name, grid in payload["grids"].items():
        if name not in wanted:
            continue
        for quadrature, ladder in (("node", node), ("bin-mean", binned)):
            if quadrature == "node" and name != payload["reference"] and args.skip_node:
                continue
            started = time.perf_counter()
            line = _to_host(ladder.lines(grid))
            key = f"{name}:{quadrature}"
            arrays[key] = line
            arrays[f"{name}:grid"] = grid
            rows[key] = {
                "yield": _bin_yield(grid, line),
                "points": int(grid.size),
                "wall_s": time.perf_counter() - started,
            }
            print(key, json.dumps(rows[key]), flush=True)
    meta = {
        "backend": BACKEND.name,
        "real": real,
        "reference": payload["reference"],
        "fingerprint": payload["fingerprint"],
        "rows": rows,
    }
    np.savez(args.out, meta=np.array(json.dumps(meta)), **arrays)
    return 0


def _load(path):
    data = np.load(path, allow_pickle=False)
    return json.loads(str(data["meta"])), data


def cmd_yield(args):
    meta, _data = _load(args.evaluation)
    rows = meta["rows"]
    reference = rows[f"{meta['reference']}:node"]["yield"]
    report = {
        "backend": meta["backend"],
        "real": meta["real"],
        "reference": f"{meta['reference']}:node",
        "reference_yield": reference,
        "relative": {key: row["yield"] / reference - 1.0 for key, row in rows.items()},
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    binned = [abs(v) for k, v in report["relative"].items() if k.endswith(":bin-mean")]
    if args.rtol is not None and max(binned) > args.rtol:
        print(f"FAIL: bin-mean yield deviation {max(binned):.3g} > {args.rtol:g}", file=sys.stderr)
        return 1
    return 0


def cmd_agree(args):
    ref_meta, ref = _load(args.reference)
    worst = 0.0
    report = {"reference": ref_meta["backend"] + "/" + ref_meta["real"], "candidates": []}
    for path in args.candidate:
        meta, data = _load(path)
        if meta["fingerprint"] != ref_meta["fingerprint"]:
            raise SystemExit(f"{path} was evaluated on different segments")
        rows = {}
        for key in data.files:
            if key in ("meta",) or key.endswith(":grid") or key not in ref.files:
                continue
            a, b = ref[key], data[key]
            peak = float(np.abs(a).max())
            significant = np.abs(a) > args.floor * peak
            pointwise = float(np.max(np.abs(b[significant] / a[significant] - 1.0)))
            yield_rel = meta["rows"][key]["yield"] / ref_meta["rows"][key]["yield"] - 1.0
            rows[key] = {"yield_rel": yield_rel, "pointwise_rel": pointwise}
            if key.endswith(":bin-mean"):
                worst = max(worst, abs(yield_rel))
        report["candidates"].append(
            {"path": path, "backend": meta["backend"], "real": meta["real"], "rows": rows}
        )
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    if args.rtol is not None and worst > args.rtol:
        print(f"FAIL: bin-mean route yield deviation {worst:.3g} > {args.rtol:g}", file=sys.stderr)
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    transport = commands.add_parser("transport", help="transport once and pickle the grids")
    transport.add_argument("--material", default="hopg")
    transport.add_argument("--energy", type=float, default=300.0)
    transport.add_argument("--tilt", type=float, default=5.0)
    transport.add_argument("--azimuth", type=float, default=95.0)
    transport.add_argument("--thickness", type=float, default=1.0e7)
    transport.add_argument("--ne", type=int, default=200)
    transport.add_argument("--seed", type=int, default=0)
    transport.add_argument("--window-samples", type=int, default=2)
    transport.add_argument("--max-reference-points", type=int, default=4_000_000)
    transport.add_argument("--out", required=True)

    evaluate = commands.add_parser("evaluate", help="evaluate every grid on one backend")
    evaluate.add_argument("--payload", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--expect-dtype", default=None)
    evaluate.add_argument("--grids", default=None, help="comma-separated subset")
    evaluate.add_argument(
        "--skip-node", action="store_true", help="node-sample only the reference grid"
    )

    yield_ = commands.add_parser("yield", help="bin-mean yields against the node reference")
    yield_.add_argument("--evaluation", required=True)
    yield_.add_argument("--json-out", default=None)
    yield_.add_argument("--rtol", type=float, default=None)

    agree = commands.add_parser("agree", help="route/precision agreement against a reference")
    agree.add_argument("--reference", required=True)
    agree.add_argument("--candidate", action="append", required=True)
    agree.add_argument("--floor", type=float, default=1e-3, help="pointwise mask, x peak")
    agree.add_argument("--json-out", default=None)
    agree.add_argument("--rtol", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    handler = {
        "transport": cmd_transport,
        "evaluate": cmd_evaluate,
        "yield": cmd_yield,
        "agree": cmd_agree,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
