"""Full-fiber reflection selector replay on the frozen #1480 ladder scenes (#1484).

One transport per scene (the #1480 ladder's seed and geometry; its segment
fingerprint is compared with the ladder's and recorded). The measured selector runs through
``score_fiber_case(..., line_yield_tolerance=...)``; the same transport is then
scored at fixed cutoffs with the unchanged opt-in model. The stop is judged on
the CXR + corrected-continuum total in the band, the quantity the net metric
measures. Writes ``summary.json`` and ``spectra.npz`` after every arm
(``complete: false`` until the last), then ``SHA256SUMS``.

Validation: reflection-yield-tolerance-selector
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from fiber_paired_1480 import file_digest, load_source, photon_axis


def _yields(result, widths):
    """Integrated CXR, continuum and total yields on the shared photon axis."""
    cxr = float(np.asarray(result["cxr"]) @ widths)
    brem = float(np.asarray(result["brem"]) @ widths)
    return dict(cxr_yield=cxr, brem_yield=brem, total_yield=cxr + brem)


def _ratio(value, reference, offset=1.0):
    """``value / reference - offset``, or ``None`` when the reference is zero or nonfinite."""
    ratio = value / reference - offset if reference else float("nan")
    return float(ratio) if np.isfinite(ratio) else None


def _model(args, cutoff):
    from pyrite.config import FiberAverage
    from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel

    average = FiberAverage(
        cutoff_invang=cutoff, azimuth_count=args.azimuths, reflection_batch_size=args.batch
    )
    return FiberCXRModel(average, args.temperature)


def run(args):
    from pyrite.energy_grid.convergence import require_identical_segments, segment_fingerprint
    from pyrite.montecarlo.runner.oom import _ensure_pool_limit
    from pyrite.montecarlo.spectrum.cxr.fiber_case import score_fiber_case

    if args.artifact is not None and args.detector is not None:
        raise ValueError("artifact detector geometry is fixed; omit --detector")
    if args.artifact is None and args.detector is None:
        args.detector = 45.0  # the #1480 ladder scenes' lab angle
    if args.histories is not None and args.histories < 1:
        raise ValueError("--histories must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("summary.json", "spectra.npz", "SHA256SUMS")):
        raise FileExistsError("use a fresh output directory")
    _ensure_pool_limit()
    started = time.perf_counter()
    grid, edges = photon_axis(*args.band, args.spacing)
    widths = np.diff(edges)
    case, tp, source = load_source(args)
    fingerprint = segment_fingerprint(tp["segs"])
    # The #1480 ladder transported on CUDA (device FP64); a CPU transport of
    # the same seed keeps the segment count but may differ in the last bits.
    frozen = dict(
        expected=args.expect_fingerprint,
        matches=None
        if not args.expect_fingerprint
        else fingerprint["digest"] == args.expect_fingerprint,
    )
    print("frozen fingerprint", json.dumps(dict(frozen, actual=fingerprint)), flush=True)
    arrays = dict(grid_eV=grid, edges_eV=edges)

    summary = dict(
        issue=1484,
        complete=False,
        source=dict(source, diagnostic_sha256=file_digest(Path(__file__))),
        fingerprint=fingerprint,
        frozen_fingerprint=frozen,
        electrons=int(tp["Ne_lines"]),
        E0_keV=float(case["E0_keV"]),
        tilt_deg=case.get("tilt_deg"),
        thickness_ang=case.get("thickness_ang"),
        theta_obs_rad=float(case["theta_obs_rad"]) if "theta_obs_rad" in case else None,
        band_eV=[float(edges[0]), float(edges[-1])],
        spacing_eV=float(widths[0]),
        azimuth_count=args.azimuths,
        temperature_K=args.temperature,
        tolerance=args.tolerance,
        cap_invang=args.cap,
        selected=None,
        prefix=None,
        fixed=[],
    )

    def flush():
        """Persist every finished arm, so a late failure keeps the earlier ones."""
        summary["wall_s"] = time.perf_counter() - started
        np.savez_compressed(output / "spectra.npz", **arrays)
        text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        (output / "summary.json").write_text(text)

    def timed(label, fn):
        t0 = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - t0
        print(label, f"{elapsed:.1f} s", flush=True)
        return value, elapsed

    selected, wall = timed(
        "selection",
        lambda: score_fiber_case(
            _model(args, args.cap), case, tp, grid, line_yield_tolerance=args.tolerance
        ),
    )
    record = selected["reflection_selection"]
    arrays["selected_cxr"], arrays["selected_brem"] = selected["cxr"], selected["brem"]
    sel = summary["selected"] = dict(
        _yields(selected, widths),
        wall_s=wall,
        record=record,
        continuum_correction=selected["continuum_correction"],
    )
    print(json.dumps({k: v for k, v in record.items() if k != "shells"}), flush=True)
    flush()
    if args.histories is not None:
        result, wall = timed(
            "prefix selection",
            lambda: score_fiber_case(
                _model(args, args.cap),
                case,
                tp,
                grid,
                line_yield_tolerance=args.tolerance,
                selection_histories=args.histories,
            ),
        )
        summary["prefix"] = dict(
            _yields(result, widths), wall_s=wall, record=result["reflection_selection"]
        )
        flush()
    for cutoff in sorted(set(args.fixed)):
        result, wall = timed(
            f"fixed {cutoff:g}", lambda c=cutoff: score_fiber_case(_model(args, c), case, tp, grid)
        )
        arrays[f"g{cutoff:g}_cxr"], arrays[f"g{cutoff:g}_brem"] = result["cxr"], result["brem"]
        row = dict(
            cutoff_invang=cutoff,
            reflections=len(result["model"]["hkls"]),
            wall_s=wall,
            continuum_correction=result["continuum_correction"],
            **_yields(result, widths),
        )
        # Positive: the fixed cutoff holds more net yield than the selected set.
        row["total_vs_selected"] = _ratio(row["total_yield"], sel["total_yield"])
        row["cxr_vs_selected"] = _ratio(row["cxr_yield"], sel["cxr_yield"])
        l1 = float(np.abs(result["cxr"] - selected["cxr"]) @ widths)
        row["cxr_l1_vs_selected"] = _ratio(l1, sel["cxr_yield"], offset=0.0)
        summary["fixed"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "continuum_correction"}), flush=True)
        flush()
    require_identical_segments(fingerprint, tp["segs"])
    summary["complete"] = True
    flush()
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_digest(output / n)}  {n}\n" for n in ("spectra.npz", "summary.json"))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, help="existing schema-2 trajectory artifact")
    parser.add_argument("--energy", type=float, default=30.0, help="fresh transport energy [keV]")
    parser.add_argument("--thickness", type=float, default=5000.0, help="slab thickness [A]")
    parser.add_argument("--tilt", type=float, default=15.0)
    parser.add_argument("--tilt-azimuth", type=float, default=180.0)
    parser.add_argument(
        "--detector", type=float, help="fresh detector polar angle [degrees] (default 45)"
    )
    parser.add_argument("--electrons", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1480)
    parser.add_argument("--band", type=float, nargs=2, default=(20.0, 25000.0))
    parser.add_argument("--spacing", type=float, default=2.0, help="maximum bin width [eV]")
    parser.add_argument("--azimuths", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--cap", type=float, default=40.0, help="selection search cap [1/A]")
    parser.add_argument("--fixed", type=float, nargs="*", default=(6.0, 8.0, 10.0, 24.0, 32.0))
    parser.add_argument("--histories", type=int, help="also select on this electron prefix")
    parser.add_argument(
        "--expect-fingerprint", help="record whether the transport digest equals this one"
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
