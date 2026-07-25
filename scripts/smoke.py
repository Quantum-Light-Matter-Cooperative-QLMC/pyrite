#!/usr/bin/env python3
"""Headless checkpoint-to-plot smoke test for cxr-mc."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from cxr_mc.config import default_settings  # noqa: E402
from cxr_mc.plots import plot_best_spectra  # noqa: E402
from cxr_mc.plots.altair_spectra import spectrum_chart  # noqa: E402
from cxr_mc.results import filter_results, records, top_geometries  # noqa: E402
from cxr_mc.run import cases_from_results, load_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_out"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    material = args.material
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] loading checkpoint: {material}")
    results = load_checkpoint(material)
    cases = cases_from_results(results)
    filtered = filter_results(results, cases)
    settings = default_settings()

    recs = records(filtered)
    if not recs:
        print(f"[smoke] FAIL: no records for {material!r}")
        return 2
    tilts = sorted({record["case"]["tilt_deg"] for record in recs})
    tilt = tilts[len(tilts) // 2]
    print(f"[smoke] {len(recs)} records, {len(tilts)} tilts; using tilt_deg={tilt:g}")

    altair_png = output_dir / f"{material}_spectrum.png"
    chart = spectrum_chart(filtered, settings, tilt_deg=tilt, include_brem=True)
    if chart is None:
        print("[smoke] FAIL: spectrum_chart returned None")
        return 3
    chart.save(str(altair_png))
    print(f"[smoke] wrote {altair_png} ({altair_png.stat().st_size} bytes)")

    matplotlib_png = output_dir / f"{material}_best_spectra.png"
    figure = plot_best_spectra(filtered, settings, top_n=6, cases=cases)
    if figure is None:
        print("[smoke] FAIL: plot_best_spectra returned None")
        return 4
    figure.savefig(matplotlib_png, dpi=110, bbox_inches="tight")
    print(f"[smoke] wrote {matplotlib_png} ({matplotlib_png.stat().st_size} bytes)")

    ranking = top_geometries(filtered, settings, top_n=10, select="quality_peak")
    print(f"[smoke] top geometries ({len(ranking)} rows):")
    print(ranking.head(10).to_string(index=False))
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
