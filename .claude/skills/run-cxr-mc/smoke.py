#!/usr/bin/env python
"""Headless smoke driver for cxr-mc — exercises the same data + plotting path
the marimo `analysis_app.py` uses, but renders straight to PNG so it runs with
no browser and no live kernel.

    uv run python .claude/skills/run-cxr-mc/smoke.py [material] [out_dir]

Loads a checkpoint (default: hopg, whichever exist under checkpoints/), rebuilds
the case list, then renders one Altair spectrum chart and one matplotlib
best-spectra grid to <out_dir>, and prints the top-geometry ranking table.
Exit 0 on success; non-zero if the checkpoint is missing or a figure is empty.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no display; render to buffers

from cxr_mc.config import default_settings  # noqa: E402
from cxr_mc.plots import plot_best_spectra  # noqa: E402
from cxr_mc.plots.altair_spectra import spectrum_chart  # noqa: E402
from cxr_mc.results import filter_results, records, top_geometries  # noqa: E402
from cxr_mc.run import cases_from_results, load_checkpoint  # noqa: E402


def main() -> int:
    material = sys.argv[1] if len(sys.argv) > 1 else "hopg"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("smoke_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- same load path as analysis_app.py's data cell -----------------------
    print(f"[smoke] loading checkpoint: {material}")
    results = load_checkpoint(material)  # {name: {E0: record}}
    cases = cases_from_results(results)
    res = filter_results(results, cases)
    settings = default_settings()

    recs = records(res)
    if not recs:
        print(f"[smoke] FAIL: no records for {material!r}", file=sys.stderr)
        return 2
    tilts = sorted({r["case"]["tilt_deg"] for r in recs})
    tilt = tilts[len(tilts) // 2]  # a mid-range polar tilt, like the app default
    print(f"[smoke] {len(recs)} records, {len(tilts)} tilts; using tilt_deg={tilt:g}")

    # --- Altair intrinsic-spectrum chart (the app's _spectra_tab) ------------
    altair_png = out_dir / f"{material}_spectrum.png"
    chart = spectrum_chart(res, settings, tilt_deg=tilt, include_brem=True)
    if chart is None:
        print("[smoke] FAIL: spectrum_chart returned None", file=sys.stderr)
        return 3
    chart.save(str(altair_png))  # vl-convert renders Vega-Lite -> PNG
    print(f"[smoke] wrote {altair_png} ({altair_png.stat().st_size} bytes)")

    # --- matplotlib best-spectra grid (the app's 'Best spectra' accordion) ---
    mpl_png = out_dir / f"{material}_best_spectra.png"
    fig = plot_best_spectra(res, settings, top_n=6, cases=cases)
    if fig is None:
        print("[smoke] FAIL: plot_best_spectra returned None", file=sys.stderr)
        return 4
    fig.savefig(mpl_png, dpi=110, bbox_inches="tight")
    print(f"[smoke] wrote {mpl_png} ({mpl_png.stat().st_size} bytes)")

    # --- top-geometry ranking table (the app's _rankings_tab) ----------------
    df = top_geometries(res, settings, top_n=10, select="quality_peak")
    print(f"[smoke] top geometries ({len(df)} rows):")
    print(df.head(10).to_string(index=False))

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
