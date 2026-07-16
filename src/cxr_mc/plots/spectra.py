"""spectra

Intrinsic-spectrum figures: by-energy, full-range, peak-vs-tilt, mosaic, comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..montecarlo import (
    convolve_detector,
    detector_efficiency,
    mosaic_psi_rad,
)
from ..results import (
    best_azimuth,
    line_fwhm_eV,
    records,
    records_for_cases,
    selection_score,
)
from ._common import (
    _best_azimuth,
    _case_title,
    _line_brem,
    _metrics_map,
    _per_tilt_figs,
)
from ._style import (
    COLORS,
    energy_color,
)


# ---- intrinsic spectra -------------------------------------------------------
def plot_tilt_panel(ax, group, settings, include_brem=True, collapse_azimuth=False):
    """One panel at fixed (energy, polar tilt): one curve per azimuth, or just
    the best azimuth if ``collapse_azimuth``."""
    group = _best_azimuth(group, collapse_azimuth)
    group = sorted(group, key=lambda r: r["case"]["tilt_azim_deg"])
    for i, r in enumerate(group):
        c = COLORS[i % len(COLORS)]
        az = r["case"]["tilt_azim_deg"]
        line_det, brem_det = _line_brem(r, settings)
        y = (line_det + brem_det) if include_brem else line_det
        ax.plot(
            r["E_grid"] / 1e3,
            y * r["scale"],
            color=c,
            ls="-",
            lw=1.1,
            label=rf"$\phi={az:.1f}\degree$",
        )
        if include_brem:
            ax.plot(r["E_grid"] / 1e3, brem_det * r["scale"], color=c, ls="--", lw=0.7)
    case = group[0]["case"]
    ax.set_title(
        _case_title(case, "", e0_keV=case["E0_keV"], tilt_fmt="g"),
        fontsize=11,
    )
    ax.set_xlabel("Photon energy (keV)", fontsize=10)
    ax.set_ylabel("Intensity (Phs/eV/s/nA)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    ax.legend(title=("dashed: brem" if include_brem else None), fontsize=8)


def _draw_by_energy(fig, trecs, settings, include_brem=True, collapse_azimuth=True):
    """Render ONE polar tilt onto ``fig`` (cleared first): every beam energy
    overlaid, INTRINSIC spectra (the detector view is the separate Eagle XO
    browser, kind='eaglexo'). Single axis -> fits the screen without scrolling."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0]
        if not grp:
            continue
        grp = _best_azimuth(grp, collapse_azimuth)
        c = energy_color(E0, energies)
        for r in sorted(grp, key=lambda r: r["case"]["tilt_azim_deg"]):
            az = r["case"]["tilt_azim_deg"]
            lbl = rf"{E0:g} keV ($\phi={az:0.1f}\degree$)"
            line_det, brem_det = _line_brem(r, settings, convolve=False)
            y = (line_det + brem_det) if include_brem else line_det
            ax.plot(r["E_grid"], y * r["scale"], color=c, lw=1.3, label=lbl)
            if include_brem:
                ax.plot(r["E_grid"], brem_det * r["scale"], color=c, ls="--", lw=0.6)
    case = trecs[0]["case"]
    tag = "best azimuth/energy" if collapse_azimuth else "all azimuths"
    ax.set_title(
        _case_title(case, f"intrinsic ({tag})"),
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.grid(alpha=0.3)
    ax.legend(title=("dashed: brem" if include_brem else None), fontsize=9)
    fig.tight_layout()


def plot_by_energy(results, settings, include_brem=True, collapse_azimuth=True):
    """One figure PER POLAR TILT, every beam energy overlaid (best azimuth when
    ``collapse_azimuth``); INTRINSIC spectra (detector view = the Eagle XO
    browser). For click-through use ``browse(results, settings, kind="by_energy")``."""
    return _per_tilt_figs(
        records(results),
        settings,
        _draw_by_energy,
        (9.5, 5.3),
        include_brem=include_brem,
        collapse_azimuth=collapse_azimuth,
    )


def _draw_full_spectrum(
    fig, trecs, settings, collapse_azimuth=True, logy=True, logx=True, floor_frac=1e-5
):
    """Render ONE polar tilt of the full measured-range view onto ``fig``: sharp
    lines + wide brem out to the beam energy, log-log, INTRINSIC (single axis; the
    detector view is the Eagle XO browser).

    Broad-spectrum view: the y-floor is set from the brem CONTINUUM (its ~1st
    percentile across the full range), not a fixed fraction of the line peak, so
    the whole bremsstrahlung shoulder out to the beam energy stays on-screen
    instead of being clipped under a tall, narrow line. ``floor_frac`` only caps
    the dynamic range (deepest allowed = floor_frac x the peak)."""
    fig.clear()
    ax = fig.subplots(1, 1)
    energies = sorted({r["case"]["E0_keV"] for r in trecs})
    ymax = 0.0
    ybrem_lo = np.inf  # smallest positive brem value shown -> the broad-spectrum floor
    xmax = 0.0
    for _i, E0 in enumerate(energies):
        grp = [r for r in trecs if r["case"]["E0_keV"] == E0 and r.get("brem_wide") is not None]
        if not grp:
            continue
        grp = _best_azimuth(grp, collapse_azimuth)
        r = grp[0]
        c = energy_color(E0, energies)
        az = r["case"]["tilt_azim_deg"]
        lbl = rf"{E0:g} keV ($\phi$={az:.1f}$\degree$)"
        Eb = r["E_grid_brem"]
        qe_b = detector_efficiency(Eb) if settings.apply_detector_qe else 1.0
        brem_wide_det = r["brem_wide"] * qe_b * r["scale"]
        xmax = max(xmax, float(Eb[-1]))  # full brem grid -> beam energy
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        total_line = (line_det + brem_det) * r["scale"]
        ax.plot(Eb, brem_wide_det, color=c, ls="--", lw=0.7, alpha=0.85)
        ax.plot(r["E_grid"], total_line, color=c, lw=1.2, label=lbl)
        ymax = max(ymax, float(np.nanmax(total_line)) if total_line.size else 0.0)
        ymax = max(ymax, float(np.nanmax(brem_wide_det)) if brem_wide_det.size else 0.0)
        bpos = brem_wide_det[np.isfinite(brem_wide_det) & (brem_wide_det > 0)]
        if bpos.size:
            ybrem_lo = min(ybrem_lo, float(np.percentile(bpos, 1)))
    case = trecs[0]["case"]
    ax.set_title(
        _case_title(case, "full measured range, intrinsic (dashed = brem)"),
        fontsize=12,
    )
    if logy and ymax > 0:
        ax.set_yscale("log")
        # floor driven by the brem continuum so the broad spectrum stays visible,
        # but never deeper than floor_frac x the peak (guards a near-zero edge)
        floor = max(ybrem_lo if np.isfinite(ybrem_lo) else 0.0, ymax * floor_frac)
        ax.set_ylim(floor, ymax * 2)
    else:
        ax.set_ylim(bottom=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    if xmax > 0:
        ax.set_xlim(50.0)  # span the full brem grid (to the beam energy)
    else:
        ax.margins(x=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_full_spectrum(results, settings, collapse_azimuth=True, logy=True, floor_frac=1e-5):
    """Full measured-range view (sharp lines on the wide brem, log-log), ONE figure
    per polar tilt. The x-axis spans the full brem grid (to the beam energy) and
    the y-floor follows the brem continuum, so the broad bremsstrahlung shoulder
    is on-screen instead of clipped under the lines (``floor_frac`` caps the depth
    at floor_frac x the peak). For click-through use
    ``browse(results, settings, kind="full")``. Needs records run with a separate
    ``E_grid_brem`` (``brem_wide`` present)."""
    recs = [r for r in records(results) if r.get("brem_wide") is not None]
    return _per_tilt_figs(
        recs,
        settings,
        _draw_full_spectrum,
        (9.5, 5.3),
        empty_msg="no wide-brem records -- set E_grid_brem in the Sweep and re-run",
        collapse_azimuth=collapse_azimuth,
        logy=logy,
        floor_frac=floor_frac,
    )


def plot_peak_vs_tilt(results, settings):
    """Overview for big sweeps: best-azimuth peak spectral flux vs polar tilt,
    one line per beam energy. Peak = max(spectrum) * scale * beam current."""
    recs = best_azimuth(records(results))
    if not recs:
        print("no results yet")
        return None
    by_E = {}
    for r in recs:
        by_E.setdefault(r["case"]["E0_keV"], []).append(r)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _i, (E0, rs) in enumerate(sorted(by_E.items())):
        rs = sorted(rs, key=lambda r: r["case"]["tilt_deg"])
        tilts = [r["case"]["tilt_deg"] for r in rs]
        peak = [float(np.max(r["spec"])) * r["scale"] * settings.beam_current_na for r in rs]
        ax.plot(tilts, peak, "o-", color=energy_color(E0, by_E), label=f"{E0:g} keV")
    ax.set_xlabel(r"polar tilt $\theta_\mathrm{tilt}$ (deg)")
    ax.set_ylabel("best-azimuth peak (Phs/eV/s)")
    ax.set_title("Peak spectral flux vs polar tilt (best azimuth per point)")
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy")
    fig.tight_layout()
    return fig


# ---- crystal mosaicity (analytic broadening) ---------------------------------
def plot_mosaic_comparison(r, settings, grades_deg=(None, 0.4, 0.8, 3.5), ax=None):
    """Detector-convolved line spectrum of ONE record ``r`` overlaid for several
    crystal mosaic grades -- WITHOUT re-running transport. The analytic mosaic model
    (montecarlo.mosaic_fwhm_eV) only changes the convolution FWHM and leaves the
    intrinsic spectrum fixed, so each grade is just a re-convolution of the same
    ``r["spec"]``. ``grades_deg`` entries are mosaic rocking-curve FWHM in degrees;
    None = perfect crystal (no mosaic term). Handy for HOPG: ZYA 0.4 / ZYB 0.8 /
    ZYH 3.5 deg. The record can come from a mosaic=False run -- the broadening is
    re-derived here per grade. Returns the Figure."""
    case = r["case"]
    E, E_pk = r["E_grid"], r["E_pk"]
    qe = detector_efficiency(E) if settings.apply_detector_qe else 1.0
    line_in = r["spec"] * qe
    psi = mosaic_psi_rad(case, E_pk)

    curves = []  # (label, fwhm, detected)
    for grade in grades_deg:
        if grade is None:
            fwhm = line_fwhm_eV(case, E_pk, None)
            lbl = "perfect"
        else:
            fwhm = line_fwhm_eV(case, E_pk, np.deg2rad(grade))
            lbl = rf"mosaic {grade:g}$\degree$"
        det = convolve_detector(E, line_in, fwhm) * r["scale"]
        curves.append((lbl, fwhm, det))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    else:
        fig = ax.figure
    ymax = 0.0
    for lbl, fwhm, det in curves:
        fin = det[np.isfinite(det)]
        if fin.size:
            ymax = max(ymax, float(fin.max()))
        ax.plot(E, det, lw=1.5, label=f"{lbl} (FWHM {fwhm:.0f} eV)")
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.08)
    # zoom to the line: +-6 of the broadest FWHM about the peak, clamped to the grid
    half = 6.0 * max(f for _, f, _ in curves)
    ax.set_xlim(max(float(E[0]), E_pk - half), min(float(E[-1]), E_pk + half))
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$E_0$={case['E0_keV']:g} keV, $\theta_\mathrm{{tilt}}$="
        rf"{case['tilt_deg']:.1f}$\degree$ — crystal-mosaic broadening",
        fontsize=12,
    )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Phs/eV/s/nA")
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
        title=(rf"$\psi$(v,g)={np.degrees(psi):.1f}$\degree$" if psi is not None else None),
    )
    return fig


def plot_best_spectra(
    results,
    settings,
    top_n=12,
    select="quality_peak",
    include_brem=True,
    cases=None,
    ncols=3,
    rel_prominence=0.03,
    line_metric="sharpness",
):
    """The top-``top_n`` geometries across the WHOLE sweep, ranked by
    results.selection_score(``select``) -- one intrinsic spectrum panel each,
    titled with the geometry, the score components, and the line quality. The
    answer to "thousands of cases, which few do I look at": instead of paging
    every polar tilt, see the best dozen at a glance. ``select`` defaults to
    peak_flux x line_quality (bright AND well-defined), not the raw peak the
    per-tilt browser collapses on -- so spurious tall spikes don't win."""
    recs = records_for_cases(results, cases)
    if not recs:
        print("no results yet")
        return None
    metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
    ranked = sorted(recs, key=lambda r: selection_score(metrics[id(r)], select), reverse=True)[
        :top_n
    ]
    all_E = {r["case"]["E0_keV"] for r in recs}
    n = len(ranked)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for k, r in enumerate(ranked):
        ax = axes[k // ncols][k % ncols]
        c, m = r["case"], metrics[id(r)]
        line_det, brem_det = _line_brem(r, settings, convolve=False)
        E = r["E_grid"] / 1e3
        col = energy_color(c["E0_keV"], all_E)
        ax.plot(
            E,
            (line_det + brem_det if include_brem else line_det) * r["scale"],
            color=col,
            lw=1.1,
        )
        if include_brem:
            ax.plot(E, brem_det * r["scale"], color=col, ls="--", lw=0.5)
        ax.set_title(
            rf"#{k + 1} {c['name'].split()[0]} {c['E0_keV']:g}keV"
            "\n"
            rf"$\theta$={c['tilt_deg']:g}$\degree$ $\phi$={c['tilt_azim_deg']:g}$\degree$  "
            rf"q={m['line_quality']:.2f}, {m['line_eV']:.0f}eV",
            fontsize=7.5,
        )
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(alpha=0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.supxlabel("Photon energy (keV)", fontsize=9)
    fig.supylabel("Intensity (Phs/eV/s/nA)", fontsize=9)
    fig.suptitle(f"Top {n} geometries by {select} (dashed = brem)", fontsize=12)
    return fig


def _separate_annotation_boxes(fig, annotations):
    """Offset labels vertically until their rendered bounding boxes do not overlap."""
    placed_boxes = []
    offsets = [0]
    for distance in range(12, 12 * (len(annotations) + 1), 12):
        offsets.extend((distance, -distance))
    for annotation in annotations:
        for offset in offsets:
            annotation.set_position((6, offset))
            fig.canvas.draw()
            box = annotation.get_window_extent(fig.canvas.get_renderer())
            if not any(box.overlaps(placed) for placed in placed_boxes):
                placed_boxes.append(box)
                break


def plot_material_comparison(
    results_by_material,
    settings,
    select="quality_peak",
    rel_prominence=0.03,
    line_metric="sharpness",
    beam_energy_keV=None,
    min_line_quality: float | None = 0.25,
):
    """Cross-material headline: for each material's results store, find the single
    BEST geometry/energy (results.selection_score ``select``) and plot its
    dominant coherent line ENERGY vs its integrated line FLUX -- one point per
    material, coloured by line-definition quality. Answers "which crystal gives
    the brightest well-defined line, and at what energy" for comparison against
    the paper's catalogue.

    ``results_by_material`` : ``{label: results_store}``, e.g. built in the
    notebook with ``{m: load_checkpoint(m) for m in CATALOG.material_keys}``
    (skip empties).  ``beam_energy_keV`` optionally restricts each material to
    that beam energy before selecting its best geometry. ``min_line_quality``
    rejects ill-defined-line candidates before ranking; ``None`` disables that
    gate. Labels are offset automatically when their rendered bounding boxes
    would overlap."""
    pts = []  # (label, line_eV, line_flux, quality, case)
    for label, results in results_by_material.items():
        recs = records(results)
        if not recs:
            continue
        metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
        candidates = [
            r
            for r in recs
            if (beam_energy_keV is None or r["case"]["E0_keV"] == beam_energy_keV)
            and (
                min_line_quality is None
                or metrics[id(r)]["line_quality"] >= min_line_quality
            )
        ]
        if select == "line_brem_ratio":
            candidates = [r for r in candidates if np.isfinite(metrics[id(r)]["line_brem_ratio"])]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: selection_score(metrics[id(r)], select))
        m = metrics[id(best)]
        pts.append((label, m["line_eV"], m["line_flux"], m["line_quality"], best["case"]))
    if not pts:
        print("no results in any material")
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sc = ax.scatter(
        [p[1] / 1e3 for p in pts],
        [p[2] for p in pts],
        c=[p[3] for p in pts],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=110,
        edgecolor="k",
        zorder=3,
    )
    annotations = []
    for label, eV, flux, _q, case in pts:
        annotations.append(
            ax.annotate(
                (
                    f"  {label} ({case['E0_keV']:g} keV, "
                    f"θ={case['tilt_deg']:g}°, φ={case['tilt_azim_deg']:g}°)"
                ),
                (eV / 1e3, flux),
                fontsize=8,
                va="center",
                xytext=(6, 0),
                textcoords="offset points",
            )
        )
    ax.set_yscale("log")
    ax.set_xlabel("dominant coherent line energy (keV)")
    ax.set_ylabel("integrated line flux at best geometry (Phs/s)")
    selection_titles = {
        "quality_peak": "highest line-definition quality",
        "peak": "highest peak flux",
        "line_brem_ratio": "highest local line-to-bremsstrahlung ratio",
    }
    selection_title = selection_titles.get(select, select.replace("_", " "))
    energy_scope = (
        "all beam energies" if beam_energy_keV is None else f"{beam_energy_keV:g} keV beam energy"
    )
    quality_scope = (
        "" if min_line_quality is None else f", line quality >= {min_line_quality:g}"
    )
    ax.set_title(
        f"Cross-material comparison — {selection_title} ({energy_scope}{quality_scope})",
        fontsize=12,
    )
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.12)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("line-definition quality")
    fig.tight_layout()
    _separate_annotation_boxes(fig, annotations)
    return fig
