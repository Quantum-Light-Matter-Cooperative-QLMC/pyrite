"""_frames

Renderer-neutral tidy-data builders. The parametric-sweep half: both
:mod:`pyrite.plots.mpl.sweeps` (matplotlib) and :mod:`pyrite.plots.altair.sweeps`
(Altair) render FROM these builders -- the per-cell/per-point "reduce every
swept case to its best record" reduction lives here exactly once. The
electron-trajectory half: :func:`_trajectory_data` runs the transport and
projects the cascade into the beam-detector plane, and :func:`survival_frame`
reduces it to a penetration/survival table; :mod:`pyrite.plots.mpl.trajectories`,
:mod:`pyrite.plots.altair.trajectories` and :mod:`pyrite.plots.plotly.trajectories`
all render from these. No matplotlib, Altair or Plotly imports; renderers turn
the returned ``pandas.DataFrame`` / dict / guard values into figures.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..campaign.sweep import fmt_thickness
from ..montecarlo import (
    simulate_trajectories,
    tilted_geometry,
)
from ..results import records, records_for_cases, selection_score
from ._common import (
    _beam_detector_basis,
    _beam_phase_space,
    _case_of,
    _groove_spec,
    _metrics_map,
)


def _ndistinct(recs, field):
    return len({r["case"][field] for r in recs if field in r["case"]})


_FALLBACK_X = ("tilt_deg", "thickness_ang", "E0_keV", "tilt_azim_deg", "B_ang2")


def _effective_x(recs, x, hue):
    """If ``x`` sweeps <2 values, substitute the first fallback knob that
    actually sweeps (and isn't the hue), so a 1-D scan/scan-frame is a
    meaningful curve instead of a vertical stack. Shared by
    :func:`pyrite.plots.mpl.sweeps.plot_metric_vs` and :func:`metric_vs_frame`."""
    if _ndistinct(recs, x) >= 2:
        return x
    return next(
        (f for f in _FALLBACK_X if f != x and f != hue and _ndistinct(recs, f) >= 2),
        x,
    )


def scan_mode(recs, x, y, heatmap_min, force=None):
    """Heatmap-vs-lines auto-pick shared by ``plot_scan`` (matplotlib) and
    ``scan_charts`` (Altair): heatmap when BOTH ``x`` and ``y`` sweep at least
    ``heatmap_min`` values, else lines. ``force`` ("heatmap"|"lines")
    overrides the pick."""
    if force in ("heatmap", "lines"):
        return force
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    return "heatmap" if nx >= heatmap_min and ny >= heatmap_min else "lines"


def pick_hue(recs, x, y, panel, hue=None):
    """Line-mode axis/hue pick shared by ``plot_scan`` and ``scan_charts``:
    the denser of ``x``/``y`` becomes the line x-axis, the sparser becomes
    ``hue`` (unless ``hue`` is given explicitly), falling back to ``panel``
    if the sparser axis doesn't vary, and finally to the sparser axis itself
    if nothing else varies (a single line). Returns ``(line_x, hue)``."""
    nx, ny = _ndistinct(recs, x), _ndistinct(recs, y)
    line_x, other = (x, y) if nx >= ny else (y, x)
    if hue is not None:
        return line_x, hue
    if _ndistinct(recs, other) >= 2:
        return line_x, other
    if _ndistinct(recs, panel) >= 2:
        return line_x, panel
    return line_x, other


# ---- axis/metric display registries (shared with pyrite.plots.mpl.sweeps) --------
# Per case field: (axis label, divide-to-display, display unit, value format).
# Lets ANY swept knob be a heatmap/scan axis with sensible labels and units.
_AXIS_SPECS = {
    "tilt_deg": ("polar tilt", 1.0, "deg", "{:g}"),
    "tilt_azim_deg": ("azimuthal tilt", 1.0, "deg", "{:g}"),
    "E0_keV": ("beam energy", 1.0, "keV", "{:g}"),
    "thickness_ang": ("thickness", 1e4, r"μm", "{:g}"),
    "B_ang2": ("B-factor", 1.0, r"$\AA^2$", "{:g}"),
}

# Line-characterization maps are meaningless where the line is ill-defined --
# either near-zero emission OR a broad ramp / a cluster of comparable peaks
# (low line_quality, see results.line_quality). Gate these by BOTH peak flux
# and line_quality. The always-well-defined maps (peak_flux, coherent_flux,
# total_flux) and the diagnostic line_quality map itself are never gated.
_FLUX_GATED = {"line_eV", "fwhm_eV", "line_frac", "line_flux"}


def _axis_disp(key, vals):
    """Swept raw values -> display units (e.g. thickness Angstrom -> microns)."""
    div = _AXIS_SPECS.get(key, (None, 1.0))[1]
    return [float(v) / div for v in vals]


def _value_label(key, v):
    """'30 keV' / '17 um' style label for one swept value."""
    if key == "thickness_ang":
        return fmt_thickness(float(v))
    _lbl, div, unit, fmt = _AXIS_SPECS.get(key, (key, 1.0, "", "{:g}"))
    return f"{fmt.format(float(v) / div)} {unit}".strip()


# (metric key, label + units, colormap)
_HEATMAP_QUANTITIES = [
    ("peak_flux", "peak spectral flux  (Phs/eV/s)", "viridis"),
    ("coherent_flux", "integrated coherent flux, all lines  (Phs/s)", "viridis"),
    ("line_flux", "integrated flux under the dominant line  (Phs/s)", "viridis"),
    ("line_eV", "dominant coherent line energy  (eV)", "plasma"),
    ("fwhm_eV", "dominant line FWHM  (eV)", "magma"),
    ("line_frac", "dominant line / total spectral flux", "cividis"),
    ("line_quality", "line-definition quality  (0-1)", "Greens"),
    ("total_flux", "total integrated flux, lines+brem  (Phs/s)", "viridis"),
]

# Metrics that are NOT in the default heatmap set (so a plain plot_scan doesn't
# grow an extra panel) but get a proper label + colormap when asked for by name,
# e.g. plot_scan(..., quantities=["coherent_brem_ratio"]). coherent_brem_ratio is
# ungated (not in _FLUX_GATED): it's the CXR/brem contrast, valid wherever brem>0.
_EXTRA_QUANTITIES = {
    "coherent_brem_ratio": (
        "coherent / incoherent-brem flux ratio  (CXR / brem)",
        "cividis",
    ),
    # finite-crystal footprint-hit fraction in [0, 1] (1 = every launched
    # electron landed on the crystal). Ungated (not in _FLUX_GATED): a pure
    # geometry diagnostic, valid regardless of line brightness.
    "hit_frac": (
        "electron footprint-hit fraction  (hits / launched)",
        "magma",
    ),
}
_METRIC_LABELS = {key: label for key, label, _ in _HEATMAP_QUANTITIES}
_METRIC_LABELS.update({k: lbl for k, (lbl, _) in _EXTRA_QUANTITIES.items()})


def _resolve_quantity(q):
    """Normalize a quantity spec to a ``(key, label, cmap)`` triple: pass triples
    through, look bare metric keys up in the default + extra registries (cmap
    falls back to viridis for an unknown key)."""
    if not isinstance(q, str):
        return tuple(q)
    if q in _EXTRA_QUANTITIES:
        lbl, cmap = _EXTRA_QUANTITIES[q]
        return (q, lbl, cmap)
    return (q, _METRIC_LABELS.get(q, q), "viridis")


def _axis_label(key):
    spec = _AXIS_SPECS.get(key)
    return key if spec is None else f"{spec[0]} ({spec[2]})"


def metric_vs_frame(
    results,
    settings,
    *,
    x="thickness_ang",
    metric="line_flux",
    hue="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    metrics=None,
):
    """Tidy long-form table for a 1-D metric scan: one row per (hue value, x value),
    ``metric`` reduced over every OTHER swept dimension to its best geometry
    (``results.selection_score`` ``select``). Mirrors the data behind
    :func:`pyrite.plots.plot_metric_vs`, including the single-valued-x fallback.
    ``x`` is in DISPLAY units (e.g. thickness in microns). Columns:
    ``x, metric, hue`` (``hue`` is the ``"30 keV"``-style value label).
    ``metrics`` is a precomputed ``_common._metrics_map`` for THESE records
    (computed here when omitted) -- multi-quantity drivers pass one shared map
    instead of re-deriving it per quantity."""
    recs = records_for_cases(results, cases)
    if not recs:
        return pd.DataFrame(columns=["x", "metric", "hue"])
    x = _effective_x(recs, x, hue)
    if metrics is None:
        metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
    div_x = _AXIS_SPECS.get(x, (None, 1.0))[1]
    rows = []
    for hv in sorted({r["case"][hue] for r in recs}):
        hr = [r for r in recs if r["case"][hue] == hv]
        for xv in sorted({r["case"][x] for r in hr}):
            cell = [r for r in hr if r["case"][x] == xv]
            best = max(cell, key=lambda r: selection_score(metrics[id(r)], select))
            rows.append(
                {
                    "x": float(xv) / div_x,
                    "metric": float(metrics[id(best)][metric]),
                    "hue": _value_label(hue, hv),
                }
            )
    return pd.DataFrame(rows, columns=["x", "metric", "hue"])


def heatmap_frame(
    results,
    settings,
    *,
    quantity="peak_flux",
    x="tilt_azim_deg",
    y="tilt_deg",
    panel="E0_keV",
    select="quality_peak",
    cases=None,
    rel_prominence=0.03,
    line_metric="sharpness",
    min_flux_frac=0.02,
    min_line_quality=0.2,
    metrics=None,
):
    """Tidy long-form table for one heatmap quantity over ``x`` x ``y``, one block
    per ``panel`` value. Each (x, y) cell is reduced to its best record
    (``selection_score`` ``select``); flux-gated quantities blank out cells with
    near-zero emission or an ill-defined line (dropped rows -> gaps), mirroring
    :func:`pyrite.plots.plot_heatmaps`. ``x`` / ``y`` are in DISPLAY units.
    Columns: ``x, y, panel, value, name, panel_raw``. ``name`` is the config
    name of the cell's best record and ``panel_raw`` its raw ``panel`` value --
    both carried so an interactive click on a cell (see
    :func:`pyrite.plots.altair.sweeps.heatmap_select_chart`) maps back to an
    exact geometry / parameter set. ``metrics`` is a precomputed
    ``_common._metrics_map`` for THESE records (computed here when omitted) --
    multi-quantity drivers pass one shared map instead of re-deriving it per
    quantity."""
    cols = ["x", "y", "panel", "value", "name", "panel_raw"]
    recs = records_for_cases(results, cases)
    if not recs:
        return pd.DataFrame(columns=cols)
    if metrics is None:
        metrics = _metrics_map(recs, settings, rel_prominence, line_metric)
    gated = quantity in _FLUX_GATED
    rows = []
    for pv in sorted({r["case"][panel] for r in recs}):
        er = [r for r in recs if r["case"][panel] == pv]
        fmax = max((metrics[id(r)]["peak_flux"] for r in er), default=0.0)
        floor = min_flux_frac * fmax
        best = {}  # (xv, yv) -> (score, rec)
        for r in er:
            ck = (r["case"][x], r["case"][y])
            s = selection_score(metrics[id(r)], select)
            if ck not in best or s > best[ck][0]:
                best[ck] = (s, r)
        for (xv, yv), (_, r) in best.items():
            m = metrics[id(r)]
            if gated and (m["peak_flux"] < floor or m["line_quality"] < min_line_quality):
                continue  # near-zero emission / ill-defined line -> blank
            rows.append(
                {
                    "x": _axis_disp(x, [xv])[0],
                    "y": _axis_disp(y, [yv])[0],
                    "panel": _value_label(panel, pv),
                    "value": float(m[quantity]),
                    "name": r["case"]["name"],
                    "panel_raw": float(pv),
                }
            )
    return pd.DataFrame(rows, columns=cols)


# ---- electron-trajectory data builders -----------------------------------------
# Renderer-neutral electron-transport + projection prep, shared by
# :mod:`pyrite.plots.mpl.trajectories` (matplotlib), :mod:`pyrite.plots.altair.trajectories`
# (Altair) and :mod:`pyrite.plots.plotly.trajectories` (Plotly) -- the same
# transport, only the renderer differs.
C_ANG_PER_FS = 2997.924580  # speed of light [Ang/fs]: age sum(L/beta)[Ang] -> fs


def _trajectory_cases(cases_or_results):
    """Flatten a build_cases list OR a results store into a list of case dicts."""
    if isinstance(cases_or_results, dict):
        return [r["case"] for r in records(cases_or_results)]
    return [_case_of(c) for c in cases_or_results]


def _trajectory_data(
    case,
    Ne,
    seed,
    *,
    beam_fwhm_mm=None,
    crystal_width_mm=None,
    crystal_height_mm=None,
) -> dict[str, Any]:
    """Simulate one case and project the cascade into the beam-detector plane.
    Returns both the projected 2D tracks and true 3D segment endpoints in sample
    coordinates, all in the same display units, plus per-segment energy/age/depth,
    slab + detector directions, and back/through fractions.

    ``beam_fwhm_mm`` (and, for a finite footprint, ``crystal_width_mm`` /
    ``crystal_height_mm``) are forwarded to :func:`simulate_trajectories`, which
    draws each electron's entry point from the Gaussian spot and projects it onto
    the tilted entrance face (grazing-incidence ``1/cos`` stretch). All default
    ``None`` -- the legacy point source entering at the origin, bit-for-bit -- so
    the 2D cross-section callers are unchanged. With a finite crystal footprint,
    entries landing off the crystal are dropped by transport (``n_missed``) and
    never appear as tracks.

    The case's own beam phase space (Twiss policy, bunch, energy spread) rides
    along via :func:`pyrite.plots._common._beam_phase_space`, so a plot shows the
    beam the run used rather than a point source standing in for it. A case that
    sets none of those keys transports exactly as before.

    Multilayer/stacked materials (``case["abs_layers"]`` set -- film-on-substrate,
    e.g. mos2-on-sapphire) are transported through the FULL stack via
    ``layers=abs_layers``, matching the spectrum runner (`montecarlo.runner`);
    without this the electron cascade (and hence the trajectory/penetration
    plots) only ever saw the top film layer, silently dropping the substrate's
    backscatter contribution and reporting only the film's thickness."""
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    phase_space = _beam_phase_space(case)
    if phase_space["transverse_distribution"] is not None:
        # A Twiss policy fixes the spot size as well as the divergence, so the
        # display FWHM would be a second, contradictory answer for <x^2>.
        beam_fwhm_mm = None
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    n_hat = -n_hat
    abs_layers = case.get("abs_layers")
    total_thickness_ang = (
        float(abs_layers[-1][1]) if abs_layers is not None else case["thickness_ang"]
    )
    segs = simulate_trajectories(
        case["E0_keV"],
        Ne,
        total_thickness_ang,
        composition=case["composition"],
        seed=seed,
        beam_dir=beam,
        layers=abs_layers,
        beam_fwhm_mm=beam_fwhm_mm,
        crystal_width_mm=crystal_width_mm,
        crystal_height_mm=crystal_height_mm,
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        **phase_space,
        # Blazed grooves (when the case carries the knob): electrons enter on the
        # relief facets, so the drawn tracks start at z in [0, groove depth) --
        # groove=None (the default for every ungrooved case) is bit-for-bit the
        # legacy flat-face entry.
        groove=_groove_spec(case),
    )
    e1, e2 = _beam_detector_basis(beam, n_hat)
    L, v, r = segs["L_ang"], segs["v_hat"], segs["r_mid"]
    start = r - 0.5 * L[:, None] * v
    u, ulab = (1e4, r"$\mu$m") if total_thickness_ang >= 1e4 else (10.0, "nm")

    # Continuous per-electron tracks (not a loose segment cloud): order segments by
    # (electron, age) so each electron's segment START points form a polyline --
    # consecutive starts share an endpoint, so they trace the real zig-zag path --
    # then break with NaN between electrons. This is what makes the tracks read as
    # paths (the old per-segment LineCollection got faint at the cool, slow tail).
    order = np.lexsort((segs["t_ang"], segs["elec_id"]))
    sx = (start @ e1)[order] / u
    sy = (start @ e2)[order] / u
    sE = segs["E_keV"][order]
    brk = np.flatnonzero(np.diff(segs["elec_id"][order]) != 0) + 1
    px = np.insert(sx, brk, np.nan)
    py = np.insert(sy, brk, np.nan)
    pE = np.insert(sE, brk, np.nan)

    z = np.array([0.0, 0.0, 1.0])  # slab normal in the sample frame
    ndet = np.array([n_hat @ e1, n_hat @ e2])
    ndet = ndet / np.linalg.norm(ndet)
    nslab = np.array([z @ e1, z @ e2])
    nn = np.linalg.norm(nslab)
    nslab = nslab / nn if nn > 1e-9 else np.array([1.0, 0.0])
    return dict(
        px=px,
        py=py,
        pE=pE,
        pts=np.column_stack([px, py]),  # for the shared-frame extent
        E=segs["E_keV"],
        t_fs=segs["t_ang"] / C_ANG_PER_FS,
        z_u=(r[:, 2] + 0.5 * L * v[:, 2])
        / u,  # depth of segment ENDPOINT; transmitted electrons reach thick exactly
        elec_id=segs["elec_id"],  # emitting electron index, per segment
        L=segs["L_ang"],
        initial_r_ang=np.asarray(segs["initial_r_ang"], dtype=float),
        initial_v_hat=np.asarray(segs["initial_v_hat"], dtype=float),
        initial_E_keV=np.asarray(segs["initial_E_keV"], dtype=float),
        initial_t0_ang=np.asarray(segs["initial_t0_ang"], dtype=float),
        start_xyz=start / u,
        end_xyz=(r + 0.5 * L[:, None] * v) / u,
        # Groove-gap flights remain diagnostic-only: they do not radiate and
        # therefore must never be appended to E/start_xyz/end_xyz.
        vacuum_start_xyz=np.asarray(segs.get("vacuum_start_ang", np.empty((0, 3))), dtype=float)
        / u,
        vacuum_end_xyz=np.asarray(segs.get("vacuum_end_ang", np.empty((0, 3))), dtype=float) / u,
        vacuum_E=np.asarray(segs.get("vacuum_E_keV", np.empty(0)), dtype=float),
        vacuum_t_fs=np.asarray(segs.get("vacuum_t_ang", np.empty(0)), dtype=float) / C_ANG_PER_FS,
        vacuum_elec_id=np.asarray(
            segs.get("vacuum_elec_id", np.empty(0, dtype=np.int64)), dtype=np.int64
        ),
        beam=np.asarray(beam, dtype=float),
        detector=np.asarray(n_hat, dtype=float),
        ndet=ndet,
        nslab=nslab,
        u=u,
        ulab=ulab,
        thick=total_thickness_ang / u,
        # internal layer boundaries (display units), excluding the final z_bot
        # (== thick, already the slab's back face) -- empty for a single slab.
        layer_bounds=(
            [float(z_bot) / u for (_, z_bot, _) in abs_layers[:-1]]
            if abs_layers is not None
            else []
        ),
        eta=100.0 * segs["n_backscattered"] / segs["Ne"],
        thru=100.0 * segs["n_transmitted"] / segs["Ne"],
        Ne=segs["Ne"],
    )


def _trajectory_frame(pts_list, pct=99.0, pad=0.12, beam_frac=0.16):
    """ONE shared (xlo, xhi, ylo, yhi) for a set of panels, from the robust
    (1st/99th-percentile) extent of all their track vertices, expanded to include
    the origin and padded. Sharing it across tilts is what makes only the slab
    rotate frame-to-frame (the old per-panel autoscale was the "scaling is
    inconsistent" complaint). Symmetric in y (beam axis centred); the left margin
    always clears the beam arrow + label."""
    pts = np.concatenate([np.asarray(s).reshape(-1, 2) for s in pts_list], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    xlo = min(0.0, float(np.percentile(pts[:, 0], 100 - pct)))
    xhi = max(0.0, float(np.percentile(pts[:, 0], pct)))
    ymax = float(np.percentile(np.abs(pts[:, 1]), pct))
    sx = max(xhi - xlo, 1e-6)
    xlo -= pad * sx
    xhi += pad * sx
    yhi = max(ymax * (1.0 + pad), 1e-6)
    aL = beam_frac * (xhi - xlo)
    xlo = min(xlo, -1.5 * aL)  # room for the beam arrow + label
    return (float(xlo), float(xhi), float(-yhi), float(yhi))


def _square_frame(frame):
    """Expand the shorter side of a (xlo, xhi, ylo, yhi) frame symmetrically so it
    is SQUARE -- no data is cropped, the extra room becomes centred margin. With
    set_aspect("equal") this lets square subplot boxes hold the tracks without the
    skinny-strip letterboxing the wide native frame produced (the trajectory-grid
    sizing fix)."""
    xlo, xhi, ylo, yhi = frame
    w, h = xhi - xlo, yhi - ylo
    if w > h:
        pad = 0.5 * (w - h)
        ylo, yhi = ylo - pad, yhi + pad
    elif h > w:
        pad = 0.5 * (h - w)
        xlo, xhi = xlo - pad, xhi + pad
    return (float(xlo), float(xhi), float(ylo), float(yhi))


def _max_depth_per_electron(data):
    """Deepest point each electron reaches (max over its segment depths), with the
    tiny negative excursions of backscattered electrons clipped to 0 -- the
    reduction behind :func:`survival_frame` (and, through it,
    :func:`pyrite.plots.mpl.trajectories.plot_penetration_survival`)."""
    max_depth = np.full(data["Ne"], -np.inf)
    np.maximum.at(max_depth, data["elec_id"], data["z_u"])
    return np.clip(max_depth[np.isfinite(max_depth)], 0.0, None)


def survival_frame(data_by_energy, *, n_bins=80, depth_frac=True):
    """Tidy long-form survival table from ``{E0_keV: _trajectory_data dict}``: one
    row per (beam energy, depth sample), ``survival`` = % of incident electrons
    reaching at least that depth. ``depth`` is depth/thickness when ``depth_frac``
    else absolute (display units). Shared by
    :func:`pyrite.plots.mpl.trajectories.plot_penetration_survival` (matplotlib)
    and :func:`pyrite.plots.altair.trajectories.penetration_survival_chart`
    (Altair). Columns: ``depth, survival, energy`` (``energy`` is the
    ``"30 keV"``-style label)."""
    rows = []
    for E0 in sorted(data_by_energy):
        d = data_by_energy[E0]
        depths = _max_depth_per_electron(d)
        thick = d["thick"]
        x = depths / thick if depth_frac else depths
        zmax = 1.0 if depth_frac else float(thick)
        zs = np.linspace(0.0, zmax, n_bins)
        surv = 100.0 * np.array([float((x >= z).mean()) if x.size else 0.0 for z in zs])
        label = _value_label("E0_keV", E0)
        for z, s in zip(zs, surv, strict=False):
            rows.append({"depth": float(z), "survival": float(s), "energy": label})
    return pd.DataFrame(rows, columns=["depth", "survival", "energy"])
