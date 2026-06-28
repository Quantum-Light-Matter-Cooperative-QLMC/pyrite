"""altair_trajectories

Altair / Vega-Lite renderers for the electron-penetration figures -- the
interactive counterparts of the matplotlib :mod:`cxr_mc.plots.trajectories`
views. Same contract as :mod:`cxr_mc.plots.altair_spectra`: these reuse the exact
electron-transport + projection prep (``trajectories._trajectory_data`` /
``_trajectory_cases``), so the physics is identical -- only the renderer differs.
Not re-exported from ``cxr_mc.plots`` (frozen export-set guard); import from the
submodule:

    from cxr_mc.plots.altair_trajectories import penetration_survival_chart

Scope:

* ``penetration_survival_chart`` -- a direct port: surviving-electron fraction vs
  depth, one monotone line per beam energy. A cheap line chart, an ideal Vega-Lite
  fit.
* ``trajectory_chart`` -- an INTERACTIVE, low-electron-count vector view of one
  penetration cross-section (energy-coloured tracks you can pan/zoom). Tracks are
  drawn as per-segment ``mark_rule`` marks (see ``track_segments_frame``), because
  a continuous ``color`` on a grouped ``mark_line`` renders nothing in Vega-Lite.
  The dense datashader raster (``plot_electron_trajectories`` /
  ``plot_trajectory_grid``)
  intentionally STAYS on matplotlib: it rasterizes tens of thousands of segments,
  which Vega-Lite cannot draw as vector marks, and it is not a performance pain
  point (datashader handles it and renders fine in marimo).

The ``*_frame`` builders take already-simulated ``_trajectory_data`` dicts so they
are unit-testable without the Monte Carlo; the ``*_chart`` entry points run the
transport then delegate. See :mod:`cxr_mc.plots.altair_spectra` for the Vega-Lite
5000-row cap note -- keep ``Ne`` modest in ``trajectory_chart`` (vector tracks).
"""

import altair as alt
import numpy as np
import pandas as pd

from .sweeps import _value_label
from .trajectories import (
    _case_of,
    _square_frame,
    _trajectory_cases,
    _trajectory_data,
    _trajectory_frame,
)


# ---- penetration / survival --------------------------------------------------
def _max_depth_per_electron(data):
    """Deepest point each electron reaches (max over its segment depths), with the
    tiny negative excursions of backscattered electrons clipped to 0 -- the same
    reduction :func:`cxr_mc.plots.plot_penetration_survival` does."""
    max_depth = np.full(data["Ne"], -np.inf)
    np.maximum.at(max_depth, data["elec_id"], data["z_u"])
    return np.clip(max_depth[np.isfinite(max_depth)], 0.0, None)


def survival_frame(data_by_energy, *, n_bins=80, depth_frac=True):
    """Tidy long-form survival table from ``{E0_keV: _trajectory_data dict}``: one
    row per (beam energy, depth sample), ``survival`` = % of incident electrons
    reaching at least that depth. ``depth`` is depth/thickness when ``depth_frac``
    else absolute (display units). Mirrors the curve behind
    :func:`cxr_mc.plots.plot_penetration_survival`. Columns:
    ``depth, survival, energy`` (``energy`` is the ``"30 keV"``-style label)."""
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


def penetration_survival_chart(
    cases_or_results,
    *,
    Ne=500,
    seed=0,
    n_bins=80,
    tilt=None,
    depth_frac=True,
    width=720,
    height=420,
):
    """Interactive surviving-electron fraction vs penetration depth, one monotone
    line per beam energy at one polar tilt. The Altair counterpart of
    :func:`cxr_mc.plots.plot_penetration_survival`. ``tilt`` picks the polar tilt
    (nearest; default closest to normal incidence). Returns an
    :class:`altair.Chart`, or ``None`` when there are no cases."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    t = min(tilts, key=lambda x: abs(x - (0.0 if tilt is None else tilt)))
    energies = sorted({c["E0_keV"] for c in cases})
    data_by_energy = {}
    for E0 in energies:
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is not None:
            data_by_energy[E0] = _trajectory_data(c, Ne, seed)
    if not data_by_energy:
        return None
    df = survival_frame(data_by_energy, n_bins=n_bins, depth_frac=depth_frac)
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = "um" if case0["thickness_ang"] >= 1e4 else "nm"
    x_title = "depth / thickness" if depth_frac else f"penetration depth ({ulab})"
    title = (
        f"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} um, "
        f"theta_tilt={t:g} deg -- electron penetration / survival"
    )
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.9)
        .encode(
            x=alt.X("depth:Q", title=x_title),
            y=alt.Y(
                "survival:Q",
                title="surviving electrons (% of N0)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color("energy:N", title="beam energy"),
            tooltip=["energy:N", "depth:Q", "survival:Q"],
        )
    )
    return chart.properties(width=width, height=height, title=title).interactive()


# ---- interactive low-Ne trajectory cross-section -----------------------------
def tracks_frame(data):
    """Tidy per-vertex track table from a ``_trajectory_data`` dict: the
    NaN-separated polylines (``px``/``py``/``pE``) split into one ``track`` group
    per electron. Columns: ``x, y, E, track`` (NaN break rows dropped). This is the
    per-vertex basis for :func:`track_segments_frame`, which turns it into the
    per-segment table :func:`trajectory_chart` actually draws (a continuous
    ``color`` on a grouped ``mark_line`` renders nothing -- see below)."""
    px, py, pE = data["px"], data["py"], data["pE"]
    track = np.cumsum(np.isnan(px))  # increments at each NaN separator
    df = pd.DataFrame({"x": px, "y": py, "E": pE, "track": track})
    return df.dropna(subset=["x", "y"]).reset_index(drop=True)


def track_segments_frame(data):
    """Per-SEGMENT table from a ``_trajectory_data`` dict: one row per consecutive
    vertex pair WITHIN a single electron's polyline -- a segment from ``(x, y)`` to
    ``(x2, y2)`` coloured by its start-vertex energy ``E``. Columns:
    ``x, y, x2, y2, E``.

    This (not :func:`tracks_frame`) is what :func:`trajectory_chart` draws, via a
    Vega-Lite ``mark_rule`` (one straight segment per row). A grouped ``mark_line``
    with a *continuous* ``color`` encoding renders NOTHING: Vega-Lite groups the
    line by the colour field, so a per-vertex-varying energy leaves one point per
    group and no path is stroked. Independent 2-point segments dodge that while
    keeping the energy gradient along the track. One row per segment (not two
    vertices) keeps the row count at ~the vertex count, under the Vega-Lite
    5000-row cap. Rows are sorted by ``E`` ascending so the highest-energy segments
    stroke last (on top) -- matching the datashader ``ds.max("E")`` intent."""
    df = tracks_frame(data)
    v = df[["x", "y", "E", "track"]].to_numpy()
    if len(v) < 2:
        return pd.DataFrame(columns=["x", "y", "x2", "y2", "E"])
    same = v[:-1, 3] == v[1:, 3]  # consecutive vertices share an electron
    seg = pd.DataFrame(
        {
            "x": v[:-1, 0][same],
            "y": v[:-1, 1][same],
            "x2": v[1:, 0][same],
            "y2": v[1:, 1][same],
            "E": v[:-1, 2][same],
        }
    )
    return seg.sort_values("E").reset_index(drop=True)


def _segment_df(p0, p1):
    return pd.DataFrame({"x": [p0[0], p1[0]], "y": [p0[1], p1[1]]})


def trajectory_chart(
    rec_or_case,
    *,
    Ne=40,
    seed=0,
    E_cut=5.0,
    frame=None,
    width=480,
):
    """Interactive vector view of ONE electron-penetration cross-section in the
    beam-detector plane: energy-coloured per-electron tracks (turbo) with the
    crystal entrance/exit faces (grey) and the beam (red) + detector (green)
    arrows. Pan/zoom and hover individual tracks. Keep ``Ne`` modest (vector
    marks, not a datashader raster -- see the module docstring). Returns an
    :class:`altair.Chart`, or ``None`` when the cascade is empty."""
    case = _case_of(rec_or_case)
    data = _trajectory_data(case, Ne, seed)
    seg_df = track_segments_frame(data)
    if seg_df.empty:
        return None
    if frame is None:
        frame = _square_frame(_trajectory_frame([data["pts"]]))
    xlo, xhi, ylo, yhi = frame
    E0 = case["E0_keV"]

    xscale = alt.Scale(domain=[xlo, xhi], nice=False)
    yscale = alt.Scale(domain=[ylo, yhi], nice=False)
    # Per-segment ``mark_rule`` (one straight 2-point segment per row), NOT a
    # grouped ``mark_line``: a continuous ``color`` on a grouped line renders
    # nothing in Vega-Lite (see track_segments_frame). x2/y2 give each segment its
    # end point; energy still colours along the path.
    tracks = (
        alt.Chart(seg_df)
        .mark_rule(strokeWidth=0.9, opacity=0.85)
        .encode(
            x=alt.X("x:Q", scale=xscale, title=f"along beam ({data['ulab']})"),
            y=alt.Y("y:Q", scale=yscale, title=f"transverse ({data['ulab']})"),
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color(
                "E:Q",
                title="electron energy (keV)",
                scale=alt.Scale(scheme="turbo", domain=[E_cut, E0]),  # type: ignore[arg-type]
            ),
        )
    )

    # crystal slab geometry: front face through origin along the slab tangent,
    # back face offset by `thick` along the slab normal (the slab rotates with tilt).
    nslab = np.asarray(data["nslab"], dtype=float)
    ndet = np.asarray(data["ndet"], dtype=float)
    thick = float(data["thick"])
    tang = np.array([-nslab[1], nslab[0]])
    W = 3.0 * max(xhi - xlo, yhi - ylo)

    # Slab shading: fill the parallelogram between the two face lines.
    # At each x the slab occupies y in [(0 - x*nslab[0])/nslab[1],
    # (thick - x*nslab[0])/nslab[1]] when nslab[1] ≠ 0; near-vertical face
    # (nslab[1] ≈ 0) falls back to a mark_rect spanning x ∈ [0, thick].
    _ns = 120
    _xs = np.linspace(xlo, xhi, _ns)
    if abs(nslab[1]) > 1e-4:
        _proj = _xs * nslab[0]
        _yf = np.clip((0.0 - _proj) / nslab[1], ylo, yhi)
        _yb = np.clip((thick - _proj) / nslab[1], ylo, yhi)
        _slab_band_df = pd.DataFrame(
            {"x": _xs, "y": np.minimum(_yf, _yb), "y2": np.maximum(_yf, _yb)}
        )
        slab_shading = (
            alt.Chart(_slab_band_df)
            .mark_area(fill="#bbbbbb", opacity=0.22, stroke=None)
            .encode(
                x=alt.X("x:Q", scale=xscale),
                y=alt.Y("y:Q", scale=yscale),
                y2="y2:Q",
            )
        )
    else:
        _x0, _x1 = min(0.0, thick) / nslab[0], max(0.0, thick) / nslab[0]
        slab_shading = (
            alt.Chart(pd.DataFrame({"x": [_x0], "x2": [_x1], "y": [ylo], "y2": [yhi]}))
            .mark_rect(fill="#bbbbbb", opacity=0.22, stroke=None)
            .encode(
                x=alt.X("x:Q", scale=xscale),
                x2="x2:Q",
                y=alt.Y("y:Q", scale=yscale),
                y2="y2:Q",
            )
        )

    faces = []
    for off in (0.0, thick):
        c = off * nslab
        faces.append(
            alt.Chart(_segment_df(-W * tang + c, W * tang + c))
            .mark_line(color="#666666", strokeWidth=1.2, opacity=0.9)
            .encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
        )

    aL = 0.16 * (xhi - xlo)
    beam = (
        alt.Chart(_segment_df((-aL, 0.0), (0.0, 0.0)))
        .mark_line(color="red", strokeWidth=2.0)
        .encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
    )
    det = (
        alt.Chart(_segment_df((0.0, 0.0), (ndet[0] * aL, ndet[1] * aL)))
        .mark_line(color="#119911", strokeWidth=2.0)
        .encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
    )

    title = (
        f"{case['name'].split()[0]}, {E0:g} keV, "
        f"theta={case.get('tilt_deg', 0.0):g} deg, phi={case.get('tilt_azim_deg', 0.0):g} deg  "
        f"-- {data['Ne']} e- ({data['eta']:.0f}% back, {data['thru']:.0f}% through)"
    )
    return (
        alt.layer(slab_shading, *faces, beam, det, tracks)
        .properties(width=width, height=width, title=title)
        .interactive()
    )
