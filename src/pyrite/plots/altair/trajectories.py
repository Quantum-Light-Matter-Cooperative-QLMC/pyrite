"""Altair trajectories

Altair / Vega-Lite renderers for the electron-penetration figures -- the
interactive counterparts of the matplotlib :mod:`pyrite.plots.mpl.trajectories`
views. Same contract as :mod:`pyrite.plots.altair.spectra`: these reuse the exact
electron-transport + projection prep (``trajectories._trajectory_data`` /
``_trajectory_cases``), so the physics is identical -- only the renderer differs.
Not re-exported from ``pyrite.plots`` (frozen export-set guard); import from the
submodule:

    from pyrite.plots.altair.trajectories import penetration_survival_chart

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
transport then delegate. See :mod:`pyrite.plots.altair.spectra` for the Vega-Lite
5000-row cap note -- keep ``Ne`` modest in ``trajectory_chart`` (vector tracks).
"""

import altair as alt
import numpy as np
import pandas as pd

from .._common import (
    _beam_detector_basis,
    _case_of,
    _groove_spec,
    groove_profile_knots,
)
from .._frames import (
    _square_frame,
    _trajectory_cases,
    _trajectory_data,
    _trajectory_frame,
    survival_frame,
)
from ._typing import _mark_chart

# Mirrors plotly.trajectories._FIELD/_GRID/font.color so trajectory_chart's
# panel reads as one system with the 3D volume it sits beside in the Trace tab.
_FIELD = "#17202A"
_GRID = "#34495E"
_TEXT = "#EDF6F9"
# Amber sawtooth overlay -- distinct from the red beam / green detector arrows
# and matches plotly.trajectories._GROOVE so the 2D and 3D groove views agree.
_GROOVE = "#E9C46A"
# Groove-period cap for the 2D overlay (see plotly.trajectories._GROOVE_MAX_PERIODS);
# larger here because a flat polyline is far cheaper than a 3D mesh.
_GROOVE_MAX_PERIODS = 400


# ---- penetration / survival --------------------------------------------------
# _max_depth_per_electron / survival_frame now live in pyrite.plots._frames,
# shared verbatim with pyrite.plots.mpl.trajectories.plot_penetration_survival;
# re-imported above for the existing ``from pyrite.plots.altair.trajectories
# import survival_frame`` call sites (tests, this module's own use below).
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
    :func:`pyrite.plots.plot_penetration_survival`. ``tilt`` picks the polar tilt
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
    chart = _mark_chart(alt.Chart(df).mark_line(strokeWidth=1.9)).encode(
        x=alt.X("depth:Q", title=x_title),
        y=alt.Y(
            "survival:Q",
            title="surviving electrons (% of N0)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color("energy:N", title="beam energy"),
        tooltip=["energy:N", "depth:Q", "survival:Q"],
    )
    return (
        chart
        # Top-level `background`, not `.configure(background=...)`: marimo's
        # vegafusion fixup (altair_chart.maybe_fix_vegafusion_background)
        # force-overrides to "transparent" whenever the top-level key is
        # unset, clobbering a config-level background. Mirrors
        # trajectory_chart's dark panel so the two read as one system when
        # they sit side by side.
        .properties(width=width, height=height, title=title, background=_FIELD)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=_TEXT, titleColor=_TEXT, gridColor=_GRID, domainColor=_GRID, tickColor=_GRID
        )
        .configure_legend(labelColor=_TEXT, titleColor=_TEXT)
        .configure_title(color=_TEXT)
        .interactive()
    )


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
    if "track_id" in data:
        e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
        start = np.asarray(data["start_xyz"])
        end = np.asarray(data["end_xyz"])
        return (
            pd.DataFrame(
                {
                    "x": start @ e1,
                    "y": start @ e2,
                    "x2": end @ e1,
                    "y2": end @ e2,
                    "E": data["E"],
                    "track_id": data["track_id"],
                    "parent_id": data["parent_id"],
                    "generation": data["generation"],
                }
            )
            .sort_values("E")
            .reset_index(drop=True)
        )
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


def vacuum_segments_frame(data):
    """Separate 2D rules for non-radiating groove-gap flights.

    Vacuum endpoints already use the trajectory display unit. Project them
    through the same beam/detector basis as material tracks without adding them
    to :func:`tracks_frame` or :func:`track_segments_frame`.
    """
    columns = ["x", "y", "x2", "y2", "E", "t_fs", "elec_id"]
    start = np.asarray(data.get("vacuum_start_xyz", np.empty((0, 3))), dtype=float)
    end = np.asarray(data.get("vacuum_end_xyz", np.empty((0, 3))), dtype=float)
    if not len(start):
        return pd.DataFrame(columns=columns)
    e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
    return pd.DataFrame(
        {
            "x": start @ e1,
            "y": start @ e2,
            "x2": end @ e1,
            "y2": end @ e2,
            "E": np.asarray(data["vacuum_E"], dtype=float),
            "t_fs": np.asarray(data["vacuum_t_fs"], dtype=float),
            "elec_id": np.asarray(data["vacuum_elec_id"], dtype=np.int64),
        },
        columns=columns,
    )


def _segment_df(p0, p1):
    return pd.DataFrame({"x": [p0[0], p1[0]], "y": [p0[1], p1[1]]})


def _groove_profile_layer(case, data, frame, xscale, yscale):
    """Blazed-groove sawtooth overlay for :func:`trajectory_chart`, or ``None``
    when the case is ungrooved / the extent needs too many teeth.

    The 2D cross-section is drawn in the BEAM-DETECTOR plane (the same basis
    ``_trajectory_data`` projects tracks into via
    :func:`~pyrite.plots._common._beam_detector_basis`), NOT a raw sample
    x/z slice, so the sawtooth surface is projected through that basis too: a
    sample-frame surface point ``(x, 0, groove_profile_z(x))`` maps to chart
    coordinates ``(P.e1, P.e2) / u``. This keeps the profile riding the entrance
    face at any tilt with no rotation of its own -- the projection IS the frame.
    Apexes touch the origin face line; valleys bulge one groove depth into the
    slab (the ``+nslab`` side the slab shading fills)."""
    spec = _groove_spec(case)
    if spec is None:
        return None
    e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
    u = data["u"]
    xlo, xhi, ylo, yhi = frame
    # Sample-frame lateral span needed to cover the visible frame along the face:
    # the sample x-axis projects into the chart with magnitude |(e1[0], e2[0])|
    # display units per Angstrom, so invert that to reach the far frame corner.
    r_frame = max(abs(xlo), abs(xhi), abs(ylo), abs(yhi))
    ax_mag = float(np.hypot(e1[0], e2[0]))
    x_max_ang = 1.6 * r_frame * u / max(ax_mag, 1e-6)
    knots = groove_profile_knots(-x_max_ang, x_max_ang, spec, max_periods=_GROOVE_MAX_PERIODS)
    if knots is None:
        return None
    xk, zk = knots
    cx = (xk * e1[0] + zk * e1[2]) / u
    cy = (xk * e2[0] + zk * e2[2]) / u
    df = pd.DataFrame({"x": cx, "y": cy, "i": np.arange(len(cx), dtype=float)})
    return _mark_chart(
        alt.Chart(df).mark_line(color=_GROOVE, strokeWidth=1.4, opacity=0.95)
    ).encode(
        x=alt.X("x:Q", scale=xscale),
        y=alt.Y("y:Q", scale=yscale),
        order=alt.Order("i:Q"),  # trace the profile in x-order, not y-sorted
    )


def trajectory_chart(
    rec_or_case,
    *,
    Ne=40,
    seed=0,
    E_cut=5.0,
    frame=None,
    width=480,
    data=None,
):
    """Interactive vector view of ONE electron-penetration cross-section in the
    beam-detector plane: energy-coloured per-electron tracks (turbo) with the
    crystal entrance/exit faces (grey) and the beam (red) + detector (green)
    arrows. Pan/zoom and hover individual tracks. Keep ``Ne`` modest (vector
    marks, not a datashader raster -- see the module docstring). Returns an
    :class:`altair.Chart`, or ``None`` when the cascade is empty."""
    case = _case_of(rec_or_case)
    if data is None:
        data = _trajectory_data(case, Ne, seed)
    seg_df = track_segments_frame(data)
    vacuum_df = vacuum_segments_frame(data)
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
    tracks = _mark_chart(alt.Chart(seg_df).mark_rule(strokeWidth=0.9, opacity=0.85)).encode(
        x=alt.X("x:Q", scale=xscale, title=f"along beam ({data['ulab']})"),
        y=alt.Y("y:Q", scale=yscale, title=f"transverse ({data['ulab']})"),
        x2="x2:Q",
        y2="y2:Q",
        color=alt.Color(
            "E:Q",
            title="electron energy (keV)",
            scale=alt.Scale(scheme="turbo", domain=[E_cut, E0]),  # type: ignore[arg-type]
        ),
        tooltip=(
            [
                alt.Tooltip("track_id:Q", title="track"),
                alt.Tooltip("parent_id:Q", title="parent"),
                alt.Tooltip("generation:Q", title="generation"),
                alt.Tooltip("E:Q", title="energy (keV)", format=".3g"),
            ]
            if "track_id" in seg_df
            else [alt.Tooltip("E:Q", title="energy (keV)", format=".3g")]
        ),
    )
    vacuum = _mark_chart(
        alt.Chart(vacuum_df).mark_rule(color="#8E9AAF", strokeWidth=0.8, opacity=0.35)
    ).encode(
        x=alt.X("x:Q", scale=xscale),
        y=alt.Y("y:Q", scale=yscale),
        x2="x2:Q",
        y2="y2:Q",
        tooltip=[
            alt.Tooltip("elec_id:Q", title="electron"),
            alt.Tooltip("E:Q", title="energy (keV)", format=".3g"),
            alt.Tooltip("t_fs:Q", title="start age (fs)", format=".3g"),
        ],
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
        slab_shading = _mark_chart(
            alt.Chart(_slab_band_df).mark_area(fill="#bbbbbb", opacity=0.22, stroke=None)
        ).encode(
            x=alt.X("x:Q", scale=xscale),
            y=alt.Y("y:Q", scale=yscale),
            y2="y2:Q",
        )
    else:
        _x0, _x1 = min(0.0, thick) / nslab[0], max(0.0, thick) / nslab[0]
        slab_shading = _mark_chart(
            alt.Chart(pd.DataFrame({"x": [_x0], "x2": [_x1], "y": [ylo], "y2": [yhi]})).mark_rect(
                fill="#bbbbbb", opacity=0.22, stroke=None
            )
        ).encode(
            x=alt.X("x:Q", scale=xscale),
            x2="x2:Q",
            y=alt.Y("y:Q", scale=yscale),
            y2="y2:Q",
        )

    faces = []
    for off in (0.0, thick):
        c = off * nslab
        faces.append(
            _mark_chart(
                alt.Chart(_segment_df(-W * tang + c, W * tang + c)).mark_line(
                    color="#666666", strokeWidth=1.2, opacity=0.9
                )
            ).encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
        )
    # internal layer boundaries (film-on-substrate stacks, e.g. mos2 on sapphire):
    # a dashed line at each interior interface, matching the matplotlib panel.
    for zb in data.get("layer_bounds", ()):
        c = float(zb) * nslab
        faces.append(
            _mark_chart(
                alt.Chart(_segment_df(-W * tang + c, W * tang + c)).mark_line(
                    color="#666666", strokeWidth=0.8, strokeDash=[4, 3], opacity=0.8
                )
            ).encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
        )

    aL = 0.16 * (xhi - xlo)
    beam = _mark_chart(
        alt.Chart(_segment_df((-aL, 0.0), (0.0, 0.0))).mark_line(color="red", strokeWidth=2.0)
    ).encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))
    det = _mark_chart(
        alt.Chart(_segment_df((0.0, 0.0), (ndet[0] * aL, ndet[1] * aL))).mark_line(
            color="#119911", strokeWidth=2.0
        )
    ).encode(x=alt.X("x:Q", scale=xscale), y=alt.Y("y:Q", scale=yscale))

    title = (
        f"{case['name'].split()[0]}, {E0:g} keV, "
        f"theta={case.get('tilt_deg', 0.0):g} deg, phi={case.get('tilt_azim_deg', 0.0):g} deg  "
        f"-- {data['Ne']} e- ({data['eta']:.0f}% back, {data['thru']:.0f}% through)"
    )
    groove = _groove_profile_layer(case, data, frame, xscale, yscale)
    overlays = [slab_shading, *faces, beam, det]
    if not vacuum_df.empty:
        overlays.append(vacuum)
    overlays.append(tracks)
    if groove is not None:
        overlays.append(groove)
    return (
        alt.layer(*overlays)
        # Top-level `background`, not `.configure(background=...)`: marimo's
        # vegafusion fixup (altair_chart.maybe_fix_vegafusion_background)
        # force-overrides to "transparent" whenever the top-level key is
        # unset, clobbering a config-level background.
        .properties(width=width, height=width, title=title, background=_FIELD)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=_TEXT, titleColor=_TEXT, gridColor=_GRID, domainColor=_GRID, tickColor=_GRID
        )
        .configure_legend(labelColor=_TEXT, titleColor=_TEXT)
        .configure_title(color=_TEXT)
        .interactive()
    )
