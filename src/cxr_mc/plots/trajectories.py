"""trajectories

Electron-trajectory and penetration/survival figures.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..montecarlo import (
    simulate_trajectories,
    tilted_geometry,
)
from ..montecarlo.groove import surface_depth_ang
from ..results import (
    records,
)
from ._style import (
    energy_color,
)

# ---- electron trajectory + penetration view ----------------------------------
# Datashader rasterizes the (tens of thousands of) trajectory line-segments into
# ONE image per panel -- fast, and tiny on disk vs a matplotlib LineCollection of
# every segment -- while matplotlib keeps the crisp slab / beam / detector overlay
# and the energy colorbar (so the nbconvert PDF export still works). The segment
# colour is the electron's kinetic energy along the track (turbo); ds.max keeps it
# crisp under the line-width antialiasing (ds.mean would blend track edges low).
C_ANG_PER_FS = 2997.924580  # speed of light [Ang/fs]: age sum(L/beta)[Ang] -> fs
_TRAJ_CMAP = "turbo"


def _case_of(rec_or_case):
    """Accept either a results record (carries 'case') or a raw case dict."""
    return rec_or_case.get("case", rec_or_case)


def _groove_spec(case):
    """Blazed :class:`~cxr_mc.montecarlo.groove.GrooveSpec` for a case carrying a
    ``groove_spacing_ang`` knob, or ``None`` when it is absent.

    Mirrors ``montecarlo.runner._transport_case``'s spec construction exactly
    (``blazed_groove_spec(spacing, theta_obs_rad, tilt_polar_rad,
    tilt_azim_rad)``) so the penetration figures transport electrons through the
    SAME relief facets the spectrum runner does -- no new physics, no new
    convention. ``blazed_groove_spec`` validates the restricted geometry
    (theta_obs = 90 deg, tilt_azim = 180 deg, 0 < tilt_polar < 90 deg) and
    raises ``ValueError`` otherwise; callers that build cases via
    ``sweep.build_cases`` never hit that because it rejects the same geometries
    up front."""
    spacing = case.get("groove_spacing_ang")
    if spacing is None:
        return None
    from ..montecarlo import blazed_groove_spec

    return blazed_groove_spec(
        spacing,
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )


def groove_profile_z(x_ang, spec):
    """Sawtooth surface depth [Ang] into the slab at sample-frame lateral position
    ``x_ang`` [Ang].

    Apexes (``z = 0``) sit at ``x = k * spacing``; the valley floor is at the
    groove depth ``spec.depth_ang``. This is the SAME closed-form profile the
    ``_z_surf`` helper in ``tests/montecarlo/test_groove.py`` evaluates (single geometric
    source: :mod:`cxr_mc.montecarlo.groove`), reused here only to DRAW the
    surface -- it introduces no new physics. numpy ufuncs only, so array input
    works elementwise."""
    return surface_depth_ang(x_ang, spec)


def groove_profile_knots(x_lo_ang, x_hi_ang, spec, *, max_periods=200):
    """Minimal sample-frame sawtooth vertices ``(x_ang, z_ang)`` covering
    ``[x_lo_ang, x_hi_ang]``: two knots per period (apex at ``z = 0``, valley
    floor at ``z = depth``), so a corrugated surface needs only ~2 vertices per
    groove instead of a dense sweep.

    Returns ``None`` when the requested span exceeds ``max_periods`` grooves --
    the caller then falls back to the flat entrance face (drawing thousands of
    teeth is neither legible nor cheap). ``x`` is returned strictly increasing so
    the vertices trace the profile directly as a polyline."""
    lam = spec.spacing_ang
    k0 = int(np.floor(x_lo_ang / lam))
    k1 = int(np.ceil(x_hi_ang / lam))
    if k1 - k0 > max_periods:
        return None
    x_valley = spec.depth_ang * np.tan(spec.tilt_polar_rad)
    xs = np.empty(2 * (k1 - k0 + 1))
    ks = np.arange(k0, k1 + 1)
    xs[0::2] = ks * lam  # apexes (z = 0)
    xs[1::2] = ks * lam + x_valley  # valley floors (z = depth)
    zs = groove_profile_z(xs, spec)
    return xs, zs


def _trajectory_cases(cases_or_results):
    """Flatten a build_cases list OR a results store into a list of case dicts."""
    if isinstance(cases_or_results, dict):
        return [r["case"] for r in records(cases_or_results)]
    return [_case_of(c) for c in cases_or_results]


def _turbo_hex(n=256):
    """The turbo colormap as a hex list (the form datashader.shade wants)."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[_TRAJ_CMAP]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def _beam_detector_basis(beam, n_hat):
    """Orthonormal 2D basis of the BEAM-DETECTOR plane: e1 = beam (-> +x, into the
    slab); e2 = the in-plane part of the detector direction (-> +y, "up"). Working
    in this plane (rather than a fixed x-z slice) keeps the beam horizontal AND the
    detector arrow pointing the right way for ANY polar/azimuthal tilt."""
    e1 = np.asarray(beam, float)
    e1 = e1 / np.linalg.norm(e1)
    nh = np.asarray(n_hat, float)
    nh = nh / np.linalg.norm(nh)
    perp = nh - np.dot(nh, e1) * e1
    if np.linalg.norm(perp) < 1e-9:  # detector ~parallel to beam: any in-plane up
        for ref in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            perp = ref - np.dot(ref, e1) * e1
            if np.linalg.norm(perp) > 1e-9:
                break
    return e1, perp / np.linalg.norm(perp)


def _beam_phase_space(case):
    """The case's own beam phase space, for the transport behind a plot.

    A trajectory plot that draws a point source for a beam the run treated as
    having finite emittance, bunch length or energy spread is a picture of a
    different beam than the one that made the spectrum. These keys are absent
    unless a profile deliberately set them, so every case built before the beam
    block existed still transports as the legacy point bunch, bit-for-bit.

    The Gaussian spot (``beam_fwhm_mm``) is deliberately NOT included: it stays
    under the caller's control, because the 2D cross-section wants a point
    source and the 3D view supplies its own display width.
    """
    return dict(
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
        transverse_distribution=case.get("transverse_distribution"),
        energy_spread_frac=case.get("energy_spread_frac"),
    )


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
    along via :func:`_beam_phase_space`, so a plot shows the beam the run used
    rather than a point source standing in for it. A case that sets none of
    those keys transports exactly as before.

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


def _draw_incident_bundle(ax, data, length):
    """Per-electron incident stubs, when the beam has a finite phase space.

    The single red arrow says where the beam comes from; it cannot say that the
    electrons arrive spread over a spot and over a range of angles. Each stub
    runs ``length`` upstream from one electron's true entry point along that
    electron's own initial direction, so the drawn bundle is the sampled
    transverse phase space itself rather than an illustration of it.

    Silently does nothing for a collimated point source -- every stub would lie
    on the arrow already there -- which keeps every pre-BeamSpec panel pixel for
    pixel unchanged.
    """
    from matplotlib.collections import LineCollection

    if data.get("initial_r_ang") is None or data.get("initial_v_hat") is None:
        return  # a hand-built panel payload (tests, notebooks) without transport
    entry = np.asarray(data["initial_r_ang"], dtype=float) / data.get("u", 1.0)
    directions = np.asarray(data["initial_v_hat"], dtype=float)
    e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
    entry_2d = np.column_stack((entry @ e1, entry @ e2))
    dir_2d = np.column_stack((directions @ e1, directions @ e2))
    spread = float(np.ptp(entry_2d, axis=0).max() + np.ptp(dir_2d, axis=0).max())
    if not spread > 0.0:
        return
    segments = np.stack((entry_2d - length * dir_2d, entry_2d), axis=1)
    ax.add_collection(
        LineCollection(segments.tolist(), colors="red", linewidths=0.6, alpha=0.55, zorder=3)
    )


def _draw_trajectory_panel(
    ax,
    data,
    frame,
    E0,
    *,
    E_cut=5.0,
    px=820,
    spread_px=1,
    cmap=None,
    label=True,
    label_fs=8.5,
):
    """Render ONE penetration cross-section into ``ax`` over the shared ``frame``:
    grey slab, datashader-rasterized energy-coloured tracks, red beam + green
    detector arrows.

    The tracks are aggregated with ``line_width=0`` so each pixel takes the true
    electron energy of the track through it -- antialiased (line_width>0) lines
    instead coverage-weight that value, which paints a bogus radial gradient
    ACROSS the line thickness (hot centre -> cool edges) rather than along the
    path. ``tf.spread`` then thickens the crisp 1-px lines back to visibility
    WITHOUT reintroducing that artifact (it copies each pixel's colour outward)."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    xlo, xhi, ylo, yhi = frame
    nslab, ndet, thick = data["nslab"], data["ndet"], data["thick"]

    # slab polygon: front face through the origin, extending `thick` into +nslab
    tang = np.array([-nslab[1], nslab[0]])
    W = 6.0 * max(xhi - xlo, yhi - ylo)
    slab = np.array([-W * tang, W * tang, W * tang + thick * nslab, -W * tang + thick * nslab])
    ax.fill(slab[:, 0], slab[:, 1], facecolor="0.80", edgecolor="0.55", lw=1.0, zorder=1)
    # internal layer boundaries (film-on-substrate stacks, e.g. mos2 on sapphire):
    # a thin line at each interior interface so the substrate is visually distinct
    # from the film, without a second fill colour per layer.
    for zb in data.get("layer_bounds", ()):
        c = zb * nslab
        ax.plot(
            [-W * tang[0] + c[0], W * tang[0] + c[0]],
            [-W * tang[1] + c[1], W * tang[1] + c[1]],
            color="0.45",
            lw=0.8,
            ls="--",
            zorder=1,
        )

    # continuous NaN-separated per-electron tracks -> datashader raster, colour =
    # electron energy (ds.max keeps it crisp under the line-width antialiasing)
    df = pd.DataFrame({"x": data["px"], "y": data["py"], "E": data["pE"]})
    asp = (yhi - ylo) / (xhi - xlo)
    cvs = ds.Canvas(
        plot_width=px,
        plot_height=max(int(px * asp), 60),
        x_range=(xlo, xhi),
        y_range=(ylo, yhi),
    )
    agg = cvs.line(df, "x", "y", agg=ds.max("E"), line_width=0)  # crisp: true E/pixel
    img = tf.shade(agg, cmap=cmap or _turbo_hex(), span=(E_cut, E0), how="linear")
    if spread_px:
        img = tf.spread(img, px=spread_px, shape="circle")  # thicken, colour kept
    ax.imshow(
        np.asarray(img.to_pil()),
        extent=(xlo, xhi, ylo, yhi),
        origin="upper",
        aspect="equal",
        interpolation="none",
        zorder=2,
    )

    # Non-radiating groove-gap flights: separate faint rules, never folded into
    # the datashader track table/aggregation above.
    vacuum_start = np.asarray(data.get("vacuum_start_xyz", np.empty((0, 3))), dtype=float)
    vacuum_end = np.asarray(data.get("vacuum_end_xyz", np.empty((0, 3))), dtype=float)
    if len(vacuum_start):
        from matplotlib.collections import LineCollection

        e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
        vacuum_segments = np.stack(
            (
                np.column_stack((vacuum_start @ e1, vacuum_start @ e2)),
                np.column_stack((vacuum_end @ e1, vacuum_end @ e2)),
            ),
            axis=1,
        )
        vacuum_energy = np.asarray(data["vacuum_E"], dtype=float)
        collection = LineCollection(
            vacuum_segments.tolist(),
            cmap=cmap or _TRAJ_CMAP,
            linewidths=0.8,
            alpha=0.35,
            zorder=3,
        )
        collection.set_clim(E_cut, E0)
        collection.set_array(vacuum_energy)
        ax.add_collection(collection)

    # beam (red) + detector (green) arrows, anchored at the entry point
    aL = 0.16 * (xhi - xlo)
    _draw_incident_bundle(ax, data, aL)
    ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(-aL, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(ndet[0] * aL, ndet[1] * aL),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#119911", lw=2.0),
        zorder=4,
    )
    if label:
        ax.text(
            -aL * 0.5,
            0.03 * (yhi - ylo),
            "beam",
            color="red",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
        tx = float(np.clip(ndet[0] * aL * 1.1, xlo + 0.06 * (xhi - xlo), xhi - 0.06 * (xhi - xlo)))
        ty = float(np.clip(ndet[1] * aL * 1.1, ylo + 0.06 * (yhi - ylo), yhi - 0.1 * (yhi - ylo)))
        ax.text(
            tx,
            ty,
            "detector",
            color="#0a6a0a",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            zorder=5,
        )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")


def _traj_colorbar(ax, E_cut, E0, label="electron energy (keV)"):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    sm = cm.ScalarMappable(norm=Normalize(E_cut, E0), cmap=_TRAJ_CMAP)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    return cb


def plot_electron_trajectories(
    rec_or_case,
    *,
    Ne=200,
    seed=0,
    frame=None,
    E_cut=5.0,
    colorbar=True,
    spread_px=1,
    label=True,
    ax=None,
):
    """One electron-penetration cross-section in the beam-detector plane: the beam
    enters horizontally at the origin (red), the crystal is the grey slab (which
    rotates with the tilt), the detector direction is the green arrow, and the
    cascade is datashader-rasterized, coloured by electron energy.

    ``frame`` (xlo, xhi, ylo, yhi) fixes the axes so repeated calls at the same
    (material, thickness, energy) share ONE frame and only the slab rotates; None
    auto-fits this case. ``ax`` draws into an existing axis (used by the grid)."""
    case = _case_of(rec_or_case)
    data = _trajectory_data(case, Ne, seed)
    if frame is None:
        frame = _trajectory_frame([data["pts"]])
    if ax is None:
        xlo, xhi, ylo, yhi = frame
        asp = (yhi - ylo) / (xhi - xlo)
        # size the FIGURE to the data aspect so the equal-aspect axes fills it
        # (no floating-title letterbox): reserve ~1.7" width for ylabel+colorbar
        # and ~1.1" height for title+xlabel, then constrained_layout packs it.
        axw = 5.4
        _, ax = plt.subplots(
            figsize=(axw + 1.7, float(np.clip(axw * asp + 1.1, 3.2, 8.4))),
            constrained_layout=True,
        )
    _draw_trajectory_panel(
        ax, data, frame, case["E0_keV"], E_cut=E_cut, spread_px=spread_px, label=label
    )
    ax.set_xlabel(f"distance along beam ({data['ulab']})")
    ax.set_ylabel(f"transverse distance ({data['ulab']})")
    ax.set_title(
        rf"{case['name'].split()[0]}, {case['E0_keV']:g} keV, "
        rf"$\theta$={case.get('tilt_deg', 0.0):g}$\degree$, "
        rf"$\phi$={case.get('tilt_azim_deg', 0.0):g}$\degree$  —  {data['Ne']} e$^-$ "
        rf"({data['eta']:.0f}% back, {data['thru']:.0f}% through)",
        fontsize=10,
    )
    if colorbar:
        _traj_colorbar(ax, E_cut, case["E0_keV"])
    return ax


def plot_trajectory_grid(
    cases_or_results,
    energy=None,
    *,
    Ne=150,
    seed=0,
    E_cut=5.0,
    spread_px=1,
    max_panels=12,
    ncols=None,
    max_width_in=13.0,
    max_height_in=11.0,
    panel_in=3.3,
    min_panel_in=2.5,
):
    """Electron-penetration cross-sections at ONE beam energy, a panel per
    (polar, azimuthal) tilt -- the trajectory analogue of plot_heatmaps. Every
    panel shares ONE (squared) frame, so across the grid ONLY the slab rotates;
    the cascade is datashader-rasterized and energy-coloured with a single shared
    colorbar.

    ``cases_or_results`` is a build_cases list or a results store; ``energy`` picks
    the beam energy (default the lowest). If both polar and azimuthal tilt are
    swept it lays out a polar x azimuth grid (like the heatmaps); otherwise it
    wraps the swept tilt into at most 3 columns (override with ``ncols``). ``Ne``
    (electrons/panel) trades detail for speed -- the electron transport, not the
    drawing, is the cost.

    Panel SIZE: each panel is SQUARE (the shared frame is squared so set_aspect
    "equal" stays exact) and targets ``panel_in`` inches (~3.3", matching
    plot_best_spectra), never shrinking below ``min_panel_in``. For more tilts than
    fit in 3 columns the figure grows TALLER (scrollable) rather than crushing
    panels -- ``max_height_in`` no longer shrinks them. ``max_panels`` caps how
    many (polar, azimuth) combos are drawn (evenly subsampled past the cap); raise
    it (and/or ``panel_in``) for a denser grid, lower it for bigger panels."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    energies = sorted({c["E0_keV"] for c in cases})
    energy = energies[0] if energy is None else min(energies, key=lambda e: abs(e - energy))
    grp = [c for c in cases if c["E0_keV"] == energy]

    polars = sorted({c["tilt_deg"] for c in grp})
    azims = sorted({c["tilt_azim_deg"] for c in grp})
    grid2d = len(polars) > 1 and len(azims) > 1
    # one representative case per (polar, azimuth) combo, in a stable order
    bycombo: dict[tuple[float, float], Any] = {}
    for c in sorted(grp, key=lambda c: (c["tilt_deg"], c["tilt_azim_deg"])):
        bycombo.setdefault((c["tilt_deg"], c["tilt_azim_deg"]), c)
    combos = list(bycombo)
    if len(combos) > max_panels:  # subsample evenly so the grid stays on-screen
        keep = np.unique(np.linspace(0, len(combos) - 1, max_panels).round().astype(int))
        combos = [combos[i] for i in keep]
        grid2d = False
    data = {cb: _trajectory_data(bycombo[cb], Ne, seed) for cb in combos}
    # square the shared frame so the panels read as squares (the native frame is
    # wide -> skinny strips); no data is cropped, set_aspect("equal") stays exact.
    frame = _square_frame(_trajectory_frame([d["pts"] for d in data.values()]))

    if grid2d:
        nrows, ncols = len(polars), len(azims)
        cell = [[(p, a) for a in azims] for p in polars]
    else:
        n = len(combos)
        ncols = ncols or min(3, n)  # cap at 3 columns; wrap the rest into ROWS
        nrows = int(np.ceil(n / ncols))
        cell = [
            [combos[r * ncols + col] if r * ncols + col < n else None for col in range(ncols)]
            for r in range(nrows)
        ]

    # SQUARE panels (the "trajectory plots are tiny / skinny" fix): the frame is
    # square so asp == 1 and pw == ph. Target ``panel_in`` (~3.3", matching
    # plot_best_spectra), clamp to the per-column width budget, never shrink below
    # min_panel_in -- for many tilts the FIGURE grows TALLER (scrollable) rather
    # than crushing every panel. ``max_height_in`` is no longer used to shrink.
    xlo, xhi, ylo, yhi = frame
    asp = (yhi - ylo) / (xhi - xlo)  # == 1 after _square_frame
    pw = min(panel_in, (max_width_in - 1.1) / ncols)
    pw = max(pw, min_panel_in)
    ph = pw * asp
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * pw + 1.1, nrows * ph + 0.7),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for r in range(nrows):
        for col in range(ncols):
            ax = axes[r][col]
            combo = cell[r][col]
            if combo is None or combo not in data:
                ax.axis("off")
                continue
            d = data[combo]
            _draw_trajectory_panel(
                ax, d, frame, energy, E_cut=E_cut, spread_px=spread_px, label=False
            )
            ax.set_title(
                rf"$\theta$={combo[0]:g}$\degree$, $\phi$={combo[1]:g}$\degree$",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            if r == nrows - 1:
                ax.set_xlabel(f"along beam ({d['ulab']})", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"transverse ({d['ulab']})", fontsize=8)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    mappable = cm.ScalarMappable(norm=Normalize(E_cut, energy), cmap=_TRAJ_CMAP)
    cb = fig.colorbar(mappable, ax=axes, shrink=0.8, aspect=30, pad=0.01)
    cb.set_label("electron energy (keV)")
    case0 = bycombo[combos[0]]
    fig.suptitle(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"{energy:g} keV — electron penetration (red beam, green detector; "
        rf"only the slab rotates)",
        fontsize=11,
    )
    return fig


def plot_penetration_survival(
    cases_or_results,
    *,
    Ne=500,
    seed=0,
    n_bins=80,
    tilt=None,
    depth_frac=True,
):
    """Surviving electron population vs penetration depth -- the fraction of the
    incident electrons (% of N0) that reach AT LEAST a depth z below the entrance
    surface, one monotonically-decreasing curve per beam energy. This is where the
    beam stops: the curve falls from 100% at the surface to 0 at the deepest
    penetration, and a higher-energy beam reaches deeper.

    Per electron the deepest segment it reaches sets its penetration depth (the
    max over its segment midpoints' depth below the slab normal); then
    ``survival(z) = (# electrons reaching depth >= z) / N0``.
    ``tilt`` selects the polar tilt (nearest; default the one closest to normal
    incidence). ``depth_frac`` plots depth as a fraction of the slab thickness (so
    thin and thick slabs overlay); set False for absolute depth."""
    cases = _trajectory_cases(cases_or_results)
    if not cases:
        print("no cases/results to plot")
        return None
    tilts = sorted({c["tilt_deg"] for c in cases})
    want = 0.0 if tilt is None else tilt
    t = min(tilts, key=lambda x: abs(x - want))
    energies = sorted({c["E0_keV"] for c in cases})

    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = 1.0
    for E0 in energies:
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is None:
            continue
        d = _trajectory_data(c, Ne, seed)
        # deepest point each electron reaches (max over its segment depths), then
        # clip the tiny negative excursions of backscattered electrons that exit
        # just above the entrance face.
        max_depth = np.full(d["Ne"], -np.inf)  # type: ignore[reportCallIssue]
        np.maximum.at(max_depth, d["elec_id"], d["z_u"])  # type: ignore[reportArgumentType]
        max_depth = np.clip(max_depth[np.isfinite(max_depth)], 0.0, None)
        thick = d["thick"]
        x = max_depth / thick if depth_frac else max_depth
        xmax = 1.0 if depth_frac else max(xmax, float(thick))
        zs = np.linspace(0.0, 1.0 if depth_frac else float(thick), n_bins)
        surv = 100.0 * np.array([float((x >= z).mean()) for z in zs])
        ax.plot(zs, surv, "-", color=energy_color(E0, energies), lw=1.9, label=f"{E0:g} keV")
    case0 = next(c for c in cases if c["tilt_deg"] == t)
    ulab = r"$\mu$m" if case0["thickness_ang"] >= 1e4 else "nm"
    xlab = "depth / thickness" if depth_frac else f"penetration depth ({ulab})"
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"surviving electrons (% of $N_0$)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(title="beam energy", fontsize=9)
    ax.set_title(
        rf"{case0['name'].split()[0]}, {case0['thickness_ang'] / 1e4:.1f} $\mu$m, "
        rf"$\theta_\mathrm{{tilt}}$={t:g}$\degree$ — electron penetration / survival",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
