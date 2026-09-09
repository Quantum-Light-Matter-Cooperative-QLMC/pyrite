"""trajectories

Electron-trajectory and penetration/survival figures.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .._common import (
    _beam_detector_basis,
    _beam_phase_space,  # noqa: F401 -- re-exported via the flat compat shim
    _case_of,
    _groove_spec,  # noqa: F401 -- re-exported via the flat compat shim
    groove_profile_knots,  # noqa: F401 -- re-exported via the flat compat shim
    groove_profile_z,  # noqa: F401 -- re-exported via the flat compat shim
)
from .._frames import (
    _square_frame,
    _trajectory_cases,
    _trajectory_data,
    _trajectory_frame,
    survival_frame,
)
from .._style import (
    energy_color,
)

# ---- electron trajectory + penetration view ----------------------------------
# Datashader rasterizes the (tens of thousands of) trajectory line-segments into
# ONE image per panel -- fast, and tiny on disk vs a matplotlib LineCollection of
# every segment -- while matplotlib keeps the crisp slab / beam / detector overlay
# and the energy colorbar (so the nbconvert PDF export still works). The segment
# colour is the electron's kinetic energy along the track (turbo); ds.max keeps it
# crisp under the line-width antialiasing (ds.mean would blend track edges low).
_TRAJ_CMAP = "turbo"


def _turbo_hex(n=256):
    """The turbo colormap as a hex list (the form datashader.shade wants)."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[_TRAJ_CMAP]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


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

    # Same data_by_energy -> tidy-table split as
    # :func:`pyrite.plots.altair.trajectories.penetration_survival_chart`: build
    # one _trajectory_data per beam energy at this tilt, then reduce every energy's
    # cascade to a survival curve in ONE shared frame builder
    # (:func:`pyrite.plots._frames.survival_frame`) so the matplotlib and Altair
    # curves are computed identically.
    data_by_energy = {}
    for E0 in energies:
        c = next((c for c in cases if c["E0_keV"] == E0 and c["tilt_deg"] == t), None)
        if c is not None:
            data_by_energy[E0] = _trajectory_data(c, Ne, seed)
    xmax = (
        1.0
        if depth_frac
        else max((float(d["thick"]) for d in data_by_energy.values()), default=1.0)
    )
    df = survival_frame(data_by_energy, n_bins=n_bins, depth_frac=depth_frac)

    fig, ax = plt.subplots(figsize=(8, 5))
    for E0 in energies:
        if E0 not in data_by_energy:
            continue
        label = f"{E0:g} keV"
        sub = df[df["energy"] == label]
        ax.plot(
            sub["depth"],
            sub["survival"],
            "-",
            color=energy_color(E0, energies),
            lw=1.9,
            label=label,
        )
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
