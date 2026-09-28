# /// script
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np

    from pyrite.api import build_configured_cases
    from pyrite.apps._design import (
        apply_altair_theme,
        apply_plotly_theme,
        page_title,
        resolved_theme,
        style_sheet,
        theme_switch,
    )
    from pyrite.apps._widgets import MaterialSelect

    # penetration_survival_chart / trajectory_chart can exceed Vega-Lite's
    # default 5000-row cap; vegafusion (shipped with marimo[recommended])
    # lifts it, falling back to disabling the cap outright.
    try:
        alt.data_transformers.enable("vegafusion")
    except Exception:
        alt.data_transformers.disable_max_rows()

    from pyrite._formatting import fmt_thickness
    from pyrite.campaign.beam_metrics import initial_state_metrics
    from pyrite.campaign.config import default_settings, trajectory_sweep
    from pyrite.materials import CATALOG
    from pyrite.plots.altair.trajectories import (
        penetration_survival_chart,
        trajectory_chart,
    )
    from pyrite.plots.plotly.camera import (
        CAMERA_PRESETS,
        DEFAULT_ANGLES,
        data_aspect_ratio,
        resolve_scene_camera,
    )
    from pyrite.plots.plotly.crystal_lattice import crystal_lattice_figure
    from pyrite.plots.plotly.render import (
        cached_render_path,
        prune_render_cache,
        render_cache_key,
        render_reveal_animation,
    )
    from pyrite.plots.plotly.trajectories import (
        VOLUME_CAMERA_ANGLES,
        trajectory_volume_data,
        trajectory_volume_figure_from_data,
        visible_trajectory_data,
    )

    return (
        CAMERA_PRESETS,
        CATALOG,
        DEFAULT_ANGLES,
        MaterialSelect,
        VOLUME_CAMERA_ANGLES,
        apply_altair_theme,
        apply_plotly_theme,
        build_configured_cases,
        cached_render_path,
        crystal_lattice_figure,
        data_aspect_ratio,
        default_settings,
        fmt_thickness,
        initial_state_metrics,
        mo,
        np,
        page_title,
        penetration_survival_chart,
        prune_render_cache,
        resolve_scene_camera,
        resolved_theme,
        render_cache_key,
        render_reveal_animation,
        style_sheet,
        theme_switch,
        trajectory_chart,
        trajectory_sweep,
        trajectory_volume_data,
        trajectory_volume_figure_from_data,
        visible_trajectory_data,
    )


@app.cell
def _(mo, page_title, style_sheet, theme_switch):
    theme_ui = theme_switch(mo)
    mo.vstack(
        [
            style_sheet(mo),
            mo.hstack(
                [
                    page_title(
                        mo,
                        "3D trajectory and structure viewer",
                        "Watch electron cascades develop inside the crystal and inspect the lattice itself.",
                        eyebrow="PyRITE",
                    ),
                    theme_ui,
                ],
                justify="space-between",
                align="start",
                wrap=True,
            ),
        ]
    )
    return (theme_ui,)


@app.cell
def _(CAMERA_PRESETS, mo):
    # Both 3D tabs expose the SAME camera control set. An explicit camera is
    # what makes a snapshot match what you see: marimo's plotly component drops
    # `scene.camera` relayout events, so a camera you drag to never reaches
    # Python -- neither the modebar PNG (which re-renders from the spec marimo
    # sent) nor the offscreen video render can see it. These controls put the
    # view back in the figure spec, where both paths read it.
    _VIEW_OPTIONS = ["default", *CAMERA_PRESETS, "custom"]

    def camera_controls(default_angles):
        """`(view, orbit axis, azimuth, elevation, zoom)` controls seeded from a figure's view."""
        _azimuth, _elevation, _distance = default_angles
        return (
            mo.ui.dropdown(_VIEW_OPTIONS, value="default", label="view"),
            # Azimuth orbits ABOUT this axis, so z alone can never bring the
            # camera over the z pole; x or y re-poles the sphere to sweep there.
            mo.ui.dropdown(["z", "x", "y"], value="z", label="orbit axis"),
            mo.ui.slider(-180, 180, value=round(_azimuth), step=1, label="azimuth"),
            mo.ui.slider(-89, 89, value=round(_elevation), step=1, label="elevation"),
            mo.ui.slider(0.5, 3.0, value=1.0, step=0.05, label="zoom"),
        )

    def dim_unless(element, active):
        """Grey out a control that currently has no effect, keeping its value visible."""
        if active:
            return element
        return element.style({"opacity": "0.4", "pointer-events": "none"})

    return camera_controls, dim_unless


@app.cell
def _(CATALOG, MaterialSelect, mo):
    # Unlike analysis_app, every view here runs transport (or draws the lattice)
    # straight from the catalog's configured scan grids, so no checkpoint is
    # needed and no material is ever disabled. Same persisted/CLI default
    # resolution as analysis_app for a consistent landing selection.
    from pyrite.apps.analyze import get_default_material, initial_material

    _options = sorted(
        (
            {"value": key, "label": CATALOG.material(key).label, "disabled": False}
            for key in CATALOG.material_keys
        ),
        key=lambda row: str(row["label"]).casefold(),
    )
    _requested = initial_material(mo.cli_args(), get_default_material())
    _values = [row["value"] for row in _options]
    _initial = _requested if _requested in _values else next(iter(_values), None)
    material_ui = mo.ui.anywidget(
        MaterialSelect(
            options=_options,
            value=_initial,
            label="Material",
            disabled=_initial is None,
        )
    )
    material_ui
    return (material_ui,)


@app.cell
def _(default_settings, material_ui):
    MATERIAL = material_ui.value["value"]
    settings = default_settings()
    return MATERIAL, settings


@app.cell
def _(CATALOG, MATERIAL, fmt_thickness, mo):
    # The penetration figures run transport directly, so use the selected
    # material's configured scan grids instead of requiring a checkpoint.
    # MATERIAL is only None with an empty catalog; keep this control cell valid
    # with inert values so the app still loads.
    if MATERIAL is None:
        _energy_values = (1.0,)
        _thickness_values = (10.0,)
        _tilt_values = (0.0,)
    else:
        _scan = CATALOG.material(MATERIAL).scan
        _energy_values = tuple(float(value) for value in _scan.energy_keV)
        _thickness_values = tuple(float(value) for value in _scan.thickness_ang)
        _tilt_values = tuple(float(value) for value in _scan.tilt_deg)

    _source_options = {"Presets": "grid", "Custom": "manual"}
    penetration_energy_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    penetration_energy_grid_ui = mo.ui.dropdown(
        {f"{value:g} keV": value for value in _energy_values},
        value=f"{_energy_values[0]:g} keV",
        label="",
    )
    penetration_energy_manual_ui = mo.ui.number(
        start=1.0, step=1.0, value=_energy_values[0], label="(keV)"
    )

    penetration_thickness_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    _default_thickness = 40000.0
    if _default_thickness not in _thickness_values:
        _default_thickness = _thickness_values[0]
    penetration_thickness_grid_ui = mo.ui.dropdown(
        {fmt_thickness(value): value for value in _thickness_values},
        value=fmt_thickness(_default_thickness),
        label="",
    )
    penetration_thickness_manual_ui = mo.ui.number(
        start=0.001,
        stop=10000.0,
        step=0.001,
        value=min(max(_default_thickness / 1e4, 4.0), 10000.0),
        label="(µm)",
    )

    penetration_tilt_source_ui = mo.ui.dropdown(_source_options, value="Presets", label="")
    if len(_tilt_values) > 1:
        default_tilt_ind = len(_tilt_values) // 2
        default_tilt_value = _tilt_values[default_tilt_ind]
    else:
        default_tilt_value = _tilt_values[0]

    penetration_tilt_grid_ui = mo.ui.dropdown(
        {f"{value:g} deg": value for value in _tilt_values},
        value=f"{default_tilt_value:g} deg",
        label="",
    )
    penetration_tilt_manual_ui = mo.ui.number(
        start=0.0, stop=89.9, step=0.1, value=_tilt_values[0], label="(deg)"
    )

    # Azimuth is not part of the material's scan grid (trajectory_sweep pins it
    # to normal incidence by default) -- offer a fixed preset list instead of a
    # Presets/Custom toggle like the other penetration knobs.
    _azim_values = (100.0, 125.0, 140.0, 165.0, 180.0)
    penetration_azim_ui = mo.ui.dropdown(
        {f"{value:g} deg": value for value in _azim_values},
        value=f"{180.0:g} deg",
        label="",
    )
    return (
        penetration_azim_ui,
        penetration_energy_grid_ui,
        penetration_energy_manual_ui,
        penetration_energy_source_ui,
        penetration_thickness_grid_ui,
        penetration_thickness_manual_ui,
        penetration_thickness_source_ui,
        penetration_tilt_grid_ui,
        penetration_tilt_manual_ui,
        penetration_tilt_source_ui,
    )


@app.cell
def _(
    penetration_azim_ui,
    penetration_energy_grid_ui,
    penetration_energy_manual_ui,
    penetration_energy_source_ui,
    penetration_thickness_grid_ui,
    penetration_thickness_manual_ui,
    penetration_thickness_source_ui,
    penetration_tilt_grid_ui,
    penetration_tilt_manual_ui,
    penetration_tilt_source_ui,
):
    penetration_energy_keV = (
        penetration_energy_grid_ui.value
        if penetration_energy_source_ui.value == "grid"
        else penetration_energy_manual_ui.value
    )
    penetration_thickness_ang = (
        penetration_thickness_grid_ui.value
        if penetration_thickness_source_ui.value == "grid"
        else penetration_thickness_manual_ui.value * 1e4
    )
    penetration_tilt_deg = (
        penetration_tilt_grid_ui.value
        if penetration_tilt_source_ui.value == "grid"
        else penetration_tilt_manual_ui.value
    )
    penetration_azim_deg = penetration_azim_ui.value
    return (
        penetration_azim_deg,
        penetration_energy_keV,
        penetration_thickness_ang,
        penetration_tilt_deg,
    )


@app.cell
def _(mo):
    # Off by default: the fitted-window 3D view exaggerates lateral scale so the
    # cascade is legible. Checking this redraws the box at the true 5x5 mm crystal
    # footprint and overlays the beam entry footprint, which outgrows the crystal
    # at grazing tilt (1/cos stretch). The ~micron cascade then collapses toward
    # the origin -- expected at true scale.
    penetration_realistic_ui = mo.ui.checkbox(value=False, label="")
    return (penetration_realistic_ui,)


@app.cell
def _(mo):
    # Transported-dataset cache. The penetration figure is a static full-reveal
    # view (orbit/hover via Plotly, no client-side playback); this cache exists
    # purely to avoid re-running Monte Carlo transport when an UNRELATED
    # control (e.g. a different tab) reruns this cell but the transport
    # parameters (case, Ne, seed, realistic, beam_fwhm) haven't changed.
    get_penetration_data, set_penetration_data = mo.state(None)
    return get_penetration_data, set_penetration_data


@app.cell
def _(mo):
    # Render-status handoff between the eager render cell (below the tabs)
    # and the lazy tab body: (render_key, error_str_or_None). Setting it
    # after a render completes reruns the tab so the freshly cached video
    # appears without another click.
    get_penetration_render_status, set_penetration_render_status = mo.state(None)
    return get_penetration_render_status, set_penetration_render_status


@app.cell
def _(mo):
    # Survival-chart cache, same pattern as get_penetration_data above.
    # penetration_survival_chart(...) runs its OWN fresh Ne=500-electron
    # transport, so without caching it would redo that work every time the
    # tab body reruns for ANY reason -- e.g. toggling Realistic mode, which
    # the survival curve does not even depend on. Keyed on the smaller set of
    # knobs the chart itself actually depends on (material, beam energy,
    # polar tilt, thickness) in the tab body below. Altair charts are plain
    # objects, so caching/reusing one is safe.
    get_penetration_survival, set_penetration_survival = mo.state(None)
    return get_penetration_survival, set_penetration_survival


@app.cell
def _(mo):
    # Regenerate's click COUNT feeds the trajectory `seed`: every click draws a
    # fresh transport, and everything else re-renders with the SAME seed until
    # Regenerate is clicked again.
    penetration_regen_ui = mo.ui.button(
        value=0, on_click=lambda value: value + 1, label="🎲 Regenerate"
    )
    penetration_ne_ui = mo.ui.slider(
        start=25, stop=300, value=50, step=5, label="Electrons (Ne)", show_value=True
    )
    penetration_beam_fwhm_ui = mo.ui.number(
        value=1.0, start=0.001, step=0.1, label="Beam FWHM (mm)"
    )
    # Blazed sawtooth entrance-face grooves (Task 6 / docs/superpowers/plans/
    # 2026-07-23-blazed-groove-geometry.md). 0 = off (flat face). Grooves are only
    # valid at azimuth = 180 deg with 0 < polar tilt < 90 deg; the tab applies
    # them only then and shows a note otherwise.
    penetration_groove_ui = mo.ui.number(
        value=0.0, start=0.0, step=1000.0, label="Groove spacing (Å, 0 = off)"
    )
    penetration_secondaries_ui = mo.ui.switch(value=False, label="generate secondaries")
    penetration_secondary_threshold_ui = mo.ui.number(
        value=1000.0, start=50.0, step=100.0, label="secondary threshold (eV)"
    )
    penetration_max_generation_ui = mo.ui.number(
        value=64, start=0, step=1, label="show through generation"
    )
    penetration_min_energy_ui = mo.ui.number(
        value=0.0, start=0.0, step=0.5, label="minimum shown energy (keV)"
    )
    return (
        penetration_beam_fwhm_ui,
        penetration_groove_ui,
        penetration_secondaries_ui,
        penetration_secondary_threshold_ui,
        penetration_max_generation_ui,
        penetration_min_energy_ui,
        penetration_ne_ui,
        penetration_regen_ui,
    )


@app.cell
def _(mo):
    # The interactive Plotly frame animation is gone: the tab shows a static
    # full-reveal figure plus an
    # explicit Render button that produces a smooth, looping fixed-camera
    # video via pyrite.plots.plotly.render.render_reveal_animation.
    # Rendering is a blocking multi-second-to-minutes kaleido/ffmpeg job, so
    # it stays opt-in behind a button rather than running on every rerun.
    penetration_render_frames_ui = mo.ui.slider(
        start=24, stop=120, value=60, step=12, label="Render frames", show_value=True
    )
    penetration_render_button_ui = mo.ui.run_button(label="Render animation")
    return penetration_render_button_ui, penetration_render_frames_ui


@app.cell
def _(VOLUME_CAMERA_ANGLES, camera_controls):
    # Seeded from the volume figure's own hand-tuned eye, so "default" is the
    # long-standing view, exactly. Named `_camera_` throughout to keep these
    # apart from the crystal TILT azimuth the transport sweep reads.
    (
        penetration_camera_view_ui,
        penetration_camera_orbit_ui,
        penetration_camera_azim_ui,
        penetration_camera_elev_ui,
        penetration_camera_zoom_ui,
    ) = camera_controls(VOLUME_CAMERA_ANGLES)
    return (
        penetration_camera_azim_ui,
        penetration_camera_elev_ui,
        penetration_camera_orbit_ui,
        penetration_camera_view_ui,
        penetration_camera_zoom_ui,
    )


@app.cell
def _(
    VOLUME_CAMERA_ANGLES,
    penetration_camera_azim_ui,
    penetration_camera_elev_ui,
    penetration_camera_orbit_ui,
    penetration_camera_view_ui,
    penetration_camera_zoom_ui,
    resolve_scene_camera,
):
    # ONE camera dict for both consumers: the lazy tab body (live figure +
    # cache-key lookup) and the eager render cell (the render itself). If they
    # disagreed, the tab would look for a cache entry the render never wrote.
    penetration_camera = resolve_scene_camera(
        penetration_camera_view_ui.value,
        penetration_camera_azim_ui.value,
        penetration_camera_elev_ui.value,
        penetration_camera_zoom_ui.value,
        default_angles=VOLUME_CAMERA_ANGLES,
        orbit_axis=penetration_camera_orbit_ui.value,
    )
    # This scene stays perspective, so the eye distance the zoom rides on is
    # read as written -- no aspect-ratio workaround needed here.
    return (penetration_camera,)


@app.cell
def _(
    MATERIAL,
    apply_altair_theme,
    apply_plotly_theme,
    build_configured_cases,
    cached_render_path,
    dim_unless,
    get_penetration_data,
    get_penetration_render_status,
    get_penetration_survival,
    initial_state_metrics,
    mo,
    np,
    penetration_azim_deg,
    penetration_azim_ui,
    penetration_beam_fwhm_ui,
    penetration_camera,
    penetration_camera_azim_ui,
    penetration_camera_elev_ui,
    penetration_camera_orbit_ui,
    penetration_camera_view_ui,
    penetration_camera_zoom_ui,
    penetration_energy_grid_ui,
    penetration_energy_keV,
    penetration_energy_manual_ui,
    penetration_energy_source_ui,
    penetration_groove_ui,
    penetration_ne_ui,
    penetration_secondaries_ui,
    penetration_secondary_threshold_ui,
    penetration_max_generation_ui,
    penetration_min_energy_ui,
    penetration_realistic_ui,
    penetration_regen_ui,
    penetration_render_button_ui,
    penetration_render_frames_ui,
    penetration_survival_chart,
    penetration_thickness_ang,
    penetration_thickness_grid_ui,
    penetration_thickness_manual_ui,
    penetration_thickness_source_ui,
    penetration_tilt_deg,
    penetration_tilt_grid_ui,
    penetration_tilt_manual_ui,
    penetration_tilt_source_ui,
    render_cache_key,
    set_penetration_data,
    set_penetration_survival,
    settings,
    trajectory_chart,
    trajectory_sweep,
    trajectory_volume_data,
    trajectory_volume_figure_from_data,
    visible_trajectory_data,
    resolved_theme,
    theme_ui,
):
    _theme = resolved_theme(theme_ui)

    def penetration_tab():
        _angle = penetration_tilt_deg
        _md = mo.md(
            "An interactive 3D electron track cutaway and a surviving-electron fraction vs depth, at the polar tilt and "
            "azimuth selected in this tab. The translucent crystal's lateral extent is fitted to the tracks; depth "
            "and layer interfaces retain true scale. Enable secondaries to generate shell collision tracks; "
            "track hover shows each parent and generation. The generation and energy controls filter both views. "
        )
        # Blazed grooves: valid only at azimuth 180 deg with 0 < polar tilt < 90.
        # Only feed the knob to trajectory_sweep when the selected azimuth admits
        # it; otherwise show a note and draw the ungrooved slab. build_cases /
        # Sweep still validate (tilt, substrate/stack, footprint) and raise -- a
        # substrate material or tilt=0 falls back to ungrooved with the reason.
        _groove_req = float(penetration_groove_ui.value or 0.0)
        if penetration_secondaries_ui.value and _groove_req > 0:
            _groove_req = 0.0  # shell transport does not support grooved geometry
            _groove_note = mo.md("*Grooves are disabled while generating secondaries.*")
        else:
            _groove_note = None
        _groove_spacing = (
            _groove_req if (_groove_req > 0.0 and penetration_azim_deg == 180.0) else None
        )
        if _groove_req > 0.0 and _groove_spacing is None:
            _groove_note = mo.md(
                f"*Grooves need azimuth = 180 deg (selected {penetration_azim_deg:g} deg); "
                "showing the ungrooved slab.*"
            )

        def _make_sweep(spacing):
            return trajectory_sweep(
                MATERIAL,
                energies=(penetration_energy_keV,),
                tilts=(penetration_tilt_deg,),
                thickness_ang=penetration_thickness_ang,
                azim_deg=penetration_azim_deg,
                groove_spacing_ang=spacing,
            )

        try:
            _sweep = _make_sweep(_groove_spacing)
            _traj = build_configured_cases(_sweep, settings)
        except ValueError as _exc:
            # geometry/material rejects grooves (tilt=0, substrate/stack, ...):
            # fall back to the flat face and surface the reason.
            _groove_spacing = None
            _groove_note = mo.md(f"*Grooves not applied: {_exc}*")
            _sweep = _make_sweep(None)
            _traj = build_configured_cases(_sweep, settings)
        if not _traj:
            return mo.vstack([_md, mo.md("*No trajectory cases.*")])
        # The sweep has one selected energy and tilt; keep the nearest-case guard
        # in case a future sweep adds a surrounding grid.
        _nc = min(_traj, key=lambda c: (abs(c["tilt_deg"] - _angle), c["E0_keV"]))
        if penetration_secondaries_ui.value:
            _nc = dict(
                _nc,
                energy_model="midpoint",
                inelastic_model="shell-soft-hard",
                inelastic_cutoff_eV=50.0,
                secondary_threshold_eV=float(penetration_secondary_threshold_ui.value),
            )

        # Survival-chart cache: penetration_survival_chart runs its OWN fresh
        # Ne=500-electron transport, so without caching it would redo that work
        # every time the tab body reruns for ANY reason -- e.g. toggling
        # Realistic mode, which the survival curve does not even depend on.
        # Keyed on just the knobs the chart itself depends on.
        _survival_key = (
            MATERIAL,
            penetration_energy_keV,
            penetration_tilt_deg,
            penetration_azim_deg,
            penetration_thickness_ang,
            _groove_spacing,
        )
        _cached_survival = get_penetration_survival()
        if _cached_survival is not None and _cached_survival[0] == _survival_key:
            _survival = _cached_survival[1]
        else:
            _survival = penetration_survival_chart(_traj, Ne=500, tilt=_angle, width=420)
            set_penetration_survival((_survival_key, _survival))
        _survival = apply_altair_theme(_survival, _theme)

        _seed = int(penetration_regen_ui.value)
        _Ne = int(penetration_ne_ui.value)
        _beam_fwhm = float(penetration_beam_fwhm_ui.value)
        _realistic = penetration_realistic_ui.value

        # The transported dataset depends only on (case, Ne, seed, realistic,
        # beam_fwhm). Cache the whole `data` dict in mo.state keyed on that
        # tuple so Monte Carlo transport runs once per parameter set, not once
        # per rerun of an unrelated control.
        _data_key = (
            _nc["name"],
            _nc["E0_keV"],
            _nc.get("tilt_deg"),
            _nc.get("tilt_azim_deg"),
            _nc.get("groove_spacing_ang"),
            _Ne,
            _seed,
            _realistic,
            _beam_fwhm,
            bool(penetration_secondaries_ui.value),
            float(penetration_secondary_threshold_ui.value)
            if penetration_secondaries_ui.value
            else None,
        )
        _cached = get_penetration_data()
        if _cached is not None and _cached[0] == _data_key:
            _data = _cached[1]
        else:
            _data = trajectory_volume_data(
                _nc, Ne=_Ne, seed=_seed, realistic=_realistic, beam_fwhm_mm=_beam_fwhm
            )
            # _nc rides along so the eager render cell (below the tabs) can
            # feed the SAME case + data to render_reveal_animation without
            # redoing transport.
            set_penetration_data((_data_key, _data, _nc))

        # Static full-reveal figure (orbit/hover intact); the old interactive
        # Plotly frame animation was intentionally removed; the static figure remains.
        # workstream 2. Smooth playback now comes from the Render button below,
        # which prerenders a fixed-camera video via render_reveal_animation.
        _max_generation = int(penetration_max_generation_ui.value)
        _min_energy = float(penetration_min_energy_ui.value)
        _view_data = visible_trajectory_data(
            _data, max_generation=_max_generation, min_energy_keV=_min_energy
        )
        _volume = trajectory_volume_figure_from_data(
            _nc, _view_data, realistic=_realistic, beam_fwhm_mm=_beam_fwhm, reveal_until_fs=None
        )
        # Full-row plot now that the 2D cross-section moved down beside the
        # survival chart; widened past Plotly's 700 default now the colorbar
        # (restored below) has room without cramping the scene.
        if _volume is not None:
            apply_plotly_theme(_volume, _theme)
            _volume.update_layout(width=900)
            # Explicit camera, so the modebar PNG and the offscreen render
            # below both frame the scene the way this tab shows it.
            _volume.update_scenes(camera=penetration_camera)

        _beam_metrics = initial_state_metrics(
            _data["initial_r_ang"],
            _data["initial_v_hat"],
            _data["initial_t0_ang"],
            _data["initial_E_keV"],
            _sweep.beam,
        )
        _gaussian_equivalent_peak_current = (
            "inf"
            if np.isinf(_beam_metrics.gaussian_equivalent_peak_current_a)
            else f"{_beam_metrics.gaussian_equivalent_peak_current_a * 1e3:.3g} mA"
        )
        _beam_diagnostics = mo.md(
            "**Sampled beam**  "
            f"σx={_beam_metrics.x.position_sigma_mm * 1e3:.3g} µm; "
            f"σy={_beam_metrics.y.position_sigma_mm * 1e3:.3g} µm; "
            f"σt={_beam_metrics.sigma_t_fs:.3g} fs; "
            f"εn,x/y={_beam_metrics.x.normalized_emittance_mm_rad:.3g}/"
            f"{_beam_metrics.y.normalized_emittance_mm_rad:.3g} mm rad; "
            f"Q={_beam_metrics.bunch_charge_pc:.3g} pC; "
            f"Ipk,Gauss-eq={_gaussian_equivalent_peak_current}; "
            f"Iavg={_beam_metrics.average_current_a * 1e9:.3g} nA"
        )

        # Render button: prerender a smooth, looping fixed-camera video of the
        # SAME reveal sequence the old client-side animation played, offscreen
        # via kaleido + ffmpeg (see pyrite.plots.plotly.render). Cached
        # under ~/.cache/pyrite/viewer-renders keyed on every parameter the
        # render depends on, so an unchanged parameter set short-circuits to
        # the existing file instead of re-rendering.
        _custom_view = penetration_camera_view_ui.value == "custom"
        _camera_controls = mo.hstack(
            [
                penetration_camera_view_ui,
                dim_unless(penetration_camera_orbit_ui, _custom_view),
                dim_unless(penetration_camera_azim_ui, _custom_view),
                dim_unless(penetration_camera_elev_ui, _custom_view),
                penetration_camera_zoom_ui,
            ],
            justify="start",
            align="center",
            gap=1.5,
            wrap=True,
        )
        _render_controls = mo.hstack(
            [penetration_render_frames_ui, penetration_render_button_ui],
            justify="start",
            align="center",
            gap=1,
            wrap=True,
        )
        _n_frames = int(penetration_render_frames_ui.value)
        _render_key = render_cache_key(
            _nc["name"],
            (*_data_key[1:5], *_data_key[9:], _max_generation, _min_energy),
            _Ne,
            _seed,
            _realistic,
            _beam_fwhm,
            _n_frames,
            12,
            penetration_camera,
        )
        _render_path = cached_render_path(_render_key, ".mp4")
        # This lazy tab body CANNOT run the render itself: marimo resets
        # run_button.value to False when the click-triggered update completes,
        # and lazy tab content is evaluated afterwards in a separate request,
        # so the value is always False here. The render happens in the eager
        # cell below the tabs; this body only DISPLAYS -- video gated on the
        # cached file existing, errors relayed via penetration_render_status.
        _render_status = get_penetration_render_status()
        _render_error = None
        if (
            _render_status is not None
            and _render_status[0] == _render_key
            and _render_status[1] is not None
        ):
            _render_error = mo.callout(mo.md(_render_status[1]), kind="warn")
        if _render_error is not None:
            _render_block = _render_error
        elif _render_path.exists():
            # File handle (not str path) so marimo serves the bytes itself
            # instead of pointing the browser at a local filesystem path.
            _render_block = mo.video(
                src=_render_path.open("rb"),
                controls=True,
                loop=True,
                width=900,
                autoplay=True,
                muted=True,
            )
        else:
            _render_block = mo.md(
                f"*Click **Render animation** for smooth playback "
                f"(~{max(1, round(_n_frames * 3 / 60))} min offscreen render; "
                f"cached per parameter set afterwards).*"
            )

        # Cache lives under ~/.cache/pyrite/viewer-renders, easy to forget
        # about; a download button lets the user pull a render straight to
        # their own filesystem without knowing that path.
        _save_row = mo.hstack(
            [
                mo.download(
                    data=_render_path.read_bytes,
                    filename=f"{str(_nc['name']).split()[0]}_{_render_key[:8]}.mp4",
                    label="Save render to disk",
                    mimetype="video/mp4",
                    disabled=not _render_path.exists(),
                )
            ],
            justify="start",
            wrap=True,
        )

        def _row_label(text):
            return mo.md(text).style({"min-width": "9rem", "display": "inline-block"})

        # Beam FWHM only matters in true-beamsize mode; gray it out (dim +
        # non-interactive) rather than hide it, so its value stays visible.
        _beam_fwhm_display = (
            penetration_beam_fwhm_ui
            if _realistic
            else penetration_beam_fwhm_ui.style({"opacity": "0.4", "pointer-events": "none"})
        )

        # Built eagerly (cheap at Ne=40); no longer behind a lazy accordion.
        # Sits beside the survival chart now, not the 3D plot, so back to its
        # own 480 default width.
        _cross_section_chart = trajectory_chart(_nc, data=_view_data, width=420)
        _cross_section_block = (
            apply_altair_theme(_cross_section_chart, _theme)
            if _cross_section_chart is not None
            else mo.md("*No tracks pass the display filters.*")
        )

        _bottom_cols = [p for p in (_cross_section_block, _survival) if p is not None]
        _bottom_row = (
            mo.hstack(_bottom_cols, justify="start", align="start", gap=1, wrap=True)
            if _bottom_cols
            else None
        )

        _parts = [
            _md,
            _beam_diagnostics,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Beam energy**"),
                                    penetration_energy_source_ui,
                                    (
                                        penetration_energy_grid_ui
                                        if penetration_energy_source_ui.value == "grid"
                                        else penetration_energy_manual_ui
                                    ),
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            mo.hstack(
                                [
                                    _row_label("True beamsize"),
                                    penetration_realistic_ui,
                                    _beam_fwhm_display,
                                ],
                                justify="end",
                                align="end",
                                gap=1,
                                wrap=True,
                            ),
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Crystal thickness**"),
                                    penetration_thickness_source_ui,
                                    (
                                        penetration_thickness_grid_ui
                                        if penetration_thickness_source_ui.value == "grid"
                                        else penetration_thickness_manual_ui
                                    ),
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            penetration_ne_ui,
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            _row_label("**Polar tilt**"),
                            penetration_tilt_source_ui,
                            (
                                penetration_tilt_grid_ui
                                if penetration_tilt_source_ui.value == "grid"
                                else penetration_tilt_manual_ui
                            ),
                        ],
                        justify="start",
                        align="center",
                        gap=1,
                        wrap=True,
                    ),
                    mo.hstack(
                        [
                            mo.hstack(
                                [
                                    _row_label("**Azimuth**"),
                                    penetration_azim_ui,
                                ],
                                justify="start",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                            mo.hstack(
                                [
                                    _row_label("**Grooves**"),
                                    penetration_groove_ui,
                                ],
                                justify="end",
                                align="center",
                                gap=1,
                                wrap=True,
                            ),
                        ],
                        justify="space-between",
                        align="center",
                        wrap=True,
                    ),
                    *((_groove_note,) if _groove_note is not None else ()),
                    mo.hstack(
                        [
                            penetration_secondaries_ui,
                            dim_unless(
                                penetration_secondary_threshold_ui, penetration_secondaries_ui.value
                            ),
                            penetration_max_generation_ui,
                            penetration_min_energy_ui,
                        ],
                        justify="start",
                        align="center",
                        gap=1,
                        wrap=True,
                    ),
                    mo.hstack([penetration_regen_ui], justify="end", wrap=True),
                ]
            ),
            *(
                p
                for p in (
                    _volume,
                    _camera_controls,
                    _render_controls,
                    _render_block,
                    _save_row,
                    _bottom_row,
                )
                if p is not None
            ),
        ]
        return mo.vstack(_parts)

    return (penetration_tab,)


@app.cell
def _(DEFAULT_ANGLES, camera_controls, mo):
    # A few unit cells in each direction (a x b x c) gives enough of the stack
    # to read layering without the scene turning to soup.
    crystal_na_ui = mo.ui.slider(1, 4, value=3, step=1, label="cells a")
    crystal_nb_ui = mo.ui.slider(1, 4, value=3, step=1, label="cells b")
    crystal_nc_ui = mo.ui.slider(1, 3, value=2, step=1, label="cells c")
    crystal_bonds_ui = mo.ui.switch(value=True, label="show bonds")
    crystal_layers_ui = mo.ui.switch(value=False, label="color by layer")
    # Arrows are off by default: they dominate the scene and only matter when
    # reading diffraction families, so the structure view starts uncluttered.
    crystal_reciprocal_show_ui = mo.ui.switch(value=False, label="show reciprocal vectors")
    crystal_reciprocal_ui = mo.ui.slider(1, 8, value=1, step=1, label="reciprocal vectors")
    # Two independent scene toggles: the gridlines are the tiled unit-cell
    # edges running THROUGH the structure (the depth cue), the axes are the
    # whole surrounding frame.
    crystal_grid_ui = mo.ui.switch(value=True, label="show gridlines")
    crystal_axes_ui = mo.ui.switch(value=True, label="show axes")
    # Orthographic by default: perspective foreshortening bends parallel cell
    # edges toward a vanishing point, so columns of atoms only line up on a
    # zone axis without it.
    crystal_ortho_ui = mo.ui.switch(value=True, label="orthographic camera")
    # crystal_lattice_figure ships no camera of its own, so the controls seed
    # from Plotly's default eye -- "default" reproduces the untouched view.
    (
        crystal_view_ui,
        crystal_orbit_ui,
        crystal_azim_ui,
        crystal_elev_ui,
        crystal_zoom_ui,
    ) = camera_controls(DEFAULT_ANGLES)
    return (
        crystal_azim_ui,
        crystal_elev_ui,
        crystal_orbit_ui,
        crystal_axes_ui,
        crystal_bonds_ui,
        crystal_grid_ui,
        crystal_layers_ui,
        crystal_na_ui,
        crystal_nb_ui,
        crystal_nc_ui,
        crystal_ortho_ui,
        crystal_reciprocal_show_ui,
        crystal_reciprocal_ui,
        crystal_view_ui,
        crystal_zoom_ui,
    )


@app.cell
def _(
    CATALOG,
    DEFAULT_ANGLES,
    MATERIAL,
    apply_plotly_theme,
    crystal_azim_ui,
    crystal_axes_ui,
    crystal_bonds_ui,
    crystal_elev_ui,
    crystal_grid_ui,
    crystal_lattice_figure,
    crystal_layers_ui,
    crystal_na_ui,
    crystal_nb_ui,
    crystal_nc_ui,
    crystal_orbit_ui,
    crystal_ortho_ui,
    crystal_reciprocal_show_ui,
    crystal_reciprocal_ui,
    crystal_view_ui,
    crystal_zoom_ui,
    data_aspect_ratio,
    dim_unless,
    mo,
    resolve_scene_camera,
    resolved_theme,
    theme_ui,
):
    _theme = resolved_theme(theme_ui)

    def crystal_tab():
        _md = mo.md(
            "### Crystal structure\n"
            "Ball-and-stick view of the selected material's unit cell tiled over "
            "a few cells. Spheres are element-colored (sized by covalent radius); "
            "switch on reciprocal vectors to overlay the strongest "
            "reciprocal-lattice families as arrows in descending strength. "
            "Adjust cell tiling, bonds, coloring, arrows, axes, and camera below."
        )
        if MATERIAL is None:
            return mo.vstack([_md, mo.md("_Select a material to view its lattice._")])
        _material = CATALOG.material(MATERIAL)
        _fig = crystal_lattice_figure(
            CATALOG.crystal(_material.crystal),
            n_a=crystal_na_ui.value,
            n_b=crystal_nb_ui.value,
            n_c=crystal_nc_ui.value,
            label=_material.label,
            show_bonds=crystal_bonds_ui.value,
            color_by="layer" if crystal_layers_ui.value else "element",
            n_reciprocal_vectors=(
                crystal_reciprocal_ui.value if crystal_reciprocal_show_ui.value else 0
            ),
        )
        apply_plotly_theme(_fig, _theme)
        _fig.update_layout(height=680)
        # Both applied after the theme, which owns axis colors but not
        # visibility. One switch takes the entire axis frame down -- walls,
        # wall rules, ticks, labels, and titles -- via axis-level ``visible``.
        _fig.update_scenes(
            xaxis={"visible": crystal_axes_ui.value},
            yaxis={"visible": crystal_axes_ui.value},
            zaxis={"visible": crystal_axes_ui.value},
        )
        # The gridlines that read as depth are the per-cell edges drawn through
        # the structure, not anything on the axis walls: that is the ``cell``
        # trace crystal_lattice_figure adds first.
        _fig.update_traces(visible=crystal_grid_ui.value, selector={"name": "cell"})
        # One explicit camera carries both the viewpoint and the projection.
        # Orthographic drops the perspective parallax so parallel lattice
        # directions stay parallel at every depth; because the camera lives in
        # the figure spec, the modebar's "Download plot as png" gets this exact
        # view instead of falling back to the default one.
        _camera = resolve_scene_camera(
            crystal_view_ui.value,
            crystal_azim_ui.value,
            crystal_elev_ui.value,
            crystal_zoom_ui.value,
            default_angles=DEFAULT_ANGLES,
            orbit_axis=crystal_orbit_ui.value,
            orthographic=crystal_ortho_ui.value,
        )
        _fig.update_scenes(camera=_camera)
        if crystal_ortho_ui.value:
            # An orthographic projection box is fixed, so the camera distance
            # the zoom rides on does nothing here. Plotly zooms an ortho scene
            # by scaling the aspect ratio instead (that is what its own scroll
            # handler does), and any aspectmode but "manual" would recompute
            # the ratio and throw this away.
            _fig.update_scenes(
                aspectmode="manual",
                aspectratio=data_aspect_ratio(_fig, crystal_zoom_ui.value),
            )
        # Angles only bite under the "custom" view; the arrow count only while
        # the arrows are drawn. Dim rather than hide, so values stay visible.
        _custom_view = crystal_view_ui.value == "custom"
        _reciprocal_count = dim_unless(crystal_reciprocal_ui, crystal_reciprocal_show_ui.value)
        _controls = mo.hstack(
            [
                crystal_na_ui,
                crystal_nb_ui,
                crystal_nc_ui,
                crystal_bonds_ui,
                crystal_layers_ui,
                crystal_reciprocal_show_ui,
                _reciprocal_count,
                crystal_grid_ui,
                crystal_axes_ui,
                crystal_ortho_ui,
            ],
            justify="start",
            gap=1.5,
            wrap=True,
        )
        _camera_row = mo.hstack(
            [
                crystal_view_ui,
                dim_unless(crystal_orbit_ui, _custom_view),
                dim_unless(crystal_azim_ui, _custom_view),
                dim_unless(crystal_elev_ui, _custom_view),
                crystal_zoom_ui,
            ],
            justify="start",
            gap=1.5,
            wrap=True,
        )
        return mo.vstack([_md, _controls, _camera_row, _fig])

    return (crystal_tab,)


@app.cell
def _(MATERIAL, crystal_tab, mo, penetration_tab):
    # Two views, one 3D scene each; lazy so the un-viewed tab never runs
    # transport or builds a lattice figure.
    if MATERIAL is None:
        view = mo.callout(
            mo.md("**No materials configured.** Check `data/catalog/`."),
            kind="info",
        )
    else:
        view = mo.ui.tabs(
            {
                "Trace": penetration_tab,
                "Structure": crystal_tab,
            },
            lazy=True,
        )
    view
    return


@app.cell
def _(
    cached_render_path,
    get_penetration_data,
    mo,
    penetration_camera,
    penetration_render_button_ui,
    penetration_render_frames_ui,
    penetration_max_generation_ui,
    penetration_min_energy_ui,
    prune_render_cache,
    render_cache_key,
    render_reveal_animation,
    set_penetration_render_status,
    visible_trajectory_data,
):
    # EAGER render cell -- must live outside the lazy tab body. marimo resets
    # run_button.value to False as soon as the click-triggered update
    # completes, and lazy tab content re-evaluates afterwards in a separate
    # request, so a render gated on the button INSIDE the tab never fires.
    # This cell reruns synchronously on click (it references the button), does
    # the blocking kaleido/ffmpeg render with a visible progress bar (shown
    # here, just below the tabs -- close to the button that triggered it, in
    # the "Trace" tab above), then flips penetration_render_status so the tab
    # body reruns and picks up the cached video file.
    mo.stop(not penetration_render_button_ui.value)
    _cached = get_penetration_data()
    mo.stop(
        _cached is None,
        mo.md("*Open the Trace tab once before rendering (no transported data yet).*"),
    )
    _data_key, _data, _nc = _cached
    _n_frames = int(penetration_render_frames_ui.value)
    # Same key recipe as the tab body: _data_key is (name, E0, tilt, azim,
    # groove, Ne, seed, realistic, beam_fwhm, secondary mode, threshold).
    _max_generation = int(penetration_max_generation_ui.value)
    _min_energy = float(penetration_min_energy_ui.value)
    _render_key = render_cache_key(
        _data_key[0],
        (*_data_key[1:5], *_data_key[9:], _max_generation, _min_energy),
        _data_key[5],
        _data_key[6],
        _data_key[7],
        _data_key[8],
        _n_frames,
        12,
        penetration_camera,
    )
    _render_path = cached_render_path(_render_key, ".mp4")
    if not _render_path.exists():
        try:
            # ~3 s/frame through kaleido at this figure size, measured on the
            # Ne=50 hopg case -- surface the wait up front so a multi-minute
            # blocking render doesn't read as a dead button.
            with mo.status.progress_bar(
                total=_n_frames,
                title="Rendering animation",
                subtitle=f"{_n_frames} frames, ~3 s each offscreen",
            ) as _bar:

                def _render_progress_cb(_k, _n, _bar=_bar):
                    _bar.update()

                render_reveal_animation(
                    _nc,
                    visible_trajectory_data(
                        _data, max_generation=_max_generation, min_energy_keV=_min_energy
                    ),
                    _render_path,
                    realistic=_data_key[7],
                    beam_fwhm_mm=_data_key[8],
                    n_frames=_n_frames,
                    fps=12,
                    camera=penetration_camera,
                    progress_cb=_render_progress_cb,
                )
            prune_render_cache()
            set_penetration_render_status((_render_key, None))
        except RuntimeError as _exc:
            set_penetration_render_status((_render_key, str(_exc)))
    else:
        set_penetration_render_status((_render_key, None))
    return


if __name__ == "__main__":
    app.run()
