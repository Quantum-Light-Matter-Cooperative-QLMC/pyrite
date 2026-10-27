# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.25.1",
# ]
#
# [tool.marimo.display]
# theme = "light"
# ///
"""Directional transported-source and internal Cherenkov energy–angle maps."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import time

    import marimo as mo

    from pyrite import fetch_tables
    from pyrite._backend import BACKEND
    from pyrite.apps._design import context_rail, page_title, style_sheet
    from pyrite.apps.angular_heatmap import (
        COMPONENT_LABELS,
        MapConfig,
        calculate_maps_reporting_cache,
        maps_npz,
        parse_tilts,
        tilt_panels,
    )
    from pyrite.apps.angular_maps import (
        form_template,
        heatmap_figure,
        material_options,
        profile_figure,
        transport_cores,
    )

    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Directional source heatmap",
                "PXR/CBS, characteristic, bremsstrahlung, planar TR and bulk Cherenkov"
                " · no detector or checkpoint",
            ),
            mo.md("""
        Select transported radiation in **ph/s/eV/sr** versus signed lab-plane
        angle (**0° = incoming beam**, ±180° = backward), or the separate internal
        Cherenkov source in **ph/electron/eV/polar bin**. Each view states its
        geometry and normalization. No aperture, filter or detector response.
        """),
        ]
    )
    return (
        BACKEND,
        COMPONENT_LABELS,
        MapConfig,
        calculate_maps_reporting_cache,
        context_rail,
        fetch_tables,
        form_template,
        heatmap_figure,
        maps_npz,
        material_options,
        mo,
        parse_tilts,
        profile_figure,
        tilt_panels,
        time,
        transport_cores,
    )


@app.cell
def _(mo):
    source_view = mo.ui.radio(
        ["Transported sources", "Cherenkov (internal bulk)"],
        value="Transported sources",
        label="Radiation view",
        inline=True,
    )
    source_view
    return (source_view,)


@app.cell
def _(fetch_tables, mo, source_view):
    mo.stop(source_view.value != "Transported sources")
    # Fresh hosted sessions (e.g. molab) start without PyRITE's pinned data;
    # a complete install returns without network access.
    try:
        with mo.status.spinner("Checking PyRITE reference data and tables…"):
            _fetched = [r["code"] for r in fetch_tables() if r["installed"]]
    except Exception as error:  # noqa: BLE001 - surface any fetch failure in the UI
        mo.stop(True, mo.callout(mo.md(f"Table fetch failed: {error}"), kind="danger"))
    tables_ready = True
    mo.md(f"Fetched PyRITE data: {', '.join(_fetched)}.") if _fetched else None
    return (tables_ready,)


@app.cell
def _(
    BACKEND,
    COMPONENT_LABELS,
    MapConfig,
    form_template,
    material_options,
    mo,
    source_view,
    transport_cores,
):
    mo.stop(source_view.value != "Transported sources")
    _d = MapConfig()
    _materials = material_options()
    _cores = transport_cores(BACKEND.name)
    scene_form = (
        mo.Html(
            form_template(
                {
                    "Beam": ("beam", "electrons", "seed"),
                    "Target": ("material", "thickness", "tilts", "tilt_azimuth"),
                    "Footprint": ("width", "height", "infinite"),
                    "Radiation": ("components", "plane", "angle_points"),
                    "Photon energy": ("start", "auto_stop", "stop", "energy_points", "log_energy"),
                    "Execution": ("reflections", "core", "batch"),
                }
            )
        )
        .batch(
            beam=mo.ui.number(1, 1e6, value=_d.beam_keV, label="Energy (keV)", full_width=True),
            electrons=mo.ui.number(
                1, 100_000_000, value=_d.electrons, label="Histories / tilt", full_width=True
            ),
            seed=mo.ui.number(0, 2**31 - 1, value=_d.seed, label="Seed", full_width=True),
            material=mo.ui.dropdown(
                _materials,
                value=next(k for k, v in _materials.items() if v == _d.material),
                label="Material",
                searchable=True,
                full_width=True,
            ),
            thickness=mo.ui.number(
                1, 1e9, value=_d.thickness_ang, label="Thickness (Å)", full_width=True
            ),
            tilts=mo.ui.text(
                value="30, 45",
                label="Tilts (deg): list or start:stop:step",
                full_width=True,
            ),
            tilt_azimuth=mo.ui.text(
                value="0", label="Target azimuths (deg): list or start:stop:step", full_width=True
            ),
            width=mo.ui.number(1e-4, 1000, value=_d.width_mm, label="Width (mm)", full_width=True),
            height=mo.ui.number(
                1e-4, 1000, value=_d.height_mm, label="Height (mm)", full_width=True
            ),
            infinite=mo.ui.checkbox(value=False, label="Infinite slab (limiting case)"),
            components=mo.ui.multiselect(
                {label: key for key, label in COMPONENT_LABELS.items()},
                value=[COMPONENT_LABELS[c] for c in _d.components],
                label="Components",
                full_width=True,
            ),
            plane=mo.ui.number(
                0, 359.9, value=0, step=0.1, label="Plane azimuth (deg)", full_width=True
            ),
            angle_points=mo.ui.number(
                4, 36_000, value=_d.angle_points, label="Angle samples", full_width=True
            ),
            start=mo.ui.number(10, 1e9, value=_d.start_eV, label="Start (eV)", full_width=True),
            auto_stop=mo.ui.checkbox(
                value=True, label="Automatic stop (TR: beam; otherwise: lines)"
            ),
            stop=mo.ui.number(20, 1e9, value=5000, label="Manual stop (eV)", full_width=True),
            energy_points=mo.ui.number(
                2, 1_000_000, value=_d.energy_points, label="Points", full_width=True
            ),
            log_energy=mo.ui.checkbox(value=_d.log_energy, label="Logarithmic spacing"),
            reflections=mo.ui.number(
                1, 256, value=_d.max_reflections, label="Max reflections", full_width=True
            ),
            core=mo.ui.dropdown(
                _cores, value=_d.transport_core, label="Transport core", full_width=True
            ),
            batch=mo.ui.number(
                1, 4096, value=_d.direction_batch, label="Direction batch", full_width=True
            ),
        )
        .form(submit_button_label="Run source maps", bordered=True)
    )
    scene_form
    return (scene_form,)


@app.cell
def _(
    BACKEND,
    MapConfig,
    calculate_maps_reporting_cache,
    mo,
    parse_tilts,
    scene_form,
    tables_ready,
    time,
):
    assert tables_ready
    _v = scene_form.value
    try:
        config = (
            MapConfig()
            if _v is None
            else MapConfig(
                material=_v["material"],
                beam_keV=_v["beam"],
                thickness_ang=_v["thickness"],
                width_mm=None if _v["infinite"] else _v["width"],
                height_mm=None if _v["infinite"] else _v["height"],
                tilts_deg=parse_tilts(_v["tilts"]),
                tilt_azimuths_deg=parse_tilts(_v["tilt_azimuth"]),
                plane_azimuth_deg=_v["plane"],
                start_eV=_v["start"],
                stop_eV=None if _v["auto_stop"] else _v["stop"],
                energy_points=int(_v["energy_points"]),
                log_energy=_v["log_energy"],
                angle_points=int(_v["angle_points"]),
                electrons=int(_v["electrons"]),
                seed=int(_v["seed"]),
                max_reflections=int(_v["reflections"]),
                transport_core=_v["core"],
                direction_batch=int(_v["batch"]),
                components=tuple(_v["components"]),
            )
        )
        _start = time.perf_counter()
        with mo.status.spinner(
            f"{BACKEND.name.upper()}: one transport per polar/azimuth orientation, then "
            f"{config.scoring_work:.3g} electron·component·direction·energy evaluations…"
        ):
            maps, cached = calculate_maps_reporting_cache(config)
        elapsed = time.perf_counter() - _start
    except ValueError as error:
        mo.stop(True, mo.callout(mo.md(str(error)), kind="danger"))
    return cached, config, elapsed, maps


@app.cell
def _(COMPONENT_LABELS, config, maps, maps_npz, mo):
    _tilts = {
        f"{m[config.components[0]].case['tilt_deg']:g}° / "
        f"{m[config.components[0]].case['tilt_azim_deg']:g}°": i
        for i, m in enumerate(maps)
    }
    _first = maps[0][config.components[0]]
    component = mo.ui.radio(
        {COMPONENT_LABELS[c]: c for c in config.components},
        value=COMPONENT_LABELS[config.components[0]],
        label="Component",
        inline=True,
    )
    current = mo.ui.number(0, 1e9, value=_first.current_na, label="Mean current (nA)")
    color = mo.ui.checkbox(value=True, label="Log color")
    decades = mo.ui.slider(1, 16, value=8, label="Log range (decades)")
    layout = mo.ui.radio(
        ["Compare selected", "Step through tilts"],
        value="Compare selected" if len(maps) <= 4 else "Step through tilts",
        label="Tilt view",
        inline=True,
    )
    selection = mo.ui.multiselect(
        options=_tilts, value=list(_tilts)[:4], label="Tilts shown (compare)"
    )
    tilt_values = sorted({float(m[config.components[0]].case["tilt_deg"]) for m in maps})
    step = mo.ui.slider(
        steps=tilt_values, value=tilt_values[0], label="Tilt (deg, step)", show_value=True
    )
    _azimuths = sorted({float(m[config.components[0]].case["tilt_azim_deg"]) for m in maps})
    azimuth_step = mo.ui.slider(
        steps=_azimuths, value=_azimuths[0], label="Target azimuth (deg)", show_value=True
    )
    cut_energy = mo.ui.number(
        float(_first.energy_eV[0]),
        float(_first.energy_eV[-1]),
        value=float(_first.energy_eV[len(_first.energy_eV) // 2]),
        label="Cut energy (eV)",
    )
    cut_angle = mo.ui.number(-180, 180, value=90, step=0.1, label="Cut angle (deg)")
    # Hosted sessions keep no generated files; the archive is built at click time.
    download = mo.download(
        data=lambda: maps_npz(config, maps, current_na=current.value),
        filename="directional_source_maps.npz",
        mimetype="application/octet-stream",
        label="Download results (.npz)",
    )
    mo.vstack(
        [
            mo.md(
                "**Display** · sliders browse cached polar/azimuth pairs, no recompute. "
                "New orientations require **Run source maps** above. "
                "TR requires the explicitly selected infinite-slab geometry."
            ),
            mo.hstack([component, layout], justify="start", gap=2, wrap=True),
            mo.hstack(
                [current, color, decades, selection, step, azimuth_step],
                justify="start",
                gap=2,
                align="end",
                wrap=True,
            ),
            mo.hstack([cut_energy, cut_angle, download], justify="start", gap=2, wrap=True),
        ]
    )
    return (
        azimuth_step,
        color,
        component,
        current,
        cut_angle,
        cut_energy,
        decades,
        layout,
        selection,
        step,
        tilt_values,
    )


@app.cell
def _(
    BACKEND,
    azimuth_step,
    cached,
    color,
    component,
    config,
    context_rail,
    current,
    cut_angle,
    cut_energy,
    decades,
    elapsed,
    heatmap_figure,
    layout,
    maps,
    mo,
    profile_figure,
    selection,
    step,
    tilt_panels,
):
    _indices = (
        [
            i
            for i, m in enumerate(maps)
            if m[component.value].case["tilt_deg"] == step.value
            and m[component.value].case["tilt_azim_deg"] == azimuth_step.value
        ]
        if layout.value == "Step through tilts"
        else sorted(selection.value)
    )
    mo.stop(not _indices, mo.md("Select at least one target tilt."))
    _shown = [maps[i] for i in _indices]
    _panels = tilt_panels(_shown, component.value, current_na=current.value)
    _first = maps[0][component.value]
    _grid = _first.energy_eV
    _ceiling = max(m[component.value].provenance["line_window_ceiling_eV"] for m in maps)
    _footprint = (
        "infinite" if config.width_mm is None else f"{config.width_mm:g} × {config.height_mm:g} mm"
    )
    _label = f"ph/s/eV/sr at {current.value:g} nA"
    mo.vstack(
        [
            context_rail(
                mo,
                {
                    "Backend": BACKEND.name,
                    "Core": config.transport_core,
                    "Elapsed": "cached" if cached else f"{elapsed:.1f} s",
                    "Footprint": _footprint,
                    "Photon grid": f"{_grid[0]:g}–{_grid[-1]:g} eV × {len(_grid)}",
                    "Line ceiling": f"{_ceiling:g} eV",
                    "Histories/orientation": config.electrons,
                },
            ),
            mo.callout(
                mo.md(
                    f"The photon grid stops at **{_grid[-1]:g} eV**, below the kinematic "
                    f"line ceiling of **{_ceiling:g} eV**. Relativistic beams blueshift "
                    "near-forward lines, so empty bands there can be window truncation, "
                    "not emission nulls. Enable *Stop at kinematic line ceiling* to check."
                ),
                kind="warn",
            )
            if _grid[-1] < _ceiling and component.value == "pxr-cbs"
            else mo.md(""),
            heatmap_figure(
                _panels,
                value_label=_label,
                logarithmic=color.value,
                dynamic_range=10.0**decades.value,
                periodic=True,
            ),
            profile_figure(
                _panels,
                energy_eV=cut_energy.value,
                angle_deg=cut_angle.value,
                value_label=_label,
                logarithmic=color.value,
            ),
        ]
    )
    return


@app.cell
def _(maps, mo):
    mo.vstack(
        [
            mo.md(f"""
    Gray cells are unsupported directions (infinite slab only); zero intensity
    takes the lowest color. The mean current defaults to the beam's
    {next(iter(maps[0].values())).current_na:g} nA (1 pC × 5 kHz) and only rescales
    cached yields. Grids are point samples: refine angles, energies and histories
    together before reading a peak height.
    """),
            mo.accordion(
                {
                    "Conventions, limits and convergence": mo.md("""
    - **Angle convention:** direction = sin θ (cos φ, sin φ, 0) + cos θ ẑ in the lab
      frame, with φ the plane azimuth. +90° points along +x at φ = 0. The +180°
      sample is not scored twice; the plot duplicates −180° only to close the circle.
    - **Escape:** each segment's emission is attenuated along the exact ray to its
      first crystal face, including side faces of the finite footprint. Directions
      parallel to the faces (θ = tilt ± 90° at zero azimuths) therefore stay a deep,
      physical attenuation minimum for mm footprints. The infinite slab is offered
      only as a limiting case; there such directions are masked as unsupported.
    - **Components:** separate maps, never summed. PXR/CBS keeps PXR–CBS
      interference. Characteristic lines are emitted isotropically, so their map
      shows escape geometry. Bremsstrahlung uses the BremsLib angular model when its
      tables are installed (`pyrite tables fetch bremslib`), otherwise isotropic EEDL.
      TR uses the production history-coherent planar crossing scorer, only for
      explicitly selected infinite slabs and supported X-ray dielectric bands.
      Bunch-coherent and channeling radiation are excluded.
    - **Photon window:** the automatic stop is the closed-form PXR/CBS bound
      ħcβ|g|max/(1 − β), capped at the beam energy. When TR is selected, automatic
      stop uses beam energy instead. Resolve/refine the relevant TR band manually;
      finite grids do not certify formation-zone or classical-emission validity.
    - **Tilts:** a comma list or `start:stop:step` range (e.g. `0:80:5`). Each tilt is
      one transport per target polar/azimuth pair; sliders browse cached pairs.
      Target azimuth rotates the sample; plane azimuth rotates the observation
      plane. Submit the form to calculate new orientations.
    - **Execution:** the array backend is fixed when the app starts. Launch with
      `pyrite app angular-heatmap launch --backend cuda` to offer the CUDA core.
      Use `pyrite.radiation.directional_sources` for scripted or remote studies.
    """)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, source_view):
    mo.stop(source_view.value != "Cherenkov (internal bulk)")
    from pyrite.apps.angular_cherenkov import cherenkov_form, submitted_config

    cone_form = cherenkov_form(mo)
    mo.vstack(
        [
            mo.md(
                "**Internal homogeneous Cherenkov source** · explicit constant index, straight "
                "fixed-speed passage. No target escape, interfaces, refraction or TIR. "
                "Target polar/azimuth settings do not affect this bulk model. "
                "This diagnostic is separate from TR and is never added to it. "
                "Choose a transparent band where the declared index is nearly constant "
                "and photon energy is far below the beam kinetic energy."
            ),
            cone_form,
        ]
    )
    return cone_form, submitted_config


@app.cell
def _(cone_form, mo, submitted_config):
    import numpy as _np

    from pyrite.apps.angular_cherenkov import VALUE_LABEL, calculate_cherenkov, cherenkov_npz

    try:
        cone_config = submitted_config(cone_form.value)
        cone_panel, _edges, _source = calculate_cherenkov(cone_config)
    except ValueError as error:
        mo.stop(True, mo.callout(mo.md(str(error)), kind="danger"))
    cone_energy = mo.ui.number(
        cone_config.start_eV,
        cone_config.stop_eV,
        value=float(cone_panel.energy_eV[len(cone_panel.energy_eV) // 2]),
        label="Cherenkov cut energy (eV)",
    )
    cone_angle = mo.ui.number(0, 180, value=45, label="Cone cut angle (deg)")
    mo.vstack(
        [
            mo.md(
                "Yield integrates each full-azimuth polar bin and the declared path; "
                "**ph/electron/eV/bin**, not a pointwise per-sr density. "
                + (
                    "Below threshold: physical zero yield."
                    if not _source.above_threshold.any()
                    else f"Cone half-angle: {_np.degrees(_source.theta_rad[0]):.3g}°."
                )
            ),
            mo.hstack([cone_energy, cone_angle]),
            mo.download(
                data=lambda: cherenkov_npz(cone_config),
                filename="cherenkov_bulk_source.npz",
                mimetype="application/octet-stream",
                label="Download Cherenkov source (.npz)",
            ),
        ]
    )
    return VALUE_LABEL, cone_angle, cone_energy, cone_panel


@app.cell
def _(VALUE_LABEL, cone_angle, cone_energy, cone_panel, heatmap_figure, mo, profile_figure):
    mo.vstack(
        [
            heatmap_figure(
                [cone_panel],
                value_label=VALUE_LABEL,
                angle_label="Polar angle about velocity (deg, full-azimuth bins)",
            ),
            profile_figure(
                [cone_panel],
                value_label=VALUE_LABEL,
                energy_eV=cone_energy.value,
                angle_deg=cone_angle.value,
                angle_label="Polar-bin centre (deg)",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
