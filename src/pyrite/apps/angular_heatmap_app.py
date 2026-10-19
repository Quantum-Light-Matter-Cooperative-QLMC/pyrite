# /// script
# [tool.marimo.display]
# theme = "light"
# ///
"""Detector-free incoherent PXR/CBS energy–angle heatmap."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pyrite.apps._design import page_title, style_sheet
    from pyrite.apps.angular_heatmap import MapConfig, calculate_maps, heatmap_figure

    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Directional source heatmap",
                "Escaped incoherent PXR/CBS · no detector or checkpoint",
            ),
            mo.md("""
        A signed lab-plane slice: **0° follows the incoming +z beam**, +90° points
        along +x at plane azimuth 0°, and ±180° is backward. Plane azimuth rotates
        the transverse basis toward +y. This is **ph/s/eV/sr**, with target escape
        absorption; no aperture, response, filters or angular integration.
        PXR–CBS interference is retained. All other radiation channels are excluded.
        """),
        ]
    )
    return MapConfig, calculate_maps, heatmap_figure, mo


@app.cell
def _(mo):
    scene_form = (
        mo.md("""
    **Source and target** {material} {beam} {thickness} {tilts} {tilt_azimuth} {plane}

    **Common explicit point grids** {start} {stop} {energy_points} {angle_points}

    **Sampling** {electrons} {seed}
    """)
        .batch(
            material=mo.ui.dropdown(
                ["silicon", "hopg", "diamond"], value="silicon", label="Material"
            ),
            beam=mo.ui.number(10, 1000, value=30, label="Beam (keV)"),
            thickness=mo.ui.number(1, 2000, value=20, label="Thickness (Å)"),
            tilts=mo.ui.text(value="30, 45", label="Target tilts (deg, comma separated; max 4)"),
            tilt_azimuth=mo.ui.number(0, 359, value=0, label="Tilt azimuth (deg)"),
            plane=mo.ui.number(0, 359, value=0, label="Observation-plane azimuth (deg)"),
            start=mo.ui.number(10, 999999, value=100, label="Photon start (eV)"),
            stop=mo.ui.number(20, 1000000, value=5000, label="Photon stop (eV)"),
            energy_points=mo.ui.number(2, 2049, value=129, label="Energy points"),
            angle_points=mo.ui.number(8, 720, value=72, label="Periodic angle samples"),
            electrons=mo.ui.number(1, 256, value=8, label="Incident macro-electrons"),
            seed=mo.ui.number(0, 1000000, value=1, label="Seed"),
        )
        .form(submit_button_label="Run source maps")
    )
    scene_form
    return (scene_form,)


@app.cell
def _(MapConfig, calculate_maps, mo, scene_form):
    _v = scene_form.value
    try:
        config = (
            MapConfig()
            if _v is None
            else MapConfig(
                material=_v["material"],
                beam_keV=_v["beam"],
                thickness_ang=_v["thickness"],
                tilts_deg=tuple(float(s.strip()) for s in _v["tilts"].split(",")),
                tilt_azimuth_deg=_v["tilt_azimuth"],
                plane_azimuth_deg=_v["plane"],
                start_eV=_v["start"],
                stop_eV=_v["stop"],
                energy_points=int(_v["energy_points"]),
                angle_points=int(_v["angle_points"]),
                electrons=int(_v["electrons"]),
                seed=int(_v["seed"]),
            )
        )
        with mo.status.spinner("Transport once per target tilt, then score source directions…"):
            maps = calculate_maps(config)
    except ValueError as error:
        mo.stop(True, mo.callout(mo.md(str(error)), kind="danger"))
    return config, maps


@app.cell
def _(maps, mo):
    current = mo.ui.number(0, 1000000, value=maps[0].current_na, label="Mean current (nA)")
    color = mo.ui.checkbox(value=True, label="Log color (zero and unsupported bins gray)")
    selection = mo.ui.multiselect(
        options={f"Tilt {r.case['tilt_deg']:g}°": i for i, r in enumerate(maps)},
        value=[f"Tilt {r.case['tilt_deg']:g}°" for r in maps],
        label="Compare target tilts",
    )
    mo.hstack([current, color, selection])
    return color, current, selection


@app.cell
def _(color, current, heatmap_figure, maps, mo, selection):
    mo.stop(not selection.value, mo.md("Select at least one target tilt."))
    heatmap_figure(
        tuple(maps[i] for i in selection.value), current_na=current.value, logarithmic=color.value
    )
    return


@app.cell
def _(config, maps, mo):
    mo.md(f"""
    Resolved input mean current: **{maps[0].current_na:g} nA** (1 pC × 5 kHz).
    Display current and color/tilt choices only redraw cached yields.
    Initial bounded example: {config.electrons} incident histories per tilt;
    {config.angle_points} periodic angles × {config.energy_points} energy points.
    No repeated +180° sample is scored. Gray in log mode includes zero;
    unsupported sample-face grazing directions are masked in every mode.

    **Convergence and limits:** grids are point samples, not energy/solid-angle bins.
    Increase both point counts and extend photon bounds before interpreting a peak;
    narrow resonances can be missed and out-of-window resonances are omitted by the
    existing source model. The thin default film deliberately broadens spectral features.
    Reflection set is capped at four; this is a quick diagnostic, not a converged yield
    certificate. Increase sampling/reflections through the reusable API and send heavy
    convergence studies to `pyrite remote`. Flat infinite slabs only in this UI;
    API also accepts flat stacks. Existing kinematic PXR/CBS and escape assumptions apply.

    `pyrite app angular-heatmap launch` · `pyrite app angular-heatmap export`
    """)
    return


if __name__ == "__main__":
    app.run()
