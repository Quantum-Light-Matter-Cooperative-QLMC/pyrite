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
    import time

    import marimo as mo

    from pyrite._backend import BACKEND
    from pyrite.apps._design import context_rail, page_title, style_sheet
    from pyrite.apps.angular_heatmap import (
        MapConfig,
        calculate_maps,
        form_template,
        heatmap_figure,
    )

    mo.vstack(
        [
            style_sheet(mo),
            page_title(
                mo,
                "Directional source heatmap",
                "Escaped incoherent PXR/CBS from a finite target · no detector or checkpoint",
            ),
            mo.md("""
        Photon intensity in **ph/s/eV/sr** versus signed observation angle in the
        lab plane (**0° = incoming beam**, ±180° = backward). Target self-absorption
        is traced to the actual crystal faces; no aperture, filter or detector response.
        """),
        ]
    )
    return (
        BACKEND,
        MapConfig,
        calculate_maps,
        context_rail,
        form_template,
        heatmap_figure,
        mo,
        time,
    )


@app.cell
def _(MapConfig, form_template, mo):
    _d = MapConfig()
    scene_form = (
        mo.Html(
            form_template(
                {
                    "Beam": ("beam", "electrons", "seed"),
                    "Target": ("material", "thickness", "tilts", "tilt_azimuth"),
                    "Footprint": ("width", "height", "infinite"),
                    "Observation": ("plane", "angle_points"),
                    "Photon energy": ("start", "auto_stop", "stop", "energy_points", "log_energy"),
                    "Execution": ("reflections", "core", "batch"),
                }
            )
        )
        .batch(
            beam=mo.ui.number(1, 100_000, value=_d.beam_keV, label="Energy (keV)", full_width=True),
            electrons=mo.ui.number(
                1, 10_000_000, value=_d.electrons, label="Histories / tilt", full_width=True
            ),
            seed=mo.ui.number(0, 2**31 - 1, value=_d.seed, label="Seed", full_width=True),
            material=mo.ui.dropdown(
                ["silicon", "hopg", "diamond"], value=_d.material, label="Material", full_width=True
            ),
            thickness=mo.ui.number(
                1, 1e8, value=_d.thickness_ang, label="Thickness (Å)", full_width=True
            ),
            tilts=mo.ui.text(value="30, 45", label="Tilts (deg)", full_width=True),
            tilt_azimuth=mo.ui.number(
                0, 359.5, value=0, step=0.5, label="Tilt azimuth (deg)", full_width=True
            ),
            width=mo.ui.number(1e-4, 1000, value=_d.width_mm, label="Width (mm)", full_width=True),
            height=mo.ui.number(
                1e-4, 1000, value=_d.height_mm, label="Height (mm)", full_width=True
            ),
            infinite=mo.ui.checkbox(value=False, label="Infinite slab (limiting case)"),
            plane=mo.ui.number(0, 359, value=0, label="Plane azimuth (deg)", full_width=True),
            angle_points=mo.ui.number(
                4, 36_000, value=_d.angle_points, label="Angle samples", full_width=True
            ),
            start=mo.ui.number(10, 1e8, value=_d.start_eV, label="Start (eV)", full_width=True),
            auto_stop=mo.ui.checkbox(value=True, label="Stop at kinematic line ceiling"),
            stop=mo.ui.number(20, 1e8, value=5000, label="Manual stop (eV)", full_width=True),
            energy_points=mo.ui.number(
                2, 1_000_000, value=_d.energy_points, label="Points", full_width=True
            ),
            log_energy=mo.ui.checkbox(value=_d.log_energy, label="Logarithmic spacing"),
            reflections=mo.ui.number(
                1, 256, value=_d.max_reflections, label="Max reflections", full_width=True
            ),
            core=mo.ui.dropdown(
                ["auto", "lockstep", "per-electron", "cuda"],
                value=_d.transport_core,
                label="Transport core",
                full_width=True,
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
def _(BACKEND, MapConfig, calculate_maps, mo, scene_form, time):
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
                tilts_deg=tuple(float(s) for s in _v["tilts"].split(",") if s.strip()),
                tilt_azimuth_deg=_v["tilt_azimuth"],
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
            )
        )
        _start = time.perf_counter()
        with mo.status.spinner(
            f"{BACKEND.name.upper()}: one transport per tilt, then "
            f"{config.scoring_work:.3g} electron·direction·energy evaluations…"
        ):
            maps = calculate_maps(config)
        elapsed = time.perf_counter() - _start
    except ValueError as error:
        mo.stop(True, mo.callout(mo.md(str(error)), kind="danger"))
    return config, elapsed, maps


@app.cell
def _(maps, mo):
    current = mo.ui.number(0, 1e9, value=maps[0].current_na, label="Mean current (nA)")
    color = mo.ui.checkbox(value=True, label="Log color")
    selection = mo.ui.multiselect(
        options={f"{r.case['tilt_deg']:g}°": i for i, r in enumerate(maps)},
        value=[f"{r.case['tilt_deg']:g}°" for r in maps],
        label="Tilts shown",
    )
    mo.vstack(
        [
            mo.md("**Display** · redraws cached maps, no recompute"),
            mo.hstack([current, color, selection], justify="start", gap=2, align="end", wrap=True),
        ]
    )
    return color, current, selection


@app.cell
def _(
    BACKEND,
    color,
    config,
    context_rail,
    current,
    elapsed,
    heatmap_figure,
    maps,
    mo,
    selection,
):
    mo.stop(not selection.value, mo.md("Select at least one target tilt."))
    _grid = maps[0].energy_eV
    _ceiling = max(r.provenance["line_window_ceiling_eV"] for r in maps)
    _footprint = (
        "infinite" if config.width_mm is None else f"{config.width_mm:g} × {config.height_mm:g} mm"
    )
    mo.vstack(
        [
            context_rail(
                mo,
                {
                    "Backend": BACKEND.name,
                    "Elapsed": "cached" if elapsed < 0.05 else f"{elapsed:.1f} s",
                    "Footprint": _footprint,
                    "Photon grid": f"{_grid[0]:g}–{_grid[-1]:g} eV × {len(_grid)}",
                    "Line ceiling": f"{_ceiling:g} eV",
                    "Histories/tilt": config.electrons,
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
            if _grid[-1] < _ceiling
            else mo.md(""),
            heatmap_figure(
                tuple(maps[i] for i in selection.value),
                current_na=current.value,
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
    takes the lowest color. The mean current
    defaults to the beam's {maps[0].current_na:g} nA (1 pC × 5 kHz) and only rescales
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
    - **Photon window:** the automatic stop is the closed-form bound
      ħcβ|g|max/(1 − β), capped at the beam energy. A manual stop below it can
      truncate lines Doppler-shifted toward the beam direction.
    - **Scope:** incoherent PXR/CBS only, with PXR–CBS interference retained.
      Bremsstrahlung, characteristic, coherent, channeling and TR emission are excluded.
      Existing kinematic PXR/CBS model limits apply; reflections are truncated to the
      *Max reflections* strongest.
    - **Execution:** the array backend is fixed at import. Launch with
      `PYRITE_MC_BACKEND=cuda pyrite app angular-heatmap launch` for a local GPU.
      Use the `pyrite.directional_source` API for scripted or remote studies.
    """)
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
