"""Reactive image-streaming interface; no geometry is sent to the browser."""

import json
import uuid
from pathlib import Path


def build_server(viewer, *, output_dir="trajectory-exports"):
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import html, vtk, vuetify3
    from vtkmodules.vtkRenderingCore import vtkCellPicker

    server = get_server(f"pyrite-trajectories-{uuid.uuid4().hex}", client_type="vue3")
    assert server is not None
    state, ctrl = server.state, server.controller
    data = viewer.data
    energy_max = float(data["E_start_keV"].max()) if "E_start_keV" in data else 0.0
    generation_max = max(0, int(data["generation"].max()))
    time_max = (
        float(viewer.mesh.cell_data["time_fs"].max()) if "time_fs" in viewer.mesh.cell_data else 0.0
    )
    x_low, x_high = viewer.x_range()
    state.update(
        dict(
            energy_min=0.0,
            generation_max=generation_max,
            time_max=time_max,
            scale="closeup",
            # A literal list attribute would be read as a (state, default) pair.
            scale_options=["closeup", "instrument"],
            clipping=False,
            clip_x=0.5 * (x_low + x_high),
            clip_min=x_low,
            clip_max=x_high,
            clip_units="angstrom",
            picked="",
            pick_status="Click a track to inspect it.",
            selected_cells=viewer.visible.n_cells,
            full_segments=viewer.plan.segments,
            message="",
        )
    )
    selected = None
    camera_at_press = None

    def notify():
        state.selected_cells = viewer.visible.n_cells
        ctrl.view_update()

    def show(info, status):
        nonlocal selected
        selected = info
        state.picked = json.dumps(info, indent=2) if info else ""
        state.pick_status = status
        viewer.highlight(info)

    @state.change("energy_min", "generation_max", "time_max", "clipping", "clip_x", "scale")
    def update(
        energy_min=0.0,
        generation_max=None,
        time_max=None,
        clipping=False,
        clip_x=0.0,
        scale="closeup",
        **_,
    ):
        if viewer.scale != scale:
            viewer.show_scale(scale)
            low, high = viewer.x_range()
            # The plane is in view units; never carry a value across scales.
            state.update(
                dict(
                    clip_min=low,
                    clip_max=high,
                    clip_x=0.5 * (low + high),
                    clip_x_ui=0.5 * (low + high),
                    clip_units="angstrom" if scale == "closeup" else "mm",
                )
            )
            clip_x = state.clip_x
        try:
            controls = dict(
                energy_min=float(energy_min),
                generation_max=int(generation_max)
                if generation_max is not None and int(generation_max) >= 0
                else None,
                time_max=float(time_max)
                if time_max is not None and "time_fs" in viewer.mesh.cell_data
                else None,
                clip_x=float(clip_x) if clipping else None,
            )
        except TypeError, ValueError:
            state.message = "Enter a numeric control value."
            return
        viewer.apply_filters(**controls)
        state.message = ""
        if selected:
            show(viewer.inspect_segment(selected["segment_id"]), state.pick_status)
        notify()

    def press(*_):
        nonlocal camera_at_press
        camera_at_press = _camera(viewer)

    def pick(event):
        # The browser sends a click after every camera drag; only a click
        # that left the camera unchanged is a pick.
        if camera_at_press is not None and _camera(viewer) != camera_at_press:
            return
        if not isinstance(event, dict) or viewer.actor is None:
            return
        position = event.get("position", {})
        if not isinstance(position, dict) or "x" not in position or "y" not in position:
            return
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.PickFromListOn()
        picker.AddPickList(viewer.actor)
        picker.Pick(float(position["x"]), float(position["y"]), 0, viewer.plotter.renderer)
        info = viewer.inspect_cell(picker.GetCellId())
        if info is None:
            state.pick_status = "No displayed track at that pixel; selection unchanged."
            return
        show(info, f"Picked segment {info['segment_id']}.")
        ctrl.view_update()

    def parent():
        if selected is None:
            state.pick_status = "Pick a track first."
        elif selected["parent_id"] < 0:
            state.pick_status = (
                "Track ancestry unavailable in this capture."
                if selected["track_id"] < 0
                else "Primary track: no parent."
            )
        elif not selected["parent_available"]:
            state.pick_status = (
                f"Parent track {selected['parent_id']} is outside the loaded selection; "
                "include its history/track when launching."
            )
        else:
            show(viewer.select_parent(selected), f"Parent track {selected['parent_id']}.")
            ctrl.view_update()

    def clear():
        show(None, "Selection cleared.")
        ctrl.view_update()

    def reset():
        viewer.reset_view()
        ctrl.view_update()

    def export(movie):
        suffix = "gif" if movie else "png"
        output = Path(output_dir) / f"capture-{uuid.uuid4().hex[:12]}.{suffix}"
        try:
            # Match the browser's current framing rather than a fixed size.
            if movie:
                viewer.movie(output, window_size=None)
            else:
                viewer.screenshot(output, window_size=None)
            payload = output.read_bytes()
        except (OSError, ValueError) as error:
            state.message = f"Export failed: {error}"
            return None
        finally:
            ctrl.view_update()
        state.message = (
            f"Export ready: {output.name} ({len(payload) / 1024:.0f} KiB); "
            f"server copy and provenance: {output}, {output.name}.json"
        )
        return payload

    ctrl.pick_track = pick
    ctrl.inspect_parent = parent
    ctrl.clear_selection = clear
    ctrl.reset_view = reset
    ctrl.trigger("export_png")(lambda: export(False))
    ctrl.trigger("export_gif")(lambda: export(True))

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Saved trajectories")
        with layout.toolbar:
            vuetify3.VSelect(
                v_model=("scale", "closeup"),
                items=("scale_options",),
                label="Scale",
                density="compact",
                hide_details=True,
                style="max-width:170px",
            )
            vuetify3.VBtn("Reset camera", click=ctrl.reset_view)
            vuetify3.VBtn(
                "Screenshot",
                click="trigger('export_png').then(data => { if (data) utils.download('pyrite-trajectories.png', data, 'image/png'); })",
            )
            vuetify3.VBtn(
                "Orbit GIF (30 frames)",
                click="trigger('export_gif').then(data => { if (data) utils.download('pyrite-trajectories-orbit.gif', data, 'image/gif'); })",
            )
        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="fill-height pa-0"):
                with vuetify3.VRow(classes="fill-height", no_gutters=True):
                    with vuetify3.VCol(
                        cols=3, classes="pa-3", style="overflow:auto; max-height:85vh"
                    ):
                        html.Div(
                            "Loaded {{ full_segments }} segments; {{ selected_cells }} displayed cells",
                            id="selection-count",
                        )
                        html.Div(
                            "Minimum energy: {{ Number(energy_min_ui).toFixed(2) }} keV",
                            id="energy-value",
                        )
                        vuetify3.VSlider(
                            v_model=("energy_min_ui", 0.0),
                            # Commit on release/key: dragging a large scene
                            # would otherwise refilter at every step.
                            end="energy_min = $event",
                            keyup="energy_min = energy_min_ui",
                            min=0,
                            max=energy_max,
                            step=max(energy_max / 100, 0.01),
                            disabled=energy_max == 0,
                            hide_details=True,
                        )
                        html.Div(
                            "Maximum generation: {{ generation_max_ui }}", id="generation-value"
                        )
                        vuetify3.VSlider(
                            v_model=("generation_max_ui", generation_max),
                            end="generation_max = $event",
                            keyup="generation_max = generation_max_ui",
                            min=0,
                            max=generation_max,
                            step=1,
                            show_ticks="always",
                            disabled=generation_max == 0,
                            hide_details=True,
                        )
                        html.Div(
                            "Maximum time: {{ Number(time_max_ui).toPrecision(3) }} fs",
                            id="time-value",
                        )
                        vuetify3.VSlider(
                            v_model=("time_max_ui", time_max),
                            end="time_max = $event",
                            keyup="time_max = time_max_ui",
                            min=0,
                            max=time_max,
                            step=max(time_max / 100, 0.001),
                            disabled=time_max == 0,
                            hide_details=True,
                        )
                        vuetify3.VSwitch(
                            v_model=("clipping", False),
                            label="Clip: keep lab x >= plane",
                            hide_details=True,
                        )
                        html.Div(
                            "Plane x: {{ Number(clip_x_ui).toPrecision(4) }} {{ clip_units }}",
                            id="clip-value",
                        )
                        vuetify3.VSlider(
                            v_model=("clip_x_ui", state.clip_x),
                            end="clip_x = $event",
                            keyup="clip_x = clip_x_ui",
                            min=("clip_min",),
                            max=("clip_max",),
                            step=("(clip_max - clip_min) / 200",),
                            disabled=("!clipping",),
                            hide_details=True,
                        )
                        html.Div("{{ pick_status }}", id="pick-status", classes="mt-2")
                        with html.Div(classes="d-flex ga-2 my-2"):
                            vuetify3.VBtn("Inspect parent", click=ctrl.inspect_parent, size="small")
                            vuetify3.VBtn("Clear", click=ctrl.clear_selection, size="small")
                        html.Pre(
                            "{{ picked }}",
                            id="picked-info",
                            style="white-space:pre-wrap; font-size:12px",
                        )
                        html.Div("{{ message }}", id="export-status")
                    with vuetify3.VCol(cols=9):
                        view = vtk.VtkRemoteView(
                            viewer.plotter.render_window,
                            ref="trajectory_view",
                            interactive_ratio=1,
                            interactive_quality=80,
                            picking_modes=("picking_modes", ["click"]),
                            click=(pick, "[$event]"),
                            interactor_events=("events", ["StartAnimation"]),
                            StartAnimation=press,
                        )
                        ctrl.view_update = view.update
    ctrl.on_server_ready.add(lambda **_: viewer.plotter.render())
    return server


def _camera(viewer):
    return [list(map(float, row)) for row in viewer.plotter.camera_position]
