"""Native desktop controls over the same selection/filter/inspection owner."""

import json
import uuid
from pathlib import Path


def show(viewer, *, output_dir, interactive=True):
    plotter = viewer.plotter
    controls = dict(energy_min=0.0, generation_max=None, time_max=None, clip_x=None)

    def change(name, value):
        controls[name] = value
        viewer.apply_filters(**controls)

    if "E_start_keV" in viewer.data:
        plotter.add_slider_widget(
            lambda v: change("energy_min", v),
            (0, float(viewer.data["E_start_keV"].max())),
            value=0,
            title="Minimum energy (keV)",
            pointa=(0.02, 0.92),
            pointb=(0.30, 0.92),
        )
    generation = int(viewer.data["generation"].max())
    if generation > 0:
        plotter.add_slider_widget(
            lambda v: change("generation_max", int(v)),
            (0, generation),
            value=generation,
            title="Generation max",
            pointa=(0.35, 0.92),
            pointb=(0.62, 0.92),
        )
    if "time_fs" in viewer.mesh.cell_data:
        maximum = float(viewer.mesh.cell_data["time_fs"].max())
        plotter.add_slider_widget(
            lambda v: change("time_max", v),
            (0, maximum),
            value=maximum,
            title="Time max (fs)",
            pointa=(0.68, 0.92),
            pointb=(0.95, 0.92),
        )

    selected = None

    def pick(picker):
        nonlocal selected
        selected = viewer.inspect_cell(picker.GetCellId())
        viewer.highlight(selected)
        if selected:
            print(json.dumps(selected, indent=2), flush=True)

    def parent():
        nonlocal selected
        selected = viewer.select_parent(selected) if selected else None
        viewer.highlight(selected)
        print(json.dumps(selected, indent=2), flush=True)

    # Server and desktop use the same captured cell IDs; no nearest-track guess.
    plotter.enable_surface_point_picking(
        callback=lambda _point, picker: pick(picker),
        use_picker=True,
        picker="cell",
        show_point=False,
        show_message=True,
    )
    plotter.add_key_event(
        "s", lambda: viewer.screenshot(Path(output_dir) / f"capture-{uuid.uuid4().hex[:12]}.png")
    )
    plotter.add_key_event(
        "m", lambda: viewer.movie(Path(output_dir) / f"capture-{uuid.uuid4().hex[:12]}.gif")
    )
    plotter.add_key_event("i", lambda: viewer.show_scale("instrument"))
    plotter.add_key_event("c", lambda: viewer.show_scale("closeup"))
    plotter.add_key_event("p", parent)
    plotter.add_key_event(
        "x", lambda: change("clip_x", 0.0 if controls["clip_x"] is None else None)
    )
    plotter.show(interactive=interactive, auto_close=interactive)
