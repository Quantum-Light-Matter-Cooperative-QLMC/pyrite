"""Lazy PyVista adapter; independent close-up and instrument coordinate groups."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .._scene_geometry import case_rotation
from ..montecarlo.trajectory_scene import export_trajectory_scene

# Same c conversion as the existing live trajectory preview. Time is recorded
# path age plus recorded entrance offset, expressed in fs.
_C_ANG_PER_FS = 2997.924580


class CaptureViewer:
    """Keep full selected tracks separate from segment-level display filters."""

    def __init__(self, plan, data, *, off_screen=False):
        import pyvista as pv

        self.pv = pv
        self.plan, self.data = plan, data
        half = data["L_ang"][:, None] * data["v_hat"] * 0.5
        slab = np.stack((data["r_mid"] - half, data["r_mid"] + half), axis=1).reshape(-1, 3)
        rotation = case_rotation(plan.case)
        origin = np.asarray((plan.scene or {}).get("sample_origin_lab_mm", [0, 0, 0]))
        lines = np.column_stack(
            (np.full(plan.segments, 2), np.arange(2 * plan.segments).reshape(-1, 2))
        ).ravel()
        self.mesh = pv.PolyData(slab @ rotation.T + origin * 1e7, lines=lines)
        for name, values in data.items():
            if name not in {"r_mid", "v_hat"}:
                self.mesh.cell_data[name] = values
        if "t_start_ang" in data:
            self.mesh.cell_data["time_fs"] = (
                data["t_start_ang"] + data.get("t0_ang", 0)
            ) / _C_ANG_PER_FS
        self.groups: dict[str, pv.MultiBlock] = {}
        pad = max(1.0, float(np.max(np.ptp(slab, axis=0))) * 0.05)
        with tempfile.TemporaryDirectory(prefix="pyrite-viewer-geometry-") as directory:
            near, far = export_trajectory_scene(
                plan.path,
                Path(directory) / "geometry.vtp",
                include_tracks=False,
                extent_ang=(slab.min(axis=0) - pad, slab.max(axis=0) + pad),
            )
            for name, path in (("closeup", near), ("instrument", far)):
                blocks = pv.read(path)
                assert isinstance(blocks, pv.MultiBlock)
                self.groups[name] = blocks
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=[1000, 700])
        self.visible = self.mesh
        self.scale = "closeup"
        self.actor = None
        self.filters = {}
        self.highlighted = None
        self.show_scale("closeup")

    def show_scale(self, scale):
        if scale not in self.groups:
            raise ValueError("scale must be closeup or instrument")
        self.scale = scale
        self.plotter.clear()
        for name, block in zip(self.groups[scale].keys(), self.groups[scale], strict=True):
            if block is not None and block.n_points:
                self.plotter.add_mesh(
                    block, name=name, color="lightgray", opacity=0.22, pickable=False
                )
        units = "angstrom" if scale == "closeup" else "mm"
        status = "recorded" if self.plan.scene else "unavailable (legacy capture)"
        self.plotter.add_text(
            f"{scale}: lab axes, {units}\ndownstream geometry: {status}", font_size=10
        )
        self.actor = None
        self.apply_filters(**self.filters)
        self.reset_view()

    def reset_view(self):
        """Restore the isometric camera fitted to the current scale's scene."""
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()  # ty: ignore[missing-argument] -- PyVista decorator typing

    def x_range(self):
        """Lab-x extent of the full selection in the current view's units."""
        unit = 1e7 if self.scale == "instrument" else 1.0
        x = self.mesh.points[:, 0] / unit
        return float(x.min()), float(x.max())

    def highlight(self, info):
        """Outline a whole inspected track, independent of display filters."""
        self.highlighted = None if info is None else (info["track_id"], info["electron_id"])
        self._draw_highlight()
        self.plotter.render()

    def _draw_highlight(self):
        self.plotter.remove_actor("highlight", render=False)  # ty: ignore[invalid-argument-type] -- PyVista decorator typing
        if self.highlighted is None:
            return
        track, history = self.highlighted
        cells = self.mesh.cell_data
        mask = cells["track_id"] == track if track >= 0 else cells["electron_id"] == history
        if not np.any(mask):
            return
        shown = self.mesh.extract_cells(mask).extract_surface(algorithm="dataset_surface")
        if self.scale == "instrument":
            shown.points /= 1e7
        self.plotter.add_mesh(
            shown, name="highlight", color="magenta", line_width=5, pickable=False, render=False
        )

    def apply_filters(self, *, energy_min=0.0, generation_max=None, time_max=None, clip_x=None):
        self.filters = dict(
            energy_min=energy_min, generation_max=generation_max, time_max=time_max, clip_x=clip_x
        )
        keep = np.ones(self.plan.segments, dtype=bool)
        if "E_start_keV" in self.data:
            keep &= self.data["E_start_keV"] >= energy_min
        if generation_max is not None:
            keep &= self.data["generation"] <= generation_max
        if time_max is not None and "time_fs" in self.mesh.cell_data:
            keep &= self.mesh.cell_data["time_fs"] <= time_max
        self.visible = self.mesh.extract_cells(keep).extract_surface(algorithm="dataset_surface")
        if self.scale == "instrument":
            self.visible.points /= 1e7
        if clip_x is not None and self.visible.n_cells:
            self.visible = self.visible.clip(normal="x", origin=(clip_x, 0, 0), invert=False)
        if self.actor is not None:
            self.plotter.remove_actor(self.actor, render=False)  # ty: ignore[invalid-argument-type] -- PyVista decorator typing
            self.actor = None
        if self.visible.n_cells:
            scalar = "E_start_keV" if "E_start_keV" in self.data else None
            self.actor = self.plotter.add_mesh(
                self.visible,
                name="tracks",
                scalars=scalar,
                cmap="turbo",
                clim=(0, max(1.0, float(np.max(self.data[scalar])))) if scalar else None,
                line_width=2,
                scalar_bar_args={"title": "Energy (keV)"},
                render=False,
            )
        self._draw_highlight()
        self.plotter.render()
        return self.visible.n_cells

    def full_track(self, track, *, history=None):
        """Captured rows for a whole track, unaffected by display filters.

        A legacy history can be inspected as a history; it is never described
        as a known secondary track.
        """
        mask = self.data["track_id"] == track if track >= 0 else self.data["electron_id"] == history
        return {name: values[mask] for name, values in self.data.items()}

    def inspect_track(self, track, *, history=None):
        full = self.full_track(track, history=history)
        if not len(full["segment_id"]):
            return None
        cell = self.visible.cell_data
        shown = (
            (cell["track_id"] == track if track >= 0 else cell["electron_id"] == history)
            if self.visible.n_cells
            else np.zeros(0, dtype=bool)
        )
        visible_ids = (
            np.unique(cell["segment_id"][shown])
            if self.visible.n_cells
            else np.empty(0, dtype=np.int64)
        )
        clipped = len(visible_ids) != len(full["segment_id"])
        if not clipped and np.any(shown):
            selected = self.visible.extract_cells(shown).extract_surface(
                algorithm="dataset_surface"
            )
            # Plane clipping can shorten an internal cell without changing a
            # track's overall bounds. Compare every displayed cell length.
            lines = selected.lines.reshape(-1, 3)[:, 1:]
            lengths = np.linalg.norm(
                selected.points[lines[:, 1]] - selected.points[lines[:, 0]], axis=1
            )
            unit = 1e7 if self.scale == "instrument" else 1.0
            clipped = not np.allclose(
                lengths * unit, selected.cell_data["L_ang"], rtol=1e-6, atol=1e-7
            )
        result = {
            key: np.asarray(value[0]).tolist()
            for key, value in full.items()
            if key not in {"r_mid", "v_hat"}
        }
        result.update(
            full_segments=len(full["segment_id"]),
            visible_segments=len(visible_ids),
            clipped=bool(clipped),
            identity="track" if track >= 0 else "history (track ancestry unavailable)",
            parent_available=bool(np.any(self.data["track_id"] == result["parent_id"]))
            if result["parent_id"] >= 0
            else False,
        )
        return result

    def inspect_cell(self, cell_id):
        """Inspect a captured segment while retaining complete-track identity."""
        if cell_id < 0 or cell_id >= self.visible.n_cells:
            return None
        return self.inspect_segment(int(self.visible.cell_data["segment_id"][cell_id]))

    def inspect_segment(self, segment_id):
        """Captured segment fields plus its whole track's display status.

        Uses unfiltered rows, so a picked segment keeps its identity and
        values when later filters hide or clip it.
        """
        (rows,) = np.nonzero(self.mesh.cell_data["segment_id"] == segment_id)
        if not len(rows):
            return None
        cell = self.mesh.cell_data
        row = rows[0]
        info = self.inspect_track(int(cell["track_id"][row]), history=int(cell["electron_id"][row]))
        assert info is not None
        info.update(
            {
                key: np.asarray(value[row]).tolist()
                for key, value in cell.items()
                if not key.startswith("vtk")
            }
        )
        return info

    def select_parent(self, info):
        """Inspect the full selected parent even when display filters hide it."""
        parent = info.get("parent_id", -1)
        return self.inspect_track(parent) if parent >= 0 else None

    @contextmanager
    def _staged_output(self, output, *, overwrite=False, metadata=None):
        output = Path(output)
        sidecar = output.with_suffix(output.suffix + ".json")
        for target in (output, sidecar):
            if target.exists() and not overwrite:
                raise FileExistsError(f"output exists: {target}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pyrite-export-", dir=output.parent) as directory:
            staged = Path(directory) / output.name
            yield staged
            staged_meta = staged.with_suffix(staged.suffix + ".json")
            staged_meta.write_text(json.dumps(metadata or self.metadata(), indent=2) + "\n")
            os.replace(staged_meta, sidecar)
            os.replace(staged, output)

    def screenshot(self, output, *, overwrite=False, window_size=(1000, 700)):
        """Write a PNG; ``window_size=None`` keeps the current (browser) framing."""
        output = Path(output)
        if output.suffix.lower() != ".png":
            raise ValueError("screenshot output must end in .png")
        size = list(window_size or self.plotter.window_size)
        with self._staged_output(
            output, overwrite=overwrite, metadata=dict(self.metadata(), image_size=size)
        ) as staged:
            self.plotter.render()
            self.plotter.screenshot(staged, window_size=_size(window_size))
        return output

    def metadata(self):
        return dict(
            artifact=str(self.plan.path),
            case_sha256=self.plan.header["case_sha256"],
            scene_sha256=self.plan.header.get("scene_sha256"),
            parameter_sha256=self.plan.header.get("parameter_sha256"),
            provenance=self.plan.provenance,
            sample_to_lab_R=case_rotation(self.plan.case).tolist(),
            sample_origin_lab_mm=(self.plan.scene or {}).get("sample_origin_lab_mm", [0, 0, 0]),
            fields=self.plan.fields,
            stored_row_ranges=self.plan.ranges,
            scale=self.scale,
            units="angstrom" if self.scale == "closeup" else "mm",
            frame="lab",
            full_segments=self.plan.segments,
            visible_segments=self.visible.n_cells,
            filters=self.filters,
        )

    def movie(self, output, *, frames=30, fps=10, overwrite=False, window_size=(1000, 700)):
        """Bounded camera orbit GIF; includes current segment filters and scale."""
        import imageio.v2 as imageio

        output = Path(output)
        if output.suffix.lower() != ".gif" or not 2 <= frames <= 120 or not 1 <= fps <= 30:
            raise ValueError("movie requires .gif, 2..120 frames and 1..30 fps")
        camera = self.plotter.camera_position
        try:
            with self._staged_output(
                output,
                overwrite=overwrite,
                metadata=dict(
                    self.metadata(),
                    frames=frames,
                    fps=fps,
                    image_size=list(window_size or self.plotter.window_size),
                ),
            ) as staged:
                with imageio.get_writer(staged, mode="I", duration=1000 / fps, loop=0) as writer:
                    for _ in range(frames):
                        self.plotter.camera.azimuth += 360 / frames
                        self.plotter.render()
                        writer.append_data(self.plotter.screenshot(window_size=_size(window_size)))  # ty: ignore[unresolved-attribute] -- imageio writer protocol
        finally:
            self.plotter.camera_position = camera
            self.plotter.render()
        return output

    def close(self):
        self.plotter.close()


def _size(window_size):
    return None if window_size is None else list(window_size)
