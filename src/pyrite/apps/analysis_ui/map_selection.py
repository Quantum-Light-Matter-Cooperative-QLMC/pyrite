"""Session-local scope and persistence guard for the experimental Plotly map."""

from __future__ import annotations

from collections.abc import Mapping

from pyrite.plots.plotly.map_select import cell_from_points, selected_cell_frame


def map_scope(context, *, energy, thickness) -> tuple:
    """Bind selections to a checkpoint and a physical map slice."""
    return (
        context.selected_material,
        context.selected_face,
        context.checkpoint_stem,
        float(energy),
        float(thickness),
    )


def selection_for_scope(points, frame, scope):
    """Validate the frontend event and attach the current physical scope."""
    picked = cell_from_points(points, frame)
    return None if picked is None else {"scope": scope, **picked}


def restore_selection(frame, stored, scope):
    """Return the selected one-row frame, or empty on a scope/cell mismatch."""
    if not isinstance(stored, Mapping) or stored.get("scope") != scope:
        return frame.iloc[0:0]
    return selected_cell_frame(frame, stored)
