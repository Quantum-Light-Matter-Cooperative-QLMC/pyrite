from collections.abc import Mapping

from pyrite.plots._decimation import DEFAULT_PLOT_RESOLUTION

from .models import AxisPair, AxisSpec


def resolve_axis_spec(values: Mapping[str, object]) -> AxisSpec:
    """Resolve scale toggles; chart interaction owns the visible domains."""
    return AxisSpec(
        x_type="log" if values.get("xlog") else "linear",
        y_type="log" if values.get("ylog") else "linear",
    )


def resolve_axis_pair(control_values: Mapping[str, object]) -> AxisPair:
    return AxisPair(
        narrow=resolve_axis_spec(control_values["narrow"]),
        broad=resolve_axis_spec(control_values["broad"]),
        resolution=str(control_values.get("resolution") or DEFAULT_PLOT_RESOLUTION),
    )
