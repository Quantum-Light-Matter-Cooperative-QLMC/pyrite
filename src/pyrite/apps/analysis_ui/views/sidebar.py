"""Analysis-app sidebar: checkpoint pickers plus the shared slice every view reads."""

from pyrite.results import records, sweep_values

from ..controls import SLICE_KEYS
from .common import optional_selector, thickness_selector

_SLICE_TEXT = {
    "energy": ("beam energy", "keV"),
    "tilt": ("polar tilt", "deg"),
    "azimuth": ("azimuth", "deg"),
}


def _slice_item(mo, name, widget, values):
    if name == "thickness":
        return thickness_selector(mo, widget, values)
    label, unit = _SLICE_TEXT[name]
    return optional_selector(mo, widget, values, lambda value: f"{label}: {value:g} {unit}")


def render_sidebar(mo, *, material_ui, checkpoint_ui, emission_ui, slice_controls, results):
    """Pickers on top, then the slice; single-valued dimensions render as text."""
    sweep = sweep_values(results) if records(results) else {}
    slice_items = [
        _slice_item(mo, name, slice_controls[name], sweep.get(key, []))
        for name, key in SLICE_KEYS.items()
    ]
    return mo.sidebar([material_ui, checkpoint_ui, emission_ui, mo.md("**Slice**"), *slice_items])
