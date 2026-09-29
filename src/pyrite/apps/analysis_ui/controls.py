from collections.abc import Sequence

import numpy as np

from pyrite._formatting import fmt_thickness
from pyrite.plots._frames import _HEATMAP_QUANTITIES
from pyrite.results import records, sweep_values, thicknesses_by_energy

# Map-view heatmap quantities: the standard sweep metrics plus the electron
# footprint-hit fraction (a geometry diagnostic pinned to a [0, 1] colour scale).
MAP_QUANTITIES = (
    *_HEATMAP_QUANTITIES,
    ("hit_frac", "electron footprint-hit fraction", "magma"),
)


def _options(values: Sequence[float], unit: str = "") -> dict[str, float | None]:
    suffix = f" {unit}" if unit else ""
    return {f"{value:g}{suffix}": value for value in values} or {"— no data —": None}


def _thickness_options(results) -> dict[str, float | None]:
    values = sweep_values(results).get("thickness_ang", []) if records(results) else []
    return {fmt_thickness(value): value for value in values} or {"— no data —": None}


def _spread_default(values: Sequence[float], count: int = 4) -> list[float]:
    if len(values) <= count:
        return list(values)
    last = len(values) - 1
    indexes = sorted({round(last * index / (count - 1)) for index in range(count)})
    return [values[index] for index in indexes]


def _grid_limits(source, key: str) -> tuple[float, float]:
    """Positive lower and full upper endpoint of the selected photon grid."""
    lows, highs = [], []
    for record in source:
        grid = record.get(key) if record.get(key) is not None else record.get("E_grid", ())
        energies = np.asarray(grid, dtype=float).ravel()
        energies = energies[np.isfinite(energies) & (energies > 0)]
        if energies.size:
            lows.append(energies.min())
            highs.append(energies.max())
    return (float(min(lows)), float(max(highs))) if lows else (1.0, 1.0)


def make_axis_controls(
    mo,
    *,
    prefix: str,
    auto: bool,
    xmin: float,
    xmax: float,
    xlog: bool,
    ylog: bool,
    include_y_domain: bool = False,
    ymin: float = 0.0,
    ymax: float = 0.0,
):
    elements = {
        "auto": mo.ui.switch(value=auto, label=f"Auto {prefix} domain"),
        "xmin": mo.ui.text(value=f"{xmin:g}", label=f"{prefix} x-min (eV)"),
        "xmax": mo.ui.text(value=f"{xmax:g}", label=f"{prefix} x-max (eV)"),
        "xlog": mo.ui.switch(value=xlog, label=f"{prefix} log x"),
        "ylog": mo.ui.switch(value=ylog, label=f"{prefix} log y"),
    }
    if include_y_domain:
        elements["ymin"] = mo.ui.text(value=f"{ymin:g}", label=f"{prefix} y-min")
        elements["ymax"] = mo.ui.text(value=f"{ymax:g}", label=f"{prefix} y-max")
    return mo.ui.dictionary(elements)


def make_spectrum_axes(
    mo,
    *,
    narrow_auto: bool,
    narrow_xmax: float,
    narrow_xmin: float = 0.0,
    narrow_xlog: bool = False,
    narrow_ylog: bool = False,
    broad_auto: bool,
    broad_xmax: float,
    broad_xmin: float = 0.0,
    broad_xlog: bool = False,
    broad_ylog: bool = True,
    include_y_domain: bool = False,
):
    return mo.ui.dictionary(
        {
            "narrow": make_axis_controls(
                mo,
                prefix="narrow",
                auto=narrow_auto,
                xmin=narrow_xmin,
                xmax=narrow_xmax,
                xlog=narrow_xlog,
                ylog=narrow_ylog,
                include_y_domain=include_y_domain,
            ),
            "broad": make_axis_controls(
                mo,
                prefix="broad",
                auto=broad_auto,
                xmin=broad_xmin,
                xmax=broad_xmax,
                xlog=broad_xlog,
                ylog=broad_ylog,
                include_y_domain=include_y_domain,
            ),
        }
    )


def axis_row(mo, axis_controls, *, include_y_domain: bool = False):
    items = [
        axis_controls["auto"],
        axis_controls["xmin"],
        axis_controls["xmax"],
    ]
    if include_y_domain:
        items.extend([axis_controls["ymin"], axis_controls["ymax"]])
    items.extend([axis_controls["xlog"], axis_controls["ylog"]])
    return mo.hstack(items, wrap=True)


def axes_panel(mo, axes, *, include_y_domain: bool = False):
    return mo.accordion(
        {
            "Axes and scaling": mo.vstack(
                [
                    axis_row(mo, axes["narrow"], include_y_domain=include_y_domain),
                    axis_row(mo, axes["broad"], include_y_domain=include_y_domain),
                ]
            )
        }
    )


SLICE_KEYS = {
    "energy": "E0_keV",
    "tilt": "tilt_deg",
    "azimuth": "tilt_azim_deg",
    "thickness": "thickness_ang",
}


def make_slice_controls(mo, results):
    """The shared sidebar slice: beam energy, polar tilt, azimuth, and thickness.

    Returns a plain ``{name: dropdown}`` mapping (keys as in :data:`SLICE_KEYS`)
    so the app can bind each dropdown to its own name and a view reruns only
    when a dimension it reads changes. Every view pins the dimensions it does
    not plot. The default thickness is the thickest slab computed at the lowest
    beam energy, so the default slice exists at every energy the watchdog kept.
    """
    source = records(results)
    values = sweep_values(results) if source else {}
    energy_options = _options(values.get("E0_keV", []), "keV")
    tilt_options = _options(values.get("tilt_deg", []), "deg")
    azimuth_options = _options(values.get("tilt_azim_deg", []), "deg")
    thickness_options = _thickness_options(results)
    inventory = thicknesses_by_energy(results) if source else {}
    default_thickness = (
        fmt_thickness(inventory[min(inventory)][-1]) if inventory else list(thickness_options)[-1]
    )
    return {
        "energy": mo.ui.dropdown(
            energy_options, value=next(iter(energy_options)), label="beam energy"
        ),
        "tilt": mo.ui.dropdown(tilt_options, value=next(iter(tilt_options)), label="polar tilt"),
        "azimuth": mo.ui.dropdown(
            azimuth_options, value=next(iter(azimuth_options)), label="azimuth"
        ),
        "thickness": mo.ui.dropdown(
            thickness_options, value=default_thickness, label="crystal thickness"
        ),
    }


VARY_OPTIONS = {"beam energy": "E0_keV", "polar tilt": "tilt_deg", "azimuth": "tilt_azim_deg"}


def make_vary_control(mo):
    """Spectra-view dimension that varies across curves (dependency-free, so it persists)."""
    return mo.ui.radio(options=VARY_OPTIONS, value="beam energy", label="Vary", inline=True)


def make_component_controls(mo):
    """Spectrum component toggles (dependency-free, so they persist across views)."""
    return mo.ui.dictionary(
        {
            "line": mo.ui.checkbox(value=False, label="show line spectrum"),
            "brem": mo.ui.checkbox(value=True, label="show brem background"),
            "characteristic": mo.ui.checkbox(
                value=False,
                label="show characteristic radiation",
            ),
        }
    )


def make_varying_control(mo, results, *, varying_key: str, unit: str):
    """Multiselect of the varying dimension's values, four spread defaults."""
    values = sweep_values(results).get(varying_key, []) if records(results) else []
    options = _options(values, unit)
    defaults = [label for label, value in options.items() if value in _spread_default(values)]
    return mo.ui.multiselect(options, value=defaults, label="values to compare")


def make_spectra_axes(mo, results):
    """Axis controls for the Spectra view (narrow line grid + broad brem grid)."""
    source = records(results)
    line_min, line_max = _grid_limits(source, "E_grid")
    brem_min, brem_max = _grid_limits(source, "E_grid_brem")
    return make_spectrum_axes(
        mo,
        narrow_auto=not source,
        narrow_xmin=line_min,
        narrow_xmax=line_max,
        broad_auto=not source,
        broad_xmin=brem_min,
        broad_xmax=brem_max,
    )


def make_detector_axes(mo, results):
    """Axis controls for the Detectors view (log scales plus manual y domains)."""
    source = records(results)
    line_min, line_max = _grid_limits(source, "E_grid")
    brem_min, brem_max = _grid_limits(source, "E_grid_brem")
    return make_spectrum_axes(
        mo,
        narrow_auto=not source,
        narrow_xmin=line_min,
        narrow_xmax=line_max,
        narrow_ylog=True,
        broad_auto=not source,
        broad_xmin=brem_min,
        broad_xmax=brem_max,
        broad_xlog=True,
        broad_ylog=True,
        include_y_domain=True,
    )


def make_map_quantity_control(mo):
    """Map-view heatmap quantity (dependency-free, so it persists)."""
    options = {label: key for key, label, _color_map in MAP_QUANTITIES}
    return mo.ui.dropdown(options, value=next(iter(options)), label="quantity")


DETECTOR_OPTIONS = {"Eagle XO": "eaglexo", "Timepix3": "timepix"}


def make_detector_choice(mo):
    """Detectors-view instrument radio (dependency-free, so it persists)."""
    return mo.ui.radio(options=DETECTOR_OPTIONS, value="Eagle XO", label="Detector", inline=True)


def make_case_axes(mo, basket=()):
    line_min, line_max = _grid_limits(basket, "E_grid")
    brem_min, brem_max = _grid_limits(basket, "E_grid_brem")
    return mo.ui.dictionary(
        {
            "line": mo.ui.checkbox(value=False, label="show line spectrum"),
            "brem": mo.ui.checkbox(value=True, label="show brem background"),
            "characteristic": mo.ui.checkbox(
                value=False,
                label="show characteristic radiation",
            ),
            "axes": make_spectrum_axes(
                mo,
                narrow_auto=not basket,
                narrow_xmin=line_min,
                narrow_xmax=line_max,
                broad_auto=not basket,
                broad_xmin=brem_min,
                broad_xmax=brem_max,
            ),
        }
    )
