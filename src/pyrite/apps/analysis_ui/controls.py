from collections.abc import Sequence

from pyrite._formatting import fmt_thickness
from pyrite.plots._frames import _HEATMAP_QUANTITIES
from pyrite.results import records, sweep_values, thicknesses_by_energy

# Map-view heatmap quantities: the standard sweep metrics plus the electron
# footprint-hit fraction (a geometry diagnostic pinned to a [0, 1] colour scale).
MAP_QUANTITIES = (
    *_HEATMAP_QUANTITIES,
    ("hit_frac", "electron footprint-hit fraction", "magma"),
)


def _options(values: Sequence[float]) -> dict[str, float | None]:
    return {f"{value:g}": value for value in values} or {"— no data —": None}


def _thickness_options(results) -> dict[str, float | None]:
    values = sweep_values(results).get("thickness_ang", []) if records(results) else []
    return {fmt_thickness(value): value for value in values} or {"— no data —": None}


def _spread_default(values: Sequence[float], count: int = 4) -> list[float]:
    if len(values) <= count:
        return list(values)
    last = len(values) - 1
    indexes = sorted({round(last * index / (count - 1)) for index in range(count)})
    return [values[index] for index in indexes]


def make_axis_controls(
    mo,
    *,
    prefix: str,
    xlog: bool,
    ylog: bool,
):
    return mo.ui.dictionary(
        {
            "xlog": mo.ui.switch(value=xlog, label=f"{prefix} log x"),
            "ylog": mo.ui.switch(value=ylog, label=f"{prefix} log y"),
        }
    )


def make_spectrum_axes(
    mo,
    *,
    narrow_xlog: bool = False,
    narrow_ylog: bool = False,
    broad_xlog: bool = False,
    broad_ylog: bool = True,
):
    return mo.ui.dictionary(
        {
            "narrow": make_axis_controls(
                mo,
                prefix="Narrow",
                xlog=narrow_xlog,
                ylog=narrow_ylog,
            ),
            "broad": make_axis_controls(
                mo,
                prefix="Broad",
                xlog=broad_xlog,
                ylog=broad_ylog,
            ),
        }
    )


def axis_row(mo, axis_controls):
    return mo.hstack([axis_controls["xlog"], axis_controls["ylog"]], gap=1, widths=[1, 1])


def axes_panel(mo, axes):
    return mo.hstack(
        [
            axis_row(mo, axes["narrow"]),
            axis_row(mo, axes["broad"]),
        ],
        gap=2,
        align="center",
        widths=[1, 1],
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
    energy_options = _options(values.get("E0_keV", []))
    tilt_options = _options(values.get("tilt_deg", []))
    azimuth_options = _options(values.get("tilt_azim_deg", []))
    thickness_options = _thickness_options(results)
    inventory = thicknesses_by_energy(results) if source else {}
    default_thickness = (
        fmt_thickness(inventory[min(inventory)][-1]) if inventory else list(thickness_options)[-1]
    )
    return {
        "energy": mo.ui.dropdown(
            energy_options, value=next(iter(energy_options)), label="Beam energy (keV)"
        ),
        "tilt": mo.ui.dropdown(tilt_options, value=next(iter(tilt_options)), label="Polar tilt (deg)"),
        "azimuth": mo.ui.dropdown(
            azimuth_options, value=next(iter(azimuth_options)), label="Azimuth (deg)"
        ),
        "thickness": mo.ui.dropdown(
            thickness_options, value=default_thickness, label="Crystal thickness"
        ),
    }


VARY_OPTIONS = {"Beam energy": "E0_keV", "Polar tilt": "tilt_deg", "Azimuth": "tilt_azim_deg"}


def make_vary_control(mo):
    """Spectra-view dimension that varies across curves (dependency-free, so it persists)."""
    return mo.ui.radio(options=VARY_OPTIONS, value="Beam energy", label="Vary", inline=True)


def make_component_controls(mo):
    """Spectrum components selected across views."""
    return mo.ui.multiselect(
        options=["Line", "Bremsstrahlung", "Characteristic"],
        value=["Line", "Bremsstrahlung"],
        label="Components",
    )


def make_varying_control(mo, results, *, varying_key: str, unit: str):
    """Multiselect of the varying dimension's values, four spread defaults."""
    values = sweep_values(results).get(varying_key, []) if records(results) else []
    options = _options(values)
    defaults = [label for label, value in options.items() if value in _spread_default(values)]
    return mo.ui.multiselect(options, value=defaults, label=f"Values to compare ({unit})")


def make_spectra_axes(mo):
    """Scale toggles for the Spectra view."""
    return make_spectrum_axes(mo)


def make_detector_axes(mo):
    """Scale toggles for the Detectors view."""
    return make_spectrum_axes(
        mo,
        narrow_ylog=True,
        broad_xlog=True,
        broad_ylog=True,
    )


def make_map_quantity_control(mo):
    """Map-view heatmap quantity (dependency-free, so it persists)."""
    options = {label: key for key, label, _color_map in MAP_QUANTITIES}
    return mo.ui.dropdown(options, value=next(iter(options)), label="Quantity")


DETECTOR_OPTIONS = {"Eagle XO": "eaglexo", "Timepix3": "timepix"}


def make_detector_choice(mo):
    """Detectors-view instrument radio (dependency-free, so it persists)."""
    return mo.ui.radio(options=DETECTOR_OPTIONS, value="Eagle XO", label="Detector", inline=True)


def make_case_axes(mo):
    return mo.ui.dictionary(
        {
            "components": make_component_controls(mo),
            "axes": make_spectrum_axes(mo),
        }
    )
