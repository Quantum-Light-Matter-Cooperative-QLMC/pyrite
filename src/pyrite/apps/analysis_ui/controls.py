import math
from collections.abc import Sequence

from pyrite._formatting import fmt_thickness
from pyrite.results import records, sweep_values, thicknesses_by_energy


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
    values = [
        float(energy)
        for record in source
        for energy in (record.get(key) if record.get(key) is not None else record.get("E_grid", ()))
        if math.isfinite(float(energy)) and float(energy) > 0
    ]
    return (min(values), max(values)) if values else (1.0, 1.0)


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


def make_energy_controls(mo, results):
    source = records(results)
    line_min, line_max = _grid_limits(source, "E_grid")
    brem_min, brem_max = _grid_limits(source, "E_grid_brem")
    tilts = sorted({record["case"]["tilt_deg"] for record in source})
    tilt_options = _options(tilts, "deg")
    thickness_options = _thickness_options(results)
    inventory = thicknesses_by_energy(results) if source else {}
    default_thickness = (
        fmt_thickness(inventory[min(inventory)][-1]) if inventory else list(thickness_options)[-1]
    )

    return mo.ui.dictionary(
        {
            "tilt": mo.ui.dropdown(
                tilt_options,
                value=next(iter(tilt_options)),
                label="polar tilt",
            ),
            "thickness": mo.ui.dropdown(
                thickness_options,
                value=default_thickness,
                label="crystal thickness",
            ),
            "line": mo.ui.checkbox(value=False, label="show line spectrum"),
            "brem": mo.ui.checkbox(value=True, label="show brem background"),
            "characteristic": mo.ui.checkbox(
                value=False,
                label="show characteristic radiation",
            ),
            "axes": make_spectrum_axes(
                mo,
                narrow_auto=not source,
                narrow_xmin=line_min,
                narrow_xmax=line_max,
                broad_auto=not source,
                broad_xmin=brem_min,
                broad_xmax=brem_max,
            ),
        }
    )


def make_dimension_controls(mo, results, *, varying_key: str):
    source = records(results)
    line_min, line_max = _grid_limits(source, "E_grid")
    brem_min, brem_max = _grid_limits(source, "E_grid_brem")
    values = sweep_values(results) if source else {}
    energies = values.get("E0_keV", [])
    thicknesses = values.get("thickness_ang", [])

    if varying_key == "tilt_deg":
        varying = values.get("tilt_deg", [])
        pinned = values.get("tilt_azim_deg", [])
        pinned_key = "pinned_angle"
        pinned_label = "azimuth"
        varying_label = "polar angles to compare"
    elif varying_key == "tilt_azim_deg":
        varying = values.get("tilt_azim_deg", [])
        pinned = values.get("tilt_deg", [])
        pinned_key = "pinned_angle"
        pinned_label = "polar tilt"
        varying_label = "azimuths to compare"
    else:
        raise ValueError(f"Unsupported varying dimension: {varying_key}")

    energy_options = _options(energies, "keV")
    pinned_options = _options(pinned, "deg")
    thickness_options = {fmt_thickness(value): value for value in thicknesses} or {
        "— no data —": None
    }
    varying_options = _options(varying, "deg")
    defaults = _spread_default(varying)

    return mo.ui.dictionary(
        {
            "energy": mo.ui.dropdown(
                energy_options,
                value=next(iter(energy_options)),
                label="beam energy",
            ),
            pinned_key: mo.ui.dropdown(
                pinned_options,
                value=next(iter(pinned_options)),
                label=pinned_label,
            ),
            "thickness": mo.ui.dropdown(
                thickness_options,
                value=list(thickness_options)[-1],
                label="thickness",
            ),
            "varying": mo.ui.multiselect(
                varying_options,
                value=[f"{value:g} deg" for value in defaults],
                label=varying_label,
            ),
            "line": mo.ui.checkbox(value=False, label="show line spectrum"),
            "brem": mo.ui.checkbox(value=True, label="show brem background"),
            "characteristic": mo.ui.checkbox(
                value=False,
                label="show characteristic radiation",
            ),
            "axes": make_spectrum_axes(
                mo,
                narrow_auto=not source,
                narrow_xmin=line_min,
                narrow_xmax=line_max,
                broad_auto=not source,
                broad_xmin=brem_min,
                broad_xmax=brem_max,
            ),
        }
    )


def make_scan_thickness_control(mo, results):
    options = _thickness_options(results)
    return mo.ui.dropdown(options, value=list(options)[-1], label="crystal thickness")


def make_heatmap_energy_control(mo, results):
    energies = sweep_values(results).get("E0_keV", []) if records(results) else []
    options = _options(energies, "keV")
    return mo.ui.dropdown(options, value=next(iter(options)), label="heatmap beam energy")


def make_detector_controls(mo, results):
    source = records(results)
    line_min, line_max = _grid_limits(source, "E_grid")
    brem_min, brem_max = _grid_limits(source, "E_grid_brem")
    values = sweep_values(results) if source else {}
    tilt_options = _options(values.get("tilt_deg", [])[::-1], "deg")
    azimuth_options = _options(values.get("tilt_azim_deg", []), "deg")
    thickness_options = _thickness_options(results)

    return mo.ui.dictionary(
        {
            "tilt": mo.ui.dropdown(
                tilt_options,
                value=next(iter(tilt_options)),
                label="polar tilt",
            ),
            "azimuth": mo.ui.dropdown(
                azimuth_options,
                value=next(iter(azimuth_options)),
                label="azimuth",
            ),
            "thickness": mo.ui.dropdown(
                thickness_options,
                value=list(thickness_options)[-1],
                label="crystal thickness",
            ),
            "axes": make_spectrum_axes(
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
            ),
        }
    )


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
