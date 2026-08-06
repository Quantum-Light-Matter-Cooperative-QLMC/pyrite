from __future__ import annotations

from cxr_mc.plots.altair_spectra import compare_spectrum_chart
from cxr_mc.results import records, select_results, sweep_values

from ..controls import axes_panel
from .common import axis_warning_block, optional_selector, thickness_selector


def render_dimension_comparison(
    mo,
    *,
    context,
    controls,
    values,
    axes,
    spec,
):
    results = context.results
    sweep = sweep_values(results) if records(results) else {}
    varying_values = list(values["varying"] or [])

    energy_widget = optional_selector(
        mo,
        controls["energy"],
        sweep.get("E0_keV", []),
        lambda value: f"beam energy: {value:g} keV",
    )
    pinned_widget = optional_selector(
        mo,
        controls["pinned_angle"],
        sweep.get(spec.pinned_angle_key, []),
        lambda value: f"{spec.pinned_angle_label}: {value:g} deg",
    )
    thickness_widget = thickness_selector(
        mo,
        controls["thickness"],
        sweep.get("thickness_ang", []),
        label="thickness",
    )

    parts = [
        mo.md(spec.description),
        mo.hstack([energy_widget, pinned_widget, thickness_widget], wrap=True),
        controls["line"],
        controls["brem"],
        axes_panel(mo, controls["axes"]),
        controls["varying"],
    ]

    warning = axis_warning_block(mo, axes)
    if warning is not None:
        parts.append(warning)

    available = sweep.get(spec.varying_key, [])
    if len(available) <= 1:
        parts.append(mo.md(f"*Only one {spec.varying_label} in this checkpoint.*"))
    if not varying_values:
        parts.append(mo.md(f"*Select at least one {spec.varying_label} above to plot.*"))
        return mo.vstack(parts)

    constraints = {
        spec.varying_key: varying_values,
        "E0_keV": values["energy"],
        spec.pinned_angle_key: values["pinned_angle"],
        "thickness_ang": values["thickness"],
    }
    constraints = {key: value for key, value in constraints.items() if value is not None}
    selected = select_results(results, **constraints)

    narrow = compare_spectrum_chart(
        selected,
        context.settings,
        hue=spec.varying_key,
        include_brem=values["brem"],
        include_line=values["line"],
        include_coherent=context.show_both_emissions,
        x_domain=axes.narrow.x_domain,
        x_type=axes.narrow.x_type,
        y_type=axes.narrow.y_type,
        band="narrow",
    )
    broad = compare_spectrum_chart(
        selected,
        context.settings,
        hue=spec.varying_key,
        include_brem=values["brem"],
        include_line=values["line"],
        include_coherent=context.show_both_emissions,
        x_domain=axes.broad.x_domain,
        x_type=axes.broad.x_type,
        y_type=axes.broad.y_type,
        band="broad",
    )

    parts.extend(
        [
            mo.md("**Narrowband**"),
            narrow if narrow is not None else mo.md("*No narrowband spectra for this slice.*"),
            mo.md("**Broadband**"),
            broad if broad is not None else mo.md("*No broadband spectra for this slice.*"),
        ]
    )
    return mo.vstack(parts)
