"""Spectra view: one dimension varies across curves, the sidebar slice pins the rest."""

from pyrite.apps._design import static_altair_chart
from pyrite.plots.altair.spectra import compare_spectrum_chart
from pyrite.results import records, select_results, sweep_values

from ..controls import axes_panel
from ..data import slice_results
from ..models import DimensionComparisonSpec
from .common import themed_chart

SPECTRA_SPECS = {
    "E0_keV": DimensionComparisonSpec(
        varying_key="E0_keV",
        varying_label="beam energy",
        varying_plural="beam energies",
        unit="keV",
        pinned=("tilt", "azimuth"),
        description=(
            "Beam energy varies across curves; polar tilt, azimuth, and thickness "
            "come from the sidebar."
        ),
    ),
    "tilt_deg": DimensionComparisonSpec(
        varying_key="tilt_deg",
        varying_label="polar tilt",
        varying_plural="polar tilts",
        unit="deg",
        pinned=("energy", "azimuth"),
        description=(
            "Polar tilt varies across curves; beam energy, azimuth, and thickness "
            "come from the sidebar."
        ),
    ),
    "tilt_azim_deg": DimensionComparisonSpec(
        varying_key="tilt_azim_deg",
        varying_label="azimuth",
        varying_plural="azimuths",
        unit="deg",
        pinned=("energy", "tilt"),
        description=(
            "Azimuth varies across curves; beam energy, polar tilt, and thickness "
            "come from the sidebar."
        ),
    ),
}


def spectra_selection(results, slice_values, spec, varying_values):
    """Records at the sidebar slice with ``spec``'s dimension free.

    Thickness goes through :func:`slice_results`, so an energy whose beam died
    before the pinned slab contributes its thickest computed slab (stamped
    ``thickness_fallback``) instead of disappearing.
    """
    pinned = {name: slice_values.get(name) for name in (*spec.pinned, "thickness")}
    selected = slice_results(results, **pinned)
    return select_results(selected, **{spec.varying_key: list(varying_values)})


def _fallback_note(mo, selected):
    fallbacks = sorted(
        {
            record["case"]["E0_keV"]
            for record in records(selected)
            if "thickness_fallback" in record["case"]
        }
    )
    if not fallbacks:
        return None
    return mo.md(
        "*Beam died before the pinned thickness for "
        + ", ".join(f"{energy:g}" for energy in fallbacks)
        + " keV — shown at the thickest computed slab (watchdog-saturated).*"
    )


def render_spectra(
    mo,
    *,
    context,
    slice_values,
    spec,
    vary_ui,
    varying_ui,
    components,
    axes_controls,
    axes,
    theme,
):
    """The Spectra view: vary radio, curve picker, components, two bands."""
    results = context.results
    component_values = set(components.value)
    parts = [
        mo.hstack([vary_ui, varying_ui], align="end", widths=[1, 1]),
        mo.md(spec.description),
        components,
        axes_panel(mo, axes_controls),
    ]

    available = sweep_values(results).get(spec.varying_key, []) if records(results) else []
    if len(available) <= 1:
        parts.append(mo.md(f"*Only one {spec.varying_label} in this checkpoint.*"))
    varying_values = list(varying_ui.value or [])
    if not varying_values:
        parts.append(mo.md(f"*Select at least one {spec.varying_label} above to plot.*"))
        return mo.vstack(parts)

    selected = spectra_selection(results, slice_values, spec, varying_values)
    note = _fallback_note(mo, selected)
    if note is not None:
        parts.append(note)

    for band, heading in (("narrow", "Narrowband"), ("broad", "Broadband")):
        band_axes = getattr(axes, band)
        chart = compare_spectrum_chart(
            selected,
            context.settings,
            hue=spec.varying_key,
            include_brem="Bremsstrahlung" in component_values,
            include_line="Line" in component_values,
            include_characteristic="Characteristic" in component_values,
            include_coherent=context.show_both_emissions,
            x_type=band_axes.x_type,
            y_type=band_axes.y_type,
            band=band,
        )
        chart = context.title_for_face(themed_chart(chart, theme))
        parts.extend(
            [
                mo.md(f"**{heading}**"),
                static_altair_chart(mo, chart)
                if chart is not None
                else mo.md(f"*No {heading.lower()} spectra for this slice.*"),
            ]
        )
    return mo.vstack(parts)
