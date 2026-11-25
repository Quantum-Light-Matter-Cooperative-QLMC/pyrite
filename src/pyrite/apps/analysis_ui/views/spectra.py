"""Spectra view: one dimension varies across curves, the sidebar slice pins the rest."""

from dataclasses import dataclass, field
from typing import Any

from pyrite.plots.plotly.spectra import prepare_compare_spectra, render_prepared_spectra
from pyrite.plots.plotly.temporal import temporal_chart
from pyrite.results import records, select_results, sweep_values

from ..controls import axes_panel
from ..data import slice_results
from ..models import DimensionComparisonSpec
from ..zoom import zoomable_chart
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


_BANDS = (("narrow", "Narrowband"), ("broad", "Broadband"))


@dataclass(frozen=True)
class SpectraPreparation:
    """Native spectra for the current slice, rebuilt only when it changes.

    Zooming re-renders from ``prepared`` without touching records (#1442).
    ``blocked`` holds the message shown instead of charts, if any.
    """

    notes: tuple[str, ...] = ()
    blocked: str | None = None
    selected: Any = None
    prepared: dict = field(default_factory=dict)


def prepare_spectra(*, context, slice_values, spec, varying_values, component_values):
    """Select the slice and build each band's native frame once."""
    results = context.results
    available = sweep_values(results).get(spec.varying_key, []) if records(results) else []
    notes = (f"*Only one {spec.varying_label} in this checkpoint.*",) if len(available) <= 1 else ()
    if not component_values:
        return SpectraPreparation(notes, "*Select at least one component above to plot.*")
    if not varying_values:
        return SpectraPreparation(
            notes, f"*Select at least one {spec.varying_label} above to plot.*"
        )
    selected = spectra_selection(results, slice_values, spec, list(varying_values))
    prepared = {
        band: prepare_compare_spectra(
            selected,
            context.settings,
            hue=spec.varying_key,
            include_brem="Bremsstrahlung" in component_values,
            include_cxr="Line" in component_values,
            include_characteristic="Characteristic" in component_values,
            include_channeling="Channeling" in component_values,
            include_transition="TR/CTR" in component_values,
            include_coherent=context.show_both_emissions,
            sum_components=True,
            band=band,
        )
        for band, _heading in _BANDS
    }
    return SpectraPreparation(notes, None, selected, prepared)


def _band_chart(mo, figure, *, band, x_type, zoom):
    if figure is None or zoom is None or band not in zoom:
        return figure
    get_viewport, set_viewport = zoom[band]
    return zoomable_chart(
        mo, figure, x_type=x_type, get_viewport=get_viewport, set_viewport=set_viewport
    )


def render_spectra(
    mo,
    *,
    context,
    prep,
    spec,
    vary_ui,
    varying_ui,
    components,
    axes_controls,
    axes,
    theme,
    zoom=None,
):
    """The Spectra view: vary radio, curve picker, components, two bands.

    ``zoom`` maps each band to its ``(get_viewport, set_viewport)`` marimo
    state; a zoom re-samples that band from ``prep`` at the chosen resolution.
    """
    from ..radiation import radiation_panel

    parts = [
        mo.hstack([vary_ui, varying_ui], align="end", widths=[1, 1]),
        mo.md(spec.description),
        components,
        radiation_panel(mo, list(records(context.results))),
        axes_panel(mo, axes_controls),
        *(mo.md(note) for note in prep.notes),
    ]
    if prep.blocked is not None:
        parts.append(mo.md(prep.blocked))
        return mo.vstack(parts)

    note = _fallback_note(mo, prep.selected)
    if note is not None:
        parts.append(note)

    for band, heading in _BANDS:
        band_axes = getattr(axes, band)
        viewport = zoom[band][0]() if zoom and band in zoom else None
        chart = render_prepared_spectra(
            prep.prepared.get(band),
            resolution=axes.resolution,
            viewport=viewport,
            x_type=band_axes.x_type,
            y_type=band_axes.y_type,
        )
        chart = context.title_for_face(themed_chart(chart, theme))
        chart = _band_chart(mo, chart, band=band, x_type=band_axes.x_type, zoom=zoom)
        parts.extend(
            [
                mo.md(f"**{heading}**"),
                chart
                if chart is not None
                else mo.md(f"*No {heading.lower()} spectra for this slice.*"),
            ]
        )
    # Opt-in line I(t) (#292): shown only when the checkpoint stored it.
    temporal = temporal_chart(prep.selected, hue=spec.varying_key)
    if temporal is not None:
        parts.extend(
            [
                mo.md("**Temporal profile**"),
                context.title_for_face(themed_chart(temporal, theme)),
            ]
        )
    return mo.vstack(parts)
