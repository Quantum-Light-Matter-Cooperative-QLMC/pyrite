from __future__ import annotations

from pyrite.apps.analyze import cached_analysis
from pyrite.materials import CATALOG
from pyrite.plots import (
    MATERIAL_COMPARISON_SUMMARY_VERSION,
    material_comparison_summary,
    select_material_comparison,
)
from pyrite.plots.altair.spectra import material_comparison_chart

from .common import themed_chart


def render_cross_material(
    mo,
    *,
    settings,
    analysis_checkpoint_manifest,
    comparison_stem,
    compare_all_ui,
    energy_ui,
    compare_all: bool,
    energy,
    theme,
):
    description = mo.md(
        "For every material with a checkpoint, show the best dominant line selected "
        "by quality × peak flux, peak flux, and local line-to-bremsstrahlung ratio. "
        "Candidate lines with quality below 0.5 are rejected."
    )
    beam_energy = None if compare_all else energy
    # A material's data may live under its own stem (a direct `<material>.pkl`)
    # or under a named catalog_profile's stem; comparison_stem resolves either
    # so a profile-only material still counts here, matching the material
    # dropdown's own availability check.
    stems = {material_key: comparison_stem(material_key) for material_key in CATALOG.material_keys}
    materials_with_data = sum(
        1
        for stem in stems.values()
        if stem is not None and (analysis_checkpoint_manifest(stem) or {}).get("n_records", 0) > 0
    )
    if materials_with_data < 2:
        return mo.vstack(
            [
                description,
                mo.md("*Run `scan_app.py` for more materials to populate this comparison.*"),
            ]
        )

    summaries = {}
    for material_key, stem in stems.items():
        if stem is None:
            continue
        summary = cached_analysis(
            stem,
            lambda results: material_comparison_summary(results, settings),
            (
                "material_comparison_summary",
                MATERIAL_COMPARISON_SUMMARY_VERSION,
                0.03,
                "sharpness",
                settings.beam_current_na,
            ),
        )
        if summary:
            summaries[material_key] = summary

    def comparison(select):
        points = []
        dropped = {}
        for material_key, summary in summaries.items():
            point, reason = select_material_comparison(
                summary,
                select=select,
                beam_energy_keV=beam_energy,
                min_line_quality=0.5,
            )
            label = CATALOG.material(material_key).label
            if point is None:
                if reason is not None:
                    dropped[label] = reason
                continue
            points.append((label, *point))
        return themed_chart(
            material_comparison_chart(
                points,
                dropped,
                select=select,
                beam_energy_keV=beam_energy,
                min_line_quality=0.5,
            ),
            theme,
        )

    return mo.vstack(
        [
            description,
            mo.hstack([compare_all_ui, energy_ui], wrap=True),
            comparison("quality_peak"),
            comparison("peak"),
            comparison("line_brem_ratio"),
        ]
    )
