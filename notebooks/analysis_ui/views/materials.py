from __future__ import annotations

from cxr_mc.analyze import cached_analysis
from cxr_mc.materials import CATALOG
from cxr_mc.plots import (
    MATERIAL_COMPARISON_SUMMARY_VERSION,
    material_comparison_summary,
    select_material_comparison,
)
from cxr_mc.plots.altair_spectra import material_comparison_chart


def render_cross_material(
    mo,
    *,
    settings,
    analysis_checkpoint_manifest,
    compare_all_ui,
    energy_ui,
    compare_all: bool,
    energy,
):
    description = mo.md(
        "For every material with a checkpoint, show the best dominant line selected "
        "by quality × peak flux, peak flux, and local line-to-bremsstrahlung ratio. "
        "Candidate lines with quality below 0.5 are rejected."
    )
    beam_energy = None if compare_all else energy
    materials_with_data = sum(
        1
        for material_key in CATALOG.material_keys
        if (analysis_checkpoint_manifest(material_key) or {}).get("n_records", 0) > 0
    )
    if materials_with_data < 2:
        return mo.vstack(
            [
                description,
                mo.md("*Run `scan_app.py` for more materials to populate this comparison.*"),
            ]
        )

    summaries = {}
    for material_key in CATALOG.material_keys:
        summary = cached_analysis(
            material_key,
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
        return material_comparison_chart(
            points,
            dropped,
            select=select,
            beam_energy_keV=beam_energy,
            min_line_quality=0.5,
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
