from __future__ import annotations

from cxr_mc.analyze import apply_emission, checkpoint_stem, load_analysis_checkpoint
from cxr_mc.config import default_settings
from cxr_mc.results import filter_results
from cxr_mc.run import cases_from_results

from .models import AnalysisContext


def _selected_value(widget) -> str | None:
    value = widget.value
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("value")
    return value


def load_context(material_widget, face_widget, profile_widget) -> AnalysisContext:
    """Resolve the UI selection and load exactly one analysis checkpoint."""

    material = _selected_value(material_widget)
    face = _selected_value(face_widget)
    profile = _selected_value(profile_widget)
    settings = default_settings()

    if profile is not None and profile != material:
        stem = profile
    elif material is not None and face is not None:
        stem = checkpoint_stem(material, face)
    else:
        stem = None

    loaded = load_analysis_checkpoint(stem) if stem is not None else None
    load_error = None
    if stem is not None and loaded is None:
        load_error = f"Checkpoint `{stem}` could not be loaded."

    raw_results = loaded or {}
    cases = cases_from_results(raw_results)
    checkpoint_results = filter_results(raw_results, cases)

    return AnalysisContext(
        selected_material=material,
        selected_face=face,
        selected_profile=profile,
        checkpoint_stem=stem,
        settings=settings,
        checkpoint_results=checkpoint_results,
        results=checkpoint_results,
        cases=cases,
        load_error=load_error,
    )


def select_emission(context: AnalysisContext, emission: str | None) -> AnalysisContext:
    selected = emission or "incoherent"
    return context.with_results(apply_emission(context.checkpoint_results, selected))
