from pyrite.apps.analyze import (
    apply_characteristic,
    apply_emission,
    checkpoint_stem,
    load_analysis_checkpoint,
)
from pyrite.campaign.config import default_settings
from pyrite.results import filter_results
from pyrite.runs.run import cases_from_results

from .models import AnalysisContext


def _selected_value(widget) -> str | None:
    value = widget.value
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("value")
    return value


def resolve_checkpoint_stem(
    material: str | None, face: str | None, profile: str | None
) -> str | None:
    """The checkpoint stem a material/face/profile selection names, or ``None``.

    A profile other than the material's own standard stem selects its checkpoint
    directly; otherwise the material and face resolve the stem.
    """
    if profile is not None and profile != material:
        return profile
    if material is not None and face is not None:
        return checkpoint_stem(material, face)
    return None


def selected_checkpoint_stem(material_widget, face_widget, profile_widget) -> str | None:
    """Resolve the picker widgets' checkpoint stem without loading the checkpoint."""
    return resolve_checkpoint_stem(
        _selected_value(material_widget),
        _selected_value(face_widget),
        _selected_value(profile_widget),
    )


def load_context(material_widget, face_widget, profile_widget) -> AnalysisContext:
    """Resolve the UI selection and load exactly one analysis checkpoint."""

    material = _selected_value(material_widget)
    face = _selected_value(face_widget)
    profile = _selected_value(profile_widget)
    settings = default_settings()
    stem = resolve_checkpoint_stem(material, face, profile)

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


def select_emission(
    context: AnalysisContext,
    emission: str | None,
    *,
    include_characteristic: bool = True,
) -> AnalysisContext:
    selected = emission or "incoherent"
    resolved = apply_emission(context.checkpoint_results, selected)
    resolved = apply_characteristic(resolved, include=include_characteristic)
    return context.with_results(resolved, emission=selected)
