from pyrite.apps.analyze import (
    apply_characteristic,
    apply_emission,
    load_analysis_checkpoint,
)
from pyrite.campaign.config import default_settings
from pyrite.results import filter_results, select_results, select_thickness
from pyrite.runs.run import DEFAULT_CHECKPOINT_DIR, cases_from_results

from .controls import SLICE_KEYS
from .models import AnalysisContext
from .pickers import checkpoint_face, checkpoint_notice


def _selected_value(widget) -> str | None:
    value = widget.value
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("value")
    return value


def selected_checkpoint_stem(checkpoint_widget) -> str | None:
    """The Checkpoint picker's stem, or ``None`` while its placeholder is shown."""
    return _selected_value(checkpoint_widget) or None


def load_context(material_widget, checkpoint_widget, checkpoint_dir=None) -> AnalysisContext:
    """Resolve the picker selection and load exactly one analysis checkpoint."""
    root = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    material = _selected_value(material_widget)
    stem = selected_checkpoint_stem(checkpoint_widget) if material is not None else None
    settings = default_settings()

    loaded = load_analysis_checkpoint(stem, root) if stem is not None else None
    load_error = None
    if stem is not None and loaded is None:
        load_error = f"Checkpoint `{stem}` could not be loaded."

    raw_results = loaded or {}
    cases = cases_from_results(raw_results)
    checkpoint_results = filter_results(raw_results, cases)

    return AnalysisContext(
        selected_material=material,
        selected_face=checkpoint_face(material, stem),
        checkpoint_stem=stem,
        settings=settings,
        checkpoint_results=checkpoint_results,
        results=checkpoint_results,
        cases=cases,
        load_error=load_error,
        notice=checkpoint_notice(material, stem, root),
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


def slice_results(results, **slice_values):
    """Pin ``results`` to sidebar slice values keyed as in :data:`SLICE_KEYS`.

    ``None`` values are skipped. ``thickness`` goes through
    :func:`select_thickness`, so an energy whose beam died before the slab keeps
    its thickest computed slab instead of disappearing.
    """
    unknown = set(slice_values) - set(SLICE_KEYS)
    if unknown:
        raise ValueError(f"unknown slice keys: {sorted(unknown)}")
    selected = results
    thickness = slice_values.get("thickness")
    if thickness is not None:
        selected = select_thickness(selected, thickness)
    constraints = {
        SLICE_KEYS[name]: value
        for name, value in slice_values.items()
        if name != "thickness" and value is not None
    }
    return select_results(selected, **constraints) if constraints else selected
