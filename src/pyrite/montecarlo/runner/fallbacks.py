"""Worker-local fallback policy and measured radiative-transport provenance."""

from contextlib import contextmanager
from functools import wraps

from ..._backend import BACKEND
from ..._physics_fallback import collect_physics_fallbacks


def _radiative_fallback_rows(audit, case, transport):
    events = [event for event in audit.preflight if event["scope"] == "entire radiative transport"]
    if not events:
        return
    segments = transport["segs"]
    xp = BACKEND.xp
    field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energies = xp.asarray(segments[field])
    layers = xp.asarray(segments.get("layer", xp.zeros(energies.shape, dtype=int)))
    compositions = (
        [layer[2] for layer in case["abs_layers"]]
        if case.get("abs_layers")
        else [case["composition"]]
    )
    for event in events:
        for element in event["elements"]:
            mask = xp.zeros(energies.shape, dtype=bool)
            for index, composition in enumerate(compositions):
                if any(symbol == element for symbol, _ in composition):
                    mask |= layers == index
            count = int(BACKEND.to_cpu(xp.count_nonzero(mask)))
            selected = energies[mask]
            span = (
                None
                if not count
                else [float(BACKEND.to_cpu(selected.min())), float(BACKEND.to_cpu(selected.max()))]
            )
            audit.record(
                event["model_requested"],
                event["model_used"],
                element,
                count,
                count,
                span,
                event["scope"],
                event["reason"],
            )


@contextmanager
def spectrum_fallbacks(case, transport, *, audit=None):
    with collect_physics_fallbacks(
        allow=case.get("allow_physics_fallback", False),
        preflight=case.get("physics_fallbacks", ()),
        audit=audit,
    ) as audit:
        _radiative_fallback_rows(audit, case, transport)
        yield audit


def measure_block_fallbacks(function):
    """Keep one audit across streamed blocks, with admission inside each worker."""

    @wraps(function)
    def measured(scorer, segments):
        snapshot = scorer.fallbacks.snapshot()
        try:
            with spectrum_fallbacks(scorer.case, {"segs": segments}, audit=scorer.fallbacks):
                return function(scorer, segments)
        except BaseException:
            scorer.fallbacks.restore(snapshot)
            raise

    return measured
