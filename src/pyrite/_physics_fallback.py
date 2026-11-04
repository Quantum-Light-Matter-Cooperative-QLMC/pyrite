"""Warning category for a physical model substituted after missing reference data.

Consumers may promote this category to an error with Python's warning filters.
The presence of this warning means the requested and executed physics differ;
selecting EEDL or another model *explicitly* is not a fallback.
"""

import os
import warnings
from contextlib import contextmanager
from contextvars import ContextVar


class PhysicsFallbackWarning(RuntimeWarning):
    """A requested physics model was replaced by a lower-fidelity surrogate."""


_AUDIT: ContextVar["FallbackAudit | None"] = ContextVar("physics_fallback_audit", default=None)


class FallbackAudit:
    """Additive row counts; no photon-grid arrays or transport state retained."""

    def __init__(self, allow: bool, preflight=()):
        self.allow = allow
        self.preflight = preflight
        self.rows: dict[tuple, dict] = {}

    def record(self, requested, used, element, total, affected, span, scope, reason):
        key = requested, used, element, scope
        event = self.rows.setdefault(
            key,
            dict(
                model_requested=requested,
                model_used=used,
                elements=[element],
                scope=scope,
                reason=reason,
                segments_total=0,
                segments_affected=0,
                energy_span_keV=None,
            ),
        )
        event["segments_total"] += total
        event["segments_affected"] += affected
        if affected:
            event["reason"] = reason
        if span is not None:
            previous = event["energy_span_keV"]
            event["energy_span_keV"] = (
                list(span)
                if previous is None
                else [min(previous[0], span[0]), max(previous[1], span[1])]
            )
        if affected and not self.allow:
            raise PhysicsFallbackWarning(
                reason + ". Physics fallback is disabled; select a covered model or pass "
                "--allow-physics-fallback."
            )

    def events(self):
        return [
            {
                **event,
                "fraction_segments_affected": event["segments_affected"] / event["segments_total"],
            }
            for event in self.rows.values()
            if event["segments_affected"]
        ]

    def snapshot(self):
        return {key: event.copy() for key, event in self.rows.items()}

    def restore(self, snapshot):
        self.rows = snapshot


@contextmanager
def collect_physics_fallbacks(*, allow=False, preflight=(), audit=None):
    """Apply one worker's strictness and return measured substitution events."""
    audit = FallbackAudit(allow, preflight) if audit is None else audit
    token = _AUDIT.set(audit)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always" if audit.allow else "error", PhysicsFallbackWarning)
            yield audit
    finally:
        _AUDIT.reset(token)


@contextmanager
def repeated_photon_block():
    """Subsequent photon blocks rescore identical rows; count only the first."""
    token = _AUDIT.set(None)
    try:
        yield
    finally:
        _AUDIT.reset(token)


def record_segment_fallback(requested, used, element, total, affected, span, scope, reason):
    audit = _AUDIT.get()
    if audit is not None:
        audit.record(requested, used, element, total, affected, span, scope, reason)


def preflight_segment_fallbacks(element, total, span):
    """Measure whole-source substitutions against the rows actually scored."""
    audit = _AUDIT.get()
    if audit is None:
        return
    for event in audit.preflight:
        if event["scope"] == "entire continuum source" and element in event["elements"]:
            audit.record(
                event["model_requested"],
                event["model_used"],
                element,
                total,
                total,
                span,
                event["scope"],
                event["reason"],
            )


def fallback_cases(cases, *, allow, events=()):
    """Carry admission and requested-model provenance across worker boundaries."""
    if not allow:
        return cases
    return [
        {**case, "allow_physics_fallback": True, "physics_fallbacks": list(events)}
        for case in cases
    ]


if os.environ.get("PYRITE_STRICT_PHYSICS_FALLBACK") == "1":
    warnings.filterwarnings("error", category=PhysicsFallbackWarning)
