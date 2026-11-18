"""Stop extension for a refused default measured line bandwidth (#1243).

The spectrum-phase truncation audit of a defaulted ``resonance-population``
axis also bounds the upper share at a ladder of candidate stops. A refusal then
rescores the kept transport up to the first rung within the share instead of
the clamped kinematic ceiling.
"""

import warnings

import numpy as np

from ..._backend import _to_cpu
from ..._cxr_grid_policy import (
    CxrGridToleranceError,
    CxrGridTruncationWarning,
    kinematic_fallback_payload,
)

__all__ = [
    "EXTENSION_LADDER_FRACTIONS",
    "MAX_EXTENSIONS",
    "extension_stop",
    "fallback_axis_payload",
    "fallback_record",
    "extension_stop_ladder",
    "measured_bandwidth_fallback",
    "spectrum_with_bandwidth_fallback",
]

#: Candidate stops a refused default may extend to: the measured stop plus
#: ``2**(-k / 3)`` of its distance to the ceiling, ``k = 0 .. 30`` (#1243). The
#: finest step is ~1e-3 of that distance; a chosen stop overshoots the exact
#: one by at most a factor ``2**(1/3)`` of its own extension.
EXTENSION_LADDER_FRACTIONS = 2.0 ** (-np.arange(30, -1, -1) / 3.0)


def extension_stop_ladder(stop_eV, ceiling_eV):
    """Increasing candidate stops in ``(stop_eV, ceiling_eV]``; the last is the ceiling.

    Validation: line-grid-resonance-bandwidth
    """
    stop, ceiling = float(stop_eV), float(ceiling_eV)
    ladder = stop + (ceiling - stop) * EXTENSION_LADDER_FRACTIONS
    ladder[-1] = ceiling
    return ladder


def extension_stop(audit, total, limit):
    """Smallest ladder stop whose upper bound is within ``limit``, and that bound.

    The ceiling is the last rung; it is returned when no lower rung passes.

    Validation: line-grid-resonance-bandwidth
    """
    stops = audit.get("extension_stops_eV")
    sums = audit.get("mass_above_extension")
    if stops is None or sums is None or total <= 0.0:
        return None
    shares = np.asarray(_to_cpu(sums), dtype=float) / total
    passing = np.flatnonzero(shares <= limit)
    index = int(passing[0]) if passing.size else len(stops) - 1
    return float(stops[index]), float(shares[index])


#: Refusals after which a defaulted axis goes straight to the ceiling. Each
#: rescore is audited on its own line set, which reflection selection may
#: change; the cap bounds the passes independent of float granularity.
MAX_EXTENSIONS = 2


def fallback_axis_payload(case, refusal, earlier=None):
    """``(payload, stop, ceiling, refusals)`` of the axis a defaulted refusal extends to.

    ``payload`` is the kinematic-shaped payload ending at ``stop``: the
    smallest :func:`extension_stop_ladder` stop within the share, or the
    clamped ceiling when none is, there is no ladder, or ``earlier`` already
    holds :data:`MAX_EXTENSIONS` refusals. An explicitly chosen measured
    bandwidth re-raises ``refusal``.

    Validation: line-grid-resonance-bandwidth
    """
    record = refusal.truncation_record
    payload = case.get("line_grid_policy")
    fallback = None if record is None or payload is None else kinematic_fallback_payload(payload)
    if fallback is None:
        raise refusal
    refusals = [*(earlier or {}).get("refusals", ()), record]
    ceiling = float(fallback["bandwidth"]["stop_eV"])
    stop = ceiling
    if len(refusals) <= MAX_EXTENSIONS:
        stop = min(float(record.get("extension_stop_eV", ceiling)), ceiling)
    fallback["bandwidth"] = {**fallback["bandwidth"], "stop_eV": stop}
    shown = "the kinematic-ceiling axis" if stop >= ceiling else "an extended axis"
    warnings.warn(
        f"{case['name']} at {case['E0_keV']:g} keV: the default measured line axis "
        f"[{record['start_eV']:g}, {record['stop_eV']:g}] eV may truncate "
        f"{record['upper_fraction_bound']:.3g} of the line yield, above the "
        f"{record['truncation_limit']:g} share; rescoring on {shown} to {stop:g} eV "
        f"(ceiling {ceiling:g} eV).",
        CxrGridTruncationWarning,
        stacklevel=3,
    )
    return fallback, stop, ceiling, refusals


def fallback_record(resolved, case, stop, ceiling, refusals):
    """``resolved`` with the ``bandwidth_fallback`` provenance of a rescored axis.

    The axis fields come from the kinematic-shaped payload; ``bandwidth_policy``
    names the measured policy that is audited on it, and ``axis_policy`` the
    payload that built it.
    """
    return {
        **resolved,
        "bandwidth_policy": case["line_grid_policy"]["bandwidth"]["policy"],
        "axis_policy": resolved.get("bandwidth_policy"),
        "bandwidth_fallback": {
            "from": case["line_grid_policy"]["bandwidth"],
            "refused": refusals[-1],
            "refusals": refusals,
            "stop_eV": stop,
            "ceiling_eV": ceiling,
        },
    }


def measured_bandwidth_fallback(case, tp, refusal):
    """``(case, tp)`` rescored on an extended axis after a defaulted refusal.

    A defaulted ``resonance-population`` stop is measured on the case's built
    reflections; automatic selection may score higher-|g| lines above it, which
    the audit sees and refuses (#1243). The transport is kept and only the line
    axis is rebuilt with the same start and resolution, up to the stop
    :func:`fallback_axis_payload` chooses.

    The returned case keeps its measured payload, so the rescore is audited
    again: automatic reflection selection reruns on the wider axis and may
    select another line set than the refused pass audited (#1243). A rescore
    that refuses in turn extends from its own stop; the ceiling only warns.

    Validation: line-grid-resonance-bandwidth
    """
    from .cxr_grid import resolve_cxr_grid

    earlier = (tp.get("diagnostic_grid") or {}).get("bandwidth_fallback")
    fallback, stop, ceiling, refusals = fallback_axis_payload(case, refusal, earlier)
    phase_cache = case.get("_spectrum_phase_cache")
    case = dict(case)
    if phase_cache is not None:
        # An allocation retry keeps finished phases and the line-roulette
        # screen; the rescored axis gets its own, so neither the refused
        # axis's screen nor the rescored lines cross over (#1243).
        case["_spectrum_phase_cache"] = phase_cache.setdefault("bandwidth_fallback", {})
    # The axis is the kinematic-shaped payload's [start, stop] at the same
    # resolution; scoring keeps the measured payload and its audit.
    E_grid, resolved = resolve_cxr_grid(
        {**case, "line_grid_policy": fallback},
        tp["segs"],
        tp["n_hat"],
        tp["Ne_lines"],
        tp["E_grid"],
        case.get("abs_layers"),
        tp["groove"],
    )
    if resolved is None:  # unreachable: a policy payload always yields a record
        raise refusal
    resolved = fallback_record(resolved, case, stop, ceiling, refusals)
    return case, {**tp, "E_grid": E_grid, "diagnostic_grid": resolved}


def spectrum_with_bandwidth_fallback(score, case, tp):
    """``(score(case, tp), case, tp)``, rescored on extended axes while a defaulted audit refuses.

    Each rescore's stop is above the refused one and at most the ceiling, where
    the audit warns instead of refusing, so the loop ends. The returned case and
    transport are the ones scored, for consumers of the final axis.

    Validation: line-grid-resonance-bandwidth
    """
    if case.get("_joint_bandwidth_fallback"):
        # Observation directions share one axis; their caller extends it.
        return score(case, tp), case, tp
    while True:
        try:
            return score(case, tp), case, tp
        except CxrGridToleranceError as refusal:
            stop = float(tp["E_grid"][-1])
            case, tp = measured_bandwidth_fallback(case, tp, refusal)
            if not float(tp["E_grid"][-1]) > stop:
                raise refusal from None
