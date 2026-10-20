"""Per-case transition-radiation record (#103).

``Physics(transition_radiation="history" | "bunch")`` adds one
``transition_radiation`` record to every lowered case; ``Case`` validation then
refuses the per-case combinations the component does not support (grooved
faces, adaptive precision, temporal profiles, non-midpoint integration).
"""

from __future__ import annotations

from dataclasses import replace

from .._numerics import Physics
from ..montecarlo.case import Case, transition_case_record


def resolve_transition_radiation(cases: list[Case], physics: Physics) -> list[Case]:
    """Attach the TR record to every case; ``"off"`` returns ``cases`` unchanged."""
    if physics.transition_radiation == "off":
        return cases
    record = transition_case_record(physics.transition_radiation)
    if physics.transition_radiation == "two-beam":
        record["lattice_origin_ang"] = list(
            physics.transition_lattice_origin_ang or (0.0, 0.0, 0.0)
        )
    return [replace(case, transition_radiation=record) for case in cases]
