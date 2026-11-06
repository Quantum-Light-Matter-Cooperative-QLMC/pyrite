"""Opt-in policy for roulette on incoherent CXR line scoring (#879).

A case holding ``line_roulette`` scores each primary history's lines either
surely or with one probability floor, reweighted by its inverse. The record is
part of the case payload, so it enters case identity; absent, nothing changes.

Validation: line-scoring-roulette
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: Versioned estimator: pre-screen, strata and draw stream of this policy.
LINE_ROULETTE_MODEL = "exact-line-mass-two-stratum-v1"

#: Case keys whose line scoring is not an additive per-history incoherent sum.
_REFUSED_KEYS = {
    "coherent_emission": "coherent emission squares a sum of amplitudes",
    "temporal_profile": "temporal profiles accumulate unweighted line boxes",
    "channeling": "channeling joins zones across a history",
    "max_dE_frac": "substeps of a flight add coherently, so a row has no separate line mass",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class LineRoulette:
    """Keep low-mass histories with probability ``floor`` and weight ``1/floor``.

    ``pilot_electrons`` leading primaries are always scored, and the
    ``1 - keep_fraction`` quantile of their exact line masses is the threshold
    at or above which later primaries are scored surely.

    Validation: line-scoring-roulette
    """

    floor: float
    keep_fraction: float = 0.1
    pilot_electrons: int = 128
    model: str = LINE_ROULETTE_MODEL

    def __post_init__(self) -> None:
        if self.model != LINE_ROULETTE_MODEL:
            raise ValueError(f"unknown line roulette model {self.model!r}")
        for name in ("floor", "keep_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
                raise ValueError(f"line roulette {name} must lie in (0, 1]")
        pilot = self.pilot_electrons
        if isinstance(pilot, bool) or not isinstance(pilot, int) or pilot < 1:
            raise ValueError("line roulette pilot_electrons must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LineRoulette:
        if not isinstance(payload, Mapping):
            raise TypeError("line_roulette must be a mapping")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown line roulette field(s) {unknown}")
        return cls(**payload)

    def validate_case(self, case: Mapping[str, Any]) -> None:
        """Refuse routes whose line tally is not a sum over primary histories."""
        for key, reason in _REFUSED_KEYS.items():
            if case.get(key):
                raise ValueError(f"line roulette does not support {key}: {reason}")
        if (case.get("transition_radiation") or {}).get("model") == "two-beam":
            raise ValueError("line roulette does not support two-beam line scoring")
