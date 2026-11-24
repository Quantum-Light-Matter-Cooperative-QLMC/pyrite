"""Opt-in detector-cone splitting of elastic events (#203, #379).

A case holding ``cone_split`` transports its primaries with splitting plus
Russian roulette at ELSEPA elastic events toward the case's single detector
direction (:mod:`pyrite.montecarlo.transport.cone_split`). Each primary plus
its weighted copies is one statistical sample. The record is part of the case
payload, so it enters case identity; absent, transport stays analog bit for bit.

Experimental: the adaptive monitor scores weighted per-history samples, and
the spectrum phase scores each power-of-two copy-weight stratum with the
production scorers (:mod:`pyrite.montecarlo.runner.cone_split_strata`); the
line-yield audit folds copy masses per history; grid resolution sees copy lines.

Validation: detector-cone-variance-reduction
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: Versioned estimator: event partition, roulettes, power-of-two weight ladder
#: and copy streams.
CONE_SPLIT_MODEL = "elastic-cone-split-ladder-v1"

#: Below this beam energy #203 measured near-unit gain (1-2 MeV h-BN).
LOW_GAIN_ENERGY_KEV = 3000.0

_ELECTRON_REST_KEV = 510.99895

#: Case keys whose transport or tallies cannot carry per-history copy weights.
_REFUSED_KEYS = {
    "coherent_emission": "coherent emission squares a sum of amplitudes over correlated copies",
    "temporal_profile": "temporal profiles accumulate unweighted line boxes",
    "channeling": "channeling transport has no split port",
    "channeling_source": "channeling transport has no split port",
    "groove_spacing_ang": "grooved entrance faces have no split port",
    "gdf_source": "GDF beams have no split port",
    "secondary_threshold_eV": "particle cascades have no split port",
    "pair_production_model": "particle cascades have no split port",
    "positron_transport": "particle cascades have no split port",
    "photon_transport": "photon transport has no split port",
    "transition_radiation": "transition radiation is not a per-history incoherent sum",
    "line_roulette": "line roulette is not composed with copy weights",
}


class ConeSplitLowGainWarning(UserWarning):
    """Detector-cone splitting requested where #203 measured no gain."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ConeSplitting:
    """Split elastic events toward the detector cone; roulette low-weight copies.

    The cone half-angle is ``cone_factor / gamma0`` about the case's detector
    direction. Elastic events below ``min_energy_fraction * E0`` stay analog.
    ``angle_scale`` (``u_ref``) applies the angle-scaled roulette; ``None``
    disables it. ``target_weight`` is the roulette target copy weight and
    ``max_copies_per_electron`` the copy buffer, which raises when full.
    A final roulette snaps each copy weight to a power of two (the weight
    ladder), so spectrum tallies can score weight strata with unchanged
    scorers. Defaults are the #203 measured values (5 MeV h-BN 1 mm).

    Validation: detector-cone-variance-reduction
    """

    cone_factor: float = 8.0
    min_energy_fraction: float = 0.4
    angle_scale: float | None = 1.0
    target_weight: float = 1e-3
    max_copies_per_electron: int = 1024
    model: str = CONE_SPLIT_MODEL

    def __post_init__(self) -> None:
        if self.model != CONE_SPLIT_MODEL:
            raise ValueError(f"unknown cone split model {self.model!r}")
        # NumPy scalars are accepted and stored as Python numbers (JSON identity).
        for name in ("cone_factor", "min_energy_fraction", "angle_scale", "target_weight"):
            value = getattr(self, name)
            if isinstance(value, numbers.Real) and not isinstance(value, bool | float):
                object.__setattr__(self, name, float(value))
        cap = self.max_copies_per_electron
        if isinstance(cap, numbers.Integral) and not isinstance(cap, bool | int):
            object.__setattr__(self, "max_copies_per_electron", int(cap))
        _positive_finite("cone_factor", self.cone_factor)
        _positive_finite("target_weight", self.target_weight)
        if self.angle_scale is not None:
            _positive_finite("angle_scale", self.angle_scale)
        fraction = self.min_energy_fraction
        if not _is_real(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError("cone split min_energy_fraction must lie in [0, 1)")
        cap = self.max_copies_per_electron
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("cone split max_copies_per_electron must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConeSplitting:
        if not isinstance(payload, Mapping):
            raise TypeError("cone_split must be a mapping")
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown cone split field(s) {unknown}")
        return cls(**payload)

    def half_angle_rad(self, E0_keV: float) -> float:
        """Cone half-angle ``cone_factor / gamma0``; must lie below pi/2."""
        angle = self.cone_factor / (1.0 + float(E0_keV) / _ELECTRON_REST_KEV)
        if not 0.0 < angle < math.pi / 2:
            raise ValueError(
                f"cone split half-angle {angle:.3g} rad at {E0_keV:g} keV is not below pi/2; "
                "lower cone_factor"
            )
        return angle

    def transport_parameters(self, E0_keV: float) -> dict[str, Any]:
        """``ConeSplit`` keywords other than the detector direction, at beam energy ``E0_keV``."""
        return {
            "half_angle_rad": self.half_angle_rad(E0_keV),
            "target_weight": float(self.target_weight),
            "max_copies_per_electron": int(self.max_copies_per_electron),
            "min_energy_keV": float(self.min_energy_fraction) * float(E0_keV),
            "angle_scale": math.inf if self.angle_scale is None else float(self.angle_scale),
            "weight_ladder": True,
        }

    def validate_case(self, case: Mapping[str, Any]) -> None:
        """Refuse configurations splitting cannot score correctly."""
        for key, reason in _REFUSED_KEYS.items():
            value = case.get(key)
            if value is not None and value is not False:
                raise ValueError(f"detector-cone splitting does not support {key}: {reason}")
        if case.get("transport_mode") == "mixed":
            raise ValueError("detector-cone splitting does not support transport_mode='mixed'")
        elastic = case.get("elastic_model", "mott")
        if elastic != "elsepa":
            raise ValueError(
                f"detector-cone splitting needs elastic_model='elsepa', not {elastic!r}"
            )
        self.half_angle_rad(case["E0_keV"])


def warn_low_gain(case: Mapping[str, Any]) -> None:
    """Warn when a split case sits where #203 measured near-unit gain."""
    E0 = float(case["E0_keV"])
    if E0 < LOW_GAIN_ENERGY_KEV:
        warnings.warn(
            f"detector-cone splitting at {E0:g} keV: below {LOW_GAIN_ENERGY_KEV:g} keV the "
            "detector-cone population is not rare and #203 measured near-unit gain; "
            "the copies cost transport time without reducing variance",
            ConeSplitLowGainWarning,
            stacklevel=3,
        )


def split_transport_core(requested: str = "auto") -> str:
    """The only core with a split port: the per-electron CPU core."""
    if requested in ("auto", "per-electron"):
        return "per-electron"
    raise ValueError(
        "detector-cone splitting runs only on the per-electron CPU transport core; "
        f"transport_core={requested!r} has no split port"
    )


def _is_real(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _positive_finite(name: str, value) -> None:
    if not _is_real(value) or value <= 0.0:
        raise ValueError(f"cone split {name} must be positive and finite")


__all__ = [
    "CONE_SPLIT_MODEL",
    "ConeSplitLowGainWarning",
    "ConeSplitting",
    "split_transport_core",
    "warn_low_gain",
]
