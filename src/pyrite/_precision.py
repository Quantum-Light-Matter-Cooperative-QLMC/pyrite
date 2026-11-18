"""Import-light adaptive sampling policy shared by API, cases, and runners."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import wraps
from typing import Any, Self

OBSERVABLES = ("cxr", "brem")

#: Field name -> serialized key. Stored ``adaptive_precision`` payloads, case and
#: dataset identities, and internal catalog tables keep the historical
#: ``*_electrons`` vocabulary (#1007), so renaming a field never re-keys a result.
STORED_KEYS = {
    "min_histories": "min_electrons",
    "max_histories": "max_electrons",
    "block_histories": "block_electrons",
    "max_history_share": "max_electron_share",
    "min_effective_histories": "min_effective_electrons",
    "pilot_histories": "pilot_electrons",
    "brem_min_histories": "brem_min_electrons",
    "brem_max_histories": "brem_max_electrons",
}
_FIELD_NAMES = {stored: name for name, stored in STORED_KEYS.items()}

#: Required keys of a serialized policy (stored vocabulary).
REQUIRED_FIELDS = ("target_rse", "max_electrons")

#: Serialized keys renamed by #1253, read from stored payloads under their new name.
LEGACY_STORED_KEYS = {"stability_blocks": "stability_checks"}

#: Keyword arguments renamed by #1253: accepted with a deprecation warning.
DEPRECATED_ARGUMENTS = {"stability_blocks": "stability_checks"}

#: Default ``min_histories`` when omitted (capped at ``max_histories``).
DEFAULT_MIN_HISTORIES = 100

#: Default for the counts that follow the now-optional ``min_histories``; keeps
#: the positional field order while still refusing an omitted count.
_REQUIRED: Any = object()
#: Omitted ``min_histories``: resolves in ``__post_init__``.
_DEFAULT_MIN: Any = None


def reject_renamed_arguments[T: type](cls: T) -> T:
    """Make ``cls(<old>_electrons=...)`` name its ``*_histories`` field (#1007).

    Arguments renamed later (:data:`DEPRECATED_ARGUMENTS`, #1253) still work,
    with a deprecation warning naming the replacement.
    """
    init = cls.__init__

    @wraps(init)
    def __init__(self, *args, **kwargs):
        for name in list(kwargs):
            if name in _FIELD_NAMES:
                raise TypeError(
                    f"{type(self).__name__}({name}=...) was removed; use {_FIELD_NAMES[name]}="
                )
            if name in DEPRECATED_ARGUMENTS:
                from ._util.deprecations import warn_api

                warn_api(f"AdaptiveSampling({name}=...)")
                kwargs[DEPRECATED_ARGUMENTS[name]] = kwargs.pop(name)
        init(self, *args, **kwargs)

    cls.__init__ = __init__  # ty: ignore[invalid-assignment]
    return cls


def _default_minimum(maximum: object) -> int:
    """``DEFAULT_MIN_HISTORIES``, lowered to a valid ``max_histories``."""
    if type(maximum) is int and maximum >= 2:
        return min(DEFAULT_MIN_HISTORIES, maximum)
    return DEFAULT_MIN_HISTORIES


def _require_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not (
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")


@reject_renamed_arguments
@dataclass(frozen=True)
class Precision:
    """Stopping-rule settings for one adaptive case.

    The rule is checked at decision points ``d_0 = min_histories``,
    ``d_{j+1} = ceil(d_j (1 + decision_growth))``, ending at ``max_histories``.
    They depend on the policy alone, so the realized count and output do not
    depend on how electrons are grouped for transport (#1253).

    Parameters
    ----------
    target_rse
        Target relative standard error of each observable's mean.
    min_histories, max_histories
        Smallest count at which the rule may stop (default: 100, at most
        ``max_histories``), and the count at which it stops regardless,
        flagged ``statistics_limited``.
    block_histories
        Deprecated (#1253): an optional transport-chunk size hint. It never
        changes decisions, realized counts, output or identity; ``None``
        sizes chunks from device fill and memory.
    observables
        Non-empty subset of ``('cxr', 'brem')`` that must all converge; a single
        name such as ``"cxr"`` is accepted.
    max_history_share
        Guard: the largest single-electron value may hold at most this share
        of the observable's sum.
    min_effective_histories
        Guard: the effective sample size ``(sum m)**2 / sum m**2`` must reach
        this floor; zero disables it.
    max_variance_of_variance
        CXR guard: ``VOV + 1/n = M4/M2**2`` (central sums) must be at most
        this value, unless empirical ``s/abs(mean)`` is at most
        ``target_rse``. The default 0.01 estimates the variance to about 10 %
        (the RSE to about 5 %); 0.1 is the looser MCNP/Geant4 convention.
        ``M4/M2**2 <= 1`` always, so 1 disables the guard.
    stability_checks, stability_fraction
        Guard: the running mean at each of the last ``k`` decision points
        ``d_j`` must lie within ``max(stability_fraction * target_rse,
        0.9 r_n sqrt((n / d_j - 1) / decision_growth))`` (relative) of the
        current mean: a fixed share of the sample, tolerant of i.i.d. drift,
        so a run does not stop on the segment a rare heavy electron just
        moved; ``k = 0`` disables.
    pilot_histories
        Optional pilot count. At the pilot the run projects
        ``N = N_pilot * max(rse / target)**2``, adds ``N`` to the decision
        points and skips the checks until it gets there.
    band_eV
        Monitor band ``(start, stop)`` in eV; ``None`` takes the case's line
        band (its policy bandwidth, else its line-grid end points).
    batch_means_band_eV
        Optional band for per-bin batch-means errors of the final ``spec`` and
        ``brem_wide``; reported only, never used to stop. Costs one extra
        line and brem reduction over the realized segments.
    brem_target_rse
        Enable independent brem-prefix stopping at this target. Requires both
        observables. ``target_rse`` then governs CXR; line transport continues
        until brem finishes, so ``Ne_brem <= Ne``. Omit for the shared rule.
    brem_min_histories, brem_max_histories
        Optional brem bounds, defaulting to the shared bounds. The brem
        maximum cannot exceed ``max_histories``. Guards and decision growth
        are shared; the brem pilot is clipped to its own bounds.
    decision_growth
        Relative spacing ``delta`` of the decision points: the stop overshoots
        the last refused point by at most ``delta`` relative, with
        ``O(log(max / min) / delta)`` checks.
    """

    target_rse: float
    min_histories: int = _DEFAULT_MIN
    max_histories: int = _REQUIRED
    block_histories: int | None = None
    observables: tuple[str, ...] = ("cxr",)
    max_history_share: float = 0.05
    min_effective_histories: float = 100.0
    max_variance_of_variance: float = 0.01
    stability_checks: int = 3
    stability_fraction: float = 0.5
    pilot_histories: int | None = None
    band_eV: tuple[float, float] | None = None
    batch_means_band_eV: tuple[float, float] | None = None
    brem_target_rse: float | None = None
    brem_min_histories: int | None = None
    brem_max_histories: int | None = None
    decision_growth: float = 0.1

    def __post_init__(self) -> None:
        if self.max_histories is _REQUIRED:
            raise TypeError(f"{type(self).__name__} requires max_histories")
        if self.min_histories is None:
            object.__setattr__(self, "min_histories", _default_minimum(self.max_histories))
        # Freeze caller-owned sequences as well as the dataclass itself. A bare
        # name is one observable, not its characters.
        raw = self.observables
        observables = (raw,) if isinstance(raw, str) else tuple(raw)
        object.__setattr__(self, "observables", observables)
        for name in ("band_eV", "batch_means_band_eV"):
            band = getattr(self, name)
            if band is not None:
                object.__setattr__(self, name, tuple(band))
        for name in ("target_rse", "decision_growth"):
            _require_positive_finite(name, getattr(self, name))
        for name in ("block_histories", "min_histories", "max_histories", "pilot_histories"):
            value = getattr(self, name)
            if name in ("block_histories", "pilot_histories") and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.min_histories <= self.max_histories:
            raise ValueError("need 2 <= min_histories <= max_histories")
        if (
            not self.observables
            or not set(self.observables) <= set(OBSERVABLES)
            or len(set(self.observables)) != len(self.observables)
        ):
            raise ValueError(f"observables must be a non-empty subset of {OBSERVABLES}")
        self._validate_unit_fractions()
        for name in ("min_effective_histories", "stability_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.stability_checks) is not int or self.stability_checks < 0:
            raise ValueError("stability_checks must be a non-negative integer")
        if self.pilot_histories is not None and not (
            self.min_histories <= self.pilot_histories <= self.max_histories
        ):
            raise ValueError("pilot_histories must lie between min_histories and max_histories")
        for name in ("band_eV", "batch_means_band_eV"):
            band = getattr(self, name)
            if band is not None and (
                len(band) != 2
                or any(isinstance(value, bool) or not math.isfinite(value) for value in band)
                or not 0.0 < band[0] < band[1]
            ):
                raise ValueError(f"{name} must contain two positive finite increasing energies")
        self._validate_brem_policy()

    def _validate_unit_fractions(self) -> None:
        for name in ("max_history_share", "max_variance_of_variance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

    def _validate_brem_policy(self) -> None:
        if self.brem_target_rse is None:
            if self.brem_min_histories is not None or self.brem_max_histories is not None:
                raise ValueError("brem count bounds require brem_target_rse")
            return
        if set(self.observables) != set(OBSERVABLES):
            raise ValueError("brem_target_rse requires both cxr and brem observables")
        policy = self.brem_policy()
        if policy.max_histories > self.max_histories:
            raise ValueError("brem_max_histories must not exceed max_histories (brem is a prefix)")

    def brem_policy(self) -> Precision:
        """Scalar brem rule for independent stopping; shares decision growth and guards."""
        minimum = self.min_histories if self.brem_min_histories is None else self.brem_min_histories
        maximum = self.max_histories if self.brem_max_histories is None else self.brem_max_histories
        pilot = self.pilot_histories
        return replace(
            self,
            target_rse=self.target_rse if self.brem_target_rse is None else self.brem_target_rse,
            min_histories=minimum,
            max_histories=maximum,
            observables=("brem",),
            pilot_histories=None if pilot is None else min(max(pilot, minimum), maximum),
            brem_target_rse=None,
            brem_min_histories=None,
            brem_max_histories=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the requested policy, without realized counts or statistics.

        Keys use the stored vocabulary (:data:`STORED_KEYS`), not field names.
        """
        # Persisted requested-policy and identity payloads retain their historical
        # vocabulary. Public in-memory observables are canonical CXR names.
        values = {STORED_KEYS.get(name, name): value for name, value in asdict(self).items()}
        # Absent optional settings must not change existing policy identities.
        for key in (
            "brem_target_rse",
            "brem_min_electrons",
            "brem_max_electrons",
            "block_electrons",
        ):
            if values[key] is None:
                del values[key]
        values["observables"] = tuple(
            "line" if name == "cxr" else name for name in self.observables
        )
        return values

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Validate a serialized (stored-vocabulary) policy and restore its fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("adaptive_precision must be a mapping")
        values = dict(payload)
        for legacy, key in LEGACY_STORED_KEYS.items():
            if legacy in values:
                values.setdefault(key, values.pop(legacy))
        stored = [STORED_KEYS.get(name, name) for name in cls.__dataclass_fields__]
        names = list(cls.__dataclass_fields__)
        unknown = sorted(_FIELD_NAMES.get(key, key) for key in set(values) - set(stored))
        if unknown:
            raise ValueError(f"unknown adaptive-sampling field(s) {unknown}; valid fields: {names}")
        missing = [_FIELD_NAMES.get(key, key) for key in REQUIRED_FIELDS if key not in values]
        if missing:
            raise ValueError(f"adaptive sampling requires {', '.join(missing)}")
        for name in ("observables", "band_eV", "batch_means_band_eV"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        if "observables" in values:
            values["observables"] = tuple(
                "cxr" if value == "line" else value for value in values["observables"]
            )
        return cls(**{_FIELD_NAMES.get(key, key): value for key, value in values.items()})

    def validate_case(self, case: Mapping[str, Any]) -> None:
        """Reject routes whose samples do not satisfy the block-driver contract."""
        if case.get("coherent_emission"):
            raise ValueError(
                "adaptive precision does not support coherent emission; use fixed counts"
            )
        # Particle cascades stop on whole primary histories (#1240).
        for key in ("gdf_source", "groove_spacing_ang"):
            if case.get(key) is not None and case.get(key) is not False:
                raise ValueError(f"adaptive precision does not support {key}; use fixed counts")


# Sampling identity generation: v1 was the unmarked mean-only guards; v2 tied
# the CXR variance gate to ``1/min_effective_histories`` (#1254); v3 gave it its
# own ``max_variance_of_variance`` threshold (#1262); v4 checks at geometric
# decision points with a drift-tolerant stability window, independent of
# transport chunks (#1253).
ADAPTIVE_SAMPLING_MODEL = "decision-points-v4"


def identity_policy(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A stored policy without its transport-only chunk hint (``block_electrons``).

    The hint never changes decisions or output (#1253), so it stays out of
    dataset identities and case content keys.
    """
    if not isinstance(payload, Mapping) or "block_electrons" not in payload:
        return payload
    return {key: value for key, value in payload.items() if key != "block_electrons"}


def _adaptive_sampling_markers(policy) -> dict[str, str]:
    """Re-key adaptive results when the stopping rule changes; fixed N is stable."""
    return {"adaptive_sampling_model": ADAPTIVE_SAMPLING_MODEL} if policy is not None else {}


_BREM_BAND_REDUCTION_MODEL = "cuda-fused-fp32-row-fp64-primary-band-v2"


def _brem_band_reduction_markers(policy, model) -> dict[str, str]:
    """Re-key adaptive BremsLib brem stops when reduction arithmetic changes.

    Fixed counts, CXR-only stopping and other continuum models keep their
    identity. This generation is backend-independent, like the other default
    numerics markers; it never changes source-table or interpolation identity.
    """
    if (
        model == "bremslib"
        and isinstance(policy, Mapping)
        and "brem" in policy.get("observables", ())
    ):
        return {"brem_band_reduction_model": _BREM_BAND_REDUCTION_MODEL}
    return {}


#: Policy for a profile that sets neither fixed history counts nor a
#: ``precision`` table (#361 user decision, 2026-10-08).
DEFAULT_PRECISION = Precision(
    target_rse=0.05,
    min_histories=200,
    max_histories=20_000,
    observables=("cxr", "brem"),
)


def default_precision_fallback(
    *,
    fixed_counts: bool,
    emission: str | None,
    gdf_beam: bool = False,
    grooved: bool = False,
) -> str | None:
    """Why a profile without a ``precision`` table keeps fixed counts, or ``None``.

    ``None`` means :data:`DEFAULT_PRECISION` applies. The reasons mirror the
    routes :meth:`Precision.validate_case` refuses, plus an explicit count.
    Particle cascades run the default too (#1240): their sample is the whole
    primary history and the realized run is the fixed run at its count.
    """
    if fixed_counts:
        return "the profile sets fixed history counts"
    if emission in ("coherent", "both"):
        return f"emission is {emission!r} (adaptive is incoherent-only)"
    if gdf_beam:
        return "the beam is a GDF particle file"
    if grooved:
        return "the target has a grooved entrance face"
    return None
