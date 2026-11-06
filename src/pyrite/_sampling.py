"""Monte Carlo history budgets: the single sampling input of ``Numerics`` (#680, #1007).

A history is one transported primary electron and everything it produces; the
budgets count histories, not physical beam electrons.

These types *select* the existing fixed/adaptive runner contracts; they do not
change stochastic estimators, RNG streams, or checkpoint payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import wraps

from ._precision import Precision, reject_renamed_arguments

_RENAMED_FIXED_ARGUMENTS = {
    "cxr_electrons": "cxr_histories",
    "brem_electrons": "brem_histories",
    "cxr_trials": "cxr_histories",
    "brem_trials": "brem_histories",
}


def _reject_renamed_fixed_arguments[T: type](cls: T) -> T:
    init = cls.__init__

    @wraps(init)
    def __init__(self, *args, **kwargs):
        for name in kwargs:
            if name in _RENAMED_FIXED_ARGUMENTS:
                raise TypeError(
                    f"FixedSampling({name}=...) was removed; use {_RENAMED_FIXED_ARGUMENTS[name]}="
                )
        init(self, *args, **kwargs)

    cls.__init__ = __init__  # ty: ignore[invalid-assignment]
    return cls


@_reject_renamed_fixed_arguments
@dataclass(frozen=True)
class FixedSampling:
    """Fixed Monte Carlo history budgets for CXR and bremsstrahlung estimators.

    The two estimators can use different counts. Their existing population
    correlation and replay semantics are preserved, not redefined here.
    """

    cxr_histories: int = 450
    brem_histories: int = 100

    def __post_init__(self) -> None:
        for name in ("cxr_histories", "brem_histories"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@reject_renamed_arguments
@dataclass(frozen=True)
class AdaptiveSampling(Precision):
    """Adaptive CXR/brem stopping on Monte Carlo history counts.

    ``to_dict()``/``from_dict()`` use the stored vocabulary (``min_electrons``,
    ``'line'``, ...) so stored ``adaptive_precision`` records and identity
    digests are unchanged by the ``*_histories`` field names.
    """


SamplingPolicy = FixedSampling | AdaptiveSampling


def adaptive_sampling(policy: Precision) -> AdaptiveSampling:
    """Return the typed policy for a catalog or ``Settings`` stopping rule."""
    if isinstance(policy, AdaptiveSampling):
        return policy
    return AdaptiveSampling(**{f.name: getattr(policy, f.name) for f in fields(Precision)})


def case_budget(policy: SamplingPolicy) -> tuple[int, int, Precision | None]:
    """Lower a policy to the case builder's ``(line, brem, adaptive)`` inputs.

    Adaptive cases request their maximum for both estimators; resolved case
    identity vocabulary (``Ne``, ``Ne_brem``, ``adaptive_precision``) is unchanged.
    """
    if isinstance(policy, FixedSampling):
        return policy.cxr_histories, policy.brem_histories, None
    return policy.max_histories, policy.max_histories, policy
