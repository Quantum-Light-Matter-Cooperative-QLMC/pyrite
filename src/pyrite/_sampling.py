"""Explicit Monte Carlo sampling modes for the Python API (issue #680).

Legacy ``Numerics(precision=...)`` and positional trial-count construction remain
valid. These types *select* the same existing fixed/adaptive runner contracts;
they do not change stochastic estimators, RNG streams, or checkpoint payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._precision import Precision


@dataclass(frozen=True)
class FixedSampling:
    """Fixed macro-electron budgets for CXR and bremsstrahlung estimators.

    The two estimators can use different counts. Their existing population
    correlation and replay semantics are preserved, not redefined here.
    """

    cxr_electrons: int = 450
    brem_electrons: int = 100

    def __post_init__(self) -> None:
        for name in ("cxr_electrons", "brem_electrons"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AdaptiveSampling(Precision):
    """Adaptive CXR/brem stopping with the existing validated estimator.

    This is a typed name for the existing :class:`~pyrite.config.Precision` contract.
    Historical ``Precision.to_dict()`` uses ``'line'`` in stored policy
    identities; this subclass deliberately inherits that serialization and
    the independent brem-prefix stopping behavior without modification.
    """


SamplingPolicy = FixedSampling | AdaptiveSampling
