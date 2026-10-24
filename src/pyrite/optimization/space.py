"""Typed, backend-neutral parameter definitions for optimization studies.

Parameter values describe *effective* scene values. No implicit rounding,
clipping or logarithmic transformations are applied when accepting candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Continuous:
    """Continuous inclusive interval; ``log=True`` requires a positive interval."""

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        lo = _finite_real(self.low, name="low")
        hi = _finite_real(self.high, name="high")
        if not isinstance(self.log, bool):
            raise TypeError("log must be a bool")
        if lo >= hi:
            raise ValueError("continuous low must be less than high")
        if self.log and lo <= 0:
            raise ValueError("logarithmic parameters must be positive")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    def validate(self, value: Any) -> float:
        number = _finite_real(value, name="continuous value")
        if not self.low <= number <= self.high:
            raise ValueError(f"continuous value {number} outside [{self.low}, {self.high}]")
        return number


@dataclass(frozen=True, slots=True)
class Integer:
    """Inclusive discrete interval, with a positive integral step."""

    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        for key in ("low", "high", "step"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"integer {key} must be an integer")
        if not isinstance(self.log, bool):
            raise TypeError("log must be a bool")
        if self.low > self.high or self.step <= 0:
            raise ValueError("invalid integer bounds or step")
        if (self.high - self.low) % self.step != 0:
            raise ValueError("integer upper bound must align with step; refusing truncation")
        if self.log and (self.low <= 0 or self.step != 1):
            raise ValueError("logarithmic integer range requires low > 0 and step=1")

    def validate(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("integer parameter requires an integer; refusing implicit rounding")
        number = int(value)
        if not self.low <= number <= self.high or (number - self.low) % self.step:
            raise ValueError(f"integer value {number} is outside allowed lattice")
        return number


@dataclass(frozen=True, slots=True)
class Categorical:
    """Explicit ordered choices. No array-valued or mutable choice objects."""

    choices: tuple[str | int | float | bool | None, ...]

    def __post_init__(self) -> None:
        raw = self.choices
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("categorical choices must be a non-string sequence")
        values = tuple(raw)
        if not values:
            raise ValueError("categorical choices must be nonempty")
        for choice in values:
            if choice is not None and not isinstance(choice, (str, int, float, bool)):
                raise TypeError("categorical choices must be scalar, JSON-compatible values")
            if isinstance(choice, float) and not isfinite(choice):
                raise ValueError("categorical values must be finite")
        if len({(type(v), v) for v in values}) != len(values):
            raise ValueError("duplicate categorical choices")
        object.__setattr__(self, "choices", values)

    def validate(self, value: Any) -> str | int | float | bool | None:
        if not any(type(value) is type(choice) and value == choice for choice in self.choices):
            raise ValueError(f"categorical value {value!r} is not an allowed choice")
        return value


Parameter = Continuous | Integer | Categorical


@dataclass(frozen=True)
class ParameterSpace:
    """Path-addressed, immutable mapping of parameter names to distributions.

    Path semantics are interpreted by :func:`apply_scene_parameters` rather than
    the optimizer. This also allows future tuning of non-scene configurations.
    """

    parameters: Mapping[str, Parameter]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("parameter space must be nonempty")
        cloned = dict(self.parameters)
        for path, param in cloned.items():
            if not isinstance(path, str) or not path:
                raise ValueError("parameter names must be nonempty strings")
            if not isinstance(param, (Continuous, Integer, Categorical)):
                raise TypeError(f"invalid parameter definition for {path!r}")
        object.__setattr__(self, "parameters", MappingProxyType(cloned))

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if set(values) != set(self.parameters):
            missing = sorted(set(self.parameters) - set(values))
            extra = sorted(set(values) - set(self.parameters))
            raise ValueError(f"parameter mismatch: missing={missing}, extra={extra}")
        return {path: param.validate(values[path]) for path, param in self.parameters.items()}


def apply_scene_parameters(base: Any, space: ParameterSpace, values: Mapping[str, Any]) -> Any:
    """Return a new public ``Scene`` using the established ``Sweep`` path resolver.

    The first implementation intentionally restricts optimization to physical
    configuration, not numerical accuracy, execution policy or presentation.
    """
    from ..campaign.model import Scene, Sweep

    if not isinstance(base, Scene):
        raise TypeError("base must be a PyRITE Scene")
    for path in space.parameters:
        if path.split(".", 1)[0].split("[", 1)[0] not in (
            "beam", "target", "detector", "filters", "acquisition", "physics", "quantum"
        ):
            raise ValueError(f"{path!r} is not a physical Scene parameter path")
    validated = space.validate(values)
    return Sweep(base=base, axes={key: (value,) for key, value in validated.items()}).expand()[0][1]
