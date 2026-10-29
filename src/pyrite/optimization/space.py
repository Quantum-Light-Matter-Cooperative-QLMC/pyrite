"""Typed, backend-neutral parameter definitions for optimization studies.

Parameter values describe *effective* scene values. No implicit rounding,
clipping or logarithmic transformations are applied when accepting candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from .._units import _is_pint_quantity


def _quantity_magnitude(value: Any, unit: Any, *, name: str) -> float:
    import numpy as np

    from .._units import _root_unit_powers

    if not _is_pint_quantity(value):
        raise TypeError(f"{name} requires a Pint quantity")
    if _root_unit_powers(value) != _root_unit_powers(unit):
        raise ValueError(f"{name} has incompatible unit {value.units!s}; expected {unit.units!s}")
    magnitude = np.asarray(value.to(str(unit.units)).magnitude)
    if magnitude.ndim != 0 or magnitude.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar quantity")
    result = _finite_real(magnitude.item(), name=name)
    if result == 0 and magnitude != 0:
        raise ValueError(f"{name} is not representable in the owner's unit")
    if magnitude.dtype.kind in "iu" and int(result) != magnitude.item():
        raise ValueError(f"{name} is not exactly representable in numeric storage")
    return result


def _interval_values(low: Any, high: Any) -> tuple[float, float]:
    if _is_pint_quantity(low) or _is_pint_quantity(high):
        reference = low if _is_pint_quantity(low) else high
        return (
            _quantity_magnitude(low, reference, name="low"),
            _quantity_magnitude(high, reference, name="high"),
        )
    return _finite_real(low, name="low"), _finite_real(high, name="high")


def _refuse_quantity(value: Any, *, name: str) -> None:
    # Legacy paths have no semantic quantity property declaring their unit.
    if _is_pint_quantity(value):
        raise TypeError(
            f"{name} must be a plain number in the addressed field's unit; "
            f"legacy paths declare no unit, so Pint quantities (received {str(value.units)!r}) "
            "are refused rather than lowered by guess"
        )


def _finite_real(value: Any, *, name: str) -> float:
    _refuse_quantity(value, name=name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Continuous:
    """Continuous interval; quantity bounds require binding to a semantic path.

    Both bounds must be quantities or both plain numbers. Quantity intervals
    are converted to the owning property unit by ``ParameterSpace.bind``.
    ``log=True`` requires a positive interval.
    """

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        lo, hi = _interval_values(self.low, self.high)
        if not isinstance(self.log, bool):
            raise TypeError("log must be a bool")
        if lo >= hi:
            raise ValueError("continuous low must be less than high")
        if self.log and lo <= 0:
            raise ValueError("logarithmic parameters must be positive")
        if not _is_pint_quantity(self.low):
            object.__setattr__(self, "low", lo)
            object.__setattr__(self, "high", hi)
        else:
            object.__setattr__(self, "low", copy(self.low))
            object.__setattr__(self, "high", copy(self.high))

    def validate(self, value: Any) -> float:
        if _is_pint_quantity(self.low):
            raise TypeError("bind quantity distributions to a Scene before validating values")
        number = _finite_real(value, name="continuous value")
        if not self.low <= number <= self.high:
            raise ValueError(f"continuous value {number} outside [{self.low}, {self.high}]")
        return number


@dataclass(frozen=True, slots=True)
class Integer:
    """Inclusive integer lattice in the bound owner's unit, without rounding.

    Quantity bounds and step are checked for integrality after Scene binding.
    For a quantity interval an omitted/plain step is in the owner's unit.
    """

    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        if _is_pint_quantity(self.low) or _is_pint_quantity(self.high):
            lo, hi = _interval_values(self.low, self.high)
            if not isinstance(self.log, bool):
                raise TypeError("log must be a bool")
            if lo > hi:
                raise ValueError("invalid integer bounds or step")
            object.__setattr__(self, "low", copy(self.low))
            object.__setattr__(self, "high", copy(self.high))
            if _is_pint_quantity(self.step):
                _quantity_magnitude(self.step, self.low, name="step")
                object.__setattr__(self, "step", copy(self.step))
            elif isinstance(self.step, bool) or not isinstance(self.step, Integral):
                raise TypeError("integer step must be an integer")
            return
        for key in ("low", "high", "step"):
            value = getattr(self, key)
            _refuse_quantity(value, name=f"integer {key}")
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
        if _is_pint_quantity(self.low):
            raise TypeError("bind quantity distributions to a Scene before validating values")
        _refuse_quantity(value, name="integer parameter")
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
    _quantity_units: Mapping[str, str] = field(default_factory=dict, init=False, repr=False)

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
        object.__setattr__(self, "_quantity_units", MappingProxyType({}))

    def bind(self, base: Any) -> ParameterSpace:
        """Convert quantity domains using each Scene property's own read-back unit.

        Semantic paths require quantity bounds and a ``with_<name>`` method.
        Legacy numeric paths refuse quantity bounds. The returned space stores
        numeric bounds and unit strings; use it when configuring an optimizer.
        Integer bounds/steps must be exactly integral in the owner's unit.
        """
        from ..campaign.model import Scene, _descend, _parts

        if not isinstance(base, Scene):
            raise TypeError("base must be a PyRITE Scene")
        parameters: dict[str, Parameter] = {}
        units: dict[str, str] = {}
        for path, param in self.parameters.items():
            parts = _parts(path)
            owner = base
            for name, indexes in parts[:-1]:
                owner = _descend(owner, name, indexes, path)
            name, indexes = parts[-1]
            current = _descend(owner, name, indexes, path)
            if _is_pint_quantity(current) and callable(getattr(owner, f"with_{name}", None)):
                if not isinstance(param, (Continuous, Integer)):
                    raise TypeError(f"{path}: semantic quantity paths require numeric intervals")
                unit = str(current.units)
                if path in self._quantity_units:
                    if self._quantity_units[path] != unit:
                        raise ValueError(f"{path}: bound owner unit changed")
                    parameters[path] = param
                else:
                    parameters[path] = _bind_interval(param, current)
                units[path] = unit
            else:
                if isinstance(param, (Continuous, Integer)):
                    _refuse_quantity(param.low, name=path)
                    _refuse_quantity(param.high, name=path)
                parameters[path] = param
        bound = ParameterSpace(parameters)
        object.__setattr__(bound, "_quantity_units", MappingProxyType(units))
        return bound

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if set(values) != set(self.parameters):
            missing = sorted(set(self.parameters) - set(values))
            extra = sorted(set(values) - set(self.parameters))
            raise ValueError(f"parameter mismatch: missing={missing}, extra={extra}")
        validated = {}
        for path, param in self.parameters.items():
            value = values[path]
            if path in self._quantity_units and _is_pint_quantity(value):
                from ..units import Q_

                value = _quantity_magnitude(value, Q_(1, self._quantity_units[path]), name=path)
                if isinstance(param, Integer):
                    value = _exact_integer(value, name=path)
            validated[path] = param.validate(value)
        return validated


def _exact_integer(value: float, *, name: str) -> int:
    if not value.is_integer():
        raise ValueError(f"{name} must be integral in the owner's unit; refusing rounding")
    return int(value)


def _bind_interval(param: Continuous | Integer, current: Any) -> Continuous | Integer:
    low = _quantity_magnitude(param.low, current, name="low")
    high = _quantity_magnitude(param.high, current, name="high")
    if isinstance(param, Continuous):
        return Continuous(low, high, log=param.log)
    step = (
        _quantity_magnitude(param.step, current, name="step")
        if _is_pint_quantity(param.step)
        else float(param.step)
    )
    return Integer(
        _exact_integer(low, name="low"),
        _exact_integer(high, name="high"),
        step=_exact_integer(step, name="step"),
        log=param.log,
    )


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
            "beam",
            "target",
            "detector",
            "filters",
            "acquisition",
            "physics",
            "quantum",
        ):
            raise ValueError(f"{path!r} is not a physical Scene parameter path")
    bound = space.bind(base)
    validated = bound.validate(values)
    axes = {}
    for key, value in validated.items():
        if key in bound._quantity_units:
            from ..units import Q_

            value = Q_(value, bound._quantity_units[key])
        axes[key] = (value,)
    return Sweep(base=base, axes=axes).expand()[0][1]
