"""Click parameter types for physical options: bare number or explicit unit string (#810).

A bare number keeps the option's canonical unit and is validated by the wrapped
base type exactly as before. A string carrying a unit (``"0.1 mm"``) is parsed
through :func:`pyrite._units.parse_unit_string`, converted to the canonical
unit, then handed to the same base type, so range checks, their messages, and
Click's help range hint apply to the converted magnitude. Pint stays unimported
until a unit string is seen.

The result is a copy of the base type whose class also mixes in
:class:`_UnitConversion`, so ``isinstance`` checks inside Click (range hints)
and the base type's metavar and name are unchanged.
"""

from __future__ import annotations

from functools import cache
from types import new_class
from typing import cast

import click

from .._units import parse_unit_string


class _UnitConversion(click.ParamType):
    """Mixin converting ``"<number> <unit>"`` to the canonical float before ``convert``."""

    unit: str
    expected: str
    unit_kind: str | None

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            text = value.strip()
            try:
                float(text)
            except ValueError:
                value = self._to_canonical(text, param, ctx)
        return super().convert(value, param, ctx)

    def _to_canonical(self, text, param, ctx):
        field = param.opts[0] if param is not None and param.opts else "value"
        try:
            magnitude = parse_unit_string(
                text, self.unit, field=field, expected=self.expected, kind=self.unit_kind
            )
            return float(magnitude)
        except (ValueError, TypeError) as exc:
            self.fail(f"{exc}; or give a bare number in {self.unit}", param, ctx)


@cache
def _unit_class(base_class: type) -> type[_UnitConversion]:
    return cast(
        "type[_UnitConversion]",
        new_class(f"Unit{base_class.__name__}", (_UnitConversion, base_class)),
    )


def unit_float(
    base: click.ParamType | type,
    unit: str,
    expected: str,
    kind: str | None = None,
) -> click.ParamType:
    """Make ``base`` also accept an explicit unit string.

    ``unit`` is the canonical Pint unit the option stores; ``expected`` names
    the dimension in errors (``"length"``). ``kind`` selects the stricter
    ``normalize_unit`` guards (``"angle"``, ``"solid_angle"``, ``"emittance"``)
    so radians, steradians and bare ratios are never implicitly exchanged.
    Declaration shorthand: ``unit_float(POSITIVE_FLOAT, "mm", "length")``.
    """
    source = click.types.convert_type(base)
    unit_class = _unit_class(type(source))
    clone = unit_class.__new__(unit_class)
    for klass in type(source).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if hasattr(source, slot):
                setattr(clone, slot, getattr(source, slot))
    getattr(clone, "__dict__", {}).update(getattr(source, "__dict__", {}))
    clone.unit = unit
    clone.expected = expected
    clone.unit_kind = kind
    return clone


def unit_pair(
    base: click.ParamType | type, unit: str, expected: str, kind: str | None = None
) -> click.Tuple:
    """Two-component option (``--size-mm X Y``) whose entries each accept units."""
    item = unit_float(base, unit, expected, kind)
    return click.Tuple((item, item))


def canonical_unit(param_type: click.ParamType) -> str | None:
    """Canonical unit of a unit-aware option type, or ``None``."""
    return param_type.unit if isinstance(param_type, _UnitConversion) else None
