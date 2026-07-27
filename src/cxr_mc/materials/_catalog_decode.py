"""Primitive decoders for the declarative material catalog schema."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import numpy as np

GridValue = int | float | Mapping[str, object]
LineGridByEnergy = Mapping[float, np.ndarray]
_GRID_KINDS = frozenset({"values", "arange", "linspace", "logspace"})


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}")

    def keys(self, raw: Mapping[str, object], path: str, allowed: set[str]) -> None:
        for key in raw:
            if key not in allowed:
                self.add(f"{path}.{key}", "unknown key")


def _readonly(values: object) -> np.ndarray:
    contiguous = np.asarray(values, dtype=np.float64).reshape(-1)
    # ``bytes`` owns immutable storage, unlike ``flags.writeable = False`` on
    # an owning ndarray, whose caller can simply re-enable writes.
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _descriptor_number(payload: Mapping[str, object], key: str) -> float:
    value = _number(payload.get(key))
    if value is None:
        raise ValueError(f"{key} must be a finite number (not bool)")
    return value


def _descriptor_num(payload: Mapping[str, object]) -> int:
    value = payload.get("num")
    if type(value) is not int or value <= 0:
        raise ValueError("num must be a positive integer")
    return value


def _descriptor_endpoint(payload: Mapping[str, object]) -> bool:
    value = payload.get("endpoint", True)
    if type(value) is not bool:
        raise ValueError("endpoint must be a boolean")
    return value


def _direction(value: object, path: str, errors: _Errors) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(v, int) or isinstance(v, bool) for v in value)
    ):
        errors.add(path, "must be three integer components")
        return None
    out = cast(tuple[int, int, int], tuple(value))
    if out == (0, 0, 0):
        errors.add(path, "must be nonzero")
        return None
    return out


def _grid(value: object, path: str, errors: _Errors) -> np.ndarray | None:
    scalar = _number(value)
    if scalar is not None:
        out = _readonly([scalar])
    elif isinstance(value, Mapping):
        value = cast(Mapping[str, object], value)
        kinds = [key for key in value if key in _GRID_KINDS]
        unknown = [key for key in value if key not in _GRID_KINDS]
        for key in unknown:
            errors.add(f"{path}.{key}", "unknown grid descriptor")
        if len(kinds) != 1 or len(value) != 1:
            errors.add(path, "must contain exactly one of values/arange/linspace/logspace")
            return None
        kind = kinds[0]
        payload = value[kind]
        try:
            if kind == "values":
                if not isinstance(payload, list):
                    raise ValueError("values must be an array")
                numeric = [_number(item) for item in payload]
                if any(item is None for item in numeric):
                    raise ValueError("values entries must be finite numbers (not bool)")
                out = _readonly(numeric)
            else:
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{kind} must be a table")
                payload = cast(Mapping[str, object], payload)
                allowed = {
                    "arange": {"start", "stop", "step"},
                    "linspace": {"start", "stop", "num", "endpoint"},
                    "logspace": {"start", "stop", "num", "endpoint", "base"},
                }[kind]
                extra = set(payload) - allowed
                if extra:
                    raise ValueError(f"unknown keys {sorted(extra)}")
                if kind == "arange":
                    if set(payload) != {"start", "stop", "step"}:
                        raise ValueError("arange requires start, stop, and step")
                    start = _descriptor_number(payload, "start")
                    stop = _descriptor_number(payload, "stop")
                    step = _descriptor_number(payload, "step")
                    if step == 0:
                        raise ValueError("step must be nonzero")
                    out = _readonly(np.arange(start, stop, step))
                else:
                    required = {"start", "stop", "num"}
                    if not required <= set(payload):
                        raise ValueError(f"{kind} requires start, stop, and num")
                    start = _descriptor_number(payload, "start")
                    stop = _descriptor_number(payload, "stop")
                    num = _descriptor_num(payload)
                    endpoint = _descriptor_endpoint(payload)
                    if kind == "logspace":
                        base = _descriptor_number(payload, "base") if "base" in payload else 10.0
                        generated = np.logspace(start, stop, num, endpoint=endpoint, base=base)
                    else:
                        generated = np.linspace(start, stop, num, endpoint=endpoint)
                    out = _readonly(generated)
        except (FloatingPointError, OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.add(path, f"invalid {kind} descriptor ({exc})")
            return None
    else:
        errors.add(path, "must be a scalar or one grid descriptor")
        return None
    if out.size == 0:
        errors.add(path, "grid must be nonempty")
        return None
    if not np.all(np.isfinite(out)):
        errors.add(path, "grid values must be finite")
        return None
    return out


def _line_grids_by_energy(
    value: object,
    energy_grid: np.ndarray | None,
    path: str,
    errors: _Errors,
    *,
    fallback_energies: frozenset[float] = frozenset(),
) -> LineGridByEnergy | None:
    """Decode a per-energy line-grid table, checking it covers ``energy_grid``.

    ``fallback_energies`` is the P0.3 interim fix (subitem 5,
    docs/cli-energy-grid-sweep-rework-plan.md): a beam energy missing a local
    line-grid row is not an error if it's already covered there (typically the
    ``standard`` source profile's own rows). The structural fix -- a shared
    per-material derived-grid store profiles merely reference -- is Phase 1.
    """
    if not isinstance(value, list) or not value:
        errors.add(path, "must be a nonempty array of line-grid entries")
        return None
    parsed: dict[float, np.ndarray] = {}
    seen_energies: set[float] = set()
    energy_indexes: dict[float, int] = {}
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        row = _table(item, item_path, errors)
        if row is None:
            continue
        errors.keys(row, item_path, {"energy_keV", "grid"})
        energy = _number(row.get("energy_keV"))
        if energy is None or energy <= 0:
            errors.add(f"{item_path}.energy_keV", "must be finite and positive")
            continue
        if energy in seen_energies:
            errors.add(f"{item_path}.energy_keV", f"duplicates beam energy {energy:g}")
            continue
        seen_energies.add(energy)
        energy_indexes[energy] = index
        grid = _grid(row.get("grid"), f"{item_path}.grid", errors)
        if grid is None or np.any(grid <= 0):
            if grid is not None:
                errors.add(f"{item_path}.grid", "values must be positive")
            continue
        parsed[energy] = grid
    if energy_grid is not None:
        configured = set(float(value) for value in energy_grid)
        mapped = set(parsed)
        missing = sorted(configured - mapped - fallback_energies)
        extra = sorted(mapped - configured)
        if missing:
            errors.add(path, f"missing beam energies {missing}")
        for energy in extra:
            index = energy_indexes[energy]
            errors.add(
                f"{path}[{index}].energy_keV",
                f"is not configured in energy_keV: {energy:g}",
            )
    return MappingProxyType(parsed) if parsed else None


def _table(value: object, path: str, errors: _Errors) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.add(path, "must be a table")
        return None
    return cast(Mapping[str, object], value)


__all__ = ["GridValue", "LineGridByEnergy"]
