"""Deterministic identity for the *definition* of a persisted optimization study.

This is intentionally NOT PyRITE's physics case-content key. It protects a
study's meaning (scene, search domain, objectives, constraints and user-named
scorer), while case keys continue to control physical-result reuse.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .space import Categorical, Continuous, Integer, ParameterSpace
from .study import Constraint, Objective

SCHEMA_VERSION = 1


def _canonical(value: Any) -> Any:
    """JSON-compatible, type-preserving snapshot; reject unknown/unstable objects.

    Avoid ``repr`` and ``pickle``: neither is a safe cross-process definition ID.
    An unsupported scene field fails closed rather than permitting unsafe resume.
    """
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical(value.value),
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("study definition contains nonfinite values")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be used in study identities")
        return {
            "ndarray_dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _canonical(value.tolist()),
        }
    if isinstance(value, Path):
        return {"path": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("study identity mapping keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported study identity type: {type(value).__name__}")


def study_definition_digest(
    base: Any,
    parameters: ParameterSpace,
    objectives: tuple[Objective, ...],
    constraints: tuple[Constraint, ...],
    *,
    evaluation_id: str,
    base_seed: int | None = None,
) -> str:
    """Hash semantics that must match before an existing study can be resumed.

    ``evaluation_id`` is a **user-controlled version for the scorer** and its
    analysis settings (e.g. ``flux-490-510eV-v2``). Arbitrary Python callback
    behavior cannot be reliably hashed. Bump this ID when the scorer changes.
    """
    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ValueError("durable studies require a nonempty evaluation_id")
    parameters = parameters.bind(base)
    specs: dict[str, Any] = {}
    for name, parameter in parameters.parameters.items():
        if not isinstance(parameter, (Continuous, Integer, Categorical)):
            raise TypeError(f"unsupported distribution {name}")
        specs[name] = _canonical(parameter)
        if name in parameters._quantity_units:
            specs[name]["unit"] = parameters._quantity_units[name]
    body = {
        "schema": SCHEMA_VERSION,
        "base": _canonical(base),
        "parameters": specs,
        "objectives": [_canonical(x) for x in objectives],
        "constraints": [_canonical(x) for x in constraints],
        "evaluation_id": evaluation_id,
        "base_seed": base_seed,
    }
    encoded = json.dumps(
        body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
