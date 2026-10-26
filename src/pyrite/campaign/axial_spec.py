"""Versioned, JSON-safe experimental axial run configuration (#624).

Pure configuration resolver shared by the public ``Physics`` facade and profile
lowering. This module intentionally does *not* admit a production axial run:
the physical handoff/depth budget and photon source in #623 remain unverified.

No discrete FFT sink, periodic wrap or above-barrier population is interpreted
as physical escape here. Scientific identity excludes execution-only ceilings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from math import gcd
from numbers import Integral, Real
from typing import Any

from .._channeling_admission import refuse_unverified_axial_run as refuse_unverified_axial_run

AXIAL_MODEL_MARKER = "axial-entrance-experimental-config-v1"
AXIAL_SCHEMA_VERSION = 1
_PENDING = {"version": 1, "kind": "pending-physical-validation"}
_DEFAULT_LIMITS = {
    "max_grid_points": 65536,
    "max_steps": 100000,
    "max_ensemble_electrons": 4096,
    "max_working_bytes": 2 * 1024**3,
}


def _fields(value: object, name: str, required: set[str], optional: set[str]) -> Mapping:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    missing = required - value.keys()
    extra = value.keys() - (required | optional)
    if missing or extra:
        raise ValueError(f"{name}: missing {sorted(missing)}; unknown {sorted(extra)}")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _number(
    value: object, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative' if nonnegative else 'numeric'}"
        )
    return result


def _vector(value: object, name: str, length: int, *, integer: bool = False) -> list:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} elements")
    fn = _integer if integer else _number
    return [fn(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _axis(value: object) -> list[int]:
    uvw = _vector(value, "axis_uvw", 3, integer=True)
    divisor = gcd(*uvw)
    if divisor == 0:
        raise ValueError("axis_uvw must be nonzero")
    return [value // divisor for value in uvw]  # retain signed direction


def _entry(raw: object, *, depth: int = 0) -> dict[str, Any]:
    """Canonical 2D pure or incoherent mixture recipe, never an executable state."""
    if depth > 4:
        raise ValueError("entry mixture nesting exceeds four levels")
    head = _fields(
        raw,
        "entry",
        {"version", "kind"},
        {
            "dimension",
            "waist_ang",
            "impact_ang",
            "tilt_offset_rad",
            "m",
            "n",
            "p",
            "ell",
            "radial_k_inv_ang",
            "components",
            "impact_average",
            "divergence",
        },
    )
    if _integer(head["version"], "entry.version") != 1:
        raise ValueError("unsupported entry recipe version")
    kind = head["kind"]
    if not isinstance(kind, str):
        raise ValueError("entry.kind must be a string")
    if kind == "mixture":
        _fields(head, "mixture", {"version", "kind", "components"}, set())
        rows = head["components"]
        if not isinstance(rows, (tuple, list)) or not rows:
            raise ValueError("mixture.components must be a nonempty list")
        components: list[dict[str, Any]] = []
        for row in rows:
            part = _fields(row, "mixture component", {"weight", "entry"}, set())
            weight = _number(part["weight"], "mixture weight", positive=True)
            child = _entry(part["entry"], depth=depth + 1)
            components.append({"weight": weight, "entry": child})
        try:
            total = math.fsum(x["weight"] for x in components)
        except OverflowError as exc:
            raise ValueError("mixture weights must have a finite sum") from exc
        if not math.isfinite(total) or total <= 0:
            raise ValueError("mixture weights must sum to a finite positive value")
        for part in components:
            part["weight"] /= total
            if part["weight"] == 0:
                raise ValueError("mixture weight underflow would erase a component")
        return {"version": 1, "kind": "mixture", "components": components}

    if kind not in {"gaussian", "hermite_gauss", "laguerre_gauss", "bessel_gauss"}:
        raise ValueError(f"unsupported axial entry kind {kind!r}")
    required = {"version", "kind", "waist_ang"}
    optional = {"dimension", "impact_ang", "tilt_offset_rad", "impact_average", "divergence"}
    if kind == "hermite_gauss":
        required |= {"m", "n"}
    elif kind == "laguerre_gauss":
        required |= {"ell", "p"}
    elif kind == "bessel_gauss":
        required |= {"ell", "radial_k_inv_ang"}
    _fields(head, f"{kind} entry", required, optional)
    if _integer(head.get("dimension", 2), "entry.dimension") != 2:
        raise ValueError("axial standard-run entry must have dimension=2")
    if head.get("impact_average", "fixed") != "fixed":
        raise ValueError("weighted impact quadrature is not yet supported in axial runs")
    if head.get("divergence", "zero") != "zero":
        raise ValueError("weighted divergence quadrature is not yet supported in axial runs")
    result: dict[str, Any] = {
        "version": 1,
        "kind": kind,
        "dimension": 2,
        "waist_ang": _number(head["waist_ang"], "entry.waist_ang", positive=True),
        "impact_ang": _vector(head.get("impact_ang", [0.0, 0.0]), "entry.impact_ang", 2),
        "tilt_offset_rad": _vector(
            head.get("tilt_offset_rad", [0.0, 0.0]), "entry.tilt_offset_rad", 2
        ),
        "impact_average": "fixed",
        "divergence": "zero",
    }
    if kind == "hermite_gauss":
        result.update(
            m=_integer(head["m"], "entry.m", minimum=0), n=_integer(head["n"], "entry.n", minimum=0)
        )
    if kind == "laguerre_gauss":
        result.update(
            ell=_integer(head["ell"], "entry.ell"), p=_integer(head["p"], "entry.p", minimum=0)
        )
    if kind == "bessel_gauss":
        result.update(
            ell=_integer(head["ell"], "entry.ell"),
            radial_k_inv_ang=_number(
                head["radial_k_inv_ang"], "entry.radial_k_inv_ang", nonnegative=True
            ),
        )
    return result


def _pending(raw: object, name: str) -> dict[str, Any]:
    value = _fields(raw, name, {"version", "kind"}, set())
    if _integer(value["version"], f"{name}.version") != 1 or value["kind"] != _PENDING["kind"]:
        raise ValueError(
            f"{name}: physical model has not been validated (#623); only explicit pending-physical-validation is permitted"
        )
    return dict(_PENDING)


def resolve_axial_spec(raw: Mapping[str, object]) -> dict[str, Any]:
    """Normalize one proposed axial recipe, without making it run-eligible.

    Inputs are JSON/TOML-compatible primitive collections; no arrays, opaque
    state callables or implicit physical boundary/source model are admitted.
    The returned mapping is detached, canonical and JSON-safe, appropriate
    for a future case identity. Execution limits are checked but not physics.
    """
    inp = _fields(
        raw,
        "channeling_axial",
        {"axis_uvw", "entry", "zone_max_ang"},
        {
            "version",
            "temperature_K",
            "cells",
            "shape",
            "dz_ang",
            "ensemble_electrons",
            "seed",
            "mechanisms",
            "boundary",
            "source",
            "limits",
            "model",
        },
    )
    if inp.get("model", "axial") != "axial":
        raise ValueError("channeling_axial.model must be 'axial'")
    if _integer(inp.get("version", 1), "channeling_axial.version") != 1:
        raise ValueError("unsupported axial configuration version")
    cells = _vector(inp.get("cells", [1, 1]), "cells", 2, integer=True)
    shape = _vector(inp.get("shape", [16, 16]), "shape", 2, integer=True)
    for i, (c, n) in enumerate(zip(cells, shape, strict=True)):
        if c < 1 or n < 2 or n % 2 or n % c:
            raise ValueError(f"axis {i}: shape must be even and divisible by positive cells")
    mechanisms = inp.get("mechanisms", ["tds", "electronic"])
    if not isinstance(mechanisms, (tuple, list)) or (
        len(mechanisms) != 2
        or sorted(map(str, mechanisms)) != ["electronic", "tds"]
        or any(not isinstance(value, str) for value in mechanisms)
    ):
        raise ValueError("standard axial candidate requires exactly tds and electronic mechanisms")
    limits = _fields(inp.get("limits", _DEFAULT_LIMITS), "limits", set(), set(_DEFAULT_LIMITS))
    ceilings = {
        name: _integer(limits.get(name, default), f"limits.{name}", minimum=1)
        for name, default in _DEFAULT_LIMITS.items()
    }
    dz = _number(inp.get("dz_ang", 5.0), "dz_ang", positive=True)
    zone = _number(inp["zone_max_ang"], "zone_max_ang", positive=True)
    count = _integer(inp.get("ensemble_electrons", 64), "ensemble_electrons", minimum=1)
    ratio = zone / dz
    if not math.isfinite(ratio) or ratio > ceilings["max_steps"]:
        raise ValueError("axial configuration exceeds declared step admission ceiling")
    steps = math.ceil(ratio)
    points = shape[0] * shape[1]
    if (
        points > ceilings["max_grid_points"]
        or steps > ceilings["max_steps"]
        or count > ceilings["max_ensemble_electrons"]
    ):
        raise ValueError(
            "axial configuration exceeds declared grid/step/ensemble admission ceilings"
        )
    # A conservative minimum for complex128 propagation and per-depth summary.
    if 256 * points + 32 * points * min(count, 128) > ceilings["max_working_bytes"]:
        raise ValueError("axial configuration exceeds minimum working-memory admission ceiling")
    return {
        "version": AXIAL_SCHEMA_VERSION,
        "model": "axial",
        "axis_uvw": _axis(inp["axis_uvw"]),
        "temperature_K": _number(inp.get("temperature_K", 293.15), "temperature_K", positive=True),
        "entry": _entry(inp["entry"]),
        "cells": cells,
        "shape": shape,
        "dz_ang": dz,
        "zone_max_ang": zone,
        "ensemble_electrons": count,
        "seed": _integer(inp.get("seed", 0), "seed", minimum=0),
        "mechanisms": ["tds", "electronic"],
        "boundary": _pending(inp.get("boundary", _PENDING), "boundary"),
        "source": _pending(inp.get("source", _PENDING), "source"),
        "limits": ceilings,
    }


def axial_spec_identity(spec: Mapping[str, object]) -> str:
    """Hash scientifically relevant canonical fields, excluding refusal ceilings.

    This *candidate* identity is not a valid transport cache key or a claim of
    model readiness. Material fingerprints and finalized physical model tags
    will additionally be required before production results can be cached.
    """
    raw = dict(spec)
    if raw.pop("model", "axial") != "axial":
        raise ValueError("expected an axial resolved specification")
    normalized = resolve_axial_spec(raw)
    payload = {k: v for k, v in normalized.items() if k != "limits"}
    encoded = json.dumps(
        {"marker": AXIAL_MODEL_MARKER, "spec": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
