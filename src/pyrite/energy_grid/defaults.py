"""Tool-owned persistent defaults for `pyrite energy-grid` (NOT the material catalog).

Stored at src/pyrite/data/line_grid_defaults.toml so local/remote derive read a single
source of standing defaults that `--set-default` / `defaults --set` update. Empty
`tilts`/`azimuths` mean "use each material's profile tilt_deg/tilt_azim_deg" --
today's behavior -- so a missing file reproduces current output exactly.
"""

from __future__ import annotations

import math
import os
import tempfile
import tomllib
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "line_grid_defaults.toml"

FALLBACK: dict = {
    "tilts": [],
    "azimuths": [],
    "thickness_ang": [1.0e7],
    "brem_step_ev": 25.0,
    "energies": [30, 40, 50, 60, 100, 150, 200, 250, 300],
    "materials": ["hopg", "diamond", "wse2", "mose2"],
}


def load_defaults() -> dict:
    merged = dict(FALLBACK)
    if DEFAULTS_PATH.exists():
        with open(DEFAULTS_PATH, "rb") as f:
            merged.update(tomllib.load(f))
    return merged


def _emit(values: dict) -> str:
    def scalar(v):
        if isinstance(v, str):
            return f'"{v}"'
        return repr(v)

    lines = ["# managed by `pyrite energy-grid defaults --set`; edit via CLI, not by hand", ""]
    for key in ("tilts", "azimuths", "thickness_ang", "energies", "materials"):
        items = ", ".join(scalar(x) for x in values[key])
        lines.append(f"{key} = [{items}]")
    lines.append(f"brem_step_ev = {values['brem_step_ev']!r}")
    lines.append("")
    return "\n".join(lines)


def _validate_number_list(values, key: str, predicate, domain: str) -> list[float | int]:
    validated = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{key}[{index}] must satisfy {domain}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}[{index}] must satisfy {domain}") from exc
        if not math.isfinite(number) or not predicate(number):
            raise ValueError(f"{key}[{index}] must satisfy {domain}")
        validated.append(value if type(value) in (int, float) else number)
    return validated


def _validated(values: dict) -> dict:
    out = dict(values)
    out["tilts"] = _validate_number_list(
        values["tilts"], "tilts", lambda value: 0 <= value < 90, "0 <= value < 90"
    )
    out["azimuths"] = _validate_number_list(
        values["azimuths"],
        "azimuths",
        lambda value: 0 <= value <= 360,
        "0 <= value <= 360",
    )
    out["thickness_ang"] = _validate_number_list(
        values["thickness_ang"], "thickness_ang", lambda value: value > 0, "value > 0"
    )
    out["energies"] = _validate_number_list(
        values["energies"], "energies", lambda value: value > 0, "value > 0"
    )
    brem_step = values["brem_step_ev"]
    if isinstance(brem_step, bool):
        raise ValueError("brem_step_ev must be a finite positive number")
    try:
        brem_step = float(brem_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("brem_step_ev must be a finite positive number") from exc
    if not math.isfinite(brem_step) or brem_step <= 0:
        raise ValueError("brem_step_ev must be a finite positive number")
    out["brem_step_ev"] = brem_step
    return out


def update_defaults(**changes) -> dict:
    values = load_defaults()
    for key, val in changes.items():
        if val is None:
            continue
        if key not in FALLBACK:
            raise KeyError(f"unknown default {key!r}")
        values[key] = val
    values = _validated(values)
    text = _emit(values)
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(DEFAULTS_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, DEFAULTS_PATH)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return values


def reset_defaults(*keys: str) -> dict:
    """Reset selected defaults, or every field when no keys are supplied.

    Angle fallbacks are empty lists by design: they mean use each material's
    catalog-profile angles. Other fields return to their built-in values.
    """
    selected = keys or tuple(FALLBACK)
    unknown = [key for key in selected if key not in FALLBACK]
    if unknown:
        raise KeyError(f"unknown default {unknown[0]!r}")
    changes = {
        key: list(FALLBACK[key]) if isinstance(FALLBACK[key], list) else FALLBACK[key]
        for key in selected
    }
    return update_defaults(**changes)
