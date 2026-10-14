"""Deprecated derivation batch presets, retained until 0.15.0.

Stored at src/pyrite/data/line_grid_defaults.toml so local/remote derive read a single
source of legacy inputs that `derive --save-default` and
`profile energy-grid defaults --save-default` update until removal. Empty
`tilts`/`azimuths` mean "use each material's profile tilt_deg/tilt_azim_deg" --
today's behavior -- so a missing file reproduces current output exactly.
"""

import math
import os
import tempfile
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "line_grid_defaults.toml"
DEPRECATED_IN = "0.13.6"
REMOVE_IN = "0.15.0"
LEGACY_WARNING = (
    "warning: persistent energy-grid derivation defaults are deprecated and will be "
    f"removed in {REMOVE_IN}; normal runs size grids automatically. For optional "
    "derivation, pass --material and --energy explicitly, plus per-invocation "
    "--polar, --azimuth, --thickness and --brem-step as needed."
)
_warned = ContextVar("energy_grid_defaults_warned", default=False)


@contextmanager
def legacy_warning_scope():
    """Deduplicate warnings across a CLI invocation and its internal forwarding."""
    token = _warned.set(_warned.get())
    try:
        yield
    finally:
        _warned.reset(token)


def warn_legacy_defaults(*, already_warned=False):
    """Warn when a derivation consumes or writes the retiring batch presets."""
    if _warned.get():
        return
    _warned.set(True)
    if already_warned:
        return
    from pyrite.console.output import emit_diagnostic

    emit_diagnostic(LEGACY_WARNING)


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

    lines = [
        "# Deprecated derivation batch presets; remove in 0.15.0.",
        "# managed by `pyrite profile energy-grid defaults --save-default`",
        "",
    ]
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
