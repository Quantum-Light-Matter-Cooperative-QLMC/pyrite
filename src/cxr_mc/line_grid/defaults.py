"""Tool-owned persistent defaults for `cxr line-grid` (NOT the material catalog).

Stored at src/cxr_mc/data/line_grid_defaults.toml so derive/submit read a single
source of standing defaults that `--set-default` / `defaults --set` update. Empty
`tilts`/`azimuths` mean "use each material's profile tilt_deg/tilt_azim_deg" --
today's behavior -- so a missing file reproduces current output exactly.
"""

from __future__ import annotations

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

    lines = ["# managed by `cxr line-grid defaults --set`; edit via CLI, not by hand", ""]
    for key in ("tilts", "azimuths", "thickness_ang", "energies", "materials"):
        items = ", ".join(scalar(x) for x in values[key])
        lines.append(f"{key} = [{items}]")
    lines.append(f"brem_step_ev = {values['brem_step_ev']!r}")
    lines.append("")
    return "\n".join(lines)


def update_defaults(**changes) -> dict:
    values = load_defaults()
    for key, val in changes.items():
        if val is None:
            continue
        if key not in FALLBACK:
            raise KeyError(f"unknown default {key!r}")
        values[key] = val
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
