"""Validated EEDL electron-impact subshell data shared by transport and scoring."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from endf_parserpy import EndfFile

from .. import DATA_DIR
from ..materials.atomic import Z_TABLE

EEDL_DATA_DIR = DATA_DIR / "characteristic_cross_sections"
EEDL_FILENAME = "EEDL.endf"
EEDL_SHA256 = "f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c"

# ENDF-6 MF=23 subshell order: designator = MT - 533, with O8/O9 before P1.
EEDL_SUBSHELL_LABELS = {
    1: "K",
    2: "L1",
    3: "L2",
    4: "L3",
    5: "M1",
    6: "M2",
    7: "M3",
    8: "M4",
    9: "M5",
    10: "N1",
    11: "N2",
    12: "N3",
    13: "N4",
    14: "N5",
    15: "N6",
    16: "N7",
    17: "O1",
    18: "O2",
    19: "O3",
    20: "O4",
    21: "O5",
    22: "O6",
    23: "O7",
    24: "O8",
    25: "O9",
    26: "P1",
    27: "P2",
    28: "P3",
    29: "P4",
    30: "P5",
    31: "P6",
    32: "P7",
    33: "P8",
    34: "P9",
    35: "P10",
    36: "P11",
    37: "Q1",
    38: "Q2",
    39: "Q3",
}


@dataclass(frozen=True, slots=True)
class EEDLSubshellTable:
    """One validated EEDL MF=23 subshell-ionization TAB1 table."""

    shell_designator: int
    binding_energy_eV: float
    projectile_energy_eV: np.ndarray
    cross_section_cm2: np.ndarray


def _readonly(array: object) -> np.ndarray:
    out = np.asarray(array, dtype=float)
    out.setflags(write=False)
    return out


def _require_finite(value: object, name: str, *, positive: bool = False) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def _section_vector(
    section: Mapping[str, object],
    field: str,
    path: Path,
    mt: int,
) -> np.ndarray:
    """Copy one endf-parserpy section vector into a finite float array."""
    try:
        values = np.asarray(section[field], dtype=float)
    except KeyError as exc:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} is missing {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} has invalid {field} values") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"{path}: EEDL MF=23/MT={mt} {field} must be a finite one-dimensional array"
        )
    return values


def _extract_eedl_subshell(
    path: Path,
    mt: int,
    section: Mapping[str, object],
) -> EEDLSubshellTable:
    """Validate one MF=23 section already decoded by endf-parserpy."""
    try:
        binding = _require_finite(
            section["EPE"],
            f"{path}: EEDL MF=23/MT={mt} EPE binding energy",
            positive=True,
        )
    except KeyError as exc:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} is missing EPE") from exc

    projectile = _section_vector(section, "Eint", path, mt)
    cross_section_barn = _section_vector(section, "sigma", path, mt)
    breakpoints_float = _section_vector(section, "NBT", path, mt)
    laws_float = _section_vector(section, "INT", path, mt)
    if projectile.size != cross_section_barn.size or projectile.size < 3:
        raise ValueError(
            f"{path}: EEDL MF=23/MT={mt} must contain at least three paired Eint/sigma samples"
        )
    if breakpoints_float.size == 0 or breakpoints_float.size != laws_float.size:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} has invalid NBT/INT arrays")
    breakpoints = np.rint(breakpoints_float).astype(int)
    laws = np.rint(laws_float).astype(int)
    if not np.array_equal(breakpoints_float, breakpoints) or not np.array_equal(laws_float, laws):
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} NBT/INT values must be integers")
    if breakpoints[-1] != projectile.size or np.any(np.diff(breakpoints) <= 0):
        raise ValueError(f"{path}: invalid EEDL MF=23/MT={mt} interpolation breakpoints")
    if np.any(laws != 2):
        raise ValueError(
            f"{path}: only ENDF interpolation law 2 (lin-lin) is supported for EEDL "
            f"cross sections; MF=23/MT={mt} declares {laws.tolist()}"
        )
    if np.any(projectile <= 0.0) or np.any(np.diff(projectile) <= 0.0):
        raise ValueError(
            f"{path}: EEDL MF=23/MT={mt} projectile energies must be positive and "
            "strictly increasing"
        )
    if np.any(cross_section_barn < 0.0):
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} cross sections must be non-negative")
    if np.any(cross_section_barn[projectile <= binding] > 0.0):
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} cross section is nonzero at/below EPE")
    if np.count_nonzero(cross_section_barn > 0.0) < 2:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} needs at least two positive samples")

    return EEDLSubshellTable(
        shell_designator=mt - 533,
        binding_energy_eV=binding,
        projectile_energy_eV=_readonly(projectile),
        cross_section_cm2=_readonly(cross_section_barn * 1.0e-24),
    )


@cache
def _load_eedl_subshell_tables(
    path: Path,
    file_size: int,
    modified_time_ns: int,
    atomic_number: int,
    element: str,
) -> tuple[EEDLSubshellTable, ...]:
    """Use endf-parserpy to load one element's EEDL subshell sections.

    ``file_size`` and ``modified_time_ns`` are cache-key sentinels. They ensure
    that replacing an explicitly supplied EEDL file in place invalidates the
    cached parsed tables.
    """
    del file_size, modified_time_ns
    saw_eedl_subshell = False
    saw_eadl_relaxation = False
    matching_materials: list[tuple[int, tuple[int, ...], Any]] = []

    with EndfFile(path, on_error="raise") as tape:
        for position in range(len(tape)):
            material = tape[position]
            section_ids = tuple(material.sections())
            subshell_mts = tuple(
                sorted(mt for mf, mt in section_ids if mf == 23 and 534 <= mt <= 572)
            )
            saw_eedl_subshell = saw_eedl_subshell or bool(subshell_mts)
            saw_eadl_relaxation = saw_eadl_relaxation or (28, 533) in section_ids
            section_atomic_number = round(float(material.za) / 1000.0)
            if section_atomic_number == atomic_number and subshell_mts:
                matching_materials.append((int(material.mat), subshell_mts, material))

        if len(matching_materials) > 1:
            mats = [mat for mat, _mts, _material in matching_materials]
            raise ValueError(
                f"{path}: multiple EEDL materials contain MF=23 subshell data "
                f"for {element} (Z={atomic_number}): MAT={mats}"
            )
        if matching_materials:
            _mat, subshell_mts, material = matching_materials[0]
            return tuple(_extract_eedl_subshell(path, mt, material[23, mt]) for mt in subshell_mts)

    if saw_eadl_relaxation and not saw_eedl_subshell:
        raise ValueError(
            f"{path}: contains EADL MF=28/MT=533 atomic-relaxation data, not "
            "electron-impact ionization cross sections; provide the companion EEDL "
            "electro-atomic file with MF=23/MT=534-572"
        )
    raise ValueError(
        f"{path}: no electroionization subshell sections for {element} (Z={atomic_number}); "
        "expected EEDL MF=23/MT=534-572"
    )


@cache
def _verify_packaged_eedl() -> None:
    """Fail closed if the packaged EEDL bytes differ from the vetted input."""
    path = EEDL_DATA_DIR / EEDL_FILENAME
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        raise FileNotFoundError(f"packaged EEDL database is missing: expected {path}") from None
    actual = digest.hexdigest()
    if actual != EEDL_SHA256:
        raise ValueError(
            f"packaged EEDL database checksum mismatch: expected {EEDL_SHA256}, got {actual}"
        )


def load_eedl_shell_ionization(
    element: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[EEDLSubshellTable, ...]:
    """Load EEDL electron-impact subshell rates without relaxation data.

    The packaged EPICS2025 tape is checksum-pinned; ``data_dir`` may instead
    name an explicit ENDF-6 file or directory. Each shell keeps its native
    projectile-energy grid, EEDL binding energy, and cross section in cm².
    This supplies total ionization rates, not a differential transfer law.
    """
    if not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None:
        raise ValueError("element must be a chemical symbol such as 'C' or 'Si'")
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    candidate = EEDL_DATA_DIR if data_dir is None else Path(data_dir)
    path = candidate if candidate.is_file() else candidate / EEDL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no EEDL shell-ionization database for {element}: expected {path}")
    if data_dir is None:
        _verify_packaged_eedl()
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    return _load_eedl_subshell_tables(
        resolved_path,
        stat.st_size,
        stat.st_mtime_ns,
        atomic_number,
        element,
    )
