"""Electron-impact characteristic x rays from EEDL and xraydb data."""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import xraydb
from endf_parserpy import EndfFile

from ... import DATA_DIR
from ..._backend import REAL, _to_cpu, xp
from ...materials.atomic import Z_TABLE
from ...materials.attenuation import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _normalize_composition,
)
from ..groove import escape_distance_ang
from .lines import (
    _clip_segments_to_cutoff,
    _escape_length,
    _observation_direction,
    _segment_escape_distance,
    _validate_groove_escape_direction,
)

CHARACTERISTIC_DATA_DIR = DATA_DIR / "characteristic_cross_sections"
CHARACTERISTIC_EEDL_FILENAME = "EEDL.endf"
CHARACTERISTIC_EEDL_SHA256 = "f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c"
CHARACTERISTIC_ENDF_PARSERPY_VERSION = package_version("endf-parserpy")
CHARACTERISTIC_XRAYDB_VERSION = package_version("xraydb")
CHARACTERISTIC_MODEL = (
    f"eedl-2025-{CHARACTERISTIC_EEDL_SHA256[:12]}/"
    f"endf-parserpy-{CHARACTERISTIC_ENDF_PARSERPY_VERSION}/"
    f"xraydb-{CHARACTERISTIC_XRAYDB_VERSION}-l-shell-ck-lorentzian-v5"
)
_MIN_RELAXATION_CUTOFF_EV = 50.0
CHARACTERISTIC_TRANSPORT_FLOOR_KEV = 1.0
_SHELL_LABELS = {
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
_SHELL_ORDER = {label: code for code, label in _SHELL_LABELS.items()}


@dataclass(frozen=True, slots=True)
class CharacteristicCrossSectionTable:
    """Elemental inner-shell ionization and xraydb relaxation data.

    EEDL subshells may have different incident-energy grids, so
    ``projectile_energy_eV_by_shell`` and
    ``ionization_cross_sections_cm2_by_shell`` contain one read-only array per
    entry in ``ionization_shell_labels``. ``vacancy_transfer`` has shape
    ``(n_ionized_shell, n_ionized_shell)`` and holds the L-shell Coster--Kronig
    vacancy redistribution of :func:`_l_shell_vacancy_transfer`.
    ``line_yield_per_vacancy`` has shape ``(n_ionized_shell, n_line)`` and is
    that transfer applied to xraydb's ``fluorescence_yield *
    conditional_line_intensity``, so row ``i`` is the photons per *primary*
    vacancy in shell ``i`` and may name lines of another L subshell.
    """

    element: str
    atomic_number: int
    recommended_cutoff_eV: float
    projectile_energy_eV_by_shell: tuple[np.ndarray, ...]
    ionization_shell_labels: tuple[str, ...]
    shell_binding_energy_eV: np.ndarray
    shell_fluorescence_yield: np.ndarray
    ionization_cross_sections_cm2_by_shell: tuple[np.ndarray, ...]
    line_labels: tuple[str, ...]
    line_initial_shell: tuple[str, ...]
    line_energy_eV: np.ndarray
    line_fwhm_eV: np.ndarray
    vacancy_transfer: np.ndarray
    line_yield_per_vacancy: np.ndarray


@dataclass(frozen=True, slots=True)
class _EEDLSubshellTable:
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


def _level_width_eV(core_widths: Mapping[str, object], level: str) -> float | None:
    """Resolve one xraydb level width, averaging unresolved level groups."""
    labels = [level]
    if "," in level:
        first, *rest = level.split(",")
        prefix_match = re.match(r"[A-Z]+", first)
        if prefix_match is not None:
            prefix = prefix_match.group(0)
            labels = [first, *(f"{prefix}{suffix}" for suffix in rest)]
    widths = []
    for label in labels:
        value = core_widths.get(label)
        try:
            width = float(str(value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(width) and width >= 0.0:
            widths.append(width)
    return float(np.mean(widths)) if widths else None


def _transition_fwhm_eV(
    core_widths: Mapping[str, object],
    *,
    element: str,
    line_label: str,
    initial_level: str,
    final_level: str,
) -> float:
    """Natural Lorentzian FWHM from the pertinent xraydb hole widths.

    Lorentzian initial- and final-hole widths add. Some xraydb transition
    labels combine unresolved final levels (for example ``M4,5``); their
    available level widths are averaged. A missing final-level width contributes
    zero, while a missing initial width is an error because it would leave the
    transition profile without a physical lifetime scale.

    Validation: characteristic-radiation
    """
    initial_width = _level_width_eV(core_widths, initial_level)
    if initial_width is None or initial_width <= 0.0:
        raise ValueError(
            f"xraydb {element} {line_label} has no positive {initial_level} core-hole width"
        )
    final_width = _level_width_eV(core_widths, final_level)
    fwhm = initial_width + (0.0 if final_width is None else final_width)
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError(f"xraydb {element} {line_label} has an invalid natural linewidth")
    return fwhm


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
) -> _EEDLSubshellTable:
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

    return _EEDLSubshellTable(
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
) -> tuple[_EEDLSubshellTable, ...]:
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


_L_SHELL_CK_ROUTES = (("L1", "L2"), ("L1", "L3"), ("L2", "L3"))


def _l_shell_vacancy_transfer(
    element: str,
    shell_labels: tuple[str, ...],
) -> np.ndarray:
    r"""Expected L-vacancy destinations per primary vacancy, from xraydb CK data.

    Row ``i`` of the returned ``(n_shell, n_shell)`` matrix is the expected
    number of vacancies each subshell holds after one primary vacancy in ``i``
    has undergone Coster--Kronig transfer:

    .. math::

        n_{L1} &= (1 - f_{12} - f_{13})\,N_{L1} \\
        n_{L2} &= (1 - f_{23})\,N_{L2} + f_{12}N_{L1} \\
        n_{L3} &= N_{L3} + f_{23}N_{L2} + f_{13}N_{L1}

    with :math:`f_{ij}` the Elam/Krause Coster--Kronig probabilities exposed by
    ``xraydb.ck_probability``. Those are *total* probabilities: :math:`f_{13}`
    already contains the L1 -> L2 -> L3 route, so it is applied to the primary
    L1 population and :math:`f_{23}` only to the primary L2 population.
    Routing the transferred :math:`f_{12}N_{L1}` through :math:`f_{23}` as well
    would count that path twice.

    Assumptions and scope:

    - Coster--Kronig moves one L hole outward without creating a second L hole,
      so every row sums to one. The outer-shell spectator vacancy left behind by
      the ejected Coster--Kronig electron is not propagated, and neither are the
      Auger daughters of any shell: every non-L row is the identity.
    - Only the L shell is redistributed. xraydb's M-shell values are not a
      probability distribution -- the finals of Cr M1 sum to 3.82 -- so they are
      excluded until EADL supplies a normalized cascade topology.
    - A route is wired only when both of its subshells are present in
      ``shell_labels``.

    Limiting case: an element with no tabulated L Coster--Kronig (Z <= 11 in
    xraydb) returns the identity, reproducing the direct-vacancy product
    ``omega_i * I_il`` exactly.

    Validation: characteristic-radiation
    """
    transfer = np.eye(len(shell_labels), dtype=float)
    index = {label: position for position, label in enumerate(shell_labels)}
    outflow: dict[str, float] = {}
    for initial, final in _L_SHELL_CK_ROUTES:
        if initial not in index or final not in index:
            continue
        probability = _require_finite(
            xraydb.ck_probability(element, initial, final),
            f"xraydb {element} {initial}->{final} Coster--Kronig probability",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"xraydb {element} {initial}->{final} Coster--Kronig probability must lie in [0, 1]"
            )
        transfer[index[initial], index[final]] = probability
        outflow[initial] = outflow.get(initial, 0.0) + probability
    for initial, total in outflow.items():
        if total > 1.0:
            raise ValueError(
                f"xraydb {element} {initial} Coster--Kronig probabilities sum to "
                f"{total:.8g}, which is not a vacancy distribution"
            )
        transfer[index[initial], index[initial]] = 1.0 - total
    return transfer


def _parse_characteristic_file(
    path: Path,
    element: str,
) -> CharacteristicCrossSectionTable:
    """Load EEDL ionization tables and join them to xraydb relaxation data."""
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    parsed_shells = _load_eedl_subshell_tables(
        resolved_path,
        stat.st_size,
        stat.st_mtime_ns,
        atomic_number,
        element,
    )
    ionization_shell_labels: list[str] = []
    binding_energies: list[float] = []
    fluorescence_yields: list[float] = []
    projectile_grids: list[np.ndarray] = []
    cross_section_tables: list[np.ndarray] = []
    line_records: list[tuple[int, str, float, float, float]] = []
    unresolved_radiative_shells: list[str] = []
    core_widths = xraydb.core_width(element)
    for subshell in parsed_shells:
        shell = _SHELL_LABELS.get(subshell.shell_designator)
        binding = subshell.binding_energy_eV
        if shell is None:
            if binding > _MIN_RELAXATION_CUTOFF_EV:
                raise ValueError(
                    f"{path}: unsupported ENDF subshell designator "
                    f"{subshell.shell_designator} "
                    f"above the {_MIN_RELAXATION_CUTOFF_EV:g} eV model cutoff"
                )
            continue
        edge = xraydb.xray_edge(element, shell)
        if edge is None:
            if binding > _MIN_RELAXATION_CUTOFF_EV:
                raise ValueError(f"{path}: xraydb has no {shell} edge for {element}")
            continue
        fluorescence_yield = _require_finite(
            edge.fyield,
            f"xraydb {element} {shell} fluorescence yield",
        )
        if not 0.0 <= fluorescence_yield <= 1.0:
            raise ValueError(f"xraydb {element} {shell} fluorescence yield must lie in [0, 1]")
        shell_row = len(ionization_shell_labels)
        ionization_shell_labels.append(shell)
        binding_energies.append(binding)
        fluorescence_yields.append(fluorescence_yield)
        projectile_grids.append(subshell.projectile_energy_eV)
        cross_section_tables.append(subshell.cross_section_cm2)
        lines = xraydb.xray_lines(element, initial_level=shell)
        if not lines and fluorescence_yield > 0.0 and binding > _MIN_RELAXATION_CUTOFF_EV:
            unresolved_radiative_shells.append(shell)
        intensity_sum = 0.0
        for label, line in lines.items():
            energy = _require_finite(
                line.energy,
                f"xraydb {element} {label} line energy",
                positive=True,
            )
            intensity = _require_finite(
                line.intensity,
                f"xraydb {element} {label} line intensity",
            )
            if not 0.0 <= intensity <= 1.0:
                raise ValueError(f"xraydb {element} {label} line intensity must lie in [0, 1]")
            if line.initial_level != shell:
                raise ValueError(
                    f"xraydb {element} {label} starts from {line.initial_level}, "
                    f"not requested shell {shell}"
                )
            intensity_sum += intensity
            line_fwhm = _transition_fwhm_eV(
                core_widths,
                element=element,
                line_label=str(label),
                initial_level=str(line.initial_level),
                final_level=str(line.final_level),
            )
            line_records.append(
                (shell_row, str(label), energy, line_fwhm, fluorescence_yield * intensity)
            )
        if lines and not 0.99 <= intensity_sum <= 1.01:
            raise ValueError(
                f"xraydb {element} {shell} line intensities sum to "
                f"{intensity_sum:.8g}, outside the accepted source-table tolerance"
            )

    if unresolved_radiative_shells:
        warnings.warn(
            f"xraydb has no emission lines for {element} shell(s) "
            f"{', '.join(unresolved_radiative_shells)} above the "
            f"{_MIN_RELAXATION_CUTOFF_EV:g} eV model cutoff; their EEDL "
            "ionization cross sections were parsed but cannot contribute photons "
            "until a complete relaxation model is supplied",
            RuntimeWarning,
            stacklevel=2,
        )

    if not ionization_shell_labels:
        raise ValueError(f"{path}: no supported EEDL subshells for {element}")
    emission = np.zeros((len(ionization_shell_labels), len(line_records)), dtype=float)
    for column, (shell_row, _label, _energy, _fwhm, probability) in enumerate(line_records):
        emission[shell_row, column] = probability
    # Row i must count photons per *primary* vacancy in i, and an L1 hole may
    # have moved to L2 or L3 by Coster--Kronig before it radiated.
    vacancy_transfer = _l_shell_vacancy_transfer(element, tuple(ionization_shell_labels))
    line_yields = vacancy_transfer @ emission

    return CharacteristicCrossSectionTable(
        element=element,
        atomic_number=atomic_number,
        recommended_cutoff_eV=_MIN_RELAXATION_CUTOFF_EV,
        projectile_energy_eV_by_shell=tuple(_readonly(grid) for grid in projectile_grids),
        ionization_shell_labels=tuple(ionization_shell_labels),
        shell_binding_energy_eV=_readonly(binding_energies),
        shell_fluorescence_yield=_readonly(fluorescence_yields),
        ionization_cross_sections_cm2_by_shell=tuple(
            _readonly(table) for table in cross_section_tables
        ),
        line_labels=tuple(label for _row, label, _energy, _fwhm, _probability in line_records),
        line_initial_shell=tuple(
            ionization_shell_labels[row]
            for row, _label, _energy, _fwhm, _probability in line_records
        ),
        line_energy_eV=_readonly(
            [energy for _row, _label, energy, _fwhm, _probability in line_records]
        ),
        line_fwhm_eV=_readonly(
            [fwhm for _row, _label, _energy, fwhm, _probability in line_records]
        ),
        vacancy_transfer=_readonly(vacancy_transfer),
        line_yield_per_vacancy=_readonly(line_yields),
    )


@cache
def _load_packaged_characteristic_cross_sections(
    element: str,
) -> CharacteristicCrossSectionTable:
    _verify_packaged_eedl()
    return _parse_characteristic_file(
        CHARACTERISTIC_DATA_DIR / CHARACTERISTIC_EEDL_FILENAME,
        element,
    )


@cache
def _verify_packaged_eedl() -> None:
    """Fail closed if the packaged EEDL bytes differ from the vetted input."""
    path = CHARACTERISTIC_DATA_DIR / CHARACTERISTIC_EEDL_FILENAME
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        raise FileNotFoundError(f"packaged EEDL database is missing: expected {path}") from None
    actual = digest.hexdigest()
    if actual != CHARACTERISTIC_EEDL_SHA256:
        raise ValueError(
            f"packaged EEDL database checksum mismatch: expected "
            f"{CHARACTERISTIC_EEDL_SHA256}, got {actual}"
        )


def load_characteristic_cross_sections(
    element: str,
    *,
    data_dir: str | Path | None = None,
) -> CharacteristicCrossSectionTable:
    """Load one element's EEDL ionization and xraydb relaxation table.

    Parameters
    ----------
    element
        Chemical symbol, such as ``"C"`` or ``"Si"``.
    data_dir
        Optional EEDL ENDF-6 file or directory containing ``EEDL.endf``. The
        checksum-pinned packaged tape is used by default.

    Returns
    -------
    CharacteristicCrossSectionTable
        Shell cross sections plus line energies, yields, and natural FWHMs.

    Notes
    -----
    Data provenance, supported ENDF sections, unresolved relaxation behavior,
    and assumptions are documented in
    ``docs/physics/radiation-physics/characteristic-radiation.md``.
    """
    if not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None:
        raise ValueError("element must be a chemical symbol such as 'C' or 'Si'")
    candidate = CHARACTERISTIC_DATA_DIR if data_dir is None else Path(data_dir)
    path = candidate if candidate.is_file() else candidate / CHARACTERISTIC_EEDL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no EEDL shell-ionization database for {element}: expected {path}; "
            "add an ENDF-6 EEDL electro-atomic file with MF=23/MT=534-572"
        )
    if data_dir is None:
        return _load_packaged_characteristic_cross_sections(element)
    return _parse_characteristic_file(path, element)


def _energy_bin_edges_and_widths(E_grid_eV: object) -> tuple[np.ndarray, np.ndarray]:
    """Nonuniform-center bin edges and widths for characteristic-line bins.

    Same convention as :func:`pyrite._grid_semantics.node_bin_edges_and_widths`
    (interior edges at the midpoint, outer edges a reflected half-width), with
    one addition: photon energy is a one-sided physical coordinate, so the low
    edge is clamped at 0 eV instead of the raw mirror reflection, which can go
    negative when the grid's first spacing exceeds ``grid[0]`` (for example a
    log-floored grid whose first two nodes are close together after a wide
    gap). No coordinate is inserted to represent that floor -- the returned
    array is still ``grid.size + 1`` edges for ``grid.size`` centres -- this
    only changes what value the existing first edge holds. The high edge is
    never clamped: no analogous physical ceiling applies to photon energy.

    Validation: characteristic-radiation
    """
    grid = np.asarray(E_grid_eV, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("characteristic spectrum requires at least two energy-bin centres")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("characteristic energy-bin centres must be finite and strictly increasing")
    edges = np.empty(grid.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = max(0.0, grid[0] - 0.5 * (grid[1] - grid[0]))
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return edges, np.diff(edges)


def _lorentzian_bin_weights(
    edges_eV: np.ndarray,
    line_energy_eV: float,
    fwhm_eV: float,
) -> np.ndarray:
    """Exact in-window bin masses for a Lorentzian normalized on all energies.

    The analytic CDF difference avoids point-sampling narrow natural lines.
    Probability outside the requested window is not redistributed into it, so
    the returned masses sum to at most one and off-grid lines retain their
    physical in-window tails.

    Validation: characteristic-radiation
    """
    gamma = 0.5 * _require_finite(fwhm_eV, "line FWHM", positive=True)
    centre = _require_finite(line_energy_eV, "line energy", positive=True)
    weights = (
        np.arctan((edges_eV[1:] - centre) / gamma) - np.arctan((edges_eV[:-1] - centre) / gamma)
    ) / np.pi
    weights = np.maximum(weights, 0.0)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Lorentzian line has no finite mass on the requested energy grid")
    return weights


#: Below this in-window captured fraction, ``mc_characteristic_spectrum`` warns
#: that a contributing line's window truncation is severe rather than staying
#: silent. Chosen well below typical coverage -- a +-25 FWHM window already
#: only captures ~98.7% of a Lorentzian's mass, so a low bar avoids flagging
#: ordinary heavy-tail truncation and reserves the warning for windows that
#: miss most of a physically relevant line.
CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION = 0.5


def characteristic_line_window_mass(
    E_grid_eV: object,
    element: str,
    *,
    data_dir: str | Path | None = None,
    relaxation_cutoff_eV: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Report each xraydb line's in-window captured and truncated mass.

    ``_lorentzian_bin_weights`` integrates each line's normalized Lorentzian
    exactly over ``E_grid_eV`` and does not redistribute the tail outside the
    window (Validation: characteristic-radiation). That leaves the truncated
    fraction computable but, inside ``mc_characteristic_spectrum``, implicit
    in the returned spectral density. This makes it an explicit, queryable
    per-line quantity instead: for every line above ``relaxation_cutoff_eV``,
    ``captured + truncated == 1`` up to floating-point rounding, by
    construction of the arctan CDF difference on the full real line.

    Returns
    -------
    dict[str, tuple[float, float]]
        Line label to ``(captured_mass, truncated_mass)``, both fractions of
        the line's total integrated yield.

    Validation: characteristic-radiation
    """
    table = load_characteristic_cross_sections(element, data_dir=data_dir)
    cutoff = (
        table.recommended_cutoff_eV
        if relaxation_cutoff_eV is None
        else _require_finite(relaxation_cutoff_eV, "relaxation_cutoff_eV", positive=True)
    )
    edges, _widths = _energy_bin_edges_and_widths(E_grid_eV)
    report: dict[str, tuple[float, float]] = {}
    for label, energy, fwhm in zip(
        table.line_labels, table.line_energy_eV, table.line_fwhm_eV, strict=True
    ):
        if energy <= cutoff:
            continue
        captured = float(_lorentzian_bin_weights(edges, energy, fwhm).sum())
        report[str(label)] = (captured, 1.0 - captured)
    return report


def _characteristic_transport_cutoff_keV(E_cut_keV: object) -> float:
    """Resolve the enforced lower validity boundary for characteristic scoring.

    The current electron transport model is not claimed accurate below 1 keV.
    Characteristic production therefore defaults to that floor and rejects a
    lower requested cutoff instead of presenting sub-keV path as validated.

    Validation: characteristic-radiation
    """
    if E_cut_keV is None:
        return CHARACTERISTIC_TRANSPORT_FLOOR_KEV
    cutoff = _require_finite(E_cut_keV, "E_cut_keV", positive=True)
    if cutoff < CHARACTERISTIC_TRANSPORT_FLOOR_KEV:
        raise ValueError(
            f"characteristic scoring requires E_cut_keV >= "
            f"{CHARACTERISTIC_TRANSPORT_FLOOR_KEV:g} keV; sub-keV electron transport "
            "is outside the validated model"
        )
    return cutoff


def _interpolate_shell_cross_sections(
    table: CharacteristicCrossSectionTable,
    segment_energy_keV: object,
):
    """Linearly interpolate EEDL partial electroionization cross sections."""
    query_eV = xp.asarray(segment_energy_keV, dtype=REAL) * 1.0e3
    columns = []
    for energy_cpu, values_cpu in zip(
        table.projectile_energy_eV_by_shell,
        table.ionization_cross_sections_cm2_by_shell,
        strict=True,
    ):
        energy = xp.asarray(energy_cpu, dtype=REAL)
        values = xp.asarray(values_cpu, dtype=REAL)
        interpolated = xp.interp(query_eV, energy, values)
        inside = (query_eV >= energy[0]) & (query_eV <= energy[-1])
        columns.append(xp.where(inside, interpolated, REAL(0.0)))
    return xp.stack(columns, axis=1)


def _xraydb_line_yields(
    table: CharacteristicCrossSectionTable,
    cutoff_eV: float,
) -> np.ndarray:
    """Return xraydb photons per primary vacancy after the energy cutoff.

    xraydb stores an edge fluorescence yield ``omega_i`` and conditional
    radiative-line intensities ``I_i,line``, whose product is the direct line
    probability for a vacancy that radiates from the shell it was created in.
    ``table.line_yield_per_vacancy`` is that product after L-shell
    Coster--Kronig redistribution (:func:`_l_shell_vacancy_transfer`), so a
    primary L1 vacancy also emits L2 and L3 lines. It still does not invent the
    Auger daughter vacancies or the M-shell Coster--Kronig topology that
    xraydb's tables do not specify.
    """
    response = np.array(table.line_yield_per_vacancy, dtype=float, copy=True)
    response[:, table.line_energy_eV <= cutoff_eV] = 0.0
    return response


def mc_characteristic_spectrum(
    segments,
    E_grid_eV,
    element=None,
    n_atoms_per_ang3=None,
    theta_obs_rad=np.deg2rad(119.0),
    n_hat=None,
    chunk=20000,
    composition=None,
    layers=None,
    groove=None,
    electron_limit=None,
    E_cut_keV=None,
    data_dir=None,
    relaxation_cutoff_eV=None,
):
    """Return the characteristic X-ray density from transport segments.

    Each xraydb transition is represented by an exactly bin-integrated natural
    Lorentzian and the result is returned in photons per eV per sr per incident
    electron. The source equation, linewidth construction, relaxation scope,
    geometry assumptions, and limiting cases are documented in
    ``docs/physics/radiation-physics/characteristic-radiation.md``.

    Parameters
    ----------
    segments
        Transport output mapping from :func:`simulate_trajectories`.
    E_grid_eV
        Strictly increasing characteristic-spectrum bin centres in eV.
    element, n_atoms_per_ang3
        Elemental target symbol and number density, superseded by ``composition``.
    theta_obs_rad, n_hat
        Polar observation angle or explicit sample-frame direction.
    chunk
        Maximum number of transport segments reduced at once.
    composition
        Compound ``(element, number_density)`` pairs in atoms per cubic angstrom.
    layers
        Optional film-first absorber stack.
    groove
        Optional supported blazed-groove escape geometry.
    electron_limit
        Optional leading macro-electron count used for normalization.
    E_cut_keV
        Optional post-transport electron-energy cutoff in keV. Characteristic
        scoring defaults to and enforces a 1 keV transport-validity floor.
    data_dir
        Optional EEDL file or containing directory.
    relaxation_cutoff_eV
        Lowest emitted line energy admitted from the relaxation data.

    Returns
    -------
    numpy.ndarray
        Characteristic density on ``E_grid_eV``.

    Validation: characteristic-radiation
    """
    if isinstance(chunk, bool) or int(chunk) <= 0:
        raise ValueError("chunk must be a positive integer")
    chunk = int(chunk)
    characteristic_cutoff_keV = _characteristic_transport_cutoff_keV(E_cut_keV)
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    all_tables = {
        el: load_characteristic_cross_sections(el, data_dir=data_dir) for el, _density in comp
    }
    minimum_cutoff = max(table.recommended_cutoff_eV for table in all_tables.values())
    if relaxation_cutoff_eV is None:
        relaxation_cutoff = minimum_cutoff
    else:
        relaxation_cutoff = _require_finite(
            relaxation_cutoff_eV,
            "relaxation_cutoff_eV",
            positive=True,
        )
        if relaxation_cutoff < minimum_cutoff:
            raise ValueError(
                f"relaxation_cutoff_eV={relaxation_cutoff:g} is below the data's "
                f"reliable minimum {minimum_cutoff:g} eV"
            )

    segments = _clip_segments_to_cutoff(segments, characteristic_cutoff_keV, comp, layers)
    Ne = segments["Ne"] if electron_limit is None else electron_limit
    if isinstance(Ne, bool) or int(Ne) <= 0:
        raise ValueError("electron_limit/segments['Ne'] must be a positive integer")

    edges, bin_widths = _energy_bin_edges_and_widths(E_grid_eV)
    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    spec = xp.zeros(E_grid.size, dtype=REAL)
    n_hat = _observation_direction(theta_obs_rad, n_hat)
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        _validate_groove_escape_direction(n_hat, groove)

    seg_elec_id = xp.asarray(segments["elec_id"])
    segment_index = xp.flatnonzero(seg_elec_id < int(Ne))
    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)[segment_index]
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[segment_index]
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    seg_E = xp.asarray(segments[E_field], dtype=REAL)[segment_index]
    if int(seg_E.size) == 0:
        return _to_cpu(spec)

    thickness = segments["thickness_ang"]
    z_mid = seg_r[:, 2]
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    if groove is not None:
        L_esc = xp.asarray(escape_distance_ang(seg_r[:, 0], z_mid, groove), dtype=REAL)
    elif finite_footprint:
        L_esc = _segment_escape_distance(segments, n_hat, xp=xp)[segment_index]
    else:
        L_esc = _escape_length(z_mid, thickness, n_hat[2])
    inv_nz = 1.0 / max(abs(float(n_hat[2])), 1.0e-12)
    severe_truncation: list[tuple[str, str, float]] = []

    for el, number_density_ang3 in comp:
        table = all_tables[el]
        xraydb_yields = _xraydb_line_yields(table, relaxation_cutoff)
        line_work = []
        for line_index, (line_energy, line_fwhm) in enumerate(
            zip(table.line_energy_eV, table.line_fwhm_eV, strict=True)
        ):
            if line_energy <= relaxation_cutoff:
                continue
            if not np.any(xraydb_yields[:, line_index] > 0.0):
                continue
            weights_cpu = _lorentzian_bin_weights(edges, line_energy, line_fwhm)
            if not np.any(weights_cpu > 0.0):
                continue
            captured = float(weights_cpu.sum())
            peak_in_window = edges[0] <= line_energy <= edges[-1]
            if peak_in_window and captured < CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION:
                # The line centre is inside the requested window but the window
                # itself is too narrow around it -- distinct from a line whose
                # centre sits entirely outside E_grid_eV and only contributes a
                # deliberately small physical tail (see
                # `characteristic_line_window_mass` for that case's breakdown).
                severe_truncation.append((el, str(table.line_labels[line_index]), captured))
            profile_cpu = weights_cpu / bin_widths
            profile = xp.asarray(profile_cpu, dtype=REAL)
            if layers is None:
                mu = _mu_total_inv_ang(
                    comp,
                    xp.asarray([line_energy], dtype=REAL),
                )[0]
                mu = xp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
                layer_mu = ()
            else:
                mu = None
                layer_mu = [
                    xp.nan_to_num(
                        _mu_total_inv_ang(c, xp.asarray([line_energy], dtype=REAL))[0],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    for _z_top, _z_bot, c in layers
                ]
            response = xp.asarray(xraydb_yields[:, line_index], dtype=REAL)
            line_work.append((profile, response, mu, layer_mu))

        if not line_work:
            continue
        line_yields = xp.zeros(len(line_work), dtype=REAL)
        for start in range(0, int(seg_E.size), chunk):
            sl = slice(start, min(start + chunk, int(seg_E.size)))
            shell_sigma_cm2 = _interpolate_shell_cross_sections(table, seg_E[sl])
            path_cm = seg_L[sl] * 1.0e-8
            for work_index, (_profile, response, mu, layer_mu) in enumerate(line_work):
                if layers is None:
                    transmission = xp.exp(-L_esc[sl] * mu)
                elif finite_footprint:
                    tau = 0.0
                    for (z_top, z_bot, _composition), mu_i in zip(layers, layer_mu, strict=True):
                        path = _layer_path_length(
                            z_mid[sl],
                            n_hat[2],
                            L_esc[sl],
                            float(z_top),
                            float(z_bot),
                        )
                        tau = tau + path * mu_i
                    transmission = xp.exp(-tau)
                else:
                    tau = 0.0
                    for (z_top, z_bot, _composition), mu_i in zip(layers, layer_mu, strict=True):
                        dz = _layer_dz(z_mid[sl], n_hat[2], float(z_top), float(z_bot))
                        tau = tau + dz * inv_nz * mu_i
                    transmission = xp.exp(-tau)
                effective_line_sigma_cm2 = shell_sigma_cm2 @ response
                line_yields[work_index] += xp.sum(
                    number_density_ang3 * 1.0e24 * path_cm * effective_line_sigma_cm2 * transmission
                )
        for (profile, _response, _mu, _layer_mu), line_yield in zip(
            line_work,
            line_yields,
            strict=True,
        ):
            spec += line_yield * profile

    if severe_truncation:
        detail = ", ".join(
            f"{el} {label} captures {captured:.1%}" for el, label, captured in severe_truncation
        )
        warnings.warn(
            f"E_grid_eV truncates {len(severe_truncation)} characteristic line(s) below "
            f"{CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION:.0%} of their physical Lorentzian mass "
            f"(not redistributed into retained bins): {detail}. Widen the window or query "
            "characteristic_line_window_mass for the full per-line breakdown.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _to_cpu(spec / (4.0 * xp.pi) / int(Ne))


__all__ = [
    "CHARACTERISTIC_DATA_DIR",
    "CHARACTERISTIC_EEDL_FILENAME",
    "CHARACTERISTIC_EEDL_SHA256",
    "CHARACTERISTIC_ENDF_PARSERPY_VERSION",
    "CHARACTERISTIC_MODEL",
    "CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION",
    "CHARACTERISTIC_TRANSPORT_FLOOR_KEV",
    "CHARACTERISTIC_XRAYDB_VERSION",
    "CharacteristicCrossSectionTable",
    "characteristic_line_window_mass",
    "load_characteristic_cross_sections",
    "mc_characteristic_spectrum",
]
