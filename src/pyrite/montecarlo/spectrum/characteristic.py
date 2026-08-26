"""Electron-impact characteristic x rays from EEDL and xraydb data."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import xraydb

from ... import DATA_DIR
from ...materials.atomic import Z_TABLE
from ...materials.attenuation import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _normalize_composition,
)
from ..._backend import REAL, _to_cpu, xp
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
CHARACTERISTIC_EEDL_SHA256 = "ce37912435e0b8002f85878f98ccf7c5840cb168f1d46af3c9e915cd16c70ccc"
CHARACTERISTIC_XRAYDB_VERSION = package_version("xraydb")
CHARACTERISTIC_MODEL = (
    f"eedl-2025-{CHARACTERISTIC_EEDL_SHA256[:12]}/"
    f"xraydb-{CHARACTERISTIC_XRAYDB_VERSION}-direct-vacancy-v1"
)
_MIN_RELAXATION_CUTOFF_EV = 50.0
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
    entry in ``ionization_shell_labels``. ``line_yield_per_vacancy`` has shape
    ``(n_ionized_shell, n_line)`` and stores ``fluorescence_yield *
    conditional_line_intensity`` from xraydb.
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
    line_yield_per_vacancy: np.ndarray


@dataclass(frozen=True, slots=True)
class _EEDLSectionIndex:
    """Cached records for EEDL subshell-ionization sections."""

    sections_by_atomic_number: dict[
        int,
        dict[int, tuple[tuple[int, str], ...]],
    ]
    saw_eedl_subshell: bool
    saw_eadl_relaxation: bool


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


def _parse_endf_float(field: str, name: str) -> float:
    """Parse an eleven-column ENDF-6 real field, including implicit exponents."""
    text = field.strip()
    if not text:
        return 0.0
    normalized = text.replace("D", "E").replace("d", "e")
    if "e" not in normalized.lower():
        match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)", normalized)
        if match is not None:
            normalized = f"{match.group(1)}e{match.group(2)}"
    try:
        value = float(normalized)
    except ValueError as exc:
        raise ValueError(f"{name}: invalid ENDF-6 real field {field!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name}: ENDF-6 value must be finite")
    return value


def _parse_endf_int(field: str, name: str) -> int:
    text = field.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        value = _parse_endf_float(field, name)
        integer = int(round(value))
        if value != integer:
            raise ValueError(f"{name}: expected an ENDF-6 integer, got {field!r}") from None
        return integer


def _endf_fields(line: str) -> tuple[str, ...]:
    data = line[:66].ljust(66)
    return tuple(data[start : start + 11] for start in range(0, 66, 11))


def _endf_record_id(line: str) -> tuple[int, int, int] | None:
    if len(line) < 75:
        return None
    try:
        return int(line[66:70]), int(line[70:72]), int(line[72:75])
    except ValueError:
        return None


@cache
def _index_eedl_sections(
    path: Path,
    file_size: int,
    modified_time_ns: int,
) -> _EEDLSectionIndex:
    """Scan one ENDF-6 file and cache every EEDL subshell section.

    ``file_size`` and ``modified_time_ns`` are cache-key sentinels supplied by
    :func:`_read_eedl_sections`; they prevent an explicit data file that was
    replaced in place from reusing a stale index.
    """
    del file_size, modified_time_ns
    sections_by_atomic_number: dict[
        int,
        dict[int, tuple[tuple[int, str], ...]],
    ] = {}
    current_key: tuple[int, int, int] | None = None
    current_records: list[tuple[int, str]] = []
    saw_eedl_subshell = False
    saw_eadl_relaxation = False

    def finish_section() -> None:
        if current_key is None or not current_records:
            return
        _mat, mf, mt = current_key
        assert mf == 23 and 534 <= mt <= 572
        head_line_number, head_line = current_records[0]
        za = _parse_endf_float(
            _endf_fields(head_line)[0],
            f"{path}:{head_line_number}: ZA",
        )
        section_atomic_number = int(round(za / 1000.0))
        if not 1 <= section_atomic_number <= 100:
            raise ValueError(
                f"{path}:{head_line_number}: invalid EEDL atomic number derived from ZA={za:g}"
            )
        element_sections = sections_by_atomic_number.setdefault(section_atomic_number, {})
        if mt in element_sections:
            raise ValueError(
                f"{path}: duplicate EEDL MF=23/MT={mt} section for Z={section_atomic_number}"
            )
        element_sections[mt] = tuple(current_records)

    with path.open("r", encoding="ascii", errors="strict", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            record_id = _endf_record_id(line)
            if record_id is None:
                continue
            _mat, mf, mt = record_id
            saw_eedl_subshell = saw_eedl_subshell or (mf == 23 and 534 <= mt <= 572)
            saw_eadl_relaxation = saw_eadl_relaxation or (mf == 28 and mt == 533)
            if record_id != current_key:
                finish_section()
                current_key = record_id
                current_records = [(line_number, line)] if mf == 23 and 534 <= mt <= 572 else []
            elif current_records:
                current_records.append((line_number, line))
    finish_section()

    return _EEDLSectionIndex(
        sections_by_atomic_number=sections_by_atomic_number,
        saw_eedl_subshell=saw_eedl_subshell,
        saw_eadl_relaxation=saw_eadl_relaxation,
    )


def _read_eedl_sections(
    path: Path,
    atomic_number: int,
    element: str,
) -> dict[int, list[tuple[int, str]]]:
    """Collect the target element's EEDL MF=23 subshell TAB1 sections."""
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    index = _index_eedl_sections(
        resolved_path,
        stat.st_size,
        stat.st_mtime_ns,
    )
    sections = index.sections_by_atomic_number.get(atomic_number)

    if sections:
        return {mt: list(records) for mt, records in sections.items()}
    if index.saw_eadl_relaxation and not index.saw_eedl_subshell:
        raise ValueError(
            f"{path}: contains EADL MF=28/MT=533 atomic-relaxation data, not "
            "electron-impact ionization cross sections; provide the companion EEDL "
            "electro-atomic file with MF=23/MT=534-572"
        )
    raise ValueError(
        f"{path}: no electroionization subshell sections for {element} (Z={atomic_number}); "
        "expected EEDL MF=23/MT=534-572"
    )


def _parse_eedl_subshell(
    path: Path,
    mt: int,
    records: list[tuple[int, str]],
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Parse one ENDF-6 File-23 TAB1 subshell cross-section section."""
    if len(records) < 4:
        raise ValueError(f"{path}: EEDL MF=23/MT={mt} section is truncated")
    header_line_number, header_line = records[1]
    header = _endf_fields(header_line)
    binding = _parse_endf_float(
        header[0],
        f"{path}:{header_line_number}: EPE binding energy",
    )
    if binding <= 0.0:
        raise ValueError(f"{path}:{header_line_number}: EPE binding energy must be positive")
    nr = _parse_endf_int(header[4], f"{path}:{header_line_number}: NR")
    npairs = _parse_endf_int(header[5], f"{path}:{header_line_number}: NP")
    if nr <= 0 or npairs < 3:
        raise ValueError(f"{path}:{header_line_number}: EEDL TAB1 needs NR>0 and NP>=3")

    payload: list[tuple[str, int]] = []
    for line_number, line in records[2:]:
        payload.extend((field, line_number) for field in _endf_fields(line) if field.strip())
    expected_fields = 2 * nr + 2 * npairs
    if len(payload) != expected_fields:
        raise ValueError(
            f"{path}: EEDL MF=23/MT={mt} TAB1 declares {expected_fields} payload "
            f"fields but contains {len(payload)}"
        )
    interpolation = payload[: 2 * nr]
    breakpoints = [
        _parse_endf_int(field, f"{path}:{line_number}: NBT")
        for field, line_number in interpolation[0::2]
    ]
    laws = [
        _parse_endf_int(field, f"{path}:{line_number}: INT")
        for field, line_number in interpolation[1::2]
    ]
    if breakpoints[-1] != npairs or any(
        later <= earlier for earlier, later in zip(breakpoints[:-1], breakpoints[1:], strict=True)
    ):
        raise ValueError(f"{path}: invalid EEDL MF=23/MT={mt} interpolation breakpoints")
    if any(law != 2 for law in laws):
        raise ValueError(
            f"{path}: only ENDF interpolation law 2 (lin-lin) is supported for EEDL "
            f"cross sections; MF=23/MT={mt} declares {laws}"
        )

    table_fields = payload[2 * nr :]
    table_values = np.asarray(
        [
            _parse_endf_float(field, f"{path}:{line_number}: TAB1 value")
            for field, line_number in table_fields
        ],
        dtype=float,
    )
    projectile = table_values[0::2]
    cross_section_barn = table_values[1::2]
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
    return mt - 533, binding, projectile, cross_section_barn * 1.0e-24


def _parse_characteristic_file(
    path: Path,
    element: str,
) -> CharacteristicCrossSectionTable:
    """Parse EEDL ionization tables and join them to xraydb relaxation data."""
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    sections = _read_eedl_sections(path, atomic_number, element)
    parsed_shells = [_parse_eedl_subshell(path, mt, sections[mt]) for mt in sorted(sections)]
    ionization_shell_labels: list[str] = []
    binding_energies: list[float] = []
    fluorescence_yields: list[float] = []
    projectile_grids: list[np.ndarray] = []
    cross_section_tables: list[np.ndarray] = []
    line_records: list[tuple[int, str, float, float]] = []
    unresolved_radiative_shells: list[str] = []
    for shell_designator, binding, projectile, cross_section_cm2 in parsed_shells:
        shell = _SHELL_LABELS.get(shell_designator)
        if shell is None:
            if binding > _MIN_RELAXATION_CUTOFF_EV:
                raise ValueError(
                    f"{path}: unsupported ENDF subshell designator {shell_designator} "
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
        projectile_grids.append(projectile)
        cross_section_tables.append(cross_section_cm2)
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
            line_records.append((shell_row, str(label), energy, fluorescence_yield * intensity))
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
    line_yields = np.zeros((len(ionization_shell_labels), len(line_records)), dtype=float)
    for column, (shell_row, _label, _energy, probability) in enumerate(line_records):
        line_yields[shell_row, column] = probability

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
        line_labels=tuple(label for _row, label, _energy, _probability in line_records),
        line_initial_shell=tuple(
            ionization_shell_labels[row] for row, _label, _energy, _probability in line_records
        ),
        line_energy_eV=_readonly([energy for _row, _label, energy, _probability in line_records]),
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
    """Load EEDL electron-impact subshell cross sections and xraydb lines.

    ``data_dir`` may name a monolithic ENDF-6 EEDL file or a directory that
    contains ``EEDL.endf``. EEDL MF=23/MT=534--572 supplies incident-energy
    grids, binding energies, and electroionization cross sections in barns.
    Fluorescence yields, line energies, and conditional line intensities come
    from xraydb. EADL MF=28 relaxation files and legacy NIST
    ``ELEMENT_Xchar.txt`` line-emission exports are intentionally not treated
    as ionization cross sections. An EEDL shell without explicit xraydb lines
    is retained and reported with ``RuntimeWarning``; its unresolved photon
    contribution is zero.
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
    grid = np.asarray(E_grid_eV, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("characteristic spectrum requires at least two energy-bin centres")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("characteristic energy-bin centres must be finite and strictly increasing")
    edges = np.empty(grid.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return edges, np.diff(edges)


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
    """Return xraydb photons per initial vacancy after the energy cutoff.

    xraydb stores an edge fluorescence yield ``omega_i`` and conditional
    radiative-line intensities ``I_i,line``. The direct line probability is
    ``omega_i * I_i,line``. This intentionally does not invent Auger daughter
    shells or Coster--Kronig cascades that xraydb's emission-line API does not
    specify.
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
    """Characteristic x-ray spectrum ``d2N/dE dOmega``.

    Returns photons / eV / sr / incident electron on ``E_grid_eV``. For each
    transported segment, element, and initially ionized subshell ``i``, the
    vacancy-production expectation is ``n * L * sigma_i(T)``. For every xraydb
    line ``line`` from that shell, the photon expectation is multiplied by
    ``omega_i * I_i,line``, where ``omega_i`` is the xraydb edge fluorescence
    yield and ``I_i,line`` is the xraydb conditional line intensity.

    Emission is isotropic and uses PyRITE's established slab, finite-prism,
    groove, and multilayer Beer--Lambert escape paths. The implementation is a
    mean track-length estimator, like PyRITE bremsstrahlung; it does not sample
    discrete ionizations or photon branches.

    Sources: ENDF-6 File 23 and the 2025 LLNL Evaluated Electron Data Library
    (EEDL) for shell-resolved electron-impact ionization cross sections; Elam,
    Ravel, and Sieber, Radiation Physics and Chemistry 63, 121--128 (2002),
    through xraydb ``xray_edge`` and ``xray_lines`` for relaxation. Assumptions:
    EEDL interpolation law 2 (lin-lin), isolated single vacancies,
    independent-atom additivity, xraydb fluorescence yields and line
    intensities, isotropic prompt x rays, straight photon escape, and
    bin-integrated delta lines. Auger-fed secondary vacancies, Coster--Kronig
    redistribution, multiple-vacancy corrections, and Auger-electron transport
    are not modeled because the xraydb emission-line API does not specify a
    complete non-radiative cascade. Limiting cases: zero path, density,
    ionization cross section, fluorescence yield, or line intensity gives zero;
    a single unattenuated line gives ``n L sigma omega I/(4*pi)``; constant-
    energy segment subdivision preserves total yield. EEDL shells without an
    explicit xraydb line list are parsed but contribute zero with a warning.

    Validation: characteristic-radiation
    """
    if isinstance(chunk, bool) or int(chunk) <= 0:
        raise ValueError("chunk must be a positive integer")
    chunk = int(chunk)
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

    segments = _clip_segments_to_cutoff(segments, E_cut_keV, comp, layers)
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

    for el, number_density_ang3 in comp:
        table = all_tables[el]
        xraydb_yields = _xraydb_line_yields(table, relaxation_cutoff)
        line_work = []
        for line_index, line_energy in enumerate(table.line_energy_eV):
            if line_energy <= relaxation_cutoff:
                continue
            bin_index = int(np.searchsorted(edges, line_energy, side="right") - 1)
            if bin_index < 0 or bin_index >= E_grid.size:
                continue
            if not np.any(xraydb_yields[:, line_index] > 0.0):
                continue
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
            line_work.append((bin_index, response, mu, layer_mu))

        if not line_work:
            continue
        line_yields = xp.zeros(len(line_work), dtype=REAL)
        for start in range(0, int(seg_E.size), chunk):
            sl = slice(start, min(start + chunk, int(seg_E.size)))
            shell_sigma_cm2 = _interpolate_shell_cross_sections(table, seg_E[sl])
            path_cm = seg_L[sl] * 1.0e-8
            for work_index, (_bin_index, response, mu, layer_mu) in enumerate(line_work):
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
        for (bin_index, _response, _mu, _layer_mu), line_yield in zip(
            line_work,
            line_yields,
            strict=True,
        ):
            spec[bin_index] += line_yield / REAL(bin_widths[bin_index])

    return _to_cpu(spec / (4.0 * xp.pi) / int(Ne))


__all__ = [
    "CHARACTERISTIC_DATA_DIR",
    "CHARACTERISTIC_EEDL_FILENAME",
    "CHARACTERISTIC_EEDL_SHA256",
    "CHARACTERISTIC_MODEL",
    "CHARACTERISTIC_XRAYDB_VERSION",
    "CharacteristicCrossSectionTable",
    "load_characteristic_cross_sections",
    "mc_characteristic_spectrum",
]
