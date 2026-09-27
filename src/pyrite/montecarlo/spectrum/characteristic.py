"""Electron-impact characteristic x rays from EEDL, EADL, and xraydb data."""

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xraydb

from ..._backend import REAL, _to_cpu, xp
from ...materials.atomic import Z_TABLE
from ...materials.attenuation import (
    _mu_total_inv_ang,
    _normalize_composition,
)
from ..eadl_relaxation import EADL_FILENAME as CHARACTERISTIC_EADL_FILENAME
from ..eadl_relaxation import EADL_SHA256 as CHARACTERISTIC_EADL_SHA256
from ..eadl_relaxation import (
    EADLRelaxation,
    _branching_scales,
    load_eadl_relaxation,
    vacancy_cascade,
)
from ..eedl_ionization import EEDL_DATA_DIR as CHARACTERISTIC_DATA_DIR
from ..eedl_ionization import EEDL_FILENAME as CHARACTERISTIC_EEDL_FILENAME
from ..eedl_ionization import EEDL_SHA256 as CHARACTERISTIC_EEDL_SHA256
from ..eedl_ionization import (
    EEDL_SUBSHELL_LABELS,
    _load_eedl_subshell_tables,
    _readonly,
    _require_finite,
    _verify_packaged_eedl,
)
from ..eedl_ionization import EEDLSubshellTable as EEDLSubshellTable
from ..eedl_ionization import load_eedl_shell_ionization as load_eedl_shell_ionization
from .lines import (
    _clip_segments_to_cutoff,
    _observation_direction,
    _validate_groove_escape_direction,
)
from .segment_escape import mean_transmission, segment_escape_paths

FluorescenceYieldSource = Literal["eadl", "elam"]
_FLUORESCENCE_YIELD_SOURCES = ("eadl", "elam")

CHARACTERISTIC_ENDF_PARSERPY_VERSION = package_version("endf-parserpy")
CHARACTERISTIC_XRAYDB_VERSION = package_version("xraydb")


def characteristic_model_marker(fluorescence_yields: FluorescenceYieldSource = "eadl") -> str:
    """Identity marker of the characteristic model for one yield choice.

    The EEDL and EADL checksums, the parser and xraydb versions, and the
    fluorescence-yield source all change the scored spectrum, so each is part
    of the checkpoint identity.
    """
    if fluorescence_yields not in _FLUORESCENCE_YIELD_SOURCES:
        raise ValueError(f"fluorescence_yields must be one of {_FLUORESCENCE_YIELD_SOURCES}")
    return (
        f"eedl-2025-{CHARACTERISTIC_EEDL_SHA256[:12]}/"
        f"eadl-2025-{CHARACTERISTIC_EADL_SHA256[:12]}/"
        f"endf-parserpy-{CHARACTERISTIC_ENDF_PARSERPY_VERSION}/"
        f"xraydb-{CHARACTERISTIC_XRAYDB_VERSION}-eadl-cascade-{fluorescence_yields}-yields-"
        "lorentzian-segment-escape-v7"
    )


CHARACTERISTIC_MODEL = characteristic_model_marker()
_MIN_RELAXATION_CUTOFF_EV = 50.0
CHARACTERISTIC_TRANSPORT_FLOOR_KEV = 1.0
_SHELL_LABELS = EEDL_SUBSHELL_LABELS
_SHELL_ORDER = {label: code for code, label in _SHELL_LABELS.items()}


@dataclass(frozen=True, slots=True)
class CharacteristicCrossSectionTable:
    """Elemental inner-shell ionization joined to the EADL relaxation cascade.

    EEDL subshells may have different incident-energy grids, so
    ``projectile_energy_eV_by_shell`` and
    ``ionization_cross_sections_cm2_by_shell`` contain one read-only array per
    entry in ``ionization_shell_labels`` -- the subshells bound above the
    minimum relaxation cutoff, which are the only ones whose primary vacancies
    can radiate. ``relaxation`` holds the EADL cascade over all occupied
    subshells, in cascade order ``relaxation_shell_labels``, and
    ``ionization_relaxation_index`` maps each ionization row into it.

    ``radiative_yield_per_decay`` has shape ``(n_relaxation_shell, n_line)``:
    the probability that one decay of a vacancy in that subshell emits the
    line. ``vacancy_transfer`` (``(n_ionization_shell, n_relaxation_shell)``)
    is the expected number of vacancies ever held by each subshell per primary
    vacancy, and ``line_yield_per_vacancy`` (``(n_ionization_shell, n_line)``)
    their product: photons per *primary* vacancy. Both are evaluated at
    ``recommended_cutoff_eV``; :func:`_cascade_line_yields` re-evaluates them
    for another cutoff.

    ``line_source[l]`` is ``"xraydb"`` when the line's energy is xraydb's
    (Elam) value for that ``(initial, final)`` level pair, or ``"eadl"`` when
    xraydb has no such line and the EADL transition energy is used.
    ``shell_fluorescence_yield`` is the radiative branching ratio used for each
    ionization subshell, EADL or Elam per ``fluorescence_yields``.
    """

    element: str
    atomic_number: int
    recommended_cutoff_eV: float
    fluorescence_yields: FluorescenceYieldSource
    projectile_energy_eV_by_shell: tuple[np.ndarray, ...]
    ionization_shell_labels: tuple[str, ...]
    shell_binding_energy_eV: np.ndarray
    shell_fluorescence_yield: np.ndarray
    ionization_cross_sections_cm2_by_shell: tuple[np.ndarray, ...]
    relaxation: EADLRelaxation
    relaxation_shell_labels: tuple[str, ...]
    ionization_relaxation_index: np.ndarray
    relaxation_fluorescence_yields: Mapping[int, float] | None
    radiative_yield_per_decay: np.ndarray
    line_labels: tuple[str, ...]
    line_initial_shell: tuple[str, ...]
    line_final_shell: tuple[str, ...]
    line_source: tuple[str, ...]
    line_energy_eV: np.ndarray
    line_fwhm_eV: np.ndarray
    vacancy_transfer: np.ndarray
    line_yield_per_vacancy: np.ndarray


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
        except TypeError, ValueError:
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


def _expand_levels(level: str) -> tuple[str, ...]:
    """Split an xraydb grouped level such as ``M4,5`` into its subshells."""
    if "," not in level:
        return (level,)
    first, *rest = level.split(",")
    prefix_match = re.match(r"[A-Z]+", first)
    if prefix_match is None:
        return (level,)
    prefix = prefix_match.group(0)
    return (first, *(f"{prefix}{suffix}" for suffix in rest))


def _xraydb_line_index(element: str) -> dict[tuple[str, str], tuple[str, Any]]:
    """Map each resolved ``(initial, final)`` subshell pair to its xraydb line.

    Grouped xraydb levels (``M4,5``) claim every component pair. A pair
    claimed by two xraydb lines would make the energy join ambiguous and is
    rejected.
    """
    index: dict[tuple[str, str], tuple[str, Any]] = {}
    for label, line in xraydb.xray_lines(element).items():
        for initial in _expand_levels(str(line.initial_level)):
            for final in _expand_levels(str(line.final_level)):
                if (initial, final) in index:
                    raise ValueError(
                        f"xraydb {element} lines {index[initial, final][0]} and {label} "
                        f"both claim the {initial}-{final} transition"
                    )
                index[initial, final] = (str(label), line)
    return index


def _elam_fluorescence_yields(element: str, relaxation: EADLRelaxation) -> dict[int, float]:
    """xraydb (Elam/Krause) ``omega_i`` for every EADL subshell xraydb tabulates."""
    yields: dict[int, float] = {}
    for designator, label in zip(
        relaxation.shell_designators, relaxation.shell_labels, strict=True
    ):
        edge = xraydb.xray_edge(element, label)
        if edge is None:
            continue
        value = _require_finite(edge.fyield, f"xraydb {element} {label} fluorescence yield")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"xraydb {element} {label} fluorescence yield must lie in [0, 1]")
        yields[designator] = value
    return yields


def _cascade_yields(
    relaxation: EADLRelaxation,
    fluorescence_yields: Mapping[int, float] | None,
    radiative_yield_per_decay: np.ndarray,
    line_energy_eV: np.ndarray,
    ionization_relaxation_index: np.ndarray,
    cutoff_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Vacancy visits and photons per primary vacancy for one relaxation cutoff.

    Returns ``(vacancy_transfer, line_yield_per_vacancy)`` with
    ``vacancy_transfer = V[primary rows]`` from :func:`vacancy_cascade` and

    .. math::

        Y_{p\ell} = \sum_i V_{pi}\,R_{i\ell},

    where :math:`R_{i\ell}` is the probability that one decay of a vacancy in
    :math:`i` emits line :math:`\ell`. Subshells bound at or below
    ``cutoff_eV`` do not decay, so their rows of :math:`R` are zero, and lines
    at or below ``cutoff_eV`` are not scored.

    Validation: characteristic-radiation
    """
    _daughters, visits = vacancy_cascade(
        relaxation, cutoff_eV, fluorescence_yields=fluorescence_yields
    )
    emission = np.array(radiative_yield_per_decay, dtype=float, copy=True)
    emission[relaxation.binding_energy_eV <= cutoff_eV, :] = 0.0
    emission[:, line_energy_eV <= cutoff_eV] = 0.0
    transfer = visits[ionization_relaxation_index]
    return transfer, transfer @ emission


def _cascade_line_yields(
    table: CharacteristicCrossSectionTable,
    cutoff_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_cascade_yields` for a loaded table at another cutoff."""
    return _cascade_yields(
        table.relaxation,
        table.relaxation_fluorescence_yields,
        table.radiative_yield_per_decay,
        table.line_energy_eV,
        table.ionization_relaxation_index,
        cutoff_eV,
    )


def _readonly_int(values: object) -> np.ndarray:
    out = np.asarray(values, dtype=int)
    out.setflags(write=False)
    return out


def _parse_characteristic_file(
    path: Path,
    eadl_path: Path | None,
    element: str,
    fluorescence_yields: FluorescenceYieldSource,
) -> CharacteristicCrossSectionTable:
    """Join EEDL ionization, the EADL cascade, and xraydb energies and widths.

    Every EADL radiative transition above the minimum relaxation cutoff becomes
    a line. Its energy and natural width are xraydb's when xraydb tabulates the
    same ``(initial, final)`` subshell pair; otherwise the energy is EADL's
    ``ETR`` and the width is still the sum of xraydb's initial- and final-level
    widths. xraydb lines with no EADL radiative counterpart (for example the
    dipole-forbidden K-L1) carry no EADL probability and are omitted.

    Validation: characteristic-radiation
    """
    if fluorescence_yields not in _FLUORESCENCE_YIELD_SOURCES:
        raise ValueError(f"fluorescence_yields must be one of {_FLUORESCENCE_YIELD_SOURCES}")
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
    relaxation = load_eadl_relaxation(element, data_dir=eadl_path)
    relaxation_row = {
        designator: row for row, designator in enumerate(relaxation.shell_designators)
    }
    scaled_yields = (
        _elam_fluorescence_yields(element, relaxation) if fluorescence_yields == "elam" else None
    )

    ionization_shell_labels: list[str] = []
    ionization_rows: list[int] = []
    binding_energies: list[float] = []
    projectile_grids: list[np.ndarray] = []
    cross_section_tables: list[np.ndarray] = []
    for subshell in parsed_shells:
        binding = subshell.binding_energy_eV
        if binding <= _MIN_RELAXATION_CUTOFF_EV:
            continue
        shell = _SHELL_LABELS.get(subshell.shell_designator)
        if shell is None or subshell.shell_designator not in relaxation_row:
            raise ValueError(
                f"{path}: EEDL subshell designator {subshell.shell_designator} for "
                f"{element} above the {_MIN_RELAXATION_CUTOFF_EV:g} eV model cutoff has "
                "no EADL relaxation data"
            )
        ionization_shell_labels.append(shell)
        ionization_rows.append(relaxation_row[subshell.shell_designator])
        binding_energies.append(binding)
        projectile_grids.append(subshell.projectile_energy_eV)
        cross_section_tables.append(subshell.cross_section_cm2)
    # H and He have no subshell above the cutoff: an empty table, not an
    # error, so hydrogenous compositions still score their other elements.

    core_widths = xraydb.core_width(element)
    xraydb_lines = _xraydb_line_index(element)
    columns: dict[str, int] = {}
    line_records: list[tuple[str, str, str, str, float, float]] = []
    emission_entries: list[tuple[int, int, float]] = []
    for row, shell in enumerate(relaxation.subshells):
        if shell.binding_energy_eV <= _MIN_RELAXATION_CUTOFF_EV:
            continue
        initial = relaxation.shell_labels[row]
        radiative_scale, _auger_scale = _branching_scales(
            shell, None if scaled_yields is None else scaled_yields.get(shell.shell_designator)
        )
        for final_designator, eadl_energy, probability in zip(
            shell.radiative_final,
            shell.radiative_energy_eV,
            shell.radiative_probability,
            strict=True,
        ):
            final = _SHELL_LABELS[int(final_designator)]
            match = xraydb_lines.get((initial, final))
            if match is None:
                label, energy, source = f"{initial}-{final}", float(eadl_energy), "eadl"
                initial_level, final_level = initial, final
            else:
                label, line = match
                energy = _require_finite(
                    line.energy, f"xraydb {element} {label} line energy", positive=True
                )
                source = "xraydb"
                initial_level, final_level = str(line.initial_level), str(line.final_level)
            if energy <= _MIN_RELAXATION_CUTOFF_EV:
                continue
            if label not in columns:
                columns[label] = len(line_records)
                line_records.append(
                    (
                        label,
                        initial_level,
                        final_level,
                        source,
                        energy,
                        _transition_fwhm_eV(
                            core_widths,
                            element=element,
                            line_label=label,
                            initial_level=initial_level,
                            final_level=final_level,
                        ),
                    )
                )
            emission_entries.append((row, columns[label], radiative_scale * probability))

    emission = np.zeros((len(relaxation.subshells), len(line_records)), dtype=float)
    for row, column, probability in emission_entries:
        emission[row, column] += probability
    fluorescence = []
    for row in ionization_rows:
        shell = relaxation.subshells[row]
        radiative_scale, _auger_scale = _branching_scales(
            shell, None if scaled_yields is None else scaled_yields.get(shell.shell_designator)
        )
        fluorescence.append(radiative_scale * shell.fluorescence_yield)
    line_energy = _readonly([record[4] for record in line_records])
    relaxation_index = _readonly_int(ionization_rows)
    transfer, line_yields = _cascade_yields(
        relaxation,
        scaled_yields,
        emission,
        line_energy,
        relaxation_index,
        _MIN_RELAXATION_CUTOFF_EV,
    )

    return CharacteristicCrossSectionTable(
        element=element,
        atomic_number=atomic_number,
        recommended_cutoff_eV=_MIN_RELAXATION_CUTOFF_EV,
        fluorescence_yields=fluorescence_yields,
        projectile_energy_eV_by_shell=tuple(_readonly(grid) for grid in projectile_grids),
        ionization_shell_labels=tuple(ionization_shell_labels),
        shell_binding_energy_eV=_readonly(binding_energies),
        shell_fluorescence_yield=_readonly(fluorescence),
        ionization_cross_sections_cm2_by_shell=tuple(
            _readonly(values) for values in cross_section_tables
        ),
        relaxation=relaxation,
        relaxation_shell_labels=relaxation.shell_labels,
        ionization_relaxation_index=relaxation_index,
        relaxation_fluorescence_yields=scaled_yields,
        radiative_yield_per_decay=_readonly(emission),
        line_labels=tuple(record[0] for record in line_records),
        line_initial_shell=tuple(record[1] for record in line_records),
        line_final_shell=tuple(record[2] for record in line_records),
        line_source=tuple(record[3] for record in line_records),
        line_energy_eV=line_energy,
        line_fwhm_eV=_readonly([record[5] for record in line_records]),
        vacancy_transfer=_readonly(transfer),
        line_yield_per_vacancy=_readonly(line_yields),
    )


@cache
def _load_packaged_characteristic_cross_sections(
    element: str,
    fluorescence_yields: FluorescenceYieldSource,
) -> CharacteristicCrossSectionTable:
    _verify_packaged_eedl()
    return _parse_characteristic_file(
        CHARACTERISTIC_DATA_DIR / CHARACTERISTIC_EEDL_FILENAME,
        None,
        element,
        fluorescence_yields,
    )


def load_characteristic_cross_sections(
    element: str,
    *,
    data_dir: str | Path | None = None,
    fluorescence_yields: FluorescenceYieldSource = "eadl",
) -> CharacteristicCrossSectionTable:
    """Load one element's EEDL ionization and EADL/xraydb relaxation table.

    Parameters
    ----------
    element
        Chemical symbol, such as ``"C"`` or ``"Si"``.
    data_dir
        Optional EEDL ENDF-6 file or directory containing ``EEDL.endf``. The
        checksum-pinned packaged tape is used by default. A directory that
        also contains ``EADL2025.ALL`` supplies the relaxation data too;
        otherwise the checksum-pinned packaged EADL is used.
    fluorescence_yields
        ``"eadl"`` (default) keeps EADL's radiative branching. ``"elam"``
        rescales each subshell's radiative branch to xraydb's Elam/Krause
        ``omega_i`` and its nonradiative branch to ``1 - omega_i``, keeping
        each branch's EADL shape.

    Returns
    -------
    CharacteristicCrossSectionTable
        Shell cross sections, the EADL cascade, and line energies, yields, and
        natural FWHMs.

    Notes
    -----
    Data provenance, supported ENDF sections, cascade assumptions, and
    validity limits are documented in
    ``docs/physics/radiation-physics/characteristic-radiation.md``.
    """
    if not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None:
        raise ValueError("element must be a chemical symbol such as 'C' or 'Si'")
    if fluorescence_yields not in _FLUORESCENCE_YIELD_SOURCES:
        raise ValueError(f"fluorescence_yields must be one of {_FLUORESCENCE_YIELD_SOURCES}")
    candidate = CHARACTERISTIC_DATA_DIR if data_dir is None else Path(data_dir)
    path = candidate if candidate.is_file() else candidate / CHARACTERISTIC_EEDL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no EEDL shell-ionization database for {element}: expected {path}; "
            "add an ENDF-6 EEDL electro-atomic file with MF=23/MT=534-572"
        )
    if data_dir is None:
        return _load_packaged_characteristic_cross_sections(element, fluorescence_yields)
    eadl_path = (
        candidate
        if candidate.is_dir() and (candidate / CHARACTERISTIC_EADL_FILENAME).is_file()
        else None
    )
    return _parse_characteristic_file(path, eadl_path, element, fluorescence_yields)


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
    """Report each characteristic line's in-window captured and truncated mass.

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
    fluorescence_yields: FluorescenceYieldSource = "eadl",
):
    """Return the characteristic X-ray density from transport segments.

    Primary vacancies are relaxed through the EADL cascade and each emitted
    transition is represented by an exactly bin-integrated natural
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
        Binding-energy bound of the relaxation cascade in eV: vacancies in
        subshells bound at or below it are neither created as primaries nor
        propagated, and photons at or below it are not scored. Defaults to and
        may not go below the data's 50 eV minimum.
    fluorescence_yields
        ``"eadl"`` (default) or ``"elam"``; see
        :func:`load_characteristic_cross_sections`. The runner and checkpoint
        identity use the default; see :func:`characteristic_model_marker`.

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
        el: load_characteristic_cross_sections(
            el, data_dir=data_dir, fluorescence_yields=fluorescence_yields
        )
        for el, _density in comp
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
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[segment_index]
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    seg_E = xp.asarray(segments[E_field], dtype=REAL)[segment_index]
    if int(seg_E.size) == 0:
        return _to_cpu(spec)

    # Escape paths from both segment ends: the yield carries the segment mean
    # of exp(-tau), not its midpoint value (issue #176).
    owner, fraction, path_start, path_end = segment_escape_paths(
        segments, segment_index, n_hat, layers=layers, groove=groove, xp=xp
    )
    seg_L = seg_L[owner] * fraction
    seg_E = seg_E[owner]
    severe_truncation: list[tuple[str, str, float]] = []

    for el, number_density_ang3 in comp:
        table = all_tables[el]
        _transfer, line_yield_per_vacancy = _cascade_line_yields(table, relaxation_cutoff)
        line_work = []
        for line_index, (line_energy, line_fwhm) in enumerate(
            zip(table.line_energy_eV, table.line_fwhm_eV, strict=True)
        ):
            if line_energy <= relaxation_cutoff:
                continue
            if not np.any(line_yield_per_vacancy[:, line_index] > 0.0):
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
            response = xp.asarray(line_yield_per_vacancy[:, line_index], dtype=REAL)
            line_work.append((profile, response, mu, layer_mu))

        if not line_work:
            continue
        line_yields = xp.zeros(len(line_work), dtype=REAL)
        for start in range(0, int(seg_E.size), chunk):
            sl = slice(start, min(start + chunk, int(seg_E.size)))
            shell_sigma_cm2 = _interpolate_shell_cross_sections(table, seg_E[sl])
            path_cm = seg_L[sl] * 1.0e-8
            for work_index, (_profile, response, mu, layer_mu) in enumerate(line_work):
                mu_by_layer = xp.asarray([mu] if layers is None else layer_mu, dtype=REAL)
                transmission = mean_transmission(
                    path_start[sl] @ mu_by_layer, path_end[sl] @ mu_by_layer, xp=xp
                )
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
    "CHARACTERISTIC_EADL_FILENAME",
    "CHARACTERISTIC_EADL_SHA256",
    "CHARACTERISTIC_EEDL_FILENAME",
    "CHARACTERISTIC_EEDL_SHA256",
    "CHARACTERISTIC_ENDF_PARSERPY_VERSION",
    "CHARACTERISTIC_MODEL",
    "CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION",
    "CHARACTERISTIC_TRANSPORT_FLOOR_KEV",
    "CHARACTERISTIC_XRAYDB_VERSION",
    "CharacteristicCrossSectionTable",
    "FluorescenceYieldSource",
    "characteristic_model_marker",
    "characteristic_line_window_mass",
    "load_characteristic_cross_sections",
    "mc_characteristic_spectrum",
]
