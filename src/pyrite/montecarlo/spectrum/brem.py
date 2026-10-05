"""Bremsstrahlung spectrum and external-background loading."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from endf_parserpy import EndfFile

from ..._backend import BACKEND, REAL, _to_cpu, xp
from ...materials.atomic import Z_TABLE
from ...materials.attenuation import (
    _finite_mu_or_raise,
    _mu_total_inv_ang,
    _normalize_composition,
)
from ...materials.crystal import ALPHA_FS
from ..eedl_ionization import eedl_path
from ..transport import TRANSPORT_ELEMENTS
from .brem_bremslib import (
    BremsLibBremsstrahlungTable,
    bremslib_segment_state,
    evaluate_bremslib,
    resolve_auto_model,
    stage_bremslib_table,
)
from .brem_unit_base import _unit_base_panels
from .characteristic import CHARACTERISTIC_EEDL_FILENAME as BREMSSTRAHLUNG_EEDL_FILENAME
from .characteristic import CHARACTERISTIC_EEDL_SHA256 as BREMSSTRAHLUNG_EEDL_SHA256
from .characteristic import (
    CHARACTERISTIC_ENDF_PARSERPY_VERSION as BREM_ENDF_PARSERPY_VERSION,
)
from .lines import (
    _clip_segments_to_cutoff,
    _observation_direction,
    _validate_groove_escape_direction,
)
from .segment_escape import mean_transmission, segment_escape_paths

_USE_JIT_BREM_REDUCTION = True
_BREMSLIB_CHUNK_CELLS = 1 << 21

BremsstrahlungModel = Literal["eedl", "bethe-heitler", "bremslib"]
BREMSSTRAHLUNG_MODEL = (
    f"eedl-2025-{BREMSSTRAHLUNG_EEDL_SHA256[:12]}/"
    f"endf-parserpy-{BREM_ENDF_PARSERPY_VERSION}-mf23-527-mf26-527-v3-unit-base-segment-escape"
)

R_E_CM2 = 7.9407877e-26  # classical electron radius squared [cm^2]
_BREM_MC2_KEV = 510.99895  # electron rest energy [keV]


class EEDLBremsstrahlungDataUnavailable(ValueError):
    """Requested element or bremsstrahlung section is absent from EEDL."""


@dataclass(frozen=True, slots=True)
class BremsstrahlungCrossSectionTable:
    """Validated EEDL total cross section and normalized photon spectra.

    Total cross sections come from MF=23/MT=527 in square centimetres. The
    photon probability densities come from the first MF=26/MT=527 subsection
    and are normalized in inverse electronvolts at each tabulated incident
    energy.
    """

    element: str
    atomic_number: int
    incident_energy_eV: np.ndarray
    total_cross_section_cm2: np.ndarray
    distribution_incident_energy_eV: np.ndarray
    photon_energy_eV_by_incident: tuple[np.ndarray, ...]
    photon_probability_density_per_eV_by_incident: tuple[np.ndarray, ...]
    photon_cumulative_probability_by_incident: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class _PreparedEEDLGrid:
    """One element's EEDL tables staged once for one output-energy grid."""

    table: BremsstrahlungCrossSectionTable
    total_energy_eV: Any
    total_cross_section_cm2: Any
    incident_panel_energy_eV: Any
    photon_probability_on_grid_per_eV: Any
    unit_base_x: Any
    panel_photon_min_eV: Any
    panel_photon_max_eV: Any
    panel_scaled_density: Any
    panel_scaled_cumulative: Any
    panel_tip_density_per_eV: Any
    minimum_incident_energy_eV: float
    maximum_incident_energy_eV: float


@dataclass(frozen=True, slots=True)
class _EEDLSegmentState:
    """Incident-energy interpolation state, stored with O(Nsegment) memory."""

    incident_energy_eV: Any
    available: Any
    lower_panel: Any
    panel_fraction: Any
    differential_scale_cm2: Any


@dataclass(frozen=True, slots=True)
class _ResolvedEEDLContext:
    prepared: _PreparedEEDLGrid
    state: _EEDLSegmentState
    all_available: bool


def _readonly(array: object) -> np.ndarray:
    out = np.asarray(array, dtype=float)
    out.setflags(write=False)
    return out


def _section_vector(section: Mapping[str, object], field: str, label: str) -> np.ndarray:
    try:
        values = np.asarray(section[field], dtype=float)
    except KeyError as exc:
        raise ValueError(f"{label} is missing {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid {field} values") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} {field} must be a finite one-dimensional array")
    return values


def _integer(value: object, label: str) -> int:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    integer = int(number)
    if not np.isfinite(number) or number != integer:
        raise ValueError(f"{label} must be an integer")
    return integer


def _validate_linlin_table(
    section: Mapping[str, object],
    sample_count: int,
    label: str,
) -> None:
    breakpoints_float = _section_vector(section, "NBT", label)
    laws_float = _section_vector(section, "INT", label)
    breakpoints = np.rint(breakpoints_float).astype(int)
    laws = np.rint(laws_float).astype(int)
    if (
        breakpoints.size == 0
        or breakpoints.size != laws.size
        or not np.array_equal(breakpoints_float, breakpoints)
        or not np.array_equal(laws_float, laws)
        or breakpoints[-1] != sample_count
        or np.any(np.diff(breakpoints) <= 0)
    ):
        raise ValueError(f"{label} has invalid NBT/INT interpolation metadata")
    if np.any(laws != 2):
        raise ValueError(
            f"{label} declares ENDF interpolation law(s) {laws.tolist()}; "
            "only law 2 (lin-lin) is supported"
        )


def _mapping_vector(mapping: Mapping[int, object], count: int, label: str) -> np.ndarray:
    try:
        values = np.asarray([mapping[index] for index in range(1, count + 1)], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain consecutive one-indexed numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain finite values")
    return values


def _extract_bremsstrahlung_table(
    path: Path,
    element: str,
    atomic_number: int,
    total_section: Mapping[str, object],
    product_section: Mapping[str, object],
) -> BremsstrahlungCrossSectionTable:
    total_label = f"{path}: EEDL MF=23/MT=527"
    incident = _section_vector(total_section, "Eint", total_label)
    total_barn = _section_vector(total_section, "sigma", total_label)
    if incident.size != total_barn.size or incident.size < 2:
        raise ValueError(f"{total_label} must contain paired Eint/sigma samples")
    _validate_linlin_table(total_section, incident.size, total_label)
    if np.any(incident <= 0.0) or np.any(np.diff(incident) <= 0.0):
        raise ValueError(f"{total_label} incident energies must be positive and increasing")
    if np.any(total_barn < 0.0):
        raise ValueError(f"{total_label} cross sections must be non-negative")

    yield_label = f"{path}: EEDL MF=26/MT=527 photon yield"
    try:
        yields = cast(Mapping[str, object], product_section["yields"])
    except KeyError as exc:
        raise ValueError(f"{yield_label} table is missing") from exc
    yield_energy = _section_vector(yields, "Eint", yield_label)
    yield_value = _section_vector(yields, "yi", yield_label)
    if yield_energy.size != yield_value.size or yield_energy.size < 2:
        raise ValueError(f"{yield_label} must contain paired Eint/yi samples")
    _validate_linlin_table(yields, yield_energy.size, yield_label)
    if np.any(yield_energy <= 0.0) or np.any(np.diff(yield_energy) <= 0.0):
        raise ValueError(f"{yield_label} incident energies must be positive and increasing")
    if not np.allclose(yield_value, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{yield_label} must declare one photon per reaction")

    spectrum_label = f"{path}: EEDL MF=26/MT=527 photon spectrum"
    try:
        subsections = cast(Mapping[int, object], product_section["subsection"])
        photon = cast(Mapping[str, object], subsections[1])
        panel_count = _integer(photon["NE"], f"{spectrum_label} NE")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{spectrum_label} subsection is missing or invalid") from exc
    if (
        _integer(photon.get("LANG", -1), f"{spectrum_label} LANG") != 1
        or _integer(photon.get("LEP", -1), f"{spectrum_label} LEP") != 2
    ):
        raise ValueError(f"{spectrum_label} requires LANG=1 and LEP=2")
    if panel_count < 2:
        raise ValueError(f"{spectrum_label} needs at least two incident-energy panels")
    _validate_linlin_table(photon, panel_count, spectrum_label)
    panel_incident = _mapping_vector(
        cast(Mapping[int, object], photon.get("E")),
        panel_count,
        f"{spectrum_label} E",
    )
    if np.any(panel_incident <= 0.0) or np.any(np.diff(panel_incident) <= 0.0):
        raise ValueError(f"{spectrum_label} incident energies must be positive and increasing")

    photon_grids: list[np.ndarray] = []
    photon_densities: list[np.ndarray] = []
    photon_cumulative: list[np.ndarray] = []
    for panel in range(1, panel_count + 1):
        panel_label = f"{spectrum_label} panel {panel}"
        try:
            point_counts = cast(Mapping[int, object], photon["NEP"])
            discrete_counts = cast(Mapping[int, object], photon["ND"])
            angular_counts = cast(Mapping[int, object], photon["NA"])
            energy_tables = cast(Mapping[int, object], photon["Ep"])
            coefficient_tables = cast(Mapping[int, object], photon["b"])
            point_count = _integer(point_counts[panel], f"{panel_label} NEP")
            nd = _integer(discrete_counts[panel], f"{panel_label} ND")
            na = _integer(angular_counts[panel], f"{panel_label} NA")
            energy = _mapping_vector(
                cast(Mapping[int, object], energy_tables[panel]),
                point_count,
                f"{panel_label} Ep",
            )
            coefficients = cast(Mapping[int, object], coefficient_tables[panel])
            density = np.asarray(
                [
                    cast(Mapping[int, object], coefficients[point])[0]
                    for point in range(1, point_count + 1)
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{panel_label} is missing or invalid") from exc
        if nd != 0 or na != 0:
            raise ValueError(f"{panel_label} requires ND=0 and NA=0")
        if point_count < 2 or density.shape != energy.shape or not np.all(np.isfinite(density)):
            raise ValueError(f"{panel_label} must contain paired finite Ep/b samples")
        if np.any(energy <= 0.0) or np.any(np.diff(energy) <= 0.0):
            raise ValueError(f"{panel_label} photon energies must be positive and increasing")
        if energy[-1] > panel_incident[panel - 1] * (1.0 + 1.0e-10):
            raise ValueError(f"{panel_label} exceeds its incident electron energy")
        if np.any(density < 0.0):
            raise ValueError(f"{panel_label} probability density must be non-negative")
        area = float(np.trapezoid(density, energy))
        if not np.isfinite(area) or not 0.999 <= area <= 1.001:
            raise ValueError(f"{panel_label} probability integrates to {area:.8g}, not unity")
        density = density / area
        cumulative = np.concatenate(
            ([0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(energy)))
        )
        photon_grids.append(_readonly(energy))
        photon_densities.append(_readonly(density))
        photon_cumulative.append(_readonly(cumulative))

    return BremsstrahlungCrossSectionTable(
        element=element,
        atomic_number=atomic_number,
        incident_energy_eV=_readonly(incident),
        total_cross_section_cm2=_readonly(total_barn * 1.0e-24),
        distribution_incident_energy_eV=_readonly(panel_incident),
        photon_energy_eV_by_incident=tuple(photon_grids),
        photon_probability_density_per_eV_by_incident=tuple(photon_densities),
        photon_cumulative_probability_by_incident=tuple(photon_cumulative),
    )


@cache
def _parse_bremsstrahlung_file(
    path: Path,
    file_size: int,
    modified_time_ns: int,
    element: str,
    atomic_number: int,
) -> BremsstrahlungCrossSectionTable:
    del file_size, modified_time_ns
    matching = []
    with EndfFile(path, on_error="raise") as tape:
        for position in range(len(tape)):
            material = tape[position]
            if round(float(material.za) / 1000.0) == atomic_number:
                matching.append(material)
        if len(matching) > 1:
            mats = [int(material.mat) for material in matching]
            raise ValueError(f"{path}: multiple EEDL materials for {element}: MAT={mats}")
        if not matching:
            raise EEDLBremsstrahlungDataUnavailable(
                f"{path}: no EEDL material for {element} (Z={atomic_number})"
            )
        material = matching[0]
        section_ids = set(material.sections())
        missing = [
            label
            for section, label in (((23, 527), "MF=23"), ((26, 527), "MF=26"))
            if section not in section_ids
        ]
        if missing:
            raise EEDLBremsstrahlungDataUnavailable(
                f"{path}: {element} lacks {' and '.join(missing)}/MT=527 bremsstrahlung data"
            )
        return _extract_bremsstrahlung_table(
            path,
            element,
            atomic_number,
            material[23, 527],
            material[26, 527],
        )


@cache
def _load_packaged_bremsstrahlung_cross_sections(
    element: str,
) -> BremsstrahlungCrossSectionTable:
    path = eedl_path()
    stat = path.stat()
    return _parse_bremsstrahlung_file(
        path.resolve(), stat.st_size, stat.st_mtime_ns, element, Z_TABLE[element]
    )


def load_bremsstrahlung_cross_sections(
    element: str,
    *,
    data_dir: str | Path | None = None,
) -> BremsstrahlungCrossSectionTable:
    """Read EEDL MF=23/527 and MF=26/527 for one element.

    The fetched file (``pyrite tables fetch eedl``) is checksum-pinned.
    ``data_dir`` is an explicit testing
    and expert override; its file is still structurally validated.

    Validation: brem-spectrum
    """
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    if data_dir is None:
        return _load_packaged_bremsstrahlung_cross_sections(element)
    path = Path(data_dir) / BREMSSTRAHLUNG_EEDL_FILENAME
    stat = path.stat()
    return _parse_bremsstrahlung_file(
        path.resolve(), stat.st_size, stat.st_mtime_ns, element, atomic_number
    )


def _brem_incident_state(T_keV):
    """Return incident momentum ``p/(m_e c)`` and beta for segment energies.

    These depend only on the emitting segment energy, not on photon energy or
    composition element, so the CUDA reduction path computes them once per
    spectrum instead of once per ``(segment, energy-block, element)``.
    """
    T = xp.asarray(T_keV, dtype=REAL)
    p_i = xp.sqrt(T * (T + 2.0 * _BREM_MC2_KEV)) / _BREM_MC2_KEV
    beta_i = p_i / (1.0 + T / _BREM_MC2_KEV)
    return p_i, beta_i


def _brem_incident_prefactor_core(L_ang, p_i, beta_i, Z, density_cm3):
    """Energy-independent part of ``n L dsigma/dk`` for one element.

    The remaining CUDA-cell work depends on the final-state momentum/beta and
    therefore still depends on photon energy. Hoisting this factor removes the
    incident-side Elwert exponential and ``1/p_i**2`` work from every energy
    block while preserving the Bethe-Heitler + Elwert expression algebraically.
    """
    zi = 2.0 * xp.pi * Z * ALPHA_FS
    den_i = 1.0 - xp.exp(-zi / beta_i)
    return (
        density_cm3
        * L_ang
        * 1.0e-8
        * (16.0 / 3.0)
        * ALPHA_FS
        * R_E_CM2
        * Z
        * Z
        * beta_i
        * den_i
        / (p_i * p_i)
    )


if hasattr(xp, "fuse"):
    _brem_incident_prefactor_core = xp.fuse()(_brem_incident_prefactor_core)


def _brem_incident_prefactor(L_ang, p_i, beta_i, Z, density_cm3):
    Z = REAL(Z)
    density_cm3 = REAL(density_cm3)
    return _brem_incident_prefactor_core(L_ang, p_i, beta_i, Z, density_cm3)


def _brem_dsigma_dk_core(T_i, k, Z):
    """Pure-elementwise Bethe-Heitler + Elwert core, one fused GPU kernel.

    ``T_i`` (kinetic energy [keV]) and ``k`` (photon energy [keV]) are the
    already-reshaped, mutually broadcasting operands (segments x grid); ``Z`` is
    the scalar atomic number. Split out of ``_brem_dsigma_dk`` so the whole chain
    (sqrt/divide/log/exp/where/maximum -- ~15 CuPy elementwise kernels, each
    allocating a full [seg, E] temporary and dominating ``cxr.brem`` at 300 keV)
    JIT-fuses to a SINGLE kernel under CuPy (see ``xp.fuse`` wrap below). On NumPy
    it runs eager with the identical ops, so it is bit-for-bit the old inline
    expression and the brem-spectrum golden is unchanged."""
    mc2 = _BREM_MC2_KEV
    T_f = T_i - k
    ok = (T_f > 1e-6) & (k > 0.0)  # k>0: no photon (and no 1/k blowup) at k=0
    T_f = xp.where(ok, T_f, 1e-6)

    p_i = xp.sqrt(T_i * (T_i + 2.0 * mc2)) / mc2
    p_f = xp.sqrt(T_f * (T_f + 2.0 * mc2)) / mc2
    beta_i = p_i / (1.0 + T_i / mc2)
    beta_f = p_f / (1.0 + T_f / mc2)

    born_log = xp.log((p_i + p_f) / xp.maximum(p_i - p_f, 1e-30))
    elwert = (
        beta_i
        / beta_f
        * (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_i))
        / (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_f))
    )

    dsig = (
        16.0
        / 3.0
        * ALPHA_FS
        * R_E_CM2
        * Z**2
        / xp.maximum(k * 1e3, 1e-30)
        / p_i**2
        * born_log
        * elwert
    )  # per eV
    return xp.where(ok, dsig, 0.0)


if hasattr(xp, "fuse"):  # CuPy exposes fuse(); NumPy/dpnp do not -> eager fallback
    _brem_dsigma_dk_core = xp.fuse()(_brem_dsigma_dk_core)


def _brem_dsigma_dk(Z, T_keV, k_eV):
    """
    Bremsstrahlung cross section differential in photon energy,
    dsigma/dk [cm^2/eV]: nonrelativistic Bethe-Heitler in Born approximation
    with the Elwert Coulomb correction (cf. Koch & Motz, Rev. Mod. Phys. 31,
    920 (1959)), evaluated with relativistic electron momenta:

        dsigma/dk = (16/3) alpha r_e^2 Z^2 (1/k) (1/p_i^2)
                    ln[(p_i+p_f)/(p_i-p_f)] * f_Elwert,
        f_Elwert  = (beta_i/beta_f) (1-exp(-2 pi Z alpha/beta_i))
                                  / (1-exp(-2 pi Z alpha/beta_f)),

    with p in units of m_e c. Broadcasts T_keV (segments) against k_eV
    (spectral grid); zero where k >= T. Adequate for Z <~ 30 and
    T <~ 100 keV; swap in Seltzer-Berger tables for better accuracy. The
    elementwise math lives in the fused ``_brem_dsigma_dk_core``.

    Validation: brem-spectrum
    """
    T_i = xp.asarray(T_keV, dtype=REAL)[:, None]
    k = xp.asarray(k_eV, dtype=REAL)[None, :] / 1e3  # keV
    # Z must arrive dtype-tagged, not as a bare Python int: cupy.fuse types an
    # untyped scalar operand by value (min_scalar_type), so the
    # 16/3*alpha*r_e^2 prefactor -- a scalar*scalar product with Z**2 -- gets
    # inferred as float16 and flushes ~3e-27 to zero, silently zeroing the
    # whole cross section on every fused (non-raw-kernel) GPU path.
    Z = REAL(Z)
    return _brem_dsigma_dk_core(T_i, k, Z)


def _prepare_eedl_grid(
    table: BremsstrahlungCrossSectionTable,
    photon_energy_eV,
) -> _PreparedEEDLGrid:
    """Stage an EEDL table and evaluate its unit-base panels on one output grid.

    The photon grid is invariant across every segment chunk. Evaluating and
    transferring these panel rows here avoids repeating that work inside
    ``_eedl_brem_dsigma_dk`` for every chunk.

    Validation: brem-spectrum
    """
    photon_energy_host = np.asarray(_to_cpu(photon_energy_eV), dtype=float)
    panels = _unit_base_panels(table)
    width = panels.photon_max_eV - panels.photon_min_eV
    tip_density = panels.scaled_density[:, -1] / width
    # Runtime interpolation between adjacent sub-panels is Cartesian. Holding
    # each sub-panel at its endpoint density up to the next endpoint keeps the
    # mixture continuous for lower-endpoint < k <= T instead of dropping the
    # lower share there; the segment normalization adds that probability.
    extension_end = np.append(panels.photon_max_eV[1:], panels.photon_max_eV[-1])
    panel_probability_host = np.stack(
        [
            np.where(
                (photon_energy_host > low + span) & (photon_energy_host <= end),
                tip,
                np.interp((photon_energy_host - low) / span, panels.x, scaled, left=0.0, right=0.0)
                / span,
            )
            for low, span, end, tip, scaled in zip(
                panels.photon_min_eV,
                width,
                extension_end,
                tip_density,
                panels.scaled_density,
                strict=True,
            )
        ],
        axis=0,
    )

    def staged(values):
        return xp.ascontiguousarray(xp.asarray(values, dtype=REAL))

    return _PreparedEEDLGrid(
        table=table,
        total_energy_eV=staged(table.incident_energy_eV),
        total_cross_section_cm2=staged(table.total_cross_section_cm2),
        incident_panel_energy_eV=staged(panels.incident_energy_eV),
        photon_probability_on_grid_per_eV=staged(panel_probability_host),
        unit_base_x=staged(panels.x),
        panel_photon_min_eV=staged(panels.photon_min_eV),
        panel_photon_max_eV=staged(panels.photon_max_eV),
        panel_scaled_density=staged(panels.scaled_density),
        panel_scaled_cumulative=staged(panels.scaled_cumulative),
        panel_tip_density_per_eV=staged(tip_density),
        minimum_incident_energy_eV=max(
            float(table.incident_energy_eV[0]),
            float(table.distribution_incident_energy_eV[0]),
        ),
        maximum_incident_energy_eV=min(
            float(table.incident_energy_eV[-1]),
            float(table.distribution_incident_energy_eV[-1]),
        ),
    )


def _panel_cdf_at(prepared: _PreparedEEDLGrid, panel, photon_eV):
    """Probability below ``photon_eV`` in each selected unit-base panel.

    Validation: brem-spectrum
    """
    low = prepared.panel_photon_min_eV[panel]
    high = prepared.panel_photon_max_eV[panel]
    query = xp.clip((photon_eV - low) / (high - low), REAL(0.0), REAL(1.0))
    x = prepared.unit_base_x
    index = xp.clip(xp.searchsorted(x, query), 1, x.size - 1)
    x0 = x[index - 1]
    y0 = prepared.panel_scaled_density[panel, index - 1]
    y1 = prepared.panel_scaled_density[panel, index]
    dx = query - x0
    return (
        prepared.panel_scaled_cumulative[panel, index - 1]
        + y0 * dx
        + REAL(0.5) * (y1 - y0) * dx * dx / (x[index] - x0)
    )


def _prepare_eedl_segment_state(
    prepared: _PreparedEEDLGrid,
    T_keV,
) -> _EEDLSegmentState:
    """Prepare total/panel interpolation and exact cutoff normalization once."""
    incident_eV = xp.asarray(T_keV, dtype=REAL) * REAL(1.0e3)
    available = (incident_eV >= REAL(prepared.minimum_incident_energy_eV)) & (
        incident_eV <= REAL(prepared.maximum_incident_energy_eV)
    )
    safe_incident_eV = xp.clip(
        incident_eV,
        REAL(prepared.minimum_incident_energy_eV),
        REAL(prepared.maximum_incident_energy_eV),
    )

    panel_energy = prepared.incident_panel_energy_eV
    panel_index = xp.clip(
        xp.searchsorted(panel_energy, safe_incident_eV),
        1,
        panel_energy.size - 1,
    )
    lower_panel = panel_index - 1
    lower_energy = panel_energy[lower_panel]
    upper_energy = panel_energy[panel_index]
    fraction = (safe_incident_eV - lower_energy) / (upper_energy - lower_energy)

    lower_cdf = _panel_cdf_at(
        prepared, lower_panel, safe_incident_eV
    ) + prepared.panel_tip_density_per_eV[lower_panel] * xp.maximum(
        safe_incident_eV - prepared.panel_photon_max_eV[lower_panel], REAL(0.0)
    )
    upper_cdf = _panel_cdf_at(prepared, panel_index, safe_incident_eV)
    normalization = lower_cdf + fraction * (upper_cdf - lower_cdf)

    total_energy = prepared.total_energy_eV
    total_sigma = prepared.total_cross_section_cm2
    total_index = xp.clip(
        xp.searchsorted(total_energy, safe_incident_eV),
        1,
        total_energy.size - 1,
    )
    x0 = total_energy[total_index - 1]
    x1 = total_energy[total_index]
    s0 = total_sigma[total_index - 1]
    s1 = total_sigma[total_index]
    sigma = s0 + (safe_incident_eV - x0) / (x1 - x0) * (s1 - s0)

    return _EEDLSegmentState(
        incident_energy_eV=xp.ascontiguousarray(incident_eV, dtype=REAL),
        available=xp.ascontiguousarray(available),
        lower_panel=xp.ascontiguousarray(lower_panel, dtype=np.uint32),
        panel_fraction=xp.ascontiguousarray(fraction, dtype=REAL),
        differential_scale_cm2=xp.ascontiguousarray(
            sigma / xp.maximum(normalization, REAL(1.0e-30)),
            dtype=REAL,
        ),
    )


def _slice_eedl_state(state: _EEDLSegmentState, selection) -> _EEDLSegmentState:
    return _EEDLSegmentState(
        incident_energy_eV=state.incident_energy_eV[selection],
        available=state.available[selection],
        lower_panel=state.lower_panel[selection],
        panel_fraction=state.panel_fraction[selection],
        differential_scale_cm2=state.differential_scale_cm2[selection],
    )


def _evaluate_prepared_eedl(
    prepared: _PreparedEEDLGrid,
    state: _EEDLSegmentState,
    photon_energy_eV,
):
    """Evaluate staged EEDL data; the portable path's only dense operation."""
    photon_eV = xp.asarray(photon_energy_eV, dtype=REAL)
    lower_pdf = prepared.photon_probability_on_grid_per_eV[state.lower_panel]
    upper_pdf = prepared.photon_probability_on_grid_per_eV[state.lower_panel + np.uint32(1)]
    mixed_pdf = lower_pdf + state.panel_fraction[:, None] * (upper_pdf - lower_pdf)
    physical = (photon_eV[None, :] > REAL(0.0)) & (
        photon_eV[None, :] <= state.incident_energy_eV[:, None]
    )
    return xp.where(
        physical,
        mixed_pdf * state.differential_scale_cm2[:, None],
        REAL(0.0),
    )


def _eedl_brem_dsigma_dk(
    element: str,
    T_keV,
    k_eV,
    *,
    table: BremsstrahlungCrossSectionTable | None = None,
    prepared: _PreparedEEDLGrid | None = None,
    state: _EEDLSegmentState | None = None,
):
    """EEDL differential cross section ``sigma(T) P(k|T)`` [cm²/eV].

    MF=23 and the MF=26 photon axis are interpolated lin-lin as declared.
    Between the decade-spaced incident panels the declared fixed-photon-energy
    law is replaced by unit-base refinement (``_unit_base_panels``); runtime
    interpolation between the refined sub-panels is at fixed photon energy.
    The mixed density is cut off at ``k <= T`` and renormalized analytically.

    Validation: brem-spectrum
    """
    if prepared is None:
        if table is None:
            table = load_bremsstrahlung_cross_sections(element)
        prepared = _prepare_eedl_grid(table, k_eV)
    if state is None:
        state = _prepare_eedl_segment_state(prepared, T_keV)
    return _evaluate_prepared_eedl(prepared, state, k_eV)


def _validate_bremsstrahlung_model(cross_section_model: str) -> BremsstrahlungModel:
    if cross_section_model not in {"eedl", "bethe-heitler", "bremslib"}:
        raise ValueError(
            "cross_section_model must be 'eedl', 'bethe-heitler', or 'bremslib'; "
            f"got {cross_section_model!r}"
        )
    return cross_section_model


def _require_bremslib_tables(
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable] | None,
) -> Mapping[str, BremsLibBremsstrahlungTable]:
    if bremslib_tables is None:
        raise ValueError(
            "cross_section_model='bremslib' needs bremslib_tables; resolve them with "
            "pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables(elements)"
        )
    return bremslib_tables


def _warn_bremslib_element_missing(element: str, stacklevel: int) -> None:
    warnings.warn(
        f"no BremsLib table supplied for {element}; its bremsstrahlung falls back to "
        "isotropic EEDL emission",
        RuntimeWarning,
        stacklevel=stacklevel + 1,
    )


def _warn_bremslib_out_of_range(table: BremsLibBremsstrahlungTable, stacklevel: int) -> None:
    warnings.warn(
        f"Z={table.atomic_number} incident energy is outside the BremsLib range "
        f"[{table.minimum_incident_energy_keV:g}, {table.maximum_incident_energy_keV:g}] keV; "
        "falling back to isotropic EEDL emission for those segments",
        RuntimeWarning,
        stacklevel=stacklevel + 1,
    )


def _bremsstrahlung_dsigma_dk(
    element: str,
    T_keV,
    k_eV,
    *,
    cross_section_model: BremsstrahlungModel = "eedl",
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable] | None = None,
):
    """Select EEDL (default), BremsLib, or the legacy Bethe--Heitler ``dsigma/dk``.

    ``"bremslib"`` evaluates the BremsLib SDCS; an element absent from
    ``bremslib_tables``, or an incident energy outside its table, falls back to
    EEDL with a ``RuntimeWarning``.

    Validation: brem-source-comparison
    Validation: brem-spectrum
    Validation: bremslib-angular-model
    """
    model = _validate_bremsstrahlung_model(cross_section_model)
    try:
        atomic_number = Z_TABLE[element]
    except KeyError as exc:
        raise ValueError(f"unknown element {element!r}") from exc
    if model == "bethe-heitler":
        return _brem_dsigma_dk(atomic_number, T_keV, k_eV)
    if model == "bremslib":
        table = _require_bremslib_tables(bremslib_tables).get(element)
        if table is None:
            _warn_bremslib_element_missing(element, stacklevel=2)
        else:
            staged = stage_bremslib_table(table)
            state = bremslib_segment_state(staged, T_keV)
            bremslib = evaluate_bremslib(staged, state, k_eV)
            if bool(np.all(np.asarray(_to_cpu(state.available), dtype=bool))):
                return bremslib
            _warn_bremslib_out_of_range(table, stacklevel=2)
            eedl = _bremsstrahlung_dsigma_dk(element, T_keV, k_eV, cross_section_model="eedl")
            return xp.where(state.available[:, None], bremslib, eedl)

    try:
        table = load_bremsstrahlung_cross_sections(element)
    except EEDLBremsstrahlungDataUnavailable as exc:
        warnings.warn(
            f"{exc}; falling back to Bethe-Heitler for {element}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _brem_dsigma_dk(atomic_number, T_keV, k_eV)

    prepared = _prepare_eedl_grid(table, k_eV)
    state = _prepare_eedl_segment_state(prepared, T_keV)
    available_host = np.asarray(_to_cpu(state.available), dtype=bool)
    all_available = bool(np.all(available_host))
    if not all_available:
        warnings.warn(
            f"{element} incident energy is outside the EEDL range "
            f"[{prepared.minimum_incident_energy_eV:g}, "
            f"{prepared.maximum_incident_energy_eV:g}] eV; falling back to "
            "Bethe-Heitler for those segments",
            RuntimeWarning,
            stacklevel=2,
        )
    eedl = _evaluate_prepared_eedl(prepared, state, k_eV)
    if all_available:
        return eedl
    bethe_heitler = _brem_dsigma_dk(atomic_number, T_keV, k_eV)
    return xp.where(state.available[:, None], eedl, bethe_heitler)


def _prepare_mc_eedl_contexts(
    composition,
    segment_energy_keV,
    photon_energy_eV,
) -> dict[str, _ResolvedEEDLContext | None]:
    """Load/stage each element once and prepare all segment interpolation state."""
    contexts: dict[str, _ResolvedEEDLContext | None] = {}
    for element, _density in composition:
        try:
            table = load_bremsstrahlung_cross_sections(element)
        except EEDLBremsstrahlungDataUnavailable as exc:
            warnings.warn(
                f"{exc}; falling back to Bethe-Heitler for {element}",
                RuntimeWarning,
                stacklevel=3,
            )
            contexts[element] = None
            continue
        prepared = _prepare_eedl_grid(table, photon_energy_eV)
        state = _prepare_eedl_segment_state(prepared, segment_energy_keV)
        available_host = np.asarray(_to_cpu(state.available), dtype=bool)
        all_available = bool(np.all(available_host))
        if not all_available:
            warnings.warn(
                f"{element} incident energy is outside the EEDL range "
                f"[{prepared.minimum_incident_energy_eV:g}, "
                f"{prepared.maximum_incident_energy_eV:g}] eV; falling back to "
                "Bethe-Heitler for those segments",
                RuntimeWarning,
                stacklevel=3,
            )
        contexts[element] = _ResolvedEEDLContext(
            prepared=prepared,
            state=state,
            all_available=all_available,
        )
    return contexts


def mc_brem_spectrum(
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
    cross_section_model: BremsstrahlungModel | Literal["auto"] = "auto",
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable] | None = None,
):
    """Return the incoherent bremsstrahlung density from transport segments.

    The result is the Beer--Lambert-attenuated track-length estimate in photons
    per eV per sr per incident electron toward ``n_hat``. EEDL MF=23/MT=527
    totals and MF=26/MT=527 photon distributions emit isotropically and are
    the ``"auto"`` fallback; the retained Bethe--Heitler backend is selectable and
    supplies missing-coverage fallback. ``"bremslib"`` weights each segment by
    the BremsLib double differential cross section at its emission angle
    ``arccos(v_hat . n_hat)``.
    Source equations, geometry assumptions, interpolation rules, and limiting
    cases are documented in ``docs/physics/radiation-physics/bremsstrahlung.md``.

    Parameters
    ----------
    segments
        Transport output mapping from :func:`simulate_trajectories`.
    E_grid_eV
        One-dimensional continuum photon-energy grid in eV.
    element, n_atoms_per_ang3
        Elemental target symbol and number density, superseded by ``composition``.
    theta_obs_rad
        Polar observation angle in radians when ``n_hat`` is absent.
    n_hat
        Optional three-component observation direction in the sample frame.
    chunk
        Maximum transport segments processed per reduction chunk.
    composition
        Compound ``(element, number_density)`` pairs in atoms per cubic angstrom.
    layers
        Optional film-first absorber stack.
    groove
        Optional supported blazed-groove escape geometry.
    electron_limit
        Optional leading macro-electron count used for normalization.
    E_cut_keV
        Optional post-transport electron-energy cutoff in keV.
    cross_section_model
        ``"auto"`` (default) uses the direction-resolved BremsLib model when
        a table resolves for every composition element (the supplied
        ``bremslib_tables``, else the installed release), and otherwise warns
        and uses ``"eedl"``. Every model needs segment directions ``v_hat``
        for the segment escape integral and raises ``ValueError`` without them.
        ``"eedl"`` selects evaluated MF=23/527 totals and normalized
        MF=26/527 photon spectra, ``"bethe-heitler"`` the retained
        analytic Bethe--Heitler + Elwert backend, and ``"bremslib"`` the
        direction-resolved BremsLib model. Missing EEDL coverage warns and
        falls back to Bethe--Heitler for affected elements or segments; an
        element without a BremsLib table, or a segment energy outside it,
        warns and falls back to isotropic EEDL.
    bremslib_tables
        Required with ``"bremslib"``: staged tables keyed by element symbol,
        from :func:`pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables`.

    Returns
    -------
    numpy.ndarray
        Continuum density in photons per incident electron per eV per sr.

    Raises
    ------
    ValueError
        If composition or requested escape geometry is inconsistent.

    Validation: brem-spectrum, finite-transverse-crystal, blazed-groove-geometry,
    bremslib-angular-model

    Validation: segment-escape-average
    Validation: substep-radiation-invariance
    """
    if segments.get("radiative", {}).get("model") == "bremslib-soft-hard":
        raise ValueError(
            "coupled radiative tracks require mc_coupled_brem_spectrum (or the soft and hard "
            "event scorers)"
        )
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    if cross_section_model == "auto":
        cross_section_model, bremslib_tables = resolve_auto_model(segments, comp, bremslib_tables)
    cross_section_model = _validate_bremsstrahlung_model(cross_section_model)
    supplied_bremslib = (
        _require_bremslib_tables(bremslib_tables) if cross_section_model == "bremslib" else {}
    )
    segments = _clip_segments_to_cutoff(segments, E_cut_keV, comp, layers)
    if electron_limit is None:
        Ne = segments["Ne"]
    else:
        Ne = electron_limit

    n_hat = _observation_direction(theta_obs_rad, n_hat)
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        _validate_groove_escape_direction(n_hat, groove)

    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    # (NE,) [1/Ang], single-slab fallback. EPDL covers 1 eV - 100 GeV; a node
    # outside it fails closed rather than escaping with unit transmission.
    mu = _finite_mu_or_raise(_mu_total_inv_ang(comp, E_grid), E_grid, "mc_brem_spectrum E_grid_eV")
    # layered (film-on-substrate) absorber: precompute each layer's mu(E_grid);
    # the per-segment z-path dz folds in inside the chunk loop. None -> single slab.
    if layers is not None:
        layer_mu = [
            _finite_mu_or_raise(_mu_total_inv_ang(c, E_grid), E_grid, "mc_brem_spectrum E_grid_eV")
            for (_, _, c) in layers
        ]

    seg_elec_id = xp.asarray(segments["elec_id"])
    # CuPy boolean gathers resolve the survivor count on the host every time.
    # Resolve once, then reuse the known-size integer index for all segment
    # arrays (including the finite-footprint escape distance below).
    brem_idx = xp.flatnonzero(seg_elec_id < Ne)

    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[brem_idx]
    # The row's path integral is a one-point quadrature of n * dsigma/dk(E(s))
    # over its length, so evaluate it at the propagator's representative energy
    # when transport supplied one. That makes an unsplit flight a midpoint rule
    # (second order in its length) instead of a left-endpoint rule (first
    # order), which is what makes the yield insensitive to how many numerical
    # substeps the flight was integrated in. Frozen rows carry no representative
    # energy and keep the historical start-energy evaluation bit-for-bit.
    # Validation: substep-radiation-invariance
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    seg_E = xp.asarray(segments[E_field], dtype=REAL)[brem_idx]
    owner, fraction, path_start, path_end = segment_escape_paths(
        segments, brem_idx, n_hat, layers=layers, groove=groove, xp=xp
    )
    seg_L = seg_L[owner] * fraction
    seg_E = seg_E[owner]
    # BremsLib keeps EEDL staged too: it is the isotropic fallback for an
    # element without a table and for segments outside a table's energy range.
    eedl_contexts = (
        _prepare_mc_eedl_contexts(comp, seg_E, E_grid)
        if cross_section_model in {"eedl", "bremslib"}
        else {}
    )
    bremslib_staged = {}
    bremslib_covers: dict[str, bool] = {}
    if cross_section_model == "bremslib":
        v_hat = xp.asarray(segments["v_hat"], dtype=REAL)[brem_idx][owner]
        seg_cos_theta = v_hat @ xp.asarray(n_hat, dtype=REAL)
        for el_i, _ in comp:
            table = supplied_bremslib.get(el_i)
            if table is None:
                _warn_bremslib_element_missing(el_i, stacklevel=2)
                continue
            covers = not seg_E.size or bool(
                (float(_to_cpu(seg_E.min())) >= table.minimum_incident_energy_keV)
                and (float(_to_cpu(seg_E.max())) <= table.maximum_incident_energy_keV)
            )
            if not covers:
                _warn_bremslib_out_of_range(table, stacklevel=2)
            bremslib_staged[el_i] = stage_bremslib_table(table)
            bremslib_covers[el_i] = covers

    spec = xp.zeros(E_grid.size, dtype=REAL)

    # GPU float32 fast paths fuse cross-section evaluation, absorption, and the
    # segment reduction. Instead of dense ``(segment, energy)`` matrices, they
    # pass only O(Nsegment) interpolation/geometry state and small staged EEDL
    # panels. The EEDL reducer is CUDA-only; other accelerators retain the
    # conservative chunked path below.
    _use_bethe_heitler_jit = (
        _USE_JIT_BREM_REDUCTION
        and cross_section_model == "bethe-heitler"
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    _use_eedl_jit = (
        _USE_JIT_BREM_REDUCTION
        and cross_section_model == "eedl"
        and BACKEND.name == "cuda"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    _use_bremslib_jit = (
        _USE_JIT_BREM_REDUCTION
        and cross_section_model == "bremslib"
        and BACKEND.name == "cuda"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_bethe_heitler_jit or _use_eedl_jit or _use_bremslib_jit:
        from .brem_jit_kernel import (
            DEFAULT_BREM_KERNEL_CONFIG,
            run_brem_reduction_kernel,
            run_bremslib_element_reduction,
            run_eedl_brem_reduction_kernel,
        )

        mu_by_layer = mu[None, :] if layers is None else xp.stack(layer_mu, axis=0)
        path_by_layer = xp.ascontiguousarray(xp.stack((path_start, path_end), axis=1), dtype=REAL)
        mu_by_layer = xp.ascontiguousarray(mu_by_layer, dtype=REAL)
        path_flat = path_by_layer.reshape(-1)
        mu_flat = mu_by_layer.reshape(-1)
        T_jit = xp.ascontiguousarray(seg_E, dtype=REAL)
        L_jit = xp.ascontiguousarray(seg_L, dtype=REAL)
        E_jit = xp.ascontiguousarray(E_grid, dtype=REAL)
        p_i_jit, beta_i_jit = _brem_incident_state(T_jit)
        p_i_jit = xp.ascontiguousarray(p_i_jit, dtype=REAL)
        beta_i_jit = xp.ascontiguousarray(beta_i_jit, dtype=REAL)
        n_abs_layers = int(path_by_layer.shape[2])

        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            incident_prefactor = xp.ascontiguousarray(
                _brem_incident_prefactor(
                    L_jit,
                    p_i_jit,
                    beta_i_jit,
                    Z_i,
                    n_i * 1e24,
                ),
                dtype=REAL,
            )
            context = eedl_contexts.get(el_i)
            staged = bremslib_staged.get(el_i)
            if cross_section_model == "bremslib" and staged is not None:
                run_bremslib_element_reduction(
                    T_jit,
                    L_jit,
                    p_i_jit,
                    incident_prefactor,
                    path_flat,
                    mu_flat,
                    E_jit,
                    staged,
                    seg_cos_theta,
                    context,
                    Z=Z_i,
                    number_density_ang3=n_i,
                    n_layers=n_abs_layers,
                    out=spec,
                )
            elif cross_section_model in {"eedl", "bremslib"} and context is not None:
                state = context.state
                eedl_incident_weight = xp.ascontiguousarray(
                    n_i * REAL(1.0e24) * L_jit * REAL(1.0e-8) * state.differential_scale_cm2,
                    dtype=REAL,
                )
                run_eedl_brem_reduction_kernel(
                    T_jit,
                    p_i_jit,
                    incident_prefactor,
                    eedl_incident_weight,
                    xp.ascontiguousarray(state.lower_panel, dtype=np.uint32),
                    xp.ascontiguousarray(state.panel_fraction, dtype=REAL),
                    xp.ascontiguousarray(state.available, dtype=REAL),
                    path_flat,
                    mu_flat,
                    E_jit,
                    context.prepared.photon_probability_on_grid_per_eV.reshape(-1),
                    Z=Z_i,
                    n_layers=n_abs_layers,
                    out=spec,
                )
            else:
                run_brem_reduction_kernel(
                    T_jit,
                    L_jit,
                    path_flat,
                    mu_flat,
                    E_jit,
                    Z=Z_i,
                    density_cm3=n_i * 1e24,
                    n_layers=n_abs_layers,
                    p_i=p_i_jit,
                    incident_prefactor=incident_prefactor,
                    out=spec,
                    config=DEFAULT_BREM_KERNEL_CONFIG,
                )
        return _to_cpu(spec / (4.0 * xp.pi) / Ne)

    M = seg_E.size
    if cross_section_model == "bremslib":
        # The direction-resolved evaluation holds several (segment, energy)
        # temporaries at once; bound them to a few million cells per chunk.
        chunk = max(1, min(int(chunk), _BREMSLIB_CHUNK_CELLS // max(int(E_grid.size), 1)))
    for j0 in range(0, M, chunk):
        sl = slice(j0, min(j0 + chunk, M))
        mu_by_layer = mu[None, :] if layers is None else xp.stack(layer_mu, axis=0)
        T_abs = mean_transmission(path_start[sl] @ mu_by_layer, path_end[sl] @ mu_by_layer, xp=xp)
        path_cm = seg_L[sl] * 1e-8
        for el_i, n_i in comp:
            context = eedl_contexts.get(el_i)
            staged = bremslib_staged.get(el_i)
            bremslib_state = None
            if staged is not None:
                bremslib_state = bremslib_segment_state(staged, seg_E[sl], seg_cos_theta[sl])
                # 4 pi d2sigma/(dk dOmega): the common 1/(4 pi) below then
                # leaves the direction-resolved density per steradian.
                dsig = REAL(4.0 * np.pi) * evaluate_bremslib(staged, bremslib_state, E_grid)
                if bremslib_covers[el_i]:
                    spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)
                    continue
                directional = dsig
            if context is not None:
                state = _slice_eedl_state(context.state, sl)
                dsig = _evaluate_prepared_eedl(context.prepared, state, E_grid)
                if not context.all_available:
                    bethe_heitler = _brem_dsigma_dk(
                        TRANSPORT_ELEMENTS[el_i]["Z"],
                        seg_E[sl],
                        E_grid,
                    )
                    dsig = xp.where(state.available[:, None], dsig, bethe_heitler)
            else:
                dsig = _brem_dsigma_dk(
                    TRANSPORT_ELEMENTS[el_i]["Z"],
                    seg_E[sl],
                    E_grid,
                )
            if bremslib_state is not None:
                dsig = xp.where(bremslib_state.available[:, None], directional, dsig)
            spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)
    return _to_cpu(spec / (4.0 * xp.pi) / Ne)


def load_external_brem(path, E_grid_eV):
    """
    Interpolate an EXTERNAL bremsstrahlung background onto the spectral grid
    -- e.g. a NIST DTSA-II simulation, which is what Zhai et al. use both
    for their simulated backgrounds (refs 96-100) and, with a PIXE-style
    numerical fit, for their experimental subtraction (SI S3).

    File format: two columns (energy [eV], intensity), whitespace- or
    comma-separated; '#' comment lines and non-numeric headers are skipped.
    The intensity must already be in DETECTED units matching your plots
    (e.g. Phs/eV/s/nA: from a DTSA-II counts export, divide counts/channel
    by channel width [eV] x live time [s] x beam current [nA]). It is
    treated as an as-detected spectrum: window efficiency and detector
    resolution are NOT re-applied. Energies outside the file's range
    interpolate to zero.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].startswith(("#", "//")):
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # header / text line
    if not rows:
        raise ValueError(f"no numeric (E, intensity) rows found in {path}")
    arr = np.array(sorted(rows))
    return np.interp(np.asarray(E_grid_eV, dtype=float), arr[:, 0], arr[:, 1], left=0.0, right=0.0)
