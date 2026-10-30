"""BremsLib bremsstrahlung: energy-differential and direction-resolved cross sections.

BremsLib v2.0 (Poškus, At. Data Nucl. Data Tables 166, 101734 (2025)) tabulates
the scaled single differential cross section (SDCS) ``chi = (k/Z**2) dsigma/dk``
and, at every ``(T1, k/T1)`` grid node, the scaled double differential cross
section ``(k/Z**2) d2sigma/(dk dOmega)`` over the photon emission angle
``theta`` measured from the incident electron direction. This module turns one
element's stored table into both cross sections at arbitrary ``(T, k, theta)``.

It holds no I/O. The physics core may not import :mod:`pyrite.xsgen`, so a
driver resolves the stored table and hands its arrays to
:func:`prepare_bremslib_table`; :mod:`pyrite.xsgen.bremslib.tables` does that
for the built-in catalogue and for locally generated tables.

The interpolation is PyRITE's own, not a port of upstream ``Interpolate_DCS``
(GPL-3). At each node the DDCS is renormalized so that the solid-angle integral
of its linear-in-``theta`` interpolant equals that node's SDCS exactly: the
angular shape is taken against its own parent SDCS, as the library defines it.
At load, each ``T1`` interval is refined with geometric sub-nodes (see
:func:`pyrite._bremslib_table._refine_incident_grid`); at evaluation the
scaled DDCS is interpolated linearly in ``k/T`` and in ``ln T`` on that
refined grid. Every such
combination is convex, so the angular integral of the
interpolated DDCS is exactly the identically interpolated SDCS -- integrating
the direction-resolved result over ``4 pi`` recovers the energy spectrum by
construction, not by a separate renormalization.

Source equations, assumptions, and limiting cases are documented in
``docs/physics/radiation-physics/bremsstrahlung.md``.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._backend import REAL, _to_cpu, xp
from ..._bremslib_table import (
    BREMSSTRAHLUNG_BREMSLIB_MODEL,
    STAGED_RATIO_COUNT,
    BremsLibBremsstrahlungTable,
    prepare_bremslib_table,
    solid_angle_integral,
    top_reduced_energy,
)
from ..._physics_fallback import PhysicsFallbackWarning

#: One millibarn in square centimetres.
_MB_CM2 = 1.0e-27


@dataclass(frozen=True, slots=True)
class _StagedBremsLib:
    """A table's arrays on the active backend."""

    table: BremsLibBremsstrahlungTable
    log_incident_energy: Any
    inner_reduced_energy: Any
    nominal_reduced_energy: Any
    top_reduced_energy: Any
    scaled_sdcs_mb: Any
    theta_rad: Any
    scaled_ddcs_mb_sr: Any
    array_module: Any = None


def stage_bremslib_table(
    table: BremsLibBremsstrahlungTable, *, array_module: Any = None
) -> _StagedBremsLib:
    """Stage interpolation arrays once per call on the selected array namespace.

    The default retains the active backend/REAL precision. An explicit namespace
    (e.g. NumPy for the weighted photon source) uses binary64 independently of
    that backend; evaluation follows the staged namespace without hidden copies.
    """
    module = xp if array_module is None else array_module
    real = REAL if array_module is None else module.float64

    def staged(values):
        return module.ascontiguousarray(module.asarray(values, dtype=real))

    return _StagedBremsLib(
        table=table,
        log_incident_energy=staged(np.log(table.incident_energy_keV)),
        # Interior breakpoints searched for the lower node; node 0 is k/T = 0.
        inner_reduced_energy=staged(table.nominal_reduced_energy[1:-1]),
        nominal_reduced_energy=staged(table.nominal_reduced_energy),
        top_reduced_energy=staged(table.top_reduced_energy),
        scaled_sdcs_mb=staged(table.scaled_sdcs_mb),
        theta_rad=staged(table.theta_rad),
        # ``(T1, theta, k/T1)`` on the device: the per-segment gather below
        # then uses adjacent advanced indices, which dpnp requires.
        scaled_ddcs_mb_sr=staged(np.swapaxes(table.scaled_ddcs_mb_sr, 1, 2)),
        array_module=array_module,
    )


@dataclass(frozen=True, slots=True)
class _BremsLibSegmentState:
    """Per-segment incident-energy bracket and node values, O(Nsegment * 18)."""

    incident_energy_keV: Any
    available: Any
    lower_row: Any
    energy_fraction: Any
    lower_values: Any
    upper_values: Any


def _table_namespace(staged):
    return (
        (xp, REAL)
        if staged.array_module is None
        else (staged.array_module, staged.array_module.float64)
    )


def _incident_bracket(staged: _StagedBremsLib, T_keV):
    xp, REAL = _table_namespace(staged)
    T = xp.asarray(T_keV, dtype=REAL)
    log_grid = staged.log_incident_energy
    table = staged.table
    available = (T >= REAL(table.minimum_incident_energy_keV)) & (
        T <= REAL(table.maximum_incident_energy_keV)
    )
    log_T = xp.clip(xp.log(xp.maximum(T, REAL(1.0e-30))), log_grid[0], log_grid[-1])
    row = xp.clip(xp.searchsorted(log_grid, log_T, side="right") - 1, 0, log_grid.size - 2)
    fraction = (log_T - log_grid[row]) / (log_grid[row + 1] - log_grid[row])
    return T, available, row, xp.clip(fraction, REAL(0.0), REAL(1.0))


def bremslib_segment_state(staged: _StagedBremsLib, T_keV, cos_theta=None):
    """Bracket each segment in ``ln T`` and gather its 2 x 18 node values.

    With ``cos_theta`` the node values are the scaled DDCS at each segment's
    emission angle, interpolated linearly in ``theta``; without it they are
    the scaled SDCS.

    Validation: bremslib-angular-model
    """
    xp, REAL = _table_namespace(staged)
    T, available, row, fraction = _incident_bracket(staged, T_keV)
    if cos_theta is None:
        lower = staged.scaled_sdcs_mb[row]
        upper = staged.scaled_sdcs_mb[row + 1]
    else:
        theta_grid = staged.theta_rad
        theta = xp.arccos(xp.clip(xp.asarray(cos_theta, dtype=REAL), REAL(-1.0), REAL(1.0)))
        cell = xp.clip(xp.searchsorted(theta_grid, theta, side="right") - 1, 0, theta_grid.size - 2)
        weight = (theta - theta_grid[cell]) / (theta_grid[cell + 1] - theta_grid[cell])
        weight = xp.clip(weight, REAL(0.0), REAL(1.0))[:, None]
        ddcs = staged.scaled_ddcs_mb_sr

        def at_angle(rows):
            left = ddcs[rows, cell]
            right = ddcs[rows, cell + 1]
            return left + weight * (right - left)

        lower = at_angle(row)
        upper = at_angle(row + 1)
    return _BremsLibSegmentState(
        incident_energy_keV=T,
        available=available,
        lower_row=row,
        energy_fraction=fraction,
        lower_values=lower,
        upper_values=upper,
    )


def _along_reduced_energy(staged, values, top, reduced):
    """Interpolate ``(M, 18)`` node values linearly in ``k/T`` at ``(M, NE)``.

    Above the top node (``k/T`` between it and 1) the top value is held: the
    library stops a 50 eV (or 1e-4 T) outgoing electron short of the tip.
    """
    xp, REAL = _table_namespace(staged)
    inner = staged.inner_reduced_energy
    nominal = staged.nominal_reduced_energy
    lower = xp.searchsorted(inner, reduced, side="right")  # 0..16
    x0 = nominal[lower]
    x1 = xp.where(
        lower == STAGED_RATIO_COUNT - 2,
        top[:, None],
        nominal[xp.minimum(lower + 1, STAGED_RATIO_COUNT - 1)],
    )
    fraction = xp.clip((reduced - x0) / (x1 - x0), REAL(0.0), REAL(1.0))
    rows = xp.arange(values.shape[0])[:, None]
    v0 = values[rows, lower]
    v1 = values[rows, lower + 1]
    return v0 + fraction * (v1 - v0)


def _evaluate(staged: _StagedBremsLib, state: _BremsLibSegmentState, photon_eV):
    """Physical cross section at photon energies broadcast against the rows.

    ``photon_eV`` is ``(1, NE)`` for a shared grid or ``(Nsegment, 1)`` for
    one photon energy per row; the result has the broadcast shape.
    """
    xp, REAL = _table_namespace(staged)
    T_eV = state.incident_energy_keV * REAL(1.0e3)
    reduced = photon_eV / xp.maximum(T_eV, REAL(1.0e-30))[:, None]
    top = staged.top_reduced_energy
    lower = _along_reduced_energy(staged, state.lower_values, top[state.lower_row], reduced)
    upper = _along_reduced_energy(staged, state.upper_values, top[state.lower_row + 1], reduced)
    scaled = lower + state.energy_fraction[:, None] * (upper - lower)
    Z = REAL(staged.table.atomic_number)
    physical = (photon_eV > REAL(0.0)) & (reduced <= REAL(1.0))
    return xp.where(
        physical,
        scaled * REAL(_MB_CM2) * Z * Z / xp.maximum(photon_eV, REAL(1.0e-30)),
        REAL(0.0),
    )


def evaluate_bremslib(staged: _StagedBremsLib, state: _BremsLibSegmentState, photon_energy_eV):
    """Return ``d sigma/dk`` [cm²/eV] or ``d2 sigma/(dk dOmega)`` [cm²/eV/sr].

    Which one depends on whether ``state`` was built with emission angles.
    Shape ``(Nsegment, NE)``; zero outside ``0 < k <= T``.

    Validation: bremslib-angular-model
    """
    xp, REAL = _table_namespace(staged)
    return _evaluate(staged, state, xp.asarray(photon_energy_eV, dtype=REAL)[None, :])


def evaluate_bremslib_rows(staged: _StagedBremsLib, state: _BremsLibSegmentState, photon_energy_eV):
    """:func:`evaluate_bremslib` with one photon energy per row; shape ``(Nsegment,)``.

    The diagonal of the ``(Nsegment, Nsegment)`` grid evaluation, without
    forming it: row ``i`` is evaluated at ``photon_energy_eV[i]``.

    Validation: bremslib-angular-model
    """
    xp, REAL = _table_namespace(staged)
    return _evaluate(staged, state, xp.asarray(photon_energy_eV, dtype=REAL)[:, None])[:, 0]


#: Electron rest energy in eV, as in ``transport.dielectric_radiative``.
_ELECTRON_REST_EV = 510998.95


def _warn_bremslib_element_missing(element: str, stacklevel: int) -> None:
    warnings.warn(
        f"no BremsLib table supplied for {element}; its bremsstrahlung falls back to "
        "isotropic EEDL emission",
        PhysicsFallbackWarning,
        stacklevel=stacklevel + 1,
    )


def _warn_bremslib_out_of_range(table: BremsLibBremsstrahlungTable, stacklevel: int) -> None:
    warnings.warn(
        f"Z={table.atomic_number} incident energy is outside the BremsLib range "
        f"[{table.minimum_incident_energy_keV:g}, {table.maximum_incident_energy_keV:g}] keV; "
        "falling back to isotropic EEDL emission for those segments",
        PhysicsFallbackWarning,
        stacklevel=stacklevel + 1,
    )


def stage_bremslib_elements(comp, tables, seg_E_keV, *, strict=False):
    """Stage each composition element's table and whether it covers ``seg_E_keV``.

    An element without a table, or a table not covering every segment energy,
    warns and leaves those segments to the isotropic EEDL fallback. ``strict``
    (dielectric-coupled scoring) refuses that fallback instead: the transport
    that produced the segments used the suppressed BremsLib DCS throughout.
    """
    staged, covers = {}, {}
    for element, _ in comp:
        table = tables.get(element)
        if table is None:
            if strict:
                raise ValueError("dielectric-coupled scoring requires all BremsLib tables")
            _warn_bremslib_element_missing(element, stacklevel=3)
            continue
        covered = not seg_E_keV.size or bool(
            (float(_to_cpu(seg_E_keV.min())) >= table.minimum_incident_energy_keV)
            and (float(_to_cpu(seg_E_keV.max())) <= table.maximum_incident_energy_keV)
        )
        if not covered:
            if strict:
                raise ValueError("dielectric-coupled BremsLib scorer cannot fall back to EEDL")
            _warn_bremslib_out_of_range(table, stacklevel=3)
        staged[element] = stage_bremslib_table(table)
        covers[element] = covered
    return staged, covers


def dielectric_scoring_plasma(plasma_eV, cross_section_model) -> float:
    """Validate the scorer's ``hbar*omega_p`` (eV); zero is the vacuum limit."""
    plasma = float(plasma_eV)
    if not np.isfinite(plasma) or plasma < 0.0:
        raise ValueError("dielectric_plasma_eV must be finite and nonnegative")
    if plasma and cross_section_model != "bremslib":
        raise ValueError("dielectric suppression requires BremsLib scoring")
    return plasma


def dielectric_scoring_factor(T_keV, photon_eV, plasma_eV):
    """Ter--Mikaelian ``k**2/(k**2 + (gamma*Ep)**2)`` on a (segment, photon) grid.

    The factor is isotropic at fixed ``(T, k)``, so it scales the DDCS without
    changing its conditional angular shape. Backend twin of
    ``spectrum.dielectric_suppression.ter_mikaelian_factor``.

    Validation: dielectric-suppression
    """
    scale = (1.0 + T_keV[:, None] * 1e3 / _ELECTRON_REST_EV) * plasma_eV
    return xp.square(photon_eV[None, :] / xp.hypot(photon_eV[None, :], scale))


def resolve_auto_model(segments, comp, bremslib_tables):
    """Pick ``"bremslib"`` or ``"eedl"`` for ``cross_section_model="auto"``.

    BremsLib needs segment directions and a table for every composition
    element; supplied tables are used as given, else the installed release is
    resolved through the driver-side loader (imported here because the physics
    core otherwise takes its tables as arguments).
    """
    if segments.get("v_hat") is None:
        return "eedl", bremslib_tables
    elements = [element for element, _ in comp]
    if bremslib_tables is not None:
        if all(element in bremslib_tables for element in elements):
            return "bremslib", bremslib_tables
        warnings.warn(
            "bremsstrahlung_model 'auto': the supplied BremsLib tables do not cover "
            f"{', '.join(elements)}; using EEDL with an isotropic photon angle",
            PhysicsFallbackWarning,
            stacklevel=3,
        )
        return "eedl", None
    from ...xsgen.bremslib.tables import load_bremsstrahlung_tables, resolve_bremsstrahlung_model

    if resolve_bremsstrahlung_model("auto", elements) != "bremslib":
        return "eedl", None
    return "bremslib", load_bremsstrahlung_tables(elements)


__all__ = [
    "resolve_auto_model",
    "dielectric_scoring_factor",
    "dielectric_scoring_plasma",
    "stage_bremslib_elements",
    "BREMSSTRAHLUNG_BREMSLIB_MODEL",
    "BremsLibBremsstrahlungTable",
    "bremslib_segment_state",
    "evaluate_bremslib",
    "evaluate_bremslib_rows",
    "prepare_bremslib_table",
    "solid_angle_integral",
    "stage_bremslib_table",
    "top_reduced_energy",
]
