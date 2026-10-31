"""Host-side BremsLib soft/hard radiative partition and photon sampling.

Validation: bremslib-radiative-partition
"""

import mmap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..._backend import BACKEND
from ...materials._transport_data import TRANSPORT_ELEMENTS
from ..spectrum.brem_bremslib import (
    BremsLibBremsstrahlungTable,
    _StagedBremsLib,
    bremslib_segment_state,
    evaluate_bremslib,
    evaluate_bremslib_rows,
    stage_bremslib_table,
)
from .dielectric_radiative import dielectric_cell_moments, dielectric_scale_eV

_MB_CM2 = 1.0e-27
_ELECTRON_REST_EV = 510998.95
# Per-row photon vectors at least this large are mapped lazily (#825).
_LAZY_ZERO_BYTES = 64 << 20


def _cell_moments(
    left: float, right: float, chi_left: float, chi_right: float
) -> tuple[float, float]:
    """Integrate ``chi(k)/k`` and ``chi(k)`` for a linear chi cell."""
    if right <= left:
        return 0.0, 0.0
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    rate = intercept * np.log(right / left) + slope * (right - left)
    first = 0.5 * (chi_left + chi_right) * (right - left)
    return float(rate), float(first)


@dataclass(frozen=True, slots=True)
class RadiativePartition:
    """One fixed-energy, fixed-cutoff BremsLib partition for one atom.

    Cross sections are per atom. ``soft_stopping_cs_eV_cm2`` is a first
    moment, while ``hard_rate_cs_cm2`` is a zeroth moment. The caller multiplies
    by atomic number density for loss per length or collision frequency.
    """

    table: BremsLibBremsstrahlungTable
    incident_energy_eV: float
    cutoff_eV: float
    photon_grid_eV: np.ndarray
    scaled_sdcs_mb: np.ndarray
    hard_cell_cs_cm2: np.ndarray
    soft_stopping_cs_eV_cm2: float
    hard_rate_cs_cm2: float
    hard_stopping_cs_eV_cm2: float
    total_stopping_cs_eV_cm2: float
    plasma_energy_eV: float = 0.0

    def sample_photon_energy(self, uniform: float) -> float:
        """Sample a hard photon from the exact piecewise-linear SDCS CDF.

        Validation: bremslib-radiative-partition
        """
        if self.hard_rate_cs_cm2 <= 0.0:
            raise ValueError("partition has no hard photons")
        if not 0.0 <= uniform < 1.0:
            raise ValueError("uniform must be in [0, 1)")
        masses = self.hard_cell_cs_cm2
        target = uniform * self.hard_rate_cs_cm2
        cumulative = np.cumsum(masses)
        cell = min(int(np.searchsorted(cumulative, target, side="right")), masses.size - 1)
        target -= float(cumulative[cell - 1]) if cell else 0.0
        left, right = self.photon_grid_eV[cell : cell + 2]
        chi_left, chi_right = self.scaled_sdcs_mb[cell : cell + 2]
        scale = _MB_CM2 * self.table.atomic_number**2
        suppression_scale = dielectric_scale_eV(self.incident_energy_eV, self.plasma_energy_eV)
        low, high = float(left), float(right)
        for _ in range(52):
            mid = 0.5 * (low + high)
            chi_mid = chi_left + (chi_right - chi_left) * (mid - left) / (right - left)
            if suppression_scale == 0.0:
                partial, _ = _cell_moments(float(left), mid, float(chi_left), float(chi_mid))
            else:
                partial, _ = dielectric_cell_moments(
                    float(left), mid, float(chi_left), float(chi_mid), suppression_scale
                )
            if scale * partial < target:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class HardRadiativePhoton:
    energy_eV: float
    direction: np.ndarray
    electron_energy_eV: float
    electron_direction: np.ndarray
    target_momentum_eV_c: np.ndarray


def build_radiative_partition(
    table: BremsLibBremsstrahlungTable,
    incident_energy_eV: float,
    cutoff_eV: float,
    *,
    plasma_energy_eV: float = 0.0,
) -> RadiativePartition:
    """Split one BremsLib SDCS at positive ``kc`` with optional dielectric suppression.

    The parent scaled SDCS is linearly interpolated in reduced photon energy.
    Its dielectric-corrected moments are integrated analytically, *not* by
    linearly resampling the suppression factor on BremsLib's coarse photon
    nodes. The vacuum limit preserves the existing unsuppressed calculation.

    Validation: bremslib-radiative-partition, dielectric-suppression
    """
    energy = float(incident_energy_eV)
    cutoff = float(cutoff_eV)
    plasma = float(plasma_energy_eV)
    if not np.isfinite(energy) or not np.isfinite(cutoff) or energy <= 0.0 or cutoff <= 0.0:
        raise ValueError("incident energy and radiative cutoff must be positive and finite")
    if not np.isfinite(plasma) or plasma < 0.0:
        raise ValueError("plasma_energy_eV must be finite and nonnegative")
    suppression_scale = dielectric_scale_eV(energy, plasma)
    T_keV = energy / 1.0e3
    if (
        T_keV < table.minimum_incident_energy_keV
        and not np.isclose(T_keV, table.minimum_incident_energy_keV, rtol=1e-12, atol=0.0)
    ) or (
        T_keV > table.maximum_incident_energy_keV
        and not np.isclose(T_keV, table.maximum_incident_energy_keV, rtol=1e-12, atol=0.0)
    ):
        raise ValueError("incident energy is outside the BremsLib table")
    if cutoff > energy:
        raise ValueError("radiative cutoff cannot exceed incident energy")

    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(staged, [energy / 1.0e3])
    row = int(BACKEND.to_cpu(state.lower_row)[0])
    tops = np.asarray(table.top_reduced_energy, dtype=float)
    nodes = np.asarray(table.nominal_reduced_energy[:-1], dtype=float)
    reduced = np.unique(np.r_[0.0, nodes, tops[row], tops[row + 1], cutoff / energy, 1.0])
    reduced = reduced[(reduced >= 0.0) & (reduced <= 1.0)]
    photon_grid = energy * reduced
    # (kc / T) * T can round below kc; keep the partition boundary exact.
    photon_grid[reduced == cutoff / energy] = cutoff
    positive = photon_grid > 0.0
    sdcs = np.zeros_like(photon_grid)
    sdcs[positive] = BACKEND.to_cpu(evaluate_bremslib(staged, state, photon_grid[positive]))[0]
    chi = sdcs * photon_grid / (_MB_CM2 * table.atomic_number**2)
    fraction = float(BACKEND.to_cpu(state.energy_fraction)[0])
    lower = float(BACKEND.to_cpu(state.lower_values)[0, 0])
    upper = float(BACKEND.to_cpu(state.upper_values)[0, 0])
    chi[0] = lower + fraction * (upper - lower)
    if np.any(~np.isfinite(chi)) or np.any(chi < 0.0):
        raise ValueError("BremsLib SDCS is not finite and nonnegative")

    scale = _MB_CM2 * table.atomic_number**2
    soft = hard_first = 0.0
    hard_cells = np.zeros(photon_grid.size - 1, dtype=float)
    for i, (left, right) in enumerate(zip(photon_grid[:-1], photon_grid[1:], strict=True)):
        if suppression_scale == 0.0:
            if left == 0.0:
                first = 0.5 * (chi[i] + chi[i + 1]) * right
                soft += scale * first
                continue
            rate, first = _cell_moments(float(left), float(right), float(chi[i]), float(chi[i + 1]))
        else:
            rate, first = dielectric_cell_moments(
                float(left), float(right), float(chi[i]), float(chi[i + 1]), suppression_scale
            )
        if right <= cutoff:
            soft += scale * first
        else:
            hard_cells[i] = scale * rate
            hard_first += scale * first
    hard_rate = float(hard_cells.sum())
    return RadiativePartition(
        table,
        energy,
        cutoff,
        photon_grid,
        chi,
        hard_cells,
        soft,
        hard_rate,
        hard_first,
        soft + hard_first,
        plasma,
    )


def sample_hard_radiative_photon(
    partition: RadiativePartition,
    electron_direction: np.ndarray,
    energy_uniform: float,
    angle_uniform: float,
    azimuth_uniform: float,
) -> HardRadiativePhoton:
    """Sample photon direction from the parent DDCS and account for momentum.

    The electron retains its direction, as in the existing segment contract;
    residual momentum is assigned to an infinitely heavy target and its recoil
    energy is neglected. BremsLib does not provide a joint photon/electron
    angular distribution. Photon propagation and downstream absorption are
    outside this host-side event contract.

    Validation: bremslib-radiative-partition
    """
    energy = partition.sample_photon_energy(energy_uniform)
    return hard_radiative_photon_at_energy(
        partition.table,
        partition.incident_energy_eV,
        energy,
        electron_direction,
        angle_uniform,
        azimuth_uniform,
    )


def hard_radiative_photon_at_energy(
    table: BremsLibBremsstrahlungTable,
    incident_energy_eV: float,
    photon_energy_eV: float,
    electron_direction: np.ndarray,
    angle_uniform: float,
    azimuth_uniform: float,
    *,
    _staged_table: _StagedBremsLib | None = None,
) -> HardRadiativePhoton:
    """Complete a sampled transport photon with its conditional BremsLib angle.

    The energy has already been drawn from the same table's hard SDCS. The
    angular CDF integrates its DDCS at that energy, and an independent uniform
    azimuth completes the photon direction. Validation: bremslib-radiative-partition.
    """
    if not 0.0 <= angle_uniform < 1.0 or not 0.0 <= azimuth_uniform < 1.0:
        raise ValueError("angle uniforms must be in [0, 1)")
    energy = float(photon_energy_eV)
    incident = float(incident_energy_eV)
    if not np.isfinite(incident) or not np.isfinite(energy) or not 0.0 < energy <= incident:
        raise ValueError("photon energy must be positive and at most the incident energy")
    incoming = np.asarray(electron_direction, dtype=float)
    if (
        incoming.shape != (3,)
        or not np.all(np.isfinite(incoming))
        or not np.isclose(np.linalg.norm(incoming), 1.0, atol=1e-10)
    ):
        raise ValueError("electron_direction must be a finite unit vector")
    theta = np.asarray(table.theta_rad, dtype=float)
    staged = stage_bremslib_table(table) if _staged_table is None else _staged_table
    state = bremslib_segment_state(staged, np.full(theta.size, incident / 1e3), np.cos(theta))
    ddcs = BACKEND.to_cpu(evaluate_bremslib(staged, state, [energy]))[:, 0]
    # Exact integral of each linear-in-theta cell; the same rule normalizes DDCS.
    widths = np.diff(theta)
    a, b = theta[:-1], theta[1:]
    cell_mass = (
        2.0
        * np.pi
        * (
            ddcs[:-1] * (np.cos(a) - np.cos(b))
            + np.diff(ddcs) / widths * (np.sin(b) - np.sin(a) - widths * np.cos(b))
        )
    )
    cumulative = np.cumsum(cell_mass)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ValueError("BremsLib DDCS has no positive angular integral")
    target = angle_uniform * float(cumulative[-1])
    cell = min(int(np.searchsorted(cumulative, target, side="right")), cell_mass.size - 1)
    target -= float(cumulative[cell - 1]) if cell else 0.0
    low, high = float(theta[cell]), float(theta[cell + 1])
    for _ in range(52):
        mid = 0.5 * (low + high)
        partial = (
            2.0
            * np.pi
            * (
                ddcs[cell] * (np.cos(a[cell]) - np.cos(mid))
                + (ddcs[cell + 1] - ddcs[cell])
                / widths[cell]
                * (np.sin(mid) - np.sin(a[cell]) - (mid - a[cell]) * np.cos(mid))
            )
        )
        if partial < target:
            low = mid
        else:
            high = mid
    polar = 0.5 * (low + high)
    reference = np.array([1.0, 0.0, 0.0]) if abs(incoming[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    transverse = np.cross(incoming, reference)
    transverse /= np.linalg.norm(transverse)
    other = np.cross(incoming, transverse)
    phi = 2.0 * np.pi * azimuth_uniform
    photon_direction = np.cos(polar) * incoming + np.sin(polar) * (
        np.cos(phi) * transverse + np.sin(phi) * other
    )
    remaining = incident - energy
    p_in = np.sqrt(incident * (incident + 2.0 * _ELECTRON_REST_EV))
    p_out = np.sqrt(remaining * (remaining + 2.0 * _ELECTRON_REST_EV))
    electron_out = incoming.copy()
    target_momentum = p_in * incoming - energy * photon_direction - p_out * electron_out
    return HardRadiativePhoton(energy, photon_direction, remaining, electron_out, target_momentum)


# Limit simultaneous angle rows, including backends that expand a row into
# multiple (18-node) BremsLib gather temporaries. This is a working-set cap,
# not a device-wide cache or a configurable change to the sampling law.
_COMPLETION_MAX_PHOTONS = 128
_COMPLETION_MAX_ANGLE_ROWS = 32_768


def _complete_hard_radiative_chunk(staged, incident, energy, incoming, uniforms):
    """Complete one element's photons using the scalar DDCS/CDF arithmetic.

    The sole host transfer is the (photon, theta) DDCS. For each photon the
    theta nodes, cell integration, side='right' search and 52 midpoint steps
    are those of :func:`hard_radiative_photon_at_energy`.
    """
    theta = np.asarray(staged.table.theta_rad, dtype=float)
    n = energy.size
    if (
        np.any(~np.isfinite(incident))
        or np.any(~np.isfinite(energy))
        or np.any((energy <= 0.0) | (energy > incident))
    ):
        raise ValueError("photon energy must be positive and at most the incident energy")
    if (
        incoming.shape != (n, 3)
        or np.any(~np.isfinite(incoming))
        or np.any(~np.isclose(np.linalg.norm(incoming, axis=1), 1.0, atol=1e-10))
    ):
        raise ValueError("electron_direction must be a finite unit vector")
    if np.any((uniforms < 0.0) | (uniforms >= 1.0)):
        raise ValueError("angle uniforms must be in [0, 1)")

    # A row of the existing backend evaluator represents one emission angle.
    # Flatten photon-major, preserving each scalar call's theta order. Unlike
    # evaluate_bremslib(..., energy), the rows evaluator does not accidentally
    # form an (Nphotons * Ntheta)**2 broadcast matrix.
    state = bremslib_segment_state(
        staged,
        np.repeat(incident / 1e3, theta.size),
        np.tile(np.cos(theta), n),
    )
    ddcs = np.asarray(
        BACKEND.to_cpu(evaluate_bremslib_rows(staged, state, np.repeat(energy, theta.size)))
    ).reshape(n, theta.size)
    widths = np.diff(theta)
    a, b = theta[:-1], theta[1:]
    cell_mass = (
        2.0
        * np.pi
        * (
            ddcs[:, :-1] * (np.cos(a) - np.cos(b))
            + np.diff(ddcs, axis=1) / widths * (np.sin(b) - np.sin(a) - widths * np.cos(b))
        )
    )
    cumulative = np.cumsum(cell_mass, axis=1)
    if np.any(~np.isfinite(cumulative[:, -1]) | (cumulative[:, -1] <= 0.0)):
        raise ValueError("BremsLib DDCS has no positive angular integral")
    target = uniforms[:, 0] * cumulative[:, -1]
    # searchsorted(..., side='right') on each monotone CDF row, including
    # zero-mass cells and exact boundaries. Do not sort photon rows by element
    # before drawing their two uniforms.
    cell = np.minimum(np.count_nonzero(cumulative <= target[:, None], axis=1), widths.size - 1)
    row = np.arange(n)
    target -= np.where(cell > 0, cumulative[row, np.maximum(cell - 1, 0)], 0.0)
    low, high = theta[cell].copy(), theta[cell + 1].copy()
    left, right = ddcs[row, cell], ddcs[row, cell + 1]
    left_theta, width = a[cell], widths[cell]
    for _ in range(52):
        mid = 0.5 * (low + high)
        partial = (
            2.0
            * np.pi
            * (
                left * (np.cos(left_theta) - np.cos(mid))
                + (right - left)
                / width
                * (np.sin(mid) - np.sin(left_theta) - (mid - left_theta) * np.cos(mid))
            )
        )
        take_low = partial < target
        low = np.where(take_low, mid, low)
        high = np.where(take_low, high, mid)
    polar = 0.5 * (low + high)

    reference = np.where(
        (np.abs(incoming[:, 0]) < 0.9)[:, None],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    transverse = np.cross(incoming, reference)
    # np.linalg.norm(transverse, axis=1) changes the reduction kernel vs the
    # scalar three-vector np.linalg.norm on some NumPy/BLAS builds (1 ULP).
    # Keep only these tiny norms scalar; all cross products, angles and momenta
    # are still vectorized, preserving the pre-batch vector bit pattern.
    transverse /= np.array([np.linalg.norm(vector) for vector in transverse])[:, None]
    other = np.cross(incoming, transverse)
    phi = 2.0 * np.pi * uniforms[:, 1]
    photon_direction = np.cos(polar)[:, None] * incoming + np.sin(polar)[:, None] * (
        np.cos(phi)[:, None] * transverse + np.sin(phi)[:, None] * other
    )
    remaining = incident - energy
    p_in = np.sqrt(incident * (incident + 2.0 * _ELECTRON_REST_EV))
    p_out = np.sqrt(remaining * (remaining + 2.0 * _ELECTRON_REST_EV))
    target_momentum = (
        p_in[:, None] * incoming - energy[:, None] * photon_direction - p_out[:, None] * incoming
    )
    return photon_direction, target_momentum


def _lazy_zeros(rows: int) -> np.ndarray:
    """A writable ``(rows, 3)`` float64 zero array whose untouched pages stay unbacked.

    Photon vectors are nonzero on few rows. ``np.zeros`` is hugepage-advised
    by NumPy, so sparse row writes back nearly every 2 MiB page, and
    ``np.zeros_like`` writes every page. A private anonymous mapping without
    hugepages backs only the 4 KiB pages actually written (#825); reads of the
    rest map the shared zero page. Small arrays, and platforms without
    ``MAP_PRIVATE``, use ``np.zeros``.
    """
    nbytes = int(rows) * 3 * np.dtype(float).itemsize
    private = getattr(mmap, "MAP_PRIVATE", None)
    if nbytes == 0 or nbytes < _LAZY_ZERO_BYTES or private is None:
        return np.zeros((rows, 3), dtype=float)
    buffer = mmap.mmap(-1, nbytes, flags=private)
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        buffer.madvise(mmap.MADV_NOHUGEPAGE)
    return np.frombuffer(buffer, dtype=float).reshape(rows, 3)


def complete_hard_radiative_events(segments, tables, seed: int) -> None:
    """Attach photon directions and recoil momenta without changing RNG order.

    Draw (polar, azimuth) in canonical positive-photon row order, then group
    by element. Each call owns its element stages; no global device cache.
    Non-photon rows retain the two zero-vector result fields.

    Validation: bremslib-radiative-partition
    """
    energies = np.asarray(segments["hard_radiative_k_eV"])
    directions = _lazy_zeros(energies.size)
    momenta = _lazy_zeros(energies.size)
    photon_rows = np.flatnonzero(energies > 0.0)
    rng = np.random.default_rng(int(seed) ^ 0xD6023FEA38B57C29)
    # Shape (N, 2) consumes exactly the same consecutive MT/PCG uniform words
    # as the former two scalar random() calls per photon (even for N=0).
    uniforms = rng.random((photon_rows.size, 2))
    by_z = {table.atomic_number: table for table in tables.values()}
    # Gather the photon rows before any float cast: a cast of a narrower
    # per-segment field would otherwise copy every row.
    element = np.asarray(segments["hard_radiative_Z"])[photon_rows]
    incident = np.asarray(segments["E_end_keV"])[photon_rows].astype(float, copy=False) * 1e3
    photon_energy = energies[photon_rows].astype(float, copy=False)
    incoming = np.asarray(segments["v_hat"])[photon_rows].astype(float, copy=False)
    for z in dict.fromkeys(element.tolist()):
        staged = stage_bremslib_table(by_z[int(z)])
        selected = np.flatnonzero(element == z)
        max_photons = max(
            1,
            min(_COMPLETION_MAX_PHOTONS, _COMPLETION_MAX_ANGLE_ROWS // staged.table.theta_rad.size),
        )
        for start in range(0, selected.size, max_photons):
            offsets = selected[start : start + max_photons]
            completed, target = _complete_hard_radiative_chunk(
                staged,
                incident[offsets],
                photon_energy[offsets],
                incoming[offsets],
                uniforms[offsets],
            )
            rows = photon_rows[offsets]
            directions[rows] = completed
            momenta[rows] = target
    segments["hard_radiative_direction"] = directions
    segments["hard_radiative_target_momentum_eV_c"] = momenta


# PENELOPE-2024 Eq. 3.154 polynomial coefficients of t, t^2, ..., t^7.
_FP_COEFFICIENTS = (
    -1.2359e-1,
    6.1274e-2,
    -3.1516e-2,
    7.7446e-3,
    -1.0595e-3,
    7.0568e-5,
    -1.8080e-6,
)


def positron_brems_factor(atomic_number, energy_eV):
    """Positron/electron radiative cross-section ratio ``F_p(Z, E)``.

    Source: PENELOPE-2024 Eqs. 3.153–3.155 (fit to Kim et al. 1986, ~0.5%),
    ``F_p = 1 - exp(sum_j a_j t^j)``, ``t = ln(1 + 10^6 E/(Z^2 m_e c^2))``;
    SBETHE ``RSTP`` applies the same factor in positron mode. The factor is
    independent of the reduced photon energy, so it scales the integrated
    and differential cross sections alike.
    Limits: ``F_p -> 1`` for ``E -> inf``; ``F_p -> 0`` as ``E -> 0``.

    Units: E in eV. Validation: positron-brems-scaling
    """
    z = np.asarray(atomic_number, dtype=float)
    t = np.log1p(1.0e6 * np.asarray(energy_eV, dtype=float) / (_ELECTRON_REST_EV * z * z))
    exponent = np.zeros_like(t)
    for coefficient in reversed(_FP_COEFFICIENTS):
        exponent = (exponent + coefficient) * t
    return 1.0 - np.exp(exponent)


def positron_bremslib_tables(
    tables: Mapping[str, BremsLibBremsstrahlungTable],
) -> dict[str, BremsLibBremsstrahlungTable]:
    """Electron BremsLib tables rescaled to positrons by ``F_p(Z, T_1)``.

    Multiplies each element's scaled SDCS and DDCS at every incident node
    ``T_1`` by :func:`positron_brems_factor` (PENELOPE-2024 Eq. 3.153): the
    photon-energy and angular shapes are the electron ones. ``key`` gains a
    ``+positron-fp`` suffix so a run identity cannot confuse the two.

    Validation: positron-brems-scaling
    """
    out = {}
    for element, table in tables.items():
        factor = positron_brems_factor(table.atomic_number, table.incident_energy_keV * 1e3)
        out[element] = replace(
            table,
            key=f"{table.key}+positron-fp",
            scaled_sdcs_mb=table.scaled_sdcs_mb * factor[:, None],
            scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * factor[:, None, None],
        )
    return out


def pack_radiative_layer_tables(
    compositions: Sequence[Sequence[tuple[str, float]]],
    tables: Mapping[str, BremsLibBremsstrahlungTable],
    min_energy_eV: float,
    max_energy_eV: float,
    *,
    dielectric: bool = False,
) -> tuple:
    """Pad the BremsLib SDCS and number densities for the exact scalar kernels.

    Every requested element must have a table covering the complete electron
    range. There is no EEDL fallback inside a coupled transport mode.

    Validation: dielectric-suppression
    """
    if not compositions or any(not comp for comp in compositions):
        raise ValueError("radiative transport needs a nonempty composition in every layer")
    n_layers = len(compositions)
    max_elements = max(len(comp) for comp in compositions)
    selected = []
    for comp in compositions:
        layer = []
        for element, density in comp:
            table = tables.get(element)
            if table is None:
                raise ValueError(
                    f"coupled radiative transport needs a BremsLib table for {element}"
                )
            if table.atomic_number != TRANSPORT_ELEMENTS[element]["Z"]:
                raise ValueError(f"BremsLib atomic number does not match {element}")
            if min_energy_eV / 1e3 < table.minimum_incident_energy_keV * (
                1.0 - 1e-12
            ) or max_energy_eV / 1e3 > table.maximum_incident_energy_keV * (1.0 + 1e-12):
                raise ValueError(
                    f"electron energy range is outside the BremsLib table for {element}"
                )
            layer.append((table, float(density) * 1e24))
        selected.append(layer)
    # The medium's plasma frequency is set by ALL constituent electrons,
    # not each element's isolated density. It is therefore shared by every
    # element in a layer, including the absorbing substrate of a stack.
    if dielectric:
        from ...materials import MediumSpec
        from ...materials.attenuation import plasma_energy_eV

        plasma_by_layer = np.array(
            [
                plasma_energy_eV(MediumSpec(f"radiative-layer-{i}", tuple(composition)))
                for i, composition in enumerate(compositions)
            ],
            dtype=float,
        )
    else:
        plasma_by_layer = np.zeros(n_layers, dtype=float)
    max_incident = max(table.incident_energy_keV.size for layer in selected for table, _ in layer)
    n_reduced = selected[0][0][0].nominal_reduced_energy.size
    n_elements = np.array([len(layer) for layer in selected], dtype=np.int32)
    n_incident = np.zeros((n_layers, max_elements), dtype=np.int32)
    atomic_number = np.zeros((n_layers, max_elements), dtype=np.int32)
    density_cm3 = np.zeros((n_layers, max_elements), dtype=float)
    incident_keV = np.zeros((n_layers, max_elements, max_incident), dtype=float)
    nominal = np.zeros((n_layers, max_elements, n_reduced), dtype=float)
    top = np.zeros_like(incident_keV)
    scaled_sdcs = np.zeros((n_layers, max_elements, max_incident, n_reduced), dtype=float)
    for layer_index, layer in enumerate(selected):
        for element_index, (table, density) in enumerate(layer):
            nT = table.incident_energy_keV.size
            if table.nominal_reduced_energy.size != n_reduced:
                raise ValueError("BremsLib tables have inconsistent reduced-energy grids")
            n_incident[layer_index, element_index] = nT
            atomic_number[layer_index, element_index] = table.atomic_number
            density_cm3[layer_index, element_index] = density
            incident_keV[layer_index, element_index, :nT] = table.incident_energy_keV
            nominal[layer_index, element_index] = table.nominal_reduced_energy
            top[layer_index, element_index, :nT] = table.top_reduced_energy
            scaled_sdcs[layer_index, element_index, :nT] = table.scaled_sdcs_mb
    return (
        n_elements,
        n_incident,
        atomic_number,
        density_cm3,
        incident_keV,
        nominal,
        top,
        scaled_sdcs,
        plasma_by_layer,
    )


def validate_radiative_args(
    radiative_model,
    radiative_cutoff_eV,
    bremslib_tables,
    *,
    energy_model,
    groove,
    keep_segments_on_device,
) -> bool:
    """Check ``simulate_trajectories``' radiative arguments; True in coupled mode."""
    if radiative_model not in ("auto", "uncoupled", "bremslib-soft-hard"):
        raise ValueError("radiative_model must be 'auto', 'uncoupled' or 'bremslib-soft-hard'")
    if radiative_model == "uncoupled":
        if radiative_cutoff_eV is not None or bremslib_tables is not None:
            raise ValueError(
                "radiative_cutoff_eV and bremslib_tables require "
                "radiative_model='bremslib-soft-hard'"
            )
        return False
    if radiative_model == "auto" and groove is not None and bremslib_tables is not None:
        warnings.warn(
            "coupled BremsLib radiative transport is unavailable for grooves; "
            "using uncoupled scoring",
            UserWarning,
            stacklevel=2,
        )
        return False
    if radiative_model == "auto" and bremslib_tables is None:
        return False
    if bremslib_tables is None:
        raise ValueError("bremslib-soft-hard requires bremslib_tables")
    if radiative_cutoff_eV is not None and (
        not np.isfinite(radiative_cutoff_eV) or radiative_cutoff_eV <= 0.0
    ):
        raise ValueError("radiative_cutoff_eV must be positive and finite")
    if energy_model != "midpoint" or groove is not None:
        raise ValueError("bremslib-soft-hard requires midpoint, ungrooved transport")
    if keep_segments_on_device:
        raise ValueError("bremslib-soft-hard completes photon rows on the host")
    return True


def radiative_core_args(
    compositions,
    tables,
    cutoff_eV,
    electron_cutoffs_keV,
    energies_keV,
    seed,
    n_electrons,
    start=0,
    *,
    moment_mode="exact",
) -> tuple:
    """Transport ``radiative_args``: stream keys, cutoff, physical and moment tables.

    Validation: dielectric-suppression
    """
    from ._jit_radiative import radiative_stream_keys

    min_cutoff_eV = float(np.min(electron_cutoffs_keV)) * 1e3
    if cutoff_eV > min_cutoff_eV:
        raise ValueError("radiative_cutoff_eV must not exceed the electron cutoff")
    maximum_eV = float(np.max(energies_keV)) * 1e3
    # Ter--Mikaelian suppression is on for every coupled run so the model has
    # no step in beam energy; at low gamma it reaches only k ~ hbar*omega_p.
    packed = pack_radiative_layer_tables(
        compositions, tables, min_cutoff_eV, maximum_eV, dielectric=True
    )
    if moment_mode == "table":
        from .radiative_moment_transport import pack_transport_moments

        plasma = pack_transport_moments(
            compositions, tables, min_cutoff_eV, maximum_eV, cutoff_eV, packed[-1]
        )
        packed = (*packed[:-1], plasma)
    elif moment_mode != "exact":
        raise ValueError("moment_mode must be table or exact")
    keys = radiative_stream_keys(seed, n_electrons, start=start)
    return (keys, float(cutoff_eV)) + packed


def add_radiative_result_fields(
    result, photon_k_eV, photon_Z, tables, cutoff_eV, seed, plasma_energy_eV_by_layer=None
) -> None:
    """Attach photon rows, their completed kinematics, and the mode's identity."""
    result["hard_radiative_k_eV"] = photon_k_eV
    result["hard_radiative_Z"] = photon_Z
    complete_hard_radiative_events(result, tables, seed)
    plasma = np.asarray(
        [] if plasma_energy_eV_by_layer is None else plasma_energy_eV_by_layer, dtype=float
    )
    result["radiative"] = {
        "model": "bremslib-soft-hard",
        **(
            {
                "dielectric_model": "ter-mikaelian-v1",
                "plasma_energy_eV_by_layer": tuple(float(x) for x in plasma),
            }
            if plasma.size and np.any(plasma > 0.0)
            else {}
        ),
        "cutoff_eV": float(cutoff_eV),
        "bremslib_tables": tuple(
            sorted(
                (element, table.atomic_number, table.key, table.digest)
                for element, table in tables.items()
            )
        ),
    }
