"""Temporal bunch-slice channeling and retarded photon-arrival spectra (#628).

Inputs are :class:`~.energy_time.IncidentPhaseSpace` records: independent
classical electrons. A static crystal makes every single-electron quantum
history a function of its entrance energy and transverse state only, so

* **temporal slices** recombine energy-node solutions with the slice's
  conditional node weights, ``O_s = sum_k P(k | s) O(T_k)``; an arrival-time
  spread alone cannot change depth populations;
* **time-resolved spectra** place each electron's depth-resolved emission at
  its retarded far-field arrival time
  ``t_obs = t_i - n_perp . r_i / c + z (1 - beta cos theta) / (beta c)``.

Two time scales are reported separately: the **bunch envelope** (spread of
``t_i``, and of its projection on the detector direction) and the
**single-electron wave train** (the retarded duration of one electron's
emission through the crystal, which bounds each photon's coherence time).
The bunch envelope never broadens the time-integrated spectrum.

The deterministic paths reuse the #626 per-energy solvers (planar
``DensityMatrixSolver``/``channeling_spectrum``, axial ``propagate_axial``)
with the configured nominal transverse entry; :func:`axial_phase_space_mcwf`
runs #627 energy-resolved MCWF trajectories with every record's own energy,
slope and impact. No longitudinal coherence (#629) or energy loss is modeled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import numpy as np

from .bloch import doppler_denominator
from .energy_time import (
    C_ANG_PER_FS,
    EnergyNodes,
    IncidentPhaseSpace,
    TimeSlices,
    quantize_energies,
    time_slices,
)

_SLICE_TAG = 0x6281
_WORK_CHUNK = 4_000_000


def _beta(kinetic_energy_eV: Any) -> np.ndarray:
    from ...materials.crystal import M_E_EV

    inverse_gamma = M_E_EV / (M_E_EV + np.asarray(kinetic_energy_eV, dtype=float))
    return np.sqrt(1.0 - inverse_gamma**2)


def retarded_delay_fs(kinetic_energy_eV: Any, depth_ang: Any, theta_rad: float = 0.0) -> Any:
    """Far-field photon delay of emission at depth ``z``: ``z (1 - beta cos theta) / (beta c)``.

    Relative to a photon emitted at the entrance by the same electron. The
    electron moves at constant speed ``beta c`` along the slab ``z`` axis and
    the detector is at angle ``theta`` in the far field (common ``R/c``
    removed). ``1 - beta cos theta`` uses the cancellation-free Doppler form.
    Limits: zero depth gives zero; forward emission gives
    ``z / (gamma^2 (1 + beta) beta c)``.

    Validation: cr-retarded-photon-arrival
    """
    denominator = doppler_denominator(kinetic_energy_eV, theta_rad)
    return (
        np.asarray(depth_ang, dtype=float) * denominator / (_beta(kinetic_energy_eV) * C_ANG_PER_FS)
    )


def projected_arrival_fs(space: IncidentPhaseSpace, theta_rad: float, phi_rad: float) -> np.ndarray:
    """Entrance arrival seen from direction ``n``: ``t_i - n_perp . r_i / c`` [fs].

    ``n_perp = sin theta (cos phi, sin phi)`` in the slab frame. This is the
    detector-side bunch envelope; for forward observation it equals ``t_i``.

    Validation: cr-retarded-photon-arrival
    """
    nx, ny = np.sin(theta_rad) * np.cos(phi_rad), np.sin(theta_rad) * np.sin(phi_rad)
    path = space.position_ang @ np.array([nx, ny])
    return space.arrival_time_fs - path / C_ANG_PER_FS


def _grid_weights(energies: np.ndarray) -> np.ndarray:
    """Trapezoid quadrature weights of a photon-energy grid [eV]."""
    if energies.size == 1:
        return np.ones(1)
    gaps = np.diff(energies)
    return 0.5 * (np.r_[gaps, 0.0] + np.r_[0.0, gaps])


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(weights @ values)
    return float(np.sqrt(weights @ (values - mean) ** 2))


def _edges(name: str, values: Any, low: int = 2, high: int = 4097) -> np.ndarray:
    edges = np.array(values, dtype=float, copy=True)
    if (
        edges.ndim != 1
        or not low <= edges.size <= high
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"{name} must be {low}..{high} finite increasing values")
    return edges


def _time_weights(
    space: IncidentPhaseSpace,
    nodes: EnergyNodes,
    depth_edges: np.ndarray,
    time_edges: np.ndarray,
    theta: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(T[k, j, m], outside[k, j], delays[k, j+1])``.

    ``T[k, j, m]`` is the incident probability, summed over node ``k``'s
    records, that window ``j``'s emission arrives in time bin ``m``,
    assuming emission uniform in depth (hence in retarded time) inside each
    window.

    Validation: cr-retarded-photon-arrival
    """
    n_nodes, n_windows, n_bins = nodes.size, depth_edges.size - 1, time_edges.size - 1
    weights = np.zeros((n_nodes, n_windows, n_bins))
    outside = np.zeros((n_nodes, n_windows))
    delays = np.zeros((n_nodes, depth_edges.size))
    base = projected_arrival_fs(space, theta, phi)
    chunk = max(1, _WORK_CHUNK // (n_windows * (n_bins + 1)))
    for k in range(n_nodes):
        delays[k] = retarded_delay_fs(nodes.energies_eV[k], depth_edges, theta)
        start, stop = delays[k, :-1], delays[k, 1:]
        records = np.flatnonzero(nodes.node_index == k)
        for first in range(0, records.size, chunk):
            index = records[first : first + chunk]
            lo = base[index, None] + start[None, :]
            span = (stop - start)[None, :, None]
            cdf = np.clip((time_edges[None, None, :] - lo[..., None]) / span, 0.0, 1.0)
            fraction = np.diff(cdf, axis=-1)
            w = space.weights[index]
            weights[k] += np.einsum("n,njm->jm", w, fraction)
            outside[k] += w @ (1.0 - fraction.sum(axis=-1))
    return weights, np.maximum(outside, 0.0), delays


@dataclass(frozen=True)
class TimeResolvedSpectrum:
    """Photon spectra per incident electron resolved by far-field arrival time.

    ``spectra[m, c, e]`` is channel ``c`` integrated over arrival-time bin
    ``m`` (``time_edges_fs``): planar channels are photons/eV/sr
    (``intensity``, ``capture``) and the Stokes numerators ``I Q``, ``I U``;
    axial channels are photons/eV (``photons``, ``helicity_plus``,
    ``helicity_minus``) on ``photon_energy_edges_eV``. ``outside`` holds the
    emission arriving outside the time window, so
    ``time_integrated = spectra.sum(0) + outside`` is the ordinary
    energy-ensemble spectrum.

    Two different durations are reported and must not be conflated:
    ``bunch_envelope_rms_fs`` is the RMS entrance arrival spread and
    ``observed_envelope_rms_fs`` its projection on the detector direction;
    ``wavetrain_duration_fs[k]`` and ``wavetrain_rms_fs[k]`` describe one
    electron's retarded emission at node ``k``, the bound on a photon's
    coherence time. ``mean_wavetrain_rms_fs`` is its incident-weighted mean.
    """

    time_edges_fs: np.ndarray
    photon_energy_eV: np.ndarray
    photon_energy_edges_eV: np.ndarray | None
    channels: tuple[str, ...]
    spectra: np.ndarray
    outside: np.ndarray
    nodes: EnergyNodes
    theta_rad: float
    phi_rad: float
    depth_edges_ang: np.ndarray
    bunch_envelope_rms_fs: float
    observed_envelope_rms_fs: float
    wavetrain_duration_fs: np.ndarray
    wavetrain_rms_fs: np.ndarray
    mean_wavetrain_rms_fs: float
    provenance: dict[str, Any]

    @property
    def time_integrated(self) -> np.ndarray:
        return self.spectra.sum(axis=0) + self.outside

    @property
    def rate_per_fs(self) -> np.ndarray:
        """``spectra`` divided by each time-bin width [per fs]."""
        return self.spectra / np.diff(self.time_edges_fs)[:, None, None]

    def channel(self, name: str) -> np.ndarray:
        return self.spectra[:, self.channels.index(name)]

    def _energy_weights(self) -> np.ndarray:
        if self.photon_energy_edges_eV is not None:
            return np.diff(self.photon_energy_edges_eV)
        return _grid_weights(self.photon_energy_eV)

    @property
    def mean_photon_energy_eV(self) -> np.ndarray:
        """Intensity-weighted mean photon energy of each time bin (``NaN`` if empty)."""
        intensity = self.spectra[:, 0] * self._energy_weights()
        total = intensity.sum(axis=1)
        first = intensity @ self.photon_energy_eV
        return np.divide(first, total, out=np.full(total.shape, np.nan), where=total > 0)

    @property
    def polarization(self) -> np.ndarray:
        """Planar normalized Stokes ``(Q, U)`` ``(M, nE, 2)``, or axial helicity ``(M, nE)``."""
        total = self.spectra[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            if "stokes_q" in self.channels:
                parts = [self.channel("stokes_q"), self.channel("stokes_u")]
                return np.stack([np.where(total > 0, p / total, np.nan) for p in parts], axis=-1)
            difference = self.channel("helicity_plus") - self.channel("helicity_minus")
            return np.where(total > 0, difference / total, np.nan)


def _assemble(
    space: IncidentPhaseSpace,
    nodes: EnergyNodes,
    window: np.ndarray,
    *,
    depth_edges: np.ndarray,
    time_edges: np.ndarray,
    theta: float,
    phi: float,
    photon_energy: np.ndarray,
    photon_edges: np.ndarray | None,
    channels: tuple[str, ...],
    energy_weights: np.ndarray,
    provenance: dict[str, Any],
) -> TimeResolvedSpectrum:
    """Fold window spectra ``window[k, j, c, e]`` through the retarded time weights."""
    weights, outside_weight, delays = _time_weights(
        space, nodes, depth_edges, time_edges, theta, phi
    )
    spectra = np.einsum("kjm,kjce->mce", weights, window)
    outside = np.einsum("kj,kjce->ce", outside_weight, window)
    # Single-electron wave train: piecewise-uniform retarded emission profile.
    emitted = np.maximum((window[:, :, 0] * energy_weights).sum(axis=-1), 0.0)
    a, b = delays[:, :-1], delays[:, 1:]
    total = emitted.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(total > 0, (emitted * 0.5 * (a + b)).sum(axis=1) / total, np.nan)
        second = np.where(
            total > 0, (emitted * (a * a + a * b + b * b) / 3.0).sum(axis=1) / total, np.nan
        )
    rms = np.sqrt(np.maximum(second - mean**2, 0.0))
    good = np.isfinite(rms)
    mean_rms = (
        float(nodes.weights[good] @ rms[good] / nodes.weights[good].sum())
        if good.any()
        else float("nan")
    )
    for array in (spectra, outside, delays, rms):
        array.setflags(write=False)
    return TimeResolvedSpectrum(
        time_edges,
        photon_energy,
        photon_edges,
        channels,
        spectra,
        outside,
        nodes,
        float(theta),
        float(phi),
        depth_edges,
        space.arrival_rms_fs,
        _weighted_rms(projected_arrival_fs(space, theta, phi), space.weights),
        delays[:, -1] - delays[:, 0],
        rms,
        mean_rms,
        provenance,
    )


def _check_prefix_depths(edges: np.ndarray, dz_ang: float, sample_ang: float) -> None:
    """Window spectra are differences of nested histories: edges must share the sample grid."""
    inner = max(1, int(round(sample_ang / dz_ang)))
    ratio = edges[1:] / (inner * dz_ang)
    # Mirror ``emission._depth_history``: it uses ceil(L / (inner dz)) segments.
    if not np.allclose(ratio, np.rint(ratio), rtol=0.0, atol=1e-9) or np.any(
        np.ceil(ratio) != np.rint(ratio)
    ):
        raise ValueError(
            "depth_edges_ang must be multiples of the estimator segment "
            f"{inner * dz_ang} angstrom (sample_ang rounded to dz_ang multiples)"
        )


def _planar_window_depths(
    thickness_ang: float,
    depth_edges_ang: Sequence[float] | None,
    dz_ang: float,
    sample_ang: float,
    theta_rad: float,
    phi_rad: float,
    surface_normal: Any,
    spectrum_kwargs: dict[str, Any],
) -> np.ndarray:
    """Validate the planar timing geometry and return the cumulative window depths.

    Pops (and refuses nonzero) analytic energy spread and path censoring from
    ``spectrum_kwargs``: both would conflict with the explicit records.
    """
    if surface_normal is not None and not np.allclose(
        np.asarray(surface_normal, float) / np.linalg.norm(surface_normal), (0.0, 0.0, 1.0)
    ):
        raise ValueError("retarded timing assumes the entrance face z = 0 normal to the beam")
    for name in ("energy_spread_eV", "path_limits_ang"):
        value = spectrum_kwargs.pop(name, None)
        if value is not None and np.any(np.asarray(value) != 0):
            raise ValueError(f"{name} is incompatible with phase-space time resolution")
    if np.ndim(theta_rad) or np.ndim(phi_rad):
        raise ValueError("time-resolved spectra take one scalar detector direction")
    if not np.isfinite(thickness_ang) or thickness_ang <= 0:
        raise ValueError("thickness_ang must be finite and positive")
    depth = _edges(
        "depth_edges_ang",
        (0.0, thickness_ang) if depth_edges_ang is None else depth_edges_ang,
        2,
        65,
    )
    if depth[0] != 0.0 or not np.isclose(depth[-1], thickness_ang, rtol=1e-12, atol=0.0):
        raise ValueError("depth_edges_ang must start at 0 and end at thickness_ang")
    if depth.size > 2:
        _check_prefix_depths(depth, dz_ang, sample_ang)
    return depth


def planar_time_resolved_spectrum(
    space: IncidentPhaseSpace,
    solver_factory: Callable[[float], Any],
    thickness_ang: float,
    photon_energy_eV: Any,
    time_edges_fs: Sequence[float],
    *,
    theta_rad: float = 0.0,
    phi_rad: float = 0.0,
    depth_edges_ang: Sequence[float] | None = None,
    max_energy_nodes: int = 17,
    dz_ang: float = 50.0,
    sample_ang: float = 500.0,
    absorption: bool = True,
    layers: Sequence | None = None,
    surface_normal: Any = None,
    **spectrum_kwargs: Any,
) -> TimeResolvedSpectrum:
    """Planar channeling radiation of a correlated bunch versus arrival time.

    Each energy node solves ``channeling_spectrum`` at the cumulative depths
    ``depth_edges_ang`` (default ``(0, thickness)``) with exit absorption
    always through the *full* slab, so successive differences are the
    window emissions exactly (prefix depth grids, linear yields). Records
    then place each window's emission at their retarded arrival. The
    detector direction is one ``(theta, phi)``; the transverse entry is the
    nominal one passed in ``spectrum_kwargs`` for every record.

    Validation: cr-retarded-photon-arrival, cr-temporal-slice-ensemble
    """
    from .emission import _resolve_composition, channeling_spectrum
    from .energy_ensemble import _planar_solver

    depth = _planar_window_depths(
        thickness_ang,
        depth_edges_ang,
        dz_ang,
        sample_ang,
        theta_rad,
        phi_rad,
        surface_normal,
        spectrum_kwargs,
    )
    time_edges = _edges("time_edges_fs", time_edges_fs)
    energies = np.asarray(photon_energy_eV, dtype=float)
    if energies.ndim != 1 or not energies.size or not np.all(np.isfinite(energies)):
        raise ValueError("photon_energy_eV must be a finite one-dimensional grid")
    nodes = quantize_energies(space, max_nodes=max_energy_nodes)
    window = np.zeros((nodes.size, depth.size - 1, 4, energies.size))
    for k, energy in enumerate(nodes.energies_eV):
        solver = _planar_solver(solver_factory, float(energy))
        slab = layers
        if absorption and slab is None:
            slab = [
                (
                    0.0,
                    float(thickness_ang),
                    _resolve_composition(solver.profile.crystal),
                )
            ]
        previous = np.zeros((4, energies.size))
        for j, length in enumerate(depth[1:]):
            result = channeling_spectrum(
                solver,
                float(length),
                energies,
                float(theta_rad),
                float(phi_rad),
                energy_spread_eV=0.0,
                dz_ang=dz_ang,
                sample_ang=sample_ang,
                absorption=absorption,
                layers=slab,
                surface_normal=surface_normal,
                **spectrum_kwargs,
            )
            intensity = np.asarray(result.spectrum, dtype=float)
            stokes = np.nan_to_num(np.asarray(result.stokes, dtype=float))
            cumulative = np.stack(
                [
                    intensity,
                    np.asarray(result.capture_spectrum),
                    intensity * stokes[..., 0],
                    intensity * stokes[..., 1],
                ]
            )
            window[k, j] = cumulative - previous
            previous = cumulative
    return _assemble(
        space,
        nodes,
        window,
        depth_edges=depth,
        time_edges=time_edges,
        theta=float(theta_rad),
        phi=float(phi_rad),
        photon_energy=energies,
        photon_edges=None,
        channels=("intensity", "capture", "stokes_q", "stokes_u"),
        energy_weights=_grid_weights(energies),
        provenance=dict(
            path="planar",
            transverse="nominal entry for every record",
            phase_space=space.digest(),
            quantization_rms_eV=nodes.quantization_rms_eV,
        ),
    )


def _deposit(
    line_energies: np.ndarray, yields: np.ndarray, edges: np.ndarray, sigma_eV: float
) -> np.ndarray:
    """Photons per eV in each bin for line yields ``(C, L)`` (delta or Gaussian response)."""
    from scipy.special import ndtr

    out = np.zeros((yields.shape[0], edges.size - 1))
    if sigma_eV == 0:
        index = np.searchsorted(edges, line_energies, side="right") - 1
        good = (index >= 0) & (index < edges.size - 1)
        for c in range(yields.shape[0]):
            np.add.at(out[c], index[good], yields[c, good])
    else:
        fractions = ndtr((edges[1:, None] - line_energies) / sigma_eV) - ndtr(
            (edges[:-1, None] - line_energies) / sigma_eV
        )
        out = yields @ fractions.T
    return out / np.diff(edges)[None, :]


def axial_time_resolved_spectrum(
    space: IncidentPhaseSpace,
    config: Any,
    photon_energy_edges_eV: Sequence[float],
    time_edges_fs: Sequence[float],
    *,
    entry: Any = None,
    max_energy_nodes: int = 9,
    detector_sigma_eV: float = 0.0,
) -> TimeResolvedSpectrum:
    """Forward axial line emission of a correlated bunch versus arrival time.

    Each node runs ``propagate_axial`` with its energy-specific line
    catalog; the stored depths of ``config`` are the emission windows (the
    accumulated per-line yields are differenced between stored depths).
    Helicity channels use each window's accumulated ``T_++`` and ``T_--``.
    Photon energies *and* retardation are forward (``theta = 0``), so the
    transverse projection term vanishes; the 4 pi line yields are timed with
    the forward delay, which is exact only for a forward detector (photons at
    ``theta ~ 1/gamma`` arrive up to about twice as late within the crystal).

    Validation: cr-retarded-photon-arrival, cr-temporal-slice-ensemble
    """
    from .axial_propagation import propagate_axial

    app = import_module("pyrite.apps.axial_channeling")
    edges = _edges("photon_energy_edges_eV", photon_energy_edges_eV)
    time_edges = _edges("time_edges_fs", time_edges_fs)
    if not np.isfinite(detector_sigma_eV) or detector_sigma_eV < 0:
        raise ValueError("detector_sigma_eV must be finite and nonnegative")
    depth = _edges("config.depths_ang", config.depths_ang, 2, 4097)
    nodes = quantize_energies(space, max_nodes=max_energy_nodes)
    window = np.zeros((nodes.size, depth.size - 1, 3, edges.size - 1))
    for k, energy in enumerate(nodes.energies_eV):
        specific = replace(config, kinetic_energy_eV=float(energy))
        diagnostic = propagate_axial(
            specific, entry=entry, observable=app._level_observable(specific)
        )
        radiation = app.axial_radiation(diagnostic)
        if radiation is None:
            continue
        lines = np.asarray(
            [radiation.catalog.photon_energy_eV(line) for line in radiation.catalog.lines]
        )
        photons = np.stack([radiation.photons(d) for d in range(depth.size)])
        plus = radiation.cumulative[:, :, 0, 0].real
        minus = radiation.cumulative[:, :, 1, 1].real
        for j in range(depth.size - 1):
            d_photons = photons[j + 1] - photons[j]
            d_plus, d_minus = plus[j + 1] - plus[j], minus[j + 1] - minus[j]
            trace = d_plus + d_minus
            f_plus = np.divide(d_plus, trace, out=np.zeros_like(trace), where=trace > 0)
            f_minus = np.divide(d_minus, trace, out=np.zeros_like(trace), where=trace > 0)
            yields = np.stack([d_photons, d_photons * f_plus, d_photons * f_minus])
            window[k, j] = _deposit(lines, yields, edges, float(detector_sigma_eV))
    return _assemble(
        space,
        nodes,
        window,
        depth_edges=depth,
        time_edges=time_edges,
        theta=0.0,
        phi=0.0,
        photon_energy=0.5 * (edges[1:] + edges[:-1]),
        photon_edges=edges,
        channels=("photons", "helicity_plus", "helicity_minus"),
        energy_weights=np.diff(edges),
        provenance=dict(
            path="axial",
            transverse="nominal entry for every record",
            phase_space=space.digest(),
            quantization_rms_eV=nodes.quantization_rms_eV,
        ),
    )


@dataclass(frozen=True)
class TemporalSliceDiagnostic:
    """Depth observables per incident arrival-time slice.

    ``observables[name][s]`` is slice ``s``'s incident-normalized mean of a
    physical, density-linear observable; ``integrated`` is the #626 result
    over the node marginal (every record, including any outside the slices).
    Rows of empty slices are zero; see ``slices.probability``.
    """

    slices: TimeSlices
    nodes: EnergyNodes
    observables: dict[str, np.ndarray]
    integrated: Any
    provenance: dict[str, Any]


def _slice_mix(slices: TimeSlices, per_node: Sequence[np.ndarray]) -> np.ndarray:
    """``O_s = sum_k P(k | s) O_k`` over a common shape.

    Validation: cr-temporal-slice-ensemble
    """
    stack = np.stack([np.asarray(item, dtype=np.result_type(item, float)) for item in per_node])
    out = np.tensordot(slices.node_weights, stack, axes=(1, 0))
    out.setflags(write=False)
    return out


def planar_temporal_slices(
    space: IncidentPhaseSpace,
    solver_factory: Callable[[float], Any],
    thickness_ang: float,
    dz_ang: float,
    slice_edges_fs: Sequence[float],
    *,
    max_energy_nodes: int = 17,
    **propagate_kwargs: Any,
) -> TemporalSliceDiagnostic:
    """Planar density-matrix depth populations per incident arrival-time slice.

    Validation: cr-temporal-slice-ensemble
    """
    from .energy_ensemble import planar_energy_diagnostic

    nodes = quantize_energies(space, max_nodes=max_energy_nodes)
    slices = time_slices(space, nodes, slice_edges_fs)
    integrated = planar_energy_diagnostic(
        nodes.ensemble(),
        solver_factory,
        thickness_ang,
        dz_ang,
        retain_node_results=True,
        **propagate_kwargs,
    )
    runs = tuple(integrated.per_node or ())
    n_bands = min(np.asarray(r.band_populations).shape[-1] for r in runs)
    observables = {
        name: _slice_mix(slices, [getattr(r, name) for r in runs])
        for name in (
            "trace",
            "bound_population",
            "free_flux",
            "overflow_flux",
            "reprojection_flux",
            "entry_flux",
        )
    }
    observables["band_populations"] = _slice_mix(
        slices, [np.asarray(r.band_populations)[..., :n_bands] for r in runs]
    )
    if runs[0].spatial_density_per_ang is not None:
        observables["spatial_density_per_ang"] = _slice_mix(
            slices, [r.spatial_density_per_ang for r in runs]
        )
    return TemporalSliceDiagnostic(
        slices, nodes, observables, integrated, dict(path="planar", phase_space=space.digest())
    )


def axial_temporal_slices(
    space: IncidentPhaseSpace,
    config: Any,
    slice_edges_fs: Sequence[float],
    *,
    entry: Any = None,
    max_energy_nodes: int = 9,
) -> TemporalSliceDiagnostic:
    """Axial physical-space density and survival per incident arrival-time slice.

    Validation: cr-temporal-slice-ensemble
    """
    from .energy_ensemble import axial_energy_diagnostic

    nodes = quantize_energies(space, max_nodes=max_energy_nodes)
    slices = time_slices(space, nodes, slice_edges_fs)
    integrated = axial_energy_diagnostic(
        nodes.ensemble(), config, entry=entry, retain_node_results=True
    )
    runs = tuple(integrated.per_node or ())
    observables = {
        "density_per_ang2": _slice_mix(slices, [r.density_per_ang2 for r in runs]),
        "surviving": _slice_mix(slices, [r.surviving for r in runs]),
    }
    if runs[0].oam_populations is not None:
        observables["oam_populations"] = _slice_mix(slices, [r.oam_populations for r in runs])
    return TemporalSliceDiagnostic(
        slices, nodes, observables, integrated, dict(path="axial", phase_space=space.digest())
    )


@dataclass(frozen=True)
class AxialPhaseSpaceMCWFResult:
    """Energy-resolved axial MCWF per arrival-time slice with correlated entries.

    ``runs[s]`` is the #627 result of slice ``s`` (``None`` if empty); its
    electrons are the slice's records in increasing record order. Slice
    means are per assigned incident electron; ``integrated_*`` are the
    count-weighted sums over slices (the slices cover every record).
    """

    slices: TimeSlices
    nodes: EnergyNodes
    runs: tuple[Any, ...]
    slice_counts: np.ndarray
    depths_ang: np.ndarray
    surviving: np.ndarray
    density_per_ang2: np.ndarray
    integrated_surviving: np.ndarray
    integrated_density_per_ang2: np.ndarray
    photon_energy_edges_eV: np.ndarray | None
    photon_density_per_eV: np.ndarray | None
    integrated_photon_density_per_eV: np.ndarray | None
    provenance: dict[str, Any]


def _correlated_entry(space: IncidentPhaseSpace, config: Any, entry: Any, impact: str):
    """Per-record ``(entry, tilt)``: centroid-relative slope and periodic impact."""
    from ...materials.channeling.axial import axial_profile
    from .states import gaussian, translate

    base = gaussian(waist_ang=0.5) if entry is None else entry
    slope = space.slope_rad - space.weights @ space.slope_rad
    tilts = np.asarray(config.tilt_rad, dtype=float)[None, :] + slope
    if impact == "entry":
        return lambda record: (base, tuple(tilts[record]))
    profile = axial_profile(config.crystal, config.axis, config.temperature_K)
    extent = np.array(
        (profile.period_ang[0] * config.cells[0], profile.period_ang[1] * config.cells[1])
    )
    offsets = np.mod(space.position_ang - space.weights @ space.position_ang, extent)
    return lambda record: (translate(base, offsets[record]), tuple(tilts[record]))


def axial_phase_space_mcwf(
    space: IncidentPhaseSpace,
    config: Any,
    *,
    slice_edges_fs: Sequence[float] | None = None,
    entry: Any = None,
    impact: str = "phase_space",
    max_energy_nodes: int = 32,
    options: Any = None,
    **mcwf_kwargs: Any,
) -> AxialPhaseSpaceMCWFResult:
    """Run #627 MCWF trajectories with each record's energy, slope and impact.

    Every equally weighted record is one electron. Its kinetic energy is its
    energy node (``max_energy_nodes`` bounds solver rebuilds; exact when the
    records are discrete), its tilt is ``config.tilt_rad`` plus its slope
    deviation from the centroid, and with ``impact="phase_space"`` the entry
    state is translated by its centroid-relative position modulo the
    supercell (``"entry"`` keeps the given entry). Slices run separately with
    slice-keyed seeds, so a slice result is statistically, not bitwise,
    comparable with an unsliced run. ``mcwf_kwargs`` go to
    :func:`~.axial_energy_mcwf.axial_energy_mcwf` (radiation edges,
    coherence pairs, observables...).

    Validation: cr-temporal-slice-ensemble
    """
    from .axial_energy_mcwf import AxialEnergyMCWFOptions, axial_energy_mcwf
    from .mcwf import mcwf_stream_key

    for name in ("entry_factory", "electron_energies_eV", "energy_bin_edges_eV"):
        if name in mcwf_kwargs:
            raise ValueError(f"{name} is derived from the phase space")
    if impact not in ("phase_space", "entry"):
        raise ValueError("impact must be 'phase_space' or 'entry'")
    if not space.uniform_weights:
        raise ValueError("MCWF needs equally weighted electrons; use IncidentPhaseSpace.resample")
    options = AxialEnergyMCWFOptions() if options is None else options
    nodes = quantize_energies(space, max_nodes=min(max_energy_nodes, options.max_bins))
    if slice_edges_fs is None:
        t = space.arrival_time_fs
        lo, hi = float(t.min()), float(t.max())
        slice_edges_fs = (lo, hi) if hi > lo else (lo - 1.0, lo + 1.0)
    slices = time_slices(space, nodes, slice_edges_fs)
    if np.any(slices.slice_index < 0):
        raise ValueError("MCWF slice edges must cover every record's arrival time")
    energies = nodes.energies_eV[nodes.node_index]
    bin_edges = nodes.bin_edges_eV()
    entry_of = _correlated_entry(space, config, entry, impact)
    runs, counts = [], []
    for s in range(slices.probability.size):
        records = np.flatnonzero(slices.slice_index == s)
        counts.append(records.size)
        if not records.size:
            runs.append(None)
            continue
        seed = options.seed
        if slices.probability.size > 1:
            seed = int(mcwf_stream_key(options.seed ^ _SLICE_TAG, s)) % 2**63

        def factory(energy, eid, _rng, records=records):
            record = int(records[eid])
            if energy != energies[record]:
                raise RuntimeError("MCWF electron order does not match phase-space records")
            return entry_of(record)

        runs.append(
            axial_energy_mcwf(
                nodes.ensemble(),
                config,
                options=replace(options, histories=int(records.size), seed=seed),
                entry_factory=factory,
                electron_energies_eV=energies[records],
                energy_bin_edges_eV=tuple(bin_edges.tolist()),
                **mcwf_kwargs,
            )
        )
    probability = np.asarray(counts, dtype=float) / space.size
    present = [r for r in runs if r is not None]
    depths = np.asarray(present[0].depths_ang)

    def stacked(attr):
        template = getattr(present[0], attr)
        if template is None:
            return None
        rows = [np.zeros_like(template) if r is None else getattr(r, attr) for r in runs]
        return np.stack(rows)

    surviving, density = stacked("surviving"), stacked("density_per_ang2")
    photons = stacked("photon_density_per_eV")
    assert surviving is not None and density is not None

    def integrate(values: np.ndarray) -> np.ndarray:
        return np.tensordot(probability, values, axes=(0, 0))

    return AxialPhaseSpaceMCWFResult(
        slices,
        nodes,
        tuple(runs),
        np.asarray(counts),
        depths,
        surviving,
        density,
        integrate(surviving),
        integrate(density),
        present[0].photon_energy_edges_eV,
        photons,
        None if photons is None else integrate(photons),
        dict(
            path="axial-mcwf",
            impact=impact,
            phase_space=space.digest(),
            quantization_rms_eV=nodes.quantization_rms_eV,
            slice_seeds="options.seed" if len(runs) == 1 else "mcwf_stream_key(seed^0x6281, s)",
        ),
    )


__all__ = [
    "AxialPhaseSpaceMCWFResult",
    "TemporalSliceDiagnostic",
    "TimeResolvedSpectrum",
    "axial_phase_space_mcwf",
    "axial_temporal_slices",
    "axial_time_resolved_spectrum",
    "planar_temporal_slices",
    "planar_time_resolved_spectrum",
    "projected_arrival_fs",
    "retarded_delay_fs",
]
