"""Finite PEC-screen optical TR/DR and CTR/CDR from physical bunch histories (#616).

CPU reference path connecting the finite, phase-bearing spectral surface fields
of :mod:`optical_finite_screen` with the physical-population/importance-sampling
contract of :class:`optical_transition.OpticalBunchSource` (#615).

No ``N**2`` macro-particle weighting: the self moment uses w, while the
independent-electron *distinct pair* moment uses w_i w_j. The result is an
expected intensity (which may have a noisy signed pair estimator), **not** a
single coherent bunch wavefront. Only ``law='realization'`` represents an
actual N-electron field that can propagate coherently to a near-field sensor.

For a finite screen the electron's transverse displacement changes the driven
metallic surface, so an analytic transverse form factor cannot in general be
multiplied onto a fixed single-electron field. Analytic/empirical offset laws
are therefore admitted here only for independent *temporal* offsets with all
sampled sources at one common intercept and arrival time. For arbitrary
transverse phase-space, supply sampled histories with their actual offsets.

All fields use the two-sided positive-frequency Fourier transform [V*s/m]
of #616, NOT the monochromatic [V/m] phasor of optical_wavefront. Angular
spectra have photons/(eV sr); cone integrals are photons/eV per bunch.
Neither this reference path nor its synthetic tests validate measured data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .optical_finite_screen import (
    _E_CHARGE_C,
    _HBAR_C_EV_M,
    _HBAR_EV_S,
    _Z0_OHM,
    PECCircularScreen,
    ScreenSpectralField,
    _wavevectors,
    circular_pec_transition_field,
)

if TYPE_CHECKING:
    from .optical_transition import OpticalBunchSource

__all__ = [
    "FiniteScreenBunchAngular",
    "FiniteScreenBunchBand",
    "finite_screen_bunch_angular",
    "finite_screen_bunch_band",
    "finite_screen_realization_field",
]


@dataclass(frozen=True, eq=False)
class FiniteScreenBunchAngular:
    """Angular self and *unclipped* pair estimates for one photon energy.

    ``yield_per_electron`` = self + (N-1) F pair (except realizations, where
    it is the exact coherent |sum E|²/N); ``bunch_yield`` = N times this.
    All maps have the same outgoing kx/ky FFT mesh; non-propagating bins are
    zero and their ``theta_rad`` entries are NaN. ``pair_yield`` can be signed
    because a distinct-pair estimator has sampling noise.
    """

    energy_eV: float
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    self_yield: np.ndarray
    pair_yield: np.ndarray
    form_factor_squared: float
    yield_per_electron: np.ndarray
    bunch_yield: np.ndarray
    physical_electrons: float
    n_histories: int
    effective_histories: float
    dx_m: float
    dy_m: float
    source_record: dict

    def collect_cone(self, *, theta_max_rad: float, theta_min_rad: float = 0.0) -> float:
        """Integrate bunch photons/eV in a forward circular angular annulus.

        The integral sums FFT angular cells with physical ``dOmega``. The
        all-forward-hemisphere limit (theta_max = pi/2) reproduces positive-z
        spectral radiation energy divided by E*e. Grid/acceptance convergence
        must be demonstrated independently; near theta = pi/2 the FFT k-grid
        under-resolves the light-circle 1/kz density and the sum varies by
        several percent with the window (compare ``hemisphere_energy_per_eV``
        for one compact source). No optics, detector efficiency or
        polarization acceptance is silently applied.
        """
        if (
            not np.isfinite(theta_min_rad)
            or not np.isfinite(theta_max_rad)
            or not 0 <= theta_min_rad < theta_max_rad <= np.pi / 2
        ):
            raise ValueError("cone bounds require 0 <= theta_min < theta_max <= pi/2")
        keep = np.isfinite(self.theta_rad) & (self.theta_rad >= theta_min_rad)
        keep &= self.theta_rad <= theta_max_rad
        if not np.any(keep):
            return 0.0
        ny, nx = self.theta_rad.shape
        k0 = self.energy_eV / _HBAR_C_EV_M
        dkx = 2 * np.pi / (nx * self.dx_m)
        dky = 2 * np.pi / (ny * self.dy_m)
        d_omega = dkx * dky / (k0**2 * np.cos(self.theta_rad[keep]))
        return float(np.sum(self.bunch_yield[keep] * d_omega))


@dataclass(frozen=True, eq=False)
class FiniteScreenBunchBand:
    """Collected spectral photon density in a declared angular cone, per bunch.

    Compatible with ``CalibratedPulseEnergyDetector.predict`` input **only
    after** the cone and any additional instrument acceptance have been
    explicitly accounted for; ``collected`` does not imply measured response.
    """

    energy_eV: np.ndarray
    collected_photons_per_eV_per_pulse: np.ndarray
    theta_min_rad: float
    theta_max_rad: float
    source_record: dict


def _shared_screen_histories(screens) -> tuple[PECCircularScreen, ...]:
    histories = tuple(screens)
    if not histories or not all(isinstance(s, PECCircularScreen) for s in histories):
        raise ValueError("at least one PECCircularScreen history is required")
    if not all(
        s.outer_radius_m == histories[0].outer_radius_m
        and s.inner_radius_m == histories[0].inner_radius_m
        for s in histories
    ):
        raise ValueError("each history must illuminate the same physical screen")
    if any(s.charge_c != histories[0].charge_c for s in histories):
        raise ValueError("source histories must share the same elementary-charge sign")
    return histories


def _validate_histories(
    screens: Sequence[PECCircularScreen], source: OpticalBunchSource, energy_eV: float
):
    histories = _shared_screen_histories(screens)
    if not np.isfinite(energy_eV) or energy_eV <= 0:
        raise ValueError("energy_eV must be positive and finite")
    law = getattr(source, "law", None)
    if law not in ("sampled", "realization", "gaussian", "empirical"):
        raise ValueError("source must use the optical bunch law of #615")
    if getattr(source, "inter_electron", "independent") == "correlated" and law != "realization":
        raise ValueError("correlated physical bunches require law='realization'")
    n = float(source.physical_electrons)
    if not np.isfinite(n) or n < 0 or (0 < n < 1):
        raise ValueError("physical electron population must be zero or >= 1")
    weights = _history_weights(source, len(histories))
    if law == "realization":
        if weights is not None:
            raise ValueError("a physical realization refuses macro-particle weights")
        if abs(n - len(histories)) > 0.5:
            raise ValueError("a physical realization requires N = number of histories")
    if law in ("gaussian", "empirical"):
        _check_temporal_only_law(source, law, histories)
    # Require distinct histories even with disparate importance weights when
    # the pair estimate will be scaled by N-1 physical electrons.
    if n > 1 and law != "realization" and len(histories) < 2:
        raise ValueError("a physical bunch with pairs needs >=2 sample histories")
    return histories, n, (np.ones(len(histories)) if weights is None else weights)


def _history_weights(source: OpticalBunchSource, n_histories: int) -> np.ndarray | None:
    weights = getattr(source, "history_weights", None)
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_histories,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("one finite positive importance weight is required per history")
    return weights


def _check_temporal_only_law(source: OpticalBunchSource, law: str, histories) -> None:
    # Finite screen breaks transverse translation invariance: even a small
    # position shift alters the induced currents, not just the far-zone phase.
    positions = np.array([[s.electron_x_m, s.electron_y_m, s.arrival_time_s] for s in histories])
    if np.any(np.ptp(positions, axis=0) > 0):
        raise ValueError("analytic/empirical offsets require offset-free histories")
    if law == "gaussian":
        cov = np.asarray(source.covariance_ang2, dtype=float)
        if cov.shape != (3, 3) or np.any(cov[1:, :] != 0) or np.any(cov[:, 1:] != 0):
            raise ValueError(
                "finite-screen gaussian law supports temporal RMS only; sample transverse offsets"
            )
    else:
        profile = np.asarray(source.profile_ang)
        if profile.ndim != 2 or profile.shape[1] != 3 or np.any(profile[:, 1:] != 0):
            raise ValueError("finite-screen empirical law supports temporal offsets only")


def _screen_fields(energy_eV, histories, *, nx, ny, dx_m, dy_m):
    for screen in histories:
        yield circular_pec_transition_field(energy_eV, screen, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m)


def _form_factor(source, energy_eV):
    if source.law in ("sampled", "realization"):
        return 1.0
    # Only temporal laws are allowed, so F² is independent of far-field angle.
    factor = float(
        np.asarray(source.form_factor_squared(np.array(energy_eV), np.array([0.0, 0.0, 1.0])))
    )
    if not np.isfinite(factor):
        raise ValueError("source temporal form factor is not finite")
    return factor


def _moments(energy_eV, histories, weights, kx, ky, kz_safe, *, nx, ny, dx_m, dy_m):
    """One pass through sampled histories; no N-history field tensor."""
    sum_fft = np.zeros((2, ny, nx), complex)
    sum_one = np.zeros((ny, nx), float)
    sum_two = np.zeros((ny, nx), float)
    for weight, field in zip(
        weights,
        _screen_fields(energy_eV, histories, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m),
        strict=True,
    ):
        ft = np.fft.fft2(field.electric_xy_v_s_m, axes=(-2, -1))
        norm = (
            np.abs(ft[0]) ** 2
            + np.abs(ft[1]) ** 2
            + np.abs((kx * ft[0] + ky * ft[1]) / kz_safe) ** 2
        )
        sum_fft += weight * ft
        sum_one += weight * norm
        sum_two += weight**2 * norm
    return sum_fft, sum_one, sum_two


def finite_screen_bunch_angular(
    energy_eV: float,
    screens: Sequence[PECCircularScreen],
    *,
    source: OpticalBunchSource,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
) -> FiniteScreenBunchAngular:
    """Estimate finite PEC-screen TR/DR self/pair far-zone angular photons.

    Each sampled electron drives the **same** PEC disc/annulus with its own
    impact point, beta and arrival-time phase. ``OpticalBunchSource`` provides
    physical charge, independent-electron offset law and positive importance
    weights. A ``realization`` is instead an actual field-level bunch sum.

    This CPU-only reference computes FFTs one history at a time (O(nx*ny)
    moment workspace). No transport-to-screen handoff, emission beyond the
    bounded ideal PEC geometry, GPU parity or experimental anchor is implied.
    """
    histories, n_physical, weights = _validate_histories(screens, source, energy_eV)
    # Generate one source first to validate the grid before allocating moments.
    reference = circular_pec_transition_field(
        energy_eV, histories[0], nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m
    )
    k0, kx, ky, kz = _wavevectors(reference)
    good = (kx * kx + ky * ky) < k0 * k0 * (1 - 1e-12)
    kz_safe = np.where(good, kz, 1.0)
    sum_fft, one, two = _moments(
        energy_eV, histories, weights, kx, ky, kz_safe, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m
    )
    # Proper vector metric, including Ez for each transverse Fourier mode.
    full = (
        np.abs(sum_fft[0]) ** 2
        + np.abs(sum_fft[1]) ** 2
        + np.abs((kx * sum_fft[0] + ky * sum_fft[1]) / kz_safe) ** 2
    )
    w_sum = float(np.sum(weights))
    w2_sum = float(np.sum(weights**2))
    g = one / w_sum
    h = two
    denominator = w_sum * w_sum - w2_sum
    if denominator > 0:
        pair = (full - h) / denominator
    else:
        pair = np.zeros_like(g)
    factor = _form_factor(source, energy_eV)
    total = (
        full / w_sum
        if source.law == "realization"
        else g + max(0.0, n_physical - 1) * factor * pair
    )
    # Angle -> spectral photons uses the exact two-sided Fourier norm from
    # optical_finite_screen.far_field_photons_per_eV_sr.
    cosine = np.where(good, kz.real / k0, 0.0)
    prefactor = (
        k0**2
        * cosine**2
        * (dx_m * dy_m) ** 2
        / (4 * np.pi**3 * _Z0_OHM * _HBAR_EV_S * energy_eV * _E_CHARGE_C)
    )
    yself = np.where(good, prefactor * g, 0.0)
    ypair = np.where(good, prefactor * pair, 0.0)
    ytotal = np.where(good, prefactor * total, 0.0)
    if np.any(ytotal < -1e-12 * max(float(np.max(yself)), 1e-300)):
        warnings.warn(
            "finite-screen CTR signed pair estimator is unresolved; "
            "returned unclipped negative spectral bins",
            RuntimeWarning,
            stacklevel=2,
        )
    theta = np.where(good, np.arccos(np.clip(cosine, -1, 1)), np.nan)
    phi = np.broadcast_to(np.arctan2(ky, kx), (ny, nx)).copy()
    return FiniteScreenBunchAngular(
        energy_eV=float(energy_eV),
        theta_rad=theta,
        phi_rad=phi,
        self_yield=yself,
        pair_yield=ypair,
        form_factor_squared=factor,
        yield_per_electron=ytotal,
        bunch_yield=n_physical * ytotal,
        physical_electrons=n_physical,
        n_histories=len(histories),
        effective_histories=w_sum**2 / w2_sum,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        source_record=source.record(),
    )


def finite_screen_realization_field(
    energy_eV: float,
    screens: Sequence[PECCircularScreen],
    *,
    source: OpticalBunchSource,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
) -> ScreenSpectralField:
    """Exactly sum physical-electron complex surface fields, for realisations.

    This full spectral field (V*s/m) can be sent through
    ``propagate_screen_field`` or ``observe_screen_green`` and later summed
    coherently across *linear* optical operators. A sampled ensemble that
    represents N >> M physical electrons does NOT have a unique realizable
    summed field, so this function explicitly rejects it.
    """
    histories, _, _ = _validate_histories(screens, source, energy_eV)
    if source.law != "realization":
        raise ValueError("only law='realization' has a physical complex bunch field")
    total = None
    for field in _screen_fields(energy_eV, histories, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m):
        if total is None:
            total = np.array(field.electric_xy_v_s_m, copy=True)
        else:
            total += field.electric_xy_v_s_m
    assert total is not None  # _validate_histories rejects empty histories
    return ScreenSpectralField(float(energy_eV), dx_m, dy_m, total)


def finite_screen_bunch_band(
    energy_eV,
    screens: Sequence[PECCircularScreen],
    *,
    source: OpticalBunchSource,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    theta_max_rad: float,
    theta_min_rad: float = 0.0,
) -> FiniteScreenBunchBand:
    """Integrate an opt-in finite-screen bunch spectrum over an angular cone.

    Output photons/(eV pulse) can feed the energy-channel operator in #617,
    provided any further instrument acceptance is explicitly modelled. The
    cone is in **vacuum far-zone angles**, not a finite-distance detector.
    """
    e = np.asarray(energy_eV, dtype=float)
    if (
        e.ndim != 1
        or e.size < 2
        or not np.all(np.isfinite(e))
        or e[0] <= 0
        or not np.all(np.diff(e) > 0)
    ):
        raise ValueError("energy_eV must be a finite positive increasing 1-D grid")
    s = tuple(screens)
    spectrum = np.array(
        [
            finite_screen_bunch_angular(
                energy, s, source=source, nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m
            ).collect_cone(theta_max_rad=theta_max_rad, theta_min_rad=theta_min_rad)
            for energy in e
        ]
    )
    return FiniteScreenBunchBand(
        energy_eV=e.copy(),
        collected_photons_per_eV_per_pulse=spectrum,
        theta_min_rad=theta_min_rad,
        theta_max_rad=theta_max_rad,
        source_record=source.record(),
    )
