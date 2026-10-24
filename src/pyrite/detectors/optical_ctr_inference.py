"""Conditional Gaussian CTR bunch-length inference from calibrated two-port scans.

This is a *forward-model fit*, not a phase reconstruction. The incident
spectral photon density is specified by separately normalized incoherent and
unit-form-factor coherent templates, already integrated over the collected
spatial/polarization acceptance. Only an independent Gaussian *temporal*
offset law is fitted; the transverse source, instrument calibration, and
spectral templates are held fixed. Its temporal intensity form factor is
``exp(-(E * sigma_t / hbar)**2)`` with RMS sigma_t, not FWHM.

The same ideal Martin--Puplett port transfer and calibrated pulse detector as
``optical_interferometer`` are used. The fitted common positive normalization
and independent constant port offsets are optional nuisance parameters. A
known fractional prior on the common normalization is supported, but this
is NOT a full uncertainty model for charge, transverse source, or optics.

A conditional profile-likelihood interval (Delta chi2=3.841 for 95%, one
parameter, independent known Gaussian voltage errors) is reported, with None
for any open interval end at the declared scan bounds. Finite-band/missing-band
ambiguity is exposed; never extrapolate the measured band to reconstruct a
unique bunch shape. Validation: optical-instrument-observable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .optical_energy import CalibratedPulseEnergyDetector
from .optical_interferometer import (
    IdealMartinPuplett,
    OpticalInterferogram,
    OpticalSpectralCoherency,
    _port_spectra,
)

__all__ = [
    "GaussianCTRTemplates",
    "OpticalVoltageScan",
    "GaussianCTRFit",
    "fit_gaussian_ctr",
    "gaussian_ctr_forward_voltage",
    "load_gaussian_ctr_fit",
]

_HBAR_EV_FS = 0.6582119569509067
_ELECTRON_VOLT_J = 1.602176634e-19
_DELTA_CHI2_95 = 3.841458820694124


def _positive_increasing(e):
    return (
        e.ndim == 1
        and e.size >= 2
        and np.all(np.isfinite(e))
        and e[0] > 0
        and np.all(np.diff(e) > 0)
    )


@dataclass(frozen=True, eq=False)
class GaussianCTRTemplates:
    """Collected self and unit-temporal-form-factor coherent photon densities.

    Both densities are photons/(eV *physical bunch*), nonnegative and already
    normalized with physical ``N`` and ``N(N-1)`` factors, respectively.
    ``coherent_envelope`` is the pair *excess* BEFORE temporal attenuation,
    not the fully coherent total yield or per-history estimate. Do NOT apply
    another physical charge or pair multiplier during fitting. Any transverse
    form factor, spatial geometry and detector acceptance are already included
    in these frozen, explicitly sourced templates.

    A declared unit Jones-linear polarization assumption is required; scalar
    intensities do not establish it. No optical response below/above the input
    energy band is inferred. This template is incompatible with complex signed
    pair estimates until those estimates are physically resolved.

    Validation: optical-instrument-observable
    """

    energy_eV: np.ndarray
    incoherent_photons_per_eV_per_pulse: np.ndarray
    coherent_envelope_photons_per_eV_per_pulse: np.ndarray
    polarization_angle_rad: float
    provenance: str

    def __post_init__(self):
        e = np.array(self.energy_eV, dtype=float, copy=True)
        if not _positive_increasing(e):
            raise ValueError("template energy_eV must be positive finite strictly increasing")
        for name in (
            "incoherent_photons_per_eV_per_pulse",
            "coherent_envelope_photons_per_eV_per_pulse",
        ):
            v = np.array(getattr(self, name), dtype=float, copy=True)
            if v.shape != e.shape or not np.all(np.isfinite(v)) or np.any(v < 0):
                raise ValueError(f"{name} must be a nonnegative finite density on energy_eV")
            v.setflags(write=False)
            object.__setattr__(self, name, v)
        if not np.isfinite(self.polarization_angle_rad):
            raise ValueError("polarization_angle_rad must be finite")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("template provenance is required")
        if not np.any(self.coherent_envelope_photons_per_eV_per_pulse > 0):
            raise ValueError(
                "coherent envelope is identically zero; bunch duration is not observable"
            )
        e.setflags(write=False)
        object.__setattr__(self, "energy_eV", e)

    def photon_density(self, rms_duration_fs: float) -> np.ndarray:
        """Collected photons/(eV pulse) under the stated Gaussian temporal law."""
        if not np.isfinite(rms_duration_fs) or rms_duration_fs < 0:
            raise ValueError("RMS bunch duration must be finite and nonnegative")
        temporal = np.exp(-np.square(self.energy_eV * rms_duration_fs / _HBAR_EV_FS))
        return (
            self.incoherent_photons_per_eV_per_pulse
            + self.coherent_envelope_photons_per_eV_per_pulse * temporal
        )

    def with_charge_jitter(
        self, *, mean_electrons: float, charge_rms_fraction: float
    ) -> GaussianCTRTemplates:
        """Shot-averaged templates for a fluctuating physical bunch population.

        The templates hold one bunch of ``mean_electrons`` = N̄: incoherent
        yield proportional to N and pair excess to N(N-1). For independent
        shot-to-shot population N with mean N̄ and RMS ``r * N̄``,
        ``<N> = N̄`` and ``<N(N-1)> = N̄^2 (1 + r^2) - N̄``, so only the pair
        envelope scales, exactly, by ``1 + N̄ r^2 / (N̄ - 1)`` (-> ``1 + r^2``
        for N̄ >> 1). Assumes the charge fluctuation is uncorrelated with the
        bunch length and transverse shape and that voltages are linear
        averages over shots. Common arrival-time jitter cancels in the
        interferogram and is not modeled here. Not a per-shot charge
        normalization: use it only for unnormalized, shot-averaged scans.
        """
        n_mean = float(mean_electrons)
        r = float(charge_rms_fraction)
        if not np.isfinite(n_mean) or n_mean <= 1:
            raise ValueError("mean_electrons must be finite and > 1")
        if not np.isfinite(r) or r < 0:
            raise ValueError("charge_rms_fraction must be finite and nonnegative")
        factor = 1.0 + n_mean * r * r / (n_mean - 1.0)
        return replace(
            self,
            coherent_envelope_photons_per_eV_per_pulse=(
                self.coherent_envelope_photons_per_eV_per_pulse * factor
            ),
            provenance=(
                f"{self.provenance}; shot-averaged charge jitter "
                f"N={n_mean:.6g}, rms={r:.6g}, pair factor={factor:.12g}"
            ),
        )

    def coherent_coverage(self, rms_duration_fs: float):
        """Energy at 1/e temporal factor and factors at the measured band edges."""
        if rms_duration_fs <= 0:
            return np.inf, (1.0, 1.0)
        ew = _HBAR_EV_FS / rms_duration_fs
        return ew, (
            float(np.exp(-((self.energy_eV[0] / ew) ** 2))),
            float(np.exp(-((self.energy_eV[-1] / ew) ** 2))),
        )


@dataclass(frozen=True, eq=False)
class OpticalVoltageScan:
    """Measured two-port *isolated-pulse* voltages and independent RMS errors.

    Values and uncertainties have shape (n_delay,2), corresponding to the
    exact mirror positions. This is NOT a lock-in response, interferogram
    derivative, or per-exposure voltage. Correlated errors, charge jitter,
    calibration uncertainty and drifts need another likelihood model.
    """

    mirror_displacement_m: np.ndarray
    voltage_v_ports: np.ndarray
    uncertainty_v_ports: np.ndarray
    provenance: str
    saturated_ports: np.ndarray | None = None

    def __post_init__(self):
        shift = np.array(self.mirror_displacement_m, dtype=float, copy=True)
        v = np.array(self.voltage_v_ports, dtype=float, copy=True)
        s = np.array(self.uncertainty_v_ports, dtype=float, copy=True)
        if (
            shift.ndim != 1
            or shift.size < 4
            or not np.all(np.isfinite(shift))
            or np.unique(shift).size < 4
        ):
            raise ValueError("scan requires at least four distinct finite mirror positions")
        if v.shape != (shift.size, 2) or s.shape != v.shape or not np.all(np.isfinite(v)):
            raise ValueError("port voltages and uncertainties must have shape (n_delay, 2)")
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("RMS voltage uncertainty must be finite and strictly positive")
        mask = (
            np.zeros(v.shape, bool)
            if self.saturated_ports is None
            else np.array(self.saturated_ports, dtype=bool, copy=True)
        )
        if mask.shape != v.shape or np.any(mask):
            raise ValueError("saturated measurements cannot use the linear-voltage fit")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("measurement provenance is required")
        for name, value in (
            ("mirror_displacement_m", shift),
            ("voltage_v_ports", v),
            ("uncertainty_v_ports", s),
            ("saturated_ports", mask),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_forward_model(cls, result: OpticalInterferogram, *, uncertainty_v_ports, provenance):
        """Explicit *synthetic* scan; not evidence of experimental validation."""
        return cls(
            result.mirror_displacement_m,
            result.expected_voltage_v_ports,
            uncertainty_v_ports,
            provenance,
            result.saturated_ports,
        )


@dataclass(frozen=True, eq=False)
class GaussianCTRFit:
    """Conditional one-parameter profile result; open CI endpoints are None."""

    rms_duration_fs: float
    confidence_interval_95_fs: tuple[float | None, float | None]
    relative_source_scale: float
    voltage_offsets_v: tuple[float, float]
    chi_square: float
    degrees_of_freedom: int
    profile_bounds_fs: tuple[float, float]
    is_identified_95: bool
    is_multimodal: bool
    turnover_energy_eV: float
    temporal_factor_at_band_edges: tuple[float, float]
    predicted_voltage_v_ports: np.ndarray
    residual_v_ports: np.ndarray
    model_assumptions: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        """Persist conditional profile result, arrays and hashes without pickles."""
        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(target.suffix + ".npz")
        info = dict(
            rms_duration_fs=self.rms_duration_fs,
            confidence_interval_95_fs=self.confidence_interval_95_fs,
            relative_source_scale=self.relative_source_scale,
            voltage_offsets_v=self.voltage_offsets_v,
            chi_square=self.chi_square,
            degrees_of_freedom=self.degrees_of_freedom,
            profile_bounds_fs=self.profile_bounds_fs,
            is_identified_95=self.is_identified_95,
            is_multimodal=self.is_multimodal,
            turnover_energy_eV=self.turnover_energy_eV,
            temporal_factor_at_band_edges=self.temporal_factor_at_band_edges,
            model_assumptions=self.model_assumptions,
            metadata=self.metadata,
        )
        np.savez(
            target,
            metadata_json=np.array(json.dumps(info, sort_keys=True)),
            predicted_voltage_v_ports=self.predicted_voltage_v_ports,
            residual_v_ports=self.residual_v_ports,
        )
        return target


def load_gaussian_ctr_fit(path) -> GaussianCTRFit:
    """Read a result saved by :meth:`GaussianCTRFit.save`."""
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["metadata_json"]))
        if info.get("metadata", {}).get("model") != "gaussian-ctr-conditional-v1":
            raise ValueError("not a Gaussian CTR conditional inference result")
        info["predicted_voltage_v_ports"] = np.array(data["predicted_voltage_v_ports"])
        info["residual_v_ports"] = np.array(data["residual_v_ports"])
    for name in (
        "confidence_interval_95_fs",
        "voltage_offsets_v",
        "profile_bounds_fs",
        "temporal_factor_at_band_edges",
        "model_assumptions",
    ):
        info[name] = tuple(info[name])
    return GaussianCTRFit(**info)


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(repr((a.shape, a.dtype.str)).encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _trapz_weights(e):
    widths = np.empty_like(e)
    widths[0] = (e[1] - e[0]) / 2
    widths[-1] = (e[-1] - e[-2]) / 2
    widths[1:-1] = (e[2:] - e[:-2]) / 2
    return widths


def _response_kernel(templates, shifts, instrument, detectors, max_elements):
    """Calibrated output voltage per incident photon density at each energy.

    This calls the *same* port transfer as ``polarized_interferogram`` rather
    than reimplementing Jones, jitter or phase conventions. Detector spectral
    integration is the same trapezoid as CalibratedPulseEnergyDetector.predict.
    """
    e = templates.energy_eV
    if shifts.size * e.size * 2 > max_elements:
        raise ValueError("inference exceeds max_spectral_elements")
    unit_source = OpticalSpectralCoherency.linearly_polarized(
        e,
        np.ones_like(e),
        angle_rad=templates.polarization_angle_rad,
        provenance=templates.provenance,
    )
    ports, _ = _port_spectra(unit_source, instrument, shifts)
    widths = _trapz_weights(e)
    for p, detector in enumerate(detectors):
        if e[0] < detector.energy_eV[0] or e[-1] > detector.energy_eV[-1]:
            raise ValueError(f"port {p} has spectral samples outside calibrated detector band")
        eta = np.interp(e, detector.energy_eV, detector.efficiency)
        ports[:, p, :] *= (eta * e * widths * _ELECTRON_VOLT_J * detector.pulse_gain_v_per_j)[
            None, :
        ]
    return ports


def gaussian_ctr_forward_voltage(
    templates: GaussianCTRTemplates,
    mirror_displacement_m,
    *,
    rms_duration_fs,
    instrument: IdealMartinPuplett,
    detectors: tuple[CalibratedPulseEnergyDetector, CalibratedPulseEnergyDetector],
    relative_source_scale: float = 1.0,
    voltage_offsets_v: tuple[float, float] = (0.0, 0.0),
    max_spectral_elements: int = 4_000_000,
) -> np.ndarray:
    """Return calibrated unsaturated model two-port voltages of Gaussian CTR."""
    shifts = np.asarray(mirror_displacement_m, dtype=float)
    if shifts.ndim != 1 or shifts.size == 0 or not np.all(np.isfinite(shifts)):
        raise ValueError("mirror_displacement_m must be a finite nonempty 1-D array")
    if (
        not isinstance(instrument, IdealMartinPuplett)
        or len(detectors) != 2
        or not all(isinstance(d, CalibratedPulseEnergyDetector) for d in detectors)
    ):
        raise TypeError("supply ideal Martin--Puplett and two calibrated detectors")
    if not np.isfinite(relative_source_scale) or relative_source_scale < 0:
        raise ValueError("relative_source_scale must be finite and nonnegative")
    offsets = np.asarray(voltage_offsets_v, dtype=float)
    if offsets.shape != (2,) or not np.all(np.isfinite(offsets)):
        raise ValueError("voltage_offsets_v must be two finite entries")
    kernel = _response_kernel(templates, shifts, instrument, detectors, max_spectral_elements)
    voltages = relative_source_scale * (kernel @ templates.photon_density(rms_duration_fs))
    return voltages + offsets[None, :]


def _profile_at_sigma(
    sigma, templates, kernel, scan, *, fit_scale, fit_offsets, scale_prior_rms_fraction
):
    model = kernel @ templates.photon_density(sigma)
    y, uncertainty = scan.voltage_v_ports, scan.uncertainty_v_ports
    values = y.flatten()
    weights = (1 / uncertainty).flatten()
    base = model.flatten()
    design = []
    base_norm = max(float(np.max(np.abs(base))), 1e-300)
    if fit_scale:
        design.append(base / base_norm)
        target = values
    else:
        target = values - base
    if fit_offsets:
        design.extend(
            (np.indices(model.shape)[1] == port).astype(float).flatten() for port in (0, 1)
        )
    if design:
        x = np.column_stack(design)
        wx, wy = x * weights[:, None], target * weights
        if fit_scale and scale_prior_rms_fraction is not None:
            prior_row = np.zeros(x.shape[1])
            prior_row[0] = 1.0 / (base_norm * scale_prior_rms_fraction)
            wx = np.vstack((wx, prior_row))
            wy = np.r_[wy, 1.0 / scale_prior_rms_fraction]
        values_hat = np.linalg.lstsq(wx, wy, rcond=None)[0]
        if fit_scale and values_hat[0] < 0:
            # Physical amplitude nonnegative; re-solve the offsets with scale 0.
            values_hat[0] = 0.0
            if fit_offsets:
                for p in range(2):
                    weight_sq = 1 / uncertainty[:, p] ** 2
                    values_hat[p + 1] = float(np.average(y[:, p], weights=weight_sq))
    else:
        values_hat = np.empty(0)
    scale = float(values_hat[0] / base_norm) if fit_scale else 1.0
    offsets = (
        tuple(map(float, values_hat[(1 if fit_scale else 0) :])) if fit_offsets else (0.0, 0.0)
    )
    predicted = scale * model + np.array(offsets)[None, :]
    resid = (y - predicted) / uncertainty
    chi2 = float(np.sum(resid**2))
    if fit_scale and scale_prior_rms_fraction is not None:
        chi2 += float(((scale - 1) / scale_prior_rms_fraction) ** 2)
    return chi2, scale, offsets, predicted


def _check_fit_arguments(
    scan,
    templates,
    instrument,
    detectors,
    sigma_bounds_fs,
    *,
    fit_scale,
    scale_prior_rms_fraction,
    profile_points,
) -> tuple[float, float]:
    if not isinstance(scan, OpticalVoltageScan) or not isinstance(templates, GaussianCTRTemplates):
        raise TypeError("provide an OpticalVoltageScan and GaussianCTRTemplates")
    lower, upper = map(float, sigma_bounds_fs)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper):
        raise ValueError("sigma_bounds_fs must be finite, positive, increasing")
    if profile_points < 21 or profile_points > 2001:
        raise ValueError("profile_points must be between 21 and 2001")
    if scale_prior_rms_fraction is not None and (
        not fit_scale or not np.isfinite(scale_prior_rms_fraction) or scale_prior_rms_fraction <= 0
    ):
        raise ValueError("scale prior needs fitted positive fractional uncertainty")
    if (
        not isinstance(instrument, IdealMartinPuplett)
        or len(detectors) != 2
        or not all(isinstance(d, CalibratedPulseEnergyDetector) for d in detectors)
    ):
        raise TypeError("supply ideal Martin--Puplett and two calibrated detectors")
    return lower, upper


def _refined_profile(profile, lower, upper, points):
    """Log-grid chi2 profile, with each interior local minimum refined."""
    grid = np.geomspace(lower, upper, points)
    values = np.array([profile(s)[0] for s in grid])
    candidates = [0, len(grid) - 1]
    candidates += [
        i
        for i in range(1, len(grid) - 1)
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]
    ]
    for i in candidates[2:]:
        solution = minimize_scalar(
            lambda z: profile(np.exp(z))[0],
            bounds=(np.log(grid[i - 1]), np.log(grid[i + 1])),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if solution.success:
            s = float(np.exp(solution.x))
            v = profile(s)[0]
            if v < values[i]:
                grid = np.append(grid, s)
                values = np.append(values, v)
    order = np.argsort(grid)
    return grid[order], values[order]


def _profile_interval(profile, grid, values, threshold):
    """Delta-chi2 interval endpoints (None when open) and multimodality flag."""
    good = values <= threshold
    # Grid components below threshold identify alternate admissible time scales.
    runs = np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) > 1) + 1)
    multimodal = len(runs) > 1
    first, last = int(np.flatnonzero(good)[0]), int(np.flatnonzero(good)[-1])
    low_ci = (
        None
        if first == 0
        else float(brentq(lambda x: profile(x)[0] - threshold, grid[first - 1], grid[first]))
    )
    high_ci = (
        None
        if last == len(grid) - 1
        else float(brentq(lambda x: profile(x)[0] - threshold, grid[last], grid[last + 1]))
    )
    return low_ci, high_ci, multimodal


def fit_gaussian_ctr(
    scan: OpticalVoltageScan,
    templates: GaussianCTRTemplates,
    *,
    instrument: IdealMartinPuplett,
    detectors: tuple[CalibratedPulseEnergyDetector, CalibratedPulseEnergyDetector],
    sigma_bounds_fs: tuple[float, float] = (1.0, 1000.0),
    fit_scale: bool = True,
    fit_offsets: bool = False,
    scale_prior_rms_fraction: float | None = None,
    profile_points: int = 101,
    max_spectral_elements: int = 4_000_000,
) -> GaussianCTRFit:
    """Fit Gaussian RMS time and conditionally profile its 95% interval.

    ``sigma_bounds_fs`` are *analysis bounds*, not physical limits; an open
    confidence endpoint means the measured band/scan did not bound that side.
    Nuisance scale multiplies both templates; port offsets must be opted into.
    Correlated/systematic errors and unknown spectral source shape are not
    absorbed into the statistical CI. The fit must not be reported as a unique
    bunch-profile reconstruction.

    Validation: optical-instrument-observable
    """
    lower, upper = _check_fit_arguments(
        scan,
        templates,
        instrument,
        detectors,
        sigma_bounds_fs,
        fit_scale=fit_scale,
        scale_prior_rms_fraction=scale_prior_rms_fraction,
        profile_points=profile_points,
    )
    kernel = _response_kernel(
        templates, scan.mirror_displacement_m, instrument, detectors, max_spectral_elements
    )
    nuisance_count = int(fit_scale) + 2 * int(fit_offsets)
    data_count = scan.voltage_v_ports.size
    if data_count <= nuisance_count + 1:
        raise ValueError("insufficient scan points for Gaussian CTR fit")

    def profile(sigma):
        return _profile_at_sigma(
            float(sigma),
            templates,
            kernel,
            scan,
            fit_scale=fit_scale,
            fit_offsets=fit_offsets,
            scale_prior_rms_fraction=scale_prior_rms_fraction,
        )

    grid, values = _refined_profile(profile, lower, upper, profile_points)
    idx = int(np.argmin(values))
    sigma = float(grid[idx])
    chi2, scale, offsets, predicted = profile(sigma)
    threshold = chi2 + _DELTA_CHI2_95
    low_ci, high_ci, multimodal = _profile_interval(profile, grid, values, threshold)
    for p, detector in enumerate(detectors):
        if detector.saturation_v is not None and np.any(predicted[:, p] >= detector.saturation_v):
            raise ValueError(
                "fitted expected voltage crosses detector saturation; linear fit invalid"
            )
    ew, factors = templates.coherent_coverage(sigma)
    metadata = {
        "model": "gaussian-ctr-conditional-v1",
        "template_provenance": templates.provenance,
        "scan_provenance": scan.provenance,
        "spectral_band_eV": [float(templates.energy_eV[0]), float(templates.energy_eV[-1])],
        "template_sha256": _digest_arrays(
            templates.energy_eV,
            templates.incoherent_photons_per_eV_per_pulse,
            templates.coherent_envelope_photons_per_eV_per_pulse,
        ),
        "scan_sha256": _digest_arrays(
            scan.mirror_displacement_m, scan.voltage_v_ports, scan.uncertainty_v_ports
        ),
        "instrument": asdict(instrument),
        "detector_response_sha256": [d.response_sha256 for d in detectors],
        "fit_scale": fit_scale,
        "fit_offsets": fit_offsets,
        "scale_prior_rms_fraction": scale_prior_rms_fraction,
        "profile_points": profile_points,
        "conditional_statistics": "Independent normal voltage errors, known variances; delta chi2 = 3.841 for one parameter; no systematic covariance",
        "missing_band": "No data or source spectrum outside the supplied band; missing-band continuation and phase remain unspecified",
        "physical_normalization": "Templates already per physical bunch; no additional N or N(N-1) multiplier",
    }
    return GaussianCTRFit(
        rms_duration_fs=sigma,
        confidence_interval_95_fs=(low_ci, high_ci),
        relative_source_scale=scale,
        voltage_offsets_v=offsets,
        chi_square=chi2,
        degrees_of_freedom=data_count - nuisance_count - 1,
        profile_bounds_fs=(lower, upper),
        is_identified_95=(low_ci is not None and high_ci is not None and not multimodal),
        is_multimodal=multimodal,
        turnover_energy_eV=ew,
        temporal_factor_at_band_edges=factors,
        predicted_voltage_v_ports=predicted,
        residual_v_ports=scan.voltage_v_ports - predicted,
        model_assumptions=(
            "Gaussian independent temporal offsets; sigma is RMS, not FWHM",
            "templates contain physical bunch population, transverse acceptance and polarization",
            "no missing-band extrapolation or unique-profile/phase reconstruction",
            "conditional independent Gaussian voltage errors; no systematic covariance",
            "ideal Jones Martin-Puplett transfer, calibration held fixed",
        ),
        metadata=json.loads(json.dumps(metadata, sort_keys=True)),
    )
