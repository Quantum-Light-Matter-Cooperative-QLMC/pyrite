"""Finite-band minimal-phase (Kramers--Kronig) CTR bunch-profile reconstruction.

A measured coherent spectrum gives only the longitudinal form-factor magnitude
``|f(omega)|``. With the convention of Leissner et al., PAC 1999, p. 2172,
Eq. (2), ``f(omega) = int rho(t) exp(i omega t) dt``, the *minimal phase* is
their Eq. (6) (Lai and Sievers, NIM A 397, 221 (1997)),

    psi(omega) = -(2 omega / pi) PV int_0^inf ln(|f(u)| / |f(omega)|) / (u^2 - omega^2) du,

and the profile is the inverse transform

    rho(t) = (1 / pi) int_0^inf |f(omega)| cos(psi(omega) - omega t) d omega.

Leissner's printed Eq. (7) has ``+ omega z / c``, i.e. ``rho(-t)`` in the
Eq. (2) convention; the two differ only by the time reversal that a magnitude
cannot identify anyway. Here ``t`` is in fs and photon energy ``E = hbar omega``
in eV.

The result is NOT a unique bunch profile. Magnitude-only data leave
unidentified: time reversal ``rho(-t)``; a common arrival-time shift (linear
phase); every zero of the analytic continuation that could lie in the other
half plane (Blaschke factors -- a non-minimal-phase profile is returned as its
minimal-phase partner); and the spectrum outside the measured band, which is
replaced by an explicit, recorded :class:`MissingBandModel`. Every
reconstruction carries ``is_unique = False`` and the list of ambiguities.

Numerics: ``ln|f|`` is piecewise linear in ``E`` between nodes; Eq. (6) is
integrated exactly segment by segment, plus the closed-form power-law tail
above the last node. The inverse transform treats ``|f|`` and ``psi``
as piecewise linear (Filon-type exact segment integrals, so the oscillation
need not be resolved by the node spacing) and is truncated at the last node.

Validation: optical-phase-retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import spence

__all__ = [
    "FormFactorUncertainty",
    "MinimalPhaseReconstruction",
    "MissingBandModel",
    "ReconstructionUncertainty",
    "minimal_phase",
    "propagate_reconstruction_uncertainty",
    "reconstruct_minimal_phase_profile",
]

_HBAR_EV_FS = 0.6582119569509067
_LOW_MODELS = ("gaussian-to-unity", "constant")
AMBIGUITIES = (
    "time reversal: rho(-t) has the same |f|",
    "arrival time: a common time shift (linear phase) is not identified",
    "zero flipping: zeros of f in the other half plane (non-minimal phase) are "
    "returned as their minimal-phase reflections",
    "missing band: the spectrum outside the measured band is the declared continuation",
)


@dataclass(frozen=True)
class MissingBandModel:
    """Declared continuation of ``|f|`` outside the measured band.

    ``low``: ``"gaussian-to-unity"`` continues ``ln|f| = ln|f(E_lo)| (E/E_lo)^2``
    so that ``|f(0)| = 1`` (normalized charge; requires ``|f(E_lo)| <= 1``);
    ``"constant"`` holds ``|f(E_lo)|`` down to ``E = 0`` (no normalization
    assumed). Above the band ``|f| ∝ E^-p`` with ``p = high_power_exponent``,
    sampled on ``high_nodes`` log-spaced nodes up to
    ``high_extent_factor * E_hi`` and continued analytically in Eq. (6) beyond.
    The profile synthesis is truncated at that last node.

    Validation: optical-phase-retrieval
    """

    low: str = "gaussian-to-unity"
    high_power_exponent: float = 2.0
    low_nodes: int = 64
    high_nodes: int = 128
    high_extent_factor: float = 8.0

    def __post_init__(self):
        if self.low not in _LOW_MODELS:
            raise ValueError(f"low continuation must be one of {_LOW_MODELS}")
        if not np.isfinite(self.high_power_exponent) or self.high_power_exponent <= 0:
            raise ValueError("high_power_exponent must be finite and positive")
        if self.low_nodes < 2 or self.high_nodes < 2:
            raise ValueError("low_nodes and high_nodes must be at least 2")
        if not np.isfinite(self.high_extent_factor) or self.high_extent_factor <= 1:
            raise ValueError("high_extent_factor must be finite and > 1")

    def describe(self) -> dict:
        """JSON-ready record of the continuation."""
        return {
            "low": self.low,
            "high": f"|f| proportional to E^-{self.high_power_exponent:.6g}",
            "low_nodes": self.low_nodes,
            "high_nodes": self.high_nodes,
            "high_extent_factor": self.high_extent_factor,
        }


def _check_band(energy_eV, form_factor_squared):
    e = np.asarray(energy_eV, dtype=float)
    f2 = np.asarray(form_factor_squared, dtype=float)
    if (
        e.ndim != 1
        or e.size < 4
        or not np.all(np.isfinite(e))
        or e[0] <= 0
        or np.any(np.diff(e) <= 0)
    ):
        raise ValueError("energy_eV must be >= 4 positive finite strictly increasing values")
    if f2.shape != e.shape or not np.all(np.isfinite(f2)) or np.any(f2 <= 0):
        raise ValueError("form_factor_squared must be positive and finite on energy_eV")
    return e, f2


def _extend(e, log_f, model: MissingBandModel):
    """Extended nodes, ln|f| on them, and the measured-node mask."""
    lo, hi = e[0], e[-1]
    e_low = np.linspace(0.0, lo, model.low_nodes + 1)[:-1]
    if model.low == "gaussian-to-unity":
        if log_f[0] > 0:
            raise ValueError("gaussian-to-unity continuation needs |f(E_lo)| <= 1")
        l_low = log_f[0] * (e_low / lo) ** 2
    else:
        l_low = np.full_like(e_low, log_f[0])
    e_high = np.geomspace(hi, model.high_extent_factor * hi, model.high_nodes + 1)[1:]
    l_high = log_f[-1] - model.high_power_exponent * np.log(e_high / hi)
    nodes = np.concatenate((e_low, e, e_high))
    measured = np.zeros(nodes.size, bool)
    measured[e_low.size : e_low.size + e.size] = True
    return nodes, np.concatenate((l_low, log_f, l_high)), measured


def _chi2(r):
    """Legendre chi function chi_2(r) = (Li2(r) - Li2(-r)) / 2 for 0 <= r <= 1."""
    return 0.5 * (spence(1.0 - r) - spence(1.0 + r))


def _safe_log(d):
    return np.log(np.where(d > 0, d, 1.0))


def minimal_phase(energy_eV, log_magnitude, *, high_power_exponent, chunk=256) -> np.ndarray:
    """Eq. (6) minimal phase [rad] at every node of a piecewise-linear ``ln|f|``.

    ``energy_eV`` starts at 0 or above and is strictly increasing;
    ``ln|f|`` is linear between nodes and ``|f| ∝ E^-p`` beyond the last node
    ``X``. Each segment integral is exact: with ``g = ln|f| - ln|f(w)|``
    linear, ``int g/(x^2-w^2) = (1/2w)[g(w) ln|x-w| - g(-w) ln(x+w)]`` (``g``
    continued linearly), and ``g(w) = 0`` on segments touching ``w``. The tail
    is ``(1/w)[g(X) artanh(w/X) - p chi_2(w/X)]``.

    Validation: optical-phase-retrieval
    """
    e = np.asarray(energy_eV, dtype=float)
    lf = np.asarray(log_magnitude, dtype=float)
    p = float(high_power_exponent)
    a, b = e[:-1], e[1:]
    beta = np.diff(lf) / np.diff(e)
    top = e[-1]
    phase = np.zeros_like(e)
    for start in range(0, e.size, chunk):
        w = e[start : start + chunk, None]
        level = lf[start : start + chunk, None]
        ga = lf[None, :-1] - level
        plus = ga + beta * (w - a)
        minus = ga + beta * (-w - a)
        touches = (a == w) | (b == w)
        plus = np.where(touches, 0.0, plus)
        seg = plus * (_safe_log(np.abs(b - w)) - _safe_log(np.abs(a - w))) - minus * (
            np.log(b + w) - _safe_log(a + w)
        )
        r = np.clip(w[:, 0] / top, 0.0, 1.0)
        g_top = lf[-1] - level[:, 0]
        # At w = X, g(X) = 0 and the artanh term vanishes.
        log_term = np.where(r < 1.0, g_top * np.arctanh(np.where(r < 1.0, r, 0.0)), 0.0)
        tail = -(2.0 / np.pi) * (log_term - p * _chi2(r))
        values = -np.sum(seg, axis=1) / np.pi + tail
        phase[start : start + chunk] = np.where(w[:, 0] > 0, values, 0.0)
    return phase


def _phi(z):
    """int_0^1 (1-v) e^{zv} dv and int_0^1 v e^{zv} dv, stable at small |z|."""
    small = np.abs(z) < 0.5
    zs = np.where(small, z, 0.0)
    za = np.where(small, 1.0, z)
    ez = np.exp(za)
    pa = np.where(small, 0.0, (ez - 1 - za) / za**2)
    pb = np.where(small, 0.0, (ez * (za - 1) + 1) / za**2)
    term, sa, sb, fact = np.ones_like(zs), 0.0, 0.0, 2.0
    for n in range(10):
        sa = sa + term / fact
        sb = sb + (n + 1) * term / fact
        term = term * zs
        fact *= n + 3
    return np.where(small, sa, pa), np.where(small, sb, pb)


def _synthesize(e, mag, phase, time_fs, chunk=128):
    """Exact inverse transform of piecewise-linear |f| and psi; 1/fs."""
    h = np.diff(e)
    dpsi = np.diff(phase)
    out = np.empty(time_fs.size)
    for start in range(0, time_fs.size, chunk):
        t = time_fs[start : start + chunk, None]
        theta_a = phase[None, :-1] - e[None, :-1] * t / _HBAR_EV_FS
        z = 1j * (dpsi[None, :] - h[None, :] * t / _HBAR_EV_FS)
        pa, pb = _phi(z)
        seg = np.exp(1j * theta_a) * h * (mag[None, :-1] * pa + mag[None, 1:] * pb)
        out[start : start + chunk] = np.real(np.sum(seg, axis=1)) / (np.pi * _HBAR_EV_FS)
    return out


def _trapezoid(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _fwhm(t, rho):
    k = int(np.argmax(rho))
    half = 0.5 * rho[k]
    left = np.flatnonzero(rho[:k] < half)
    right = np.flatnonzero(rho[k:] < half)
    if left.size == 0 or right.size == 0:
        return None
    i, j = left[-1], k + right[0]
    t_lo = np.interp(half, [rho[i], rho[i + 1]], [t[i], t[i + 1]])
    t_hi = np.interp(half, [rho[j], rho[j - 1]], [t[j], t[j - 1]])
    return float(t_hi - t_lo)


def _spectral_minima(e, mag, depth):
    """Interior local minima of |f| below ``depth`` x the geometric mean of the
    adjacent peaks (reached by climbing monotonically left and right).

    A heuristic flag for zeros near the real axis, not a zero locator.
    """
    found = []
    for i in range(1, e.size - 1):
        if mag[i] <= mag[i - 1] and mag[i] < mag[i + 1]:
            left = i
            while left > 0 and mag[left - 1] >= mag[left]:
                left -= 1
            right = i
            while right < e.size - 1 and mag[right + 1] >= mag[right]:
                right += 1
            if mag[i] < depth * np.sqrt(mag[left] * mag[right]):
                found.append(float(e[i]))
    return tuple(found)


@dataclass(frozen=True, eq=False)
class MinimalPhaseReconstruction:
    """Minimal-phase profile of a finite-band ``|f|``; never a unique profile.

    ``energy_eV``/``form_factor_magnitude``/``phase_rad`` are on the extended
    nodes (``measured`` masks the input band). ``profile_per_fs`` is
    ``rho(t)`` on ``time_fs``; its area is ``|f(0)|`` up to window and
    high-frequency truncation (``window_area``). ``rms_duration_fs`` and
    ``fwhm_fs`` are of the signed reconstructed profile on the window (NaN
    when its area or variance there is not positive, e.g. long negative
    tails of a ``"constant"`` low continuation).
    ``negative_lobe_fraction`` (``-int min(rho,0) / int |rho|``) above a
    few percent signals data or continuation inconsistent with a
    nonnegative minimal-phase profile. ``spectral_minima_eV`` are deep
    in-band minima of ``|f|`` (heuristic, relative to the adjacent peaks):
    candidate zeros near the real axis whose half plane (and hence the
    profile) is not identified.
    """

    energy_eV: np.ndarray
    form_factor_magnitude: np.ndarray
    phase_rad: np.ndarray
    measured: np.ndarray
    time_fs: np.ndarray
    profile_per_fs: np.ndarray
    window_area: float
    rms_duration_fs: float
    fwhm_fs: float | None
    negative_lobe_fraction: float
    extrapolated_power_fraction: float
    truncated_magnitude_fraction: float
    spectral_minima_eV: tuple[float, ...]
    missing_band: dict
    provenance: str
    ambiguities: tuple[str, ...] = AMBIGUITIES
    is_unique: bool = False
    metadata: dict = field(default_factory=dict)

    def time_reversed(self) -> tuple[np.ndarray, np.ndarray]:
        """The equally admissible ``rho(-t)`` as ``(time_fs, profile_per_fs)``."""
        return -self.time_fs[::-1], self.profile_per_fs[::-1].copy()


def _rms(t, rho):
    area = _trapezoid(rho, t)
    if not area > 0:
        return float("nan")
    mean = _trapezoid(t * rho, t) / area
    var = _trapezoid((t - mean) ** 2 * rho, t) / area
    return float(np.sqrt(var)) if var > 0 else float("nan")


def _power_fractions(nodes, log_f, measured, p):
    mag = np.exp(log_f)
    top, mag_top = nodes[-1], mag[-1]
    total2 = _trapezoid(mag**2, nodes) + (mag_top**2 * top / (2 * p - 1) if 2 * p > 1 else np.inf)
    band = nodes[measured]
    inside2 = _trapezoid(mag[measured] ** 2, band)
    tail1 = mag_top * top / (p - 1) if p > 1 else np.inf
    return float(1 - inside2 / total2), float(tail1 / (tail1 + _trapezoid(mag, nodes)))


def reconstruct_minimal_phase_profile(
    energy_eV,
    form_factor_squared,
    time_fs,
    *,
    missing_band: MissingBandModel,
    provenance: str,
    minima_depth: float = 0.5,
) -> MinimalPhaseReconstruction:
    """Minimal-phase (Kramers--Kronig) profile from a measured ``|f(E)|^2``.

    ``form_factor_squared`` is the longitudinal form factor squared on the
    measured band, already divided by the single-electron spectrum,
    ``N(N-1)``, the transverse form factor and the instrument response.
    ``time_fs`` is the evaluation window (strictly increasing). The
    continuation is ``missing_band`` and is recorded in the result.

    Validation: optical-phase-retrieval
    """
    if not isinstance(missing_band, MissingBandModel):
        raise TypeError("missing_band must be a MissingBandModel")
    if not provenance or not provenance.strip():
        raise ValueError("provenance is required")
    e, f2 = _check_band(energy_eV, form_factor_squared)
    t = np.asarray(time_fs, dtype=float)
    if t.ndim != 1 or t.size < 8 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_fs must be >= 8 finite strictly increasing values")
    nodes, log_f, measured = _extend(e, 0.5 * np.log(f2), missing_band)
    p = missing_band.high_power_exponent
    phase = minimal_phase(nodes, log_f, high_power_exponent=p)
    mag = np.exp(log_f)
    rho = _synthesize(nodes, mag, phase, t)
    extrapolated, truncated = _power_fractions(nodes, log_f, measured, p)
    absolute = _trapezoid(np.abs(rho), t)
    for array in (nodes, mag, phase, measured, t, rho):
        array.setflags(write=False)
    return MinimalPhaseReconstruction(
        energy_eV=nodes,
        form_factor_magnitude=mag,
        phase_rad=phase,
        measured=measured,
        time_fs=t,
        profile_per_fs=rho,
        window_area=_trapezoid(rho, t),
        rms_duration_fs=_rms(t, rho),
        fwhm_fs=_fwhm(t, rho),
        negative_lobe_fraction=float(-_trapezoid(np.minimum(rho, 0), t) / absolute),
        extrapolated_power_fraction=extrapolated,
        truncated_magnitude_fraction=truncated,
        spectral_minima_eV=_spectral_minima(e, np.sqrt(f2), minima_depth),
        missing_band=missing_band.describe(),
        provenance=provenance,
        metadata={
            "model": "minimal-phase-kk-v1",
            "source": "Leissner et al. PAC 1999 Eqs. (2),(6),(7); Lai & Sievers NIM A 397 (1997)",
            "convention": "f(omega) = int rho(t) exp(i omega t) dt; rho(t) = (1/pi) int |f| cos(psi - omega t)",
            "measured_band_eV": [float(e[0]), float(e[-1])],
        },
    )


@dataclass(frozen=True)
class FormFactorUncertainty:
    """Declared uncertainties of a derived ``|f|^2`` for Monte Carlo propagation.

    - ``form_factor_squared_rms``: independent absolute RMS per measured node
      (spectral noise), or ``None``.
    - ``charge_rms_fraction``: relative RMS of the bunch population ``N``;
      ``|f|^2 ∝ 1/N^2`` on every node together (one draw per sample).
    - ``transverse_rms_fraction`` / ``response_rms_fraction``: relative RMS
      (scalar or per node) of the transverse form factor and the instrument
      response that divide the measured spectrum (``|f|^2`` is divided by
      ``1 + rms * eps``). Each is a *fully
      correlated* systematic: one standard-normal draw per sample scales the
      whole array. This conservative choice keeps smooth shape errors intact
      rather than averaging them as independent noise.

    Draws are Gaussian; nonpositive samples are floored and values above 1
    are capped at 1 (both counted).

    Validation: optical-phase-retrieval
    """

    form_factor_squared_rms: np.ndarray | None = None
    charge_rms_fraction: float = 0.0
    transverse_rms_fraction: float | np.ndarray = 0.0
    response_rms_fraction: float | np.ndarray = 0.0

    def sample(self, f2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One perturbed ``|f|^2`` (before flooring/capping)."""
        out = np.array(f2, dtype=float)
        if self.form_factor_squared_rms is not None:
            sd = np.broadcast_to(np.asarray(self.form_factor_squared_rms, float), f2.shape)
            out = out + sd * rng.standard_normal(f2.shape)
        for rel in (self.transverse_rms_fraction, self.response_rms_fraction):
            # The uncertain factor divides the measured spectrum.
            factor = 1.0 + np.broadcast_to(np.asarray(rel, float), f2.shape) * rng.standard_normal()
            out = out / np.maximum(factor, 1e-6)
        charge = 1.0 + self.charge_rms_fraction * rng.standard_normal()
        return out / max(charge, 1e-6) ** 2

    def describe(self) -> dict:
        """JSON-ready summary."""

        def _summary(v):
            a = np.asarray(v, float)
            return float(a) if a.ndim == 0 else [float(a.min()), float(a.max())]

        return {
            "spectral_noise": None
            if self.form_factor_squared_rms is None
            else _summary(self.form_factor_squared_rms),
            "charge_rms_fraction": float(self.charge_rms_fraction),
            "transverse_rms_fraction": _summary(self.transverse_rms_fraction),
            "response_rms_fraction": _summary(self.response_rms_fraction),
            "correlation": "spectral noise independent per node; charge, transverse and response each fully correlated across nodes",
        }


@dataclass(frozen=True, eq=False)
class ReconstructionUncertainty:
    """Monte Carlo spread of the minimal-phase reconstruction.

    The band describes propagation of the declared inputs through the SAME
    minimal-phase map and continuation; it does not cover the reconstruction
    ambiguities (time reversal, zero flipping, missing band), which no
    magnitude uncertainty can bound. The RMS interval uses only draws with a
    defined RMS (``undefined_rms_draws`` counts the rest); ``clipped_draws``
    counts draws floored at a positive value or capped at ``|f| = 1``.
    """

    nominal: MinimalPhaseReconstruction
    profile_mean_per_fs: np.ndarray
    profile_std_per_fs: np.ndarray
    profile_interval_95_per_fs: tuple[np.ndarray, np.ndarray]
    rms_duration_draws_fs: np.ndarray
    rms_duration_interval_95_fs: tuple[float, float]
    undefined_rms_draws: int
    clipped_draws: int
    draws: int
    seed: int
    uncertainty: dict


def propagate_reconstruction_uncertainty(
    energy_eV,
    form_factor_squared,
    time_fs,
    *,
    missing_band: MissingBandModel,
    uncertainty: FormFactorUncertainty,
    provenance: str,
    draws: int = 200,
    seed: int = 0,
) -> ReconstructionUncertainty:
    """Propagate charge, transverse, response and noise uncertainty by sampling.

    Validation: optical-phase-retrieval
    """
    if not isinstance(uncertainty, FormFactorUncertainty):
        raise TypeError("uncertainty must be a FormFactorUncertainty")
    if draws < 10:
        raise ValueError("draws must be at least 10")
    nominal = reconstruct_minimal_phase_profile(
        energy_eV, form_factor_squared, time_fs, missing_band=missing_band, provenance=provenance
    )
    e, f2 = _check_band(energy_eV, form_factor_squared)
    t = nominal.time_fs
    rng = np.random.default_rng(seed)
    floor = 1e-12 * float(f2.max())
    profiles = np.empty((draws, t.size))
    rms = np.empty(draws)
    clipped = 0
    for k in range(draws):
        sample = uncertainty.sample(f2, rng)
        bounded = np.clip(sample, floor, 1.0)
        clipped += int(np.any(bounded != sample))
        nodes, log_f, _ = _extend(e, 0.5 * np.log(bounded), missing_band)
        phase = minimal_phase(nodes, log_f, high_power_exponent=missing_band.high_power_exponent)
        profiles[k] = _synthesize(nodes, np.exp(log_f), phase, t)
        rms[k] = _rms(t, profiles[k])
    lo, hi = np.percentile(profiles, [2.5, 97.5], axis=0)
    finite = rms[np.isfinite(rms)]
    interval = (
        (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
        if finite.size
        else (float("nan"), float("nan"))
    )
    return ReconstructionUncertainty(
        nominal=nominal,
        profile_mean_per_fs=profiles.mean(axis=0),
        profile_std_per_fs=profiles.std(axis=0),
        profile_interval_95_per_fs=(lo, hi),
        rms_duration_draws_fs=rms,
        rms_duration_interval_95_fs=interval,
        undefined_rms_draws=int(draws - finite.size),
        clipped_draws=clipped,
        draws=draws,
        seed=seed,
        uncertainty=uncertainty.describe(),
    )
