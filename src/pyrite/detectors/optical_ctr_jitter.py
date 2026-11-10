"""Shot-to-shot bunch-charge jitter in the Gaussian CTR scan likelihood (#617).

One scan value is the mean of ``shots_per_point`` isolated-pulse voltages. A
shot with population ``N = q * Nbar`` (``<q> = 1``, ``Var q = r^2``) gives,
before readout noise, the two port voltages

    V_p = a_p * q + c_p * N (N - 1) / (Nbar (Nbar - 1))
        = alpha_p * q + beta_p * q^2,
    alpha_p = a_p - c_p / (Nbar - 1),   beta_p = c_p * Nbar / (Nbar - 1),

where ``a_p`` and ``c_p`` are the incoherent and pair voltages of the mean
bunch. The same shots feed both ports, so the jitter is fully correlated
between ports at one mirror position and independent between positions.

``normalization="none"`` (raw shot average)
    mean  ``a_p + c_p * (1 + Nbar r^2 / (Nbar - 1))``  (exact, any law);
    covariance  ``[alpha alpha' r^2 + (alpha beta' + beta alpha') 2 r^2
    + beta beta' (4 r^2 + 2 r^4)] / shots``, which uses the Gaussian third
    and fourth central moments of ``q`` (``mu_3 = 0``, ``mu_4 = 3 r^4``).

``normalization="per-shot-charge"`` (each shot divided by its measured ``q``)
    ``V_p / q = alpha_p + beta_p * q``, so mean ``a_p + c_p`` and covariance
    ``beta beta' r^2 / shots``. Both are exact for any charge law with
    ``q > 0``; no pair-factor bias remains.

Assumptions: charge uncorrelated with bunch duration, transverse shape and
arrival time; baselines subtracted before normalization; the supplied readout
uncertainty is the independent per-port RMS of the reported (averaged,
normalized) value; the charge measurement itself is exact. The covariance
depends on the fitted parameters, so the fit minimizes the Gaussian
``-2 ln L = sum_i [d_i^T C_i^-1 d_i + ln(det C_i / det C_i0)]`` with ``C_i0``
the readout-only covariance; it reduces to chi-square at ``r = 0``.

Validation: optical-instrument-observable
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["ChargeJitterModel"]

_NORMALIZATIONS = ("none", "per-shot-charge")


@dataclass(frozen=True)
class ChargeJitterModel:
    """Declared shot-to-shot population jitter of an averaged voltage scan.

    ``mean_electrons`` is the population ``Nbar`` of the bunch the templates
    describe, ``charge_rms_fraction`` its relative RMS ``r`` (must leave
    ``q > 0``: ``r < 1``, and the Gaussian moments of the raw mode need
    ``r << 1``), ``shots_per_point`` the number of independent shots averaged
    into each scan value, identical for every mirror position.

    Validation: optical-instrument-observable
    """

    mean_electrons: float
    charge_rms_fraction: float
    shots_per_point: int
    normalization: str = "none"

    def __post_init__(self):
        if not np.isfinite(self.mean_electrons) or self.mean_electrons <= 1:
            raise ValueError("mean_electrons must be finite and > 1")
        r = self.charge_rms_fraction
        if not np.isfinite(r) or not 0 <= r < 1:
            raise ValueError("charge_rms_fraction must lie in [0, 1)")
        n = self.shots_per_point
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("shots_per_point must be a positive integer")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_NORMALIZATIONS}")

    @property
    def pair_mean_factor(self) -> float:
        """Multiplier of the pair voltage in the expected scan value."""
        if self.normalization == "per-shot-charge":
            return 1.0
        n, r = float(self.mean_electrons), float(self.charge_rms_fraction)
        return 1.0 + n * r * r / (n - 1.0)

    def port_covariance(self, incoherent_v, coherent_v) -> np.ndarray:
        """Jitter covariance [V^2] of the two ports, shape (n_delay, 2, 2).

        ``incoherent_v`` and ``coherent_v`` are the mean-bunch voltages
        ``a`` and ``c`` with shape (n_delay, 2). Readout noise is not included.
        """
        a = np.asarray(incoherent_v, dtype=float)
        c = np.asarray(coherent_v, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or c.shape != a.shape:
            raise ValueError("port voltages must have shape (n_delay, 2)")
        n, r2 = float(self.mean_electrons), float(self.charge_rms_fraction) ** 2
        alpha = a - c / (n - 1.0)
        beta = c * n / (n - 1.0)

        def outer(u, v):
            return u[:, :, None] * v[:, None, :]

        if self.normalization == "per-shot-charge":
            cov = outer(beta, beta) * r2
        else:
            cov = (
                outer(alpha, alpha) * r2
                + (outer(alpha, beta) + outer(beta, alpha)) * 2 * r2
                + outer(beta, beta) * (4 * r2 + 2 * r2 * r2)
            )
        return cov / float(self.shots_per_point)

    def describe(self) -> dict:
        return {
            "mean_electrons": float(self.mean_electrons),
            "charge_rms_fraction": float(self.charge_rms_fraction),
            "shots_per_point": int(self.shots_per_point),
            "normalization": self.normalization,
            "pair_mean_factor": self.pair_mean_factor,
            "moments": (
                "exact first/second moments of q"
                if self.normalization == "per-shot-charge"
                else "Gaussian third/fourth central moments of q"
            ),
        }


def _evaluate(scale, incoherent, coherent, scan, jitter, fit_offsets):
    """Quadratic form, log-determinant excess, offsets and prediction."""
    y, readout = scan.voltage_v_ports, scan.uncertainty_v_ports
    model = scale * (incoherent + jitter.pair_mean_factor * coherent)
    cov = jitter.port_covariance(scale * incoherent, scale * coherent)
    cov[:, 0, 0] += readout[:, 0] ** 2
    cov[:, 1, 1] += readout[:, 1] ** 2
    weight = np.linalg.inv(cov)
    delta = y - model
    offsets = np.zeros(2)
    if fit_offsets:
        # Constant port offsets enter linearly: generalized least squares.
        offsets = np.linalg.solve(weight.sum(axis=0), np.einsum("iab,ib->a", weight, delta))
        delta = delta - offsets[None, :]
    quadratic = float(np.einsum("ia,iab,ib->", delta, weight, delta))
    log_det = float(np.sum(np.log(np.linalg.det(cov)) - 2 * np.log(readout[:, 0] * readout[:, 1])))
    return quadratic, log_det, offsets, model + offsets[None, :]


def jitter_profile_at_sigma(
    incoherent,
    coherent,
    scan,
    jitter: ChargeJitterModel,
    *,
    fit_scale: bool,
    fit_offsets: bool,
    scale_prior_rms_fraction: float | None,
    scale_guess: float,
):
    """Profile ``-2 ln L`` over the common scale and port offsets.

    ``incoherent``/``coherent`` are the mean-bunch port voltages (n_delay, 2)
    at the trial duration and unit scale. Offsets are solved exactly for each
    scale; the scale is a bounded one-dimensional search within ``e^{+-3}`` of
    ``scale_guess`` because the covariance depends on it. Returns
    ``(objective, scale, offsets, predicted, quadratic_form)``.
    """

    def objective(scale):
        quadratic, log_det, offsets, predicted = _evaluate(
            scale, incoherent, coherent, scan, jitter, fit_offsets
        )
        value = quadratic + log_det
        if scale_prior_rms_fraction is not None:
            value += ((scale - 1) / scale_prior_rms_fraction) ** 2
        return value, offsets, predicted, quadratic

    scale = 1.0
    if fit_scale:
        center = np.log(scale_guess if scale_guess > 0 else 1.0)
        solution = minimize_scalar(
            lambda z: objective(float(np.exp(z)))[0],
            bounds=(center - 3.0, center + 3.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        scale = float(np.exp(solution.x))
    value, offsets, predicted, quadratic = objective(scale)
    return value, scale, (float(offsets[0]), float(offsets[1])), predicted, quadratic
