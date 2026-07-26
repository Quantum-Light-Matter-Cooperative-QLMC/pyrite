"""External-background fitting and comparison for validation studies.

Production simulation remains in :mod:`cxr_mc.montecarlo`.  This module owns
analysis-only operations used to compare a simulated or measured spectrum with
an externally generated, already detector-normalized bremsstrahlung spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .montecarlo import load_external_brem


@dataclass(frozen=True)
class BackgroundFit:
    """Weighted scale-only fit of an external background to an experiment."""

    scale: float
    scale_std: float
    reduced_chi2: float
    n_points: int


@dataclass(frozen=True)
class BackgroundComparison:
    """Shape and absolute-normalization comparison between two backgrounds."""

    integrated_ratio: float
    normalized_rmse: float
    correlation: float
    n_points: int


def fit_external_background(
    energy_eV: ArrayLike,
    intensity: ArrayLike,
    external_path: str | PathLike[str],
    *,
    sigma: ArrayLike | None = None,
    fit_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], BackgroundFit]:
    r"""Fit one non-negative scale factor for an external background.

    The external spectrum is ingested by :func:`load_external_brem`, preserving
    its detected-unit contract.  For selected sideband samples, this minimizes

    .. math:: \chi^2(a)=\sum_i[(y_i-a b_i)/\sigma_i]^2

    and therefore uses
    :math:`a=\sum_i w_i b_i y_i/\sum_i w_i b_i^2`.  With exact data
    ``y = a*b`` the residual vanishes; with unit uncertainties this reduces to
    ordinary least squares through the origin.

    ``fit_mask`` must exclude coherent and characteristic peaks.  This
    scale-only sideband fit is cxr-mc's documented analysis method; Zhai et al.
    state that DTSA-II plus a numerical PIXE method was used, but do not publish
    enough algorithmic detail to claim an exact reconstruction.

    Validation: external-brem-subtraction
    """
    energy = np.asarray(energy_eV, dtype=float)
    observed = np.asarray(intensity, dtype=float)
    if energy.ndim != 1 or observed.shape != energy.shape:
        raise ValueError("energy_eV and intensity must be matching 1-D arrays")

    background = np.asarray(load_external_brem(external_path, energy), dtype=float)
    uncertainty = np.ones_like(observed) if sigma is None else np.asarray(sigma, dtype=float)
    if uncertainty.shape != observed.shape:
        raise ValueError("sigma must match intensity")
    selected = (
        np.ones(observed.shape, dtype=bool)
        if fit_mask is None
        else np.asarray(fit_mask, dtype=bool)
    )
    if selected.shape != observed.shape:
        raise ValueError("fit_mask must match intensity")
    selected &= (
        np.isfinite(energy)
        & np.isfinite(observed)
        & np.isfinite(background)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
        & (background > 0.0)
    )
    n_points = int(np.count_nonzero(selected))
    if n_points < 2:
        raise ValueError("background fit needs at least two finite selected points")

    b = background[selected]
    y = observed[selected]
    w = 1.0 / np.square(uncertainty[selected])
    denominator = float(np.dot(w, np.square(b)))
    scale = float(np.dot(w * b, y) / denominator)
    if scale < 0.0:
        raise ValueError("best-fit external-background scale is negative")
    residual = y - scale * b
    chi2 = float(np.dot(w, np.square(residual)))
    return background, BackgroundFit(
        scale=scale,
        scale_std=float(np.sqrt(1.0 / denominator)),
        reduced_chi2=chi2 / (n_points - 1),
        n_points=n_points,
    )


def subtract_external_background(
    energy_eV: ArrayLike,
    intensity: ArrayLike,
    external_path: str | PathLike[str],
    *,
    sigma: ArrayLike | None = None,
    fit_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], BackgroundFit]:
    """Fit then subtract an external background on the experimental grid."""
    observed = np.asarray(intensity, dtype=float)
    background, fit = fit_external_background(
        energy_eV,
        observed,
        external_path,
        sigma=sigma,
        fit_mask=fit_mask,
    )
    fitted = fit.scale * background
    return observed - fitted, fitted, fit


def compare_external_background(
    energy_eV: ArrayLike,
    model_intensity: ArrayLike,
    external_path: str | PathLike[str],
    *,
    comparison_mask: ArrayLike | None = None,
) -> BackgroundComparison:
    """Compare cxr-mc and external backgrounds without rescaling either one."""
    energy = np.asarray(energy_eV, dtype=float)
    model = np.asarray(model_intensity, dtype=float)
    if energy.ndim != 1 or model.shape != energy.shape:
        raise ValueError("energy_eV and model_intensity must be matching 1-D arrays")
    external = np.asarray(load_external_brem(external_path, energy), dtype=float)
    selected = (
        np.ones(model.shape, dtype=bool)
        if comparison_mask is None
        else np.asarray(comparison_mask, dtype=bool)
    )
    if selected.shape != model.shape:
        raise ValueError("comparison_mask must match model_intensity")
    selected &= np.isfinite(model) & np.isfinite(external) & (external > 0.0)
    n_points = int(np.count_nonzero(selected))
    if n_points < 2:
        raise ValueError("background comparison needs at least two overlapping points")
    x = energy[selected]
    m = model[selected]
    e = external[selected]
    external_area = float(np.trapezoid(e, x))
    correlation = (
        float(np.corrcoef(m, e)[0, 1]) if np.std(m) > 0.0 and np.std(e) > 0.0 else float("nan")
    )
    return BackgroundComparison(
        integrated_ratio=float(np.trapezoid(m, x) / external_area),
        normalized_rmse=float(np.sqrt(np.mean(np.square(m - e))) / np.mean(e)),
        correlation=correlation,
        n_points=n_points,
    )
