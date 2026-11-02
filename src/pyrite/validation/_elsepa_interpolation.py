"""Deterministic envelopes for the proposed ELSEPA energy-node mixture.

Source: the node law of ``elsepa-elastic-sampling`` and the square expansion
derived in docs/validation/beam-transport/elsepa-energy-mixture-moments.md.
Assumptions: normalized, nonnegative piecewise-linear densities on [0, 1];
positive panel masses. All quantities are dimensionless. Identical nodes and
either endpoint weight give zero moment difference.

Validation: elsepa-energy-mixture-moments
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normalized_row(mu: NDArray[np.float64], dcs: ArrayLike):
    values = np.asarray(dcs, dtype=np.float64)
    if values.shape != mu.shape or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("DCS must be finite, nonnegative and match the angular grid")
    masses = np.diff(mu) * (values[:-1] + values[1:]) / 2
    norm = masses.sum()
    if not np.isfinite(norm) or norm <= 0 or np.any(masses <= 0):
        raise ValueError("each angular panel must have positive probability")
    pdf = values / norm
    cdf = np.r_[0.0, np.cumsum(masses) / norm]
    cdf[-1] = 1.0
    return pdf, cdf


def _moments(mu, pdf):
    """Exact local polynomial integrals, avoiding differences of close powers."""
    x, h = mu[:-1], np.diff(mu)
    p, q = pdf[:-1], pdf[1:]
    mass = (p + q) / 2
    first = (h * (x * mass + h * (p + 2 * q) / 6)).sum()
    second = (h * (x * x * mass + x * h * (p + 2 * q) / 3 + h * h * (p + 3 * q) / 12)).sum()
    return float(first), float(second)


def _quantile(mu, pdf, cdf, u):
    """Vectorized inverse used only by deterministic validation, never transport."""
    panel = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, mu.size - 2)
    h = mu[panel + 1] - mu[panel]
    p = pdf[panel]
    slope = (pdf[panel + 1] - p) / h
    mass = np.maximum(u - cdf[panel], 0)
    root = np.sqrt(np.maximum(p * p + 2 * slope * mass, 0))
    denominator = p + root
    offset = np.divide(2 * mass, denominator, out=np.zeros_like(mass), where=denominator > 0)
    return mu[panel] + np.clip(offset, 0, h)


def _difference_envelopes(mu, rows, cdfs, left, right):
    """Extrema of Q1-Q0 on cells contained in one panel of each node.

    On such a panel p(Q(u))**2 = p_j**2 + 2*slope*(u-F_j).
    The derivative of Q1-Q0 vanishes only when these two affine functions
    coincide. Flat derivative differences need no interior candidate.
    Validation: elsepa-energy-mixture-moments
    """
    middle = left + (right - left) / 2
    q_left, q_right, slopes, squares = [], [], [], []
    for pdf, cdf in zip(rows, cdfs, strict=True):
        panel = np.clip(np.searchsorted(cdf, middle, side="right") - 1, 0, mu.size - 2)
        slope = (pdf[panel + 1] - pdf[panel]) / (mu[panel + 1] - mu[panel])
        slopes.append(slope)
        squares.append(pdf[panel] ** 2 + 2 * slope * (left - cdf[panel]))
        q_left.append(_quantile(mu, pdf, cdf, left))
        q_right.append(_quantile(mu, pdf, cdf, right))
    dl, dr = q_left[1] - q_left[0], q_right[1] - q_right[0]
    difference = 2 * (slopes[1] - slopes[0])
    offset = np.divide(
        squares[0] - squares[1],
        difference,
        out=np.full_like(left, np.inf),
        where=difference != 0,
    )
    inside = (offset > 0) & (offset < right - left)
    candidate = left + np.where(inside, offset, 0)
    ds = _quantile(mu, rows[1], cdfs[1], candidate) - _quantile(mu, rows[0], cdfs[0], candidate)
    low = np.minimum(np.minimum(dl, dr), ds)
    high = np.maximum(np.maximum(dl, dr), ds)
    lower_square = np.where(low * high <= 0, 0, np.minimum(low * low, high * high))
    upper_square = np.maximum(low * low, high * high)
    return lower_square, upper_square


def _pair_moment_bounds(mu: ArrayLike, dcs0: ArrayLike, dcs1: ArrayLike, *, subdivisions: int = 16):
    """Enclose the two laws' moment gap for every log-energy weight f in [0,1].

    For independent uniform node selection and quantile uniforms, the first
    moment of Y=1-cos(theta) agrees exactly. The second-moment excess of the
    mixture is 4*f*(1-f)*D, D=int (Q1-Q0)**2 du. This routine bounds D using
    exact cell extrema, not a sampled maximum or quadrature error estimate.
    The mathematical envelopes are evaluated in FP64, with a 64-epsilon
    absolute padding; these are not directed-rounding interval certificates.

    Source/assumptions: module docstring. Limit: identical nodes give D=0.
    Validation: elsepa-energy-mixture-moments
    """
    mu = np.asarray(mu, dtype=np.float64)
    if (
        mu.ndim != 1
        or mu.size < 2
        or mu[0] != 0
        or mu[-1] != 1
        or np.any(~np.isfinite(mu))
        or np.any(np.diff(mu) <= 0)
    ):
        raise ValueError("mu must increase strictly from zero to one")
    if not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    rows, cdfs = zip(_normalized_row(mu, dcs0), _normalized_row(mu, dcs1), strict=True)
    moments = np.array([_moments(mu, pdf) for pdf in rows])
    breaks = np.unique(np.r_[cdfs[0], cdfs[1]])
    # Subdivide each common-CDF panel; no cell straddles a density knot.
    edges = (
        breaks[:-1, None] + np.diff(breaks)[:, None] * np.arange(subdivisions + 1) / subdivisions
    )
    left, right = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    lower, upper = _difference_envelopes(mu, rows, cdfs, left, right)
    padding = 64 * np.finfo(np.float64).eps
    d_lower = max(0.0, float(np.sum((right - left) * lower)) - padding)
    d_upper = float(np.sum((right - left) * upper)) + padding
    if np.array_equal(rows[0], rows[1]):
        d_lower = d_upper = 0.0
    a, b = moments[:, 1]
    # Max_f Delta M2/M2_quantile = D / ((sqrt(a)+sqrt(b))**2-D).
    denominator = (np.sqrt(a) + np.sqrt(b)) ** 2 - d_upper
    relative_upper = d_upper / denominator if denominator > 0 else float("inf")
    relative_lower = d_lower / ((np.sqrt(a) + np.sqrt(b)) ** 2 - d_lower)
    return {
        "first_moment_difference": 0.0,
        "node_first_moments": (2 * moments[:, 0]).tolist(),
        "node_second_moments": (4 * moments[:, 1]).tolist(),
        "max_second_moment_difference_lower": d_lower,
        "max_second_moment_difference_upper": d_upper,
        "max_relative_second_moment_difference_upper": float(relative_upper),
        "max_relative_second_moment_difference_lower": float(relative_lower),
        "absolute_max_weight": 0.5,
        "relative_max_weight": float(np.sqrt(a) / (np.sqrt(a) + np.sqrt(b))),
        "cells": int(left.size),
        "fp64_absolute_padding": padding,
    }
