"""Primary-history moments for opt-in cross-engine performance studies."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def primary_moments(values: ArrayLike) -> dict[str, int | float | str | None]:
    """Return raw moments and sample-mean SE, including zero-score primaries.

    ``values`` contains one finite score per independent incident primary.
    A zero sample variance is reported but never qualifies a precision target.
    For zero nonzero scores, give the exact one-sided 95% upper bound on the
    probability that an independent primary scores nonzero, not on its yield.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("need at least two finite independent primary scores")
    n = int(values.size)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n))
    return {
        "n": n,
        "sum": float(values.sum()),
        "sum_squares": float(np.dot(values, values)),
        "mean": mean,
        "se": se,
        "nonzero": int(np.count_nonzero(values)),
        "rse": se / abs(mean) if mean != 0 and se > 0 else None,
        "precision_status": "estimable" if mean != 0 and se > 0 else "unresolved",
        "nonzero_probability_upper95_if_zero": (
            -math.expm1(math.log(0.05) / n) if np.count_nonzero(values) == 0 else None
        ),
    }


def grouped_scores(
    primary_ids: ArrayLike, scores: ArrayLike, n_primaries: int
) -> NDArray[np.float64]:
    """Sum correlated emissions/segments within primaries, preserving misses."""
    ids = np.asarray(primary_ids)
    scores = np.asarray(scores, dtype=np.float64)
    if (
        n_primaries < 1
        or ids.ndim != 1
        or scores.shape != ids.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or np.any(ids >= n_primaries)
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("invalid primary ids, scores or incident population")
    return np.bincount(ids, weights=scores, minlength=n_primaries)
