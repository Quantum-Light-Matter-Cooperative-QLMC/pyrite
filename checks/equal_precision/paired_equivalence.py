"""Finite-sample paired-primary mean bounds for opt-in numerical equivalence."""

import math

import numpy as np
from scipy.stats import beta

SOURCE = "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf"


def mismatch_upper(k, n, delta):
    """One-sided exact Clopper-Pearson upper probability for nonzero differences."""
    if not (0 <= k <= n and n > 0 and 0 < delta < 1):
        raise ValueError("invalid binomial bound inputs")
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(delta) / n)
    return float(beta.isf(delta, k + 1, n - k))


def paired_interval(reference, candidate, *, bound, margin, comparisons, alpha=0.05):
    """Joint-error-budgeted intersection of empirical Bernstein and mismatch bounds.

    Assumes independent incident-primary pairs and fixed, known [0,bound]
    supports. Split alpha/comparisons over two methods; the Bernstein method
    also splits over two tails. Identical samples retain positive uncertainty.
    Numerical method diagnostic; no physical-model validation claim.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("equal one-dimensional primary score populations required")
    if not (
        math.isfinite(bound)
        and bound > 0
        and math.isfinite(margin)
        and margin > 0
        and comparisons >= 1
        and 0 < alpha < 1
    ):
        raise ValueError("invalid support, margin or confidence parameters")
    if not (
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and min(a.min(), b.min()) >= 0
        and max(a.max(), b.max()) <= bound
    ):
        raise ValueError("nonfinite score or known support violated")
    d = b - a
    n = d.size
    mean = float(d.mean())
    variance = float(np.var(d, ddof=1))
    log = math.log(8 * comparisons / alpha)
    radius = math.sqrt(2 * variance * log / n) + 14 * bound * log / (3 * (n - 1))
    eb = [max(-bound, mean - radius), min(bound, mean + radius)]
    nonzero = int(np.count_nonzero(d))
    upper = mismatch_upper(nonzero, n, alpha / (2 * comparisons))
    discordance = [-bound * upper, bound * upper]
    interval = [max(eb[0], discordance[0]), min(eb[1], discordance[1])]
    status = "unresolved"
    if interval[0] >= -margin and interval[1] <= margin:
        status = "passed"
    elif interval[0] > margin or interval[1] < -margin:
        status = "failed"
    reference_se = float(np.std(a, ddof=1) / math.sqrt(n))
    candidate_se = float(np.std(b, ddof=1) / math.sqrt(n))
    return dict(
        reference_standard_error=reference_se,
        candidate_standard_error=candidate_se,
        reference_rse=reference_se / float(a.mean()) if a.mean() > 0 else None,
        candidate_rse=candidate_se / float(b.mean()) if b.mean() > 0 else None,
        status=status,
        n=n,
        reference_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
        difference=mean,
        difference_ci=interval,
        margin=margin,
        known_score_bound=bound,
        nonzero_pairs=nonzero,
        mismatch_probability_upper=upper,
        paired_standard_error=math.sqrt(variance / n),
        empirical_bernstein_ci=eb,
        mismatch_ci=discordance,
        comparisons=comparisons,
        joint_confidence=1 - alpha,
        method="Two-method Bonferroni intersection; Maurer-Pontil Thm4 two-sided and exact binomial mismatch upper bound.",
        source=SOURCE,
    )


def coupling_record(seed, histories):
    """Fingerprint ordered counter streams outside measured transport timing."""
    import hashlib
    from pathlib import Path

    from pyrite.montecarlo.transport import kinematics
    from pyrite.montecarlo.transport._jit_radiative import radiative_stream_keys

    owner = Path(kinematics.__file__).parent
    sources = {
        name: hashlib.sha256((owner / name).read_text(encoding="utf-8").encode()).hexdigest()
        for name in [
            "kinematics.py",
            "api.py",
            "_jit_radiative.py",
            "hard_radiative.py",
            "_jit_kernel.py",
            "_jit_device.py",
            "batching.py",
        ]
    }
    sources["pyrite_adapter.py"] = hashlib.sha256(
        Path(__file__).with_name("pyrite_adapter.py").read_text(encoding="utf-8").encode()
    ).hexdigest()
    return dict(
        scheme="pyrite.counter-primary.v1",
        seed=int(seed),
        histories=int(histories),
        ordered_primary_range=[0, int(histories)],
        score_index_equals_global_primary=True,
        transport_keys_sha256=hashlib.sha256(
            kinematics.stream_keys(seed, histories).tobytes()
        ).hexdigest(),
        radiation_keys_sha256=hashlib.sha256(
            radiative_stream_keys(seed, histories).tobytes()
        ).hexdigest(),
        sources=sources,
    )


def validate_coupling(reference, candidate):
    """Require audited same-primary streams/order, not merely matching seed labels."""
    first, second = reference.get("primary_coupling"), candidate.get("primary_coupling")
    if not first or first != second:
        raise ValueError("primary coupling provenance mismatch")
    if first.get("scheme") != "pyrite.counter-primary.v1" or not first.get(
        "score_index_equals_global_primary"
    ):
        raise ValueError("unknown primary coupling scheme/order")
    if reference["seed"] != candidate["seed"] or reference["histories"] != candidate["histories"]:
        raise ValueError("paired seed/population mismatch")
    expected = coupling_record(reference["seed"], reference["histories"])
    for field in [
        "seed",
        "histories",
        "ordered_primary_range",
        "transport_keys_sha256",
        "radiation_keys_sha256",
    ]:
        if first.get(field) != expected[field]:
            raise ValueError("primary stream fingerprint mismatch")

    if first.get("sources") != expected["sources"]:
        raise ValueError("primary coupling source mismatch")
