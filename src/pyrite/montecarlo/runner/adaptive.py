"""Adaptive electron counts: sequential stopping on a target error (#361, #1253).

``Numerics(sampling=AdaptiveSampling(...))`` selects this path through the public runner.
:func:`run_case_adaptive` retains the internal compatibility interface. Both
transport electron chunks through :mod:`.block_transport`, sized for the device
by :class:`.stream_sizing.AdaptiveChunkPlan`, and buffer grid-independent
per-electron scalars:

* ``"line"`` -- the per-electron line mass ``sum w pi / a_w`` over the lines a
  fixed monitor band keeps, the quantity of the #201 line-yield audit;
* ``"brem"`` -- the per-electron bremsstrahlung integral over the same band.

Neither depends on the line grid, which is resolved only once transport has
stopped, from every realized electron, before the single spectrum reduction.
The rule is checked at geometric decision points ``d_{j+1} = ceil(d_j (1 +
decision_growth))`` from ``min_electrons`` to ``max_electrons``; the values of
each segment between points are folded into the moments at once, so decisions
do not depend on the chunks. The stop is the first point ``n`` at which, for
every chosen observable, the relative standard error of the mean is at most
``target_rse`` and the heavy-tail guards hold (largest single-electron share,
effective sample size, CXR variance sampling, stability of the running mean
over the last decision points); at ``max_electrons`` the case completes
flagged ``statistics_limited``. A stop inside a chunk keeps the prefix by
transporting it again, so the realized case is the fixed-N case at the
realized count, bit for bit: the stop changes how many electrons run, never
how they run.

Estimator, rule, guards and their limiting cases are derived in
``docs/validation/beam-transport/adaptive-sample-size-stopping.md``; the user
description is ``docs/computation/statistical-methods.md``.

Validation: adaptive-sample-size-stopping
"""

import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction
from itertools import chain
from typing import Any

import numpy as np

from ..._backend import _to_cpu
from ..._cxr_grid_policy import CxrYieldStatisticsWarning
from ..._precision import ADAPTIVE_SAMPLING_MODEL, Precision
from ..._precision import OBSERVABLES as OBSERVABLES
from ..spectrum.cxr._kernels import _SEG_ARRAYS
from .adaptive_report import limit_summary
from .artifacts import _case_geometry
from .electron_blocks import _take
from .stream_sizing import AdaptiveChunkPlan

#: ``measure(start, stop, block) -> {observable: per-electron values}``.
Measure = Callable[[int, int, Mapping[str, Any]], Mapping[str, Any]]


# Internal compatibility name; the public policy lives below the domain packages.
AdaptiveSettings = Precision


class RunningMoments:
    """Exact running moments of one non-negative per-electron observable.

    Each block is reduced where its values live (device or host) to its count,
    sum, centred moments through fourth order, sum of squares and maximum;
    blocks merge by
    Chan et al.'s pairwise update, so the mean and variance equal a one-pass
    Welford over the concatenated values up to rounding.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.m4 = 0.0
        self.total = 0.0
        self.total_sq = 0.0
        self.largest = 0.0

    def add_block(self, values) -> None:
        if values.size == 0:
            return
        values = values.astype("float64")
        nb = int(values.size)
        block_sum = float(values.sum())
        block_mean = block_sum / nb
        centered = values - block_mean
        squared = centered * centered
        block_m2 = float(squared.sum())
        block_m3 = float((squared * centered).sum())
        block_m4 = float((squared * squared).sum())
        block_sq = float((values * values).sum())
        block_max = float(values.max())
        n = self.n + nb
        delta = block_mean - self.mean
        na = self.n
        # Expand (x - combined_mean)^p in each block, p = 3, 4.
        # Use the old M2/M3 throughout; updating either first corrupts M4.
        self.m4 += (
            block_m4
            + delta**4 * na * nb * (na * na - na * nb + nb * nb) / n**3
            + 6 * delta**2 * (na * na * block_m2 + nb * nb * self.m2) / n**2
            + 4 * delta * (na * block_m3 - nb * self.m3) / n
        )
        self.m3 += (
            block_m3
            + delta**3 * na * nb * (na - nb) / n**2
            + 3 * delta * (na * block_m2 - nb * self.m2) / n
        )
        self.mean += delta * nb / n
        self.m2 += block_m2 + delta * delta * self.n * nb / n
        self.n = n
        self.total += block_sum
        self.total_sq += block_sq
        self.largest = max(self.largest, block_max)

    def relative_se(self) -> float | None:
        """``s / (sqrt(n) |mean|)``; ``None`` below two samples or at zero mean."""
        if self.n < 2 or self.mean == 0.0:
            return None
        return math.sqrt(self.m2 / (self.n - 1) / self.n) / abs(self.mean)

    def max_share(self) -> float | None:
        return self.largest / self.total if self.total > 0.0 else None

    def effective_sample_size(self) -> float:
        return self.total**2 / self.total_sq if self.total_sq > 0.0 else 0.0

    def variance_of_variance(self) -> float | None:
        """Geant4/MCNP diagnostic ``M4/M2**2 - 1/n`` (dimensionless).

        For independent histories with finite fourth moment, its square root
        is an asymptotic plug-in estimate of relative variance uncertainty.
        A constant signal returns
        zero; fewer than two samples has no estimate. This cannot expose an
        entirely unsampled population.

        Validation: adaptive-sample-size-stopping
        """
        if self.n < 2:
            return None
        if self.m2 == 0.0:
            return 0.0
        return max(0.0, (math.sqrt(self.m4) / self.m2) ** 2 - 1.0 / self.n)


def vov_trend_slope(points) -> float | None:
    """Least-squares slope of ``ln VOV`` against ``ln n`` over the last half.

    ``points`` are ``(n, VOV)`` at successive decision points. Only points with
    ``n >= n_last / 2`` and ``VOV > 0`` enter; fewer than three such
    points, or a single distinct ``n``, gives ``None``. A population whose
    fourth moment the run has sampled has ``VOV ~ (kappa - 1)/n``, so the slope
    tends to -1 (MCNP checks VOV falls as ``1/N``); a heavy score drawn late
    makes it shallower or positive. Reported only; it never gates a stop.

    Validation: adaptive-sample-size-stopping
    """
    if not points:
        return None
    half = points[-1][0] / 2.0
    kept = [(math.log(n), math.log(v)) for n, v in points if n >= half and v is not None and v > 0]
    if len(kept) < 3:
        return None
    mx = sum(x for x, _ in kept) / len(kept)
    my = sum(y for _, y in kept) / len(kept)
    sxx = sum((x - mx) ** 2 for x, _ in kept)
    if sxx == 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in kept) / sxx


def _excess(move: float, limit: float) -> float:
    """``move / limit``, with a zero limit (``stability_fraction = 0`` and no
    drift allowance) admitting only a zero move."""
    if move == 0.0:
        return 0.0
    return math.inf if limit == 0.0 else move / limit


def decision_points(minimum: int, maximum: int, growth: float) -> Iterator[int]:
    """Electron counts at which the stopping rule is checked: a geometric grid.

    ``d_0 = minimum`` and ``d_{j+1} = min(maximum, max(d_j + 1, ceil(d_j (1 + growth))))``,
    ending at ``maximum``. The monitor adds the pilot count and the pilot's
    projection as points of their own. The grid depends on the policy only, never on how
    electrons are transported, and holds ``O(log(maximum / minimum) / growth)``
    points.

    Validation: adaptive-sample-size-stopping
    """
    # ``growth`` as the decimal it was written as (0.1 is exactly 1/10, not
    # the binary double), so the ceiling is exact integer arithmetic.
    ratio = Fraction(repr(float(growth)))
    num, den = ratio.numerator, ratio.denominator
    point = int(minimum)
    while point < maximum:
        yield point
        point = min(int(maximum), point + max(1, -(-point * num // den)))
    yield int(maximum)


def _with_point(points: Iterator[int], extra: int) -> Iterator[int]:
    """The increasing stream ``points`` with ``extra`` merged in once."""
    added = False
    for point in points:
        if not added and extra <= point:
            added = True
            if extra < point:
                yield extra
        yield point


#: Standard deviations of i.i.d. running-mean drift per ``sqrt(decision_growth)``
#: the stability guard admits. Below one, a single electron's jump (about
#: ``r_n`` over one decision segment) stays refused at the point it arrives.
STABILITY_DRIFT = 0.9


class StoppingMonitor:
    """The stopping rule at policy decision points, fed by transport chunks.

    ``on_block(start, stop, block)`` measures one transport chunk and buffers
    its per-electron values; chunks are contiguous from zero, and a chunk at
    ``start == 0`` resets the state, so a restarted transport (an OOM retry)
    is measured once. ``should_stop(n)`` then visits every decision point
    ``d <= n`` not yet visited: it folds the values ``[d_prev, d)`` into the
    moments as one segment and applies the rule at ``d``. A stop sets
    ``realized = d``, which may lie inside the last chunk; the driver keeps
    only the prefix ``[0, d)``.

    Folding one decision segment at a time makes the moments, and so every
    decision, a function of the per-electron values and the policy alone:
    the same seed and settings give the same realized count under any chunk
    sizes.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self, settings: AdaptiveSettings, measure: Measure):
        self.settings = settings
        self.measure = measure
        self.replayed_transport = False
        self._reset()

    def _reset(self):
        s = self.settings
        self.moments = {name: RunningMoments() for name in s.observables}
        self.history: list[tuple[int, dict[str, float]]] = []
        self.vov_history: dict[str, list[tuple[int, float | None]]] = {
            name: [] for name in self.moments
        }
        self.checks = 0
        self.pilot: dict[str, Any] | None = None
        self.projected: int | None = None
        self.stop_reason: str | None = None
        self.realized: int | None = None
        self._points = decision_points(s.min_histories, s.max_histories, s.decision_growth)
        if s.pilot_histories is not None and s.pilot_histories < s.max_histories:
            self._points = _with_point(self._points, s.pilot_histories)
        self._next: int | None = next(self._points)
        self._values: dict[str, list[np.ndarray]] = {name: [] for name in s.observables}
        self._measured = 0
        #: A count inside the last chunk past which its transport is invalid
        #: (set only by the independent brem freeze); ``None`` otherwise.
        self.cut_at: int | None = None

    @property
    def measured(self) -> int:
        """Electrons whose values are buffered or folded."""
        return self._measured

    def on_block(self, start: int, stop: int, block) -> None:
        if start == 0:
            self._reset()
        if start != self._measured:
            raise ValueError(f"chunk starts at {start}, expected {self._measured}")
        values = self.measure(start, stop, block)
        for name in self.moments:
            value = values[name]
            if value.ndim != 1 or value.size != stop - start:
                raise ValueError(f"{name} must contain one value per electron in the block")
            self._values[name].append(np.asarray(_to_cpu(value), dtype=np.float64))
        self._measured = stop

    def truncate(self, n: int) -> None:
        """Forget buffered values of electrons ``>= n``; ``n`` must not precede a fold."""
        folded = self.moments[next(iter(self.moments))].n
        if not folded <= n <= self._measured:
            raise ValueError(f"cannot truncate to {n}: {folded} folded, {self._measured} measured")
        keep = n - folded
        for name, parts in self._values.items():
            joined = np.concatenate(parts) if parts else np.empty(0)
            self._values[name] = [joined[:keep]]
        self._measured = n

    def _fold(self, point: int) -> None:
        """Fold buffered values ``[n, point)`` as one segment; record the means at ``point``."""
        for name, moments in self.moments.items():
            parts = self._values[name]
            joined = np.concatenate(parts) if len(parts) > 1 else parts[0]
            take = point - moments.n
            moments.add_block(joined[:take])
            self._values[name] = [joined[take:]]
        self.history.append((point, {name: m.mean for name, m in self.moments.items()}))
        for name, m in self.moments.items():
            self.vov_history[name].append((point, m.variance_of_variance()))

    def pending(self, n: int) -> Iterator[int]:
        """Fold and yield each unvisited decision point ``<= min(n, measured)``."""
        limit = min(n, self._measured)
        while self._next is not None and self._next <= limit:
            point = self._next
            self._next = next(self._points, None)
            self._fold(point)
            yield point

    def _stability(self, name: str) -> tuple[float, float] | None:
        """Largest move of the running mean over the last k decision points, and its limit.

        Returns ``(move, limit)`` at the binding point: the move
        ``|mean(d_j) - mean(n)| / |mean(n)|`` and the tolerance
        ``max(f eps, z r_n sqrt((n / d_j - 1) / delta))`` with ``z = 0.9``; the
        second term admits the i.i.d. drift of a running mean over a fixed
        fraction of the sample. ``None`` before ``k + 1`` points or at zero
        mean; ``(0, f eps)`` when ``k = 0``.

        Validation: adaptive-sample-size-stopping
        """
        s = self.settings
        k = s.stability_checks
        floor = s.stability_fraction * s.target_rse
        if k == 0:
            return 0.0, floor
        if len(self.history) < k + 1:
            return None
        n, latest = self.history[-1]
        current = latest[name]
        if current == 0.0:
            return None
        rse = self.moments[name].relative_se() or 0.0
        worst = (0.0, floor)
        for point, means in self.history[-k - 1 : -1]:
            move = abs(means[name] - current) / abs(current)
            drift = STABILITY_DRIFT * rse * math.sqrt((n / point - 1.0) / s.decision_growth)
            limit = max(floor, drift)
            if _excess(move, limit) > _excess(*worst):
                worst = (move, limit)
        return worst

    def observable_status(self, name: str) -> dict[str, Any]:
        s = self.settings
        m = self.moments[name]
        rse, share, ess = m.relative_se(), m.max_share(), m.effective_sample_size()
        stability = self._stability(name)
        move, limit = (None, None) if stability is None else stability
        vov = m.variance_of_variance()
        cv = None if rse is None else rse * math.sqrt(m.n)
        # VOV + 1/n = M4/M2**2 <= threshold; M4/M2**2 <= 1, so 1 disables
        # (checked first: rounding may put the ratio an ulp above 1).
        # The near-constant policy exception uses EMPIRICAL CV <= target,
        # not a bound on population variability or any unseen tail.
        variance_passes = (
            name != "cxr"
            or vov is not None
            and (
                (cv is not None and cv <= s.target_rse)
                or s.max_variance_of_variance >= 1.0
                or vov + 1.0 / m.n <= s.max_variance_of_variance
            )
        )
        passes = {
            "relative_se": rse is not None and rse <= s.target_rse,
            "max_electron_share": share is not None and share <= s.max_history_share,
            "effective_sample_size": ess >= s.min_effective_histories,
            "variance_sampling": variance_passes,
            "stability": stability is not None and stability[0] <= stability[1],
        }
        return {
            "n_electrons": m.n,
            "mean": m.mean,
            "relative_se": rse,
            "max_electron_share": share,
            "effective_sample_size": ess,
            "variance_of_variance": vov,
            "vov_trend_slope": vov_trend_slope(self.vov_history[name]),
            "coefficient_of_variation": cv,
            "stability": move,
            "stability_limit": limit,
            "passes": passes,
            "converged": all(passes.values()),
        }

    def _project(self, n: int) -> int:
        """``N = n (rse / target)**2`` for the worst observable, within ``[n, max]``."""
        s = self.settings
        worst = 0.0
        for moments in self.moments.values():
            rse = moments.relative_se()
            worst = max(worst, math.inf if rse is None else (rse / s.target_rse) ** 2)
        if not math.isfinite(worst):
            return s.max_histories
        return min(max(math.ceil(n * worst), n), s.max_histories)

    def decide(self, n: int) -> bool:
        """Apply the rule at the folded decision point ``n``; ``True`` stops there."""
        s = self.settings
        at_max = n >= s.max_histories
        if s.pilot_histories is not None and not at_max:
            if n < s.pilot_histories:
                return False
            if self.pilot is None:
                self.projected = self._project(n)
                self.pilot = {
                    "n_electrons": n,
                    "relative_se": {k: m.relative_se() for k, m in self.moments.items()},
                    "projected_electrons": self.projected,
                }
                if self._next is not None and n < self.projected < s.max_histories:
                    # Add N to the grid: the first check after the pilot falls
                    # on N itself, not on the grid point above it.
                    self._points = _with_point(chain((self._next,), self._points), self.projected)
                    self._next = next(self._points)
            if self.projected is not None and n < self.projected:
                return False
        self.checks += 1
        if all(self.observable_status(name)["converged"] for name in self.moments):
            self.stop_reason, self.realized = "converged", n
            return True
        if at_max:
            self.stop_reason, self.realized = "max_electrons", n
            return True
        return False

    def should_stop(self, n: int, recheck_at: int | None = None) -> bool:
        """Visit the decision points up to ``n``; ``True`` once one stops (``realized``).

        With ``recheck_at`` (realized-range replay) the points before it are
        only folded, and the rule is applied at ``recheck_at`` alone.
        """
        for point in self.pending(n):
            if (recheck_at is None or point == recheck_at) and self.decide(point):
                return True
        return False

    def next_point(self) -> int | None:
        """The next unvisited decision point, or ``None`` after the maximum."""
        return self._next

    def record(self) -> dict[str, Any]:
        """The case's statistics record (settings, guards, stop, per observable)."""
        s = self.settings
        return {
            "mode": "adaptive",
            "precision": s.to_dict(),
            "target_rse": s.target_rse,
            "min_electrons": s.min_histories,
            "max_electrons": s.max_histories,
            "decision_growth": s.decision_growth,
            "observables": list(s.observables),
            "sampling_model": ADAPTIVE_SAMPLING_MODEL,
            "guards": {
                "max_electron_share": s.max_history_share,
                "min_effective_electrons": s.min_effective_histories,
                "max_variance_of_variance": s.max_variance_of_variance,
                "stability_checks": s.stability_checks,
                "stability_fraction": s.stability_fraction,
            },
            "band_eV": None if s.band_eV is None else list(s.band_eV),
            "realized_electrons": self.realized,
            "realized_electrons_brem": self.realized,
            "stop_reason": self.stop_reason,
            "statistics_limited": self.stop_reason != "converged",
            "checks": self.checks,
            "pilot": self.pilot,
            "replayed_transport": self.replayed_transport,
            "statistics": {name: self.observable_status(name) for name in self.moments},
        }


def run_stopping_rule(
    settings: AdaptiveSettings, measure: Measure, chunk: int | None = None
) -> dict[str, Any]:
    """Apply the stopping rule to ``measure`` without transport.

    The monitor the runner uses, over a supplied per-electron population in
    chunks of ``chunk`` electrons (default: one chunk per decision point);
    ``block`` is ``None``. The record does not depend on ``chunk``. For tests
    and studies of the rule itself.
    """
    from .independent import IndependentStoppingMonitor

    monitor = (
        StoppingMonitor(settings, measure)
        if settings.brem_target_rse is None
        else IndependentStoppingMonitor(settings, measure)
    )
    start = 0
    while (point := monitor.next_point()) is not None:
        stop = point if chunk is None else min(start + chunk, settings.max_histories)
        monitor.on_block(start, stop, None)
        start = stop
        if monitor.should_stop(stop):
            break
        if monitor.cut_at is not None:
            monitor.truncate(monitor.cut_at)
            start = monitor.cut_at
    return monitor.record()


def _cxr_band(case, E_grid) -> tuple[float, float]:
    """The case's line band: its policy bandwidth, else its grid end points."""
    payload = case.get("line_grid_policy")
    bandwidth = payload.get("bandwidth") if isinstance(payload, Mapping) else None
    if isinstance(bandwidth, Mapping) and {"start_eV", "stop_eV"} <= set(bandwidth):
        return float(bandwidth["start_eV"]), float(bandwidth["stop_eV"])
    return float(E_grid[0]), float(E_grid[-1])


def band_weights(E_grid, start_eV: float, stop_eV: float) -> np.ndarray:
    """Integrate the piecewise-linear density over the clipped band (weights in eV).

    Interpolate at band edges between nodes; a narrow band without an interior
    node still has positive width. Outside the grid, no density is assumed.
    """
    grid = np.asarray(E_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("E_grid must contain at least two finite increasing energies")
    spacing = np.diff(grid)
    if np.any(spacing <= 0.0):
        raise ValueError("E_grid must contain at least two finite increasing energies")
    if not math.isfinite(start_eV) or not math.isfinite(stop_eV) or start_eV >= stop_eV:
        raise ValueError("band edges must be finite and increasing")
    left = np.maximum(grid[:-1], start_eV)
    right = np.minimum(grid[1:], stop_eV)
    width = np.maximum(right - left, 0.0)
    upper_weight = width * ((left - grid[:-1]) + (right - grid[:-1])) / (2.0 * spacing)
    weights = np.zeros(grid.size, dtype=float)
    weights[:-1] += width - upper_weight
    weights[1:] += upper_weight
    return weights


def case_measure(case, settings: AdaptiveSettings) -> Measure:
    """Per-electron line mass and brem band integral of one transport block.

    The block carries block-local electron ids, so each call normalizes over
    its own ``stop - start`` electrons and returns one value per electron.

    Validation: adaptive-sample-size-stopping
    """
    from .emission_adapters import EmissionBatch, select_emission

    emission = select_emission(case)
    emission.preflight(case, adaptive_observables=settings.observables)
    E_grid, E_brem, _beam, n_hat, groove = _case_geometry(case)
    start_eV, stop_eV = settings.band_eV or _cxr_band(case, E_grid)
    abs_layers = case.get("abs_layers")
    band_grid = np.array([start_eV, stop_eV])
    brem_q = band_weights(E_brem, start_eV, stop_eV) if "brem" in settings.observables else None

    def measure(start, stop, block):
        n = stop - start
        values = {}
        if "cxr" in settings.observables:
            # Only ``electron_mass`` is read: skip the grid reduction, which was
            # ~85% of the measure on the GPU (#1286); the line masses are unchanged.
            audit: dict[str, Any] = {
                "start_eV": start_eV,
                "stop_eV": stop_eV,
                "n_electrons": n,
                "mass_only": True,
            }
            with np.errstate(over="ignore", divide="ignore"):
                emission.cxr.evaluate(
                    EmissionBatch(
                        block,
                        band_grid,
                        case,
                        n_hat,
                        abs_layers,
                        groove,
                        electron_limit=n,
                        truncation_audit=audit,
                    ),
                    "spec",
                )
            mass = audit.get("electron_mass")
            values["cxr"] = np.zeros(n) if mass is None else mass
        if "brem" in settings.observables:
            values["brem"] = np.asarray(
                emission.continuum.evaluate(
                    EmissionBatch(
                        block,
                        E_brem,
                        case,
                        n_hat,
                        abs_layers,
                        groove,
                        electron_limit=n,
                        electron_band_weights=brem_q,
                    ),
                    "brem_wide",
                ).values
            )
        return values

    return measure


def _electron_rows(segments, start: int, stop: int):
    """Rows of electrons ``[start, stop)``; ids, scalars and per-electron fields kept."""
    key = "electron_id" if "electron_id" in segments else "elec_id"
    ids = np.asarray(_to_cpu(segments[key]))
    view = dict(segments)
    if ids.size and bool(np.all(ids[1:] >= ids[:-1])):
        rows: Any = slice(*np.searchsorted(ids, [start, stop], side="left"))
    else:
        rows = np.flatnonzero((ids >= start) & (ids < stop))
    for field in _SEG_ARRAYS:
        if field in view:
            view[field] = _take(view[field], rows)
    return view


#: Batches of the realized run for per-bin batch means (#1253).
BATCH_MEANS_BATCHES = 20


def batch_spans(n: int, batches: int = BATCH_MEANS_BATCHES) -> list[tuple[int, int]]:
    """``min(batches, n)`` contiguous electron ranges of sizes ``floor`` or ``ceil(n / K)``."""
    k = min(int(batches), int(n))
    bounds = [b * n // k for b in range(k + 1)] if k else [0]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _batch_summary(energies, batches, sizes):
    """Weighted batch means: ``S = sum (n_b / n) S_b``, ``se^2 = sum n_b (S_b - S)^2 / ((K - 1) n)``.

    Equal ``n_b = n / K`` gives the textbook ``s(S_b) / sqrt(K)``.

    Validation: adaptive-sample-size-stopping
    """
    values = np.asarray(batches, dtype=float)
    weights = np.asarray(sizes, dtype=float)
    n, k = float(weights.sum()), len(sizes)
    mean = (weights[:, None] * values).sum(axis=0) / n
    if k >= 2:
        spread = (weights[:, None] * (values - mean) ** 2).sum(axis=0) / (k - 1)
        se = np.sqrt(spread / n)
    else:
        se = np.full(mean.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean > 0.0, se / np.abs(mean), np.nan)
    finite = rel[np.isfinite(rel)]
    return {
        "E_eV": energies,
        "mean": mean,
        "standard_error": se,
        "relative_se": rel,
        "max_relative_se": float(finite.max()) if finite.size else None,
    }


def batch_means(case, transport, settings: AdaptiveSettings) -> dict[str, Any]:
    """Per-bin batch-means standard errors of ``spec`` and ``brem_wide`` in a band.

    The realized ``Ne`` (and, separately, ``Ne_brem``) electrons are split into
    :data:`BATCH_MEANS_BATCHES` contiguous batches of near-equal size,
    independent of how they were transported. Each batch's own line and brem
    spectra on the final grids, normalized per electron, give a weighted
    batch-means standard error of the run's spectrum in each bin. Reported
    after the final reduction; never used to stop, since the line grid is
    unknown while transport runs.

    Validation: adaptive-sample-size-stopping
    """
    from .emission_adapters import EmissionBatch, select_emission

    emission = select_emission(case)
    emission.preflight(case, streaming=True)
    if settings.batch_means_band_eV is None:
        raise ValueError("batch means need settings.batch_means_band_eV")
    start_eV, stop_eV = settings.batch_means_band_eV
    segs = transport["segs"]
    n = int(segs["Ne"])
    n_brem = int(transport["Ne_brem"])
    E_grid, E_brem = np.asarray(transport["E_grid"]), np.asarray(transport["E_brem"])
    cxr_bins = (E_grid >= start_eV) & (E_grid <= stop_eV)
    brem_bins = (E_brem >= start_eV) & (E_brem <= stop_eV)
    abs_layers = case.get("abs_layers")

    def batch(start, stop, energies, limit):
        return EmissionBatch(
            _electron_rows(segs, start, stop),
            energies,
            case,
            transport["n_hat"],
            abs_layers,
            transport.get("groove"),
            electron_limit=limit,
        )

    cxr_spans, brem_spans = batch_spans(n), batch_spans(n_brem)
    cxr_batches, brem_batches = [], []
    for start, stop in cxr_spans:
        # Normalized by n; n / n_b turns it into this batch's own mean.
        line = emission.cxr.evaluate(batch(start, stop, E_grid, n), "spec").values
        cxr_batches.append(np.asarray(_to_cpu(line))[cxr_bins] * n / (stop - start))
    for start, stop in brem_spans:
        brem = emission.continuum.evaluate(batch(start, stop, E_brem, n_brem), "brem_wide").values
        brem_batches.append(np.asarray(_to_cpu(brem))[brem_bins] * n_brem / (stop - start))
    return {
        "band_eV": [float(start_eV), float(stop_eV)],
        "n_batches": len(cxr_spans),
        **({"n_batches_brem": len(brem_spans)} if settings.brem_target_rse is not None else {}),
        "spec": _batch_summary(E_grid[cxr_bins], cxr_batches, [b - a for a, b in cxr_spans]),
        "brem_wide": _batch_summary(
            E_brem[brem_bins], brem_batches, [b - a for a, b in brem_spans]
        ),
    }


def adaptive_transport_core(case, settings: AdaptiveSettings, requested: str = "auto") -> str:
    """The per-electron/CUDA core an adaptive case runs on; lockstep is refused.

    ``"auto"`` takes the core a ``max_electrons`` run would, except that the
    lockstep CPU core becomes the per-electron CPU core. An explicit or
    environment-pinned lockstep core is refused.
    """
    from .. import runner

    loop_case = {key: value for key, value in case.items() if key != "adaptive_precision"}
    loop_case.update(Ne=settings.max_histories, Ne_brem=settings.max_histories)
    core = runner._case_transport_core(loop_case, requested)
    if core == "lockstep" and requested == "auto":
        core = runner._case_transport_core(loop_case, "per-electron")
    if core == "lockstep":
        raise ValueError(
            "adaptive electron counts need the per-electron or CUDA transport core; "
            f"{requested!r} resolved to 'lockstep'"
        )
    return core


def transport_case_adaptive(
    case: Mapping[str, Any],
    settings: Precision,
    *,
    transport_core: str = "auto",
    record_timing: bool = False,
    keep_segments_on_device: bool = False,
    block_consumer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Transport an adaptive case and retain its realized sampling provenance.

    The requested case remains the identity input. Returned segments, grids,
    counts, and statistics describe the realized population. Both serial and
    split transport/spectrum scheduling use this seam.

    ``block_consumer`` makes the run measure-only (#1280): the monitor reads
    each chunk, no chunk is kept, and ``segs`` holds only the realized
    additive counts. Streamed adaptive scoring then runs the fixed case at
    the realized pair (:func:`.streamed_scoring.run_case_streamed`).

    Validation: adaptive-sample-size-stopping
    """
    from .. import runner
    from ..case import refuse_channeling, refuse_transition_radiation

    refuse_channeling(case, "adaptive precision")
    refuse_transition_radiation(case, "adaptive precision")
    settings.validate_case(case)
    if block_consumer is not None and settings.batch_means_band_eV is not None:
        raise ValueError("batch means need the retained adaptive transport")
    if settings.batch_means_band_eV is not None:
        from .emission_adapters import select_emission

        select_emission(case).preflight(case, streaming=True)
    core = adaptive_transport_core(case, settings, transport_core)
    loop_case = {key: value for key, value in case.items() if key != "adaptive_precision"}
    loop_case.update(Ne=settings.max_histories, Ne_brem=settings.max_histories)
    from .independent import case_stopping_monitor

    monitor = case_stopping_monitor(loop_case, settings)
    transport = runner._transport_case(
        loop_case,
        record_timing,
        transport_core=core,
        keep_segments_on_device=keep_segments_on_device,
        block_electrons=AdaptiveChunkPlan.for_core(core, settings.block_histories),
        block_monitor=monitor,
        block_consumer=block_consumer,
    )
    statistics = monitor.record()
    statistics["transport_core"] = core
    transport["adaptive_sampling"] = statistics
    transport["_adaptive_precision"] = settings.to_dict()
    if statistics["statistics_limited"]:
        warn_statistics_limited(case, settings, statistics)
    return transport


def warn_statistics_limited(case, settings: Precision, statistics: Mapping[str, Any]) -> None:
    """Warn that an adaptive case stopped without meeting its rule, and why (#1267)."""
    if settings.brem_target_rse is None:
        stop = f"reached max_histories={settings.max_histories}"
    else:
        stop = (
            f"stopped at Ne={statistics['realized_electrons']}, "
            f"Ne_brem={statistics['realized_electrons_brem']}"
        )
    warnings.warn(
        f"{case.get('name', 'case')} at {case['E0_keV']:g} keV: adaptive stopping {stop} "
        f"without converging. {limit_summary(statistics)}. Treat the result as "
        "statistics-limited; the full per-observable record is in adaptive_sampling.",
        CxrYieldStatisticsWarning,
        stacklevel=3,
    )


def run_case_adaptive(
    case: Mapping[str, Any],
    settings: Precision,
    *,
    transport_core: str = "auto",
    record_timing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper returning the realized fixed-N result and statistics.

    Public callers select :class:`~pyrite.config.AdaptiveSampling` on ``Numerics`` and use
    ``simulate`` or ``run_case``; their output carries ``adaptive_sampling``.
    This helper retains the internal two-value return convention.

    Validation: adaptive-sample-size-stopping
    """
    from .. import runner

    transport = transport_case_adaptive(
        case, settings, transport_core=transport_core, record_timing=record_timing
    )
    result = runner._spectrum_case(case, transport, record_timing)
    statistics = result.pop("adaptive_sampling")
    return result, statistics


def case_transport_core(case, requested="auto"):
    """Resolve which transport core a case dict runs on.

    Mirrors what ``simulate_trajectories`` will decide for this case: transport
    covers both electron populations, so the count that matters is
    ``max(Ne, Ne_brem)``, and ``auto`` keeps a grooved entrance face on the lockstep core.
    """

    from ..transport import resolve_transport_core

    if case.get("transport_mode") == "mixed":
        # No lockstep specialization: every CPU route is the per-electron core.
        core = resolve_transport_core(requested, max(case.get("Ne") or 0, case.get("Ne_brem") or 0))
        return core if core == "cuda" or requested != "auto" else "per-electron"
    if case.get("adaptive_precision") is not None:
        return adaptive_transport_core(
            case, Precision.from_dict(case["adaptive_precision"]), requested
        )
    return resolve_transport_core(
        requested,
        max(case.get("Ne") or 0, case.get("Ne_brem") or 0),
        groove=case.get("groove_spacing_ang"),
    )


def _adaptive_worker_call(function, case, *args, **kwargs):
    """Replace only the pool's automatic lockstep pin for one adaptive case.

    Explicit inherited CPU-core choices remain authoritative. Restore the pool
    default afterwards, so a fixed case in a mixed batch keeps its old draws.
    """
    from ..._util.env import env_value, set_canonical_env
    from .. import runner

    if runner._WORKER_INHERITED_CORE not in ("", "auto", "cuda"):
        return function(case, *args, **kwargs)
    original = env_value("PYRITE_MC_TRANSPORT_CORE", "")
    set_canonical_env("PYRITE_MC_TRANSPORT_CORE", "per-electron")
    try:
        return function(case, *args, **kwargs)
    finally:
        set_canonical_env("PYRITE_MC_TRANSPORT_CORE", original)


def _run_adaptive_case_worker(case, *args, **kwargs):
    from .. import runner

    return _adaptive_worker_call(runner.run_case, case, *args, **kwargs)


def _transport_adaptive_case_worker(case, *args, **kwargs):
    from .. import runner

    return _adaptive_worker_call(runner._transport_case, case, *args, **kwargs)


def transport_requested_precision(case, record_timing=False, **options):
    """Validate runner-only options and dispatch a serialized precision policy."""
    if options.pop("trajectory_capture", None) is not None:
        raise ValueError("adaptive precision does not support trajectory capture; use fixed counts")
    if (
        options.pop("block_electrons", None) is not None
        or options.pop("block_monitor", None) is not None
    ):
        raise ValueError("adaptive precision owns block_electrons and the stopping monitor")
    return transport_case_adaptive(
        case,
        Precision.from_dict(case["adaptive_precision"]),
        record_timing=record_timing,
        **options,
    )


def attach_sampling(case, transport, output) -> None:
    """Attach realized statistics and optional final-grid batch means to output."""
    if "adaptive_sampling" not in transport:
        return
    sampling = dict(transport["adaptive_sampling"])
    precision = Precision.from_dict(transport["_adaptive_precision"])
    if precision.batch_means_band_eV is not None:
        sampling["batch_means"] = batch_means(case, transport, precision)
    output["adaptive_sampling"] = sampling


def realized_case(case, transport):
    """Use realized counts for reduction while retaining the requested identity."""
    if "adaptive_sampling" not in transport:
        return case
    return {**case, "Ne": transport["Ne_lines"], "Ne_brem": transport["Ne_brem"]}


def validate_directions(case, directions) -> None:
    """Reject directional scoring before starting an adaptive transport."""
    if case.get("adaptive_precision") is not None and directions is not None:
        raise ValueError(
            "adaptive precision does not support observation directions; use fixed counts"
        )
