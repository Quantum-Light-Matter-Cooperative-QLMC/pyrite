"""Adaptive electron counts: sequential block-wise stopping on a target error (#361).

``Numerics.precision`` selects this path through the public runner.
:func:`run_case_adaptive` retains the internal compatibility interface. Both
transport electron blocks
``[kB, (k+1)B)`` through :mod:`.block_transport` and, after each block, folds
grid-independent per-electron scalars into running moments:

* ``"line"`` -- the per-electron line mass ``sum w pi / a_w`` over the lines a
  fixed monitor band keeps, the quantity of the #201 line-yield audit;
* ``"brem"`` -- the per-electron bremsstrahlung integral over the same band.

Neither depends on the line grid, which is resolved only once transport has
stopped, from every realized electron, before the single spectrum reduction.
The stop is the first block boundary ``n`` at which, for every chosen
observable, the relative standard error of the mean is at most
``target_rse`` and the heavy-tail guards hold (largest single-electron share,
effective sample size, stability of the running mean over the last blocks),
subject to ``min_electrons``; at ``max_electrons`` the case completes flagged
``statistics_limited``. The realized case is the fixed-N case at the realized
count, bit for bit, so the stop changes how many electrons run, never how
they run.

Estimator, rule, guards and their limiting cases are derived in
``docs/computation/statistical-methods.md``.

Validation: adaptive-sample-size-stopping
"""

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..._backend import _to_cpu
from ..._line_grid_policy import LineYieldStatisticsWarning
from ..._precision import OBSERVABLES as OBSERVABLES
from ..._precision import Precision
from ..spectrum.lines._kernels import _SEG_ARRAYS
from .artifacts import _case_geometry
from .block_transport import electron_blocks
from .electron_blocks import _take

#: ``measure(start, stop, block) -> {observable: per-electron values}``.
Measure = Callable[[int, int, Mapping[str, Any]], Mapping[str, Any]]


# Internal compatibility name; the public policy lives below the domain packages.
AdaptiveSettings = Precision


class RunningMoments:
    """Exact running moments of one non-negative per-electron observable.

    Each block is reduced where its values live (device or host) to its count,
    sum, centred second moment, sum of squares and maximum; blocks merge by
    Chan et al.'s pairwise update, so the mean and variance equal a one-pass
    Welford over the concatenated values up to rounding.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
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
        block_m2 = float(((values - block_mean) ** 2).sum())
        block_sq = float((values * values).sum())
        block_max = float(values.max())
        n = self.n + nb
        delta = block_mean - self.mean
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


class StoppingMonitor:
    """Block hooks for :func:`.block_transport.transport_electron_blocks`.

    ``on_block`` measures each block and folds it into the moments;
    ``should_stop`` applies the stopping rule at the block's end. A block at
    ``start == 0`` resets the state, so a restarted transport (an OOM retry)
    is measured once. Decisions depend only on the measured values, so the
    same seed and settings give the same realized count.

    Validation: adaptive-sample-size-stopping
    """

    def __init__(self, settings: AdaptiveSettings, measure: Measure):
        self.settings = settings
        self.measure = measure
        self.replayed_transport = False
        self._reset()

    def _reset(self):
        self.moments = {name: RunningMoments() for name in self.settings.observables}
        self.history: list[tuple[int, dict[str, float]]] = []
        self.checks = 0
        self.pilot: dict[str, Any] | None = None
        self.projected: int | None = None
        self.stop_reason: str | None = None
        self.realized: int | None = None

    def on_block(self, start: int, stop: int, block) -> None:
        if start == 0:
            self._reset()
        values = self.measure(start, stop, block)
        for name, moments in self.moments.items():
            if values[name].ndim != 1 or values[name].size != stop - start:
                raise ValueError(f"{name} must contain one value per electron in the block")
            moments.add_block(values[name])
        self.history.append((stop, {name: m.mean for name, m in self.moments.items()}))

    def _stability(self, name: str) -> float | None:
        """Largest relative move of the running mean over the last k block ends."""
        k = self.settings.stability_blocks
        if k == 0:
            return 0.0
        if len(self.history) < k + 1:
            return None
        current = self.history[-1][1][name]
        if current == 0.0:
            return None
        return max(abs(means[name] - current) for _, means in self.history[-k - 1 : -1]) / abs(
            current
        )

    def observable_status(self, name: str) -> dict[str, Any]:
        s = self.settings
        m = self.moments[name]
        rse, share, ess = m.relative_se(), m.max_share(), m.effective_sample_size()
        stability = self._stability(name)
        passes = {
            "relative_se": rse is not None and rse <= s.target_rse,
            "max_electron_share": share is not None and share <= s.max_electron_share,
            "effective_sample_size": ess >= s.min_effective_electrons,
            "stability": stability is not None and stability <= s.stability_fraction * s.target_rse,
        }
        return {
            "n_electrons": m.n,
            "mean": m.mean,
            "relative_se": rse,
            "max_electron_share": share,
            "effective_sample_size": ess,
            "stability": stability,
            "passes": passes,
            "converged": all(passes.values()),
        }

    def _project(self, n: int) -> int:
        s = self.settings
        worst = 0.0
        for moments in self.moments.values():
            rse = moments.relative_se()
            worst = max(worst, math.inf if rse is None else (rse / s.target_rse) ** 2)
        if not math.isfinite(worst):
            return s.max_electrons
        block = s.block_electrons
        projected = math.ceil(n * worst / block) * block
        return min(max(projected, n), s.max_electrons)

    def should_stop(self, n: int) -> bool:
        s = self.settings
        at_max = n >= s.max_electrons
        if n < s.min_electrons and not at_max:
            return False
        if s.pilot_electrons is not None and not at_max:
            if n < s.pilot_electrons:
                return False
            if self.pilot is None:
                self.projected = self._project(n)
                self.pilot = {
                    "n_electrons": n,
                    "relative_se": {k: m.relative_se() for k, m in self.moments.items()},
                    "projected_electrons": self.projected,
                }
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

    def record(self) -> dict[str, Any]:
        """The case's statistics record (settings, guards, stop, per observable)."""
        s = self.settings
        return {
            "mode": "adaptive",
            "precision": s.to_dict(),
            "target_rse": s.target_rse,
            "min_electrons": s.min_electrons,
            "max_electrons": s.max_electrons,
            "block_electrons": s.block_electrons,
            "observables": list(s.observables),
            "guards": {
                "max_electron_share": s.max_electron_share,
                "min_effective_electrons": s.min_effective_electrons,
                "stability_blocks": s.stability_blocks,
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


def run_stopping_rule(settings: AdaptiveSettings, measure: Measure) -> dict[str, Any]:
    """Apply the stopping rule to ``measure`` without transport.

    The same block loop and monitor the runner uses, over a supplied
    per-electron population; ``block`` is ``None``. For tests and studies of
    the rule itself.
    """
    monitor = StoppingMonitor(settings, measure)
    for start, stop in electron_blocks(settings.max_electrons, settings.block_electrons):
        monitor.on_block(start, stop, None)
        if monitor.should_stop(stop):
            break
    return monitor.record()


def _line_band(case, E_grid) -> tuple[float, float]:
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
    from .. import runner

    E_grid, E_brem, _beam, n_hat, groove = _case_geometry(case)
    start_eV, stop_eV = settings.band_eV or _line_band(case, E_grid)
    abs_layers = case.get("abs_layers")
    band_grid = np.array([start_eV, stop_eV])
    brem_q = band_weights(E_brem, start_eV, stop_eV) if "brem" in settings.observables else None

    def measure(start, stop, block):
        n = stop - start
        values = {}
        if "line" in settings.observables:
            audit: dict[str, Any] = {"start_eV": start_eV, "stop_eV": stop_eV, "n_electrons": n}
            with np.errstate(over="ignore", divide="ignore"):
                runner._lines_for_segments(
                    block,
                    band_grid,
                    case,
                    n_hat,
                    abs_layers,
                    groove,
                    coherent=False,
                    Ne=n,
                    truncation_audit=audit,
                )
            mass = audit.get("electron_mass")
            values["line"] = np.zeros(n) if mass is None else mass
        if "brem" in settings.observables:
            values["brem"] = np.asarray(
                runner._brem_wide_from_segments(
                    block,
                    E_brem,
                    case,
                    n_hat,
                    abs_layers,
                    groove=groove,
                    Ne=n,
                    electron_band_weights=brem_q,
                )
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


def batch_means(case, transport, settings: AdaptiveSettings) -> dict[str, Any]:
    """Per-bin batch-means standard errors of ``spec`` and ``brem_wide`` in a band.

    Each equal block of the realized run is one batch: its own line and brem
    spectra on the final grids, normalized per electron. The batch means'
    sample standard deviation over ``sqrt(k)`` estimates the standard error of
    the run's spectrum in each bin. Reported after the final reduction; never
    used to stop, since the line grid is unknown while transport runs.

    Validation: adaptive-sample-size-stopping
    """
    from .. import runner

    if settings.batch_means_band_eV is None:
        raise ValueError("batch means need settings.batch_means_band_eV")
    start_eV, stop_eV = settings.batch_means_band_eV
    segs = transport["segs"]
    n = int(segs["Ne"])
    blocks = electron_blocks(n, settings.block_electrons)
    E_grid, E_brem = np.asarray(transport["E_grid"]), np.asarray(transport["E_brem"])
    line_bins = (E_grid >= start_eV) & (E_grid <= stop_eV)
    brem_bins = (E_brem >= start_eV) & (E_brem <= stop_eV)
    abs_layers = case.get("abs_layers")
    line_batches, brem_batches = [], []
    for start, stop in blocks:
        rows = _electron_rows(segs, start, stop)
        scale = n / (stop - start)  # Ne = n normalization -> this batch's own mean
        line = runner._lines_for_segments(
            rows,
            E_grid,
            case,
            transport["n_hat"],
            abs_layers,
            transport.get("groove"),
            coherent=False,
            Ne=n,
        )
        brem = runner._brem_wide_from_segments(
            rows, E_brem, case, transport["n_hat"], abs_layers, groove=transport.get("groove"), Ne=n
        )
        line_batches.append(np.asarray(_to_cpu(line))[line_bins] * scale)
        brem_batches.append(np.asarray(_to_cpu(brem))[brem_bins] * scale)
    k = len(blocks)

    def summary(energies, batches):
        values = np.asarray(batches, dtype=float)
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(k) if k >= 2 else np.full(mean.shape, np.nan)
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

    return {
        "band_eV": [float(start_eV), float(stop_eV)],
        "n_batches": k,
        "batch_electrons": settings.block_electrons,
        "spec": summary(E_grid[line_bins], line_batches),
        "brem_wide": summary(E_brem[brem_bins], brem_batches),
    }


def adaptive_transport_core(case, settings: AdaptiveSettings, requested: str = "auto") -> str:
    """The per-electron/CUDA core an adaptive case runs on; lockstep is refused.

    ``"auto"`` takes the core a ``max_electrons`` run would, except that the
    lockstep CPU core becomes the per-electron CPU core. An explicit or
    environment-pinned lockstep core is refused.
    """
    from .. import runner

    loop_case = {key: value for key, value in case.items() if key != "adaptive_precision"}
    loop_case.update(Ne=settings.max_electrons, Ne_brem=settings.max_electrons)
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
) -> dict[str, Any]:
    """Transport an adaptive case and retain its realized sampling provenance.

    The requested case remains the identity input. Returned segments, grids,
    counts, and statistics describe the realized population. Both serial and
    split transport/spectrum scheduling use this seam.

    Validation: adaptive-sample-size-stopping
    """
    from .. import runner

    settings.validate_case(case)
    core = adaptive_transport_core(case, settings, transport_core)
    loop_case = {key: value for key, value in case.items() if key != "adaptive_precision"}
    loop_case.update(Ne=settings.max_electrons, Ne_brem=settings.max_electrons)
    monitor = StoppingMonitor(settings, case_measure(loop_case, settings))
    transport = runner._transport_case(
        loop_case,
        record_timing,
        transport_core=core,
        keep_segments_on_device=keep_segments_on_device,
        block_electrons=settings.block_electrons,
        block_monitor=monitor,
    )
    statistics = monitor.record()
    statistics["transport_core"] = core
    transport["adaptive_sampling"] = statistics
    transport["_adaptive_precision"] = settings.to_dict()
    if statistics["statistics_limited"]:
        worst = {name: status["relative_se"] for name, status in statistics["statistics"].items()}
        warnings.warn(
            f"{case.get('name', 'case')} at {case['E0_keV']:g} keV: the adaptive run "
            f"reached max_electrons={settings.max_electrons} without meeting "
            f"target_rse={settings.target_rse:g} and its guards (relative SE {worst}); "
            "treat the result as statistics-limited.",
            LineYieldStatisticsWarning,
            stacklevel=2,
        )
    return transport


def run_case_adaptive(
    case: Mapping[str, Any],
    settings: Precision,
    *,
    transport_core: str = "auto",
    record_timing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper returning the realized fixed-N result and statistics.

    Public callers select a :class:`~pyrite.Precision` on ``Numerics`` and use
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
    ``max(Ne, Ne_brem)``, and a grooved entrance face stays on the lockstep core.
    """

    from ..transport import resolve_transport_core

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
    from ..._env import env_value, set_canonical_env
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
