"""Case-local line-grid policy: bandwidth, resolution, tolerance, precedence.

A production run must not need ``pyrite material energy-grid derive`` first. When
no explicit grid and no stored catalog row cover a case, ``build_cases`` attaches
a *policy* instead of coordinates, and the runner resolves the coordinates from
the run's own trajectories, before the spectrum phase and without a second Monte
Carlo job.

Two policies are deliberately separate (issue #101):

bandwidth
    Where the line grid starts and stops. The automatic default is the
    ``kinematic-ceiling`` bound below -- closed form, no simulation. The stored
    catalog artifacts instead carry ``coverage-0.95``, a *bandwidth* policy
    measured by ``energy_grid.derive``: 95% of integrated coherent-line
    intensity. Neither is an accuracy statement. ``resonance-population``
    (#192, opt-in) keeps the ceiling as a cap but stops where the case's own
    resonance population and characteristic lines leave at most
    :data:`DEFAULT_BANDWIDTH_TRUNCATION` of their mass above the axis; the
    spectrum phase audits that share with the production weights.

resolution
    How finely that interval is sampled. Owned by the sinc Nyquist estimator
    (:func:`pyrite.montecarlo.spectrum.diagnostics.sinc_feature_spacing`) under a
    per-observable relative tolerance. Never silently coarsened: a spacing that
    cannot be met inside the point budget or the backend ULP floor raises.

windows
    Optional piecewise refinement (``feature-windows``): a backbone at the
    maximum spacing plus fine windows seeded from kinematics, absorption edges
    and characteristic lines (:mod:`pyrite._line_windows`). Off unless a
    per-call or stored policy asks for it; a policy without windows keeps its
    historical payload bit-for-bit.

quadrature
    How the ``sinc^2`` line profile is evaluated on those coordinates (#116).
    ``node`` (default) samples it at each node; ``bin-mean`` writes the
    closed-form mean over each node's bin, which makes integrated yield exact
    at any spacing but is a yield-only choice -- it smooths peak height and
    width. One explicit switch, not a per-observable field: a run produces one
    line spectrum, so the choice is the caller's statement of what that
    spectrum is for. Off unless a per-call or stored policy asks for it; a
    ``node`` policy keeps its historical payload bit-for-bit.

This module is a leaf on purpose. ``energy_grid`` imports ``montecarlo.runner``,
so the runner cannot import ``energy_grid``; and #64 may move the energy-grid CLI
entirely. Only ``numpy``, ``pyrite._env``, ``pyrite.paths``, the grid leaves
``_grid_semantics`` and ``_line_windows``, and the pure
``pyrite.materials.crystal`` kinematics are used here.
"""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._env import env_value
from ._grid_semantics import (
    resolution_num,
    validate_backend_coordinates,
    validate_backend_spacing,
)
from ._line_windows import WindowPlan
from .paths import cache_dir

__all__ = [
    "AUTOMATIC_BANDWIDTH_POLICY",
    "BANDWIDTH_POLICIES",
    "BANDWIDTH_PROXY_SAFETY",
    "AUTOMATIC_RESOLUTION_POLICY",
    "DEFAULT_LOCAL_HALO_LIMIT",
    "LOCAL_RESOLUTION_POLICY",
    "RESOLUTION_POLICIES",
    "COVERAGE_BANDWIDTH_POLICY",
    "DEFAULT_BANDWIDTH_COVERAGE",
    "DEFAULT_BANDWIDTH_TRUNCATION",
    "FLOAT32_LINESHAPE_BINADE_EV",
    "LINE_GRID_POLICY_SCHEMA",
    "LINE_QUADRATURES",
    "QUADRATURE_LINE_GRID_POLICY_SCHEMA",
    "RESONANCE_BANDWIDTH_POLICY",
    "RESONANCE_LINE_GRID_POLICY_SCHEMA",
    "OBSERVABLE_CLASSES",
    "WINDOWED_LINE_GRID_POLICY_SCHEMA",
    "WINDOW_POLICY",
    "LineGridPolicy",
    "LineShapePrecisionWarning",
    "cached_coordinates",
    "coordinate_cache_key",
    "environment_overrides_present",
    "line_quadrature_from_payload",
    "kinematic_line_stop_eV",
    "lineshape_precision_warning",
    "line_start_eV",
    "resolve_line_grid_policy",
    "resolved_coordinates",
    "store_coordinates",
    "windowed_coordinates",
]

#: Payload version. Bump when a field changes meaning: the payload is hashed
#: into case identity, so a silent reinterpretation would reuse stale results.
LINE_GRID_POLICY_SCHEMA = 1
#: Payload version of a policy carrying a ``windows`` block. A policy without
#: one keeps :data:`LINE_GRID_POLICY_SCHEMA` and its payload bit-for-bit, so
#: existing automatic case identities do not move.
WINDOWED_LINE_GRID_POLICY_SCHEMA = 2
#: Payload version of a policy selecting a non-default line quadrature (#116),
#: with or without windows. A ``node`` policy keeps schema 1 or 2 unchanged.
QUADRATURE_LINE_GRID_POLICY_SCHEMA = 3
#: Payload version of a policy selecting the ``resonance-population`` bandwidth
#: (#192). Every other policy keeps its schema and payload unchanged.
RESONANCE_LINE_GRID_POLICY_SCHEMA = 4

#: Line quadratures, default first. Restated from
#: ``montecarlo.spectrum.lines._bin_quadrature`` so this module stays a leaf; a
#: test pins the two against each other.
LINE_QUADRATURES = ("node", "bin-mean")
DEFAULT_LINE_QUADRATURE = LINE_QUADRATURES[0]

#: Window policy name and defaults. Restated from
#: ``montecarlo.spectrum.line_seeds`` so this module stays a leaf; a test pins
#: the two against each other.
WINDOW_POLICY = "feature-windows"
DEFAULT_WINDOW_SAMPLES_PER_FEATURE = 8
DEFAULT_WINDOW_TAIL_WIDTHS = 2.0
DEFAULT_WINDOW_PROVIDERS = ("pxr-kinematic", "absorption-edge", "characteristic")
_WINDOW_KEYS = ("providers", "samples_per_feature", "tail_widths")

#: Observables whose accuracy tolerance is set independently. One global
#: tolerance is refused by design (#101): an intrinsic-source spectrum and a
#: detector-convolved count rate do not need the same grid.
OBSERVABLE_CLASSES = ("intrinsic_source", "detected_counts")

#: Built-in per-observable relative accuracy tolerances. Precision intrinsic
#: yield is held an order tighter than detected counts, which are dominated by
#: response convolution and Monte Carlo statistics rather than by grid error.
#:
#: Each value is the *grid* share of the total error, not the total error of a
#: spectrum: transport and Monte Carlo statistics are separate additive terms
#: that the refinement ladder holds fixed rather than bounds. The split of each
#: total among bandwidth, quadrature, interpolation and backend precision is
#: allocated in ``docs/validation/beam-transport/line-spectrum-error-budget.md``
#: (table ``tbl-line-budget-allocation``); this mapping is that table's
#: intrinsic-source and detected-counts column totals.
DEFAULT_RTOL: Mapping[str, float] = {"intrinsic_source": 1.0e-3, "detected_counts": 1.0e-2}

#: BANDWIDTH policy of the stored ``E_grid_line_by_energy`` artifacts: the
#: energy below which 95% of integrated PXR/CBS line intensity falls
#: (characteristic emission excluded). It states
#: how much spectrum is covered, NOT how accurately it is sampled.
DEFAULT_BANDWIDTH_COVERAGE = 0.95
COVERAGE_BANDWIDTH_POLICY = "coverage-0.95"
AUTOMATIC_BANDWIDTH_POLICY = "kinematic-ceiling"
#: Trajectory-measured bandwidth under the ceiling cap (#192). Opt-in only.
RESONANCE_BANDWIDTH_POLICY = "resonance-population"
BANDWIDTH_POLICIES = (AUTOMATIC_BANDWIDTH_POLICY, RESONANCE_BANDWIDTH_POLICY)
#: Upper-edge truncation share of ``resonance-population``: the bandwidth row of
#: ``tbl-line-budget-allocation``, a fraction of integrated incoherent line
#: yield. The spectrum phase gates the production-weight truncation on it; the
#: edge is chosen from the ``t_L**2`` proxy at half of it. Measured on 5 MeV
#: h-BN (#192), the true loss sat 2.5-5x below the proxy target, and the audit
#: bound is itself 2x the exact far tail.
DEFAULT_BANDWIDTH_TRUNCATION = 1.0e-4
BANDWIDTH_PROXY_SAFETY = 2.0
AUTOMATIC_RESOLUTION_POLICY = "sinc-nyquist"
#: Energy-dependent spacing from the measured resonance population (#192).
#: Opt-in, and only under the ``resonance-population`` bandwidth that measures
#: that population.
LOCAL_RESOLUTION_POLICY = "resonance-local"
RESOLUTION_POLICIES = (AUTOMATIC_RESOLUTION_POLICY, LOCAL_RESOLUTION_POLICY)
#: Per-line tail share outside the fine halo of ``resonance-local``: the
#: quadrature-backbone row of ``tbl-line-budget-allocation`` split in two.
DEFAULT_LOCAL_HALO_LIMIT = 1.0e-4

#: Coarsest automatic spacing. Matches the catalog's historical 3 eV line-grid
#: convention, so automatic resolution is never worse than the legacy grids.
DEFAULT_MAX_SPACING_EV = 3.0
#: Backend coordinate-precision safety factor, shared with ``energy_grid.derive``.
#: Holds the backend-precision row of ``tbl-line-budget-allocation`` (1e-4 of the
#: intrinsic-source share); #109 measured float32-versus-FP64 deviation flat in
#: ``spacing/ulp``, so this collapse floor, not a spacing floor, is the control.
DEFAULT_BACKEND_SAFETY_ULPS = 8.0
#: Refuse rather than coarsen beyond this many line coordinates.
DEFAULT_MAX_POINTS = 600_000
#: Round the automatic ``stop`` up to a multiple of this, so nearby cases share
#: a bandwidth (and therefore a cache entry and a checkpoint identity).
STOP_ROUND_TO_EV = 100.0

_ENV_RTOL = "PYRITE_ENERGY_GRID_RTOL"
_ENV_MAX_SPACING = "PYRITE_ENERGY_GRID_MAX_SPACING_EV"
_ENV_ULPS = "PYRITE_ENERGY_GRID_ULPS"
_ENV_MAX_POINTS = "PYRITE_ENERGY_GRID_MAX_POINTS"


def _env_rtol_name(observable: str) -> str:
    return f"{_ENV_RTOL}_{observable.upper()}"


#: Every environment name that participates in the precedence chain. Presence of
#: any one makes automatic resolution outrank a stored catalog row.
ENVIRONMENT_NAMES = (
    _ENV_RTOL,
    *(_env_rtol_name(name) for name in OBSERVABLE_CLASSES),
    _ENV_MAX_SPACING,
    _ENV_ULPS,
    _ENV_MAX_POINTS,
)


def environment_overrides_present() -> bool:
    """True when any ``PYRITE_ENERGY_GRID_*`` policy value is set."""
    return any(env_value(name) is not None for name in ENVIRONMENT_NAMES)


def line_start_eV(energy_keV: float) -> float:
    """Line-grid ``start_eV`` convention: 10 eV at <=60 keV beam, 50 eV above.

    Lives here rather than in ``energy_grid.bounds`` so ``campaign`` and the
    runner can reach it without importing the driver-side energy-grid package;
    ``energy_grid.bounds`` re-exports it unchanged.
    """
    return 10.0 if float(energy_keV) <= 60.0 else 50.0


def kinematic_line_stop_eV(
    g_magnitudes_invang,
    energy_keV: float,
    *,
    round_to_eV: float = STOP_ROUND_TO_EV,
) -> float:
    """Closed-form upper bound on any PXR/CBS line energy, in eV.

    Source equation: the resonance the line kernels solve,
    ``E_res = hbar c (v . g) / (1 - n_hat . v)``
    (``montecarlo/spectrum/lines/_kernels.py::_line_kin_core``; ledger rows
    ``line-energy-dispersion`` and ``finite-time-lineshape``). Writing
    ``v = beta c v_hat`` and maximizing numerator and denominator independently
    over direction gives ``|v . g| <= beta |g|`` and ``1 - n_hat . v >= 1 - beta``,
    hence

        ``E_res <= hbar c beta |g|_max / (1 - beta)``.

    Assumptions: the emission and observation directions are unconstrained, so
    this is a bound and not an estimate -- multiple scattering may point an
    electron anywhere, and at 300 keV the true stop measured by
    ``energy_grid.derive`` sits well above the fixed-geometry value
    ``1 - beta cos(theta_obs)`` would give. The reflection set is the case's own
    ``hkl_list``; reflections outside it radiate nothing in this model.
    ``beta`` is the incident speed, the largest the electron ever has, so
    slowing down only moves lines down.

    Limiting cases: ``beta -> 0`` gives ``E_res -> 0`` (a stopped electron
    radiates no line); at fixed ``beta`` the bound is linear in ``|g|_max``,
    i.e. the ``n``-th order reflection sits ``n`` times higher, which is the
    observed ``(002)``/``(004)`` ordering.

    Validation: line-grid-kinematic-bandwidth
    """
    magnitudes = np.atleast_1d(np.asarray(g_magnitudes_invang, dtype=float))
    finite = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    if finite.size == 0:
        raise ValueError("kinematic line bandwidth needs at least one positive |g|")
    energy = float(energy_keV)
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("beam energy must be finite and positive")
    # Local restatement of materials.crystal.beta_from_Ee; kept inline so this
    # leaf stays importable from every layer that needs a grid.
    gamma = 1.0 + energy * 1.0e3 / _ELECTRON_REST_EV
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    bound = _HBARC_EV_ANG * beta * float(finite.max()) / (1.0 - beta)
    return float(math.ceil(bound / float(round_to_eV)) * float(round_to_eV))


#: hbar c in eV*Angstrom and the electron rest energy in eV. Identical to
#: ``materials.crystal.HBARC_EV_ANG`` / ``M_E_EV``; restated so this module
#: stays a dependency-free leaf (a test pins the two against each other).
_HBARC_EV_ANG = 1973.269804
_ELECTRON_REST_EV = 510998.95


class LineGridToleranceError(ValueError):
    """A requested tolerance cannot be met within the declared budget."""


class LineShapePrecisionWarning(UserWarning):
    """A float32 line window reaches the binade where lineshape exceeds its share."""


#: Lower edge of the ``[2**14, 2**15)`` eV float32 binade. Measured on identical
#: segments (diamond 300 keV), a window reaching it keeps yield and centroid
#: inside their backend-precision shares but not dominant-line FWHM
#: (1.0-1.8e-3 against 1e-3); see
#: ``docs/validation/beam-transport/line-spectrum-error-budget.md``.
FLOAT32_LINESHAPE_BINADE_EV = 16384.0


def lineshape_precision_warning(record: Mapping[str, Any]) -> str | None:
    """The float32 lineshape warning a resolved-grid record calls for, if any."""
    reach = record.get("float32_lineshape_window_eV")
    if reach is None:
        return None
    return (
        f"a float32 line window reaches {reach:g} eV, in the "
        f">= {FLOAT32_LINESHAPE_BINADE_EV:g} eV binade where dominant-line FWHM was "
        "measured at 1-2x its 1e-3 backend-precision share (yield and centroid stay "
        "inside theirs); set PYRITE_FP64=1 if lineshape is gated"
    )


@dataclass(frozen=True, slots=True)
class LineGridPolicy:
    """Resolved, result-affecting line-grid policy for one case.

    Every field here enters the case payload and therefore case, checkpoint, and
    cache identity. ``sources`` records where each value came from so provenance
    can report the precedence decision, and is identity-bearing too: the same
    number supplied by a different layer is the same physics, but recording the
    layer is what makes a precedence regression visible in a stored result.
    """

    bandwidth_policy: str
    start_eV: float
    stop_eV: float
    resolution_policy: str
    max_spacing_eV: float
    aliased_weight_limit: float
    rtol: tuple[tuple[str, float], ...]
    backend_safety_ulps: float
    max_points: int
    sources: tuple[tuple[str, str], ...]
    windows: tuple[tuple[str, Any], ...] | None = None
    quadrature: str = DEFAULT_LINE_QUADRATURE
    bandwidth_truncation: float | None = None
    halo_limit: float | None = None

    def payload(self) -> dict[str, Any]:
        """Canonical JSON-able payload carried on the case."""
        payload = {
            "schema": LINE_GRID_POLICY_SCHEMA,
            "bandwidth": {
                "policy": self.bandwidth_policy,
                "start_eV": self.start_eV,
                "stop_eV": self.stop_eV,
            },
            "resolution": {
                "policy": self.resolution_policy,
                "max_spacing_eV": self.max_spacing_eV,
                "aliased_weight_limit": self.aliased_weight_limit,
                "backend_safety_ulps": self.backend_safety_ulps,
                "max_points": self.max_points,
            },
            "rtol": dict(self.rtol),
            "sources": dict(self.sources),
        }
        if self.windows is not None:
            windows = dict(self.windows)
            payload["schema"] = WINDOWED_LINE_GRID_POLICY_SCHEMA
            payload["windows"] = {**windows, "providers": list(windows["providers"])}
        if self.quadrature != DEFAULT_LINE_QUADRATURE:
            payload["schema"] = QUADRATURE_LINE_GRID_POLICY_SCHEMA
            payload["quadrature"] = self.quadrature
        if self.bandwidth_policy == RESONANCE_BANDWIDTH_POLICY:
            # stop_eV stays the closed-form ceiling: the cap the measured edge
            # may not exceed, and the placeholder grid's extent.
            payload["schema"] = RESONANCE_LINE_GRID_POLICY_SCHEMA
            payload["bandwidth"] = {
                **payload["bandwidth"],
                "truncation_limit": self.bandwidth_truncation,
            }
        if self.resolution_policy == LOCAL_RESOLUTION_POLICY:
            payload["resolution"] = {**payload["resolution"], "halo_limit": self.halo_limit}
        return payload

    @property
    def strictest_rtol(self) -> float:
        """The governing tolerance: the tightest across observable classes."""
        return min(value for _, value in self.rtol)


def _float_env(name: str, *, minimum: float, maximum: float | None = None) -> float | None:
    raw = env_value(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from None
    if not math.isfinite(value) or value <= minimum or (maximum is not None and value >= maximum):
        upper = "" if maximum is None else f" and < {maximum:g}"
        raise ValueError(f"{name} must be > {minimum:g}{upper}, got {raw!r}")
    return value


def _int_env(name: str, *, minimum: int) -> int | None:
    raw = env_value(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {raw!r}")
    return value


def _rtol_mapping(value: object, label: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} rtol must be a mapping of observable class to tolerance")
    unknown = sorted(set(value) - set(OBSERVABLE_CLASSES))
    if unknown:
        raise ValueError(
            f"{label} rtol names unknown observable classes {unknown}; "
            f"known classes are {list(OBSERVABLE_CLASSES)}"
        )
    return {str(name): float(entry) for name, entry in value.items()}


def _window_block(value: object, label: str) -> dict[str, Any] | None:
    """Normalize a ``windows`` request: ``False`` disables, ``True`` takes defaults."""
    if value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} windows must be a bool or a mapping")
    unknown = sorted(set(value) - set(_WINDOW_KEYS))
    if unknown:
        raise ValueError(
            f"{label} windows names unknown keys {unknown}; known keys are {list(_WINDOW_KEYS)}"
        )
    samples = value.get("samples_per_feature", DEFAULT_WINDOW_SAMPLES_PER_FEATURE)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"{label} windows samples_per_feature must be an integer >= 1")
    tail = float(value.get("tail_widths", DEFAULT_WINDOW_TAIL_WIDTHS))
    if not math.isfinite(tail) or tail <= 0.0:
        # zero leaves a single-valued reflection population an empty window
        raise ValueError(f"{label} windows tail_widths must be finite and positive")
    requested = value.get("providers", DEFAULT_WINDOW_PROVIDERS)
    if isinstance(requested, str):
        raise TypeError(f"{label} windows providers must be a sequence of provider names")
    providers = tuple(requested)
    if (
        not providers
        or any(not isinstance(name, str) or not name for name in providers)
        or len(set(providers)) != len(providers)
    ):
        raise ValueError(f"{label} windows providers must be unique non-empty names")
    return {
        "policy": WINDOW_POLICY,
        "providers": providers,
        "samples_per_feature": samples,
        "tail_widths": tail,
    }


def _quadrature(value: object, label: str) -> str:
    if value not in LINE_QUADRATURES:
        raise ValueError(
            f"{label} quadrature must be one of {list(LINE_QUADRATURES)}, got {value!r}"
        )
    return str(value)


def line_quadrature_from_payload(payload: Mapping[str, Any] | None) -> str:
    """The line quadrature a policy payload selects; ``node`` when absent.

    Validation: sinc-bin-integration
    """
    if payload is None:
        return DEFAULT_LINE_QUADRATURE
    return _quadrature(payload.get("quadrature", DEFAULT_LINE_QUADRATURE), "policy payload")


def resolve_line_grid_policy(
    *,
    start_eV: float,
    stop_eV: float,
    bandwidth_policy: str = AUTOMATIC_BANDWIDTH_POLICY,
    per_call: Mapping[str, object] | None = None,
    stored: Mapping[str, object] | None = None,
) -> LineGridPolicy:
    """Resolve line-grid policy through the standard precedence chain.

    Precedence, highest first: explicit per-call/API value, ``PYRITE_*``
    environment, stored configuration (the catalog/profile artifact), built-in
    default. This mirrors ``cli._config.resolve`` without importing it -- #101
    forbids coupling the resolver to the CLI, which #64 may move.

    Per-observable tolerances are kept separate. ``PYRITE_ENERGY_GRID_RTOL`` is
    the global *fallback* for observables with no override; a per-observable
    ``PYRITE_ENERGY_GRID_RTOL_<OBSERVABLE>`` outranks it.
    """
    per_call = dict(per_call or {})
    stored = dict(stored or {})
    sources: dict[str, str] = {}

    def _resolve(key: str, env_name: str, env_reader, default):
        if key in per_call and per_call[key] is not None:
            sources[key] = "per-call"
            return per_call[key]
        environment = env_reader(env_name)
        if environment is not None:
            sources[key] = env_name
            return environment
        if key in stored and stored[key] is not None:
            sources[key] = "stored configuration"
            return stored[key]
        sources[key] = "built-in default"
        return default

    max_spacing = float(
        _resolve(
            "max_spacing_eV",
            _ENV_MAX_SPACING,
            lambda name: _float_env(name, minimum=0.0),
            DEFAULT_MAX_SPACING_EV,
        )
    )
    ulps = float(
        _resolve(
            "backend_safety_ulps",
            _ENV_ULPS,
            lambda name: _float_env(name, minimum=0.0),
            DEFAULT_BACKEND_SAFETY_ULPS,
        )
    )
    max_points = int(
        _resolve(
            "max_points",
            _ENV_MAX_POINTS,
            lambda name: _int_env(name, minimum=2),
            DEFAULT_MAX_POINTS,
        )
    )

    global_rtol = _float_env(_ENV_RTOL, minimum=0.0, maximum=1.0)
    per_call_rtol = _rtol_mapping(per_call.get("rtol"), "per-call")
    stored_rtol = _rtol_mapping(stored.get("rtol"), "stored")
    rtol: dict[str, float] = {}
    for observable in OBSERVABLE_CLASSES:
        key = f"rtol.{observable}"
        if observable in per_call_rtol:
            sources[key] = "per-call"
            rtol[observable] = float(per_call_rtol[observable])
            continue
        specific = _float_env(_env_rtol_name(observable), minimum=0.0, maximum=1.0)
        if specific is not None:
            sources[key] = _env_rtol_name(observable)
            rtol[observable] = specific
            continue
        if global_rtol is not None:
            sources[key] = _ENV_RTOL
            rtol[observable] = global_rtol
            continue
        if observable in stored_rtol:
            sources[key] = "stored configuration"
            rtol[observable] = float(stored_rtol[observable])
            continue
        sources[key] = "built-in default"
        rtol[observable] = DEFAULT_RTOL[observable]

    for observable, value in rtol.items():
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"rtol[{observable}] must lie in (0, 1), got {value!r}")
    if not math.isfinite(max_spacing) or max_spacing <= 0.0:
        raise ValueError("max_spacing_eV must be finite and positive")
    if not math.isfinite(stop_eV) or not math.isfinite(start_eV) or stop_eV <= start_eV:
        raise ValueError("line-grid stop must be finite and greater than start")

    # The sinc estimator's aliased-weight budget IS the accuracy knob: the
    # t_L^2-weighted fraction of features narrower than the step is a
    # first-order proxy for the relative spectral error the step commits, so
    # the budget is set to the governing (tightest) requested tolerance.
    strictest = min(rtol.values())
    sources["aliased_weight_limit"] = "derived from rtol"

    # Windows follow per-call > stored > built-in (off). No environment layer:
    # the PYRITE_ENERGY_GRID_* names would otherwise outrank stored rows for a
    # structural choice. A per-call False disables stored windows. The source
    # is recorded only when windows are on, keeping window-free payloads
    # unchanged.
    windows = None
    for layer, label in ((per_call, "per-call"), (stored, "stored configuration")):
        if layer.get("windows") is not None:
            windows = _window_block(layer["windows"], label)
            if windows is not None:
                sources["windows"] = label
            break

    # Quadrature follows the same per-call > stored > built-in order as windows,
    # and for the same reason has no environment layer. The source is recorded
    # only for a non-default choice, keeping ``node`` payloads unchanged.
    # Validation: sinc-bin-integration
    quadrature = DEFAULT_LINE_QUADRATURE
    for layer, label in ((per_call, "per-call"), (stored, "stored configuration")):
        if layer.get("quadrature") is not None:
            quadrature = _quadrature(layer["quadrature"], label)
            if quadrature != DEFAULT_LINE_QUADRATURE:
                sources["quadrature"] = label
            break

    # Bandwidth follows per-call > stored > the caller's argument, with no
    # environment layer, like windows and quadrature. The source is recorded
    # only for a non-default choice, keeping historical payloads unchanged.
    bandwidth_policy = str(bandwidth_policy)
    for layer, label in ((per_call, "per-call"), (stored, "stored configuration")):
        if layer.get("bandwidth") is not None:
            chosen = str(layer["bandwidth"])
            if chosen not in BANDWIDTH_POLICIES:
                raise ValueError(
                    f"{label} line-grid bandwidth must be one of {list(BANDWIDTH_POLICIES)}, "
                    f"got {chosen!r}"
                )
            if chosen != bandwidth_policy:
                sources["bandwidth"] = label
            bandwidth_policy = chosen
            break
    # Resolution follows the same per-call > stored order, with no environment
    # layer; the source is recorded only for a non-default choice.
    resolution_policy = AUTOMATIC_RESOLUTION_POLICY
    for layer, label in ((per_call, "per-call"), (stored, "stored configuration")):
        if layer.get("resolution") is not None:
            chosen = str(layer["resolution"])
            if chosen not in RESOLUTION_POLICIES:
                raise ValueError(
                    f"{label} line-grid resolution must be one of {list(RESOLUTION_POLICIES)}, "
                    f"got {chosen!r}"
                )
            if chosen != resolution_policy:
                sources["resolution"] = label
            resolution_policy = chosen
            break
    halo_limit = None
    if resolution_policy == LOCAL_RESOLUTION_POLICY:
        if bandwidth_policy != RESONANCE_BANDWIDTH_POLICY:
            raise ValueError(
                "the resonance-local resolution reads the measured resonance population "
                "and needs the resonance-population bandwidth"
            )
        halo_limit = DEFAULT_LOCAL_HALO_LIMIT
    truncation = None
    if bandwidth_policy == RESONANCE_BANDWIDTH_POLICY:
        if windows is not None:
            raise ValueError(
                "the resonance-population bandwidth takes its budget share from the "
                "feature-window row and cannot be combined with windows"
            )
        truncation = DEFAULT_BANDWIDTH_TRUNCATION

    return LineGridPolicy(
        bandwidth_policy=bandwidth_policy,
        start_eV=float(start_eV),
        stop_eV=float(stop_eV),
        resolution_policy=resolution_policy,
        max_spacing_eV=max_spacing,
        aliased_weight_limit=strictest,
        rtol=tuple(sorted(rtol.items())),
        backend_safety_ulps=ulps,
        max_points=max_points,
        sources=tuple(sorted(sources.items())),
        windows=None if windows is None else tuple(sorted(windows.items())),
        quadrature=quadrature,
        bandwidth_truncation=truncation,
        halo_limit=halo_limit,
    )


def resolved_coordinates(
    payload: Mapping[str, Any],
    target_spacing_eV: float,
    *,
    dtype=np.float32,
    stop_eV: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn a measured target spacing into the resolved uniform line grid.

    ``target_spacing_eV`` comes from the sinc Nyquist estimator run on the case's
    own trajectories. Failure to satisfy the point budget or the backend ULP
    floor raises :class:`LineGridToleranceError` with the unmet tolerance and an
    actionable correction; neither is ever met by coarsening.

    ``stop_eV`` replaces the payload's stop for a measured bandwidth; it must
    lie inside the payload's ``(start, stop]``, which is then a cap.
    """
    bandwidth = payload["bandwidth"]
    resolution = payload["resolution"]
    start = float(bandwidth["start_eV"])
    stop = float(bandwidth["stop_eV"])
    if stop_eV is not None:
        if not start < float(stop_eV) <= stop:
            raise ValueError(
                f"measured line-grid stop {stop_eV!r} eV lies outside ({start:g}, {stop:g}] eV"
            )
        stop = float(stop_eV)
    maximum_spacing = float(resolution["max_spacing_eV"])
    max_points = int(resolution["max_points"])
    safety_ulps = float(resolution["backend_safety_ulps"])
    rtol = dict(payload["rtol"])
    governing = min(rtol, key=lambda name: rtol[name])

    target = min(float(target_spacing_eV), maximum_spacing)
    if not math.isfinite(target) or target <= 0.0:
        raise LineGridToleranceError(
            "automatic line-grid resolution produced a non-positive target spacing; "
            "supply an explicit grid or set PYRITE_ENERGY_GRID_MAX_SPACING_EV"
        )
    num = resolution_num(start, stop, target)
    if num > max_points:
        raise LineGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r}: it needs {num} coordinates at {target:g} eV over "
            f"[{start:g}, {stop:g}] eV, above the {max_points}-point budget. Raise the "
            "budget with PYRITE_ENERGY_GRID_MAX_POINTS, relax the tolerance with "
            f"PYRITE_ENERGY_GRID_RTOL_{governing.upper()}, narrow the bandwidth, or "
            "supply an explicit grid. The grid is not coarsened automatically."
        )
    try:
        actual = validate_backend_spacing(start, stop, num, dtype=dtype, safety_ulps=safety_ulps)
    except ValueError as exc:
        raise LineGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r} in backend precision: {exc}"
        ) from None
    grid = np.linspace(start, stop, num)
    record = {
        "schema": int(payload.get("schema", LINE_GRID_POLICY_SCHEMA)),
        "bandwidth_policy": str(bandwidth["policy"]),
        "resolution_policy": str(resolution["policy"]),
        "start_eV": start,
        "stop_eV": stop,
        "num": int(num),
        "target_spacing_eV": float(target),
        "actual_spacing_eV": float(actual),
        "rtol": rtol,
        "governing_observable": governing,
        "sources": dict(payload.get("sources", {})),
    }
    return grid, record


def windowed_coordinates(
    payload: Mapping[str, Any],
    plan: WindowPlan,
    *,
    dtype=np.float32,
    stop_eV: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate a window plan against the policy budget and backend precision.

    The windowed counterpart of :func:`resolved_coordinates`. The plan must span
    the policy bandwidth on a backbone at the declared maximum spacing. A point
    budget or backend ULP shortfall raises :class:`LineGridToleranceError`; the
    plan is never coarsened. Precision is judged interval by interval, so a fine
    window low on the axis is held to the ulp at its own energy.

    A ``resonance-local`` resolution plans its own pieces without a windows
    block, and ``stop_eV`` then replaces the payload's measured-bandwidth cap.
    """
    bandwidth = payload["bandwidth"]
    resolution = payload["resolution"]
    windows = payload.get("windows")
    local = resolution["policy"] == LOCAL_RESOLUTION_POLICY
    if windows is None and not local:
        raise ValueError("line-grid policy payload carries no windows block")
    start = float(bandwidth["start_eV"])
    stop = float(bandwidth["stop_eV"])
    if stop_eV is not None:
        if not start < float(stop_eV) <= stop:
            raise ValueError(
                f"measured line-grid stop {stop_eV!r} eV lies outside ({start:g}, {stop:g}] eV"
            )
        stop = float(stop_eV)
    backbone = float(resolution["max_spacing_eV"])
    if (plan.start_eV, plan.stop_eV, plan.backbone_spacing_eV) != (start, stop, backbone):
        raise ValueError(
            "window plan bounds and backbone must equal the policy bandwidth and maximum spacing"
        )
    rtol = dict(payload["rtol"])
    governing = min(rtol, key=lambda name: rtol[name])
    max_points = int(resolution["max_points"])
    num = plan.num
    if num > max_points:
        raise LineGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r}: its {len(plan.seeds)} feature windows need {num} coordinates "
            f"over [{start:g}, {stop:g}] eV, above the {max_points}-point budget. Raise the "
            "budget with PYRITE_ENERGY_GRID_MAX_POINTS, relax the tolerance with "
            f"PYRITE_ENERGY_GRID_RTOL_{governing.upper()}, lower the windows' "
            "samples_per_feature, narrow the bandwidth, or supply an explicit grid. The grid "
            "is not coarsened automatically."
        )
    grid = plan.coordinates()
    try:
        smallest = validate_backend_coordinates(
            grid, dtype=dtype, safety_ulps=float(resolution["backend_safety_ulps"])
        )
    except ValueError as exc:
        raise LineGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r} in backend precision: {exc}"
        ) from None
    record = {
        "schema": int(payload.get("schema", WINDOWED_LINE_GRID_POLICY_SCHEMA)),
        "bandwidth_policy": str(bandwidth["policy"]),
        "resolution_policy": str(resolution["policy"]),
        "window_policy": str(resolution["policy"] if windows is None else windows["policy"]),
        "start_eV": start,
        "stop_eV": stop,
        "num": int(num),
        "backbone_spacing_eV": backbone,
        "min_spacing_eV": float(np.diff(grid).min()),
        "min_cast_spacing_eV": float(smallest),
        "rtol": rtol,
        "governing_observable": governing,
        "sources": dict(payload.get("sources", {})),
        "window_plan": plan.payload(),
    }
    if np.dtype(dtype) == np.float32:
        reach = max(
            (hi for _lo, hi, spacing in plan.pieces if spacing < backbone),
            default=0.0,
        )
        if reach >= FLOAT32_LINESHAPE_BINADE_EV:
            record["float32_lineshape_window_eV"] = float(reach)
    return grid, record


def coordinate_cache_key(payload: Mapping[str, Any], resolution_inputs: Mapping[str, Any]) -> str:
    """Content address for one automatically resolved grid.

    Keyed by the policy plus every case input the resolution depends on
    (material, geometry, beam, seed, electron count). Purely a speed cache: it
    never participates in case identity, so a cold and a warm cache produce the
    same result key.
    """
    encoded = json.dumps(
        {"policy": payload, "inputs": resolution_inputs},
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode()
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _cache_path(key: str):
    return cache_dir() / "line-grids" / key[:2] / f"{key}.json"


def cached_coordinates(key: str) -> dict[str, Any] | None:
    """Read one content-addressed resolved grid, or ``None``."""
    path = _cache_path(key)
    try:
        with path.open("rb") as stream:
            record = json.load(stream)
    except OSError, ValueError:
        return None
    if not isinstance(record, dict) or record.get("schema") not in (
        LINE_GRID_POLICY_SCHEMA,
        WINDOWED_LINE_GRID_POLICY_SCHEMA,
        QUADRATURE_LINE_GRID_POLICY_SCHEMA,
    ):
        return None
    return record


def store_coordinates(key: str, record: Mapping[str, Any]) -> None:
    """Write one content-addressed resolved grid; failures are not fatal.

    The cache lives under the user cache directory and never touches the
    selected profile or the material catalog: ``pyrite run`` must not mutate
    configuration as a side effect of resolving a grid.
    """
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(dict(record), stream, sort_keys=True)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise
    except OSError:
        return
