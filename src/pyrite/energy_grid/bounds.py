"""Pure helpers for pyrite.energy_grid.derive: turn a simulated
coherent-line spectrum into a coverage energy, and a coverage energy into a
catalog-ready line-grid ``stop``/``num`` pair.

See docs/adr/0005-energy-grid-schema-decisions.md.
"""

import numpy as np
from scipy.constants import physical_constants

_ELECTRON_REST_KEV = physical_constants["electron mass energy equivalent in MeV"][0] * 1.0e3


class CoverageGridTooNarrow(ValueError):
    """The ``coverage`` fraction is only reached in the final grid bin, so the
    diagnostic grid ceiling almost certainly clips the true emission tail and
    any reported bound would be silently pinned to the ceiling (issue_notes.md
    #1: no silent truncation without an explicit override)."""

    DEFAULT_MESSAGE = (
        "E_grid width is insufficient to support full X-ray spectral bandwidth. "
        "Either widen `--grid-stop`, or, if truncation/narrowband operation is "
        "acceptable, pass `allow_shortfall=True`."
    )

    def __init__(self, message=DEFAULT_MESSAGE):
        super().__init__(message)


def coverage_energy(
    E_grid: np.ndarray,
    spec: np.ndarray,
    coverage: float = 0.9,
    *,
    allow_shortfall: bool = False,
) -> float:
    """The smallest energy on ``E_grid`` at or below which the trapezoidal-
    integrated ``spec`` reaches ``coverage`` (0-1) of its total integral over
    ``E_grid``.

    Returns ``E_grid[0]`` when ``spec`` integrates to zero: a geometry with no
    emission intensity places no requirement on the grid width, and must not be
    mistaken for the widest requirement when candidates are ranked by this value.

    The integral only spans ``E_grid``, so any emission above ``E_grid[-1]`` is
    invisible: when ``coverage`` is not reached until the final bin, the true
    coverage energy lies beyond the ceiling and the returned value is a floor
    pinned to it, not a real bound. That case is detected as ``truncated`` and,
    unless ``allow_shortfall`` is set, is refused rather than reported silently.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if E_grid.size < 2:
        raise ValueError("E_grid must have at least two points")
    increments = np.diff(E_grid) * (spec[:-1] + spec[1:]) / 2.0
    total = increments.sum()
    if total <= 0.0:
        return float(E_grid[0])
    cumulative = np.cumsum(increments) / total
    index = int(np.searchsorted(cumulative, coverage))
    # `cumulative` is normalized to end at 1.0, so `index` can only reach the
    # last bin; landing there means `coverage` was not met until the ceiling.
    truncated = index >= cumulative.size - 1
    if truncated and not allow_shortfall:
        raise CoverageGridTooNarrow()
    index = min(index, cumulative.size - 1)
    return float(E_grid[index + 1])


def margined_stop(raw_energy_eV: float, margin: float = 0.15, round_to: float = 100.0) -> float:
    """Add a safety margin above ``raw_energy_eV`` and round up to the
    nearest ``round_to`` eV, so the catalog ``stop`` clears the measured
    coverage energy even for materials/geometries the empirical scan didn't
    sample exactly."""
    margined = raw_energy_eV * (1.0 + margin)
    return float(np.ceil(margined / round_to) * round_to)


def line_shift_fraction(
    energy_keV: float,
    energy_spread_frac: float,
    cos_theta_obs: float = 0.0,
) -> float:
    """Fractional line-energy shift a relative beam energy spread produces.

    The catalog's ``E_grid_line`` window is derived at the nominal case energy,
    so a beam with ``energy_spread_frac`` set emits a line that is displaced
    from where the grid was cut. This says by how much, in units of the line
    energy, so it can be compared against the ``margined_stop`` headroom.

    The PXR resonance (``montecarlo/spectrum/lines.py::_line_kin_core``) is
    ``omega = v.g / (1 - n.v)``, so at fixed reciprocal-lattice vector ``g`` and
    observation direction ``n``, differentiating in ``beta`` at
    ``n.v = beta cos(theta_obs)`` gives

    ``domega/omega = (dbeta/beta) / (1 - beta cos(theta_obs))``

    and the beam's *energy* deviation ``delta = dT/T`` enters through
    ``gamma = 1 + T/m_e c^2``, ``dbeta/beta = (gamma - 1) delta / (gamma^3 beta^2)``.
    The two combine into ``domega/omega = S delta`` with the returned ``S``
    factored out.

    Assumptions: first order in ``delta`` (the resonance is not linear in beta,
    so this overestimates slightly at large spread), and a fixed emission
    direction -- the spread is taken to move the line, not to redistribute it
    over angle.

    Limiting cases: ``delta -> 0`` gives no shift; the nonrelativistic
    ``gamma -> 1`` limit gives ``S -> delta/2`` (``omega ~ beta``, and
    ``beta ~ sqrt(T)``); at 90 degrees observation ``cos_theta_obs = 0`` drops
    the Doppler denominator entirely.

    Validation: beam-energy-spread-injection
    """
    gamma = 1.0 + float(energy_keV) / _ELECTRON_REST_KEV
    beta_sq = 1.0 - 1.0 / gamma**2
    doppler = 1.0 - np.sqrt(beta_sq) * float(cos_theta_obs)
    sensitivity = (gamma - 1.0) / (gamma**3 * beta_sq * doppler)
    return float(abs(sensitivity * float(energy_spread_frac)))


def spacing_num(start_eV: float, stop_eV: float, target_spacing_eV: float = 3.0) -> int:
    """The endpoint-inclusive ``num`` for ``linspace(start_eV, stop_eV, num)``
    closest to ``target_spacing_eV`` uniform spacing -- the convention already
    used by every ``E_grid_line_by_energy`` row in materials.toml."""
    return int(round((stop_eV - start_eV) / target_spacing_eV)) + 1


def resolution_num(start_eV: float, stop_eV: float, maximum_spacing_eV: float) -> int:
    """Endpoint-inclusive count whose actual spacing never exceeds the limit."""
    values = (float(start_eV), float(stop_eV), float(maximum_spacing_eV))
    if not all(np.isfinite(value) for value in values) or maximum_spacing_eV <= 0.0:
        raise ValueError("grid bounds and maximum spacing must be finite; spacing must be positive")
    if stop_eV <= start_eV:
        raise ValueError("grid stop must be greater than start")
    return int(np.ceil((stop_eV - start_eV) / maximum_spacing_eV)) + 1


def validate_backend_spacing(
    start_eV: float,
    stop_eV: float,
    num: int,
    *,
    dtype=np.float32,
    safety_ulps: float = 8.0,
) -> float:
    """Validate a generated linspace against backend coordinate precision."""
    if not np.isfinite(safety_ulps) or safety_ulps <= 0.0:
        raise ValueError("backend ULP safety factor must be finite and positive")
    if int(num) != num or num < 2:
        raise ValueError("grid num must be an integer of at least 2")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise ValueError("backend grid dtype must be floating point")
    spacing = (float(stop_eV) - float(start_eV)) / (int(num) - 1)
    largest = resolved_dtype.type(max(abs(float(start_eV)), abs(float(stop_eV))))
    minimum = float(safety_ulps) * abs(float(np.spacing(largest)))
    if spacing < minimum:
        raise ValueError(
            f"requested line-grid spacing {spacing:g} eV is below the {safety_ulps:g}-ULP "
            f"safety floor {minimum:g} eV for {resolved_dtype.name} at {float(largest):g} eV; "
            "run with PYRITE_FP64=1 or choose a physically justified coarser tolerance"
        )
    cast = np.linspace(start_eV, stop_eV, int(num), dtype=float).astype(resolved_dtype)
    cast_steps = np.diff(cast)
    collapsed = int(np.count_nonzero(cast_steps <= 0.0))
    if collapsed:
        raise ValueError(
            f"generated line grid is not strictly increasing after {resolved_dtype.name} cast "
            f"({collapsed} collapsed intervals); run with PYRITE_FP64=1 or choose a physically "
            "justified coarser tolerance"
        )
    minimum_cast_step = float(cast_steps.min())
    if minimum_cast_step < minimum:
        raise ValueError(
            f"generated line-grid intervals reach {minimum_cast_step:g} eV after "
            f"{resolved_dtype.name} cast, below the {safety_ulps:g}-ULP safety floor "
            f"{minimum:g} eV; run with PYRITE_FP64=1 or choose a physically justified "
            "coarser tolerance"
        )
    return spacing


def line_start_eV(energy_keV: float) -> float:
    """Catalog convention for a line-grid row's ``start_eV`` when none is
    already recorded: 10 eV floor at <=60 keV beam energy, 50 eV above."""
    return 10.0 if float(energy_keV) <= 60.0 else 50.0
