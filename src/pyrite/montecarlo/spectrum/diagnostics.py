"""Radiation-side error estimators for frozen-energy segment evaluation.

The line and bremsstrahlung kernels evaluate every per-flight emission
coefficient at the flight's START energy (``E_keV``). These host-side
estimators bound the error that choice introduces, per physical flight, before
any spectrum is computed:

- :func:`cxr_endpoint_resonance_drift` -- how far a flight's CXR resonance
  energy ``E_res = hbar_c * v.g / (1 - v.n)`` sweeps between its endpoints,
  in units of the flight's own sinc linewidth. A drift of order one linewidth
  means the frozen-energy line is smeared across its own width.
- :func:`brem_endpoint_quadrature_error` -- the relative difference between
  the kernel's left-endpoint evaluation of the per-flight bremsstrahlung
  integral and a midpoint evaluation, per spectral grid.

Both read ``E_end_keV`` when transport supplied it (``energy_model="midpoint"``)
and otherwise predict the end energy with the same left-endpoint Joy--Luo
stopping rule the transport diagnostics use, so they apply to frozen runs
unchanged. They are opt-in, host-only passes: calling them is the only path
that downloads resident device segment fields, mirroring
``simulate_trajectories(collect_diagnostics=True)``.

Warning thresholds are calibration constants measured in
``docs/validation/beam-transport/radiation-error-estimators.md`` -- observed
convergence of the actual spectra, not a priori targets. Each function warns
when the p99 of its primary metric exceeds the threshold.

Validation: radiation-error-estimators
"""

import warnings

import numpy as np

from ...materials.attenuation import _normalize_composition
from ...materials.crystal import HBARC_EV_ANG
from .._backend import _to_cpu
from ..transport import TRANSPORT_ELEMENTS, _percentile_summary, beta_from_keV
from .brem import _brem_dsigma_dk
from .lines import _observation_direction

# Measured calibration defaults; see the validation doc named above. Drift is
# in units of the flight's sinc half-width to its first zero; the brem error
# is the grid-integrated relative yield difference. Drift: 1% of flights
# sweeping more than one full linewidth corresponded to a 4--10% line-region
# spectral change. Quadrature: the measured actual grid-L1 change ran at
# ~0.4x the p99 estimate, so 1e-2 flags a ~0.4% continuum change, above every
# measured case (p99 <= 8.6e-3).
DEFAULT_RESONANCE_DRIFT_WARN = 1.0
DEFAULT_BREM_QUADRATURE_WARN = 1.0e-2


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def _stopping_keV_per_ang(E_keV, composition):
    """Joy--Luo CSDA stopping power [keV/Ang] (negative), matching transport."""
    total = np.zeros_like(E_keV)
    for element, n_i in composition:
        params = TRANSPORT_ELEMENTS[element]
        Z = params["Z"]
        J = params["J_keV"]
        k = 0.731 + 0.0688 * np.log10(float(Z))
        total += (n_i / 0.602214076) * float(Z) * np.log(1.166 * (E_keV + k * J) / J)
    return -7.85e-4 * total / E_keV


def _flight_E_end_keV(segments, composition, layers):
    """Flight end energies: transported when present, left-endpoint predicted else.

    The prediction reapplies the transport core's frozen stopping rule, the
    same rule ``_flight_diagnostic_summary`` uses, so a frozen run's estimator
    input is exactly the end state its own propagation rule implies.
    """
    if segments.get("E_end_keV") is not None:
        return _host(segments["E_end_keV"]).astype(float, copy=False)
    if composition is None and layers is None:
        raise ValueError(
            "segments carry no E_end_keV; pass composition (or layers) so the "
            "end energy can be predicted with the frozen stopping rule"
        )
    E_start = _host(segments["E_keV"]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    if layers is None:
        # One composition applies to every row, whatever the layer field says.
        return E_start + _stopping_keV_per_ang(E_start, composition) * length
    layer_index = _host(segments["layer"]).astype(np.int64, copy=False)
    E_end = np.empty_like(E_start)
    for index, comp in enumerate([item[2] for item in layers]):
        mask = layer_index == index
        E_end[mask] = E_start[mask] + _stopping_keV_per_ang(E_start[mask], comp) * length[mask]
    return E_end


def cxr_endpoint_resonance_drift(
    segments,
    g_vectors,
    n_hat=None,
    theta_obs_rad=np.deg2rad(119.0),
    composition=None,
    layers=None,
    warn_threshold=None,
):
    """Per-flight CXR resonance drift between the flight's endpoints.

    The line kernel freezes ``v = beta(E_start) * v_hat`` for the whole flight,
    so each (flight, reflection) radiates at one resonance energy
    ``E_res = HBARC_EV_ANG * v.g / (1 - v.n)``. The decelerating electron's
    true resonance sweeps; this estimator evaluates ``E_res`` at both endpoint
    speeds and reports the sweep in units of the flight's sinc linewidth --
    the half-width to the first zero of the kernel's finite-interaction-time
    factor ``t_L sinc(a_width (E - E_res)/pi)``,

    ``W = 2 pi HBARC_EV_ANG / (dnm * t_L)``,  ``t_L = L / beta(E_start)``.

    ``g_vectors``: (G, 3) lab-frame reciprocal vectors [1/Ang], the oriented
    set the caller's ``mc_spectrum`` run sums over. The per-flight metric is
    the worst drift over G. Flights whose start-energy resonance falls below
    the kernel's hard 10 eV floor are excluded (they radiate nothing).

    Returns fixed-size percentile summaries (p50/p90/p99/max) of the
    per-flight worst drift in linewidths and in eV, and warns when the p99
    drift exceeds ``warn_threshold`` (default
    ``DEFAULT_RESONANCE_DRIFT_WARN``, calibrated in
    ``docs/validation/beam-transport/radiation-error-estimators.md``).

    Validation: radiation-error-estimators
    """
    if warn_threshold is None:
        warn_threshold = DEFAULT_RESONANCE_DRIFT_WARN
    E_start = _host(segments["E_keV"]).astype(float, copy=False)
    E_end = _flight_E_end_keV(segments, composition, layers)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    v_hat = _host(segments["v_hat"]).astype(float, copy=False)
    g = np.asarray(g_vectors, dtype=float).reshape(-1, 3)
    n_hat = _observation_direction(theta_obs_rad, n_hat)

    beta_start = beta_from_keV(E_start)
    beta_end = beta_from_keV(E_end)
    # (M, G) projection factors; each endpoint's resonance uses its own
    # Doppler denominator, while the kernel's frozen line uses the start one.
    c1 = v_hat @ g.T
    c2 = v_hat @ n_hat
    dnm = 1.0 - beta_start * c2
    omega_start = (beta_start[:, None] * c1) / dnm[:, None]
    omega_end = (beta_end[:, None] * c1) / (1.0 - beta_end * c2)[:, None]
    radiating = (HBARC_EV_ANG * omega_start) > 10.0

    t_L = length / beta_start
    width = 2.0 * np.pi * HBARC_EV_ANG / (dnm * t_L)  # (M,) half-width [eV]
    drift_eV = np.where(
        radiating, HBARC_EV_ANG * np.abs(omega_end - omega_start), -np.inf
    ).max(axis=1)
    drift_linewidths = np.where(
        radiating,
        HBARC_EV_ANG * np.abs(omega_end - omega_start) / width[:, None],
        -np.inf,
    ).max(axis=1)
    valid = drift_eV > -np.inf
    n_flights = int(valid.sum())
    drift_eV = drift_eV[valid]
    drift_linewidths = drift_linewidths[valid]

    summary = _percentile_summary(drift_linewidths)
    warned = summary["p99"] is not None and summary["p99"] > warn_threshold
    if warned:
        warnings.warn(
            f"CXR endpoint resonance drift p99 {summary['p99']:.3g} linewidths "
            f"exceeds the calibrated warning threshold {warn_threshold:.3g}; "
            "the frozen-energy line kernel smears these flights across their "
            "own width",
            stacklevel=2,
        )
    return {
        "n_flights": n_flights,
        "resonance_drift_linewidths": summary,
        "resonance_drift_eV": _percentile_summary(drift_eV),
        "warn_threshold": float(warn_threshold),
        "warned": bool(warned),
    }


def brem_endpoint_quadrature_error(
    segments,
    E_grid_eV,
    element=None,
    n_atoms_per_ang3=None,
    composition=None,
    layers=None,
    warn_threshold=None,
    chunk=8192,
):
    """Per-flight bremsstrahlung left-endpoint versus midpoint quadrature error.

    The brem kernel integrates each flight's emission as
    ``n * dsigma/dk(T_start) * L`` -- a left-endpoint rectangle rule in the
    electron energy. This estimator re-evaluates the same Bethe-Heitler/Elwert
    integrand at the flight's midpoint energy and reports, per flight, the
    grid-integrated relative yield difference and the worst per-bin relative
    difference (bins below 1e-6 of the flight's peak bin excluded: the escape
    factor and ``n * L`` cancel in every ratio). Flights contribute per their
    own layer's composition.

    Returns fixed-size percentile summaries plus a warning when the p99
    integrated error exceeds ``warn_threshold`` (default
    ``DEFAULT_BREM_QUADRATURE_WARN``, calibrated in
    ``docs/validation/beam-transport/radiation-error-estimators.md``).

    Validation: radiation-error-estimators
    """
    if warn_threshold is None:
        warn_threshold = DEFAULT_BREM_QUADRATURE_WARN
    if layers is None:
        compositions = [_normalize_composition(element, n_atoms_per_ang3, composition)]
    else:
        compositions = [item[2] for item in layers]
    E_start = _host(segments["E_keV"]).astype(float, copy=False)
    E_end = _flight_E_end_keV(
        segments, None if layers is not None else compositions[0], layers
    )
    layer_index = _host(segments["layer"]).astype(np.int64, copy=False)
    E_grid = np.asarray(E_grid_eV, dtype=float)
    # Trapezoid weights turn the per-bin densities into an integrated yield.
    weights = np.empty_like(E_grid)
    weights[1:-1] = 0.5 * (E_grid[2:] - E_grid[:-2])
    weights[0] = 0.5 * (E_grid[1] - E_grid[0])
    weights[-1] = 0.5 * (E_grid[-1] - E_grid[-2])

    n_flights = E_start.size
    integrated = np.zeros(n_flights, dtype=float)
    max_bin = np.zeros(n_flights, dtype=float)
    for lo in range(0, n_flights, chunk):
        hi = min(lo + chunk, n_flights)
        T_start = E_start[lo:hi]
        T_mid = 0.5 * (E_start[lo:hi] + E_end[lo:hi])
        # Emission is additive over elements, each with its own Z^2 weight;
        # the per-layer composition only reweights the same sum.
        y_start = np.zeros((hi - lo, E_grid.size), dtype=float)
        y_mid = np.zeros((hi - lo, E_grid.size), dtype=float)
        for index, layer_comp in enumerate(compositions):
            mask = layer_index[lo:hi] == index
            if not mask.any():
                continue
            ys = np.zeros((int(mask.sum()), E_grid.size), dtype=float)
            ym = np.zeros((int(mask.sum()), E_grid.size), dtype=float)
            for element_i, n_i in layer_comp:
                Z = TRANSPORT_ELEMENTS[element_i]["Z"]
                # _to_cpu keeps the estimator host-only even when the global
                # backend has xp on a device.
                ys += n_i * _to_cpu(_brem_dsigma_dk(Z, T_start[mask], E_grid))
                ym += n_i * _to_cpu(_brem_dsigma_dk(Z, T_mid[mask], E_grid))
            y_start[mask] = ys
            y_mid[mask] = ym
        diff = np.abs(y_start - y_mid)
        integrated[lo:hi] = (diff * weights[None, :]).sum(axis=1) / (
            (y_mid * weights[None, :]).sum(axis=1) + 1e-300
        )
        significant = y_mid >= 1e-6 * y_mid.max(axis=1, keepdims=True)
        max_bin[lo:hi] = np.where(
            significant & (y_mid > 0.0), diff / (y_mid + 1e-300), 0.0
        ).max(axis=1)

    summary = _percentile_summary(integrated)
    warned = summary["p99"] is not None and summary["p99"] > warn_threshold
    if warned:
        warnings.warn(
            f"bremsstrahlung endpoint quadrature p99 relative error "
            f"{summary['p99']:.3g} exceeds the calibrated warning threshold "
            f"{warn_threshold:.3g}; the left-endpoint rule misestimates the "
            "per-flight continuum integral",
            stacklevel=2,
        )
    return {
        "n_flights": int(n_flights),
        "integrated_relative_error": summary,
        "max_bin_relative_error": _percentile_summary(max_bin),
        "warn_threshold": float(warn_threshold),
        "warned": bool(warned),
    }
