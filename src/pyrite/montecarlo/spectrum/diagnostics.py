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
and otherwise predict the end energy with the same left-endpoint stopping rule
the transport diagnostics use, so they apply to frozen runs
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

from ..._backend import _to_cpu
from ...materials.attenuation import _normalize_composition
from ...materials.crystal import HBARC_EV_ANG
from ..transport import (
    _percentile_summary,
    beta_from_keV,
    spliced_stopping_keV_per_ang,
)
from .brem import BremsstrahlungModel, _bremsstrahlung_dsigma_dk
from .lines import _SEG_ARRAYS, _observation_direction

# Measured calibration defaults; see the validation doc named above. Drift is
# in units of the flight's sinc half-width to its first zero; the brem error
# is the grid-integrated relative yield difference. Drift: 1% of flights
# sweeping more than one full linewidth corresponded to a 4--10% line-region
# spectral change. Quadrature: the measured actual grid-L1 change ran at
# ~0.4x the p99 estimate, so 1e-2 flags a ~0.4% continuum change, above every
# measured case (p99 <= 8.6e-3).
DEFAULT_RESONANCE_DRIFT_WARN = 1.0
DEFAULT_BREM_QUADRATURE_WARN = 1.0e-2


def sinc_feature_spacing(
    segments,
    n_hat,
    *,
    electron_limit=None,
    aliased_weight_limit=0.01,
):
    """Largest uniform step resolving all but a weighted tail of sinc features.

    The line kernels use ``sinc(a_width * (E - E_res) / pi)**2`` with
    ``a_width = (1 - v.n) * t_L / (2 HBARC_EV_ANG)``. Thus the Nyquist step
    to the first zero is ``pi / a_width``. Select its lower weighted quantile,
    using the kernel's leading ``t_L**2`` intensity factor as an inexpensive
    pre-spectrum proxy. This is policy derived from the existing line model,
    not a new radiation equation.

    Returns ``(step_eV, aliased_weight_fraction, n_segments)``. Invalid
    kinematics are ignored; an empty valid population fails closed.
    """
    if not 0.0 <= float(aliased_weight_limit) < 1.0:
        raise ValueError("aliased_weight_limit must be in [0, 1)")
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energy = _host(segments[energy_field]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    direction = _host(segments["v_hat"]).astype(float, copy=False)
    if electron_limit is not None:
        line = _host(segments["elec_id"]) < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]

    beta = beta_from_keV(energy)
    t_L = length / beta
    denominator = 1.0 - beta * (direction @ np.asarray(n_hat, dtype=float))
    width = 2.0 * np.pi * HBARC_EV_ANG / (denominator * t_L)
    weight = t_L * t_L
    valid = (
        np.isfinite(width)
        & np.isfinite(weight)
        & (width > 0.0)
        & (weight > 0.0)
        & (denominator > 0.0)
    )
    width, weight = width[valid], weight[valid]
    if width.size == 0:
        raise ValueError("cannot derive sinc spacing: diagnostic transport has no valid segments")

    order = np.argsort(width, kind="stable")
    width, weight = width[order], weight[order]
    cumulative = np.cumsum(weight)
    total = float(cumulative[-1])
    index = int(np.searchsorted(cumulative, float(aliased_weight_limit) * total, side="left"))
    step = float(width[min(index, width.size - 1)])
    aliased = float(weight[width < step].sum() / total)
    return step, aliased, int(width.size)


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def _stopping_keV_per_ang(E_keV, composition):
    """CSDA stopping power [keV/Ang] (negative), matching transport exactly.

    The frozen-rule replay below is only meaningful if it reapplies the *same*
    rule the cores used, so this delegates rather than restating the model.
    """
    return spliced_stopping_keV_per_ang(composition, E_keV)


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


def subdivide_flights(segments, composition=None, layers=None, max_dE_frac=0.0, max_substeps=64):
    """Split each physical flight into numerical substeps at fixed geometry.

    The instrument for the substep-invariance contract: a physical flight is a
    straight constant-direction ray, so a substep's midpoint and length follow
    exactly from its parent's without any transport state. Only the numerical
    sampling of the energy and clock along the flight refines. That is what
    separates this from re-running transport at a tighter ``max_dE_frac``, which
    also moves the sampled collision points and so decorrelates the trajectories
    (``docs/validation/beam-transport/energy-controlled-propagation.md``).

    Each substep chain reproduces the transport core's explicit midpoint RK2:
    predict from ``E_start``, evaluate stopping at
    ``(E_start + E_pred)/2``, and advance the clock by
    ``ds / beta(E_repr)``. The emitted rows carry the same
    ``E_start_keV``/``E_end_keV``/``E_repr_keV``/``t_start_ang``/``t_end_ang``
    schema an energy-controlled transport run would have produced for the same
    flights.

    ``max_dE_frac=0.0`` returns one row per flight, i.e. the caller's input
    row set with a rebuilt (and identical) end state. Substep counts are
    ``ceil(|dE|/E / max_dE_frac)`` per flight, clipped at ``max_substeps``, and
    substeps are equal in LENGTH rather than in energy loss -- a uniform
    refinement of the same integral, not a replay of the core's step control.

    Returns ``(rows, parent)`` where ``parent[i]`` is the input row index that
    emitted output row ``i``.

    Validation: substep-radiation-invariance
    """
    if max_dE_frac < 0.0:
        raise ValueError("max_dE_frac must be >= 0")
    if max_substeps < 1:
        raise ValueError("max_substeps must be >= 1")
    compositions = [composition] if layers is None else [item[2] for item in layers]
    if compositions[0] is None:
        raise ValueError("pass composition (or layers) so the substep chain can be integrated")

    E_start = _host(segments["E_keV"]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    v_hat = _host(segments["v_hat"]).astype(float, copy=False)
    r_mid = _host(segments["r_mid"]).astype(float, copy=False)
    t_start = _host(segments["t_ang"]).astype(float, copy=False)
    layer_index = _host(segments["layer"]).astype(np.int64, copy=False)
    n_rows = E_start.size

    def _stopping(E_keV, rows):
        """Layer-resolved stopping for the given row selection."""
        if layers is None:
            return _stopping_keV_per_ang(E_keV, compositions[0])
        out = np.empty_like(E_keV)
        for index, comp in enumerate(compositions):
            mask = layer_index[rows] == index
            if mask.any():
                out[mask] = _stopping_keV_per_ang(E_keV[mask], comp)
        return out

    all_rows = np.arange(n_rows)
    if max_dE_frac == 0.0:
        n_sub = np.ones(n_rows, dtype=np.int64)
    else:
        loss = np.maximum(0.0, -_stopping(E_start, all_rows) * length / E_start)
        n_sub = np.minimum(
            np.maximum(1.0, np.ceil(loss / max_dE_frac)), float(max_substeps)
        ).astype(np.int64)

    offsets = np.concatenate(([0], np.cumsum(n_sub)[:-1]))
    total = int(n_sub.sum())
    parent = np.repeat(all_rows, n_sub)
    within = np.arange(total) - np.repeat(offsets, n_sub)

    step_len = length / n_sub
    r_entry = r_mid - 0.5 * length[:, None] * v_hat
    sub_r_mid = r_entry[parent] + v_hat[parent] * ((within + 0.5) * step_len[parent])[:, None]

    sub_E_start = np.empty(total)
    sub_E_end = np.empty(total)
    sub_t_start = np.empty(total)
    sub_t_end = np.empty(total)
    E_cur = E_start.copy()
    t_cur = t_start.copy()
    for k in range(int(n_sub.max()) if n_rows else 0):
        active = np.nonzero(n_sub > k)[0]
        slot = offsets[active] + k
        E_here = E_cur[active]
        ds = step_len[active]
        E_pred = E_here + _stopping(E_here, active) * ds
        E_next = E_here + _stopping(0.5 * (E_here + E_pred), active) * ds
        E_repr = 0.5 * (E_here + E_next)
        sub_E_start[slot] = E_here
        sub_E_end[slot] = E_next
        sub_t_start[slot] = t_cur[active]
        t_cur[active] += ds / beta_from_keV(E_repr)
        sub_t_end[slot] = t_cur[active]
        E_cur[active] = E_next

    flight_id = (
        _host(segments["flight_id"]).astype(np.int64, copy=False)
        if segments.get("flight_id") is not None
        else _flight_index_within_electron(_host(segments["elec_id"]))
    )
    elec_id = _host(segments["elec_id"]).astype(np.int64, copy=False)
    # Every per-row array is rebuilt; scalars and per-electron arrays pass
    # through. A parent-length row array left in place would survive a row mask
    # at the wrong length and silently desynchronize.
    out = {key: value for key, value in segments.items() if key not in _SEG_ARRAYS}
    out.update(
        {
            "r_mid": sub_r_mid,
            "v_hat": v_hat[parent],
            "L_ang": step_len[parent],
            "E_keV": sub_E_start,
            "E_start_keV": sub_E_start,
            "E_end_keV": sub_E_end,
            "E_repr_keV": 0.5 * (sub_E_start + sub_E_end),
            "t_ang": sub_t_start,
            "t_start_ang": sub_t_start,
            "t_end_ang": sub_t_end,
            "t0_ang": _host(segments["t0_ang"]).astype(float, copy=False)[parent],
            "elec_id": elec_id[parent],
            "electron_id": elec_id[parent],
            "flight_id": flight_id[parent],
            "substep_id": within,
            "layer": layer_index[parent],
        }
    )
    return out, parent


def _flight_index_within_electron(elec_id):
    """Zero-based monotonic flight index per electron for unlabelled rows.

    Transport emits rows grouped by electron and ordered along each trajectory,
    which is the ordering this reproduces. Only used when the producer predates
    ``flight_id`` -- an energy-controlled run supplies its own.
    """
    elec_id = np.asarray(elec_id).astype(np.int64, copy=False)
    index = np.zeros(elec_id.size, dtype=np.int64)
    if elec_id.size:
        new_electron = np.empty(elec_id.size, dtype=bool)
        new_electron[0] = True
        new_electron[1:] = elec_id[1:] != elec_id[:-1]
        run_start = np.maximum.accumulate(np.where(new_electron, np.arange(elec_id.size), 0))
        index = np.arange(elec_id.size) - run_start
    return index


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
    drift_eV = np.where(radiating, HBARC_EV_ANG * np.abs(omega_end - omega_start), -np.inf).max(
        axis=1
    )
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
    cross_section_model: BremsstrahlungModel = "eedl",
):
    """Per-flight bremsstrahlung left-endpoint versus midpoint quadrature error.

    The brem kernel integrates each flight's emission as
    ``n * dsigma/dk(T_start) * L`` -- a left-endpoint rectangle rule in the
    electron energy. This estimator re-evaluates the same Bethe-Heitler/Elwert
    integrand at the flight's midpoint energy and reports, per flight, the
    configured cross-section model at the flight's midpoint energy and reports,
    per flight, the grid-integrated relative yield difference and the worst per-bin relative
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
    E_end = _flight_E_end_keV(segments, None if layers is not None else compositions[0], layers)
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
                # _to_cpu keeps the estimator host-only even when the global
                # backend has xp on a device.
                ys += n_i * _to_cpu(
                    _bremsstrahlung_dsigma_dk(
                        element_i,
                        T_start[mask],
                        E_grid,
                        cross_section_model=cross_section_model,
                    )
                )
                ym += n_i * _to_cpu(
                    _bremsstrahlung_dsigma_dk(
                        element_i,
                        T_mid[mask],
                        E_grid,
                        cross_section_model=cross_section_model,
                    )
                )
            y_start[mask] = ys
            y_mid[mask] = ym
        diff = np.abs(y_start - y_mid)
        integrated[lo:hi] = (diff * weights[None, :]).sum(axis=1) / (
            (y_mid * weights[None, :]).sum(axis=1) + 1e-300
        )
        significant = y_mid >= 1e-6 * y_mid.max(axis=1, keepdims=True)
        max_bin[lo:hi] = np.where(significant & (y_mid > 0.0), diff / (y_mid + 1e-300), 0.0).max(
            axis=1
        )

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
