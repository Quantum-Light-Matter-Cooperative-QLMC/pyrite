"""Bounded optical TR/CTR: wiring, source contract, form factors and limits (#615).

Validation: optical-planar-tr, optical-ctr-population
"""

import dataclasses

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from pyrite.materials.optical import gold_rakic_response
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.runner.transition_emission import (
    case_layer_eps,
    case_optical_transition,
)
from pyrite.montecarlo.spectrum.coherent_population import (
    CoherentSamplingError,
    CoherentSamplingWarning,
)
from pyrite.montecarlo.spectrum.optical_transition import (
    C_ANG_PER_FS,
    OPTICAL_TRANSITION_MODEL,
    OpticalBunchSource,
    gaussian_coherent_band_eV,
    load_optical_transition,
    optical_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation import beta_from_kinetic_eV
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    BoundaryCrossings,
    _take,
    boundary_crossings,
    crossing_transition_radiation,
)

_AU = gold_rakic_response()
_HOPG = CATALOG.crystal("hopg")
_ENERGY = np.array([1.4, 1.9, 2.5, 3.1])
_BACK = np.array([np.sin(0.9), 0.0, -np.cos(0.9)])
_GRID = (100.0, 2000.0, 20.0)


def _response(eps, medium="matched"):
    nodes = np.array([0.1, 10.0])
    return dataclasses.replace(
        _AU, medium=medium, segments=((nodes, np.full(2, eps, dtype=complex)),)
    )


@pytest.fixture(scope="module")
def transported():
    # Plumbing only: HOPG transport tables (no Au transport tables) with Au optics.
    fields = dict(
        name="optical wiring", crystal="hopg", composition=_HOPG.composition,
        hkl_list=[(0, 0, 2)], B_ang2=0.8, E0_keV=30.0, thickness_ang=5e4,
        crystal_width_mm=None, crystal_height_mm=None, beam_fwhm_mm=None, surface_hkl=None,
        E_grid=_GRID, E_grid_line=_GRID, E_grid_brem=(0.0, 500.0, 50.0),
        theta_obs_rad=0.3, dtheta_obs_rad=np.deg2rad(2.0), tilt_deg=0.0, tilt_azim_deg=0.0,
        domega_sr=1e-4, beam_uvw=(0, 0, 1), mosaic_fwhm_rad=None, mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1, abs_layers=None, layer_radiators=None, brem_file=None, Ne=12,
        Ne_brem=8, seed=11, spec_chunk=None, brem_chunk=None, energy_model="midpoint",
        elastic_model="elsepa",
        transition_radiation={"model": "planar-stack", "coherence": "history"},
    )  # fmt: skip
    case = Case(**fields)
    tp = runner._transport_case(case)
    return case, dict(tp, E_grid=_ENERGY, n_hat=_BACK)


def _crossings(case, tp):
    crossings = boundary_crossings(tp["segs"], [(0.0, case.thickness_ang, case.composition)])
    return _take(crossings, np.flatnonzero(crossings.electron_id < tp["Ne_brem"]))


# --- Plan 1: wiring, normalization and provenance ---------------------------------


def test_case_wiring_matches_the_history_and_bunch_scorers(transported, tmp_path):
    case, tp = transported
    m = int(tp["Ne_brem"])
    eps = case_layer_eps(case, _ENERGY, layer_responses=(_AU,))
    mine = _crossings(case, tp)
    history = crossing_transition_radiation(mine, _ENERGY, _BACK, eps, n_electrons=m)
    one = OpticalBunchSource.sampled(1.602176634e-7, "one electron")  # N = 1
    single = case_optical_transition(
        case, tp, layer_responses=(_AU,), source=one, formation_tolerance=None
    )
    np.testing.assert_allclose(single.yield_per_electron, history, rtol=1e-12)
    np.testing.assert_allclose(single.self_yield, history, rtol=1e-12)
    # A realization is the plain coherent sum of the sampled bunch (N = M).
    real = OpticalBunchSource.realization(m * 1.602176634e-7, "eight transported electrons")
    bunch = case_optical_transition(
        case,
        tp,
        layer_responses=(_AU,),
        source=real,
        formation_tolerance=None,
        energy_budget=None,
    )
    plain = crossing_transition_radiation(
        mine, _ENERGY, _BACK, eps, n_electrons=m, coherence="bunch"
    )
    np.testing.assert_allclose(bunch.yield_per_electron, plain, rtol=1e-12)
    np.testing.assert_allclose(bunch.bunch_yield, m * plain, rtol=1e-12)
    meta = bunch.metadata
    assert meta["model"] == OPTICAL_TRANSITION_MODEL and meta["n_histories"] == m
    assert meta["materials"][0]["response_digest"] == _AU.digest
    assert meta["materials"][0]["dataset_id"] == _AU.dataset_id
    assert meta["source"]["law"] == "realization" and "emitting face" in meta["reference"]
    saved = load_optical_transition(bunch.save(tmp_path / "optical"))
    np.testing.assert_array_equal(saved.yield_per_electron, bunch.yield_per_electron)
    assert saved.metadata == meta


def test_case_wiring_refuses_unsupported_band_charge_and_footprint(transported):
    case, tp = transported
    source = OpticalBunchSource.sampled(1.0, "1 pC")
    with pytest.raises(ValueError, match="unsupported energy"):
        case_optical_transition(
            case, dict(tp, E_grid=np.array([0.1, 1.0])), layer_responses=(_AU,), source=source
        )
    with pytest.raises(ValueError, match="bunch_charge_pc"):
        case_optical_transition(
            dict(case, bunch_charge_pc=2.0), tp, layer_responses=(_AU,), source=source
        )
    crossings = dataclasses.replace(_crossings(case, tp), n_side_exits=1)
    with pytest.raises(ValueError, match="laterally infinite"):
        optical_transition_radiation(
            crossings, _ENERGY, _BACK, [_AU.epsilon(_ENERGY)], source=source, n_histories=8
        )


# --- Synthetic straight histories (exact limits) ----------------------------------

_BETA = 0.6
_V = np.array([0.0, 0.0, _BETA])


def _film(foil=2e3, eps_layers=1):
    """Entry and exit of one straight electron through ``eps_layers`` film layers."""
    planes = np.linspace(0.0, foil, eps_layers + 1)
    t = planes / _BETA
    k = planes.size
    return BoundaryCrossings(
        electron_id=np.zeros(k, int), track_id=np.zeros(k, int), charge=-np.ones(k, np.int8),
        position_ang=_V * t[:, None], direction=np.tile([0.0, 0.0, 1.0], (k, 1)),
        kinetic_energy_keV=np.full(k, M_E_EV * (1 / np.sqrt(1 - _BETA**2) - 1) / 1e3),
        time_ang=t, interface=np.arange(k),
        normal=np.tile([0.0, 0.0, 1.0], (k, 1)), layer_before=np.r_[-1, np.arange(k - 1)],
        layer_after=np.r_[np.arange(1, k), -1] if k > 2 else np.array([0, -1]),
        z_interfaces_ang=planes, n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _bunch(t0_ang, dx_ang, foil=2e3):
    one = _film(foil)
    m, k = len(t0_ang), len(one.interface)
    position = np.repeat(one.position_ang[None], m, 0)
    position[:, :, 0] += np.asarray(dx_ang)[:, None]
    fields = {
        f.name: np.concatenate([getattr(one, f.name)] * m)
        for f in dataclasses.fields(one)
        if isinstance(getattr(one, f.name), np.ndarray) and f.name != "z_interfaces_ang"
    }
    fields.update(
        electron_id=np.repeat(np.arange(m), k), track_id=np.repeat(np.arange(m), k),
        position_ang=position.reshape(-1, 3),
        time_ang=(one.time_ang[None] + np.asarray(t0_ang)[:, None]).reshape(-1),
    )  # fmt: skip
    return dataclasses.replace(one, **fields)


_FWD = np.array([np.sin(0.7), 0.0, np.cos(0.7)])
_E = np.array([1.5, 2.0, 2.8])
_PC1 = 1.602176634e-7  # pC of one electron


def _score(crossings, source, m, eps=None, energy=_E, n=_FWD):
    eps = [_AU.epsilon(energy)] if eps is None else eps
    return optical_transition_radiation(
        crossings, energy, n, eps, source=source, n_histories=m
    ).yield_per_electron


# --- Plan 4: Maxwell limits through the optical path ------------------------------


def test_matched_medium_and_vanishing_film_radiate_nothing():
    one = OpticalBunchSource.sampled(_PC1, "one")
    matched = [_response(1.0 + 0j).epsilon(_E)]
    assert np.all(_score(_film(), one, 1, eps=matched) == 0.0)
    thin = _score(_film(foil=1e-6), one, 1)
    assert np.max(thin) < 1e-12 * np.max(_score(_film(), one, 1))


def test_finite_gold_permittivity_matches_closed_form_ginzburg_frank():
    from pyrite.montecarlo.spectrum.transition_radiation import ginzburg_frank_interface

    theta = np.linspace(0.05, 1.5, 15)
    n = np.stack([np.sin(theta), 0 * theta, -np.cos(theta)], -1)
    entrance = _take(_film(foil=1e6), np.array([0]))  # charge entering bulk gold
    one = OpticalBunchSource.sampled(_PC1, "one")
    for energy in (1.4, 2.5, 3.3):
        eps = _AU.epsilon(energy)
        got = optical_transition_radiation(
            entrance, energy, n, [eps], source=one, n_histories=1
        ).yield_per_electron
        closed = ginzburg_frank_interface(eps, -_BETA, theta, energy)
        np.testing.assert_allclose(got, closed, rtol=1e-9)


def test_perfect_conductor_limit_recovers_ginzburg_frank():
    theta = np.linspace(0.05, 1.5, 12)
    n = np.stack([np.sin(theta), 0 * theta, -np.cos(theta)], -1)
    exit_only = _take(_film(foil=1e5), np.array([0]))  # entrance face, backward emission
    conductor = [_response(-1e12 + 1e9j, "conductor").epsilon(2.0)]
    one = OpticalBunchSource.sampled(_PC1, "one")
    got = optical_transition_radiation(
        exit_only, 2.0, n, conductor, source=one, n_histories=1
    ).yield_per_electron
    beta = _BETA
    pc = (
        ALPHA_FS
        * beta**2
        * np.sin(theta) ** 2
        / (np.pi**2 * 2.0 * (1 - beta**2 * np.cos(theta) ** 2) ** 2)
    )
    np.testing.assert_allclose(got, pc, rtol=1e-6)


# --- Plan 2/3/4: population, weights, form factors --------------------------------


def test_population_limits_n1_f0_f1_and_sample_count_invariance():
    single = _score(_film(), OpticalBunchSource.sampled(_PC1, "one"), 1)
    charge = 1e4 * _PC1
    for m in (2, 5, 17):
        same = _bunch(np.zeros(m), np.zeros(m))
        # F = 1 (identical in-phase sampled histories): N I_1 per electron.
        f1 = _score(same, OpticalBunchSource.sampled(charge, "point bunch"), m)
        np.testing.assert_allclose(f1, 1e4 * single, rtol=1e-10)
        # F -> 0 (Gaussian law far beyond its turnover): I_1 per electron.
        long = OpticalBunchSource.gaussian(charge, "long", rms_t_fs=1e3)
        np.testing.assert_allclose(_score(same, long, m), single, rtol=1e-12)
        # N = 1 keeps only the self term whatever the law.
        alone = OpticalBunchSource.gaussian(_PC1, "one", rms_t_fs=0.1)
        np.testing.assert_allclose(_score(same, alone, m), single, rtol=1e-12)


def test_gaussian_law_is_exact_and_matches_sampled_offsets():
    m, sigma_t_fs, sigma_x_um = 3000, 0.25, 0.05
    charge = 1e6 * _PC1
    law = OpticalBunchSource.gaussian(charge, "gauss", rms_t_fs=sigma_t_fs, rms_x_um=sigma_x_um)
    same = _bunch(np.zeros(4), np.zeros(4))
    single = _score(_film(), OpticalBunchSource.sampled(_PC1, "one"), 1)
    omega = _E / HBARC_EV_ANG
    sig_t, sig_x = sigma_t_fs * C_ANG_PER_FS, sigma_x_um * 1e4
    factor = np.exp(-((omega * sig_t) ** 2) - (omega * _FWD[0] * sig_x) ** 2)
    np.testing.assert_allclose(law.form_factor_squared(_E, _FWD), factor, rtol=1e-12)
    expected = (1 + (1e6 - 1) * factor) * single
    np.testing.assert_allclose(_score(same, law, 4), expected, rtol=1e-10)
    rng = np.random.default_rng(5)
    sampled = _bunch(rng.normal(0, sig_t, m), rng.normal(0, sig_x, m))
    got = _score(sampled, OpticalBunchSource.sampled(charge, "sampled gauss"), m)
    sigma = (1.0 - factor) * np.sqrt(2.0 / (m * factor))
    assert np.all(np.abs(got / expected - 1.0) < 3.5 * sigma)
    assert np.all((factor > 0.1) & (factor < 0.9))
    # Applying the analytic factor to histories that carry offsets is refused.
    with pytest.raises(ValueError, match="twice"):
        _score(sampled, law, m)


def test_importance_weights_are_consistent_and_never_squared_as_charge():
    m, sigma_t, wide = 4000, 700.0, 1000.0  # Å of crossing-time spread
    rng = np.random.default_rng(9)
    t0 = rng.normal(0, wide, m)
    weights = np.exp(-0.5 * (t0 / sigma_t) ** 2) / np.exp(-0.5 * (t0 / wide) ** 2)
    bunch = _bunch(t0, np.zeros(m))
    charge = 1e5 * _PC1
    weighted = _score(bunch, OpticalBunchSource.sampled(charge, "is", weights=weights), m)
    rescaled = _score(bunch, OpticalBunchSource.sampled(charge, "is", weights=1e6 * weights), m)
    np.testing.assert_allclose(rescaled, weighted, rtol=1e-10)  # scale-free: not charges
    unit = _score(bunch, OpticalBunchSource.sampled(charge, "unit"), m)
    np.testing.assert_allclose(
        _score(bunch, OpticalBunchSource.sampled(charge, "u", weights=np.ones(m)), m),
        unit, rtol=1e-12,
    )  # fmt: skip
    single = _score(_film(), OpticalBunchSource.sampled(_PC1, "one"), 1)
    factor = np.exp(-((_E / HBARC_EV_ANG * sigma_t) ** 2))
    expected = (1 + (1e5 - 1) * factor) * single
    effective = weights.sum() ** 2 / np.sum(weights**2)
    tolerance = 4.0 * (1.0 - factor) * np.sqrt(2.0 / (effective * factor))
    assert np.all(np.abs(weighted / expected - 1.0) < tolerance)


def test_empirical_samples_profile_microbunching_and_resolution():
    charge = 1e6 * _PC1
    rng = np.random.default_rng(2)
    sigma = 700.0
    draws = np.stack([rng.normal(0, sigma, 20000), np.zeros(20000), np.zeros(20000)], -1)
    samples = OpticalBunchSource(
        "empirical", charge, "draws", profile_ang=draws, profile_kind="samples",
        time_resolution_ang=50.0,
    )  # fmt: skip
    exact = np.exp(-((_E / HBARC_EV_ANG * sigma) ** 2))
    got = samples.form_factor_squared(_E, _FWD)
    assert np.all(np.abs(got - exact) < 4 * np.sqrt(2 * exact / 20000) + 1e-4)
    # A density on a fine grid: its transform, no finite-sample bias term.
    t = np.linspace(-1.5e5, 1.5e5, 24001)
    density = np.exp(-0.5 * (t / sigma) ** 2)
    grid = np.stack([t, 0 * t, 0 * t], -1)
    profile = OpticalBunchSource(
        "empirical", charge, "density", profile_ang=grid, profile_weights=density,
        profile_kind="profile", time_resolution_ang=t[1] - t[0],
    )  # fmt: skip
    np.testing.assert_allclose(profile.form_factor_squared(_E, _FWD), exact, rtol=1e-6)
    # Microbunching at period lambda_m: a coherent harmonic at E = 2 pi hbar c / lambda_m.
    period = 2 * np.pi * HBARC_EV_ANG / 2.0
    train = np.exp(-0.5 * (t / (5 * period)) ** 2) * (1 + np.cos(2 * np.pi * t / period))
    micro = OpticalBunchSource(
        "empirical", charge, "train", profile_ang=grid, profile_weights=train,
        profile_kind="profile", time_resolution_ang=t[1] - t[0],
    )  # fmt: skip
    harmonic = micro.form_factor_squared(np.array([1.0, 2.0]), np.array([0.0, 0.0, 1.0]))
    assert harmonic[1] > 0.2 > harmonic[0] / 10
    coarse = dataclasses.replace(profile, time_resolution_ang=1e4)
    with pytest.raises(ValueError, match="does not resolve"):
        coarse.form_factor_squared(_E, _FWD)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: OpticalBunchSource("sampled", 1.0, "x", covariance_ang2=np.eye(3)), "twice"),
        (
            lambda: OpticalBunchSource("sampled", 1.0, "x", inter_electron="correlated"),
            "correlated",
        ),
        (lambda: OpticalBunchSource("realization", 1.0, "x", history_weights=np.ones(2)), "macro"),
        (lambda: OpticalBunchSource("gaussian", 1.0, "x"), "covariance"),
        (lambda: OpticalBunchSource("thermal", 1.0, "x"), "unsupported"),
        (lambda: OpticalBunchSource.sampled(1.0, "x", weights=np.array([1.0, -1.0])), "positive"),
        (lambda: OpticalBunchSource.sampled(1.0, ""), "provenance"),
        (lambda: OpticalBunchSource.sampled(1e-9, "x"), "at least one"),
        (
            lambda: OpticalBunchSource(
                "empirical", 1.0, "x", profile_ang=np.zeros((3, 3)), profile_kind="samples"
            ),
            "resolution",
        ),
    ],
)
def test_unsupported_source_contracts_are_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_realization_and_pair_estimate_contracts():
    same = _bunch(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="realization needs"):
        _score(same, OpticalBunchSource.realization(5 * _PC1, "five"), 3)
    with pytest.raises(CoherentSamplingError, match="two sampled"):
        _score(_film(), OpticalBunchSource.sampled(10 * _PC1, "ten"), 1)
    half = np.pi * HBARC_EV_ANG / 2.0
    opposed = _bunch(np.array([0.0, half]), np.zeros(2))
    with pytest.warns(CoherentSamplingWarning, match="returned unclipped"):
        y = _score(opposed, OpticalBunchSource.sampled(10 * _PC1, "x"), 2, energy=np.array([2.0]))
    assert y[0] < 0.0


def test_gaussian_coherent_window_for_the_catalog_200_fs_bunch():
    # Catalog envelope_rms_fs is the RMS crossing-time width at normal incidence.
    n_1pc = 1e-12 / 1.602176634e-19
    band = gaussian_coherent_band_eV(200.0, n_1pc, 1e-3)
    hbar = HBARC_EV_ANG / C_ANG_PER_FS
    assert hbar / 200.0 == pytest.approx(3.291e-3, rel=1e-3)  # E_1/e
    assert band == pytest.approx(hbar / 200.0 * np.sqrt(np.log((n_1pc - 1) / 1e-3)))
    assert 0.015 < band < 0.016  # turnover needs sub-0.01 eV optical support
    assert band < _AU.bands_eV[0][0] / 10
    assert gaussian_coherent_band_eV(200.0, 1.0, 1e-3) == 0.0
    law = OpticalBunchSource.gaussian(1.0, "catalog gaussian_200fs", rms_t_fs=200.0)
    assert law.form_factor_squared(0.01, np.array([0, 0, 1.0])) == pytest.approx(
        np.exp(-((0.01 / (hbar / 200.0)) ** 2))
    )


def test_relativistic_beta_helper_matches_film_kinematics():
    t_keV = M_E_EV * (1 / np.sqrt(1 - _BETA**2) - 1) / 1e3
    assert beta_from_kinetic_eV(t_keV * 1e3) == pytest.approx(_BETA)


# --- Plan 6: formation and energy-budget refusals -----------------------------------


def test_straight_runs_and_formation_bound_on_transported_histories(transported):
    case, tp = transported
    from pyrite.montecarlo.spectrum.optical_transition_validity import formation_error_bound

    crossings = _crossings(case, tp)
    entry = crossings.layer_before == -1
    assert np.all(np.isinf(crossings.straight_before_ang[entry]))
    assert np.all(np.isinf(crossings.straight_after_ang[crossings.layer_after == -1]))
    finite = np.r_[crossings.straight_before_ang[~entry], crossings.straight_after_ang[entry]]
    assert np.all((finite > 0) & (finite < 5e5))
    eps = case_layer_eps(case, _ENERGY, layer_responses=(_AU,))
    bound = formation_error_bound(crossings, _ENERGY, _BACK, eps)
    assert bound.shape == _ENERGY.shape and np.all(np.isfinite(bound) & (bound >= 0))
    # Infinitely long straight runs remove the bound; zero runs give the full sensitivity.
    straight = dataclasses.replace(
        crossings,
        straight_before_ang=np.full(len(crossings), np.inf),
        straight_after_ang=np.full(len(crossings), np.inf),
    )
    assert np.all(formation_error_bound(straight, _ENERGY, _BACK, eps) == 0.0)
    kinked = dataclasses.replace(
        crossings,
        straight_before_ang=np.where(entry, np.inf, 0.0),
        straight_after_ang=np.where(crossings.layer_after == -1, np.inf, 0.0),
    )
    assert np.all(formation_error_bound(kinked, _ENERGY, _BACK, eps) >= bound)
    with pytest.raises(ValueError, match="formation-zone bound"):
        case_optical_transition(
            case, tp, layer_responses=(_AU,), source=OpticalBunchSource.sampled(_PC1, "one"),
            formation_tolerance=0.0 if np.max(bound) > 0 else -1.0, energy_budget=None,
        )  # fmt: skip


@pytest.mark.filterwarnings(
    "ignore::pyrite.montecarlo.spectrum.coherent_population.CoherentSamplingWarning"
)
def test_energy_budget_records_and_refuses_unbounded_coherence(transported):
    case, tp = transported
    one = OpticalBunchSource.sampled(_PC1, "one")
    result = case_optical_transition(
        case, tp, layer_responses=(_AU,), source=one, formation_tolerance=None
    )
    fraction = result.metadata["validity"]["energy_fraction"]
    assert 0.0 < fraction < 1e-6
    # A 1 nC point-like bunch keeps its pair term at the band edges: refused.
    point = OpticalBunchSource.gaussian(1000.0, "short", rms_t_fs=1e-4)
    with pytest.raises(ValueError, match="band edge"):
        case_optical_transition(
            case, tp, layer_responses=(_AU,), source=point, formation_tolerance=None
        )
    with pytest.raises(ValueError, match="kinetic energy"):
        case_optical_transition(
            case, tp, layer_responses=(_AU,), source=one, formation_tolerance=None,
            energy_budget=1e-12,
        )  # fmt: skip
