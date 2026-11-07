"""Opt-in mixed (class II) elastic transport with a random hinge (issue #894).

Validation: mixed-elastic-split
Validation: mixed-elastic-soft-moments
Validation: mixed-elastic-hinge
"""

from functools import partial

import numpy as np
import pytest

from pyrite.config import FixedSampling
from pyrite.montecarlo.case import _ABSENT
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories
from pyrite.montecarlo.transport import host_pass
from pyrite.montecarlo.transport.events import (
    EVENT_CUTOFF,
    EVENT_DELTA,
    EVENT_ELASTIC,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_SOFT_HINGE,
    EVENT_SUBSTEP,
)
from pyrite.montecarlo.transport.mixed_elastic import (
    MIXED_MAX_STEP_LAYER_FRACTION,
    MixedElasticParameters,
    _hinge_moments,
    _invert_cdf_row,
    _layer_hard_rate_bound,
    _mixed_hard_element,
    _mixed_hard_quantile_scalar,
    _mixed_hard_rate_scalar,
    _mixed_soft_rates_scalar,
    _panel_moments,
    _sample_hinge_mu,
    layer_step_limits_ang,
    mean_angular_deflection,
    mixed_elastic_tables,
    validate_mixed_parameters,
)
from pyrite.montecarlo.transport.scattering import (
    _elsepa_bracket,
    _elsepa_rate_scalar,
    pack_elsepa_tables,
)

MU = np.concatenate([[0.0], np.geomspace(1e-9, 1.0, 400)])
ENERGIES_KEV = np.geomspace(20.0, 2000.0, 12)
N_CM3 = 5e22


def _sr_table(Z=14):
    """Screened-Rutherford law in ELSEPA table form, analytic and table-free."""
    alpha = 3.4e-3 * Z**0.67 / ENERGIES_KEV
    rel = (ENERGIES_KEV + 511.0) / (ENERGIES_KEV + 1024.0)
    sigma = 5.21e-21 * Z * Z * 4.0 * np.pi / ENERGIES_KEV**2 / (alpha * (1.0 + alpha)) * rel**2
    dcs = (sigma * alpha * (1.0 + alpha) / (4.0 * np.pi))[:, None] / (
        MU[None, :] + alpha[:, None]
    ) ** 2
    return {
        "energy_eV": ENERGIES_KEV * 1e3,
        "total_elastic_cm2": sigma,
        "mu": MU,
        "dcs_cm2_sr": dcs,
    }


@pytest.fixture(scope="module")
def group():
    return pack_elsepa_tables([[_sr_table()]], [np.array([N_CM3])])


def _soft_rates(group, tables, E_keV):
    _, start, length, logE, log_rate, *_ = group
    _, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi = tables
    return _mixed_soft_rates_scalar(
        E_keV, logE, log_rate, t1_lo, t1_hi, t2_lo, t2_cross, t2_hi, start[0, 0], length[0, 0]
    )


def _quantile_mu(group, row, f, xi):
    cdf, pdf, mu = group[5], group[6], group[7]
    lower = _invert_cdf_row(cdf[row], pdf[row], mu, xi)
    upper = _invert_cdf_row(cdf[row + 1], pdf[row + 1], mu, xi)
    return (1.0 - f) * lower + f * upper


@pytest.mark.parametrize("c1", [0.01, 0.05, 0.2])
def test_cutoff_bounds_the_mean_deflection_per_hard_step(group, c1):
    hard_fraction = mixed_elastic_tables(group, c1)[0]
    mean_mu = mean_angular_deflection(group[6], group[7])

    assert np.all((hard_fraction > 0.0) & (hard_fraction <= 1.0))
    active = hard_fraction < 1.0
    assert active.any()
    # lam_h = lam_el / hard_fraction = c1 lam_1, with lam_1 = lam_el / (2 <mu>).
    np.testing.assert_allclose(hard_fraction[active], 2.0 * mean_mu[active] / c1, rtol=1e-14)
    assert np.all(2.0 * mean_mu[~active] >= c1)


def test_zero_c1_has_no_soft_part(group):
    hard_fraction, *moments = mixed_elastic_tables(group, 0.0)

    assert np.all(hard_fraction == 1.0)
    assert all(np.all(column == 0.0) for column in moments)


@pytest.mark.parametrize("f", [0.0, 0.3, 1.0])
def test_soft_moments_are_those_of_the_detailed_quantile_law(group, f):
    """Exact at any energy in a panel, not only at the nodes."""
    tables = mixed_elastic_tables(group, 0.05)
    row = 8
    xi_cut = 1.0 - tables[0][row]
    assert 0.0 < xi_cut < 1.0
    log_e = np.log(ENERGIES_KEV[row]) + f * (np.log(ENERGIES_KEV[row + 1] / ENERGIES_KEV[row]))
    E_keV = float(np.exp(log_e))
    # Below the bracket's upper end so the row stays this panel; f = 1 is the
    # next node and evaluated by the panel's own row through the clamp below.
    E_keV = min(E_keV, ENERGIES_KEV[row + 1] * (1.0 - 1e-12))

    xi = xi_cut * (np.arange(2_000_000) + 0.5) / 2_000_000
    mu = _quantile_mu(group, row, f, xi)
    first = xi_cut * mu.mean()
    second = xi_cut * (mu * mu).mean()

    _, start, length, logE, log_rate, *_ = group
    rate = _elsepa_rate_scalar(E_keV, logE, log_rate, start[0, 0], length[0, 0])
    soft_1, soft_2 = _soft_rates(group, tables, E_keV)
    assert soft_1 / (2.0 * rate) == pytest.approx(first, rel=2e-5)
    assert soft_1 / (2.0 * rate) - soft_2 / (6.0 * rate) == pytest.approx(second, rel=2e-4)


def test_soft_and_hard_parts_close_on_the_detailed_first_moment(group):
    tables = mixed_elastic_tables(group, 0.05)
    row = 8
    f = 0.4
    xi_cut = 1.0 - tables[0][row]
    E_keV = float(np.exp(np.log(ENERGIES_KEV[row]) + f * np.log(ENERGIES_KEV[9] / ENERGIES_KEV[8])))
    _, start, length, logE, log_rate, *_ = group
    rate = _elsepa_rate_scalar(E_keV, logE, log_rate, start[0, 0], length[0, 0])
    means = mean_angular_deflection(group[6], group[7])
    detailed = (1.0 - f) * means[row] + f * means[row + 1]

    xi = xi_cut + (1.0 - xi_cut) * (np.arange(2_000_000) + 0.5) / 2_000_000
    hard = (1.0 - xi_cut) * _quantile_mu(group, row, f, xi).mean()
    soft = _soft_rates(group, tables, E_keV)[0] / (2.0 * rate)

    assert soft + hard == pytest.approx(detailed, rel=1e-5)


def test_hard_rate_and_quantile_share_the_panel_fraction(group):
    hard_fraction = mixed_elastic_tables(group, 0.05)[0]
    _, start, length, logE, log_rate, *_ = group
    E_keV = float(np.sqrt(ENERGIES_KEV[8] * ENERGIES_KEV[9]))
    begin, count = start[0, 0], length[0, 0]

    full = _elsepa_rate_scalar(E_keV, logE, log_rate, begin, count)
    hard = _mixed_hard_rate_scalar(E_keV, logE, log_rate, hard_fraction, begin, count)
    assert hard == pytest.approx(full * hard_fraction[8], rel=1e-15)
    assert _mixed_hard_quantile_scalar(0.0, E_keV, logE, hard_fraction, begin, count) == (
        pytest.approx(1.0 - hard_fraction[8], rel=1e-15)
    )
    assert _mixed_hard_quantile_scalar(1.0, E_keV, logE, hard_fraction, begin, count) == 1.0


@pytest.mark.parametrize(
    ("step_cm", "soft_1", "soft_2"),
    [(1e-5, 2e2, 6e2), (1e-4, 2e2, 5.9e2), (1e-3, 2e2, 5.5e2), (5e-3, 1e2, 2.9e2)],
)
def test_hinge_distribution_reproduces_both_lewis_moments(step_cm, soft_1, soft_2):
    """PENELOPE-2024 Eqs. (4.28)-(4.32), integrated over the uniform deviate."""
    first, second = _hinge_moments(step_cm, soft_1, soft_2)
    assert first == pytest.approx(0.5 * (1.0 - np.exp(-step_cm * soft_1)), rel=1e-12)
    assert first - second == pytest.approx((1.0 - np.exp(-step_cm * soft_2)) / 6.0, rel=1e-12)

    # Tail probability can be 1e-6, so the deviate grid resolves u -> 1.
    u = 1.0 - np.geomspace(1e-14, 1.0, 400_001)[::-1]
    mu = np.array([_sample_hinge_mu(first, second, value) for value in u])

    assert np.all((mu >= 0.0) & (mu <= 1.0))
    assert np.trapezoid(mu, u) == pytest.approx(first, rel=1e-3)
    assert np.trapezoid(mu * mu, u) == pytest.approx(second, rel=1e-2)


def test_hinge_without_soft_scattering_does_not_deflect():
    assert _sample_hinge_mu(*_hinge_moments(1e-4, 0.0, 0.0), 0.7) == 0.0


@pytest.mark.parametrize(
    ("c1", "c2", "max_step"),
    [
        (-0.1, 0.05, None),
        (0.3, 0.05, None),
        (0.05, 0.0, None),
        (0.05, 0.3, None),
        (0.05, 0.05, 0.0),
    ],
)
def test_parameters_outside_the_scheme_are_refused(c1, c2, max_step):
    with pytest.raises(ValueError, match="mixed_"):
        validate_mixed_parameters(c1, c2, max_step)


def test_zero_c1_is_valid_without_c2():
    validate_mixed_parameters(0.0, 0.0)


def test_step_limit_is_the_layer_fraction_or_the_explicit_limit():
    top, bottom = np.array([0.0, 1e4]), np.array([1e4, 1e6])
    np.testing.assert_allclose(layer_step_limits_ang(top, bottom), [1e2, 9.9e3])
    np.testing.assert_allclose(layer_step_limits_ang(top, bottom, 0.5), [1e2, 5e3])


def _transport(mixed, *, Ne=60, seed=11, thickness=2e4, E0=200.0, **kwargs):
    arguments = {
        "E0_keV": E0,
        "Ne": Ne,
        "thickness_ang": thickness,
        "element": "Si",
        "n_atoms_per_ang3": N_CM3 * 1e-24,
        "E_cut_keV": 30.0,
        "seed": seed,
        "max_steps": 200_000,
        "energy_model": "midpoint",
        "transport_core": "per-electron",
        "elastic_tables": [[_sr_table()]],
        "radiative_model": "uncoupled",
        **kwargs,
    }
    return simulate_trajectories(**arguments, mixed_elastic=mixed)


def test_zero_c1_is_bitwise_detailed_transport():
    """The theta_c -> 0 limit, exactly: no hinge, no extra draws, same rows."""
    detailed = _transport(None)
    mixed = _transport(MixedElasticParameters(c1=0.0, c2=0.0))

    arrays = [key for key, value in detailed.items() if isinstance(value, np.ndarray)]
    assert "L_ang" in arrays
    for key in arrays:
        np.testing.assert_array_equal(mixed[key], detailed[key], err_msg=key)


def test_mixed_rows_stay_continuous_and_deflect_only_at_events():
    result = _transport(MixedElasticParameters(c1=0.1), max_dE_frac=0.05)
    kind = result["event_kind"]
    electron = result["electron_id"]
    same = electron[1:] == electron[:-1]
    assert np.any(kind == EVENT_SOFT_HINGE)
    assert np.any(kind == EVENT_ELASTIC)

    half = 0.5 * result["L_ang"][:, None] * result["v_hat"]
    start = result["r_mid"] - half
    end = result["r_mid"] + half
    np.testing.assert_allclose(start[1:][same], end[:-1][same], atol=1e-5)
    np.testing.assert_allclose(
        result["E_start_keV"][1:][same], result["E_end_keV"][:-1][same], rtol=1e-12
    )
    turned = np.any(result["v_hat"][1:] != result["v_hat"][:-1], axis=1)
    straight = same & np.isin(kind[:-1], [EVENT_SUBSTEP, EVENT_DELTA])
    assert not np.any(turned[straight])
    assert np.any(turned[same & (kind[:-1] == EVENT_SOFT_HINGE)])


def test_steps_respect_the_layer_limit():
    thickness = 2e4
    result = _transport(MixedElasticParameters(c1=0.2, c2=0.2), thickness=thickness)
    assert result["L_ang"].max() <= MIXED_MAX_STEP_LAYER_FRACTION * thickness * (1.0 + 1e-12)


def test_mixed_transport_is_much_cheaper_and_reproducible():
    thick = {"E0": 1000.0, "thickness": 1e7}
    detailed = _transport(None, **thick)
    mixed = _transport(MixedElasticParameters(), **thick)
    again = _transport(MixedElasticParameters(), **thick)

    assert mixed["L_ang"].size < 0.25 * detailed["L_ang"].size
    np.testing.assert_array_equal(mixed["L_ang"], again["L_ang"])
    np.testing.assert_array_equal(mixed["v_hat"], again["v_hat"])


def _lewis_rates(group, E_keV):
    """Detailed ``(Sigma, 1/lam_1, 1/lam_2)`` [1/cm] at ``E_keV``, on the sampler's own law."""
    _, start, length, log_e, log_rate, cdf, pdf, mu, _ = group
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), log_e, int(start[0, 0]), int(length[0, 0]))
    rate = np.exp(log_rate[row] + f * (log_rate[row + 1] - log_rate[row]))
    q1_lo, q1_hi, q2_lo, q_cross, q2_hi = _panel_moments(
        cdf[row], pdf[row], cdf[row + 1], pdf[row + 1], mu, 1.0
    )
    first = (1.0 - f) * q1_lo + f * q1_hi
    second = (1.0 - f) ** 2 * q2_lo + 2.0 * f * (1.0 - f) * q_cross + f**2 * q2_hi
    return rate, 2.0 * rate * first, 6.0 * rate * (first - second)


def _mid_slab_transport(monkeypatch, parameters, *, Ne, thickness=1e7):
    """Tracks that start mid-slab and end at the cutoff, so the medium looks infinite."""
    sample = host_pass.initial_beam_positions

    def from_mid_slab(*args, **kwargs):
        positions, slopes, entry = sample(*args, **kwargs)
        positions = positions.copy()
        positions[:, 2] += 0.5 * thickness
        return positions, slopes, entry

    monkeypatch.setattr(host_pass, "initial_beam_positions", from_mid_slab)
    return _transport(
        parameters, Ne=Ne, E0=170.0, E_cut_keV=100.0, thickness=thickness, max_steps=1_000_000
    )


@pytest.mark.parametrize(("c1", "c2"), [(0.05, 0.05), (0.2, 0.2)])
def test_mixed_tracks_follow_the_lewis_integrals_across_nodes(monkeypatch, group, c1, c2):
    """Hard events and both angular moments against noise-free integrals.

    The track crosses two table nodes, where the frozen panel cutoff makes the
    hard rate jump. Stopping is continuous, so ``E(s)`` is the same for every
    track and the references are deterministic.
    """
    Ne = 20_000
    result = _mid_slab_transport(monkeypatch, MixedElasticParameters(c1=c1, c2=c2), Ne=Ne)
    electron, kind = result["electron_id"], result["event_kind"]
    last = np.r_[np.nonzero(np.diff(electron))[0], electron.size - 1]
    assert last.size == Ne
    assert np.all(kind[last] == EVENT_CUTOFF)

    # The energy along one track, piecewise linear in its rows.
    track = electron == electron[0]
    length = result["L_ang"][track]
    start, end = result["E_start_keV"][track], result["E_end_keV"][track]
    nodes, weights = np.polynomial.legendre.leggauss(4)
    tables = mixed_elastic_tables(group, c1)
    hard_integral = first_integral = second_integral = 0.0
    for row_length, row_start, row_end in zip(length, start, end, strict=True):
        for node, weight in zip(nodes, weights, strict=True):
            E_keV = row_start + 0.5 * (node + 1.0) * (row_end - row_start)
            hard = _mixed_hard_rate_scalar(
                E_keV, group[3], group[4], tables[0], int(group[1][0, 0]), int(group[2][0, 0])
            )
            _, inverse_1, inverse_2 = _lewis_rates(group, E_keV)
            ds_cm = 0.5 * weight * row_length * 1e-8
            hard_integral += hard * ds_cm
            first_integral += inverse_1 * ds_cm
            second_integral += inverse_2 * ds_cm

    counts = np.count_nonzero(kind == EVENT_ELASTIC) / Ne
    assert counts == pytest.approx(hard_integral, abs=4.0 * np.sqrt(hard_integral / Ne))

    cosine = result["v_hat"][last, 2]
    p2 = 1.5 * cosine**2 - 0.5
    for measured, integral in ((cosine, first_integral), (p2, second_integral)):
        expected = np.exp(-integral)
        error = np.std(measured) / np.sqrt(Ne)
        assert abs(np.mean(measured) - expected) < 4.0 * error


def _exit_summary(result):
    electron = result["electron_id"]
    last = np.r_[np.nonzero(np.diff(electron))[0], electron.size - 1]
    kind = result["event_kind"][last]
    return (
        last.size,
        np.count_nonzero(kind == 4),
        np.count_nonzero(kind == 5),
        result["E_keV"][last],
    )


def test_mixed_exits_agree_with_detailed_transport_within_statistics():
    thick = {"Ne": 1500, "thickness": 1e5}
    n_d, back_d, trans_d, _ = _exit_summary(_transport(None, **thick))
    n_m, back_m, trans_m, _ = _exit_summary(_transport(MixedElasticParameters(), seed=23, **thick))
    assert n_d == n_m == 1500
    for count_d, count_m in ((back_d, back_m), (trans_d, trans_m)):
        sigma = np.sqrt(count_d + count_m)
        assert abs(count_d - count_m) < 4.0 * max(sigma, 1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"transport_core": "lockstep"},
        {"energy_model": "frozen", "max_dE_frac": 0.0},
    ],
)
def test_ineligible_transport_is_refused(overrides):
    with pytest.raises(ValueError, match="transport_mode='mixed'"):
        _transport(MixedElasticParameters(), **overrides)


@pytest.mark.parametrize("parameters", [MixedElasticParameters(), MixedElasticParameters(0.0, 0.0)])
def test_urban_straggling_is_refused(parameters):
    """Urban losses leave the hazard window, including in the detailed limit's core."""
    with pytest.raises(ValueError, match="Urban straggling"):
        _transport(parameters, straggling=True)


def test_shell_soft_sampler_still_runs_with_straggling():
    rows = _transport(MixedElasticParameters(), straggling=True, **_shell_kwargs())
    assert np.any(rows["event_kind"] == EVENT_SOFT_HINGE)


def _shell_kwargs():
    from pyrite.montecarlo import shell_configuration
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    return {
        "stopping_tables": [resolve_catalog_table("silicon").arrays()],
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "inelastic_materials": ["silicon"],
    }


def _radiative_kwargs():
    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    return {
        "radiative_model": "bremslib-soft-hard",
        "radiative_cutoff_eV": 1000.0,
        "bremslib_tables": {"Si": table},
    }


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(_shell_kwargs, id="shell"),
        pytest.param(_radiative_kwargs, id="radiative"),
        pytest.param(lambda: {**_shell_kwargs(), **_radiative_kwargs()}, id="shell-radiative"),
    ],
)
def test_collision_specializations_keep_the_detailed_limit_and_run_mixed(extra):
    kwargs = extra()
    detailed = _transport(None, **kwargs)
    limit = _transport(MixedElasticParameters(c1=0.0, c2=0.0), **kwargs)
    mixed = _transport(MixedElasticParameters(), **kwargs)

    for key, value in detailed.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(limit[key], value, err_msg=key)
    assert np.any(mixed["event_kind"] == EVENT_SOFT_HINGE)
    assert np.all(np.isfinite(mixed["E_end_keV"]))


@pytest.mark.parametrize("c1", [0.05, 0.2])
def test_hazard_bound_covers_the_hard_rate_and_its_elements(group, c1):
    tables = mixed_elastic_tables(group, c1)
    start, length = group[1], group[2]
    # The window straddles table nodes, where the frozen cutoff makes the rate jump.
    E_low, E_high = 25.0, 170.0
    bound = _layer_hard_rate_bound(
        E_low, E_high, 0, 1, group[3], group[4], tables[0], start, length
    )
    energies = np.geomspace(E_low, E_high, 4001)
    rates = [
        _mixed_hard_rate_scalar(E, group[3], group[4], tables[0], start[0, 0], length[0, 0])
        for E in energies
    ]
    assert bound >= max(rates)
    # Tight: the supremum sits at a panel's edge, which the fine grid approaches.
    assert bound == pytest.approx(max(rates), rel=2e-3)

    E_keV = 100.0
    rate = _mixed_hard_rate_scalar(E_keV, group[3], group[4], tables[0], start[0, 0], length[0, 0])
    element = partial(
        _mixed_hard_element, E_keV=E_keV, L=0, n_el=1, logE_flat=group[3],
        log_rate_flat=group[4], hard_fraction=tables[0], start=start, length=length,
    )  # fmt: skip
    assert element(target=0.999 * rate) == 0
    assert element(target=1.001 * rate) == -1


@pytest.mark.parametrize(
    ("extra", "event", "Ne", "minimum"),
    [
        (_shell_kwargs, EVENT_HARD_INELASTIC, 400, 100),
        (_radiative_kwargs, EVENT_HARD_RADIATIVE, 4000, 20),
    ],
    ids=["inelastic", "radiative"],
)
def test_thinned_hard_processes_keep_the_detailed_event_count(extra, event, Ne, minimum):
    """Rejected proposals restart their own clock and leave the others running."""
    arguments = {"Ne": Ne, "thickness": 1e6, **extra()}
    detailed = np.count_nonzero(_transport(None, **arguments)["event_kind"] == event)
    mixed = _transport(MixedElasticParameters(), seed=23, **arguments)
    count = np.count_nonzero(mixed["event_kind"] == event)
    assert detailed >= minimum
    assert abs(count - detailed) < 4.0 * np.sqrt(count + detailed)
    assert np.any(mixed["event_kind"] == EVENT_DELTA)


def test_compound_rates_add_over_elements():
    composition = [("Mo", 0.0186), ("Se", 0.0372)]
    tables = {"elastic_tables": [[_sr_table(42), _sr_table(34)]], "composition": composition}
    detailed = _transport(None, element=None, n_atoms_per_ang3=None, **tables)
    limit = _transport(
        MixedElasticParameters(c1=0.0, c2=0.0), element=None, n_atoms_per_ang3=None, **tables
    )
    mixed = _transport(MixedElasticParameters(), element=None, n_atoms_per_ang3=None, **tables)

    np.testing.assert_array_equal(limit["L_ang"], detailed["L_ang"])
    assert np.any(mixed["event_kind"] == EVENT_SOFT_HINGE)


def _mixed_physics(**kwargs):
    from pyrite import Physics

    return Physics(transport_mode="mixed", **kwargs)


def test_physics_default_adds_no_case_keys():
    from pyrite import Physics
    from pyrite._mixed_elastic_policy import mixed_case_keys

    physics = Physics()
    assert physics.transport_mode == "single-scattering"
    assert (
        mixed_case_keys(
            physics.transport_mode, physics.mixed_c1, physics.mixed_c2, physics.mixed_max_step_um
        )
        == {}
    )


def test_physics_allows_straggling_with_the_shell_soft_sampler():
    assert _mixed_physics(straggling=True, inelastic_model="shell-soft-hard").straggling


def test_physics_mixed_resolves_default_parameters():
    from pyrite._mixed_elastic_policy import mixed_case_keys

    physics = _mixed_physics(mixed_max_step_um=2.0)
    assert mixed_case_keys(
        physics.transport_mode, physics.mixed_c1, physics.mixed_c2, physics.mixed_max_step_um
    ) == {
        "transport_mode": "mixed",
        "mixed_c1": 0.05,
        "mixed_c2": 0.05,
        "mixed_max_step_um": 2.0,
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mixed_c1": 0.1}, "transport_mode='mixed'"),
        ({"transport_mode": "condensed"}, "transport_mode must be"),
        ({"transport_mode": "mixed", "mixed_c1": 0.3}, "mixed_c1"),
        ({"transport_mode": "mixed", "mixed_c2": -0.1}, "mixed_c2"),
        ({"transport_mode": "mixed", "mixed_max_step_um": 0.0}, "mixed_max_step_um"),
        ({"transport_mode": "mixed", "elastic_model": "mott"}, "elastic_model='elsepa'"),
        (
            {"transport_mode": "mixed", "straggling": True, "inelastic_model": "continuous"},
            "Urban straggling",
        ),
    ],
)
def test_physics_refuses_invalid_mixed_settings(kwargs, message):
    from pyrite import Physics

    with pytest.raises(ValueError, match=message):
        Physics(**kwargs)


def _mixed_scene():
    import pyrite as pr

    # Thick: on a target thin against the hard mean free path, the layer step
    # limit makes mixed transport cost more rows than detailed scattering.
    beam = pr.Beam(
        energy_keV=1000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon", thickness_ang=1_000_000.0, footprint=pr.geometry.Footprint(5.0, 5.0)
    )
    return pr.Scene(beam, target, pr.Detector())


def _lowered(physics=None):
    import numpy as np

    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    grid = np.array([1000.0, 1050.0])
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=1, brem_histories=1),
        energy_bins=EnergyBins(cxr=grid, brem=grid),
    )
    return api.build_case(_mixed_scene(), numerics, physics=physics)


def test_default_case_carries_no_mixed_keys():
    case = _lowered()
    assert "transport_mode" not in case
    assert not any(key.startswith("mixed_") for key in case)


def test_mixed_case_records_identity_and_routes_to_per_electron():
    from pyrite.montecarlo.runner.adaptive import case_transport_core
    from pyrite.montecarlo.runner.case_tables import _case_elastic_kwargs

    case = _lowered(_mixed_physics(mixed_c1=0.1))
    assert case["transport_mode"] == "mixed"
    assert (case["mixed_c1"], case["mixed_c2"]) == (0.1, 0.05)
    assert "mixed_max_step_um" not in case
    assert _case_elastic_kwargs(case)["mixed_elastic"] == (0.1, 0.05, None)
    assert case_transport_core(case, "auto") == "per-electron"
    assert case_transport_core(case, "cuda") == "cuda"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mixed_c1": 0.5}, "mixed_c1"),
        ({"transport_mode": "condensed"}, "transport_mode must be"),
        ({"coherent_emission": True}, "coherent emission"),
        ({"straggling": True, "inelastic_model": _ABSENT, "inelastic_cutoff_eV": _ABSENT}, "Urban"),
        ({"temporal_profile": True}, "temporal profiles"),
        ({"groove_spacing_ang": 100.0}, "grooved"),
    ],
)
def test_case_refuses_ineligible_mixed_combinations(overrides, message):
    from dataclasses import replace

    case = _lowered(_mixed_physics())
    with pytest.raises(ValueError, match=message):
        replace(case, **overrides)


def test_public_mixed_physics_cuts_rows():
    import pyrite as pr
    from pyrite.campaign.trajectories import simulate_trajectories

    scene = _mixed_scene()
    numerics = pr.Numerics(sampling=FixedSampling(cxr_histories=40, brem_histories=40), seed=7)
    rows = [
        simulate_trajectories(scene.beam, scene.target, physics=physics, numerics=numerics)[
            "E_end_keV"
        ].size
        for physics in (pr.Physics(), _mixed_physics())
    ]
    assert rows[1] < rows[0]


def test_mixed_case_records_model_marker_and_default_does_not():
    from pyrite._physics_provenance import resolved_physics_view

    markers = dict(
        resolved_physics_view(_lowered(_mixed_physics()), table_markers={}).model_markers
    )
    assert markers["mixed_elastic_model"] == "penelope-random-hinge.v2"
    assert "mixed_elastic_model" not in dict(
        resolved_physics_view(_lowered(), table_markers={}).model_markers
    )
