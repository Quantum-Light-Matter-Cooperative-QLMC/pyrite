"""Opt-in mixed (class II) elastic transport with a random hinge (issue #894).

Validation: mixed-elastic-split
Validation: mixed-elastic-soft-moments
Validation: mixed-elastic-hinge
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories
from pyrite.montecarlo.transport.events import (
    EVENT_DELTA,
    EVENT_ELASTIC,
    EVENT_SOFT_HINGE,
    EVENT_SUBSTEP,
)
from pyrite.montecarlo.transport.mixed_elastic import (
    MIXED_MAX_STEP_LAYER_FRACTION,
    MixedElasticParameters,
    _hinge_moments,
    _invert_cdf_row,
    _mixed_hard_quantile_scalar,
    _mixed_hard_rate_scalar,
    _mixed_soft_rates_scalar,
    _sample_hinge_mu,
    layer_step_limits_ang,
    mean_angular_deflection,
    mixed_elastic_tables,
    validate_mixed_parameters,
)
from pyrite.montecarlo.transport.scattering import _elsepa_rate_scalar, pack_elsepa_tables

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
    np.testing.assert_allclose(layer_step_limits_ang(top, bottom), [1e3, 9.9e4])
    np.testing.assert_allclose(layer_step_limits_ang(top, bottom, 1.0), [1e3, 1e4])


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
    thick = {"E0": 1000.0, "thickness": 1e6}
    detailed = _transport(None, **thick)
    mixed = _transport(MixedElasticParameters(), **thick)
    again = _transport(MixedElasticParameters(), **thick)

    assert mixed["L_ang"].size < 0.25 * detailed["L_ang"].size
    np.testing.assert_array_equal(mixed["L_ang"], again["L_ang"])
    np.testing.assert_array_equal(mixed["v_hat"], again["v_hat"])


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
