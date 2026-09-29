"""Closed-form ``sinc^2`` bin integration (issue #116).

Pins the antiderivative and its ``Si`` evaluator, the mass identity on uniform
and windowed grids, both limits, the ``mc_spectrum`` opt-in on both routes, the
explicit refusals, and that node sampling stays the default.

Validation: sinc-bin-integration
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import sici

from pyrite._backend import REAL
from pyrite._line_windows import FeatureSeed, build_window_plan
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths
from pyrite.montecarlo.spectrum.lines import (
    SpectrumRequest,
    _accumulate_batched,
    _accumulate_per_hkl,
    _finalize_spectrum,
    _policy,
    _prepare_spectrum,
)
from pyrite.montecarlo.spectrum.lines import _bin_quadrature as bq
from tests.helpers import host_backend_only, scaled_rtol, to_device, to_host
from tests.helpers.backend import IS_FLOAT32
from tests.montecarlo.test_spectrum_phases import B_ANG2, CRYSTAL, HKL, N_HAT, _segments

# The #116 acceptance spacings: the catalog's 3 eV and an eighth of it.
SPACINGS_EV = (3.0, 0.375)
# Lines chosen to cover a narrow (a = 1e3/eV), a resolved, and a broad profile,
# one centred on a bin edge, one near each window end.
A_WIDTH = np.array([1.0e3, 30.0, 3.0, 0.5, 30.0])
E_RES = np.array([8500.1, 8501.0, 8003.3, 8998.0, 8000.0])


def _uniform_axis(spacing, start=8000.0, stop=8999.0):
    """Nodes whose bins tile exactly ``[start, stop]`` (a whole number of bins)."""
    count = round((stop - start) / spacing)
    return start + (np.arange(count) + 0.5) * spacing


def _windowed_axis():
    """A #101 piecewise axis: 3 eV backbone with two fine windows."""
    seeds = [
        FeatureSeed("test", "a", 8500.0, 5.0, 5.0, 0.01),
        FeatureSeed("test", "b", 8003.0, 20.0, 20.0, 0.2),
    ]
    return build_window_plan(7990.0, 9010.0, 3.0, seeds).coordinates()


# ---- antiderivative -----------------------------------------------------------
@pytest.mark.parametrize("x", [0.0, 1.0e-9, 0.3, 1.0, 2.5, 7.3, 40.0, -0.7, -4.1])
def test_antiderivative_matches_quadrature(x):
    reference = quad(lambda u: np.sinc(u) ** 2, 0.0, x, limit=400, epsabs=1e-15)[0]
    assert float(bq.sincsq_antiderivative(x)) == pytest.approx(reference, abs=5e-14)


def test_antiderivative_is_odd_and_tends_to_one_half():
    x = np.array([1.0e3, 1.0e6, 1.0e12])
    assert np.all(np.abs(bq.sincsq_antiderivative(x) - 0.5) <= 1.0 / x)
    np.testing.assert_array_equal(bq.sincsq_antiderivative(-x), -bq.sincsq_antiderivative(x))


def test_the_handoff_formula_is_wrong_by_a_factor_two_in_si():
    """Guard against the #101 handoff's ``Si(2 pi u)/(2 pi)`` spelling."""
    x = 3.0
    handoff = sici(2 * np.pi * x)[0] / (2 * np.pi) - np.sin(np.pi * x) ** 2 / (np.pi**2 * x)
    reference = quad(lambda u: np.sinc(u) ** 2, 0.0, x, limit=400)[0]
    assert abs(handoff - reference) > 0.2
    assert float(bq.sincsq_antiderivative(x)) == pytest.approx(reference, abs=1e-13)


def test_si_asymptotic_branch_joins_the_library_branch():
    t = np.array([48.0, 60.0, 200.0])
    # Below ~1e3 the library subtraction still has ~1e-13 relative accuracy.
    np.testing.assert_allclose(bq.si_minus_half_pi(t), sici(t)[0] - np.pi / 2, rtol=1e-12)
    below = np.nextafter(48.0, 0.0)
    assert float(bq.si_minus_half_pi(below)) == pytest.approx(
        float(bq.si_minus_half_pi(48.0)), abs=1e-14
    )


def test_tail_complement_has_no_cancellation_far_out():
    """``R(x) = -1/(2 pi^2 x) (1 + O(1/x))``: relative accuracy, not just absolute.

    Subtracting ``F`` from ``1/2`` would leave only ``eps / R ~ 1e-6`` relative
    accuracy at ``x = 1e10``; the tail form keeps the ``O(1/x)`` correction.
    """
    x = np.array([1.0e4, 1.0e7, 1.0e10])
    relative = bq.sincsq_tail(x) * (-2 * np.pi**2 * x) - 1.0
    assert np.all(np.abs(relative) <= 1.0 / x)


# ---- mass identity ------------------------------------------------------------
@pytest.mark.parametrize("spacing", SPACINGS_EV)
def test_bin_masses_sum_to_the_whole_line_minus_the_reported_truncation(spacing):
    edges, _ = _energy_bin_edges_and_widths(_uniform_axis(spacing))
    masses = bq.sincsq_bin_masses(A_WIDTH, E_RES, edges)
    captured, truncated = bq.sincsq_window_mass(A_WIDTH, E_RES, edges)
    np.testing.assert_allclose(masses.sum(axis=1) + truncated, np.pi / A_WIDTH, rtol=1e-14)
    np.testing.assert_allclose(masses.sum(axis=1), captured, rtol=1e-14)
    assert np.all(truncated >= 0.0)


def test_bin_mass_is_independent_of_spacing():
    coarse, fine = (
        bq.sincsq_bin_masses(A_WIDTH, E_RES, _energy_bin_edges_and_widths(_uniform_axis(s))[0])
        for s in SPACINGS_EV
    )
    np.testing.assert_allclose(coarse.sum(axis=1), fine.sum(axis=1), rtol=1e-13)
    # Eight fine bins tile each coarse bin exactly, so the masses nest too.
    np.testing.assert_allclose(
        fine.reshape(fine.shape[0], -1, 8).sum(axis=2), coarse, rtol=1e-9, atol=1e-15
    )


def test_mass_identity_holds_on_a_nonuniform_windowed_grid():
    grid = _windowed_axis()
    assert np.ptp(np.diff(grid)) > 1.0, "the axis must really be nonuniform"
    edges, widths = _energy_bin_edges_and_widths(grid)
    masses = bq.sincsq_bin_masses(A_WIDTH, E_RES, edges)
    _, truncated = bq.sincsq_window_mass(A_WIDTH, E_RES, edges)
    np.testing.assert_allclose(masses.sum(axis=1) + truncated, np.pi / A_WIDTH, rtol=1e-14)
    # the density written into the spectrum is the mass per bin width
    density = bq._host_bin_mean(A_WIDTH, E_RES, edges, 1.0 / widths)
    np.testing.assert_allclose(density * widths, masses, rtol=1e-12, atol=1e-18)


# ---- limits -------------------------------------------------------------------
def test_vanishing_bin_width_recovers_node_sampling():
    grid = np.linspace(8000.0, 8010.0, 200_001)
    edges, widths = _energy_bin_edges_and_widths(grid)
    mean = bq._host_bin_mean([2.0], [8005.1234], edges, 1.0 / widths)[0]
    node = np.sinc(2.0 * (grid - 8005.1234) / np.pi) ** 2
    np.testing.assert_allclose(mean, node, atol=1e-8)


def test_infinite_width_parameter_puts_all_mass_in_the_resonance_bin():
    edges, _ = _energy_bin_edges_and_widths(np.arange(8000.0, 8010.0, 1.0))
    fractions = []
    for a_width in (1.0e1, 1.0e3, 1.0e6):
        masses = bq.sincsq_bin_masses([a_width], [8004.3], edges)[0]
        assert int(masses.argmax()) == 4  # bin [8003.5, 8004.5]
        fractions.append(masses.max() / (np.pi / a_width))
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0, abs=1e-5)


# ---- mc_spectrum opt-in -------------------------------------------------------
_KW = {"crystal": CRYSTAL, "hkl_list": [HKL], "B_ang2": B_ANG2, "n_hat": N_HAT}


def _long_flights():
    """Long flights: a sinc feature far narrower than 3 eV, so nodes alias."""
    segs = _segments(8, n_elec=2)
    segs["L_ang"] = np.full(8, 2000.0)
    return segs


def _yield(grid, spec):
    return float(np.sum(to_host(spec) * _energy_bin_edges_and_widths(grid)[1]))


def test_bin_mean_yield_is_spacing_independent_where_node_sampling_aliases():
    segs = _long_flights()
    dense = _uniform_axis(0.002, 701.0, 1499.0)
    reference = _yield(dense, mc_spectrum(segs, dense, **_KW))
    for spacing in SPACINGS_EV:
        grid = _uniform_axis(spacing, 701.0, 1499.0)
        binned = _yield(grid, mc_spectrum(segs, grid, line_quadrature="bin-mean", **_KW))
        # float32 casts the dense reference's nodes by up to an ulp.
        assert binned == pytest.approx(reference, rel=1e-3 if IS_FLOAT32 else 1e-5)
    coarse = _uniform_axis(3.0, 701.0, 1499.0)
    aliased = _yield(coarse, mc_spectrum(segs, coarse, **_KW))
    assert abs(aliased / reference - 1.0) > 1e-2, "fixture no longer aliases"


def test_node_quadrature_is_the_unchanged_default():
    segs = _long_flights()
    grid = _uniform_axis(3.0, 701.0, 1499.0)
    np.testing.assert_array_equal(
        to_host(mc_spectrum(segs, grid, **_KW)),
        to_host(mc_spectrum(segs, grid, line_quadrature="node", **_KW)),
    )


def test_per_hkl_and_batched_routes_agree_under_bin_mean():
    grid = _uniform_axis(3.0, 701.0, 1499.0)
    request = SpectrumRequest(
        segments=_long_flights(), E_grid_eV=grid, line_quadrature="bin-mean", **_KW
    )
    batched = _prepare_spectrum(request)
    _accumulate_batched(batched)
    per_hkl = _prepare_spectrum(request)
    _accumulate_per_hkl(per_hkl)
    np.testing.assert_allclose(
        to_host(_finalize_spectrum(batched)),
        to_host(_finalize_spectrum(per_hkl)),
        rtol=max(1e-10, 500.0 * float(np.finfo(REAL).eps)),
        atol=0.0,
    )


def test_fused_and_fallback_routes_agree(monkeypatch):
    """Off CUDA float32 both spellings take the fallback; on it, the fused kernel."""
    grid = _uniform_axis(3.0, 701.0, 1499.0)
    fused = to_host(mc_spectrum(_long_flights(), grid, line_quadrature="bin-mean", **_KW))
    monkeypatch.setattr(_policy, "_USE_JIT_LINE_REDUCTION", False)
    fallback = to_host(mc_spectrum(_long_flights(), grid, line_quadrature="bin-mean", **_KW))
    np.testing.assert_allclose(fused, fallback, rtol=scaled_rtol(1e-12, eps_multiple=64.0))


def test_components_are_supported_and_sum_to_the_total_parts():
    grid = _uniform_axis(3.0, 701.0, 1499.0)
    total = mc_spectrum(_long_flights(), grid, line_quadrature="bin-mean", **_KW)
    both = mc_spectrum(_long_flights(), grid, line_quadrature="bin-mean", components=True, **_KW)
    np.testing.assert_allclose(
        to_host(both[0]), to_host(total), rtol=scaled_rtol(1e-12, eps_multiple=8.0)
    )
    assert np.all(to_host(both[1]) >= 0.0) and np.all(to_host(both[2]) >= 0.0)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"coherent": True}, "incoherent-only"),
        ({"sinc_cutoff": 30.0}, "sinc_cutoff"),
        ({"line_quadrature": "trapezoid"}, "line_quadrature must be one of"),
    ],
)
def test_unsupported_combinations_are_refused(over, match):
    kw = {"line_quadrature": "bin-mean", **over}
    with pytest.raises(ValueError, match=match):
        mc_spectrum(_long_flights(), _uniform_axis(3.0, 701.0, 1499.0), **_KW, **kw)


@host_backend_only("flight-grouped reduction is host-only")
def test_numerical_substeps_are_refused():
    with pytest.raises(ValueError, match="numerical substeps"):
        mc_spectrum(
            _segments(flights=True),
            _uniform_axis(3.0, 701.0, 1499.0),
            line_quadrature="bin-mean",
            **_KW,
        )


def test_host_evaluator_blocks_do_not_change_the_result(monkeypatch):
    grid = _uniform_axis(3.0, 701.0, 1499.0)
    edges, inv_width = bq.bin_axis(grid)
    a = to_device(np.linspace(0.5, 50.0, 7), REAL)
    e_r = to_device(np.linspace(800.0, 1400.0, 7), REAL)
    whole = bq.sincsq_bin_lineshape(a, e_r, edges, inv_width)
    monkeypatch.setattr(bq, "_HOST_BLOCK_ELEMENTS", 2 * grid.size)
    np.testing.assert_array_equal(
        to_host(bq.sincsq_bin_lineshape(a, e_r, edges, inv_width)), to_host(whole)
    )


# Far-field envelope beyond BIN_MEAN_EXACT_WIDTHS (Validation: sinc-bin-far-envelope)


def _hybrid_population(seed=5, count=200):
    rng = np.random.default_rng(seed)
    width = np.exp(rng.uniform(np.log(0.05), np.log(30.0), count))
    return np.pi / width, rng.uniform(3000.0, 17000.0, count), width


@pytest.mark.parametrize("step", [0.5, 3.0])
@pytest.mark.parametrize("exact_widths", [16.0, bq.BIN_MEAN_EXACT_WIDTHS])
def test_far_envelope_yield_error_stays_inside_its_bound(step, exact_widths):
    """Per-line yield change <= 3/(2 pi^3 K^2); pointwise <= 1/(pi^2 K^2) of the peak."""
    grid = np.arange(0.0, 20000.0 + step / 2, step)
    edges, widths = _energy_bin_edges_and_widths(grid)
    a, e_r, width = _hybrid_population()
    exact = bq._host_bin_mean(a, e_r, edges, 1.0 / widths)
    hybrid = bq._host_bin_mean(a, e_r, edges, 1.0 / widths, exact_widths=exact_widths)
    yield_error = np.abs(((hybrid - exact) * widths).sum(axis=1)) / width
    assert yield_error.max() <= 3.0 / (2.0 * np.pi**3 * exact_widths**2)
    assert np.abs(hybrid - exact).max() <= 1.0 / (np.pi**2 * exact_widths**2)


def test_far_envelope_leaves_near_bins_exact_and_uses_the_closed_form_beyond():
    grid = np.arange(0.0, 2000.0, 0.5)
    edges, widths = _energy_bin_edges_and_widths(grid)
    a, e_r = np.array([np.pi / 2.0]), np.array([1000.3])
    exact = bq._host_bin_mean(a, e_r, edges, 1.0 / widths)[0]
    hybrid = bq._host_bin_mean(a, e_r, edges, 1.0 / widths, exact_widths=64.0)[0]
    x = (edges - e_r[0]) / 2.0
    far = (x[:-1] >= 64.0) | (x[1:] <= -64.0)
    assert far.any() and (~far).any()
    np.testing.assert_array_equal(hybrid[~far], exact[~far])
    np.testing.assert_allclose(
        hybrid[far], 1.0 / (2.0 * np.pi**2 * x[:-1][far] * x[1:][far]), rtol=1e-6
    )


def test_production_lineshape_defaults_to_the_far_envelope():
    grid = np.arange(0.0, 2000.0, 0.5)
    edges, inv_width = bq.bin_axis(grid)
    a, e_r = np.array([np.pi / 2.0]), np.array([1000.3])
    default = np.asarray(bq.sincsq_bin_lineshape(a, e_r, edges, inv_width))
    hybrid = bq._host_bin_mean(
        a, e_r, np.asarray(edges), np.asarray(inv_width), exact_widths=bq.BIN_MEAN_EXACT_WIDTHS
    )
    exact = np.asarray(bq.sincsq_bin_lineshape(a, e_r, edges, inv_width, exact_widths=None))
    np.testing.assert_allclose(default, hybrid.astype(default.dtype), rtol=1e-6)
    assert not np.array_equal(default, exact)


@pytest.mark.parametrize("exact_widths", [16.0, bq.BIN_MEAN_EXACT_WIDTHS])
def test_far_envelope_bound_holds_at_mev_resonances_on_narrow_lines(exact_widths):
    """E_res / w ~ 1.6e8: the resonance and edges must keep their float32
    remainders, or the far set shifts by ulp(E_res)/w widths (fresh-context
    verification, #192)."""
    # Far bins on one side only, so a shifted far set cannot cancel across sides.
    width = np.array([0.01, 0.02, 0.05])
    e_r = np.array([1_600_000.0123, 1_600_000.0567, 1_600_000.0901])
    grid = np.arange(1_599_999.9, 1_600_020.0, 0.005)
    edges, widths = _energy_bin_edges_and_widths(grid)
    a = np.pi / width
    exact = bq._host_bin_mean(a, e_r, edges, 1.0 / widths)
    hybrid = bq._host_bin_mean(a, e_r, edges, 1.0 / widths, exact_widths=exact_widths)
    yield_error = np.abs(((hybrid - exact) * widths).sum(axis=1)) / width
    assert yield_error.max() <= 3.0 / (2.0 * np.pi**3 * exact_widths**2)
    # A dropped edge remainder mostly telescopes out of the yield but not per bin.
    assert np.abs(hybrid - exact).max() <= 1.0 / (np.pi**2 * exact_widths**2)


def test_far_envelope_never_claims_the_core_of_a_sub_mev_width_line():
    """w = 0.1 meV at 1.6 MeV: ulp(E)/w ~ 1250 widths. Without the edge
    float32 remainders every edge collapses onto one head, ~123 widths from the
    resonance, and the resonance bins would classify as far."""
    width, e_r = np.array([0.0001]), np.array([1_600_000.01234])
    grid = np.arange(1_600_000.0, 1_600_000.05, 0.00005)
    edges, widths = _energy_bin_edges_and_widths(grid)
    a = np.pi / width
    exact = bq._host_bin_mean(a, e_r, edges, 1.0 / widths)
    hybrid = bq._host_bin_mean(a, e_r, edges, 1.0 / widths, exact_widths=64.0)
    assert np.abs(hybrid - exact).max() <= 1.0 / (np.pi**2 * 64.0**2)
