"""Support-windowed node reduction with the analytic far tail (#1191).

The windowed reductions are compared with the full-node reference kernels on
line sets and on contiguous substep flights (chirp, absorption, refraction).
With windows covering the whole grid they reproduce the reference to roundoff;
at ``x_c = 100`` they stay well inside the #1143 margins (yield/centroid 1e-3,
mass-L1 1e-2). Validation: node-support-window-tail
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._cpu_reduce import (
    accumulate_grouped_formation,
    accumulate_sincsq,
)
from pyrite.montecarlo.spectrum.cxr._node_window import (
    NodeTail,
    accumulate_grouped_windowed,
    accumulate_sincsq_windowed,
    grouped_window_plan,
    uniform_step,
)

HBARC = 1973.269804
E_BAND = np.arange(500.0, 4000.0, 0.09375)


def _yield_l1(ref, got):
    return got.sum() / ref.sum() - 1.0, np.abs(got - ref).sum() / ref.sum()


def _rows(E_grid, x_c, E_r, a, weights):
    refs = [np.zeros(E_grid.size) for _ in weights]
    accumulate_sincsq(E_r, a, weights, E_grid, refs)
    tail = NodeTail(E_grid, len(weights), x_c)
    got = [np.zeros(E_grid.size) for _ in weights]
    accumulate_sincsq_windowed(E_r, a, weights, E_grid, got, tail, range(len(weights)))
    tail.apply(got)
    return refs, got


def _contiguous_flights(rng, n_flights, *, chirp=0.01, mu_max=5e-4, max_rows=32):
    """Straight flights split into contiguous substeps, as the transport does.

    Per row: formation lines, phases referenced to the flight's first row, and
    two polarization coefficients proportional to the row duration.
    """
    keys = ("E_vac", "a_vac", "half_dL", "apb", "bma", "q", "d", "g", "L", "c1", "c2")
    cols = {k: [] for k in keys}
    starts = [0]
    for _ in range(n_flights):
        n = int(rng.integers(1, max_rows + 1))
        length = rng.uniform(200.0, 5000.0)
        beta0, cos, vg = rng.uniform(0.5, 0.9), rng.uniform(-0.5, 0.7), rng.uniform(0.2, 1.4)
        frac = rng.uniform(0.3, 1.3, n)
        Ls = frac / frac.sum() * length
        s0 = np.concatenate(([0.0], np.cumsum(Ls)[:-1]))
        beta = beta0 - rng.uniform(0.0, chirp) * (s0 + Ls / 2) / length
        T = Ls / beta
        t_mid = np.concatenate(([0.0], np.cumsum(T)[:-1])) + T / 2
        s_mid = s0 + Ls / 2
        L_start = np.maximum(rng.uniform(0, 3000) + rng.uniform(-0.8, 0.8) * s0, 0.0)
        L_end = np.maximum(L_start + rng.uniform(-0.8, 0.8) * Ls, 0.0)
        mu = rng.uniform(0.0, mu_max)
        a_end, b_end = np.exp(-mu * L_start / 2), np.exp(-mu * L_end / 2)
        d = t_mid - cos * s_mid
        L_mid = 0.5 * (L_start + L_end)
        amp = rng.uniform(0.5, 1.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        for k, v in (
            ("E_vac", HBARC * beta * vg / (1 - beta * cos)),
            ("a_vac", (1 - beta * cos) * T / (2 * HBARC)),
            ("half_dL", 0.5 * (L_end - L_start)),
            ("apb", a_end + b_end),
            ("bma", b_end - a_end),
            ("q", mu * (L_end - L_start) / 4),
            ("d", d - d[0]),
            ("g", vg * (s_mid - s_mid[0])),
            ("L", L_mid - L_mid[0]),
            ("c1", amp * T),
            ("c2", 0.3j * amp * T),
        ):
            cols[k].append(v)
        starts.append(starts[-1] + n)
    c = {k: np.concatenate(v) for k, v in cols.items()}
    lines = tuple(c[k] for k in keys[:6])
    return lines, c["d"], c["g"], c["L"], [c["c1"], c["c2"]], np.array(starts)


def _grids(E_grid):
    delta = 2e-5 * (1000.0 / E_grid) ** 2
    return E_grid, E_grid / HBARC, delta * E_grid / HBARC


def _grouped(case, grids, x_c):
    lines, d, g, L, coefs, starts = case
    ref = np.zeros(grids[0].size)
    accumulate_grouped_formation(lines, d, g, L, coefs, starts, grids, ref, 0.5)
    tail = NodeTail(grids[0], 1, x_c)
    got = np.zeros(grids[0].size)
    accumulate_grouped_windowed(lines, d, g, L, coefs, starts, grids, got, 0.5, tail)
    tail.apply([got])
    return ref, got


def test_uniform_step_accepts_float32_rounding_and_rejects_ragged_grids():
    E = np.arange(1000.0, 2000.0, 0.25)
    assert uniform_step(E) == (1000.0, pytest.approx(0.25))
    assert uniform_step(E.astype(np.float32)) is not None
    ragged = E.copy()
    ragged[10] += 0.05
    assert uniform_step(ragged) is None
    assert uniform_step(np.geomspace(1000.0, 2000.0, 100)) is None
    with pytest.raises(ValueError, match="uniform"):
        NodeTail(ragged, 1, 100.0)


def test_rows_match_full_node_within_tail_bound():
    """In-band, padding-region and far poles; three weight targets."""
    rng = np.random.default_rng(1191)
    n = 1500
    E_r = rng.uniform(-3000.0, 15000.0, n)
    a = np.pi / rng.uniform(1.0, 60.0, n)
    weights = [rng.uniform(0.5, 1.0, n) for _ in range(3)]
    refs, got = _rows(E_BAND, 100.0, E_r, a, weights)
    for ref, out in zip(refs, got, strict=True):
        dy, l1 = _yield_l1(ref, out)
        assert abs(dy) < 5e-5
        assert l1 < 3e-4


def test_rows_with_grid_wide_windows_reproduce_the_reference():
    rng = np.random.default_rng(7)
    E_r = rng.uniform(400.0, 4100.0, 300)
    a = np.pi / rng.uniform(1.0, 20.0, 300)
    refs, got = _rows(E_BAND, 1e7, E_r, a, [rng.uniform(0.5, 1.0, 300)])
    np.testing.assert_allclose(got[0], refs[0], rtol=0, atol=1e-9 * refs[0].max())


def test_far_pole_whose_window_reaches_the_band_is_evaluated_directly():
    """A wide line resonating far outside a short grid (direct mode)."""
    E = np.arange(1000.0, 1010.0, 0.1)
    E_r = np.array([1500.0, 1003.0, -600.0])
    a = np.array([0.02, 3.0, 0.05])
    refs, got = _rows(E, 100.0, E_r, a, [np.array([1.0, 1e-3, 1.0])])
    dy, l1 = _yield_l1(refs[0], got[0])
    assert abs(dy) < 1e-4 and l1 < 1e-3


def test_contiguous_flights_match_full_node_within_margins():
    rng = np.random.default_rng(1142)
    case = _contiguous_flights(rng, 120)
    grids = _grids(E_BAND)
    ref, got = _grouped(case, grids, 100.0)
    dy, l1 = _yield_l1(ref, got)
    E = grids[0]
    centroid = (got * E).sum() / got.sum() - (ref * E).sum() / ref.sum()
    assert abs(dy) < 5e-5
    assert l1 < 2e-3
    assert abs(centroid) < 1e-5 * (ref * E).sum() / ref.sum()


def test_grouped_window_grows_with_the_flight_not_its_substeps():
    """Splitting a straight flight leaves window, moments and output unchanged."""
    E = np.arange(1500.0, 2500.0, 0.05)
    grids = (E, E / HBARC, np.zeros(E.size))
    beta, cos, vg, length = 0.7, 0.2, 1.4, 3000.0

    def split(n):
        Ls = np.full(n, length / n)
        s_mid = (np.arange(n) + 0.5) * Ls
        T = Ls / beta
        d = s_mid / beta - cos * s_mid
        one = np.ones(n)
        lines = (
            HBARC * beta * vg / (1 - beta * cos) * one,
            (1 - beta * cos) * T / (2 * HBARC),
            0 * one,
            2 * one,
            0 * one,
            0 * one,
        )
        return lines, d - d[0], vg * (s_mid - s_mid[0]), 0 * one, [T + 0j], np.array([0, n])

    outs, plans = [], []
    for n in (1, 16, 1024):
        case = split(n)
        tail = NodeTail(E, 1, 100.0)
        plans.append(grouped_window_plan(tail, *case[:5], case[5], grids))
        out = np.zeros(E.size)
        accumulate_grouped_windowed(*case, grids, out, 1.0, tail)
        tail.apply([out])
        outs.append(out)
    for plan, out in zip(plans[1:], outs[1:], strict=True):
        for unsplit, substeps in zip(plans[0], plan, strict=True):
            np.testing.assert_allclose(substeps, unsplit, rtol=1e-9, atol=1e-12 * np.max(unsplit))
        np.testing.assert_allclose(out, outs[0], rtol=0, atol=1e-9 * outs[0].max())


def test_grouped_with_grid_wide_windows_reproduces_the_reference():
    rng = np.random.default_rng(3)
    case = _contiguous_flights(rng, 30, max_rows=8)
    ref, got = _grouped(case, _grids(E_BAND), 1e7)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-8 * ref.max())


def test_non_contiguous_rows_widen_their_windows():
    """Random rows break junction cancellation; the slope-gap guard widens W."""
    from tests.montecarlo.spectrum.test_grouped_formation_reduce import _case

    rng = np.random.default_rng(1152)
    lines, d, g, L, coefs, bounds, grids = _case(rng, n_E=3001)
    ref, got = _grouped((lines, d, g, L, coefs, bounds), grids, 100.0)
    dy, l1 = _yield_l1(ref, got)
    assert abs(dy) < 1e-5 and l1 < 1e-4


@pytest.mark.parametrize("substeps", [False, True])
def test_spectrum_option_matches_full_node(substeps):
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum, subdivide_flights
    from tests.montecarlo.spectrum.test_substep_invariance import (
        CARBON,
        CXR_KWARGS,
        _flights,
    )

    grid = np.arange(700.0, 1300.0, 0.05)
    rows = _flights(Ne=6)
    if substeps:
        rows, _ = subdivide_flights(rows, composition=CARBON, max_dE_frac=1e-3)
    kwargs = dict(CXR_KWARGS, components=not substeps)
    ref = mc_cxr_spectrum(rows, grid, node_support_cutoff=None, **kwargs)
    got = mc_cxr_spectrum(rows, grid, node_support_cutoff=100.0, **kwargs)
    for r, o in zip(np.atleast_2d(ref), np.atleast_2d(got), strict=True):
        dy, l1 = _yield_l1(r, o)
        assert abs(dy) < 1e-4 and l1 < 2e-3


def test_default_is_automatic_window_and_none_is_reference():
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum, subdivide_flights
    from tests.montecarlo.spectrum.test_substep_invariance import (
        CARBON,
        CXR_KWARGS,
        _flights,
    )

    grid = np.arange(700.0, 1300.0, 0.05)
    rows, _ = subdivide_flights(_flights(Ne=4), composition=CARBON, max_dE_frac=1e-3)
    kwargs = dict(CXR_KWARGS, components=False)
    default = mc_cxr_spectrum(rows, grid, **kwargs)
    np.testing.assert_array_equal(
        default, mc_cxr_spectrum(rows, grid, node_support_cutoff="auto", **kwargs)
    )
    ref = mc_cxr_spectrum(rows, grid, node_support_cutoff=None, **kwargs)
    assert not np.array_equal(default, ref)
    dy, l1 = _yield_l1(ref, default)
    assert abs(dy) < 1e-4 and l1 < 2e-3


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"node_support_cutoff": 0.0}, "finite and positive"),
        ({"node_support_cutoff": 100.0, "coherent": True}, "incoherent"),
        ({"node_support_cutoff": 100.0, "sinc_cutoff": 50.0}, "exclusive"),
        ({"node_support_cutoff": 100.0, "line_quadrature": "bin-mean"}, "node"),
    ],
)
def test_spectrum_option_refusals(extra, match):
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum
    from tests.montecarlo.spectrum.test_substep_invariance import (
        CXR_GRID_EV,
        CXR_KWARGS,
        _flights,
    )

    with pytest.raises(ValueError, match=match):
        mc_cxr_spectrum(_flights(Ne=2), CXR_GRID_EV, **CXR_KWARGS, **extra)


def test_grouped_affine_dispersion_uses_effective_slope_and_pole():
    # Independent identity: v=(a-h*b)E-a*Ev-h*c, not a*(E-Ek).
    E = np.linspace(500.0, 1500.0, 10001)
    lines = tuple(np.array([v]) for v in (990.0, 1.0, 10000.0, 2.0, 0.0, 0.0))
    zero = np.zeros(1)
    tail = NodeTail(E, 1, 100.0)
    plan = grouped_window_plan(
        tail,
        lines,
        zero,
        zero,
        zero,
        [np.ones(1, dtype=complex)],
        np.array([0, 1]),
        (E, E / HBARC, 1e-6 * E),
    )
    node, _, _, _, m2, m3, m4 = plan
    assert tail.E0 + node[0] * tail.step == pytest.approx(1000.0)
    assert m2[0] == pytest.approx(0.5 / 0.99**2)
    assert m3[0] == pytest.approx(0.0, abs=1e-10)
    assert m4[0] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("small_cutoff", [0.5, 1.0])
def test_grouped_small_cutoff_uses_full_reference(small_cutoff):
    case = _contiguous_flights(np.random.default_rng(91), 3, max_rows=3)
    ref, got = _grouped(case, _grids(np.linspace(500.0, 4000.0, 1001)), small_cutoff)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * ref.max())


def test_grouped_nonpositive_affine_slope_uses_full_reference():
    E = np.linspace(500.0, 1500.0, 1001)
    lines = tuple(np.array([v]) for v in (990.0, 1.0, 1e6, 2.0, 0.0, 0.0))
    zero = np.zeros(1)
    case = lines, zero, zero, zero, [np.ones(1, dtype=complex)], np.array([0, 1])
    ref, got = _grouped(case, (E, E / HBARC, 2e-6 * E), 100.0)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * ref.max())


def test_stationary_dispersion_beat_forces_full_grid():
    # The beat derivative .1 - 1000*d'(E) crosses zero inside the band,
    # even though its band-centred secant derivative is .1.
    E = np.linspace(1000.0, 2000.0, 1001)
    lines = tuple(np.full(2, v) for v in (1500.0, 1.0, 0.0, 2.0, 0.0, 0.0))
    d = np.array([0.0, 0.1 * HBARC])
    L = np.array([0.0, 1000.0])
    grids = E, E / HBARC, 1e-6 * (E - 1500.0) ** 2
    case = lines, d, np.zeros(2), L, [np.ones(2, dtype=complex)], np.array([0, 2])
    plan = grouped_window_plan(NodeTail(E, 1, 100.0), *case, grids)
    assert plan[1][0] == 0 and plan[2][0] == E.size - 1
    ref, got = _grouped(case, grids, 100.0)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * ref.max())


def test_automatic_policy_retains_exact_unresolved_node_samples():
    # Integer-pi samples have zero wings; replacing these by the cycle average
    # would introduce a finite bias, so automatic scoring must keep them exact.
    E = np.arange(500.0, 1505.0, 5.0)
    er, a = np.array([1000.0]), np.array([np.pi / 5])
    weights = [np.ones(1)]
    ref = [np.zeros(E.size)]
    accumulate_sincsq(er, a, weights, E, ref)
    tail = NodeTail(E, 1, 100.0)
    tail.resolved_only = True
    got = [np.zeros(E.size)]
    accumulate_sincsq_windowed(er, a, weights, E, got, tail, [0])
    tail.apply(got)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "extra", [{"coherent": True}, {"sinc_cutoff": 50.0}, {"line_quadrature": "bin-mean"}]
)
def test_automatic_policy_keeps_unsupported_routes_on_reference(extra):
    from pyrite.montecarlo.spectrum.cxr._setup import (
        SpectrumRequest,
        _node_tail,
        _refuse_node_support_cutoff,
    )

    E = np.arange(500.0, 1500.0, 1.0)
    request = SpectrumRequest(
        segments={},
        E_grid_eV=E,
        crystal="hopg",
        hkl_list=[],
        node_support_cutoff="auto",
        **extra,
    )
    _refuse_node_support_cutoff(request)
    assert _node_tail(request, E) is None
