import numpy as np
import pytest

from pyrite.montecarlo import transport as t


def _synthetic_layer(two_elements=False):
    Z = np.array([6.0, 14.0] if two_elements else [6.0], dtype=np.float64)
    J = np.array([0.081, 0.173] if two_elements else [0.081], dtype=np.float64)
    k = 0.731 + 0.0688 * np.log10(Z)
    density_ang3 = np.array([0.10, 0.05] if two_elements else [0.10], dtype=np.float64)
    n_cm3 = density_ang3 * 1e24
    coeff = (density_ang3 / 0.602214076) * Z

    sr_rate_numer = 5.21e-21 * Z * Z * 4.0 * np.pi * n_cm3
    z17 = Z**1.7
    mott_numer = 3.0e-18 * z17 * n_cm3
    mott_denom1 = 0.005 * z17
    mott_denom2 = 0.0007 * Z * Z
    sr_joy_numer = 3.4e-3 * Z**0.67
    # rho, Z and A all cancel in the Joy-Luo/Berger-Seltzer ratio (both laws
    # carry the same rho Z/A), so the crossover depends only on Z -- through k --
    # and J. The mass number below is a placeholder that never reaches the answer.
    E_cross = np.array(
        [t._bs_joy_luo_crossover_keV(Z_i, 2.0 * Z_i, J_i) for Z_i, J_i in zip(Z, J, strict=True)],
        dtype=np.float64,
    )

    return (
        [J],
        [Z],
        [k],
        [coeff],
        [E_cross],
        [sr_rate_numer],
        [mott_numer],
        [mott_denom1],
        [mott_denom2],
        [sr_joy_numer],
    )


def _lut_coord(lut, E):
    """Vectorised twin of ``_lut_index_frac_scalar``: one log, one multiply."""
    x = np.clip((np.log(E) - lut.log_E_min) * lut.inv_dlogE, 0.0, lut.n_energy - 1.0)
    i = np.minimum(np.floor(x).astype(np.int64), lut.n_energy - 2)
    return i, x - i


def _interp_1d(table, lut, E):
    i, f = _lut_coord(lut, E)
    return table[i] + f * (table[i + 1] - table[i])


def _interp_2d(table, row, lut, E):
    i, f = _lut_coord(lut, E)
    return table[row, i] + f * (table[row, i + 1] - table[row, i])


def test_sr_transport_lut_matches_direct_scalar_physics():
    tables = _synthetic_layer()
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = tables
    lut = t.build_transport_energy_lut(
        5.0,
        30.0,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_xc,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None]],
        config=t.TransportLUTConfig(min_points=2),
    )

    rng = np.random.default_rng(1234)
    energies = rng.uniform(5.0, 30.0, 2000)
    rate_exact = np.array([t._scatter_rates_sr_scalar(E, L_sr[0][0], L_sj[0][0]) for E in energies])
    dEds_exact = np.array(
        [
            t._dEds_spliced_compound_scalar(L_Js[0], L_ks[0], L_coeffs[0], L_xc[0], 0.0, E)
            for E in energies
        ]
    )
    inv_beta_exact = 1.0 / np.array([t.beta_from_keV_scalar(E) for E in energies])

    np.testing.assert_allclose(_interp_2d(lut.total_rate, 0, lut, energies), rate_exact, rtol=1e-5)
    np.testing.assert_allclose(_interp_2d(lut.dEds, 0, lut, energies), dEds_exact, rtol=1e-5)
    np.testing.assert_allclose(_interp_1d(lut.inv_beta, lut, energies), inv_beta_exact, rtol=1e-5)


def test_transport_lut_uses_the_sbethe_material_table_for_stopping():
    tables = _synthetic_layer()
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = tables
    log_energy, log_stopping = t.prepare_sbethe_stopping_table(
        {
            "stopping_energy_eV": np.array([5.0e3, 10.0e3, 30.0e3]),
            "stopping_eV_per_angstrom": np.array([9.0, 6.0, 3.0]),
        }
    )
    lut = t.build_transport_energy_lut(
        5.0,
        30.0,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_xc,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None]],
        config=t.TransportLUTConfig(min_points=32),
        stopping_tables=[(log_energy, log_stopping)],
    )
    energies = np.exp(lut.log_E_min + np.arange(lut.n_energy) / lut.inv_dlogE)
    energies[0], energies[-1] = 5.0, 30.0

    np.testing.assert_allclose(
        lut.dEds[0],
        t.sbethe_stopping_keV_per_ang(log_energy, log_stopping, energies),
        rtol=2e-15,
    )


def test_transport_lut_cdf_is_normalized_and_monotone():
    tables = _synthetic_layer(two_elements=True)
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = tables
    lut = t.build_transport_energy_lut(
        5.0,
        60.0,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_xc,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None, None]],
        config=t.TransportLUTConfig(min_points=2),
    )

    live = lut.cdf[0, :2]
    assert np.all(np.diff(live, axis=0) >= 0.0)
    np.testing.assert_array_equal(live[-1], np.ones(lut.n_energy))


def test_lut_index_clamps_both_endpoints():
    log_E_min = np.log(5.0)
    i, f = t._lut_index_frac_scalar(1.0, log_E_min, 40.0, 101)
    assert i == 0
    assert f == 0.0

    i, f = t._lut_index_frac_scalar(1000.0, log_E_min, 40.0, 101)
    assert i == 99
    assert f == 1.0


def test_lut_index_is_exact_on_grid_nodes_and_linear_between():
    log_E_min = np.log(2.0)
    inv_dlogE = 40.0
    for node in (0, 1, 7, 99):
        E = np.exp(log_E_min + node / inv_dlogE)
        i, f = t._lut_index_frac_scalar(E, log_E_min, inv_dlogE, 101)
        assert (i, round(f, 9)) in ((node, 0.0), (node - 1, 1.0))
    E = np.exp(log_E_min + 7.25 / inv_dlogE)
    i, f = t._lut_index_frac_scalar(E, log_E_min, inv_dlogE, 101)
    assert i == 7
    assert f == pytest.approx(0.25, abs=1e-12)


def _build(E_min, E_max, two=False, **kwargs):
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = _synthetic_layer(two)
    return t.build_transport_energy_lut(
        E_min,
        E_max,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_xc,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None, None]] if two else [[None]],
        config=t.TransportLUTConfig(min_points=2, **kwargs),
    )


def test_grid_is_uniform_in_log_energy_and_hits_both_bounds():
    lut = _build(0.1, 1.0e5)
    nodes = np.exp(lut.log_E_min + np.arange(lut.n_energy) / lut.inv_dlogE)
    ratio = nodes[1:] / nodes[:-1]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    assert nodes[0] == pytest.approx(0.1, rel=1e-12)
    assert nodes[-1] == pytest.approx(1.0e5, rel=1e-12)
    # The default resolution is a target, and it is met to within one node.
    assert lut.intervals_per_decade == pytest.approx(1024.0, rel=1e-3)


def test_adjacent_nodes_stay_distinct_in_float32():
    lut = _build(0.1, 1.0e5)
    nodes = np.exp(lut.log_E_min + np.arange(lut.n_energy) / lut.inv_dlogE)
    narrowed = nodes.astype(np.float32)
    assert np.all(np.diff(narrowed) > 0.0)


def test_off_grid_values_track_direct_physics_across_every_decade():
    # The 5-30 keV test above does not certify six decades. Probe each decade
    # densely, on deliberately off-node energies, against the scalar physics the
    # direct cores evaluate.
    E_min, E_max = 0.1, 1.0e5
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = _synthetic_layer()
    lut = _build(E_min, E_max)

    rng = np.random.default_rng(99)
    for decade in range(-1, 5):
        lo, hi = 10.0**decade, 10.0 ** (decade + 1)
        E = np.exp(rng.uniform(np.log(lo), np.log(hi), 500))
        rate = np.array([t._scatter_rates_sr_scalar(e, L_sr[0][0], L_sj[0][0]) for e in E])
        dEds = np.array(
            [
                t._dEds_spliced_compound_scalar(L_Js[0], L_ks[0], L_coeffs[0], L_xc[0], 0.0, e)
                for e in E
            ]
        )
        inv_beta = 1.0 / np.array([t.beta_from_keV_scalar(e) for e in E])
        np.testing.assert_allclose(_interp_2d(lut.total_rate, 0, lut, E), rate, rtol=1e-5)
        np.testing.assert_allclose(_interp_2d(lut.dEds, 0, lut, E), dEds, rtol=5e-5)
        np.testing.assert_allclose(_interp_1d(lut.inv_beta, lut, E), inv_beta, rtol=1e-5)


def test_off_grid_values_track_direct_physics_at_the_cutoff_and_the_stopping_join():
    E_min, E_max = 0.1, 1.0e5
    L_Js, L_Zs, L_ks, L_coeffs, L_xc, L_sr, L_mn, L_d1, L_d2, L_sj = _synthetic_layer()
    lut = _build(E_min, E_max)

    # The Joy-Luo/Berger-Seltzer crossover is a kink, not a jump: the two laws
    # agree in value there, so no interval smooths across a discontinuity. The
    # straddling interval still loses one order of convergence, which is why
    # dE/ds carries a looser bound here than the smooth rates do.
    joins = np.asarray(L_xc[0], dtype=np.float64)
    offsets = np.array([-1e-3, -1e-5, 0.0, 1e-5, 1e-3])
    probes = np.concatenate(
        [
            joins[:, None] * (1.0 + offsets),
            np.array([[E_min, E_min * 1.001, E_max, E_max * 0.999]]),
        ],
        axis=None,
    )
    probes = np.clip(probes, E_min, E_max)
    dEds = np.array(
        [
            t._dEds_spliced_compound_scalar(L_Js[0], L_ks[0], L_coeffs[0], L_xc[0], 0.0, e)
            for e in probes
        ]
    )
    rate = np.array([t._scatter_rates_sr_scalar(e, L_sr[0][0], L_sj[0][0]) for e in probes])
    np.testing.assert_allclose(_interp_2d(lut.dEds, 0, lut, probes), dEds, rtol=5e-5)
    np.testing.assert_allclose(_interp_2d(lut.total_rate, 0, lut, probes), rate, rtol=1e-5)

    # Continuity of the spliced model at the join, to machine scale.
    for E_cross in joins:
        below = t._dEds_spliced_compound_scalar(
            L_Js[0], L_ks[0], L_coeffs[0], L_xc[0], 0.0, E_cross * (1.0 - 1e-12)
        )
        above = t._dEds_spliced_compound_scalar(
            L_Js[0], L_ks[0], L_coeffs[0], L_xc[0], 0.0, E_cross * (1.0 + 1e-12)
        )
        assert abs(above - below) / abs(below) < 1e-9


def test_endpoint_clamp_reads_the_terminal_intervals_exactly():
    lut = _build(1.0, 100.0)
    last = lut.n_energy - 1
    for E in (1e-30, 0.5, 1.0):
        i, f = t._lut_index_frac_scalar(E, lut.log_E_min, lut.inv_dlogE, lut.n_energy)
        assert (i, f) == (0, 0.0)
        assert _interp_1d(lut.inv_beta, lut, np.array([E]))[0] == lut.inv_beta[0]
    # Strictly outside the grid the upper clamp is exact and identical on both
    # backends, which is what the CPU/CUDA endpoint contract needs. Exactly at
    # E_max the coordinate is log(E_max) recomputed, which can land one ulp
    # short of the last node; the interpolant is then still the last node to
    # rounding, but the clamp branch is not the one taken.
    for E in (1e6, np.nextafter(100.0, np.inf) * 1.000001):
        i, f = t._lut_index_frac_scalar(E, lut.log_E_min, lut.inv_dlogE, lut.n_energy)
        assert (i, f) == (last - 1, 1.0)
        assert _interp_1d(lut.inv_beta, lut, np.array([E]))[0] == lut.inv_beta[last]
    i, f = t._lut_index_frac_scalar(100.0, lut.log_E_min, lut.inv_dlogE, lut.n_energy)
    assert i in (last - 1, last - 2)
    assert _interp_1d(lut.inv_beta, lut, np.array([100.0]))[0] == pytest.approx(
        lut.inv_beta[last], rel=1e-12
    )


def test_build_reports_an_unmet_tolerance_instead_of_coarsening_silently():
    with pytest.warns(t.TransportLUTToleranceWarning, match="capped the requested"):
        lut = _build(0.1, 1.0e5, max_points=64, rel_tol=1e-9)
    assert lut.n_energy == 64
    assert not lut.tolerance_met
    assert lut.max_rel_interp_error > 1e-9
    assert lut.intervals_per_decade < 1024.0


def test_build_meets_its_default_tolerance_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", t.TransportLUTToleranceWarning)
        lut = _build(0.1, 1.0e5, two=True)
    assert lut.tolerance_met
    assert lut.max_rel_interp_error <= 1e-4
    # The smooth tables are far inside the contract; the residual is the O(h)
    # term the stopping crossover leaves in the single interval straddling it.
    assert lut.max_rel_interp_error > 1e-6
    assert min(_synthetic_layer(True)[4][0]) * 0.99 < lut.worst_energy_keV


# ---- CUDA endpoint parity -----------------------------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")


@pytest.mark.hardware
@requires_cuda
def test_device_and_host_lut_reads_agree_bit_for_bit_including_the_clamps():
    from pyrite.montecarlo._cupy_jit import jit
    from pyrite.montecarlo.transport._jit_device import _lut_lerp_at

    lut = _build(0.5, 500.0)
    table = np.ascontiguousarray(lut.total_rate[0])

    @jit.rawkernel()
    def probe(table, n_energy, log_E_min, inv_dlogE, energies, out):
        i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        if i < energies.size:
            out[i] = _lut_lerp_at(table, np.int32(0), n_energy, log_E_min, inv_dlogE, energies[i])

    energies = np.array(
        [
            1e-6,  # far below the grid: lower clamp
            0.25,  # below E_min: lower clamp
            0.5,  # exactly E_min
            0.5000001,
            7.3,  # interior, deliberately off-node
            123.456,
            499.999,
            500.0,  # exactly E_max
            5000.0,  # above the grid: upper clamp
            1e12,
        ],
        dtype=np.float64,
    )

    d_out = cupy.empty(energies.size, dtype=cupy.float64)
    probe(
        (1,),
        (64,),
        (
            cupy.asarray(table),
            np.int32(lut.n_energy),
            np.float64(lut.log_E_min),
            np.float64(lut.inv_dlogE),
            cupy.asarray(energies),
            d_out,
        ),
    )
    gpu = cupy.asnumpy(d_out)

    cpu = np.empty_like(energies)
    for k, E in enumerate(energies):
        i, f = t._lut_index_frac_scalar(E, lut.log_E_min, lut.inv_dlogE, lut.n_energy)
        cpu[k] = t._lut_lerp_1d(table, i, f)

    # Both clamps must be exactly the terminal node, on both backends.
    assert gpu[0] == table[0]
    assert gpu[1] == table[0]
    assert gpu[-1] == table[lut.n_energy - 1]
    assert gpu[-2] == table[lut.n_energy - 1]
    np.testing.assert_array_equal(cpu[[0, 1, -2, -1]], gpu[[0, 1, -2, -1]])
    # The interior reads share the same arithmetic; only the device libm log
    # may differ, which stays far inside the table's own interpolation error.
    np.testing.assert_allclose(cpu, gpu, rtol=1e-12)


def test_degenerate_span_collapses_to_two_distinct_nodes():
    # E_max <= E_min is widened to one ulp; the requested point count then asks
    # for more nodes than the coordinate can separate, so the build backs off to
    # the finest grid whose adjacent nodes stay distinct instead of emitting a
    # table with repeated energies and an infinite inverse spacing.
    lut = _build(7.0, 7.0)
    assert lut.n_energy == 2
    assert np.isfinite(lut.inv_dlogE)
    assert lut.E_max_keV > lut.E_min_keV
