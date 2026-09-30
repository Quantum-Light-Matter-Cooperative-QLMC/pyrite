"""ELSEPA tabulated elastic sampling (issue #89).

Validation: elsepa-elastic-sampling
"""

from pathlib import Path

import numpy as np
import pytest

from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.scattering import (
    _elsepa_rate_scalar,
    _sample_cos_theta_elsepa,
    elsepa_angular_pdf,
    pack_elsepa_tables,
)
from pyrite.xsgen.elsepa import parse_dcs, table_arrays


@pytest.fixture(scope="module")
def reference():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "xsgen"
        / "elsepa"
        / "test-run-output"
        / "dcs_1p000e03.dat"
    )
    return parse_dcs(path.read_bytes())


def _sr_sigma_cm2(Z, E_keV):
    """Total cross section of PyRITE's analytic screened-Rutherford model."""
    alpha = 3.4e-3 * Z**0.67 / E_keV
    rel = (E_keV + 511.0) / (E_keV + 1024.0)
    return 5.21e-21 * Z * Z * 4.0 * np.pi / E_keV**2 / (alpha * (1.0 + alpha)) * rel**2


def _sr_table(Z, energies_keV, mu):
    """ELSEPA-format table holding the analytic screened-Rutherford model.

    ``dsigma/dOmega = sigma/(4 pi) * a(1+a)/(mu+a)^2`` integrates to
    ``sigma`` over ``dOmega = 4 pi dmu``, the distribution the ``"sr"`` model
    samples by exact inversion.
    """
    E = np.asarray(energies_keV, dtype=float)
    alpha = 3.4e-3 * Z**0.67 / E
    sigma = _sr_sigma_cm2(Z, E)
    dcs = (sigma / (4.0 * np.pi))[:, None] * (alpha * (1.0 + alpha))[:, None]
    dcs = dcs / (mu[None, :] + alpha[:, None]) ** 2
    return {"energy_eV": E * 1e3, "total_elastic_cm2": sigma, "mu": mu, "dcs_cm2_sr": dcs}


def test_pdf_cdf_match_the_stored_table_cdf(reference):
    pdf, cdf = elsepa_angular_pdf(reference.mu, reference.dcs_cm2_sr)
    stored = table_arrays([reference], energies_ev=[1_000.0])["angular_cdf"]

    np.testing.assert_allclose(cdf, stored, rtol=0.0, atol=1e-14)
    assert np.trapezoid(pdf[0], reference.mu) == pytest.approx(1.0, rel=1e-12)


def _group(table, n_cm3=1.0):
    return pack_elsepa_tables([[table]], [np.array([n_cm3])])


def _two_node_table(reference):
    energy = np.array([1_000.0, 2_000.0])
    return {
        "energy_eV": energy,
        "total_elastic_cm2": np.array([4.0e-16, 1.0e-16]),
        "mu": reference.mu,
        "dcs_cm2_sr": np.vstack([reference.dcs_cm2_sr, reference.dcs_cm2_sr]),
    }


def test_rate_is_log_log_between_nodes_and_exact_at_them(reference):
    has, start, length, logE, log_rate, *_ = _group(_two_node_table(reference), n_cm3=5e22)

    at_node = _elsepa_rate_scalar(1.0, logE, log_rate, start[0, 0], length[0, 0])
    geometric = _elsepa_rate_scalar(np.sqrt(2.0), logE, log_rate, start[0, 0], length[0, 0])

    assert has[0, 0]
    assert at_node == pytest.approx(5e22 * 4.0e-16, rel=1e-12)
    assert geometric == pytest.approx(5e22 * 2.0e-16, rel=1e-12)


def test_inversion_reproduces_the_tabulated_cdf(reference):
    """Kolmogorov distance of exact inversion is at the sampling noise floor."""
    group = _group(_two_node_table(reference))
    _, start, length, logE, _, cdf, pdf, mu = group
    rng = np.random.default_rng(89)
    xi = rng.random(200_000)

    samples = np.array(
        [
            0.5
            * (
                1.0
                - _sample_cos_theta_elsepa(1.0, x, logE, cdf, pdf, mu, start[0, 0], length[0, 0])
            )
            for x in xi
        ]
    )
    empirical = np.searchsorted(np.sort(samples), mu, side="right") / samples.size

    # 1.63/sqrt(N): the 99% Kolmogorov critical value; the seed is fixed.
    assert np.max(np.abs(empirical - cdf[0])) < 1.63 / np.sqrt(samples.size)


def test_sampling_is_monotone_in_the_uniform_and_exact_at_the_endpoints(reference):
    _, start, length, logE, _, cdf, pdf, mu = _group(_two_node_table(reference))
    xi = np.linspace(0.0, 1.0 - 1e-12, 4001)

    cos_t = np.array(
        [
            _sample_cos_theta_elsepa(1.5, x, logE, cdf, pdf, mu, start[0, 0], length[0, 0])
            for x in xi
        ]
    )

    assert np.all(np.diff(cos_t) <= 0.0)
    assert cos_t[0] == pytest.approx(1.0)
    assert cos_t[-1] >= -1.0


def test_first_moment_matches_the_analytic_screened_rutherford_value(reference):
    """Limiting case: an SR DCS table reproduces 2a[(1+a)ln(1+1/a) - 1]."""
    Z, E_keV = 14.0, 30.0
    table = _sr_table(Z, [E_keV, 2 * E_keV], reference.mu)
    _, start, length, logE, _, cdf, pdf, mu = _group(table)
    alpha = 3.4e-3 * Z**0.67 / E_keV
    expected = 2.0 * alpha * ((1.0 + alpha) * np.log1p(1.0 / alpha) - 1.0)
    xi = (np.arange(400_000) + 0.5) / 400_000  # stratified: deterministic quadrature

    one_minus_cos = np.array(
        [
            1.0 - _sample_cos_theta_elsepa(E_keV, x, logE, cdf, pdf, mu, start[0, 0], length[0, 0])
            for x in xi
        ]
    ).mean()

    assert one_minus_cos == pytest.approx(expected, rel=5e-3)


def test_sampled_moments_reproduce_elsepas_own_transport_cross_sections(reference):
    """Real DCS: ``<1 - P_l(cos theta)> = sigma_l / sigma`` for l = 1, 2.

    ELSEPA reports ``sigma_l = 2 pi int [1 - P_l(cos theta)] DCS sin(theta) dtheta``;
    with ``mu = (1 - cos theta)/2`` the weights are ``2 mu`` and
    ``6 mu (1 - mu)``. Stratified draws make this a deterministic quadrature;
    the Hg 1 keV panel agrees to about 2e-5, the native-grid trapezoid floor.
    """
    _, start, length, logE, _, cdf, pdf, mu = _group(_two_node_table(reference))
    xi = (np.arange(100_000) + 0.5) / 100_000
    sampled_mu = np.array(
        [
            0.5
            * (
                1.0
                - _sample_cos_theta_elsepa(1.0, x, logE, cdf, pdf, mu, start[0, 0], length[0, 0])
            )
            for x in xi
        ]
    )

    sigma = reference.total_elastic_cm2
    assert (2.0 * sampled_mu).mean() == pytest.approx(reference.transport1_cm2 / sigma, rel=1e-4)
    assert (6.0 * sampled_mu * (1.0 - sampled_mu)).mean() == pytest.approx(
        reference.transport2_cm2 / sigma, rel=1e-4
    )


def test_packing_rejects_mismatched_grids_and_layers(reference):
    table = _two_node_table(reference)
    other = dict(table, mu=reference.mu * 0.5, dcs_cm2_sr=table["dcs_cm2_sr"])

    with pytest.raises(ValueError, match="angular grid"):
        pack_elsepa_tables([[table, other]], [np.array([1.0, 1.0])])
    with pytest.raises(ValueError, match="one table per element"):
        pack_elsepa_tables([[table]], [np.array([1.0, 1.0])])


_SI_N = 0.04994


def _run(core, **kwargs):
    return simulate_trajectories(
        30.0,
        4000,
        2.0e5,
        composition=[("Si", _SI_N)],
        E_cut_keV=5.0,
        seed=11,
        transport_core=core,
        **kwargs,
    )


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
def test_sr_shaped_elsepa_tables_reproduce_sr_transport(reference, core):
    """End to end: identical physics through the two samplers agrees statistically."""
    energies = np.geomspace(4.0, 40.0, 41)
    table = _sr_table(14.0, energies, reference.mu)

    sr = _run(core, elastic_model="sr")
    elsepa = _run(core, elastic_model="elsepa", elastic_tables=[[table]])

    n = sr["Ne"]
    p_sr, p_el = sr["n_backscattered"] / n, elsepa["n_backscattered"] / n
    sigma = np.sqrt(p_sr * (1 - p_sr) / n + p_el * (1 - p_el) / n)
    assert abs(p_sr - p_el) < 4.0 * sigma, (p_sr, p_el)
    mean_sr = sr["L_ang"].sum() / n
    mean_el = elsepa["L_ang"].sum() / n
    assert mean_el == pytest.approx(mean_sr, rel=0.03)


def test_elsepa_requires_tables_and_covering_energies(reference):
    table = _sr_table(14.0, np.geomspace(6.0, 40.0, 11), reference.mu)

    with pytest.raises(ValueError, match="elastic_tables is required"):
        _run("lockstep", elastic_model="elsepa")
    with pytest.raises(ValueError, match="elastic_tables is required"):
        _run("lockstep", elastic_model="sr", elastic_tables=[[table]])
    with pytest.raises(ValueError, match="within ELSEPA table"):
        _run("lockstep", elastic_model="elsepa", elastic_tables=[[table]])


def test_flight_diagnostics_use_the_elsepa_hazard(reference):
    table = _sr_table(14.0, np.geomspace(4.0, 40.0, 41), reference.mu)

    out = _run(
        "lockstep", elastic_model="elsepa", elastic_tables=[[table]], collect_diagnostics=True
    )

    summary = out["transport_diagnostics"]
    assert summary["n_flights"] == out["L_ang"].size
    assert np.isfinite(summary["relative_hazard_change"]["max"])


def _require_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        has_cuda = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        has_cuda = False
    if not has_cuda:
        pytest.skip("no CUDA device")


@pytest.mark.hardware
def test_cuda_elsepa_first_step_matches_the_cpu_reference(reference):
    """Same stream, same draw order: the first row agrees to a few ulp."""
    _require_cuda()
    table = _sr_table(14.0, np.geomspace(4.0, 40.0, 41), reference.mu)
    kwargs = dict(elastic_model="elsepa", elastic_tables=[[table]])

    cpu = _run("per-electron", **kwargs)
    gpu = _run("cuda", **kwargs)

    first_cpu = np.flatnonzero(np.r_[True, np.diff(cpu["electron_id"]) != 0])
    first_gpu = np.flatnonzero(np.r_[True, np.diff(gpu["electron_id"]) != 0])
    np.testing.assert_allclose(
        gpu["L_ang"][first_gpu], cpu["L_ang"][first_cpu], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        gpu["v_hat"][first_gpu[1:] - 1], cpu["v_hat"][first_cpu[1:] - 1], rtol=0.0, atol=1e-9
    )


@pytest.mark.hardware
def test_cuda_elsepa_agrees_statistically_with_the_cpu_core(reference):
    _require_cuda()
    table = _sr_table(14.0, np.geomspace(4.0, 40.0, 41), reference.mu)
    kwargs = dict(elastic_model="elsepa", elastic_tables=[[table]])

    cpu = _run("per-electron", **kwargs)
    gpu = _run("cuda", **kwargs)

    n = cpu["Ne"]
    p_cpu, p_gpu = cpu["n_backscattered"] / n, gpu["n_backscattered"] / n
    sigma = np.sqrt(p_cpu * (1 - p_cpu) / n + p_gpu * (1 - p_gpu) / n)
    assert abs(p_cpu - p_gpu) < 4.0 * sigma
    assert gpu["L_ang"].sum() / n == pytest.approx(cpu["L_ang"].sum() / n, rel=0.03)
