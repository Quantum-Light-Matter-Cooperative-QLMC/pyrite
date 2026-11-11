"""Planar fibre unraveling vs the planar Lindblad operators and density-matrix solver.

Oracles are independent planar code: ``operators._site_densities``, the
full-transfer ``C(x, x)`` Toeplitz and binned kernels of
``tds_kernel``/``electronic_kernel``, ``overflow.overflow_spectrum`` and
``DensityMatrixSolver.propagate``.
"""

from functools import cache

import numpy as np
import pytest
from scipy import stats

from pyrite.materials.channeling.potential import planar_profile
from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
    _Engine,
    _fold,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_tds_continuum import TailRecord, string_jump
from pyrite.montecarlo.channeling.density import DensityMatrixSolver
from pyrite.montecarlo.channeling.operators import (
    _site_densities,
    electronic_kernel,
    tds_kernel,
)
from pyrite.montecarlo.channeling.overflow import overflow_spectrum
from pyrite.montecarlo.channeling.planar_fiber import planar_fiber_inputs, plane_sites

ENERGY = 5.0e6
BUILDERS = {"tds": tds_kernel, "electronic": electronic_kernel}


@cache
def _profile(crystal="silicon", hkl=(1, 1, 0)):
    return planar_profile(crystal, hkl, 293.15)


@pytest.mark.parametrize(
    ("crystal", "hkl"), [("silicon", (1, 1, 0)), ("silicon", (1, 0, 0)), ("diamond", (1, 1, 1))]
)
def test_plane_sites_reproduce_site_densities(crystal, hkl):
    profile = _profile(crystal, hkl)
    sites = plane_sites(profile)
    reference = _site_densities(profile, 6)
    g = profile.g_inv_ang
    for h in range(-6, 7):
        mine = sum(s.density_inv_ang3 * np.exp(-1j * h * g * s.position_ang) for s in sites)
        assert mine == pytest.approx(sum(v[h + 6] for v in reference.values()), abs=1e-14)
    assert all(0.0 <= s.position_ang < profile.period_ang for s in sites)
    assert len({s.position_ang for s in sites}) == len(sites)  # coincident planes merged


@pytest.mark.parametrize("mechanism", ["tds", "electronic"])
@pytest.mark.parametrize("hkl", [(1, 1, 0), (1, 0, 0)])
def test_loss_matches_full_transfer_planar_diagonal(mechanism, hkl):
    """Strings/kicks give the planar ``C(x, x)`` Toeplitz with every transfer."""
    profile = _profile("silicon", hkl)
    inputs = planar_fiber_inputs(profile, ENERGY, order_max=6, mechanisms=(mechanism,))
    loss = _Engine(inputs.channels, inputs.hbar_v_eV_ang).loss
    kernel = BUILDERS[mechanism](profile, ENERGY, n_cells=2, max_order=6)
    reference = DensityMatrixSolver(profile, ENERGY, kernel)._loss_toeplitz
    np.testing.assert_allclose(loss, reference, rtol=0, atol=1e-6 * np.abs(reference).max())


def _coherent_state(inputs):
    """Normalized superposition of the three lowest k = 0 states with distinct phases."""
    energies, vectors = inputs.channels.spectrum((0.0, 0.0))
    chi = vectors[:, :3] @ np.array([0.7, 0.5j, -0.5])
    return chi / np.linalg.norm(chi)


def _expanded_rates(profile, mechanism, chi, order, cells, transfer_order):
    """Planar-kernel gain per output wavevector bin for |chi><chi| at kappa = 0."""
    kernel = BUILDERS[mechanism](profile, ENERGY, n_cells=cells, max_order=transfer_order)
    size = 2 * transfer_order + 1
    padded = np.zeros(size, dtype=complex)
    padded[transfer_order - order : transfer_order + order + 1] = chi
    plane = np.zeros((cells, size, size), dtype=complex)
    plane[0] = np.outer(padded, padded.conj())
    spectrum = overflow_spectrum(plane, kernel, reference_kernel=kernel)
    width = profile.g_inv_ang / cells
    bins = np.rint(spectrum.wavevectors_inv_ang / width).astype(int).ravel()
    return bins, spectrum.rates_per_ang.ravel(), spectrum.full_loss_rate


def test_tds_jump_momentum_law_matches_overflow_spectrum():
    """Sampled string outputs bin onto the planar kernel's coherent gain."""
    profile, order, cells = _profile(), 4, 8
    inputs = planar_fiber_inputs(profile, ENERGY, order_max=order, mechanisms=("tds",))
    engine = _Engine(inputs.channels, inputs.hbar_v_eV_ang)
    chi = _coherent_state(inputs)
    bins, rates, full_loss = _expanded_rates(profile, "tds", chi, order, cells, 40)
    rates_by_bin = {b: rates[bins == b].sum() for b in np.unique(bins)}
    string_rates = np.array([channel.rate(chi) for channel in engine.strings])
    assert string_rates.sum() == pytest.approx(full_loss, rel=1e-5)
    rng, n = np.random.default_rng(11), 24_000
    width = profile.g_inv_ang / cells
    drawn = []
    for _ in range(n):
        channel = engine.strings[rng.choice(len(string_rates), p=string_rates / string_rates.sum())]
        q_rel, _ = channel.sample(chi, rng)
        drawn.append(int(np.rint(q_rel[0] / width)))
    counts = {b: c for b, c in zip(*np.unique(drawn, return_counts=True), strict=True)}
    expected = {b: n * r / full_loss for b, r in rates_by_bin.items()}
    keys = [b for b, e in expected.items() if e >= 20.0]
    observed = np.array([counts.get(b, 0) for b in keys], dtype=float)
    mean = np.array([expected[b] for b in keys])
    tail_obs, tail_exp = n - observed.sum(), n - mean.sum()
    chi2 = float(np.sum((observed - mean) ** 2 / mean) + (tail_obs - tail_exp) ** 2 / tail_exp)
    assert stats.chi2.sf(chi2, len(keys)) > 1e-3


def _kick_rates_binned(inputs, chi, width):
    """Exact azimuth-averaged kick gain per output bin (arcsine law of q cos(theta))."""
    channels = inputs.channels
    g = channels.reciprocal_inv_ang[0]
    out: dict[int, float] = {}
    for column, q in zip(channels.kick_columns[:, :, 0], channels.kick_q_inv_ang, strict=True):
        amplitude = np.convolve(chi, column)
        half = (amplitude.size - 1) // 2
        for o, rate in zip(range(-half, half + 1), np.abs(amplitude) ** 2, strict=True):
            centre = g * o
            lo = int(np.floor((centre - q) / width - 0.5))
            hi = int(np.ceil((centre + q) / width + 0.5))
            edges = (np.arange(lo, hi + 1) + 0.5) * width - centre
            cdf = 1.0 - np.arccos(np.clip(edges / q, -1.0, 1.0)) / np.pi
            for b, mass in zip(range(lo + 1, hi + 1), np.diff(cdf), strict=True):
                out[b] = out.get(b, 0.0) + rate * mass
    return out


def test_kick_momentum_law_matches_planar_electronic_kernel():
    profile, order, cells = _profile(), 4, 8
    inputs = planar_fiber_inputs(profile, ENERGY, order_max=order, mechanisms=("electronic",))
    chi = _coherent_state(inputs)
    width = profile.g_inv_ang / cells
    bins, rates, full_loss = _expanded_rates(profile, "electronic", chi, order, cells, 24)
    mine = _kick_rates_binned(inputs, chi, width)
    assert sum(mine.values()) == pytest.approx(full_loss, rel=1e-6)
    reference = {b: rates[bins == b].sum() for b in np.unique(bins)}
    support = sorted(set(reference) | {b for b, v in mine.items() if v > 1e-6 * full_loss})
    a = np.array([mine.get(b, 0.0) for b in support])
    r = np.array([reference.get(b, 0.0) for b in support])
    # Different q quadratures (exact nodes vs binned Abel weights on a log-q
    # table); the total-variation distance bounds the law difference.
    assert 0.5 * np.abs(a / a.sum() - r / r.sum()).sum() < 0.02


def _compare_with_solver(mechanisms, n, seed_offset):
    profile, order, margin, depth = _profile(), 16, 30.0, 3000.0
    kernel = None
    for mechanism in mechanisms:
        part = BUILDERS[mechanism](profile, ENERGY, n_cells=32, max_order=order)
        kernel = part if kernel is None else kernel + part
    solver = DensityMatrixSolver(profile, ENERGY, kernel, margin_eV=margin)
    result = solver.propagate(depth, 10.0, record_every=100)
    inputs = planar_fiber_inputs(
        profile, ENERGY, order_max=order, margin_eV=margin, mechanisms=mechanisms
    )
    assert inputs.cutoff_eV == pytest.approx(solver.cutoff_eV)
    window = np.zeros((2 * order + 1, 1), dtype=complex)
    window[0, 0] = 1.0
    events = unravel_fiber_outgoing(
        inputs.channels,
        inputs.entry_orders,
        carrier_inv_ang=inputs.carrier_inv_ang,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        window=window,
        max_depth_ang=depth,
        incident_ids=range(seed_offset, seed_offset + n),
        seed=3,
        search_step_ang=100.0,
    )
    weight = events[0].weight
    assert weight == pytest.approx(1.0 - result.entry_flux, abs=1e-6)
    depths = np.array([e.depth_ang for e in events])
    kinds = np.array([e.kind for e in events])
    for z, trace, free, overflow in zip(
        result.z_ang[1:],
        result.trace[1:],
        result.free_flux[1:],
        result.overflow_flux[1:],
        strict=True,
    ):
        tallies = {
            "trace": (trace, np.mean((kinds == "retained") | (depths > z))),
            "free": (free, np.mean((kinds == "outgoing") & (depths <= z))),
            "overflow": (overflow, np.mean(np.isin(kinds, ("unresolved", "tail")) & (depths <= z))),
        }
        for name, (dense, fraction) in tallies.items():
            sigma = np.sqrt(max(dense * (1.0 - dense), 1e-4 / n) / n)
            assert abs(weight * fraction - dense) < 4.5 * sigma, (name, z)


def test_unraveling_matches_density_matrix_solver_tds():
    """Retained trace, in-window free flux and overflow at M = 16 (window conventions negligible)."""
    _compare_with_solver(("tds",), 5000, 0)


@pytest.mark.slow
def test_unraveling_matches_density_matrix_solver_tds_electronic():
    _compare_with_solver(("tds", "electronic"), 20_000, 50_000)


def test_tail_records_carry_out_of_plane_momentum():
    """Out-of-window string outputs keep the exact (k_x, q_y) draw; nothing folds in y."""
    profile, order = _profile(), 2
    inputs = planar_fiber_inputs(profile, ENERGY, order_max=order, mechanisms=("tds",))
    engine = _Engine(inputs.channels, inputs.hbar_v_eV_ang)
    rng = np.random.default_rng(4)
    chi = inputs.entry_orders
    tails = 0
    for _ in range(2000):
        result = string_jump(
            engine.strings[0],
            chi,
            (0.0, 0.0),
            (order, 0),
            (order, 0),
            rng,
            lambda moved: _fold(moved, engine.g),
        )
        if isinstance(result, TailRecord):
            tails += 1
            assert abs(result.momentum_inv_ang[0]) > (order + 0.5) * profile.g_inv_ang * (1 - 1e-12)
        else:
            kappa, out = result
            assert abs(kappa[0]) <= 0.5 * profile.g_inv_ang
            assert out.shape == (2 * order + 1,)
    assert tails > 0


def test_refusals():
    profile = _profile()
    with pytest.raises(ValueError, match="outgoing_order_max"):
        planar_fiber_inputs(profile, ENERGY, order_max=4, outgoing_order_max=3)
    with pytest.raises(ValueError, match="unknown mechanisms"):
        planar_fiber_inputs(profile, ENERGY, order_max=4, mechanisms=("phonon",))
    with pytest.raises(ValueError, match="entrance order"):
        planar_fiber_inputs(profile, ENERGY, order_max=1, incidence_rad=0.05)
