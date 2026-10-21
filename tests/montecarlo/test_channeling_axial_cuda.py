"""Axial CUDA device parity, trajectory streams and callback contract (#620)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("real CUDA device required")
    return cp


def _case(shape=(12, 12), mechanisms=("tds", "electronic")):
    profile = axial_profile("silicon", (0, 0, 1))
    dissipator = axial_dissipator(profile, 1e6, (1, 1), shape, mechanisms=mechanisms)
    return dissipator, profile.on_grid((1, 1), shape)


def test_backend_and_callback_refusals():
    from pyrite.montecarlo.channeling._axial_cuda import _callback_check

    d, potential = _case()
    with pytest.raises(ValueError, match="backend"):
        AxialMCWF(d, 1e6, potential, backend="sycl")
    solver = AxialMCWF(d, 1e6, potential, backend="cuda")
    with pytest.raises(ValueError, match="cuda_compatible"):
        solver.run(states.gaussian(waist_ang=0.6), [0], 1, observable=lambda p, t: p)
    with pytest.raises(ValueError, match="batch_size"):
        solver.run(states.gaussian(waist_ang=0.6), [0], 1, batch_size=0)
    with pytest.raises(ValueError, match="cuda_compatible"):
        _callback_check(None, lambda e, p, t: p)


def test_cpu_jump_history_preserves_batch_and_recording_contract():
    d, potential = _case()
    entry = states.gaussian(waist_ang=0.6)
    solver = AxialMCWF(d, 1e6, potential)
    first = solver.run(entry, [0, 1000, 2000], 24, seed=17, step_ang=10, batch_size=7)
    second = solver.run(entry, [2000], 24, seed=17, step_ang=10, batch_size=24)
    assert first.jumps == second.jumps
    for kind, counts in first.jump_history.items():
        assert counts[0] == 0 and np.all(np.diff(counts) >= 0)
        assert counts[-1] == first.jumps[kind] == second.jump_history[kind][-1]


def test_cuda_admission_reserves_recording_finalization_and_mixtures():
    from pyrite._backend import BackendResourceError, NumPyBackend
    from pyrite.montecarlo.channeling._axial_cuda import _admit
    from pyrite.montecarlo.channeling.axial_mcwf import density_observable

    d, potential = _case()
    solver = AxialMCWF(d, 1e6, potential)
    backend = NumPyBackend()
    backend.xp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (2**30, 2**30)))
    )
    # Three coexisting recording copies alone exceed this explicit16MiB budget;
    # reserving only one copy would incorrectly admit it.
    with pytest.raises(BackendResourceError, match="immutable data and scratch"):
        _admit(solver, backend, 1, 16 * 2**20, 10000, density_observable, 1)
    # Immutable sampled mixtures must be counted before device upload too.
    with pytest.raises(BackendResourceError, match="immutable data and scratch"):
        _admit(solver, backend, 1, 2**20, 1, density_observable, 1000)


def test_cuda_jump_equations_match_cpu():
    cp = _cuda()
    from pyrite.montecarlo.channeling._axial_cuda_dissipator import DeviceDissipator

    d, _ = _case()
    dev = DeviceDissipator(d)
    state = d.grid.sample(states.gaussian(waist_ang=0.6, center_ang=(0.3, 0.5)))
    density = np.abs(state) ** 2 / d.grid.norm(state)
    offsets = np.array([[0.03, -0.05], [0.1, 0.02]])
    device_density = cp.asarray(density)
    for source in range(len(d.tds_sources)):
        host_coeff = d.tds_weight_spectrum(source, density)
        coeff = dev.tds_weight_spectrum(source, device_density)
        np.testing.assert_allclose(cp.asnumpy(coeff), host_coeff, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(
            cp.asnumpy(dev.tds_phase(source, offsets)),
            d.tds_phase(source, offsets),
            rtol=1e-10,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            cp.asnumpy(dev.tds_weight(coeff, offsets)),
            d.tds_weight(host_coeff, offsets),
            rtol=1e-10,
            atol=1e-13,
        )
        edges = np.linspace(0, 0.8, 8)
        np.testing.assert_allclose(
            cp.asnumpy(dev.tds_weight_envelope(coeff, edges)),
            d.tds_weight_envelope(host_coeff, edges),
            rtol=1e-10,
            atol=1e-13,
        )
    np.testing.assert_allclose(
        cp.asnumpy(dev.electronic_expectations(device_density)),
        d.electronic_expectations(density),
        rtol=1e-10,
        atol=1e-12,
    )
    q = float(d.electronic_q_inv_ang[len(d.electronic_q_inv_ang) // 2])
    np.testing.assert_allclose(
        cp.asnumpy(dev.smoothed_density(q)), d.smoothed_density(q), rtol=1e-10, atol=1e-12
    )


def test_cuda_philox_published_vector_full_key_and_draw_partition():
    cp = _cuda()
    from pyrite.montecarlo.channeling._axial_cuda_rng import _DRAW, ElectronRNG

    # Random123 published Philox4x32-10 counter=key=0 known-answer vector.
    words = [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    expected = [((words[i] >> 5) * 2**26 + (words[i + 1] >> 6)) / 2**53 for i in (0, 2)]
    out = cp.empty(2, dtype=cp.float64)
    _DRAW((1,), (2,), (np.uint64(0), np.uint64(0), np.uint64(2), out))
    np.testing.assert_array_equal(cp.asnumpy(out), expected)
    low = ElectronRNG(17, 0)
    low.key = np.uint64(1)
    high = ElectronRNG(17, 0)
    high.key = np.uint64((1 << 32) + 1)
    assert not np.array_equal(cp.asnumpy(low.random(64)), cp.asnumpy(high.random(64)))
    whole, split = ElectronRNG(17, 42), ElectronRNG(17, 42)
    np.testing.assert_array_equal(
        cp.asnumpy(whole.random(64)),
        cp.asnumpy(cp.concatenate([split.random(7), split.random(57)])),
    )


def test_cuda_rejection_acceptance_handles_strided_real_weights():
    cp = _cuda()
    from pyrite.montecarlo.channeling._axial_cuda_jumps import _accepted

    weights = cp.asarray([0.2 + 7j, 0.8 + 9j, 0.5 + 11j]).real
    shells = cp.asarray([0, 0, 0], dtype=cp.int64)
    envelope = cp.asarray([1.0])
    uniforms = cp.asarray([0.5, 0.5, 0.6])
    assert _accepted(weights, envelope, shells, uniforms) == 1
    weights[2] = 1.1
    with pytest.raises(RuntimeError, match="exceeded"):
        _accepted(weights, envelope, shells, uniforms)


def test_cuda_displacement_proposal_matches_numpy_inverse_cdf():
    cp = _cuda()
    from pyrite.montecarlo.channeling._axial_cuda_jumps import _proposals
    from pyrite.montecarlo.channeling.axial_mcwf import _SHELL_EDGES

    proposal = np.arange(1, len(_SHELL_EDGES), dtype=float)
    draws = np.random.default_rng(620).random((4, 64))
    draws[:, 0] = 0
    draws[:, 1] = np.nextafter(1.0, 0.0)
    shells = np.searchsorted(np.cumsum(proposal), draws[0] * proposal.sum(), "right")
    decay = np.exp(-0.5 * _SHELL_EDGES**2)
    radius = 0.1 * np.sqrt(
        -2 * np.log(decay[shells + 1] + (decay[shells] - decay[shells + 1]) * draws[1])
    )
    offsets = radius[:, None] * np.stack(
        [np.cos(2 * np.pi * draws[2]), np.sin(2 * np.pi * draws[2])], axis=1
    )
    actual_shells, actual_offsets = _proposals(cp.asarray(proposal), cp.asarray(draws), 0.1)
    np.testing.assert_array_equal(cp.asnumpy(actual_shells), shells)
    np.testing.assert_allclose(cp.asnumpy(actual_offsets), offsets, rtol=1e-13, atol=1e-14)


def test_cuda_deterministic_propagation_and_device_callbacks():
    cp = _cuda()
    d, potential = _case()
    zeros = np.zeros(d.grid.shape)
    d = replace(d, tds_grid_rate=zeros, tds_sink_rate=zeros, electronic_rate=zeros)
    entry = states.gaussian(waist_ang=0.6)
    kwargs = dict(depths_ang=[0, 100], n_electrons=4, seed=7, step_ang=10)
    host = AxialMCWF(d, 1e6, potential).run(entry, **kwargs)
    device = AxialMCWF(d, 1e6, potential, backend="cuda").run(entry, **kwargs)
    np.testing.assert_allclose(device.observables, host.observables, rtol=2e-11, atol=1e-12)
    np.testing.assert_array_equal(device.surviving, 1)
    seen = []

    def observe(phi, tilt):
        assert isinstance(phi, cp.ndarray)
        seen.append(True)
        return cp.stack([cp.sum(cp.abs(phi) ** 2) * d.grid.cell_area_ang2])

    observe.cuda_compatible = True
    result = AxialMCWF(d, 1e6, potential, backend="cuda").run(entry, **kwargs, observable=observe)
    assert seen
    np.testing.assert_allclose(result.observables, 1, atol=2e-12)


@pytest.mark.slow
def test_cuda_stream_batch_recording_and_stochastic_parity():
    _cuda()
    d, potential = _case()
    entry = states.gaussian(waist_ang=0.6)
    solver = AxialMCWF(d, 1e6, potential, backend="cuda")
    kwargs = dict(n_electrons=128, seed=17, step_ang=10)
    first = solver.run(entry, [2000], batch_size=32, **kwargs)
    second = solver.run(entry, [0, 500, 2000], batch_size=7, **kwargs)
    assert first.jumps == second.jumps
    np.testing.assert_array_equal(np.sort(first.sink_depths_ang), np.sort(second.sink_depths_ang))
    np.testing.assert_allclose(
        first.observables[-1], second.observables[-1], rtol=1e-11, atol=1e-12
    )
    assert first.jumps["tds"] > 0 and first.jumps["electronic"] > 0
    host = AxialMCWF(d, 1e6, potential).run(entry, [2000], **kwargs)
    # Predeclared conservative independent-ensemble Hoeffding bound (four sigma
    # worst-case Bernoulli variance) for survival and four quadrant populations.
    tolerance = 4 * np.sqrt(0.5 / kwargs["n_electrons"])
    np.testing.assert_allclose(first.surviving, host.surviving, atol=tolerance, rtol=0)
    for x in (slice(0, 6), slice(6, 12)):
        for y in (slice(0, 6), slice(6, 12)):
            a = first.observables[-1, x, y].sum() * d.grid.cell_area_ang2
            b = host.observables[-1, x, y].sum() * d.grid.cell_area_ang2
            assert abs(a - b) < tolerance


@pytest.mark.slow
def test_cuda_replicated_jump_population_and_survival_statistics():
    _cuda()
    d, potential = _case()
    entry = states.gaussian(waist_ang=0.6)
    summaries = []
    for backend in ("cpu", "cuda"):
        solver = AxialMCWF(d, 1e6, potential, backend=backend)
        rows = []
        for seed in range(8):
            result = solver.run(entry, [2000], 64, seed=seed, step_ang=10)
            populations = [
                result.observables[-1, x, y].sum() * d.grid.cell_area_ang2
                for x in (slice(0, 6), slice(6, 12))
                for y in (slice(0, 6), slice(6, 12))
            ]
            rows.append(
                [
                    result.surviving[-1],
                    *populations,
                    *(result.jumps[kind] / 64 for kind in ("tds", "sink", "electronic")),
                ]
            )
        summaries.append(np.asarray(rows))
    host, device = summaries
    # Independent seed ensembles: predeclared four combined estimated SEMs,
    # with a one-electron floor for rare-event/zero sample-variance channels.
    tolerance = 4 * np.sqrt(host.var(axis=0, ddof=1) / 8 + device.var(axis=0, ddof=1) / 8) + 1 / 64
    assert np.all(np.abs(host.mean(axis=0) - device.mean(axis=0)) < tolerance)
