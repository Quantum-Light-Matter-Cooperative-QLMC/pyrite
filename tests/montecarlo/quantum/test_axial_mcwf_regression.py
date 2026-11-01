"""Independent right-insertion, RNG and actual axial adapter oracles."""

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.montecarlo.channeling.axial_mcwf_regression import (
    AuxiliaryAddress,
    axial_mcwf_regression,
    four_map_regression,
)

ADDRESS = AuxiliaryAddress(620, 17, 3, 1)


def test_complex_right_insertion_matches_closed_unitary_dense_oracle():
    psi = np.array([1.0, 1j]) / np.sqrt(2)
    b = np.array([[0.2j, 0.7], [0.0, -0.1]])
    current = np.array([[0.0, 0.3j], [0.8, 0.2]])
    h = np.array([[1.0, 0.1j], [-0.1j, 2.0]])
    depths = [0.0, 0.4, 1.3]

    def evolve(state, seed, count):
        return np.array(
            [
                np.vdot(expm(-1j * h * z) @ state, current @ expm(-1j * h * z) @ state)
                for z in depths
            ]
        )[:, None]

    actual = four_map_regression(
        psi,
        lambda state: b @ state,
        evolve,
        record_shape=(3, 1),
        address=ADDRESS,
        n_replicas=3,
        n_auxiliary=1,
    )
    insertion = np.outer(psi, psi.conj()) @ b.conj().T
    expected = [
        np.trace(current @ expm(-1j * h * z) @ insertion @ expm(1j * h * z)) for z in depths
    ]
    np.testing.assert_allclose(actual.values[:, 0], expected, atol=4e-16)
    np.testing.assert_allclose(actual.sem_real, 0, atol=1e-16)
    np.testing.assert_allclose(actual.sem_imag, 0, atol=1e-16)
    np.testing.assert_allclose(actual.insertion_scale, np.linalg.norm(b @ psi))


def test_amplitude_damping_incident_normalization_and_subnormalized_parent():
    psi = np.array([0.0, np.sqrt(0.3)])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])
    survival = np.array([1.0, 0.7, 0.1])

    def evolve(state, seed, count):
        rho = np.outer(state, state.conj())
        values = []
        for probability in survival:
            k0 = np.diag([1.0, np.sqrt(probability)])
            k1 = np.sqrt(1 - probability) * lower
            values.append(np.trace(lower @ (k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T)))
        return np.array(values)[:, None]

    actual = four_map_regression(
        psi,
        lambda state: lower @ state,
        evolve,
        record_shape=(3, 1),
        address=ADDRESS,
        n_replicas=4,
        n_auxiliary=2,
    )
    np.testing.assert_allclose(actual.values[:, 0], 0.3 * np.sqrt(survival), atol=2e-16)
    attenuated = four_map_regression(
        psi,
        lambda state: lower @ state,
        lambda *args: 0.2 * evolve(*args),
        record_shape=(3, 1),
        address=ADDRESS,
        n_replicas=4,
        n_auxiliary=2,
    )
    np.testing.assert_allclose(attenuated.values, 0.2 * actual.values)


def test_replica_complex_errors_follow_combined_four_branch_samples():
    psi = np.array([1.0, 0.0])
    b = np.array([[0.3, 0.4j], [0.7, -0.2]])
    observed = []

    def noisy_map(state, seed, count):
        rng = np.random.default_rng(seed)
        value = rng.normal(size=2) + 1j * rng.normal(size=2)
        observed.append(value)
        return value

    actual = four_map_regression(
        psi,
        lambda state: b @ state,
        noisy_map,
        record_shape=(2,),
        address=ADDRESS,
        n_replicas=5,
        n_auxiliary=3,
    )
    scale = np.linalg.norm(b @ psi)
    expected = []
    for replica in range(5):
        total = 0j
        for branch, phase in enumerate([1.0, -1.0, 1j, -1j]):
            chi = psi + phase * (b @ psi) / scale
            total += phase * scale * np.vdot(chi, chi).real * observed[4 * replica + branch] / 4
        expected.append(total)
    expected = np.array(expected)
    np.testing.assert_allclose(actual.replicas, expected, atol=2e-16)
    np.testing.assert_allclose(actual.sem_real, expected.real.std(axis=0, ddof=1) / np.sqrt(5))
    np.testing.assert_allclose(actual.sem_imag, expected.imag.std(axis=0, ddof=1) / np.sqrt(5))
    centered = expected - expected.mean(axis=0)
    np.testing.assert_allclose(
        actual.covariance_real_imag, (centered.real * centered.imag).sum(axis=0) / 20
    )
    assert actual.auxiliary_trajectories == 5 * 4 * 3
    assert np.any(actual.values.imag != 0)


def test_zero_parent_insertion_and_zero_auxiliary_branch_skip_work():
    calls = []

    def evolve(state, seed, count):
        calls.append(seed)
        return np.array([1.0])

    for parent, apply in [
        (np.zeros(2), lambda state: state),
        (np.array([1.0, 0.0]), lambda state: state * 0),
    ]:
        actual = four_map_regression(parent, apply, evolve, record_shape=(1,), address=ADDRESS)
        np.testing.assert_array_equal(actual.values, [0.0])
    assert calls == []
    actual = four_map_regression(
        np.array([1.0, 0.0]),
        lambda state: state,
        evolve,
        record_shape=(1,),
        address=ADDRESS,
        n_replicas=2,
        n_auxiliary=1,
    )
    np.testing.assert_allclose(actual.values, [1.0])
    assert calls == [ADDRESS.seed(r, b) for r in range(2) for b in (0, 2, 3)]


def test_namespace_matches_actual_philox_keys_and_primary_separation():
    from pyrite.montecarlo.channeling.mcwf import _rng, mcwf_stream_key

    root = ADDRESS.root_seed
    keys = []
    for parent, depth, component, replica, branch in itertools.product(
        [0, 17], [0, 3], [0, 1], [0, 2], range(4)
    ):
        address = AuxiliaryAddress(root, parent, depth, component)
        base = (
            0x9 << 60 | parent << 40 | depth << 30 | component << 26 | replica << 16 | branch << 14
        )
        seed = address.seed(replica, branch)
        for local in [0, 1, 2**14 - 1]:
            actual = int(mcwf_stream_key(seed, local))
            assert actual == int(mcwf_stream_key(root, base + local))
            np.testing.assert_array_equal(
                _rng(seed, local).random(3), _rng(root, base + local).random(3)
            )
            keys.append(actual)
    assert len(set(keys)) == len(keys)
    assert set(keys).isdisjoint(int(mcwf_stream_key(root, e)) for e in range(128))


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_seed", -1),
        ("parent_id", 2**20),
        ("earlier_depth_index", 1024),
        ("insertion_component", 16),
        ("parent_id", True),
    ],
)
def test_namespace_range_validation(field, value):
    fields = dict(root_seed=620, parent_id=17, earlier_depth_index=3, insertion_component=1)
    fields[field] = value
    with pytest.raises(ValueError):
        AuxiliaryAddress(**fields)


def test_budgets_refuse_before_insertion_or_evolution():
    def forbidden(*args):
        raise AssertionError("callback before admission")

    for kwargs, message in [
        ({"max_auxiliary_trajectories": 1}, "trajectory budget"),
        ({"max_working_bytes": 1}, "memory budget"),
        ({"n_auxiliary": 2**14 + 1}, "n_auxiliary"),
    ]:
        with pytest.raises(ValueError, match=message):
            four_map_regression(
                np.ones(2), forbidden, forbidden, record_shape=(2,), address=ADDRESS, **kwargs
            )


@pytest.fixture
def zero_solver():
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF

    profile = axial_profile("silicon", (0, 0, 1), 293.15)
    dissipator = axial_dissipator(profile, 5e6, (1, 1), (4, 4), mechanisms=())
    return AxialMCWF(dissipator, 5e6, np.zeros((4, 4)))


def test_actual_axial_adapter_free_grid_unitary_array_reconstruction_and_carrier(zero_solver):
    from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
    from pyrite.montecarlo.channeling.states import from_array

    solver = zero_solver
    rng = np.random.default_rng(620)
    phi = rng.normal(size=solver.grid.shape) + 1j * rng.normal(size=solver.grid.shape)
    phi /= np.sqrt(solver.grid.norm(phi))
    spacing = tuple(e / n for e, n in zip(solver.grid.extent_ang, solver.grid.shape, strict=True))
    np.testing.assert_allclose(
        solver.grid.sample(from_array(phi, spacing_ang=spacing, origin_ang=(0, 0))), phi, atol=1e-14
    )
    insertion = np.exp(2j * np.pi * solver.grid.x[:, None] / solver.grid.extent_ang[0])
    current_multiplier = (0.1 + 0.2j) * insertion.conj()
    tilt = (0.17, -0.24)
    seen = []

    def current(state, kt):
        seen.append(kt)
        return np.array(
            [np.sum(state.conj() * current_multiplier * state) * solver.grid.cell_area_ang2]
        )

    actual = axial_mcwf_regression(
        solver,
        phi,
        lambda state, kt: insertion * state,
        current,
        [0.0, 5.0, 10.0],
        address=ADDRESS,
        tilt_inv_ang=tilt,
        n_replicas=2,
        n_auxiliary=1,
        batch_size=1,
    )
    energy = (
        HBAR2_2M_EV_ANG2
        / solver.gamma
        * ((solver.grid.kx[:, None] + tilt[0]) ** 2 + (solver.grid.ky[None, :] + tilt[1]) ** 2)
    )
    expected = []
    for z in [0.0, 5.0, 10.0]:
        phase = np.exp(-1j * energy * z / solver.hbar_v)
        p = np.fft.ifft2(np.fft.fft2(phi) * phase)
        b = np.fft.ifft2(np.fft.fft2(insertion * phi) * phase)
        expected.append(np.sum(b.conj() * current_multiplier * p) * solver.grid.cell_area_ang2)
    np.testing.assert_allclose(actual.values[:, 0], expected, atol=1e-14)
    np.testing.assert_allclose(seen, np.broadcast_to(tilt, (len(seen), 2)), atol=1e-16)


def test_actual_adapter_refuses_cuda_and_nonstep_depths(zero_solver):
    solver = zero_solver
    solver.backend = "cuda"
    with pytest.raises(ValueError, match="CPU only"):
        axial_mcwf_regression(
            solver,
            np.ones(solver.grid.shape),
            lambda *a: None,
            lambda *a: None,
            [0.0, 5.0],
            address=ADDRESS,
        )
    solver.backend = "cpu"
    with pytest.raises(ValueError, match="multiples"):
        axial_mcwf_regression(
            solver,
            np.ones(solver.grid.shape),
            lambda *a: None,
            lambda *a: None,
            [0.0, 3.0],
            address=ADDRESS,
        )
