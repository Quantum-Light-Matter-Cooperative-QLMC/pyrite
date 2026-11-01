"""CUDA photoelectric/EADL parity, capacity refills and signed energy budgets."""

from math import fsum

import numpy as np
import pytest

from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
from pyrite.montecarlo.transport._photon_cuda_relaxation import host_relaxation_table
from pyrite.montecarlo.transport.photoelectric import sample_photoelectric
from pyrite.montecarlo.transport.photon_histories import _isotropic
from pyrite.montecarlo.transport.photon_photoelectric_cuda import sample_photoelectric_cuda


@pytest.fixture
def cp():
    from pyrite._backend import BackendUnavailableError, select_cupy_backend

    try:
        return select_cupy_backend().xp
    except BackendUnavailableError as error:
        pytest.skip(str(error))


def _check_event(cp, actual, row, element, energy, ray, uniforms, cutoff):
    model, table = load_eadl_relaxation(element), host_relaxation_table(element)
    stream = iter(uniforms)
    expected = sample_photoelectric(element, energy, ray, model, cutoff, stream)
    directions = [_isotropic(stream) for _ in expected.relaxation.transitions]
    nt, nh = int(actual["n_transitions"][row]), int(actual["n_terminal"][row])
    assert nt == len(expected.relaxation.transitions)
    assert nh == len(expected.relaxation.terminal_vacancies)
    assert int(actual["subshell_designator"][row]) == expected.subshell_designator
    assert int(actual["draws_used"][row]) == expected.draws_used + 2 * nt
    np.testing.assert_allclose(
        float(actual["electron_energy_eV"][row]), expected.electron_energy_eV, rtol=1e-14
    )
    np.testing.assert_allclose(
        cp.asnumpy(actual["electron_direction"][row]),
        expected.electron_direction,
        rtol=2e-12,
        atol=2e-14,
    )
    ids = cp.asnumpy(actual["transition_ids"][row, :nt])
    expected_ids = []
    for t in expected.relaxation.transitions:
        parent = model.shell_designators.index(t.vacancy.shell_designator)
        offset = 0 if t.kind == "photon" else len(model.subshells[parent].radiative_probability)
        expected_ids.append(table["offsets"][parent] + offset + t.transition_index)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(
        cp.asnumpy(actual["emission_direction"][row, :nt]),
        np.asarray(directions).reshape(nt, 3),
        rtol=2e-12,
        atol=2e-14,
    )
    holes = cp.asnumpy(actual["terminal_shell_indices"][row, :nh])
    np.testing.assert_array_equal(
        table["designators"][holes],
        [v.shell_designator for v in expected.relaxation.terminal_vacancies],
    )
    terminal = fsum(table["binding"][i] for i in holes)
    defects = fsum(
        table["binding"][table["parent"][i]]
        - fsum(table["binding"][d] for d in (table["first"][i], table["second"][i]) if d >= 0)
        - table["energy"][i]
        for i in ids
    )
    assert terminal == expected.relaxation.energy_budget.terminal_binding_eV
    assert defects == expected.relaxation.energy_budget.transition_energy_defect_eV
    closure = (
        float(actual["electron_energy_eV"][row])
        + terminal
        + defects
        + fsum(table["energy"][i] for i in ids)
    )
    assert closure == pytest.approx(energy, rel=3e-15, abs=1e-10)


@pytest.mark.parametrize("element", ["C", "Si", "W"])
def test_same_draw_photoelectron_and_full_cascade_match_cpu(cp, element):
    rng = np.random.default_rng(341_904)
    energy = np.geomspace(1e3, 1e7, 24)
    rays = rng.normal(size=(len(energy), 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    uniforms = rng.random((len(energy), 4096))
    actual = sample_photoelectric_cuda(element, energy, rays, uniforms, cutoff_eV=100, capacity=128)
    np.testing.assert_array_equal(cp.asnumpy(actual["status"]), 0)
    for row, (e, ray, draws) in enumerate(zip(energy, rays, uniforms, strict=True)):
        _check_event(cp, actual, row, element, e, ray, draws, 100)


def test_photoelectric_prefix_and_capacity_retries_keep_the_same_history(cp):
    args = ("W", [2e5], [[0, 0, 1]])
    draws = np.zeros((1, 4096))
    short = sample_photoelectric_cuda(*args, draws[:, :2], cutoff_eV=100, require_complete=False)
    np.testing.assert_array_equal(cp.asnumpy(short["status"]), [1])
    with pytest.raises(ValueError, match="sequence exhausted"):
        sample_photoelectric_cuda(*args, draws[:, :2], cutoff_eV=100)
    small = sample_photoelectric_cuda(
        *args, draws, cutoff_eV=100, capacity=1, require_complete=False
    )
    np.testing.assert_array_equal(cp.asnumpy(small["status"]), [4])
    with pytest.raises(BufferError, match="capacity exhausted"):
        sample_photoelectric_cuda(*args, draws, cutoff_eV=100, capacity=1)
    full = sample_photoelectric_cuda(*args, draws, cutoff_eV=100, capacity=128)
    _check_event(cp, full, 0, "W", 2e5, np.array([0, 0, 1]), draws[0], 100)


def test_terminal_primary_consumes_no_relaxation_or_isotropic_draws(cp):
    draws = np.zeros((1, 16))
    actual = sample_photoelectric_cuda("W", [2e5], [[0, 0, 1]], draws, cutoff_eV=1e9, capacity=1)
    _check_event(cp, actual, 0, "W", 2e5, np.array([0, 0, 1]), draws[0], 1e9)
    assert int(actual["n_transitions"][0]) == 0
    assert int(actual["draws_used"][0]) == 4


def test_photoelectric_inputs_fail_before_device_probe(monkeypatch):
    from pyrite import _backend

    monkeypatch.setattr(
        _backend, "select_cupy_backend", lambda *args: pytest.fail("unexpected probe")
    )
    with pytest.raises(ValueError, match="cutoff"):
        sample_photoelectric_cuda("Si", [1e5], [[0, 0, 1]], [[0.1]], cutoff_eV=0)
    with pytest.raises(ValueError, match="capacity"):
        sample_photoelectric_cuda("Si", [1e5], [[0, 0, 1]], [[0.1]], cutoff_eV=100, capacity=0)
