"""Characteristic segment chunk: own control, grid-independent device sizing (#1455)."""

import numpy as np
import pytest

import pyrite.montecarlo.runner as runner
from pyrite.montecarlo.spectrum import _characteristic_fused, characteristic

FINE_GRID = 600_000  # sinc-oversampled line grid of a wide window
DEVICE_BUDGET = 8 << 30


@pytest.fixture
def device_policy(monkeypatch):
    """Size as on a GPU without one: device budget on, no pinned chunks."""
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "device_budget_bytes", DEVICE_BUDGET)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "spec_budget_mb", 1920)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "brem_chunk", 0)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "characteristic_chunk", 0)


def _segments(seed=1455, n_electrons=4_000, per_electron=8):
    rng = np.random.default_rng(seed)
    n = n_electrons * per_electron
    v = rng.normal(size=(n, 3))
    return {
        "r_mid": np.column_stack((np.zeros(n), np.zeros(n), rng.uniform(1.0, 99.0, n))),
        "v_hat": v / np.linalg.norm(v, axis=1, keepdims=True),
        "L_ang": rng.uniform(0.5, 50.0, n),
        "E_keV": rng.uniform(12.0, 190.0, n),
        "elec_id": np.repeat(np.arange(n_electrons), per_electron),
        "layer": np.zeros(n, dtype=int),
        "Ne": n_electrons,
        "thickness_ang": 1000.0,
    }


def test_device_chunk_ignores_the_line_grid_and_brem(device_policy):
    case = {"composition": [("W", 0.063)], "brem_chunk": 64}
    coarse = runner._characteristic_chunk(case, 1_000)
    fine = runner._characteristic_chunk(case, FINE_GRID)
    legacy = runner._admit_chunk(runner._adaptive_chunk(FINE_GRID), FINE_GRID)

    assert coarse == fine
    assert fine > 1_000 * legacy
    assert fine <= characteristic._AUTO_CHUNK_ROWS
    # The chunk's own intermediates stay within the spectrum budget.
    n_shell = characteristic._max_ionization_shells(["W"])
    used = 8 * (
        fine * characteristic._chunk_row_elements(n_shell) + characteristic._chunk_fixed_elements()
    )
    assert used <= 1_920_000_000


def test_explicit_then_pinned_then_automatic(device_policy, monkeypatch):
    case = {"composition": [("W", 0.063)]}
    automatic = runner._characteristic_chunk(case, FINE_GRID)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "characteristic_chunk", 5_000)
    assert runner._characteristic_chunk(case, FINE_GRID) == 5_000
    assert runner._characteristic_chunk({**case, "characteristic_chunk": 700}, FINE_GRID) == 700
    monkeypatch.setattr(runner._RESOURCE_POLICY, "characteristic_chunk", 0)
    assert runner._characteristic_chunk(case, FINE_GRID) == automatic


def test_device_admission_caps_an_oversized_request(device_policy, monkeypatch):
    monkeypatch.setattr(runner._RESOURCE_POLICY, "device_budget_bytes", 1 << 30)
    case = {"composition": [("W", 0.063)], "characteristic_chunk": 10**9}
    admitted = runner._characteristic_chunk(case, FINE_GRID)
    n_shell = characteristic._max_ionization_shells(["W"])
    per_row = 8 * characteristic._chunk_row_elements(n_shell)
    fixed = 8 * characteristic._chunk_fixed_elements()
    assert admitted == ((1 << 30) - fixed) // per_row


def test_lighter_elements_admit_wider_chunks(device_policy):
    light = runner._characteristic_chunk({"composition": [("Si", 0.05)]}, FINE_GRID)
    heavy = runner._characteristic_chunk({"composition": [("W", 0.063)]}, FINE_GRID)
    stack = runner._characteristic_chunk(
        {
            "composition": [("Si", 0.05)],
            "abs_layers": [(0.0, 100.0, [("W", 0.063)]), (100.0, 1000.0, [("Si", 0.05)])],
        },
        FINE_GRID,
    )
    assert light >= heavy
    assert stack == heavy


def test_cpu_keeps_its_grid_sized_default():
    assert not runner._RESOURCE_POLICY.gpu or pytest.skip("CPU policy only")
    expected = runner._admit_chunk(runner._adaptive_chunk(2_000), 2_000)
    assert runner._characteristic_chunk({"brem_chunk": 64}, 2_000) == expected


def test_new_sizing_cuts_launch_sets_and_agrees_to_fp64_rounding(device_policy, monkeypatch):
    """One launch set per (chunk, line group); the sum only reorders (#1313)."""
    case = {"composition": [("W", 0.063)]}
    old_chunk = runner._admit_chunk(runner._adaptive_chunk(FINE_GRID), FINE_GRID)
    new_chunk = runner._characteristic_chunk(case, FINE_GRID)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    launches = []
    segment_sum = _characteristic_fused.segment_sum

    def counted(weighted):
        launches.append(weighted.shape)
        return segment_sum(weighted)

    monkeypatch.setattr(_characteristic_fused, "segment_sum", counted)
    segments = _segments()
    energy = np.linspace(1.0e3, 1.2e4, 200)

    def score(chunk):
        launches.clear()
        spectrum = characteristic.mc_characteristic_spectrum(
            segments, energy, composition=case["composition"], chunk=chunk
        )
        return spectrum, len(launches)

    old, old_launches = score(old_chunk)
    new, new_launches = score(new_chunk)
    odd, _ = score(997)

    assert old_chunk < segments["L_ang"].size <= new_chunk
    assert new_launches == 1
    assert old_launches >= segments["L_ang"].size // old_chunk
    assert old.max() > 0.0
    for chunked in (old, odd):
        np.testing.assert_allclose(chunked, new, rtol=1.0e-12, atol=1.0e-12 * new.max())


def test_env_pin_is_independent_of_the_brem_pin(monkeypatch):
    from pyrite.montecarlo.runner.chunking import _resolve_runner_resource_policy

    monkeypatch.setenv("PYRITE_MC_BREM_CHUNK", "2048")
    monkeypatch.setenv("PYRITE_MC_CHARACTERISTIC_CHUNK", "65536")
    policy = _resolve_runner_resource_policy()
    assert (policy.brem_chunk, policy.characteristic_chunk) == (2048, 65536)
    monkeypatch.delenv("PYRITE_MC_CHARACTERISTIC_CHUNK")
    assert _resolve_runner_resource_policy().characteristic_chunk == 0
