"""Prescribed-state scorer controls; no histories, tables or CUDA required."""

import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

HARNESS = Path(__file__).resolve().parents[2] / "checks/geant4_performance/pyrite_bench.py"
NS = runpy.run_path(str(HARNESS))


def _score(monkeypatch, angular=4, energy=None, mode="fixed"):
    from pyrite.montecarlo.spectrum import brem_events

    calls = []

    def scorer(_tracks, nodes, **kwargs):
        calls.append((len(nodes), tuple(kwargs["n_hat"])))
        # Known cubic spectrum; first incident primary is an explicit zero.
        integral = np.dot((nodes / 1000) ** 3, kwargs["electron_band_weights"] / 1000)
        return np.array([0.0, integral])

    monkeypatch.setattr(brem_events, "mc_coupled_brem_spectrum", scorer)
    ctx = dict(
        composition=[("Si", 1.0)],
        energy_keV=60,
        thickness_ang=1,
        edges=[30.0],
        cutoff_eV=1000,
        brem={},
        chunk=100,
        mode=mode,
        order=angular,
        rtol=1e-3,
        orders=[],
    )
    if energy is not None:
        ctx["energy_order"] = energy
    scores = NS["continuous_cone_scores"]({}, 2, dict(center_deg=90, half_angle_deg=10), ctx)
    return scores, calls


@pytest.mark.parametrize("mode", ["fixed", "point"])
def test_independent_energy_order_changes_only_energy_nodes(monkeypatch, mode):
    coarse, a = _score(monkeypatch, angular=4, energy=4, mode=mode)
    fine, b = _score(monkeypatch, angular=4, energy=16, mode=mode)
    assert len(a) == len(b)
    assert [direction for _, direction in a] == [direction for _, direction in b]
    assert all(n_b == 4 * n_a for (n_a, _), (n_b, _) in zip(a, b, strict=True))
    assert min(n for n, _ in a) == 8
    solid_angle = 2 * math.pi * (1 - math.cos(math.radians(10)))
    expected = solid_angle * (60**4 - 10**4) / 4
    assert fine["cone_source_photons_ge10kev"][0] == 0
    assert fine["cone_source_photons_ge10kev"][1] == pytest.approx(expected)
    for key in coarse:
        np.testing.assert_allclose(coarse[key], fine[key], rtol=2e-14, atol=0)


def test_angular_refinement_holds_explicit_energy_rule_fixed(monkeypatch):
    _, a = _score(monkeypatch, angular=4, energy=8)
    _, b = _score(monkeypatch, angular=8, energy=8)
    assert {n for n, _ in a} == {n for n, _ in b}
    assert min(n for n, _ in a) == 16
    assert len(b) == 4 * len(a)


@pytest.mark.parametrize("mode", ["fixed", "point"])
def test_omitted_energy_control_preserves_coupled_scores_bit_for_bit(monkeypatch, mode):
    original, a = _score(monkeypatch, angular=8, mode=mode)
    explicit, b = _score(monkeypatch, angular=8, energy=8, mode=mode)
    assert a == b
    for key in original:
        np.testing.assert_array_equal(original[key], explicit[key])


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--estimator", "continuous"],
        ["--estimator", "continuous", "--cone-mode", "fixed", "--tables-only"],
    ],
)
def test_unused_energy_control_fails_before_tables_or_cuda(tmp_path, monkeypatch, extra):
    output = tmp_path / "unused.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(HARNESS),
            "si-300keV-100um",
            "--histories",
            "2",
            "--seed",
            "301",
            "--output",
            str(output),
            "--cone-energy-order",
            "8",
            *extra,
        ],
    )
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(SystemExit) as error:
        NS["main"]()
    assert error.value.code == 2
    assert not output.exists()
