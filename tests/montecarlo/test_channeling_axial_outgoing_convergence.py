"""Paired convergence / return-deficit estimators of checks/axial_outgoing_convergence.py."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

CHECKS = Path(__file__).parents[2] / "checks"


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(CHECKS / "axial_outgoing_convergence.py"))


def _arm(kind, depth, bound, cps, weight=0.5, start=0, seed=0):
    kind, depth = np.asarray(kind, dtype=np.int8), np.asarray(depth, dtype=float)
    alive = (kind[:, None] == 0) | (depth[:, None] > cps[None, :])
    return {
        "kind": kind,
        "depth": depth,
        "alive": alive,
        "bound": np.where(alive, bound, 0.0),
        "checkpoints": cps,
        "weight": weight,
        "ids": (start, start + len(kind)),
        "info": {"seed": seed},
    }


def test_bound_weight_sums_states_below_the_barrier(study):
    c = np.array([0.6, 0.0, 0.8j])
    assert study["bound_weight"](np.array([-3.0, 1.0, 9.0]), c, 8.0) == pytest.approx(0.36)
    assert study["bound_weight"](np.array([-3.0, 1.0, 7.0]), c, 8.0) == pytest.approx(1.0)


def test_observables_tally_terminal_kinds_and_integrate_exposure(study):
    cps = np.array([0.0, 1.0, 2.0, 3.0])
    arm = _arm([0, 1, 2], [3.0, 1.5, 0.5], np.full((3, 4), 0.5), cps)
    obs = study["observables"](arm, depths_ang=(1.0, 3.0))
    np.testing.assert_allclose(obs["retained"], 0.5 * np.array([[1, 1], [1, 0], [0, 0]]))
    np.testing.assert_allclose(obs["outgoing"], 0.5 * np.array([[0, 0], [0, 1], [0, 0]]))
    np.testing.assert_allclose(obs["unresolved"], 0.5 * np.array([[0, 0], [0, 0], [1, 1]]))
    # N_B = 0.25 while alive; trapezoid over the 1 Å grid.
    np.testing.assert_allclose(obs["bound_exposure_ang"][:, 1], [0.75, 0.375, 0.125])


def test_paired_difference_uses_per_incident_differences(study):
    rng = np.random.default_rng(4)
    cps = np.array([0.0, 1.0, 2.0, 3.0])
    n = 400
    bound = rng.uniform(size=(n, 4))
    base = _arm(np.zeros(n), np.full(n, 3.0), bound, cps)
    shifted = _arm(np.zeros(n), np.full(n, 3.0), 0.9 * bound, cps)
    out = study["paired"](shifted, base, depths_ang=(1.0, 3.0))
    expected = 0.5 * (0.9 - 1.0) * bound[:, [1, 3]]
    np.testing.assert_allclose([m for m, _ in out["bound"]], expected.mean(0))
    np.testing.assert_allclose([s for _, s in out["bound"]], expected.std(0, ddof=1) / np.sqrt(n))
    # Exact proportional change: relative deficit 0.1 with zero paired error.
    for value, sem in out["bound_relative_deficit"] + out["bound_exposure_ang_relative_deficit"]:
        assert value == pytest.approx(0.1)
        assert sem == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="same seed"):
        study["paired"](_arm(np.zeros(n), np.full(n, 3.0), bound, cps, seed=1), base)


def test_chunks_refuse_survival_disagreement_and_gaps(study, tmp_path):
    cps = np.array([0.0, 1.0, 2.0])

    def save(name, start, kind, depth, alive=None):
        arm = _arm(kind, depth, np.ones((len(kind), 3)), cps, start=start)
        info = {
            "arm": "x",
            "seed": 0,
            "start": start,
            "count": len(kind),
            "entrance_weight": 0.5,
            "seconds": 1.0,
        }
        path = tmp_path / name
        np.savez(
            path,
            kind=arm["kind"],
            depth=arm["depth"],
            alive=arm["alive"] if alive is None else alive,
            bound=arm["bound"],
            lattice_jumps=np.zeros(len(kind)),
            kick_jumps=np.zeros(len(kind)),
            checkpoints=cps,
            info=json.dumps(info),
        )
        return path

    a, b = save("a.npz", 0, [0, 1], [2.0, 0.5]), save("b.npz", 2, [0], [2.0])
    merged = study["load_arm"]([b, a])
    assert merged["ids"] == (0, 3)
    np.testing.assert_array_equal(merged["kind"], [0, 1, 0])
    with pytest.raises(ValueError, match="contiguous"):
        study["load_arm"]([a, save("c.npz", 5, [0], [2.0])])
    with pytest.raises(ValueError, match="survival"):
        study["load_arm"]([save("d.npz", 0, [1], [0.5], alive=np.ones((1, 3), bool))])


def test_return_deficit_is_positive_when_absorption_drops_bound_weight(study):
    cps = np.array([0.0, 1.0, 2.0, 3.0])
    n = 50
    bound = np.random.default_rng(1).uniform(0.2, 1.0, size=(n, 4))
    reference = _arm(np.zeros(n), np.full(n, 3.0), bound, cps)
    absorbing = _arm(np.zeros(n), np.full(n, 3.0), 0.8 * bound, cps)
    out = study["return_deficit"](absorbing, reference, depths_ang=(3.0,))
    assert out["bound"][0][0] == pytest.approx(0.5 * 0.2 * bound[:, 3].mean())
    assert out["bound_exposure_ang_relative_deficit"][0][0] == pytest.approx(0.2)


def test_common_restricts_both_arms_to_shared_incidents(study):
    cps = np.array([0.0, 1.0])
    long = _arm(np.zeros(6), np.full(6, 1.0), np.arange(12.0).reshape(6, 2), cps)
    short = _arm(np.zeros(4), np.full(4, 1.0), np.ones((4, 2)), cps)
    a, b = study["common"](short, long)
    assert a["ids"] == b["ids"] == (0, 4)
    np.testing.assert_array_equal(b["bound"], long["bound"][:4])
    with pytest.raises(ValueError, match="no incident"):
        study["common"](short, _arm(np.zeros(2), np.full(2, 1.0), np.ones((2, 2)), cps, start=9))
