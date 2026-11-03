"""Policy pilot accounting checks; no transport or GPU allocations."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks.line_grid_candidates_286 import uniform_cells
from checks.line_policy_286 import policy_specs, summarize


class Native:
    def native_score(self, grid, density):
        return SimpleNamespace(events=np.ones(2))


class Half:
    def score(self, grid, density, **kwargs):
        return density / 2


@pytest.mark.parametrize("quadrature", ["node", "bin-mean"])
def test_integrals_respect_quadrature_and_shared_detector_cut(quadrature):
    grid = np.array([100.0, 200.0, 400.0])
    density = np.array([1.0, 2.0, 4.0])
    result, shape, native = summarize(
        grid,
        density,
        quadrature,
        uniform_cells(100, 250, 6),
        {"timepix3_counts": Native(), "eaglexo_counts": Half()},
        np.array([50.0, 150.0, 300.0, 500.0]),
    )
    if quadrature == "node":
        assert result["yield"] == pytest.approx(750)
        assert result["centroid_eV"] == pytest.approx(280)
        assert result["eaglexo_counts"] == pytest.approx(131.25)
    else:
        assert result["yield"] == pytest.approx(1200)
        assert result["centroid_eV"] == pytest.approx(331.25)
        assert result["eaglexo_counts"] == pytest.approx(125)
    assert shape.sum() == pytest.approx(result["yield"])
    assert native["timepix3_counts"] == 2


def test_policy_matrix_keeps_quadrature_control_and_local_rules():
    policies = dict(policy_specs([3, 12, 25, 50]))
    assert policies["default_node"] == {"bandwidth": "kinematic-ceiling"}
    assert policies["full_bin"]["quadrature"] == "bin-mean"
    for spacing in [3, 12, 25, 50]:
        local = policies[f"adaptive_{spacing}"]
        assert local["max_spacing_eV"] == spacing
        assert local["quadrature"] == "bin-mean"
        assert "halo_limit" not in local


def test_cpu_refused_before_transport(tmp_path, monkeypatch):
    import sys

    from checks.line_policy_286 import main

    monkeypatch.setattr("pyrite._backend.xp", np)
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["pilot", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not output.exists()


def test_gpu_exhaustion_is_recorded_and_later_policy_arms_continue(tmp_path, monkeypatch):
    import sys

    from checks import line_policy_286 as pilot

    class GPUOutOfMemory(Exception):
        pass

    pool = SimpleNamespace(total_bytes=lambda: 0, free_all_blocks=lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=GPUOutOfMemory)),
            get_default_memory_pool=lambda: pool,
        ),
    )
    args = SimpleNamespace(
        thickness_ang=1e7,
        electrons=20,
        detector_max_ev=500,
        repeats=1,
        reference_spacing_ev=3,
        spacings=[3, 12],
        output=tmp_path / "pilot.json",
    )

    def case_builder(args, material, energy, seed, policy):
        return {
            "line_grid_policy": {"bandwidth": {"start_eV": 100.0, "stop_eV": 500.0}},
            "policy": policy,
            "line_quadrature": policy.get("quadrature", "node"),
        }

    def resolver(case, ladder, directory):
        # One failed arm must not discard later successful controls/adaptive arms.
        if case["policy"].get("quadrature") == "node":
            raise GPUOutOfMemory("test allocation")
        return np.linspace(100.0, 500.0, 10), {}

    monkeypatch.setattr(pilot, "build_case", case_builder)
    monkeypatch.setattr(
        "pyrite.energy_grid.convergence_case.CaseLadder",
        lambda case: SimpleNamespace(fingerprint="same", transport_wall_s=0),
    )
    monkeypatch.setattr(pilot, "resolve", resolver)
    monkeypatch.setattr(
        pilot,
        "score",
        lambda ladder, case, grid, repeats: (
            np.ones(grid.size),
            {"first_s": 0.0, "warm_s": [0.0], "audit": None},
        ),
    )
    values = {"yield": 1.0, "centroid_eV": 200.0, "timepix3_counts": 1.0, "eaglexo_counts": 1.0}
    monkeypatch.setattr(
        pilot, "summarize", lambda *args: (values, np.ones(4), {"timepix3_counts": 1.0})
    )
    saved = []
    pilot.run_case(args, "hopg", 29999, 1, saved.append)
    row = saved[-1]
    assert row["complete"]
    assert row["arms"]["population_node"]["exception_type"] == "GPUOutOfMemory"
    assert "refused" not in row["arms"]["adaptive_12"]
    assert "common_adaptive_12_uniform_12" in row["arms"]


@pytest.mark.parametrize("fail", [False, True])
def test_profile_restores_reducer_hooks_and_preserves_score(monkeypatch, fail):
    import sys

    from checks import line_cost_286 as cost
    from checks import line_policy_286 as pilot
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import cxr_grid

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(pilot, "synchronize", lambda: None)
    monkeypatch.setattr("pyrite._backend._to_cpu", lambda x: x)
    monkeypatch.setattr(cxr_grid, "cxr_truncation_audit", lambda *a, **kw: {})
    monkeypatch.setattr(cxr_grid, "check_cxr_truncation", lambda *a: {"passed": True})
    originals = (
        cost._batched.run_bin_mean_reduction_kernel,
        cost._bin_tree.build_cxr_tree,
        cost._bin_tree.run_tree_reduction,
    )

    def calculate(*args, **kwargs):
        assert cost._batched.run_bin_mean_reduction_kernel != originals[0]
        if fail:
            raise RuntimeError("scoring failure")
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(runner, "_cxr_for_segments", calculate)
    ladder = SimpleNamespace(transport={"Ne_lines": 20, "n_hat": [0, 0, 1]}, _segments_device={})
    if fail:
        with pytest.raises(RuntimeError, match="scoring failure"):
            pilot.score(ladder, {"_profile_line_reducers": True}, np.arange(3), 1)
    else:
        density, timing = pilot.score(ladder, {"_profile_line_reducers": True}, np.arange(3), 1)
        np.testing.assert_array_equal(density, [1, 2, 3])
        assert timing["audit"] == {"passed": True}
        assert len(timing["reducer_profiles"]) == 2
        assert all(p["reduce_s"] == 0 for p in timing["reducer_profiles"])
    assert (
        cost._batched.run_bin_mean_reduction_kernel,
        cost._bin_tree.build_cxr_tree,
        cost._bin_tree.run_tree_reduction,
    ) == originals
