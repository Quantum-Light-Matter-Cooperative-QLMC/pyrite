"""Guard the diagnostic CPU interpolation layout and the source-selection boundary."""

import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

CHECKS = Path(__file__).parents[2] / "checks/equal_precision"


def _helper():
    path = CHECKS / "moment_transport_cpu_device.py"
    spec = importlib.util.spec_from_file_location("test_cpu_moment_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.lookup_layer_moment_cpu


def _packed(plasma, grids, moments):
    counts = np.asarray([len(a) for a in grids], dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    return np.r_[
        plasma, starts, counts, *grids, *(np.log(g) for g in grids), np.concatenate(moments).ravel()
    ].astype(np.float64)


def _evaluate(lookup, layer, energy, packed):
    # Other input arrays do not matter: they are read only by the exact event
    # call, which is outside the diagnostic helper.
    return lookup(
        np.array([1, 1], dtype=np.int32),
        np.empty((2, 1), dtype=np.int32),
        np.empty((2, 1), dtype=np.int32),
        np.empty((2, 1)),
        np.empty((2, 1, 1)),
        np.empty((2, 1, 1)),
        np.empty((2, 1, 1)),
        np.empty((2, 1, 1)),
        layer,
        energy,
        1000.0,
        np.empty(1),
        packed,
    )


def test_cpu_helper_interpolates_float64_tables_without_touching_plasma():
    lookup = _helper()
    grids = [np.array([1e3, 2e3, 8e3]), np.array([1e3, 4e3])]
    values = [
        np.array([[1.0, 0.0, 9.0], [3.0, 4.0, 8.0], [7.0, 8.0, 7.0]]),
        np.array([[2.0, 0.0, 1.0], [8.0, 6.0, 2.0]]),
    ]
    packed = _packed(np.array([12.0, 34.0]), grids, values)
    assert packed[:2].tolist() == [12.0, 34.0]
    assert _evaluate(lookup, 0, 1000.0, packed) == (1.0, 0.0)
    assert _evaluate(lookup, 1, 4000.0, packed) == (8.0, 6.0)
    assert np.allclose(_evaluate(lookup, 0, 4000.0, packed), (5.0, 6.0), atol=1e-14)
    with pytest.raises(ValueError, match="outside prepared"):
        _evaluate(lookup, 0, 999.0, packed)
    with pytest.raises(ValueError, match="outside prepared"):
        _evaluate(lookup, 1, np.nan, packed)


def test_cpu_wrapper_rejects_invalid_core_and_arm(monkeypatch):
    path = CHECKS / "benchmark_moment_transport_cpu.py"
    parser = runpy.run_path(str(path))["parse_args"]
    for extra in (["--core", "cuda"], ["--core", "per-electron", "--arm", "detector"]):
        monkeypatch.setattr(sys, "argv", [str(path), "table", "audit.json", "scene", *extra])
        with pytest.raises(SystemExit) as err:
            parser()
        assert err.value.code == 2


def test_cpu_kernel_call_inventory_is_exactly_three_plus_one():
    source = (Path(__file__).parents[2] / "src/pyrite/montecarlo/transport/cores.py").read_text()
    assert source.count("radiative_layer_moments_scalar(") == 1
    # Each of the three sites: inline table lookup + out-of-line exact quadrature (#841).
    assert source.count("_moments.table_moments(") == 3
    assert source.count("_moments.exact_moments(") == 3
    transformed = source
    assert transformed.count("radiative_layer_moments_scalar(") == 1
    # The retained exact call is the physical row-end sampler prerequisite.
    assert transformed.index("radiative_layer_moments_scalar(") > transformed.index(
        "if event_j == EVENT_HARD_RADIATIVE:"
    )
