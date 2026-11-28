"""Experimental history fiber campaign identity, RNG and routing (#1527)."""

import numpy as np
import pytest

from pyrite.config import FiberAverage
from pyrite.montecarlo.runner import fiber_emission
from pyrite.montecarlo.spectrum.cxr._fiber_history import (
    FIBER_ROLL_STREAM, HistoryRollSeed, primary_roll_rows,
)
from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel
from pyrite.montecarlo.transport.kinematics import counter_uniforms


def test_history_rng_matches_transport_counter_and_is_sparse():
    ids = np.array([12, 0, 9, 12, 3], dtype=np.int64)
    seed = HistoryRollSeed(101)
    root = np.random.SeedSequence(101, spawn_key=FIBER_ROLL_STREAM)
    key = int(root.generate_state(1, dtype=np.uint64)[0])
    expected = (np.pi / 3) * counter_uniforms(key, 13, 1)[:, 0][ids]
    np.testing.assert_array_equal(seed.for_ids(ids), expected)
    np.testing.assert_array_equal(seed.for_ids(ids[:2]), expected[:2])
    angles, rows = primary_roll_rows(ids, seed, xp=np)
    np.testing.assert_array_equal(angles[rows], expected)
    assert np.all(seed.for_ids(np.array([0, 2**40])) < np.pi / 3)
    assert not np.array_equal(seed.for_ids(ids), HistoryRollSeed(102).for_ids(ids))


def test_history_identity_ignores_unused_quadrature_count():
    q = FiberCXRModel.from_numerics(FiberAverage(cutoff_invang=4))
    assert set(q.case_record()) == {
        "model", "crystal", "cutoff_invang", "azimuth_count", "temperature_K", "thermal_source"
    }
    h = FiberCXRModel.from_numerics(FiberAverage(cutoff_invang=4, roll_sampling="history"))
    other = FiberCXRModel.from_numerics(FiberAverage(
        cutoff_invang=4, roll_sampling="history", azimuth_count=128,
    ))
    assert h.case_record() != q.case_record()
    assert h.case_record() == other.case_record()
    assert FiberCXRModel.from_case_record(h.case_record()).case_record() == h.case_record()
    with pytest.raises(ValueError, match="RNG version"):
        FiberCXRModel.from_case_record(dict(h.case_record(), roll_rng="unknown"))
    with pytest.raises(ValueError, match="roll_sampling"):
        FiberAverage(cutoff_invang=4, roll_sampling="bad")


@pytest.mark.parametrize("change,reason", [
    ({"line_grid_policy": {}}, "explicit fixed line grid"),
    ({"_diagnostic_line_grid": {}}, "explicit fixed line grid"),
    ({"adaptive_precision": {}}, "fixed electron counts"),
    ({"recip_miscut_rad": (0.01, 0)}, "miscut"),
])
def test_history_case_refuses_unsupported_modes(change, reason):
    with pytest.raises(ValueError, match=reason):
        fiber_emission._validate_history_case({"seed": 101, **change})


def test_history_configuration_lowers_to_fixed_campaign():
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, Spot
    from pyrite.campaign.lowering import lower_simulation
    from pyrite.campaign.resolution import resolve_simulation
    from pyrite.config import CxrReflections, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab
    from pyrite.montecarlo.trajectories import case_digest

    scene = Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    physics = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")

    def lower(mode, grid):
        numerics = Numerics(
            sampling=FixedSampling(cxr_histories=2, brem_histories=2),
            cxr_reflections=CxrReflections(fiber=FiberAverage(cutoff_invang=4, roll_sampling=mode)),
            energy_bins=None if grid is None else EnergyBins(cxr=grid),
        )
        return lower_simulation(resolve_simulation(scene, physics=physics, numerics=numerics))

    grid = np.arange(100., 4000., 20.)
    q = lower("quadrature", grid)
    h = lower("history", grid)
    assert h["cxr_fiber"]["roll_sampling"] == "history"
    assert h["line_quadrature"] == "bin-mean"
    assert h.get("line_grid_policy") is None
    assert case_digest(q) != case_digest(h)
    with pytest.raises(ValueError, match="explicit fixed line grid"):
        lower("history", None)


def test_history_campaign_scores_each_reflection_once(monkeypatch):
    from pyrite.montecarlo import runner

    model = FiberCXRModel.from_numerics(FiberAverage(
        cutoff_invang=4, roll_sampling="history", reflection_batch_size=3,
    ))
    domain = model.prepare()
    calls = []

    def score(segs, grid, case, *args, **kwargs):
        hkls = case["hkl_list"]
        roll = case.get("_fiber_history_rolls")
        assert kwargs["Ne"] == 9
        assert kwargs["coherent"] is False
        assert len(hkls) <= 3
        assert all((h[:2] != (0, 0)) == isinstance(roll, HistoryRollSeed) for h in hkls)
        if roll is not None:
            assert roll.seed == 101
        calls.extend(hkls)
        return np.full(grid.shape, len(hkls))

    monkeypatch.setattr(runner, "_cxr_for_segments", score)
    monkeypatch.setattr(fiber_emission, "fiber_domain", lambda case: domain)
    case = {"seed": 101, "cxr_fiber": model.case_record(), "azimuth_rad": 0.2}
    grid = np.array([100., 200.])
    result = fiber_emission.fiber_lines({}, grid, case, None, None, None, Ne=9)
    assert sorted(calls) == sorted(domain.hkls)
    np.testing.assert_array_equal(result, np.full(grid.shape, len(domain.hkls)))
    with pytest.raises(ValueError, match="truncation audits"):
        fiber_emission.fiber_lines({}, grid, case, None, None, None, extras=(False, {}))
