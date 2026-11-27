"""Bandwidth extension of the in-process shard-sum route (#1395)."""

import numpy as np
import pytest

from pyrite._cxr_grid_policy import CxrGridToleranceError, CxrGridTruncationWarning
from pyrite.montecarlo.runner import shard_scoring, streamed_line_grid
from tests.montecarlo.runner.test_shard_scoring import _case


@pytest.mark.parametrize("explicit", [False, True])
def test_refused_shard_axis_replays_retained_rows(monkeypatch, explicit):
    case = dict(_case())
    payload = case["line_grid_policy"]
    case["line_grid_policy"] = {
        **payload,
        "bandwidth": {"policy": "resonance-population", "start_eV": 50.0, "stop_eV": 300.0},
        "sources": {"bandwidth": "per-call"} if explicit else {},
    }
    case["reflection_selection"] = None
    rows = [{"test_shard": 0}, {"test_shard": 1}]
    first = np.array([50.0, 100.0])
    initial = shard_scoring.ShardScoringPlan(
        first, {}, {}, None, shard_scoring._plan_digest(first, None)
    )
    refusal = CxrGridToleranceError("forced measured bandwidth refusal")
    refusal.truncation_record = {
        "start_eV": 50.0,
        "stop_eV": 100.0,
        "upper_fraction_bound": 0.01,
        "truncation_limit": 1e-4,
        "extension_stop_eV": 200.0,
    }
    seen = []
    fed = []
    plans = []

    def resolve_axis(work, block_electrons, **kwargs):
        assert block_electrons is None
        assert work["line_grid_policy"]["bandwidth"]["policy"] == "kinematic-ceiling"
        stop = work["line_grid_policy"]["bandwidth"]["stop_eV"]
        assert stop == 200.0
        kwargs["feed"](fed.append)
        return np.array([50.0, stop]), {"bandwidth_policy": "kinematic-ceiling"}, {}

    def score(work, shard, plan):
        seen.append((shard["test_shard"], plan.digest))
        return {"plan": plan.digest}

    def merge(work, plan, partials):
        assert all(partial["plan"] == plan.digest for partial in partials)
        plans.append(plan)
        if plan.digest == initial.digest:
            raise refusal
        return {"E_grid": plan.E_grid}

    monkeypatch.setattr(shard_scoring, "resolve_shard_plan", lambda *_: initial)
    monkeypatch.setattr(streamed_line_grid, "resolve_streamed_line_grid", resolve_axis)
    monkeypatch.setattr(shard_scoring, "score_history_shard", score)
    monkeypatch.setattr(shard_scoring, "merge_shard_spectra", merge)
    if explicit:
        with pytest.raises(CxrGridToleranceError) as caught:
            shard_scoring.run_case_shard_scored(case, 1, shards=rows)
        assert caught.value is refusal
        assert not fed
        assert seen == [(0, initial.digest), (1, initial.digest)]
        return
    with pytest.warns(CxrGridTruncationWarning, match="extended axis"):
        got = shard_scoring.run_case_shard_scored(case, 1, shards=rows)
    np.testing.assert_array_equal(got["E_grid"], [50.0, 200.0])
    assert fed == rows
    final = plans[-1]
    assert final.digest != initial.digest
    assert final.grid_record is not None
    assert final.grid_record["bandwidth_fallback"]["refusals"] == [refusal.truncation_record]
    assert seen == [(0, initial.digest), (1, initial.digest), (0, final.digest), (1, final.digest)]
