"""Unsupported axial mappings refuse before production work or result mutation."""

from types import SimpleNamespace

import pytest

from pyrite._precision import Precision
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import artifacts, channeling_emission
from pyrite.montecarlo.runner.adaptive import transport_case_adaptive
from pyrite.montecarlo.trajectories import TrajectoryArtifactError
from pyrite.results.store import store_result


@pytest.fixture(params=["record", "selectors"])
def axial_case(request):
    # A raw compatibility mapping bypasses Scene/profile and typed Case gates.
    if request.param == "record":
        return {"channeling": {"model": "axial", "axis_uvw": [0, 0, 1]}}
    return {"channeling_model": "axial", "channeling_axial": {"axis_uvw": [0, 0, 1]}}


@pytest.mark.parametrize(
    "consume",
    [
        lambda case: runner.run_case(case),
        lambda case: runner._transport_case(case, transport_core="cuda"),
        lambda case: runner._transport_case(case, trajectory_capture=object()),
        lambda case: runner._transport_case(case, block_electrons=4),
        lambda case: runner._spectrum_case(case, {}),
        lambda case: runner.run_cases([case], engine="cpu", max_workers=0, progress=False),
        lambda case: transport_case_adaptive(case, Precision(0.1, 40, 60, 20)),
        lambda case: channeling_emission.case_channeling_spectrum(case, {}),
        lambda case: channeling_emission.rescore_channeling_spectrum(case, [1000.0]),
    ],
    ids=["run", "cuda", "capture", "blocks", "spectrum", "batch", "adaptive", "source", "rescore"],
)
def test_raw_axial_mapping_refuses_before_setup(axial_case, consume):
    # Required geometry, grids, transport arrays and material inputs are absent:
    # touching any of them instead of refusing raises a different exception.
    with pytest.raises(ValueError, match="#623"):
        consume(axial_case)


def test_axial_result_is_refused_before_mutating_store(axial_case):
    results = {"existing": {}}
    with pytest.raises(ValueError, match="#623"):
        store_result(results, axial_case, {})
    assert results == {"existing": {}}


@pytest.mark.parametrize("streaming", [False, True])
def test_axial_replay_refuses_before_inputs_or_segments(axial_case, monkeypatch, streaming):
    artifact = SimpleNamespace(case=axial_case)
    monkeypatch.setattr(artifacts, "read_trajectory_artifact", lambda *_a, **_kw: artifact)
    with pytest.raises(TrajectoryArtifactError, match="#623"):
        if streaming:
            artifacts.stream_spectrum_from_artifact("unused.h5")
        else:
            artifacts.spectrum_from_artifact(artifact)
