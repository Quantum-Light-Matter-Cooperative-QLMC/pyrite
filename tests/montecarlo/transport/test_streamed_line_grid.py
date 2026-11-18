"""Live streamed scoring with a population-resolved line grid (#1001).

The streamed path resolves the grid in a first transport pass over the same
electron blocks, then scores in a second. Its grid must equal the joined
path's bit for bit, its spectra within summation order, and its provenance
records must match.
"""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._cxr_grid_policy import resolve_cxr_grid_policy
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.console.config import output_dir
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.streamed_line_grid import WINDOWED_POLICY_REFUSAL, _drain
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def _case(policy=None):
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    case = {**build_cases(sweep)[0], "Ne": 24, "Ne_brem": 30}
    bandwidth = case["line_grid_policy"]["bandwidth"]
    case["line_grid_policy"] = (
        policy
        or resolve_cxr_grid_policy(
            start_eV=bandwidth["start_eV"],
            stop_eV=bandwidth["stop_eV"],
            per_call={"resolution": "sinc-nyquist"},
        ).payload()
    )
    assert case["line_grid_policy"]["bandwidth"]["policy"]
    return case


def _measured_policy(case):
    bandwidth = case["line_grid_policy"]["bandwidth"]
    return resolve_cxr_grid_policy(
        start_eV=bandwidth["start_eV"],
        stop_eV=bandwidth["stop_eV"],
        per_call={
            "bandwidth": "resonance-population",
            "resolution": "resonance-local",
            "quadrature": "bin-mean",
        },
    ).payload()


def _clear_grid_cache():
    shutil.rmtree(output_dir("cache"), ignore_errors=True)


def _joined(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


def _assert_matches(streamed, reference):
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    for key in SPECTRA:
        # Summation order; the windowed far tail (#1191) puts peak-scale rounding in low bins.
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key
    for key in ("line_grid_resolved", "line_grid_diagnostic"):
        expected = {k: v for k, v in reference[key].items() if k != "cache"}
        actual = {k: v for k, v in streamed[key].items() if k != "cache"}
        _assert_record_close(actual, expected, key)


def _assert_record_close(actual, expected, path):
    """Equal records, with floats at summation-order tolerance.

    The truncation audit sums line masses block by block, like the spectra.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_record_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=0, nan_ok=True), path
    else:
        np.testing.assert_equal(actual, expected, err_msg=path)


@pytest.fixture(scope="module", params=["sinc-nyquist", "resonance-local"])
def policy_case(request):
    case = _case()
    if request.param == "resonance-local":
        case = _case(_measured_policy(case))
    return case


@pytest.mark.parametrize("block", [5, 24, 1000])
def test_streamed_policy_grid_matches_the_joined_path(policy_case, block):
    _clear_grid_cache()
    reference = _joined(policy_case)
    _clear_grid_cache()
    streamed = run_case_streamed(policy_case, block, transport_core="per-electron")
    assert streamed["_streamed_line_grid_pass"]["transported"]
    assert streamed["_streamed_line_grid_pass"]["evidence_bytes"] > 0
    _assert_matches(streamed, reference)


def test_a_cached_grid_skips_the_grid_pass(monkeypatch):
    case = _case()
    _clear_grid_cache()
    reference = _joined(case)  # stores the grid
    calls = []
    transport = runner._transport_case

    def counted(*args, **kwargs):
        calls.append(kwargs.get("block_consumer"))
        return transport(*args, **kwargs)

    monkeypatch.setattr(runner, "_transport_case", counted)
    streamed = run_case_streamed(case, 10, transport_core="per-electron")
    assert len(calls) == 1
    assert streamed["_streamed_line_grid_pass"] == {
        "transported": False,
        "evidence_bytes": 0,
        "spilled": False,
        "spill_peak_bytes": 0,
    }
    assert streamed["line_grid_resolved"]["cache"] == "hit"
    _assert_matches(streamed, reference)


def test_windowed_policies_are_refused_before_transport(monkeypatch):
    case = _case()
    case["line_grid_policy"] = {**case["line_grid_policy"], "windows": {"providers": []}}
    assert streaming_refusal(case, live=True) == WINDOWED_POLICY_REFUSAL
    assert streaming_refusal(case) is None  # artifact replay reads a final grid
    monkeypatch.setattr(runner, "_transport_case", None)
    with pytest.raises(ValueError, match="windowed line_grid_policy"):
        run_case_streamed(case, 10)


def test_drain_joins_blocks_in_order_and_releases_them():
    blocks = [np.arange(3.0), np.empty(0), np.arange(3.0, 7.0)]
    expected = np.concatenate(blocks)
    joined = _drain(blocks)
    np.testing.assert_array_equal(joined, expected)
    assert joined.dtype == expected.dtype and blocks == []
    assert _drain([]).size == 0


@pytest.mark.parametrize("block", [5, 24, 1000])
@pytest.mark.parametrize("quadrature", ["node", "bin-mean"])
def test_oversampled_policy_is_refused_before_transport(monkeypatch, block, quadrature):
    case = _case()
    band = case["line_grid_policy"]["bandwidth"]
    case["line_grid_policy"] = resolve_cxr_grid_policy(
        start_eV=band["start_eV"],
        stop_eV=band["stop_eV"],
        per_call={"resolution": "sinc-oversampled", "quadrature": quadrature, "max_points": 1000},
    ).payload()
    assert "whole-population line evidence" in streaming_refusal(case, live=True)
    assert streaming_refusal(case) is None
    monkeypatch.setattr(runner, "_transport_case", None)
    with pytest.raises(ValueError, match="sinc-oversampled.*use joined scoring"):
        run_case_streamed(case, block, transport_core="per-electron")


def test_over_budget_kinematic_evidence_gap_against_joined_resolver():
    from pyrite.montecarlo.runner.cxr_grid import _kinematic_cxr_grid
    from pyrite.montecarlo.runner.streamed_line_grid import BlockGridEvidence
    from pyrite.montecarlo.spectrum.cxr_seeds import ResonancePopulation

    payload = resolve_cxr_grid_policy(
        start_eV=50.0,
        stop_eV=1000.0,
        per_call={
            "resolution": "sinc-oversampled",
            "quadrature": "node",
            "max_points": 1200,
            "bandwidth": "kinematic-ceiling",
        },
    ).payload()
    case, direction = {"name": "line-grid probe", "E0_keV": 30.0}, np.array([0.0, 0.0, 1.0])
    population = ResonancePopulation("line", np.array([500.0]), np.ones(1), np.array([0.01]))
    joined = SimpleNamespace(populations=lambda *_: [population])
    grid, record = _kinematic_cxr_grid(payload, case, [direction], 1.0, joined)
    assert record["local_axis"]["uniform_points"] > 1200 >= grid.size
    streamed = BlockGridEvidence(case, payload, direction, 1)
    # This was the late failure after the streamed grid transport pass.
    with pytest.raises(ValueError, match="holds no lines for this band"):
        _kinematic_cxr_grid(payload, case, [direction], 1.0, streamed)
