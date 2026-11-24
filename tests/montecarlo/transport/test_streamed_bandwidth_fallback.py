"""A refused default line axis extends on the live streamed route too (#1243).

Remote job cleo_sub-31 (HOPG 300 keV, pol=45): the streamed adaptive route
audited its measured bandwidth, found 4.98e-3 of the line yield above the
edge and aborted, where the retained route rescores on an extended axis. The
streamed route has no transport to rescore, so it reruns its grid and scoring
passes on the extension; counter-addressed streams make that the same
histories, so it must match the retained rescore up to summation order.
"""

import shutil
from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import _to_cpu
from pyrite._cxr_grid_policy import (
    RESONANCE_BANDWIDTH_POLICY,
    CxrGridToleranceError,
    CxrGridTruncationWarning,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import case_driver, cxr_grid
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from pyrite.paths import cache_dir

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def _hbn_case(cxr_grid_policy=None):
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=1000.0))
    if cxr_grid_policy is not None:
        sweep = replace(sweep, cxr_grid_policy=cxr_grid_policy)
    return build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]


def _strict_check(monkeypatch, refusals):
    """Refuse the first ``refusals`` audited axes at half their measured upper share.

    Returns the distinct audited stops in order; a route may check one audit
    more than once.
    """
    check = cxr_grid.check_cxr_truncation
    stops = []

    def strict(case, audit):
        stop = float(audit["stop_eV"])
        if stop not in stops:
            stops.append(stop)
        if stops.index(stop) >= refusals:
            return check(case, audit)
        total = float(_to_cpu(audit["line_mass"]))
        share = float(_to_cpu(audit["mass_above"])) / total
        payload = case["line_grid_policy"]
        bandwidth = {**payload["bandwidth"], "truncation_limit": 0.5 * share}
        return check({**case, "line_grid_policy": {**payload, "bandwidth": bandwidth}}, audit)

    # The retained route binds the name at import; the streamed one reads cxr_grid.
    monkeypatch.setattr(case_driver, "check_cxr_truncation", strict)
    monkeypatch.setattr(cxr_grid, "check_cxr_truncation", strict)
    return stops


def _clear_grid_cache():
    shutil.rmtree(cache_dir(), ignore_errors=True)


def _retained(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


@pytest.mark.slow
@pytest.mark.parametrize("refusals", [1, 2])
def test_streamed_refusal_extends_like_the_retained_rescore(monkeypatch, refusals):
    """Validation: line-grid-resonance-bandwidth"""
    case = _hbn_case()
    assert case["line_grid_policy"]["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY

    _clear_grid_cache()
    retained_calls = _strict_check(monkeypatch, refusals)
    with pytest.warns(CxrGridTruncationWarning, match="rescoring on"):
        reference = _retained(case)
    _clear_grid_cache()
    streamed_calls = _strict_check(monkeypatch, refusals)
    with pytest.warns(CxrGridTruncationWarning, match="rescoring on"):
        streamed = run_case_streamed(case, 1, transport_core="per-electron")

    assert len(streamed_calls) == len(retained_calls) == refusals + 1
    np.testing.assert_allclose(streamed_calls, retained_calls, rtol=1e-12)
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    for key in SPECTRA:
        scale = np.abs(reference[key]).max()
        np.testing.assert_allclose(
            streamed[key], reference[key], rtol=1e-11, atol=1e-14 * scale, err_msg=key
        )
    fallback = streamed["line_grid_resolved"]["bandwidth_fallback"]
    expected = reference["line_grid_resolved"]["bandwidth_fallback"]
    assert len(fallback["refusals"]) == len(expected["refusals"]) == refusals
    assert fallback["stop_eV"] == expected["stop_eV"] == streamed["E_grid"][-1]
    assert fallback["refused"]["stop_eV"] < fallback["stop_eV"]
    assert "truncation_audit" in streamed["line_grid_resolved"]
    grid_pass = streamed["_streamed_line_grid_pass"]
    assert grid_pass["fallback_transports"] == 2 * refusals


@pytest.mark.slow
def test_streamed_explicit_measured_bandwidth_still_refuses(monkeypatch):
    """An explicitly chosen measured bandwidth never falls back.

    Validation: line-grid-resonance-bandwidth
    """
    _strict_check(monkeypatch, 1)
    _clear_grid_cache()
    case = _hbn_case({"bandwidth": RESONANCE_BANDWIDTH_POLICY})

    with pytest.raises(CxrGridToleranceError, match="bandwidth share"):
        run_case_streamed(case, 1, transport_core="per-electron")


@pytest.mark.slow
def test_streamed_unrefused_axis_runs_no_fallback_pass():
    _clear_grid_cache()
    streamed = run_case_streamed(_hbn_case(), 1, transport_core="per-electron")

    assert streamed["_streamed_line_grid_pass"]["fallback_transports"] == 0
    assert "bandwidth_fallback" not in streamed["line_grid_resolved"]
