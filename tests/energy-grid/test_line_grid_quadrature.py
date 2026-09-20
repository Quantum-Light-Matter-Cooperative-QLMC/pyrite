"""Line-quadrature opt-in through the line-grid policy (issue #116).

Pins the ``quadrature`` policy field (off by default, schema 3 only when
selected), its precedence, its case key and identity, the early refusals, the
cache schema, and that the runner hands the case's choice to ``mc_spectrum``.

Validation: sinc-bin-integration
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from pyrite import _line_grid_policy
from pyrite._line_grid_policy import (
    LINE_GRID_POLICY_SCHEMA,
    LINE_QUADRATURES,
    QUADRATURE_LINE_GRID_POLICY_SCHEMA,
    WINDOWED_LINE_GRID_POLICY_SCHEMA,
    cached_coordinates,
    line_quadrature_from_payload,
    resolve_line_grid_policy,
    store_coordinates,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.spectrum.lines import _bin_quadrature


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in _line_grid_policy.ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_line_grid_policy, "cache_dir", lambda: tmp_path)


def _sweep(energy_keV=77.0, **policy):
    sweep = material_sweep("hopg")
    return replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=energy_keV),
        line_grid_policy=policy or None,
    )


def _policy(**per_call):
    return resolve_line_grid_policy(start_eV=10.0, stop_eV=10_000.0, per_call=per_call or None)


# ---- policy field -------------------------------------------------------------
def test_quadratures_match_the_kernel_module():
    assert LINE_QUADRATURES == _bin_quadrature.LINE_QUADRATURES
    assert LINE_QUADRATURES[0] == _bin_quadrature.NODE_QUADRATURE


def test_node_quadrature_keeps_the_historical_payload():
    default = _policy().payload()
    assert default["schema"] == LINE_GRID_POLICY_SCHEMA
    assert "quadrature" not in default and "quadrature" not in default["sources"]
    assert _policy(quadrature="node").payload() == default
    windowed = _policy(windows=True).payload()
    assert _policy(windows=True, quadrature="node").payload() == windowed
    assert windowed["schema"] == WINDOWED_LINE_GRID_POLICY_SCHEMA


@pytest.mark.parametrize("windows", [False, True])
def test_bin_mean_bumps_the_schema_and_records_its_source(windows):
    payload = _policy(quadrature="bin-mean", windows=windows).payload()
    assert payload["schema"] == QUADRATURE_LINE_GRID_POLICY_SCHEMA
    assert payload["quadrature"] == "bin-mean"
    assert payload["sources"]["quadrature"] == "per-call"
    assert ("windows" in payload) is windows
    assert json.loads(json.dumps(payload)) == payload
    assert line_quadrature_from_payload(payload) == "bin-mean"
    assert line_quadrature_from_payload(None) == "node"


def test_quadrature_precedence_is_per_call_then_stored():
    stored = resolve_line_grid_policy(
        start_eV=10.0, stop_eV=10_000.0, stored={"quadrature": "bin-mean"}
    ).payload()
    assert stored["sources"]["quadrature"] == "stored configuration"
    overridden = resolve_line_grid_policy(
        start_eV=10.0,
        stop_eV=10_000.0,
        per_call={"quadrature": "node"},
        stored={"quadrature": "bin-mean"},
    ).payload()
    assert overridden == _policy().payload()


def test_unknown_quadrature_is_refused():
    with pytest.raises(ValueError, match="quadrature must be one of"):
        _policy(quadrature="simpson")
    with pytest.raises(ValueError, match="quadrature must be one of"):
        line_quadrature_from_payload({"quadrature": "simpson"})


def test_schema_three_cache_records_are_readable():
    store_coordinates("ab" * 16, {"schema": QUADRATURE_LINE_GRID_POLICY_SCHEMA, "num": 3})
    assert cached_coordinates("ab" * 16) == {"schema": QUADRATURE_LINE_GRID_POLICY_SCHEMA, "num": 3}


# ---- case identity ------------------------------------------------------------
def test_bin_mean_sets_the_case_key_and_changes_identity():
    node = build_cases(_sweep(), n_electrons=2, n_electrons_brem=2)[0]
    binned = build_cases(_sweep(quadrature="bin-mean"), n_electrons=2, n_electrons_brem=2)[0]
    assert "line_quadrature" not in node
    assert binned["line_quadrature"] == "bin-mean"
    assert binned["line_grid_policy"]["quadrature"] == "bin-mean"
    # an explicit per-call node policy is the automatic default, key for key
    explicit_node = build_cases(_sweep(quadrature="node"), n_electrons=2, n_electrons_brem=2)[0]
    assert case_content_key(explicit_node) == case_content_key(node)
    assert case_content_key(binned) != case_content_key(node)


def test_the_case_refuses_bin_mean_with_coherent_emission_or_substeps():
    with pytest.raises(ValueError, match="incoherent-only"):
        build_cases(
            _sweep(quadrature="bin-mean"),
            n_electrons=2,
            n_electrons_brem=2,
            coherent_emission=True,
        )
    with pytest.raises(ValueError, match="max_dE_frac"):
        build_cases(
            _sweep(quadrature="bin-mean"),
            n_electrons=2,
            n_electrons_brem=2,
            energy_model="midpoint",
            max_dE_frac=0.05,
        )
    case = build_cases(_sweep(), n_electrons=2, n_electrons_brem=2)[0]
    with pytest.raises(ValueError, match="absent or 'bin-mean'"):
        Case(**{**dict(case), "line_quadrature": "node"})


# ---- runner -------------------------------------------------------------------
def test_runner_forwards_the_case_quadrature(monkeypatch):
    seen = []

    def fake_spectrum(segs, E_grid, **kwargs):
        seen.append(kwargs["line_quadrature"])
        return np.zeros(len(E_grid))

    monkeypatch.setattr(runner, "mc_spectrum", fake_spectrum)
    grid = np.linspace(100.0, 200.0, 11)
    for case, expected in (
        (build_cases(_sweep(), n_electrons=2, n_electrons_brem=2)[0], "node"),
        (
            build_cases(_sweep(quadrature="bin-mean"), n_electrons=2, n_electrons_brem=2)[0],
            "bin-mean",
        ),
    ):
        runner._lines_for_segments({}, grid, dict(case), np.array([0.0, 0.0, -1.0]), None, None)
        assert seen[-1] == expected
