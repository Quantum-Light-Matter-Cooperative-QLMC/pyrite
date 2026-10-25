"""Pin the external photon comparison, including its unresolved W discrepancy.

These tests audit stored independent-history blocks; no simulation or external
engine is run. They do not certify photon transport as anchored.
"""

import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[2] / "checks" / "photon_full_track"


@pytest.fixture(scope="module")
def comparison():
    spec = importlib.util.spec_from_file_location("cases", BENCH / "cases.py")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    with patch.dict(sys.modules, {"cases": cases}):
        return runpy.run_path(str(BENCH / "compare.py"))


def test_fraction_keeps_paired_block_covariance(comparison):
    # A fixed R/T in every history block has zero ratio SE, even when
    # numerator and denominator individually have substantial scatter.
    entry = {"total": [[4], [8], [12]], "redist": [[1], [2], [3]]}
    mean, se = comparison["_fraction"](entry, [1, 1, 1], 0, 1, np.array([0, 1]))
    assert mean == 0.25
    assert se == 0


def test_net_line_keeps_side_window_covariance(comparison):
    # Each block has peak - left - right = 2; independent window errors
    # would give a spurious positive uncertainty.
    entry = {"total": [[1, 4, 1], [2, 6, 2], [3, 8, 3]]}
    assert comparison["_line"](entry, "total", [1, 1, 1], np.array([0, 1, 3, 4]), 1, 3) == (
        2,
        0,
    )


def test_committed_reduction_checksums():
    manifest = json.loads((BENCH / "provenance.json").read_text())
    assert len(manifest["committed_reduction_sha256"]) == 16
    for path, digest in manifest["committed_reduction_sha256"].items():
        assert hashlib.sha256((BENCH / path).read_bytes()).hexdigest() == digest


def test_missing_case_cannot_pass(comparison, monkeypatch, tmp_path):
    monkeypatch.setitem(comparison["main"].__globals__, "HERE", tmp_path)
    with pytest.raises(FileNotFoundError):
        comparison["main"]([])


def test_compaction_preserves_window_block_samples(comparison):
    edges = comparison["edges_eV"](0.3)
    rng = np.random.default_rng(341)
    entry = {
        label: rng.random((4, len(edges) - 1)).tolist() for label in ("total", "direct", "redist")
    }
    record = {"case": "w_0p3mev", "edges_eV": edges.tolist(), "apertures": {"R": entry}}
    compact = comparison["compact_record"](record)["apertures"]["R"]
    for lo, hi in compact["windows_eV"]:
        for label in ("total", "direct", "redist"):
            np.testing.assert_array_equal(
                comparison["_counts"](compact, label, lo, hi, edges),
                comparison["_counts"](entry, label, lo, hi, edges),
            )


@pytest.mark.parametrize("case", [f"{t}_{e}mev" for t in ("si", "w") for e in ("0p3", 1, 3, 10)])
def test_committed_comparison_keeps_known_discrepancy_visible(comparison, case):
    record = comparison["load_record"](BENCH / "results" / f"{case}.json")
    assert record["events"] == 20_000
    assert record["meta"]["first_batches"] == list(range(0, 400, 20))
    assert record["format"] == "photon-full-track-windows.v1"
    report = comparison["compare_case"](case)
    failures = {
        (aperture, band, observable)
        for aperture, bands in report.items()
        for band, row in bands.items()
        for observable, result in row.items()
        if isinstance(result, dict) and result["gated"] and not result["pass_"]
    }
    expected = (
        {("R", "1-10keV", "redistributed"), ("R", "1-10keV", "redistributed_fraction")}
        if case == "w_0p3mev"
        else set()
    )
    assert failures == expected
