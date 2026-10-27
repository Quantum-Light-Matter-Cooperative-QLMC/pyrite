"""Pin the external photon comparison, including its unresolved W discrepancy.

These tests audit stored independent-history blocks; no simulation or external
engine is run. They do not certify photon transport as anchored.
"""

import gzip
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
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


@pytest.fixture(scope="module")
def macros():
    cases = ModuleType("cases")
    cases.__dict__.update(runpy.run_path(str(BENCH / "cases.py")))
    with patch.dict(sys.modules, {"cases": cases}):
        return runpy.run_path(str(BENCH / "make_macros.py"))


def test_complete_cascade_changes_only_atomic_settings_and_output_name(macros):
    original = macros["macro"]("w_0p3mev", 0)
    matched = macros["macro"]("w_0p3mev", 0, complete_cascade=True)
    restored = matched.replace("/process/em/auger true\n", "/process/em/auger false\n")
    restored = restored.replace("/process/em/deexcitationIgnoreCut true\n", "")
    restored = restored.replace("w_0p3mev_s00_cascade", "w_0p3mev_s00")
    assert restored == original
    assert "/testem/stack/killSecondaries 3\n" in matched
    assert "/random/setSeeds 34100 5\n" in matched


def test_cascade_logs_require_completed_histories_and_settings(comparison, tmp_path):
    path = tmp_path / "cascade.log.gz"
    header = "Auger electron cascade enabled 1\n### === Ignore cuts flag: 1\n"
    with gzip.open(path, "wt") as stream:
        stream.write(header)
    verify = comparison["verify_cascade_logs"]
    with pytest.raises(ValueError, match="completed"):
        verify("w_0p3mev", [path], 100)
    with gzip.open(path, "at") as stream:
        stream.write(" The run was 100 e- of 300 keV through 100 um of Tungsten\n")
    verify("w_0p3mev", [path], 100)
    with pytest.raises(ValueError, match="primary count"):
        verify("w_0p3mev", [path], 200)
    with pytest.raises(ValueError, match="nonduplicate"):
        verify("w_0p3mev", [path, path], 100)
    with gzip.open(path, "wt") as stream:
        stream.write(header.replace("Ignore cuts flag: 1", "Ignore cuts flag: 0"))
        stream.write(" The run was 100 e- of 300 keV through 100 um of Tungsten\n")
    with pytest.raises(ValueError, match="not enabled"):
        verify("w_0p3mev", [path], 100)


def test_cascade_comparison_cannot_pass_without_historical_gate_coverage(comparison, tmp_path):
    record = comparison["load_record"](BENCH / "reference" / "w_0p3mev.json")
    path, output = tmp_path / "reference.json", tmp_path / "report.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="verified complete-cascade"):
        comparison["main"](["compare-cascade", "w_0p3mev", str(path)])
    record["relaxation"] = "complete-cascade"
    for entry in record["apertures"].values():
        for label in ("total", "direct", "redist"):
            entry[label] = np.zeros_like(entry[label]).tolist()
    path.write_text(json.dumps(record))
    assert comparison["main"](["compare-cascade", "w_0p3mev", str(path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["scope"] == "w_0p3mev"
    assert report["gated_count"] == 0
    assert report["missing_gates"]


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


def test_incident_gamma_direct_classification(comparison, tmp_path):
    path = tmp_path / "gamma.log.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("GAM 0 20000 1 0 0 1 0 0 primary 1\n")
        stream.write("GAM 1 8200 -1 0 0 -1 0 0 phot 1\n")
        stream.write("GAM 2 19000 -1 0 0 -1 0 0 primary 0\n")
    rows = comparison["_parse"]([path], 0.1, 100, source_creator="primary")
    np.testing.assert_array_equal(rows[-1], [True, False, False])
    # The full-track electron-source convention remains unchanged.
    assert not comparison["_parse"]([path], 0.1, 100)[-1].any()


@pytest.fixture(scope="module")
def diagnostic(comparison):
    cases, reducer = ModuleType("cases"), ModuleType("compare")
    cases.__dict__.update(runpy.run_path(str(BENCH / "cases.py")))
    reducer.__dict__.update(comparison)
    with patch.dict(sys.modules, {"cases": cases, "compare": reducer}):
        return runpy.run_path(str(BENCH / "monoenergetic.py"))


def test_matched_photon_tally_keeps_zero_histories(diagnostic):
    # Two photons from one incident history must stay correlated; 99 blocks
    # with no detections still contribute to the mean and standard error.
    record = diagnostic["tally"](
        np.array([0, 0]),
        np.array([8200, 8200]),
        np.zeros((2, 3)),
        np.tile([0, 0, -1], (2, 1)),
        np.array([False, False]),
        100,
        creator=np.array(["phot", "phot"]),
    )
    mean, se = diagnostic["summarize"](record)["R"]["creator:phot"]["La"]
    assert mean == pytest.approx(0.02)
    assert se == pytest.approx(0.02)
    assert np.asarray(record["apertures"]["R"]["redist"]).sum() == 2
    assert np.asarray(record["apertures"]["T"]["total"]).sum() == 0
    with pytest.raises(ValueError, match="event id"):
        diagnostic["tally"](
            np.array([-1]),
            np.array([8200]),
            np.zeros((1, 3)),
            np.array([[0, 0, -1]]),
            np.array([False]),
            100,
        )


def test_matched_photon_reduction_rejects_incomplete_or_wrong_run(diagnostic, tmp_path):
    path = tmp_path / "run.log.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("GAM 0 20000 1 0 0 1 0 0 primary 1\n")
    with pytest.raises(ValueError, match="completed"):
        diagnostic["verify_geant4_run"](path, 20, 100)
    with gzip.open(path, "at") as stream:
        stream.write(" The run was 100 gamma of 20 keV through 100 um of Tungsten\n")
    diagnostic["verify_geant4_run"](path, 20, 100)
    with pytest.raises(ValueError, match="does not match"):
        diagnostic["verify_geant4_run"](path, 12, 100)
    with pytest.raises(ValueError, match="does not match"):
        diagnostic["verify_geant4_run"](path, 20, 1000)


@pytest.mark.parametrize("energy", [12, 20, 30])
def test_matched_photon_evidence_preserves_history_blocks(diagnostic, energy):
    directory = BENCH / "monoenergetic-results"
    manifest = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert len(manifest["committed_reduction_sha256"]) == 12
    assert manifest["ledger_status"] == "discrepancy"
    for setting in ("auger-off", "auger-cut", "cascade", "pyrite"):
        name = f"w_{energy}kev_{setting}"
        data = (directory / f"{name}.json.gz").read_bytes()
        assert (
            hashlib.sha256(data).hexdigest()
            == manifest["committed_reduction_sha256"][f"{name}.json.gz"]
        )
        record = json.loads(gzip.decompress(data))
        assert record["events"] == 100_000
        assert record["energy_keV"] == energy
        rebuilt = diagnostic["summarize"](record)
        for aperture, labels in record["apertures"].items():
            assert np.asarray(labels["total"]).shape[0] == 100
            np.testing.assert_array_equal(
                labels["total"], np.asarray(labels["direct"]) + labels["redist"]
            )
            for label, windows in rebuilt[aperture].items():
                for window, mean_se in windows.items():
                    np.testing.assert_allclose(
                        summary[name][aperture][label][window], mean_se, rtol=1e-14, atol=0
                    )


def test_complete_cascade_w_reference_clears_failures_without_losing_gates(comparison, tmp_path):
    directory = BENCH / "cascade-results"
    manifest = json.loads((directory / "provenance.json").read_text())
    for name, digest in manifest["committed_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert (
        manifest["original_pyrite_result_sha256"]
        == hashlib.sha256((BENCH / "results" / "w_0p3mev.json.gz").read_bytes()).hexdigest()
    )
    reference = comparison["load_record"](directory / "reference.json.gz")
    assert reference["events"] == 10_000_000
    assert len(reference["log_sha256"]) == 10
    path = tmp_path / "report.json"
    assert not comparison["main"](
        ["compare-cascade", "w_0p3mev", str(directory / "reference.json.gz"), "--output", str(path)]
    )
    fresh = json.loads(path.read_text())
    stored = json.loads((directory / "report.json").read_text())
    assert fresh["gated_count"] == stored["gated_count"] == 11
    assert fresh["missing_gates"] == fresh["failures"] == []
    for aperture, bands in fresh["matched"].items():
        for band, observables in bands.items():
            if "skipped" in observables:
                continue
            for observable, result in observables.items():
                pinned = stored["matched"][aperture][band][observable]
                assert result["gated"] == pinned["gated"]
                assert result["pass_"] == pinned["pass_"]
                np.testing.assert_allclose(result["pyrite"], pinned["pyrite"], rtol=1e-14)
                np.testing.assert_allclose(result["geant4"], pinned["geant4"], rtol=1e-14)


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


def test_complete_cascade_matrix_retains_all_100_gates(comparison, tmp_path):
    directory = BENCH / "cascade-matrix-results"
    manifest = json.loads((directory / "provenance.json").read_text())
    for name, digest in manifest["committed_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    total, shards = 11, 0  # Prior retained W 300 keV rerun has 11 gates.
    for case, expected in manifest["cases"].items():
        source = BENCH / "results" / f"{case}.json.gz"
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == manifest["original_pyrite_sha256"][case]
        )
        reference = comparison["load_record"](directory / f"{case}.json.gz")
        count = len(reference["log_sha256"])
        shards += count
        assert reference["events"] == count * 1_000_000
        output = tmp_path / f"{case}.json"
        assert not comparison["main"](
            ["compare-cascade", case, str(directory / f"{case}.json.gz"), "--output", str(output)]
        )
        fresh = json.loads(output.read_text())
        stored = json.loads((directory / f"{case}-report.json").read_text())
        assert fresh == stored
        assert fresh["failures"] == fresh["missing_gates"] == []
        assert fresh["gated_count"] == expected["gated_count"]
        total += fresh["gated_count"]
    assert shards == manifest["shards"] == 49
    assert total == manifest["total_gated_count_including_prior_w"] == 100
