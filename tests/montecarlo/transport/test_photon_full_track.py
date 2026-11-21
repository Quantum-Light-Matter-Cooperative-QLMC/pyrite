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

BENCH = Path(__file__).resolve().parents[3] / "checks" / "photon_full_track"


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


def _assert_report_matches(actual, expected):
    """Keep report structure/verdicts exact; permit binary64 arithmetic rounding."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_report_matches(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, pinned in zip(actual, expected, strict=True):
            _assert_report_matches(value, pinned)
    elif isinstance(expected, float):
        # One ULP in hypot can propagate through the 3*sigma limit. Four ULPs
        # cover the two rounded operations without a physical/absolute tolerance.
        np.testing.assert_array_max_ulp(actual, expected, maxulp=4)
    else:
        assert actual == expected


@pytest.mark.parametrize("changed", [{"pass_": False}, {"gated": False}, {"sigma": 0.1}])
def test_report_comparison_rejects_changed_gates_and_numerics(comparison, changed):
    gate = comparison["_gate"]((1.0, 0.01), (1.0, 0.01), 0.1)
    with pytest.raises(AssertionError):
        _assert_report_matches({**gate, **changed}, gate)


@pytest.mark.parametrize(
    "round_toward", [None, -np.inf, np.inf], ids=["native", "down-ulp", "up-ulp"]
)
def test_complete_cascade_matrix_retains_all_100_gates(
    comparison, tmp_path, monkeypatch, round_toward
):
    # libm/SIMD hypot can round one ULP differently across supported platforms.
    if round_toward is not None:
        hypot = comparison["np"].hypot
        monkeypatch.setattr(
            comparison["np"],
            "hypot",
            lambda *args, **kwargs: np.nextafter(hypot(*args, **kwargs), round_toward),
        )
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
        _assert_report_matches(fresh, stored)
        assert fresh["failures"] == fresh["missing_gates"] == []
        assert fresh["gated_count"] == expected["gated_count"]
        total += fresh["gated_count"]
    assert shards == manifest["shards"] == 49
    assert total == manifest["total_gated_count_including_prior_w"] == 100


@pytest.fixture(scope="module")
def alloy(comparison):
    cases, reducer = ModuleType("cases"), ModuleType("compare")
    cases.__dict__.update(runpy.run_path(str(BENCH / "cases.py")))
    reducer.__dict__.update(comparison)
    with patch.dict(sys.modules, {"cases": cases, "compare": reducer}):
        return runpy.run_path(str(BENCH / "alloy_fluorescence.py"))


def test_alloy_composition_matches_geant4_nist_gaas(alloy):
    (ga, n_ga), (as_, n_as) = alloy["composition"]()
    assert (ga, as_) == ("Ga", "As") and n_ga == n_as
    # Back to mass density: 5.31 g/cm^3 for one Ga and one As atom per unit.
    rho = n_ga * 1e24 * (69.723 + 74.9216) / 6.02214076e23
    assert rho == pytest.approx(5.31, rel=1e-12)
    text = alloy["macro"](11.5, 100, 7)
    for line in (
        "/testem/det/setAbsMat G4_GALLIUM_ARSENIDE",
        "/testem/det/setAbsThick 0.2 mm",
        "/process/em/auger true",
        "/process/em/deexcitationIgnoreCut true",
        "/gun/energy 11.5 keV",
        "/gun/position -0.1000010 0 0 mm",
        "/testem/stack/killSecondaries 3",
    ):
        assert line in text.splitlines()


def test_alloy_line_windows_are_single_bins_on_both_sides_of_edges(alloy):
    edges = alloy["EDGES_EV"]
    for lo, hi in alloy["WINDOWS"].values():
        assert lo in edges and hi in edges and hi - lo == 500
    # As K-alpha/beta must sit above the Ga K edge for secondary excitation;
    # 11.5 keV must stay below the As K edge (11.867 keV).
    assert alloy["WINDOWS"]["As_Ka"][0] > alloy["GA_K_EDGE_EV"]
    assert set(alloy["GATED"][11.5]) == {"Ga_Ka", "Ga_Kb"}


def test_alloy_reduction_rejects_wrong_material_or_run(alloy):
    verify = alloy["verify_geant4_log"]
    good = " The run was 100 gamma of 15 keV through 200 um  of G4_GALLIUM_ARSENIDE\n"
    verify(good, 15.0, 100)
    with pytest.raises(ValueError, match="completed"):
        verify("GAM 0 15000 1 0 0 1 0 0 primary 1\n", 15.0, 100)
    with pytest.raises(ValueError, match="does not match"):
        verify(good, 30.0, 100)
    with pytest.raises(ValueError, match="does not match"):
        verify(good, 15.0, 1000)
    with pytest.raises(ValueError, match="G4_GALLIUM_ARSENIDE"):
        verify(good.replace("G4_GALLIUM_ARSENIDE", "Tungsten"), 15.0, 100)


def _alloy_record(alloy, rate, *, engine="x", energy=15.0, blocks=100, events=1_000_000):
    """Fixed per-block Ga K-alpha counts on R with small block scatter."""
    record = alloy["tally"](
        np.array([0]), np.array([9200.0]), np.zeros((1, 3)), np.array([[0, 0, -1.0]]),
        np.array([False]), 100,
    )  # fmt: skip
    size = events // blocks
    counts = np.zeros((blocks, len(alloy["EDGES_EV"]) - 1), dtype=np.int64)
    column = int(np.searchsorted(alloy["EDGES_EV"], 9200.0, side="right") - 1)
    counts[:, column] = np.round(rate * size + np.tile([-3, 3], blocks // 2)).astype(int)
    record["apertures"]["R"]["total"] = counts.tolist()
    record.update(block_sizes=[size] * blocks, format="f", engine=engine, energy_keV=energy)
    return record


def test_alloy_gate_detects_missing_secondary_fluorescence(alloy):
    g = _alloy_record(alloy, 0.01)
    report = alloy["compare"](_alloy_record(alloy, 0.0102), g)
    assert report["R"]["Ga_Ka"]["gated"] and report["R"]["Ga_Ka"]["pass_"]
    # A 15% deficit, the scale of an omitted As->Ga contribution, must fail.
    report = alloy["compare"](_alloy_record(alloy, 0.0085), g)
    assert alloy["failures"](report) == ["R/Ga_Ka"]
    # Windows with fewer than MIN_COUNTS Geant4 counts only report.
    assert not report["T"]["Ga_Ka"]["gated"]
    with pytest.raises(ValueError, match="energy"):
        alloy["compare"](_alloy_record(alloy, 0.01, energy=30.0), g)


def test_alloy_merge_concatenates_independent_shards(alloy):
    a = _alloy_record(alloy, 0.01, engine="p")
    b = _alloy_record(alloy, 0.01, engine="p")
    a.update(seed=1, first_history=0, ga_k_excitation=dict(ga_above_k_edge=10, indirect=2))
    b.update(seed=1, first_history=1_000_000, ga_k_excitation=dict(ga_above_k_edge=30, indirect=8))
    merged = alloy["merge"]([a, b])
    assert len(merged["block_sizes"]) == 200
    assert len(merged["apertures"]["R"]["total"]) == 200
    g = _alloy_record(alloy, 0.01)
    assert alloy["compare"](merged, g)["indirect_ga_k_fraction"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="disagree"):
        alloy["merge"]([a, _alloy_record(alloy, 0.01, engine="other")])


def test_alloy_livermore_arm_changes_only_physics_constructor(alloy):
    penelope = alloy["macro"](15.0, 100, 7).splitlines()
    livermore = alloy["macro"](15.0, 100, 7, "emlivermore").splitlines()
    changed = [(a, b) for a, b in zip(penelope, livermore, strict=True) if a != b]
    assert changed == [
        ("/testem/phys/addPhysics empenelope", "/testem/phys/addPhysics emlivermore")
    ]
    with pytest.raises(ValueError, match="physics"):
        alloy["macro"](15.0, 100, 7, "emstandard")


ALLOY = BENCH / "alloy-results"
#: Pre-registered Penelope reference: Ga K deficit near the Ga K edge
PENELOPE_FAILURES = [
    "gaas_11p5kev/R/Ga_Ka",
    "gaas_11p5kev/R/Ga_Kb",
    "gaas_15kev/R/Ga_Ka",
    "gaas_30kev/R/Ga_Ka",
    "gaas_30kev/T/Ga_Ka",
]


def test_alloy_evidence_checksums():
    manifest = json.loads((ALLOY / "provenance.json").read_text())
    assert len(manifest["committed_record_sha256"]) == 9
    for path, digest in manifest["committed_record_sha256"].items():
        assert hashlib.sha256((ALLOY / path).read_bytes()).hexdigest() == digest


def _alloy_report(alloy, arm):
    report = {}
    for energy in alloy["ENERGIES_KEV"]:
        stem = alloy["name"](energy)
        p = alloy["_load"](ALLOY / f"{stem}_pyrite.json.gz")
        g = alloy["_load"](ALLOY / f"{stem}_geant4{arm}.json.gz")
        assert len(p["block_sizes"]) == len(g["block_sizes"]) == 1000
        assert sum(p["block_sizes"]) == sum(g["block_sizes"]) == 1_000_000
        report[stem] = alloy["compare"](p, g)
    return report


def test_alloy_penelope_reference_keeps_ga_k_deficit_visible(alloy):
    report = _alloy_report(alloy, "")
    failed = [f"{case}/{f}" for case, r in report.items() for f in alloy["failures"](r)]
    assert failed == PENELOPE_FAILURES
    gated = sum(e["gated"] for r in report.values() for a in ("R", "T") for e in r[a].values())
    assert gated == 7
    # The near-edge primary baseline carries the largest deficit.
    assert report["gaas_11p5kev"]["R"]["Ga_Ka"]["ratio"] == pytest.approx(1.928, abs=1e-3)
    assert report["gaas_15kev"]["R"]["As_Ka"]["pass_"]


def test_alloy_livermore_arm_passes_every_preregistered_gate(alloy):
    report = _alloy_report(alloy, "_emlivermore")
    assert [f for r in report.values() for f in alloy["failures"](r)] == []
    gated = [
        (case, a, w)
        for case, r in report.items()
        for a in ("R", "T")
        for w, e in r[a].items()
        if e["gated"]
    ]
    # The Livermore reference holds 1000+ counts in one more window (8 gated).
    assert len(gated) == 8
    # Secondary fluorescence is a large share of Ga K excitation only once As K
    # is excited, so the passing 15/30 keV Ga K gates are sensitive to it.
    assert report["gaas_11p5kev"]["indirect_ga_k_fraction"] < 0.02
    for case in ("gaas_15kev", "gaas_30kev"):
        assert report[case]["indirect_ga_k_fraction"] > 0.3
        assert report[case]["R"]["Ga_Ka"]["gated"] and report[case]["R"]["Ga_Ka"]["pass_"]


@pytest.fixture(scope="module")
def escape(comparison):
    cases, reducer = ModuleType("cases"), ModuleType("compare")
    cases.__dict__.update(runpy.run_path(str(BENCH / "cases.py")))
    reducer.__dict__.update(comparison)
    with patch.dict(sys.modules, {"cases": cases, "compare": reducer}):
        return runpy.run_path(str(BENCH / "annihilation_escape.py"))


def test_escape_macros_place_sources_in_testem5_frame(escape):
    ext = escape["macro"]("w_ext", 100, 7).splitlines()
    internal = escape["macro"]("si_int", 100, 7).splitlines()
    assert "/gun/position -2.0000010 0 0 mm" in ext  # 1 nm before the -T/2 face
    assert "/gun/position 0.0000000 0 0 mm" in internal  # slab centre
    assert "/testem/det/setAbsMat Tungsten" in ext
    assert "/testem/det/setAbsThick 25 mm" in internal
    assert "/gun/energy 510.99895 keV" in ext
    livermore = escape["macro"]("w_ext", 100, 7, "emlivermore").splitlines()
    assert [(a, b) for a, b in zip(ext, livermore, strict=True) if a != b] == [
        ("/testem/phys/addPhysics empenelope", "/testem/phys/addPhysics emlivermore")
    ]


def test_escape_observables_align_with_histogram_edges(escape):
    edges = escape["EDGES_EV"]
    assert np.all(np.diff(edges) > 0)
    for _, _, lo, hi, _, _ in escape["OBSERVABLES"]:
        assert lo in edges and hi in edges
    # The 511 keV line sits strictly inside the direct window's last bins.
    assert edges[-1] > escape["ENERGY_KEV"] * 1e3 > 504.0e3


def _escape_record(escape, config, direct_rate, *, engine="x", events=1_000_000):
    record = escape["tally"](
        config, np.array([0]), np.array([511e3]), np.array([[0, 0, 1e9]]),
        np.array([[0, 0, 1.0]]), np.array([True]), 100,
    )  # fmt: skip
    size = events // 100
    counts = np.zeros((100, len(escape["EDGES_EV"]) - 1), dtype=np.int64)
    column = int(np.searchsorted(escape["EDGES_EV"], 511e3, side="right") - 1)
    counts[:, column] = np.round(direct_rate * size + np.tile([-5, 5], 50)).astype(int)
    record["apertures"]["T"]["direct"] = counts.tolist()
    record.update(block_sizes=[size] * 100, format="f", engine=engine)
    return record


def test_escape_direct_gate_tolerance(escape):
    g = _escape_record(escape, "w_ext", 0.02)
    ok = escape["compare"](_escape_record(escape, "w_ext", 0.0203), g)
    assert ok["T"]["direct_511"]["gated"] and ok["T"]["direct_511"]["pass_"]
    bad = escape["compare"](_escape_record(escape, "w_ext", 0.0206), g)
    assert escape["failures"](bad) == ["T/direct_511"]
    assert "W_Ka" in bad["R"]
    assert (
        "W_Ka"
        not in escape["compare"](
            _escape_record(escape, "si_ext", 0.02), _escape_record(escape, "si_ext", 0.02)
        )["R"]
    )
    with pytest.raises(ValueError, match="configuration"):
        escape["compare"](_escape_record(escape, "w_int", 0.02), g)


def test_escape_reduction_rejects_wrong_run(escape):
    verify = escape["verify_geant4_log"]
    good = (
        " The run was 100 gamma of 510.999 keV through 4 mm  of Tungsten (density: 19.3 g/cm3 )\n"
    )
    verify(good, "w_ext", 100)
    with pytest.raises(ValueError, match="does not match"):
        verify(good, "w_ext", 1000)
    with pytest.raises(ValueError, match="Silicon"):
        verify(good, "si_ext", 100)
    with pytest.raises(ValueError, match="completed"):
        verify("GAM 0 511000 0 0 0 1 0 0 primary 1\n", "w_ext", 100)


def test_alloy_windows_hold_the_documented_eadl_lines(alloy):
    # The adjacent single-bin windows split As K-alpha: Ka2 (K-L2, 10.46 keV)
    # falls in the window labelled Ga_Kb, Ka1 (K-L3, 10.50 keV) only 0.7 eV
    # above the 10.5 keV edge. Gate decisions stay like-for-like only while
    # these placements hold, so pin them against the packaged EADL energies.
    from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation

    def window(energy):
        hits = [n for n, (lo, hi) in alloy["WINDOWS"].items() if lo <= energy < hi]
        return hits[0] if hits else None

    def k_lines(element):
        k = load_eadl_relaxation(element).subshells[0]
        assert k.shell_designator == 1
        return dict(zip(k.radiative_final.tolist(), k.radiative_energy_eV.tolist(), strict=True))

    ga, as_ = k_lines("Ga"), k_lines("As")
    assert window(ga[3]) == window(ga[4]) == "Ga_Ka"  # K-L2, K-L3
    assert window(as_[3]) == "Ga_Kb"  # As Ka2 shares the Ga K-beta window
    assert window(as_[4]) == "As_Ka" and as_[4] - 10500.0 < 5.0
    assert window(ga[7]) == "Ga_Kb"  # K-M3


ESCAPE = BENCH / "escape-results"


def test_escape_evidence_checksums():
    manifest = json.loads((ESCAPE / "provenance.json").read_text())
    assert len(manifest["committed_record_sha256"]) == 12
    for path, digest in manifest["committed_record_sha256"].items():
        assert hashlib.sha256((ESCAPE / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("arm", ["", "_emlivermore"])
def test_escape_passes_every_gate_against_both_geant4_arms(escape, arm):
    gated = 0
    for config in escape["CONFIGS"]:
        p = escape["_load"](ESCAPE / f"{config}_pyrite.json.gz")
        g = escape["_load"](ESCAPE / f"{config}_geant4{arm}.json.gz")
        assert sum(p["block_sizes"]) == sum(g["block_sizes"]) == 1_000_000
        report = escape["compare"](p, g)
        assert escape["failures"](report) == []
        gated += sum(e["gated"] for a in report.values() for e in a.values())
        # Narrow-beam 511 keV escape is the sharpest check: within 0.4%.
        direct = report["T"]["direct_511"]
        assert direct["gated"] and abs(direct["ratio"] - 1) < 0.004
    assert gated == 14


def test_escape_w_backscatter_shape_follows_doppler_broadening(escape):
    # Below the pre-registered 1000-count threshold. Free-electron Compton
    # gave 0.20 (w_ext, 300-504 keV: no K/L-shell Doppler tail) and 0.51
    # (w_int, 100-300 keV: narrow single-backscatter peak). The #1231 rerun
    # with impulse-approximation broadening and Compton-vacancy relaxation
    # (job 14196, commit 8fd3b1a16) brings both within counting statistics.
    rows = {}
    for config in ("w_ext", "w_int"):
        report = escape["compare"](
            escape["_load"](ESCAPE / f"{config}_pyrite.json.gz"),
            escape["_load"](ESCAPE / f"{config}_geant4.json.gz"),
        )
        rows[config] = report["R"]
    for config, window, ratio in (
        ("w_ext", "redist_300-504keV", 25 / 30),
        ("w_int", "redist_100-300keV", 81 / 80),
    ):
        row = rows[config][window]
        assert not row["gated"] and row["pass_"]
        assert row["ratio"] == pytest.approx(ratio)


def test_shard_merges_reject_duplicates(escape, alloy):
    record = escape["tally"](
        "w_ext", np.array([0]), np.array([511e3]), np.array([[0, 0, 1e9]]),
        np.array([[0, 0, 1.0]]), np.array([True]), 100,
    )  # fmt: skip
    record.update(format="f", engine="p", seed=1, first_history=0)
    with pytest.raises(ValueError, match="duplicate"):
        escape["merge"]([record, dict(record)])
    other = dict(record, first_history=100)
    assert len(escape["merge"]([record, other])["block_sizes"]) == 200
    g = _alloy_record(alloy, 0.01)
    g.update(seed=1, first_history=0)
    with pytest.raises(ValueError, match="duplicate"):
        alloy["merge"]([g, dict(g)])
