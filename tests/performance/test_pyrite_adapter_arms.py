"""#510 PyRITE adapter arms: scene loading, primary grouping and transport-only work."""

import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "checks/equal_precision"


@pytest.fixture(scope="module")
def adapter():
    return runpy.run_path(str(ROOT / "pyrite_adapter.py"))


def rows():
    """Five primaries, several rows each; event 0 is an interior substep."""
    return {
        "electron_id": np.array([0, 0, 1, 2, 2, 2, 3, 4]),
        "event_kind": np.array([0, 5, 4, 0, 0, 7, 6, 5]),
        "E_end_keV": np.array([700.0, 650.0, 400.0, 1, 1, 9.0, 300.0, 120.0]),
        "hard_radiative_k_eV": np.array([0, 2e4, 0, 6e4, 2e5, 0, 0, 0.0]),
    }


def test_transport_scores_group_rows_by_primary(adapter):
    scores = adapter["transport_scores"](rows(), 5)
    assert scores["transmitted"].tolist() == [1, 0, 0, 0, 1]
    assert scores["backscattered"].tolist() == [0, 1, 0, 0, 0]
    assert scores["stopped"].tolist() == [0, 0, 1, 0, 0]
    assert scores["side_exited"].tolist() == [0, 0, 0, 1, 0]
    # Only the terminal transmitted row's energy counts; others score zero.
    assert scores["transmitted_energy_keV_per_primary"].tolist() == [650, 0, 0, 0, 120]
    assert not any(name.startswith("photons") for name in scores)
    source = adapter["score_segments"](rows(), 5)
    assert source["photons_ge10kev"].tolist() == [1, 0, 2, 0, 0]
    with pytest.raises(RuntimeError, match="terminal"):
        adapter["transport_scores"](rows(), 6)


def test_transport_only_arm_is_uncoupled_and_refuses_photon_rows(adapter):
    assert adapter["radiative_arguments"]("transport", "none", 1000.0, {"W": 1}) == {
        "radiative_model": "uncoupled",
        "radiative_cutoff_eV": None,
        "bremslib_tables": None,
    }
    retained = adapter["radiative_arguments"]("transport", "bremslib-soft-hard", 1000.0, {"W": 1})
    assert retained["radiative_model"] == "bremslib-soft-hard"
    assert adapter["radiative_arguments"]("detector", "none", 1000.0, {})["radiative_cutoff_eV"]
    with pytest.raises(RuntimeError, match="radiative rows"):
        adapter["check_transport_only"](rows(), "none")
    adapter["check_transport_only"](rows(), "bremslib-soft-hard")
    forbidden = adapter["forbidden_module"]
    assert forbidden("pyrite.montecarlo.spectrum.cxr._batched", "source")
    assert not forbidden("pyrite.montecarlo.spectrum.brem_events", "detector")
    assert forbidden("pyrite.montecarlo.spectrum.brem_events", "transport")
    assert forbidden("pyrite.perf.detector_scoring", "transport", "bremslib-soft-hard")
    assert forbidden("pyrite.montecarlo.spectrum.brem_bremslib", "transport")
    # Retained BremsLib loss samples hard photons from the staged tables in transport.
    assert not forbidden(
        "pyrite.montecarlo.spectrum.brem_bremslib", "transport", "bremslib-soft-hard"
    )
    assert not forbidden("pyrite.montecarlo.transport.cores", "transport")
    # The brem estimator's shared geometry helpers live in the CXR package.
    kernels = "pyrite.montecarlo.spectrum.cxr._kernels"
    assert not forbidden(kernels, "detector", "none", "_observation_direction")
    assert forbidden(kernels, "detector", "none", "chi_g")
    assert forbidden(kernels, "transport", "none", "_observation_direction")


def test_scenes_resolve_from_manifest_and_matrix(adapter):
    raw, spec, path = adapter["load_scene"]("si-300-100um-0deg")
    assert path.name == "manifest.json" and raw["id"] == "si-300-100um-0deg"
    assert spec["composition"] == [("Si", raw["number_density_ang3"])] and not spec["detector"]
    raw, spec, path = adapter["load_scene"]("tungsten-100kev-tr3-0deg-det135-5")
    assert path.name == "matrix.json" and spec["catalog_key"] == "tungsten"
    assert spec["detector"] == {"center_deg": 135, "half_angle_deg": 5}
    assert spec["thickness_um"] == raw["thickness_um"] and spec["unsupported"] == []
    _raw, spec, _path = adapter["load_scene"]("silicon-800kev-tr0.3-0deg-secondaries")
    assert "secondary_transport" in spec["unsupported"]
    with pytest.raises(SystemExit):
        adapter["load_scene"]("no-such-scene")


class _Table:
    key, digest, manifest = "k", "d", {"arrays_sha256": "a"}
    tables = ()
    arrays = {}


def test_transport_arm_writes_comparison_record_without_emission_work(
    adapter, monkeypatch, tmp_path
):
    seen = {}

    def simulate(**kwargs):
        seen.setdefault("calls", []).append(kwargs)
        n, start = kwargs["Ne"], kwargs["_electron_start"]
        ids = np.arange(n)
        return {
            "electron_id": ids,
            "event_kind": np.where((ids + start) % 2 == 0, 5, 4),
            "E_end_keV": np.full(n, 90.0),
            "v_hat": np.zeros((n, 3)),
            "L_ang": np.ones(n),
        }

    def refuse(*args, **kwargs):
        raise AssertionError("emission or BremsLib work in the transport-only arm")

    entry = types.SimpleNamespace(tables=[_Table()], arrays={})
    stopping = types.SimpleNamespace(key="s", digest="d", manifest={"arrays_sha256": "a"})
    stopping.arrays = dict
    monkeypatch.setattr("pyrite.montecarlo.transport.simulate_trajectories", simulate)
    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.resolve_layer_tables", lambda c: [entry])
    monkeypatch.setattr(
        "pyrite.xsgen.sbethe.catalog.resolve_composition_table", lambda k, c: stopping
    )
    monkeypatch.setattr("pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables", refuse)
    monkeypatch.setattr("pyrite.montecarlo.spectrum.brem_events.mc_coupled_brem_spectrum", refuse)
    output = tmp_path / "t.json"
    argv = ["pyrite_adapter.py", "silicon-100kev-tr0.03-0deg", "--histories", "6", "--seed", "3",
            "--core", "per-electron", "--arm", "transport", "--batch-histories", "4",
            "--audit-cxr", "--output", str(output)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    adapter["main"]()
    record = json.loads(output.read_text())
    assert record["schema"] == "pyrite.equal-precision.comparison.v1"
    assert record["kind"] == "transport" and record["status"] == "ok"
    # Warmup and measurement, each in two counter-addressed blocks (4 + 2 primaries).
    assert [c["_electron_start"] for c in seen["calls"]] == [0, 4, 0, 4]
    assert {c["radiative_model"] for c in seen["calls"]} == {"uncoupled"}
    assert {c["bremslib_tables"] for c in seen["calls"]} == {None}
    assert record["moments"]["transmitted"]["mean"] == pytest.approx(0.5)
    assert record["moments"]["transmitted"]["n"] == 6
    assert record["transmitted_energy"]["mean_keV"] == 90.0
    assert not any(k.startswith(("photons", "cone")) for k in record["moments"])
    provenance = record["provenance"]
    assert provenance["radiative_energy_loss"]["radiative_model"] == "uncoupled"
    assert provenance["cxr_execution_audit"] == {"enabled": True, "calls": []}
    assert provenance["tables"]["bremslib"] == []
    raw = np.load(output.with_suffix(".npz"))
    assert raw["transmitted"].tolist() == [1, 0, 1, 0, 1, 0]


def test_unsupported_scene_keeps_an_explicit_outcome(adapter, monkeypatch, tmp_path):
    output = tmp_path / "u.json"
    argv = ["pyrite_adapter.py", "silicon-800kev-tr0.3-0deg-div5mrad", "--histories", "10",
            "--seed", "1", "--core", "per-electron", "--arm", "transport",
            "--output", str(output)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    adapter["main"]()
    record = json.loads(output.read_text())
    assert record["status"] == "unsupported-physics" and "divergence" in record["reason"]
    assert "moments" not in record
