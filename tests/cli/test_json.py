import json
import pickle

import pytest

from pyrite.cli import json as cli_json


def test_result_is_one_versioned_json_value_with_newline():
    result = cli_json.JsonResult("cxr.test", {"value": 3})

    rendered = result.dumps()

    assert rendered.endswith("\n") and rendered.count("\n") == 1
    assert json.loads(rendered) == {
        "schema": "cxr.test",
        "schema_version": 1,
        "ok": True,
        "payload": {"value": 3},
        "errors": [],
    }


def test_remote_jobs_retains_valid_rows_and_reports_partial_errors():
    result = cli_json.remote_jobs(
        "20260719-120000-abcd1234\t0048291\tFalse\thopg hbn\trunning hbn [2/2]\n"
        "bad row\n"
        "j2\t-\t?\thopg\tFAILED\x1b[31m\n"
    )

    assert result.ok is False
    assert len(result.payload["jobs"]) == 2
    first = result.payload["jobs"][0]
    assert first["scheduler_job_id"] == "0048291"
    assert first["materials"] == ["hopg", "hbn"]
    assert first["terminal"] is False
    assert result.payload["jobs"][1]["last_event"] == "FAILED?[31m"
    assert result.errors[0]["code"] == "malformed_job"


def test_remote_status_has_stable_types_sanitized_logs_and_partial_errors():
    result = cli_json.remote_status(
        {
            "JOB": "j1",
            "META": (
                "kind: rebrem\nmaterials: hopg hbn\nsubmitted_at: 2026-07-25T12:00:00-07:00\n"
            ),
            "STATE": "done now",
            "SQUEUE": (
                "job_id=0042|state=COMPLETED|partition=gpu|elapsed=1:02:03|"
                "left=00:00|nodes=2|reason=None"
            ),
            "PROGRESS": (
                '{"material":"hopg","total_cases":5,"cached_cases":2,'
                '"completed_new_cases":3,"state":"done"}\nnot-json\n'
            ),
            "LOG": "fine\nbad\x1b[31m\n",
        }
    )

    assert result.ok is False
    assert result.payload["job"]["submitted_at"] == "2026-07-25T19:00:00Z"
    assert result.payload["job"]["terminal"] is True
    assert result.payload["scheduler"]["elapsed_seconds"] == 3723
    assert result.payload["scheduler"]["nodes"] == 2
    assert result.payload["progress"]["completed_cases"] == 5
    assert result.payload["recent_log_lines"] == ["fine", "bad?[31m"]
    assert result.errors[0]["code"] == "malformed_progress"


def test_line_grid_defaults_encodes_units_and_source():
    result = cli_json.line_grid_defaults(
        {
            "energies": [30, 40],
            "tilts": [1],
            "azimuths": [2],
            "thickness_ang": [1e7],
            "brem_step_ev": 25,
            "materials": ["hopg"],
        },
        source="persisted",
    )

    assert result.payload == {
        "energies_keV": [30.0, 40.0],
        "tilts_deg": [1.0],
        "azimuths_deg": [2.0],
        "thickness_ang": [1e7],
        "brem_step_eV": 25.0,
        "materials": ["hopg"],
        "source": "persisted",
    }


def test_line_grid_show_has_grids_brem_and_provenance():
    materials = {"hopg": {}}
    energy_grids = {
        "hopg": {
            "line_by_energy": [
                {
                    "energy_keV": 30,
                    "grid": {
                        "linspace": {
                            "start": 10,
                            "stop": 2600,
                            "num": 864,
                            "endpoint": True,
                        }
                    },
                    "source": "manual",
                }
            ],
        }
    }
    brem_by_material = {"hopg": {"start": 0, "stop": 30000, "step": 25}}
    provenance = {
        "hopg": {
            "line": {"30": {"note": "reviewed"}},
            "brem": {"source": "derived"},
        }
    }

    result = cli_json.line_grid_show(materials, energy_grids, brem_by_material, provenance)

    item = result.payload["materials"][0]
    assert item["line_grids"][0]["grid"]["stop_eV"] == 2600.0
    assert item["line_grids"][0]["provenance"] == {
        "source": "manual",
        "note": "reviewed",
    }
    assert item["brem_grid"]["step_eV"] == 25.0


def test_line_grid_show_unknown_material_is_structured_failure():
    result = cli_json.line_grid_show({}, {}, {}, {}, selected="ghost")

    assert result.payload == {"profile": "standard", "materials": []}
    assert result.errors == (
        {
            "code": "unknown_material",
            "message": "material is not configured",
            "item": "ghost",
        },
    )


def test_archives_retains_unreadable_entries(tmp_path):
    shelf = tmp_path / "archive"
    shelf.mkdir()
    with open(shelf / "good.pkl", "wb") as stream:
        pickle.dump({"a": {30.0: {}, 40.0: {}}}, stream)
    (shelf / "bad.pkl").write_bytes(b"bad")

    def load(path):
        with open(path, "rb") as stream:
            return pickle.load(stream)

    result = cli_json.archives(tmp_path, loader=load)

    assert [item["label"] for item in result.payload["archives"]] == ["bad", "good"]
    assert result.payload["archives"][0]["readable"] is False
    assert result.payload["archives"][1]["record_count"] == 2
    assert result.errors[0]["item"] == "bad"


def test_operation_summary_represents_partial_failure_and_resumability():
    result = cli_json.operation_summary(
        "remote-pull",
        ["hopg", "hbn"],
        ["hopg"],
        checkpoints=["checkpoints/hopg.pkl"],
        elapsed_seconds=1.25,
        resumable=True,
        material_errors={"hbn": "copy failed\x1b"},
    )

    assert result.ok is False
    assert result.payload["failed_materials"] == ["hbn"]
    assert result.payload["material_errors"] == {"hbn": "copy failed?"}
    assert result.payload["resumable"] is True
    assert result.errors[0]["item"] == "hbn"


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_operation_summary_rejects_invalid_elapsed(elapsed):
    with pytest.raises(ValueError, match="elapsed_seconds"):
        cli_json.operation_summary("scan", [], [], elapsed_seconds=elapsed)
