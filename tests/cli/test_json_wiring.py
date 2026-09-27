import json
import pickle

import pytest

from pyrite import energy_grid
from pyrite.checkpoints import recompute
from pyrite.cli import command as root_command
from pyrite.cli.commands import archive
from pyrite.cli.commands import blaze as blaze_cli
from pyrite.cli.commands import job as job_cli
from pyrite.cli.commands import recompute as recompute_cli
from pyrite.cli.commands import remote as remote_cli
from pyrite.cli.commands import scan as scan_cli
from pyrite.cli.commands.energy_grid_surface import material_command, profile_command
from pyrite.energy_grid import apply as _energy_grid_apply  # noqa: F401 - binds the submodule
from pyrite.energy_grid import defaults as _energy_grid_defaults  # noqa: F401 - binds it
from pyrite.remote import lifecycle, viewer
from pyrite.runs import blaze, scan
from tests.helpers.cli import invoke


def _document(result, *, exit_code=0):
    assert result.exit_code == exit_code
    assert result.stderr == ""
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    return json.loads(result.stdout)


def test_remote_jobs_json_is_one_sanitized_envelope(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "jobs_raw",
        lambda: "j1\t0042\tFalse\trebrem\thopg hbn\trunning\x1b[31m\n",
    )

    document = _document(invoke(job_cli.command, ["list", "-o", "json"]))

    assert document["schema"] == "cxr.remote.jobs"
    assert document["payload"]["jobs"][0]["scheduler_job_id"] == "0042"
    assert document["payload"]["jobs"][0]["command"] == "rebrem"
    assert document["payload"]["jobs"][0]["last_event"] == "running?[31m"


def test_remote_jobs_human_output_accepts_six_field_transport(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "jobs_raw",
        lambda: "j1\t0042\tFalse\trebrem\thopg hbn\trunning\n",
    )

    result = invoke(job_cli.command, ["list"])

    assert result.exit_code == 0
    assert "j1" in result.stdout
    assert "hopg, hbn" in result.stdout
    assert result.stderr == ""


def test_output_selector_is_the_only_machine_output_door(monkeypatch):
    """`--json` retired at 0.3.0; `-o/--output` carries the contract alone."""
    monkeypatch.setattr(
        viewer,
        "jobs_raw",
        lambda: "j1\t0042\tFalse\trun\thopg\tdone\n",
    )

    canonical = invoke(root_command, ["job", "list", "-o", "json"])
    retired = invoke(root_command, ["job", "list", "--json"])
    wide = invoke(root_command, ["job", "list", "-o", "wide"])
    help_result = invoke(root_command, ["job", "list", "--help"])

    assert canonical.exit_code == wide.exit_code == 0
    assert canonical.stderr == wide.stderr == ""
    assert retired.exit_code == 2
    assert "No such option" in retired.stderr
    assert "-o, --output [table|json|wide]" in help_result.stdout
    assert "--json" not in help_result.stdout


def test_remote_status_json_fetches_full_detail(monkeypatch):
    seen = {}

    def status_sections(jobid, detail):
        seen.update(jobid=jobid, detail=detail)
        return (
            {
                "JOB": "j1",
                "STATE": "done now",
                "META": "materials: hopg\nkind: scan\n",
                "SQUEUE": "job_id=7|state=COMPLETED|nodes=1",
                "PROGRESS": (
                    '{"material":"hopg","total_cases":2,"cached_cases":1,'
                    '"completed_new_cases":1,"state":"done"}'
                ),
                "LOG": "complete",
            },
            "raw",
        )

    monkeypatch.setattr(viewer, "status_sections", status_sections)

    document = _document(invoke(job_cli.command, ["status", "j1", "-o", "json"]))

    assert seen == {"jobid": "j1", "detail": 2}
    assert document["schema"] == "cxr.remote.status"
    assert document["payload"]["progress"]["completed_cases"] == 2
    assert document["payload"]["recent_log_lines"] == ["complete"]


def test_remote_jobs_runtime_failure_is_json_and_nonzero(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "jobs_raw",
        lambda: (_ for _ in ()).throw(SystemExit("ssh failed")),
    )

    document = _document(invoke(job_cli.command, ["list", "-o", "json"]), exit_code=1)

    assert document["ok"] is False
    assert document["payload"] == {"jobs": []}
    assert document["errors"][0]["message"] == "ssh failed"


def test_line_grid_defaults_json_is_read_only(monkeypatch, tmp_path):
    defaults_path = tmp_path / "defaults.toml"
    defaults_path.write_text("")
    monkeypatch.setattr(energy_grid.defaults, "DEFAULTS_PATH", defaults_path)
    monkeypatch.setattr(
        energy_grid.defaults,
        "load_defaults",
        lambda: {
            "energies": [30],
            "tilts": [],
            "azimuths": [],
            "thickness_ang": [1e7],
            "brem_step_ev": 25,
            "materials": ["hopg"],
        },
    )

    document = _document(invoke(profile_command, ["defaults", "-o", "json"]))

    assert document["schema"] == "cxr.energy-grid.defaults"
    assert document["payload"]["source"] == "persisted"
    result = invoke(profile_command, ["defaults", "--save-default", "-o", "json"])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_line_grid_show_json_reads_catalog_and_provenance(monkeypatch, tmp_path):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        """
[materials.hopg]

[energy_grids.hopg]
line_by_energy = [
  { energy_keV = 30, grid = { linspace = { start = 1, stop = 2, num = 3 } }, source = "derived" },
]

[profiles.standard]
E_grid_brem = { arange = { start = 0, stop = 10, step = 1 } }
"""
    )
    monkeypatch.setattr(energy_grid.apply, "_CATALOG_PATH", catalog)
    monkeypatch.setattr(energy_grid.apply._provenance, "load", lambda: {})

    document = _document(invoke(material_command, ["show", "hopg", "-o", "json"]))

    assert document["schema"] == "cxr.energy-grid.show"
    material = document["payload"]["materials"][0]
    assert material["line_grids"][0]["grid"]["points"] == 3
    assert material["brem_grid"]["step_eV"] == 1.0


def test_line_grid_show_json_resolves_selected_profile_artifact(monkeypatch, tmp_path):
    from pyrite import _energy_grid_artifacts as artifacts

    identity = artifacts.artifact_identity(
        "hopg",
        [{"energy_keV": 30, "start_eV": 10, "stop_eV": 2800, "num": 900}],
        {"start_eV": 0, "stop_eV": 150000, "step_eV": 25},
        [30],
    )
    stored = artifacts.write_artifact(tmp_path / "energy-grid-artifacts", identity)
    catalog = tmp_path / "materials.toml"
    catalog.write_text(
        f"""
[materials.hopg]

[energy_grids.hopg]
line_by_energy = [
  {{ energy_keV = 30, grid = {{ linspace = {{ start = 1, stop = 2, num = 3 }} }}, source = "derived" }},
]

[profiles.standard]
E_grid_brem = {{ arange = {{ start = 0, stop = 10, step = 1 }} }}

[profiles.campaign]
energy_grid_refs = {{ hopg = "{stored.digest}" }}
"""
    )
    monkeypatch.setattr(energy_grid.apply, "_CATALOG_PATH", catalog)
    monkeypatch.setattr(energy_grid.apply._provenance, "load", lambda: {})

    document = _document(
        invoke(
            material_command,
            ["show", "hopg", "--profile", "campaign", "-o", "json"],
        )
    )

    assert document["payload"]["profile"] == "campaign"
    material = document["payload"]["materials"][0]
    assert material["artifact_sha256"] == stored.digest
    assert material["line_grids"][0]["grid"]["stop_eV"] == 2800.0
    assert material["brem_grid"]["stop_eV"] == 150000.0


def test_archives_json_retains_unreadable_entry(monkeypatch, tmp_path):
    shelf = tmp_path / "archive"
    shelf.mkdir()
    with (shelf / "good.pkl").open("wb") as stream:
        pickle.dump({"a": {30.0: {}}}, stream)
    (shelf / "bad.pkl").write_bytes(b"bad")
    monkeypatch.setattr(archive, "DEFAULT_ROOT", str(tmp_path))

    document = _document(invoke(archive.archives_command, ["-o", "json"]), exit_code=1)

    assert document["schema"] == "cxr.archives"
    assert [item["label"] for item in document["payload"]["archives"]] == ["bad", "good"]
    assert document["payload"]["archives"][1]["record_count"] == 1


@pytest.mark.parametrize(
    ("complete", "exit_code", "resumable"),
    [(True, 0, False), (False, 75, True)],
)
def test_run_json_suppresses_human_output_and_preserves_resumable_exit(
    monkeypatch, complete, exit_code, resumable
):
    def run_material(_args, material, max_seconds=None):
        print(f"progress {material} {max_seconds}")
        return complete

    monkeypatch.setattr(scan, "_run_material", run_material)

    document = _document(
        invoke(scan_cli.command, ["standard", "-m", "hopg", "-o", "json"]), exit_code=exit_code
    )

    assert document["schema"] == "cxr.operation-summary"
    assert document["payload"]["operation"] == "run"
    assert document["payload"]["resumable"] is resumable
    assert "progress" not in json.dumps(document)


def test_blaze_json_suppresses_human_output(monkeypatch):
    monkeypatch.setattr(blaze, "run", lambda _args: print("human progress"))

    document = _document(
        invoke(
            blaze_cli.command,
            ["hopg", "--energy", "30", "--spacing", "1e-6", "-o", "json"],
        )
    )

    assert document["payload"]["operation"] == "blaze"
    assert document["payload"]["completed_materials"] == ["hopg"]


@pytest.mark.parametrize(
    ("command", "name"),
    [(recompute_cli.brem_command, "rebrem"), (recompute_cli.line_command, "reline")],
)
def test_recompute_json_uses_status_for_partial_summary(monkeypatch, command, name):
    target = f"{name}_checkpoints"

    def _driver(*_args, summary_status, **_kwargs):
        print("human progress")
        summary_status.update(hopg={"complete": True}, hbn={"complete": False})
        raise SystemExit(75)

    monkeypatch.setattr(recompute, target, _driver)

    document = _document(
        invoke(command, ["hopg", "hbn", "-o", "json"]),
        exit_code=75,
    )

    assert document["payload"]["operation"] == name
    assert document["payload"]["completed_materials"] == ["hopg"]
    assert document["payload"]["failed_materials"] == ["hbn"]
    assert document["payload"]["resumable"] is True


@pytest.mark.parametrize(
    ("command", "low_level"),
    [
        (recompute_cli.brem_command, "repair_checkpoint"),
        (recompute_cli.line_command, "reline_checkpoint"),
    ],
)
def test_recompute_json_marks_low_level_exception_failed(monkeypatch, command, low_level):
    from pyrite.checkpoints import recompute

    def fail(*_args, **_kwargs):
        raise OSError("checkpoint failed")

    monkeypatch.setattr(recompute, low_level, fail)

    document = _document(
        invoke(command, ["hopg", "--profile", "standard", "-o", "json"]), exit_code=1
    )

    assert document["payload"]["completed_materials"] == []
    assert document["payload"]["failed_materials"] == ["hopg"]
    assert document["errors"][0]["message"] == "checkpoint failed"


def test_remote_pull_json_retains_success_when_one_item_fails(monkeypatch):
    def pull(_materials, *, summary, **_kwargs):
        print("human progress")
        summary.update(
            completed=["hopg"],
            failed=["hbn"],
            errors={"hbn": "scp failed"},
        )
        raise SystemExit("partial pull")

    monkeypatch.setattr(lifecycle, "pull", pull)

    document = _document(
        invoke(remote_cli.command, ["pull", "hopg", "hbn", "-o", "json"]),
        exit_code=1,
    )

    assert document["payload"]["operation"] == "remote-pull"
    assert document["payload"]["completed_materials"] == ["hopg"]
    assert document["payload"]["failed_materials"] == ["hbn"]
    assert document["errors"][0]["message"] == "scp failed"
