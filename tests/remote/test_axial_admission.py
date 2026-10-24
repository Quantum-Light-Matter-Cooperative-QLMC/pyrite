"""Axial profiles refuse locally before remote probes, staging, or submission."""

import json
from argparse import Namespace
from dataclasses import replace

import pytest
from click.testing import CliRunner

from pyrite.campaign import profiles
from pyrite.cli.commands import remote as remote_cli
from pyrite.cli.commands import scan as scan_cli
from pyrite.remote import config, queue, transport
from pyrite.runs import scan


@pytest.fixture
def axial_profile(monkeypatch):
    scan._load_runtime()
    resolve = profiles.profile_run_settings

    def axial_settings(*args, **kwargs):
        return replace(
            resolve(*args, **kwargs),
            channeling_model="axial",
            channeling_axial={
                "axis_uvw": [0, 0, 1],
                "zone_max_ang": 1000.0,
                "entry": {"version": 1, "kind": "gaussian", "waist_ang": 0.5},
            },
            radiative_model="uncoupled",
            precision=None,
        )

    monkeypatch.setattr(profiles, "profile_run_settings", axial_settings)
    monkeypatch.setattr(
        scan, "material_sweep", lambda *_a, **_kw: pytest.fail("material sweep resolved")
    )


def test_scan_refuses_axial_before_material_resolution(axial_profile):
    with pytest.raises(ValueError, match="#623"):
        scan._resolved_run(Namespace(fidelity="full", catalog_profile="standard"), "hopg")


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"quick": True},
        {"cpu_only": True, "performance_profile": "standard", "chunk_minutes": 0},
    ],
    ids=["full", "quick", "cpu-profile"],
)
def test_axial_submission_refuses_before_any_remote_effect(
    axial_profile, monkeypatch, dry_run, options
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("remote operation started")

    monkeypatch.setattr(queue, "_refuse_if_busy", forbidden)
    monkeypatch.setattr(queue, "_refuse_if_profile_live", forbidden)
    monkeypatch.setattr(transport, "sync_code", forbidden)
    monkeypatch.setattr(transport, "_ssh_capture", forbidden)
    monkeypatch.setattr(queue, "_stage_job_script", forbidden)
    monkeypatch.setattr(queue, "_submit_staged_job", forbidden)
    monkeypatch.setattr(config, "remote_host", forbidden)
    with pytest.raises(ValueError, match="#623"):
        queue.start_queue(["hopg"], dry_run=dry_run, **options)


@pytest.mark.parametrize("output", ["table", "json"])
def test_local_cli_axial_refusal_is_clean_and_writes_no_checkpoint(axial_profile, tmp_path, output):
    checkpoint_dir = tmp_path / "checkpoints"
    result = CliRunner().invoke(
        scan_cli.command,
        ["standard", "-m", "hopg", "--checkpoint-dir", str(checkpoint_dir), "--output", output],
        catch_exceptions=False,
    )
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    if output == "json":
        payload = json.loads(result.stdout)
        assert "#623" in json.dumps(payload)
        assert result.stderr == ""
    else:
        assert "#623" in result.stderr
        assert result.stdout == ""
    assert not checkpoint_dir.exists()


def test_remote_cli_axial_refusal_is_a_runtime_diagnostic(axial_profile, monkeypatch):
    monkeypatch.setattr(
        queue, "_refuse_if_busy", lambda *_a: pytest.fail("remote busy probe started")
    )
    result = CliRunner().invoke(
        remote_cli.start_command,
        ["standard", "-m", "hopg", "--headless"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "#623" in result.stderr
    assert "Traceback" not in result.output
