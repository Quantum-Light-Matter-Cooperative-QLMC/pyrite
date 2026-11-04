"""Missing BremsLib data refuses work unless the run explicitly permits a surrogate."""

import json

import pytest
from click.testing import CliRunner

from pyrite._physics_fallback import PhysicsFallbackWarning
from pyrite.cli.commands import scan as scan_cli
from pyrite.runs import scan
from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.bremslib import tables


@pytest.fixture
def missing_bremslib(monkeypatch):
    def missing(*_args, **_kwargs):
        raise TableNotFoundError("test catalog has no BremsLib tables")

    monkeypatch.setattr(tables, "resolve_element_table", missing)


@pytest.mark.parametrize("json_output", [False, True])
def test_missing_tables_refuse_before_cases_or_checkpoint(
    monkeypatch, tmp_path, missing_bremslib, json_output
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("missing tables must stop the run before cases or compute")

    scan._load_runtime()
    monkeypatch.setattr(scan, "build_cases", forbidden)
    monkeypatch.setattr(scan, "run_sweep", forbidden)
    checkpoint = tmp_path / "checkpoints"
    result = CliRunner().invoke(
        scan_cli.command,
        [
            "standard",
            "-m",
            "hopg",
            "--no-progress",
            "--checkpoint-dir",
            str(checkpoint),
            *(["--output", "json"] if json_output else []),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "pyrite tables fetch bremslib" in result.output
    assert "Traceback" not in result.output
    assert not checkpoint.exists()
    if json_output:
        assert result.stderr == ""
        json.loads(result.stdout)
    else:
        assert result.stdout == ""
        assert result.stderr.startswith("Error:")


@pytest.mark.slow
def test_opt_in_records_source_and_transport_fallbacks_in_campaign_lock(
    monkeypatch, tmp_path, missing_bremslib
):
    scan._load_runtime()
    seen = []

    def run_sweep(cases, _results, **kwargs):
        # Exercise real preflight/case construction and completion persistence;
        # leave the expensive Monte Carlo calculation outside this CLI test.
        assert cases
        assert all(case.get("bremsstrahlung_model", "eedl") == "eedl" for case in cases)
        assert all("radiative_model" not in case for case in cases)
        seen.append(kwargs["dataset_identity"])
        return True

    monkeypatch.setattr(scan, "run_sweep", run_sweep)
    checkpoint = tmp_path / "checkpoints"
    with pytest.warns(PhysicsFallbackWarning):
        result = CliRunner().invoke(
            scan_cli.command,
            [
                "standard",
                "-m",
                "hopg",
                "--no-progress",
                "--checkpoint-dir",
                str(checkpoint),
                "--allow-physics-fallback",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert len(seen) == 1
    locks = list(checkpoint.rglob("cxr.lock.json"))
    assert len(locks) == 1
    identity = json.loads(locks[0].read_text())["dataset_identity"]
    assert identity["physics_fallbacks"] == seen[0]["physics_fallbacks"]
    events = identity["physics_fallbacks"]
    assert [event["model_used"] for event in events] == ["eedl", "uncoupled"]
    assert all(event["elements"] == ["C"] for event in events)
    assert all(event["energy_span_keV"] is None for event in events)
    assert all(event["fraction_segments_affected"] is None for event in events)
