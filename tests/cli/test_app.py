"""Public dispatch contract for the grouped interactive applications."""

import importlib

from click.testing import CliRunner

from pyrite import cli
from pyrite.cli.commands import app_analysis as analyze
from pyrite.cli.commands import app_validation as check
from pyrite.cli.commands import app_viewer as viewer


def test_app_help_imports_only_the_group(monkeypatch):
    imported: list[str] = []
    original = importlib.import_module

    def track(name):
        imported.append(name)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", track)
    result = CliRunner().invoke(cli.command, ["app", "--help"])

    assert result.exit_code == 0
    assert "analysis" in result.output
    assert "viewer" in result.output
    assert "validation" in result.output
    assert imported == ["pyrite.cli.commands.app"]


def test_analysis_leaf_launches_and_export_dispatches(monkeypatch):
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kwargs: (
            launched.setdefault("material", material),
            launched.update(**kwargs),
        ),
    )
    monkeypatch.setattr(
        "pyrite.cli.commands.export._export", lambda stem: launched.update(export=stem)
    )

    runner = CliRunner()
    launch = runner.invoke(cli.command, ["app", "analysis", "launch", "mose2"])
    smoke = runner.invoke(cli.command, ["app", "analysis", "launch", "--smoke"])
    exported = runner.invoke(cli.command, ["app", "analysis", "export", "report"])

    assert launch.exit_code == 0
    assert launched["material"] == "mose2"
    assert smoke.exit_code == 0
    assert launched["smoke"] is True
    assert exported.exit_code == 0
    assert launched["export"] == "report"


def test_viewer_export_never_launches(monkeypatch):
    launched = []
    monkeypatch.setattr(
        viewer.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    result = CliRunner().invoke(
        cli.command, ["app", "viewer", "export", "mose2", "--stem", "trace"]
    )

    assert result.exit_code == 0
    assert launched[0][0][0] == viewer._smoke_command("mose2", "results/trace.html")


def test_validation_leaf_and_export_dispatch_without_cross_mode_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(check, "_launch", lambda **kwargs: calls.append(("launch", kwargs)))
    monkeypatch.setattr(
        check, "_export", lambda *args, **kwargs: calls.append(("export", args, kwargs))
    )

    runner = CliRunner()
    launch = runner.invoke(cli.command, ["app", "validation", "launch", "--watch"])
    exported = runner.invoke(
        cli.command, ["app", "validation", "export", "--outdir", "out", "--ne", "11"]
    )

    assert launch.exit_code == 0
    assert calls[0] == ("launch", {"edit": False, "watch": True})
    assert exported.exit_code == 0
    assert calls[1] == ("export", ("out",), {"ne": 11, "ne_brem": 200, "ne_supp": 200})


def test_implicit_app_launch_retired_in_favour_of_the_explicit_leaf(monkeypatch):
    """The bare group launched the app through 0.2.x; 0.3.0 requires `launch`."""
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kwargs: launched.update(material=material, **kwargs),
    )

    implicit = CliRunner().invoke(cli.command, ["app", "analysis", "mose2"])
    explicit = CliRunner().invoke(cli.command, ["app", "analysis", "launch", "mose2"])

    assert implicit.exit_code == 2
    assert explicit.exit_code == 0
    assert launched["material"] == "mose2"
