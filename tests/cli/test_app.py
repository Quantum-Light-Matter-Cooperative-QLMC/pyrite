"""Public dispatch contract for the grouped interactive applications."""

import importlib

from click.testing import CliRunner

from pyrite import cli
from pyrite.cli.commands import app_analysis as analyze
from pyrite.cli.commands import app_validation as check
from pyrite.cli.commands import app_viewer as viewer
from pyrite.cli.commands import app_views as views


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
    assert "pixels" in result.output
    assert "compare" in result.output
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


def test_viewer_export_defaults_to_pyrite_stem(monkeypatch):
    launched = []
    monkeypatch.setattr(
        viewer.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    viewer._export(None, "mose2")

    assert launched[0][0][0] == viewer._smoke_command("mose2", "results/pyrite_viewer_mose2.html")


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


def test_pixels_and_compare_leaves_launch_their_own_notebooks(monkeypatch):
    launched = []
    monkeypatch.setattr(
        views, "_launch", lambda app, material, **kwargs: launched.append((app, material, kwargs))
    )
    monkeypatch.setattr(views, "get_analysis_default", lambda: "wse2")

    runner = CliRunner()
    pixels = runner.invoke(cli.command, ["app", "pixels", "launch", "mose2", "--smoke"])
    compare = runner.invoke(cli.command, ["app", "compare", "launch", "--tunnel"])

    assert pixels.exit_code == 0, pixels.output
    assert compare.exit_code == 0, compare.output
    assert launched[0][0] is views.PIXELS and launched[0][1] == "mose2"
    assert launched[0][2]["smoke"] is True
    # No MATERIAL: the analysis app's persisted default seeds the picker.
    assert launched[1][0] is views.COMPARE and launched[1][1] == "wse2"
    assert launched[1][2]["tunnel"] is True


def test_pixels_and_compare_never_persist_a_default():
    for leaf in ("pixels", "compare"):
        result = CliRunner().invoke(cli.command, ["app", leaf, "launch", "-d", "hopg"])
        assert result.exit_code == 2
        assert "No such option" in result.output


def test_split_app_material_falls_back_to_hopg(monkeypatch):
    monkeypatch.setattr(views, "get_analysis_default", lambda: None)

    assert views.resolve_material(None) == "hopg"
    assert views.resolve_material("mose2") == "mose2"


def test_split_app_commands_target_their_notebooks_and_ports():
    assert views.PIXELS.notebook.endswith("pixel_app.py")
    assert views.COMPARE.notebook.endswith("compare_app.py")
    ports = {analyze.TUNNEL_PORT, viewer.TUNNEL_PORT, views.PIXELS.tunnel_port}
    assert len(ports | {views.COMPARE.tunnel_port}) == 4

    run = views._command(views.PIXELS, "hopg", tunnel=True, no_token=True)
    assert run[3:8] == ["run", "--port", "2720", "--no-token", views.PIXELS.notebook]
    assert run[-2:] == ["--material", "hopg"]
    smoke = views._smoke_command(views.COMPARE, "hopg", "out.html")
    assert smoke[3:7] == ["export", "html", views.COMPARE.notebook, "--output"]
    assert smoke[-2:] == ["--material", "hopg"]


def test_split_app_export_never_launches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        views.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(views, "get_analysis_default", lambda: None)

    named = CliRunner().invoke(cli.command, ["app", "pixels", "export", "mose2", "--stem", "px"])
    default = CliRunner().invoke(cli.command, ["app", "compare", "export"])

    assert named.exit_code == 0 and default.exit_code == 0
    assert calls[0][0][0] == views._smoke_command(views.PIXELS, "mose2", "results/px.html")
    assert calls[1][0][0] == views._smoke_command(
        views.COMPARE, "hopg", "results/pyrite_compare_hopg.html"
    )
    assert calls[1][1]["env"]["PYRITE_ANALYZE_INITIAL"] == "hopg"
