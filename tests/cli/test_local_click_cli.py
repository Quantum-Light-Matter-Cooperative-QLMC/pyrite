import subprocess
import sys

import pytest

from pyrite.checkpoints import _checkpoint_io, slim
from pyrite.cli.commands import app_analysis as analyze
from pyrite.cli.commands import app_validation as check
from pyrite.cli.commands import app_viewer as viewer
from pyrite.cli.commands import archive, check_config, export
from pyrite.cli.commands import blaze as blaze_cli
from pyrite.cli.commands import recompute as recompute_cli
from pyrite.cli.commands import scan as scan_cli
from pyrite.cli.commands import slim as slim_cli
from pyrite.cli.commands.scan import performance_command
from pyrite.runs import blaze, scan
from tests.helpers.cli import assert_clean_result, invoke

LOCAL_COMMANDS = [
    scan_cli.command,
    performance_command,
    blaze_cli.command,
    analyze.command,
    viewer.command,
    check.command,
    export.command,
    slim_cli.command,
    recompute_cli.brem_command,
    recompute_cli.line_command,
    archive.archive_command,
    archive.restore_command,
    archive.archives_command,
    archive.union_command,
    check_config.command,
]


@pytest.mark.parametrize("command", LOCAL_COMMANDS, ids=lambda command: command.name)
def test_every_local_click_help_path_is_clean(command):
    result = invoke(command, ["--help"])
    assert_clean_result(result)
    assert result.stdout.startswith("Usage:")
    assert result.stdout.endswith("\n")


def _capture(monkeypatch, module, handler_name):
    seen = {}

    def handler(args):
        seen.update(vars(args))

    monkeypatch.setattr(module, handler_name, handler)
    return seen


def test_run_click_dispatches_profile_material_and_zero_workers(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(
        scan_cli.command,
        ["standard", "-m", "hopg", "--workers", "0"],
    )
    assert_clean_result(result)
    assert seen == {
        "material": "hopg",
        "all": False,
        "actually_all": False,
        "workers": 0,
        "fidelity": "full",
        "catalog_profile": "standard",
        "quick": False,
        "n_families": None,
        "beam_uvw": None,
        "checkpoint_dir": "checkpoints",
        "max_minutes": None,
        "performance_profile": None,
        "performance_interval": 5.0,
        "performance_dir": None,
        "cache_read": True,
        "cache_write": True,
        "progress_file": None,
        "no_progress": False,
        "verbose": 0,
    }


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--no-cache"], (False, False)),
        (["--recompute"], (False, True)),
    ],
)
def test_run_cache_flag_precedence(monkeypatch, extra, expected):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(scan_cli.command, ["standard", "-m", "hopg", *extra])
    assert_clean_result(result)
    assert (seen["cache_read"], seen["cache_write"]) == expected


def test_run_cache_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(scan, "run", lambda _args: None)
    result = invoke(
        scan_cli.command,
        ["standard", "-m", "hopg", "--no-cache", "--recompute"],
    )
    assert result.exit_code == 2
    assert "--no-cache and --recompute are mutually exclusive" in result.stderr


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], (False, False)),
        (["--no-cache"], (False, False)),
        (["--recompute"], (False, True)),
    ],
)
def test_perf_cache_flag_precedence(monkeypatch, extra, expected):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(performance_command, ["standard", "-m", "hopg", *extra])
    assert_clean_result(result)
    assert (seen["cache_read"], seen["cache_write"]) == expected


def test_run_fidelity_dispatch_and_quick_conflict(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(scan_cli.command, ["standard", "-m", "hopg", "--fidelity", "survey"])
    assert_clean_result(result)
    assert seen["fidelity"] == "survey"

    conflict = invoke(
        scan_cli.command, ["standard", "-m", "hopg", "--fidelity", "survey", "--quick"]
    )
    assert conflict.exit_code == 2
    assert "cannot be combined" in conflict.stderr


def test_blaze_preserves_one_flag_many_values_syntax(monkeypatch):
    seen = _capture(monkeypatch, blaze, "run")
    result = invoke(
        blaze_cli.command,
        [
            "hopg",
            "--energy",
            "30",
            "40",
            "--spacing",
            "1e-6",
            "--polar",
            "10",
            "20",
            "--workers",
            "0",
        ],
    )
    assert_clean_result(result)
    assert seen["energy"] == [30.0, 40.0]
    assert seen["spacing"] == [1e-6]
    assert seen["angles"] == [10.0, 20.0]
    assert seen["workers"] == 0


def test_blaze_short_help_after_variadic_value_remains_eager():
    result = invoke(blaze_cli.command, ["hopg", "--energy", "30", "-h"])

    assert_clean_result(result)
    assert result.stdout.startswith("Usage:")


def test_blaze_unknown_option_after_variadic_value_is_not_swallowed():
    result = invoke(blaze_cli.command, ["hopg", "--energy", "30", "--unknown"])

    assert result.exit_code == 2
    assert "No such option '--unknown'" in result.stderr


@pytest.mark.parametrize(
    ("module", "command", "handler", "argv", "expected"),
    [
        (
            analyze,
            analyze.command,
            "_cli",
            ["hopg", "--save-default", "--smoke"],
            {"material": "hopg", "default": True, "smoke": True},
        ),
        (
            check,
            check.command,
            "_cli",
            ["--export", "--ne", "11"],
            {"export": True, "ne": 11, "outdir": "figures"},
        ),
        (
            slim,
            slim_cli.command,
            "_cli",
            ["in.pkl", "--line-only"],
            {
                "checkpoint": "in.pkl",
                "line_only": True,
                "compresslevel": _checkpoint_io.DEFAULT_LEVEL,
            },
        ),
        (
            recompute_cli,
            recompute_cli.brem_command,
            "_brem_cli",
            ["hopg", "w", "--ne-brem", "1000"],
            {"material": ["hopg", "w"], "all": False, "ne_brem": 1000},
        ),
        (
            recompute_cli,
            recompute_cli.line_command,
            "_line_cli",
            ["--all", "--line-step", "5"],
            {"material": [], "all": True, "line_step": 5.0},
        ),
        (
            check_config,
            check_config.command,
            "_run",
            ["catalog.toml"],
            {"manifest": "catalog.toml"},
        ),
    ],
)
def test_local_click_dispatches_legacy_handler(
    monkeypatch, module, command, handler, argv, expected
):
    seen = _capture(monkeypatch, module, handler)
    result = invoke(command, argv)
    assert_clean_result(result)
    for key, value in expected.items():
        actual = seen[key]
        if key == "manifest":
            actual = str(actual)
        assert actual == value


def test_export_click_dispatches_default_stem(monkeypatch):
    seen = []
    monkeypatch.setattr(export, "_export", seen.append)
    assert_clean_result(invoke(export.command))
    assert seen == [None]


@pytest.mark.parametrize(
    ("command", "handler_name", "argv", "expected"),
    [
        (
            archive.archive_command,
            "_cli_archive",
            ["hopg", "keeper", "--force"],
            {"stem": "hopg", "label": "keeper", "force": True},
        ),
        (
            archive.restore_command,
            "_cli_restore",
            ["keeper", "--as", "hopg"],
            {"label": "keeper", "stem": "hopg", "force": False},
        ),
        (archive.archives_command, "_cli_archives", [], {}),
        (
            archive.union_command,
            "_cli_union",
            ["hopg", "keeper", "--no-archive", "--delete-archive"],
            {
                "stem": "hopg",
                "label": "keeper",
                "no_archive": True,
                "delete_archive": True,
                "force": False,
            },
        ),
    ],
)
def test_archive_click_dispatch(command, handler_name, argv, expected, monkeypatch):
    seen = _capture(monkeypatch, archive, handler_name)
    assert_clean_result(invoke(command, argv))
    assert seen == expected


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        (scan_cli.command, ["standard", "-m", "hopg", "--workers", "-1"]),
        (blaze_cli.command, ["hopg", "--energy", "0", "--spacing", "1e-6"]),
        (blaze_cli.command, ["hopg", "--energy", "30", "--spacing", "1e-6", "--polar", "90"]),
        (check.command, ["--ne", "0"]),
        (recompute_cli.brem_command, ["hopg", "--save-every", "0"]),
        (recompute_cli.line_command, ["hopg", "--line-step", "nan"]),
    ],
)
def test_local_click_numeric_domains_are_usage_errors(command, argv):
    result = invoke(command, argv)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr


def test_slim_dataset_modes_are_mutually_exclusive():
    result = invoke(slim_cli.command, ["in.pkl", "--brem-only", "--line-only"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "mutually exclusive" in result.stderr


@pytest.mark.parametrize(
    ("command", "argv", "message"),
    [
        (recompute_cli.brem_command, [], "needs material"),
        (recompute_cli.line_command, ["hopg", "--all"], "--all does not take"),
        (analyze.command, ["--save-default"], "--save-default requires MATERIAL"),
        (scan_cli.command, ["standard", "-m", "not-a-material"], "not a configured material"),
    ],
)
def test_local_selection_errors_fail_at_click_boundary(command, argv, message):
    result = invoke(command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize(
    "flag",
    [
        "--all",
        "--actually-all",
        "--include-unverified-dw",
        "--include-high-energy",
        "--high-energy-min-kev",
        "--beam-uvw",
        "--beam-transverse-fwhm-x-mm",
        "--beam-transverse-fwhm-y-mm",
        "--beam-bunch-length-fs",
        "--beam-long-shape",
        "--beam-rep-rate-hz",
        "--beam-bunch-charge-pc",
    ],
)
def test_run_rejects_removed_selection_and_beam_flags(flag):
    result = invoke(scan_cli.command, ["standard", flag])
    assert result.exit_code == 2
    assert f"No such option '{flag}'" in result.stderr


class _FakeCatalog:
    """Duck-typed stand-in for ``MaterialCatalog``'s profile surface --
    ``validate_catalog_profile`` only ever touches ``profile_names`` and
    ``profile_materials``."""

    def __init__(self, profile_names, memberships, material_keys=("hopg", "mos2")):
        self.profile_names = profile_names
        self._memberships = memberships
        self.material_keys = material_keys

    def profile_materials(self, name):
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}")
        return self._memberships.get(name)


def test_run_unknown_profile_is_usage_error():
    result = invoke(scan_cli.command, ["bogus"])
    assert result.exit_code == 2
    assert "unknown profile 'bogus'" in result.stderr
    assert "standard" in result.stderr


def test_run_profile_membership_is_default_selection(monkeypatch):
    import pyrite.materials as materials_pkg

    monkeypatch.setattr(
        materials_pkg, "CATALOG", _FakeCatalog(("standard", "narrowed"), {"narrowed": ("hopg",)})
    )
    seen = {}

    def fake_run(args):
        seen["materials"] = scan._selected(args)

    monkeypatch.setattr(scan, "run", fake_run)
    result = invoke(scan_cli.command, ["narrowed"])
    assert_clean_result(result)
    assert seen["materials"] == ["hopg"]


def test_run_explicit_material_outside_profile_is_usage_error(monkeypatch):
    import pyrite.materials as materials_pkg

    monkeypatch.setattr(
        materials_pkg, "CATALOG", _FakeCatalog(("standard", "narrowed"), {"narrowed": ("hopg",)})
    )
    result = invoke(scan_cli.command, ["narrowed", "-m", "mos2"])
    assert result.exit_code == 2
    assert "does not include" in result.stderr
    assert "hopg" in result.stderr


def test_resolve_profile_materials_uses_profile_or_catalog_order(monkeypatch):
    import pyrite.materials as materials_pkg

    monkeypatch.setattr(
        materials_pkg,
        "CATALOG",
        _FakeCatalog(
            ("standard", "narrowed"),
            {"narrowed": ("hopg",)},
            material_keys=("mos2", "hopg"),
        ),
    )
    assert scan.resolve_profile_materials("standard") == ["mos2", "hopg"]
    assert scan.resolve_profile_materials("narrowed") == ["hopg"]
    assert scan.resolve_profile_materials("narrowed", "hopg") == ["hopg"]


def test_resolved_run_threads_catalog_profile_into_material_sweep(monkeypatch):
    """Regression: `_resolved_run` used to compute `catalog_profile` only after
    building `sweep`, so `material_sweep()` always resolved the standard
    profile's grid regardless of `--profile` -- the checkpoint identity/stem
    were correctly tagged `sub_100keV` while the simulated parameters (energy
    grid, thickness, etc.) silently stayed standard's. Caught live: `pyrite
    remote profile runs still used standard-profile 250 keV cases."""
    import types

    from pyrite.campaign.config import material_sweep as real_material_sweep

    calls = []

    def fake_material_sweep(material, **kwargs):
        calls.append(kwargs)
        # Content doesn't matter here -- only that `_resolved_run` forwarded
        # `catalog_profile`; build a real (standard-profile) Sweep so the
        # rest of `_resolved_run` (dataset_identity/variant_stem) has valid
        # data to chew on.
        return real_material_sweep(material, catalog_profile="standard")

    monkeypatch.setattr(scan, "material_sweep", fake_material_sweep)
    args = types.SimpleNamespace(catalog_profile="sub_100keV")

    scan._resolved_run(args, "hopg")

    assert calls[0]["catalog_profile"] == "sub_100keV"


def test_standalone_click_usage_error_preserves_exit_and_streams():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrite._entry.scan",
            "standard",
            "-m",
            "hopg",
            "--workers",
            "-1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Invalid value for '--workers'" in completed.stderr
    assert "Traceback" not in completed.stderr
