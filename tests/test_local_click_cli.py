from __future__ import annotations

import subprocess
import sys

import pytest

from cxr_mc import (
    analyze,
    archive,
    blaze,
    check,
    check_config,
    export,
    rebrem,
    reline,
    scan,
    slim,
    viewer,
)
from tests.cli_helpers import assert_clean_result, invoke

LOCAL_COMMANDS = [
    scan.command,
    blaze.command,
    analyze.command,
    viewer.command,
    check.command,
    export.command,
    slim.command,
    rebrem.command,
    reline.command,
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


def test_scan_click_dispatches_defaults_and_zero_workers(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(
        scan.command,
        ["hopg", "--workers", "0", "--beam-uvw", "1", "0", "-1"],
    )
    assert_clean_result(result)
    assert seen == {
        "material": "hopg",
        "all": False,
        "actually_all": False,
        "include_unverified_dw": False,
        "include_high_energy": False,
        "high_energy_min_kev": None,
        "workers": 0,
        "fidelity": "full",
        "catalog_profile": "standard",
        "quick": False,
        "n_families": None,
        "beam_uvw": (1, 0, -1),
        "checkpoint_dir": "checkpoints",
        "max_minutes": None,
        "progress_file": None,
        "no_progress": False,
    }


def test_scan_fidelity_dispatch_and_quick_conflict(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(scan.command, ["hopg", "--fidelity", "survey"])
    assert_clean_result(result)
    assert seen["fidelity"] == "survey"

    conflict = invoke(scan.command, ["hopg", "--fidelity", "survey", "--quick"])
    assert conflict.exit_code == 2
    assert "cannot be combined" in conflict.stderr


def test_blaze_preserves_one_flag_many_values_syntax(monkeypatch):
    seen = _capture(monkeypatch, blaze, "run")
    result = invoke(
        blaze.command,
        [
            "hopg",
            "--energy",
            "30",
            "40",
            "--spacing",
            "1e-6",
            "--angles",
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
    result = invoke(blaze.command, ["hopg", "--energy", "30", "-h"])

    assert_clean_result(result)
    assert result.stdout.startswith("Usage:")


def test_blaze_unknown_option_after_variadic_value_is_not_swallowed():
    result = invoke(blaze.command, ["hopg", "--energy", "30", "--unknown"])

    assert result.exit_code == 2
    assert "No such option '--unknown'" in result.stderr


@pytest.mark.parametrize(
    ("module", "command", "handler", "argv", "expected"),
    [
        (
            analyze,
            analyze.command,
            "_cli",
            ["hopg", "--default", "--smoke"],
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
            slim.command,
            "_cli",
            ["in.pkl", "--line-only"],
            {"checkpoint": "in.pkl", "line_only": True, "compresslevel": 6},
        ),
        (
            rebrem,
            rebrem.command,
            "_cli",
            ["hopg", "w", "--ne-brem", "1000"],
            {"material": ["hopg", "w"], "all": False, "ne_brem": 1000},
        ),
        (
            reline,
            reline.command,
            "_cli",
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
        (scan.command, ["hopg", "--workers", "-1"]),
        (scan.command, ["hopg", "--beam-uvw", "0", "0", "0"]),
        (blaze.command, ["hopg", "--energy", "0", "--spacing", "1e-6"]),
        (blaze.command, ["hopg", "--energy", "30", "--spacing", "1e-6", "--angles", "90"]),
        (check.command, ["--ne", "0"]),
        (rebrem.command, ["hopg", "--save-every", "0"]),
        (reline.command, ["hopg", "--line-step", "nan"]),
    ],
)
def test_local_click_numeric_domains_are_usage_errors(command, argv):
    result = invoke(command, argv)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value" in result.stderr


def test_slim_dataset_modes_are_mutually_exclusive():
    result = invoke(slim.command, ["in.pkl", "--brem-only", "--line-only"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "mutually exclusive" in result.stderr


@pytest.mark.parametrize(
    ("command", "argv", "message"),
    [
        (scan.command, [], "needs a material"),
        (scan.command, ["hopg", "--all"], "--all does not take"),
        (scan.command, ["hopg", "-A"], "-A/--actually-all does not take"),
        (scan.command, ["--all", "-A"], "already includes --all"),
        (
            scan.command,
            ["-A", "--include-unverified-dw"],
            "already includes --include-unverified-dw",
        ),
        (scan.command, ["-A", "--include-high-energy"], "already includes --include-high-energy"),
        (scan.command, ["--include-unverified-dw"], "--include-unverified-dw requires --all"),
        (scan.command, ["--include-high-energy"], "--include-high-energy requires --all"),
        (rebrem.command, [], "needs material"),
        (reline.command, ["hopg", "--all"], "--all does not take"),
        (analyze.command, ["--default"], "--default requires MATERIAL"),
        (scan.command, ["not-a-material"], "not a configured material"),
    ],
)
def test_local_selection_errors_fail_at_click_boundary(command, argv, message):
    result = invoke(command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_scan_all_include_flags_dispatch(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(
        scan.command,
        [
            "--all",
            "--include-unverified-dw",
            "--include-high-energy",
            "--high-energy-min-kev",
            "200",
        ],
    )
    assert_clean_result(result)
    assert seen["all"] is True
    assert seen["include_unverified_dw"] is True
    assert seen["include_high_energy"] is True
    assert seen["high_energy_min_kev"] == 200.0


def test_scan_actually_all_dispatch(monkeypatch):
    seen = _capture(monkeypatch, scan, "run")
    result = invoke(scan.command, ["-A"])
    assert_clean_result(result)
    assert seen["actually_all"] is True
    assert seen["all"] is False


@pytest.mark.parametrize(
    ("argv", "expected_len"),
    [
        (["--all"], 21),
        (["--all", "--include-unverified-dw"], 30),
        (["--all", "--include-high-energy"], 25),
        (["--all", "--include-unverified-dw", "--include-high-energy"], 34),
        (["-A"], 46),
    ],
)
def test_scan_selection_flags_resolve_expected_material_counts(monkeypatch, argv, expected_len):
    seen = {}

    def fake_run(args):
        seen["materials"] = scan._selected(args)

    monkeypatch.setattr(scan, "run", fake_run)
    result = invoke(scan.command, argv)
    assert_clean_result(result)
    assert len(seen["materials"]) == expected_len
    assert len(seen["materials"]) == len(set(seen["materials"]))


def test_scan_include_high_energy_floors_energy_grid(monkeypatch):
    seen = {}

    def fake_run(args):
        scan._selected(args)
        seen["floor_map"] = args.high_energy_floor_map

    monkeypatch.setattr(scan, "run", fake_run)
    result = invoke(scan.command, ["--all", "--include-high-energy"])
    assert_clean_result(result)
    assert set(seen["floor_map"]) == {"tise2", "gep", "ges", "rese2"}
    assert all(value == 150.0 for value in seen["floor_map"].values())


def test_scan_direct_material_ignores_floor_unless_high_energy_tagged(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["materials"] = scan._selected(args)
        seen["floor_map"] = dict(args.high_energy_floor_map)

    monkeypatch.setattr(scan, "run", fake_run)

    tagged = invoke(scan.command, ["tise2", "--high-energy-min-kev", "200"])
    assert_clean_result(tagged)
    assert seen["floor_map"] == {"tise2": 200.0}

    untagged = invoke(scan.command, ["hopg", "--high-energy-min-kev", "200"])
    assert_clean_result(untagged)
    assert seen["floor_map"] == {}


class _FakeCatalog:
    """Duck-typed stand-in for ``MaterialCatalog``'s profile surface --
    ``validate_catalog_profile`` only ever touches ``profile_names`` and
    ``profile_materials``."""

    def __init__(self, profile_names, memberships):
        self.profile_names = profile_names
        self._memberships = memberships

    def profile_materials(self, name):
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}")
        return self._memberships.get(name)


def test_scan_unknown_profile_is_usage_error():
    result = invoke(scan.command, ["hopg", "--profile", "bogus"])
    assert result.exit_code == 2
    assert "unknown profile 'bogus'" in result.stderr
    assert "standard" in result.stderr


def test_scan_all_profile_membership_silently_intersects(monkeypatch):
    import cxr_mc.materials as materials_pkg

    monkeypatch.setattr(
        materials_pkg, "CATALOG", _FakeCatalog(("standard", "narrowed"), {"narrowed": ("hopg",)})
    )
    seen = {}

    def fake_run(args):
        seen["materials"] = scan._selected(args)

    monkeypatch.setattr(scan, "run", fake_run)
    result = invoke(scan.command, ["--all", "--profile", "narrowed"])
    assert_clean_result(result)
    assert seen["materials"] == ["hopg"]


def test_scan_explicit_material_outside_profile_is_usage_error(monkeypatch):
    import cxr_mc.materials as materials_pkg

    monkeypatch.setattr(
        materials_pkg, "CATALOG", _FakeCatalog(("standard", "narrowed"), {"narrowed": ("hopg",)})
    )
    result = invoke(scan.command, ["mos2", "--profile", "narrowed"])
    assert result.exit_code == 2
    assert "does not include" in result.stderr
    assert "hopg" in result.stderr


def test_resolved_run_threads_catalog_profile_into_material_sweep(monkeypatch):
    """Regression: `_resolved_run` used to compute `catalog_profile` only after
    building `sweep`, so `material_sweep()` always resolved the standard
    profile's grid regardless of `--profile` -- the checkpoint identity/stem
    were correctly tagged `sub_100keV` while the simulated parameters (energy
    grid, thickness, etc.) silently stayed standard's. Caught live: `cxr
    remote submit --all --profile sub_100keV` still ran 250 keV cases."""
    import types

    from cxr_mc.config import material_sweep as real_material_sweep

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
        [sys.executable, "-m", "cxr_mc.scan", "hopg", "--workers", "-1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Invalid value for '--workers'" in completed.stderr
    assert "Traceback" not in completed.stderr
