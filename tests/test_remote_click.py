"""Focused Click-contract tests for ``cxr remote``."""

from __future__ import annotations

import click
import pytest

from cxr_mc import remote
from cxr_mc._remote import lifecycle, viewer
from tests.cli_helpers import assert_clean_result, invoke

REMOTE_COMMANDS = (
    "run",
    "rebrem",
    "reline",
    "jobs",
    "status",
    "logs",
    "stop",
    "reap",
    "pull",
    "profile",
    "performance",
    "clear",
    "prune",
    "prune-jobs",
    "sync",
    "validate",
    "check",
)


def test_remote_exports_click_group():
    assert isinstance(remote.command, click.Group)
    assert set(remote.command.commands) == set(REMOTE_COMMANDS)


@pytest.mark.parametrize("name", REMOTE_COMMANDS)
def test_every_remote_click_help_path_is_offline(name):
    result = invoke(remote.command, [name, "--help"])

    assert_clean_result(result)
    assert f"Usage: remote {name} " in result.stdout


def test_remote_performance_commands_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "list_remote_performance",
        lambda: calls.append(("list",)),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda profile: calls.append(("pull", profile)),
    )
    monkeypatch.setattr(
        lifecycle,
        "prune_remote_performance",
        lambda profiles, **kwargs: calls.append(("prune", profiles, kwargs)),
    )

    assert_clean_result(invoke(remote.command, ["performance", "list"]))
    assert_clean_result(invoke(remote.command, ["performance", "pull", "baseline"]))
    assert_clean_result(
        invoke(remote.command, ["performance", "prune", "baseline", "--yes"])
    )

    assert calls == [
        ("list",),
        ("pull", "baseline"),
        ("prune", ["baseline"], {"all_profiles": False, "yes": True}),
    ]


def test_legacy_remote_profile_pull_warns_once(monkeypatch):
    monkeypatch.setattr(lifecycle, "pull_performance_profile", lambda _profile: None)

    result = invoke(remote.command, ["profile", "pull", "baseline"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr.count("is deprecated") == 1
    assert "cxr remote performance pull" in result.stderr


def test_run_click_defaults_and_zero_meanings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(
        remote.command, ["run", "standard", "-m", "hopg", "--workers", "0", "--headless"]
    )

    assert_clean_result(result)
    assert calls == [
        (
            ["hopg"],
            {
                "quick": False,
                "fidelity": "full",
                "workers": 0,
                "parallel_materials": None,
                "chunk_minutes": 10.0,
                "no_sync": False,
                "dry_run": False,
                "high_energy_min_kev": None,
                "catalog_profile": "standard",
                "performance_profile": None,
                "performance_repetitions": 1,
                "performance_interval": 5.0,
                "spec_chunk": None,
                "brem_chunk": None,
                "nsys": False,
            },
        )
    ]


def test_run_perf_flags_and_level9_reach_workflow(monkeypatch):
    queued = []
    pulled = []
    performance_pulled = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: queued.append((materials, kwargs)) or "job",
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(remote.state, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(remote.state, "_completed_materials", lambda _jobid, materials: materials)
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda profile: performance_pulled.append(profile),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda stems, **kwargs: pulled.append((stems, kwargs)),
    )

    result = invoke(
        remote.command,
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--perf",
            "--perf-reps",
            "1",
            "--perf-interval",
            "2",
            "--chunk-minutes",
            "0",
            "--level9",
        ],
    )

    assert_clean_result(result)
    assert queued[0][1]["performance_repetitions"] == 1
    assert queued[0][1]["performance_interval"] == 2.0
    assert performance_pulled == ["standard"]
    assert pulled[0][1]["level9"] is True


def test_hidden_remote_aliases_remain_callable():
    assert_clean_result(invoke(remote.command, ["check", "--help"]))

    root_help = invoke(remote.command, ["--help"])
    command_lines = {
        line.split()[0]
        for line in root_help.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert {"run", "validate"}.issubset(command_lines)
    assert command_lines.isdisjoint({"scan", "submit", "start", "check"})


@pytest.mark.parametrize(
    "argv, option",
    [
        (["run", "standard", "-m", "hopg", "--workers", "-1"], "--workers"),
        (["run", "standard", "-m", "hopg", "--chunk-minutes", "-1"], "--chunk-minutes"),
        (["rebrem", "hopg", "--ne-brem", "0"], "--ne-brem"),
        (["rebrem", "hopg", "--step", "nan"], "--step"),
        (["reline", "hopg", "--line-ne", "0"], "--line-ne"),
        (["reline", "hopg", "--line-step", "inf"], "--line-step"),
        (["reap", "--min-age-minutes", "-0.1"], "--min-age-minutes"),
        (["check", "--ne", "0"], "--ne"),
        (["check", "--tmd-azimuth", "nan"], "--tmd-azimuth"),
        (["check", "--tmd-azimuth", "-inf"], "--tmd-azimuth"),
    ],
)
def test_remote_numeric_domains_fail_at_click_boundary(argv, option):
    result = invoke(remote.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert option in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "argv, message",
    [
        (["run", "standard", "-m", "hopg", "--quick", "--grid"], "drop --grid"),
        (["pull", "hopg", "--brem-only", "--line-only"], "mutually exclusive"),
        (["stop"], "needs material"),
        (["clear"], "needs material"),
        (["prune", "--all", "--profile", "sub_100keV"], "cannot be combined"),
        (["check", "--follow"], "requires --detached"),
        (["check", "--pull", "--detached"], "mutually exclusive"),
    ],
)
def test_remote_incompatible_click_inputs_are_usage_errors(argv, message):
    result = invoke(remote.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_remote_prune_defaults_to_standard_preview(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "prune_remote",
        lambda **kwargs: calls.append(kwargs),
    )

    result = invoke(remote.command, ["prune"])

    assert_clean_result(result)
    assert calls == [{"all_profiles": False, "catalog_profile": None, "yes": False}]


@pytest.mark.parametrize("command_name", ["run", "rebrem", "reline"])
def test_fidelity_dispatches_cleanly(monkeypatch, command_name):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(
        lifecycle,
        "start_rebrem_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(
        lifecycle,
        "start_reline_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)

    argv = [command_name, "hopg", "--fidelity", "survey", "--no-sync"]
    if command_name in ("rebrem", "reline"):
        argv.append("--dry-run")
    else:
        argv = [
            command_name,
            "standard",
            "-m",
            "hopg",
            "--fidelity",
            "survey",
            "--no-sync",
            "--headless",
        ]

    result = invoke(remote.command, argv)

    assert_clean_result(result)
    assert calls[0]["fidelity"] == "survey"


def test_run_profile_and_material_dispatch(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")
        material_keys = ("hopg", "mose2")

        def profile_materials(self, _name):
            return None  # no membership row -> every candidate stays

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append(kwargs) or "job",
    )

    result = invoke(
        remote.command,
        ["run", "sub_100keV", "-m", "hopg", "--headless"],
    )

    assert_clean_result(result)
    assert calls[0]["catalog_profile"] == "sub_100keV"


def test_run_profile_with_membership_defaults_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")
        material_keys = ("hopg", "mose2")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote.command, ["run", "sub_100keV", "--headless"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg", "mose2"]
    assert calls[0][1]["catalog_profile"] == "sub_100keV"


def test_run_profile_without_membership_uses_manifest_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg
    import cxr_mc.scan as scan

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")
        material_keys = ("mose2", "hopg")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["hopg"])
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote.command, ["run", "sub_100keV", "--headless"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg"]


def test_pull_hash_option_dispatches_and_requires_one_qualified_selector(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "hopg@sub_100keV", "--hash", "abc123"])
    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]
    assert calls[0][1]["hash_prefix"] == "abc123"

    ambiguous = invoke(remote.command, ["pull", "hopg", "wse2", "--hash", "abc123"])
    assert ambiguous.exit_code == 2
    assert "requires exactly one MATERIAL@PROFILE" in ambiguous.stderr


@pytest.mark.parametrize(
    ("flag", "dataset"),
    [("--brem-only", "brem"), ("--line-only", "line")],
)
def test_partial_pull_all_forwards_full_material_list(monkeypatch, tmp_path, flag, dataset):
    manifest = tmp_path / "materials.txt"
    manifest.write_text('materials = ["hopg", "hbn", "mos2"]\n')
    monkeypatch.setattr(remote.config, "MATS_FILE", manifest)
    seen = {}
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: seen.update(materials=materials, kwargs=kwargs),
    )

    result = invoke(remote.command, ["pull", "--all", flag])

    assert_clean_result(result)
    assert seen["materials"] == ["hopg", "hbn", "mos2"]
    assert seen["kwargs"]["dataset"] == dataset
    assert seen["kwargs"]["grid"] is False


@pytest.mark.parametrize("status", [1, 130])
def test_logs_click_propagates_follow_status(monkeypatch, status):
    monkeypatch.setattr(viewer, "tail_logs", lambda _jobid, _follow: status)

    result = invoke(remote.command, ["logs", "--follow"])

    assert_clean_result(result, exit_code=status)


def test_remote_click_preserves_resumable_exit(monkeypatch):
    monkeypatch.setattr(viewer, "list_jobs", lambda: (_ for _ in ()).throw(SystemExit(75)))

    result = invoke(remote.command, ["jobs"])

    assert_clean_result(result, exit_code=75)


def test_stop_previews_by_default_and_yes_executes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "stop_jobs",
        lambda materials, all_jobs, *, yes, profile: calls.append(
            (materials, all_jobs, yes, profile)
        ),
    )

    preview = invoke(remote.command, ["stop", "hopg"])
    confirmed = invoke(remote.command, ["stop", "hopg", "--yes"])

    assert_clean_result(preview)
    assert_clean_result(confirmed)
    assert calls == [(["hopg"], False, False, None), (["hopg"], False, True, None)]


def test_stop_profile_dispatches_and_rejects_combinations(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "stop_jobs",
        lambda materials, all_jobs, *, yes, profile: calls.append(
            (materials, all_jobs, yes, profile)
        ),
    )

    ok = invoke(remote.command, ["stop", "--profile", "sub_100keV", "--yes"])
    assert_clean_result(ok)
    assert calls == [([], False, True, "sub_100keV")]

    with_materials = invoke(remote.command, ["stop", "hopg", "--profile", "sub_100keV"])
    assert with_materials.exit_code == 2
    assert "--profile does not take material names or --all" in with_materials.stderr

    with_all = invoke(remote.command, ["stop", "--all", "--profile", "sub_100keV"])
    assert with_all.exit_code == 2

    bare = invoke(remote.command, ["stop"])
    assert bare.exit_code == 2
    assert "--profile" in bare.stderr


def test_clear_implicit_profile_uses_manifest_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg
    import cxr_mc.scan as scan

    class _FakeCatalog:
        profile_names = ("standard",)

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["hopg", "hbn"])
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "clear_remote",
        lambda materials, yes, **kwargs: calls.append((materials, yes, kwargs)),
    )

    result = invoke(remote.command, ["clear", "--profile", "standard"])

    assert_clean_result(result)
    assert calls == [(["hopg", "hbn"], False, {"catalog_profile": "standard"})]


def test_pull_profile_expands_membership_to_qualified_selectors(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV", "mose2@sub_100keV"]


def test_pull_positional_profile_expands_membership(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV", "mose2@sub_100keV"]


def test_pull_profile_qualifies_explicit_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "hopg", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]

    with_all = invoke(remote.command, ["pull", "--all", "--profile", "sub_100keV"])
    assert with_all.exit_code == 0
    assert "warning:" in with_all.stderr
    assert "ignoring --all" in with_all.stderr

    with_selector = invoke(remote.command, ["pull", "hopg@sub_100keV", "--profile", "sub_100keV"])
    assert with_selector.exit_code == 2
    assert "drop the @PROFILE selector" in with_selector.stderr


def test_pull_profile_without_membership_uses_manifest_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg
    import cxr_mc.scan as scan

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    monkeypatch.setattr(scan, "load_all_materials", lambda: ["mose2", "hopg"])
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["mose2@sub_100keV", "hopg@sub_100keV"]


def test_pull_material_option_narrows_positional_profile(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote.command, ["pull", "sub_100keV", "-m", "hopg"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]


def test_pull_material_option_rejects_material_outside_profile(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg",)

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())

    result = invoke(remote.command, ["pull", "sub_100keV", "-m", "hbn"])

    assert result.exit_code == 2
    assert "profile 'sub_100keV' does not include hbn" in result.stderr
