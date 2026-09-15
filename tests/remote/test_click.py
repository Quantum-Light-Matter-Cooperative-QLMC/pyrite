"""Focused Click-contract tests for ``pyrite remote``."""

from __future__ import annotations

import click
import pytest

from pyrite import remote
from pyrite.cli.commands import _remote_actions
from pyrite.cli.commands import remote as remote_cli
from pyrite.remote import lifecycle, viewer
from tests.helpers.cli import assert_clean_result, invoke

REMOTE_COMMANDS = (
    "gc",
    "pull",
    "performance",
    "rm",
    "prune-jobs",
    "sync",
)


def assert_legacy_run_result(result, *, stderr=""):
    assert_clean_result(result, stderr=stderr)


def test_remote_exports_click_group():
    assert isinstance(remote_cli.command, click.Group)
    assert set(remote_cli.command.commands) == set(REMOTE_COMMANDS)


@pytest.mark.parametrize("name", REMOTE_COMMANDS)
def test_every_remote_click_help_path_is_offline(name):
    result = invoke(remote_cli.command, [name, "--help"])

    assert_clean_result(result)
    assert f"Usage: remote {name} " in result.stdout


def test_sync_verbose_flag_enables_raw_ssh_echo(monkeypatch):
    from pyrite.remote import transport

    seen = []

    def fake_sync_code():
        seen.append(transport._VERBOSE)

    monkeypatch.setattr(transport, "sync_code", fake_sync_code)

    assert_clean_result(invoke(remote_cli.command, ["sync"]))
    assert_clean_result(invoke(remote_cli.command, ["sync", "-v"]))

    assert seen == [False, True]
    assert transport._VERBOSE is False


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

    assert_clean_result(invoke(remote_cli.command, ["performance", "list"]))
    assert_clean_result(invoke(remote_cli.command, ["performance", "pull", "baseline"]))
    assert_clean_result(invoke(remote_cli.command, ["performance", "rm", "baseline", "--yes"]))

    assert calls == [
        ("list",),
        ("pull", "baseline"),
        ("prune", ["baseline"], {"all_profiles": False, "yes": True}),
    ]


def test_run_click_defaults_and_zero_meanings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(
        remote_cli.start_command, ["standard", "-m", "hopg", "--workers", "0", "--headless"]
    )

    assert_legacy_run_result(result)
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
                "cpu": False,
                "cpu_only": False,
                "no_cache": False,
                "recompute": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--cpu", "--cpu-only"], "--cpu and --cpu-only are mutually exclusive"),
        (["--cpu-only", "--nsys"], "--cpu-only cannot be combined with --nsys"),
        (["--cpu", "--chunk-minutes", "1"], "--cpu requires --chunk-minutes 0"),
        (["--cpu-only", "--chunk-minutes", "1"], "--cpu-only requires --chunk-minutes 0"),
        (["--cpu-only", "--perf-reps", "2"], "cannot be combined with --perf-reps"),
        (
            ["--cpu-only", "--perf-interval", "2"],
            "cannot be combined with --perf-interval",
        ),
    ],
)
def test_run_cpu_incompatible_inputs_fail_before_submission(monkeypatch, flags, message):
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda *_args, **_kwargs: pytest.fail("invalid CPU flags must not submit"),
    )

    result = invoke(remote_cli.start_command, ["standard", "-m", "hopg", *flags])

    assert result.exit_code == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("flag", "cpu", "cpu_only"),
    [("--cpu", True, False), ("--cpu-only", False, True)],
)
def test_run_cpu_flags_imply_performance_and_monolithic_dispatch(monkeypatch, flag, cpu, cpu_only):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(
        remote_cli.start_command,
        ["standard", "-m", "hopg", flag, "--headless"],
    )

    assert_legacy_run_result(result)
    assert calls[0][1]["performance_profile"] == "standard"
    assert calls[0][1]["chunk_minutes"] == 0.0
    assert calls[0][1]["cpu"] is cpu
    assert calls[0][1]["cpu_only"] is cpu_only


def test_combined_cpu_failure_pulls_retained_primary_performance_artifacts(monkeypatch):
    pulled = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda *_args, **_kwargs: "job")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(remote.state, "_job_succeeded", lambda _jobid: False)
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda profile: pulled.append(profile),
    )

    result = invoke(remote_cli.start_command, ["standard", "-m", "hopg", "--cpu"])

    assert_legacy_run_result(
        result,
        stderr="CPU phase failed; pulling retained primary performance artifacts\nperformance profiling used isolated job-local checkpoints; skipping automatic checkpoint pull\n",
    )
    assert pulled == ["standard"]


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
        remote.state, "_materials_needing_pull", lambda _jobid, materials: list(materials)
    )
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
        remote_cli.start_command,
        [
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

    assert_legacy_run_result(result)
    assert queued[0][1]["performance_repetitions"] == 1
    assert queued[0][1]["performance_interval"] == 2.0
    assert performance_pulled == ["standard"]
    assert pulled[0][1]["level9"] is True


def test_zhai_execution_dispatches_without_a_deprecation_diagnostic(monkeypatch):
    monkeypatch.setattr(_remote_actions, "remote_check", lambda **_kwargs: None)

    result = invoke(remote_cli.check_command, [])

    assert result.exit_code == 0
    assert "is deprecated" not in result.stderr


def test_zhai_pull_dispatches_to_the_cache_retrieval_path(monkeypatch):
    pulled = []
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: pulled.append("pull"))

    result = invoke(remote_cli.check_command, ["--pull"])

    assert result.exit_code == 0
    assert pulled == ["pull"]


def test_zhai_detached_follow_submits_and_attaches(monkeypatch):
    attached = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **_kwargs: "job")
    monkeypatch.setattr(viewer, "attach", lambda jobid: attached.append(jobid))

    result = invoke(remote_cli.check_command, ["--detached", "--follow"])

    assert result.exit_code == 0
    assert attached == ["job"]


def test_pull_zhai_preset_dispatches_without_checkpoint_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: calls.append("pull"))

    result = invoke(remote_cli.command, ["pull", "--preset", "zhai"])
    incompatible = invoke(remote_cli.command, ["pull", "hopg", "--preset", "zhai"])
    output = invoke(remote_cli.command, ["pull", "--preset", "zhai", "-o", "wide"])

    assert_clean_result(result)
    assert calls == ["pull"]
    assert incompatible.exit_code == 2
    assert "does not take checkpoint option(s): material" in incompatible.stderr
    assert output.exit_code == 2
    assert "does not take checkpoint option(s): output" in output.stderr


@pytest.mark.parametrize(
    "target, argv, option",
    [
        ("start_command", ["standard", "-m", "hopg", "--workers", "-1"], "--workers"),
        (
            "start_command",
            ["standard", "-m", "hopg", "--chunk-minutes", "-1"],
            "--chunk-minutes",
        ),
        ("rebrem_command", ["hopg", "--ne-brem", "0"], "--ne-brem"),
        ("rebrem_command", ["hopg", "--step", "nan"], "--step"),
        ("reline_command", ["hopg", "--line-ne", "0"], "--line-ne"),
        ("reline_command", ["hopg", "--line-step", "inf"], "--line-step"),
        ("check_command", ["--ne", "0"], "--ne"),
        ("check_command", ["--tmd-azimuth", "nan"], "--tmd-azimuth"),
        ("check_command", ["--tmd-azimuth", "-inf"], "--tmd-azimuth"),
    ],
)
def test_remote_numeric_domains_fail_at_click_boundary(target, argv, option):
    result = invoke(getattr(remote_cli, target), argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert option in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "argv, message",
    [
        (["pull", "hopg", "--brem-only", "--line-only"], "mutually exclusive"),
        (["rm"], "needs material"),
        (["gc", "--all", "--profile", "sub_100keV"], "cannot be combined"),
    ],
)
def test_remote_incompatible_click_inputs_are_usage_errors(argv, message):
    result = invoke(remote_cli.command, argv)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_remote_gc_runs_both_reclamations_with_standard_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "prune_remote",
        lambda **kwargs: calls.append(("prune", kwargs)),
    )
    monkeypatch.setattr(
        lifecycle,
        "reap_reservations",
        lambda **kwargs: calls.append(("reap", kwargs)),
    )

    result = invoke(remote_cli.command, ["gc"])

    assert_clean_result(result)
    assert calls == [
        ("prune", {"all_profiles": False, "catalog_profile": None, "yes": False}),
        ("reap", {"min_age_minutes": 5.0, "yes": False}),
    ]


@pytest.mark.parametrize("command_name", ["start", "rebrem", "reline"])
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

    if command_name in ("rebrem", "reline"):
        argv = ["hopg", "--fidelity", "survey", "--no-sync", "--dry-run"]
    else:
        argv = [
            "standard",
            "-m",
            "hopg",
            "--fidelity",
            "survey",
            "--no-sync",
            "--headless",
        ]

    result = invoke(getattr(remote_cli, f"{command_name}_command"), argv)

    assert_clean_result(result)
    assert calls[0]["fidelity"] == "survey"


@pytest.mark.parametrize(
    ("command_name", "starter"),
    [
        ("rebrem", "start_rebrem_queue"),
        ("reline", "start_reline_queue"),
    ],
)
def test_remote_recompute_detach_skips_viewer_and_pull(monkeypatch, command_name, starter):
    monkeypatch.setattr(lifecycle, starter, lambda _materials, **_kwargs: "job")
    monkeypatch.setattr(
        viewer,
        "attach",
        lambda _jobid: pytest.fail("detached recompute must not attach"),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda *_args, **_kwargs: pytest.fail("detached recompute must not pull"),
    )

    result = invoke(getattr(remote_cli, f"{command_name}_command"), ["hopg", "--detach"])

    assert_clean_result(result)


def test_run_profile_and_material_dispatch(monkeypatch):
    import pyrite.materials as materials_pkg

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
        remote_cli.start_command,
        ["sub_100keV", "-m", "hopg", "--headless"],
    )

    assert_legacy_run_result(result)
    assert calls[0]["catalog_profile"] == "sub_100keV"


def test_run_profile_with_membership_defaults_materials(monkeypatch):
    import pyrite.materials as materials_pkg

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

    result = invoke(remote_cli.start_command, ["sub_100keV", "--headless"])

    assert_legacy_run_result(result)
    assert calls[0][0] == ["hopg", "mose2"]
    assert calls[0][1]["catalog_profile"] == "sub_100keV"


def test_run_profile_without_membership_uses_catalog_materials(monkeypatch):
    import pyrite.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")
        material_keys = ("mose2", "hopg")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote_cli.start_command, ["sub_100keV", "--headless"])

    assert_legacy_run_result(result)
    assert calls[0][0] == ["mose2", "hopg"]


def test_pull_hash_option_dispatches_and_requires_one_qualified_selector(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote_cli.command, ["pull", "hopg@sub_100keV", "--hash", "abc123"])
    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]
    assert calls[0][1]["hash_prefix"] == "abc123"

    ambiguous = invoke(remote_cli.command, ["pull", "hopg", "wse2", "--hash", "abc123"])
    assert ambiguous.exit_code == 2
    assert "requires exactly one MATERIAL@PROFILE" in ambiguous.stderr


@pytest.mark.parametrize(
    ("flag", "dataset"),
    [("--brem-only", "brem"), ("--line-only", "line")],
)
def test_partial_pull_all_forwards_full_material_list(monkeypatch, tmp_path, flag, dataset):
    import pyrite.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard",)
        material_keys = ("hopg", "hbn", "mos2")

        def profile_materials(self, _name):
            return self.material_keys

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    seen = {}
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: seen.update(materials=materials, kwargs=kwargs),
    )

    result = invoke(remote_cli.command, ["pull", "--all", flag])

    assert_clean_result(result)
    assert seen["materials"] == ["hopg", "hbn", "mos2"]
    assert seen["kwargs"]["dataset"] == dataset
    assert seen["kwargs"]["grid"] is False


def test_clear_implicit_profile_uses_catalog_materials(monkeypatch):
    import pyrite.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard",)
        material_keys = ("hopg", "hbn")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "clear_remote",
        lambda materials, yes, **kwargs: calls.append((materials, yes, kwargs)),
    )

    result = invoke(remote_cli.command, ["rm", "--profile", "standard"])

    assert_clean_result(result)
    assert calls == [(["hopg", "hbn"], False, {"catalog_profile": "standard"})]


def test_pull_profile_expands_membership_to_qualified_selectors(monkeypatch):
    import pyrite.materials as materials_pkg

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

    result = invoke(remote_cli.command, ["pull", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV", "mose2@sub_100keV"]


def test_pull_positional_profile_expands_membership(monkeypatch):
    import pyrite.materials as materials_pkg

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

    result = invoke(remote_cli.command, ["pull", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV", "mose2@sub_100keV"]


def test_pull_profile_qualifies_explicit_materials(monkeypatch):
    import pyrite.materials as materials_pkg

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

    result = invoke(remote_cli.command, ["pull", "hopg", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]

    with_all = invoke(remote_cli.command, ["pull", "--all", "--profile", "sub_100keV"])
    assert with_all.exit_code == 0
    assert "warning:" in with_all.stderr
    assert "ignoring --all" in with_all.stderr

    with_selector = invoke(
        remote_cli.command, ["pull", "hopg@sub_100keV", "--profile", "sub_100keV"]
    )
    assert with_selector.exit_code == 2
    assert "drop the @PROFILE selector" in with_selector.stderr


def test_pull_profile_without_membership_uses_catalog_materials(monkeypatch):
    import pyrite.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")
        material_keys = ("mose2", "hopg")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda materials, **kwargs: calls.append((materials, kwargs)),
    )

    result = invoke(remote_cli.command, ["pull", "--profile", "sub_100keV"])

    assert_clean_result(result)
    assert calls[0][0] == ["mose2@sub_100keV", "hopg@sub_100keV"]


def test_pull_material_option_narrows_positional_profile(monkeypatch):
    import pyrite.materials as materials_pkg

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

    result = invoke(remote_cli.command, ["pull", "sub_100keV", "-m", "hopg"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg@sub_100keV"]


def test_pull_material_option_rejects_material_outside_profile(monkeypatch):
    import pyrite.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg",)

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())

    result = invoke(remote_cli.command, ["pull", "sub_100keV", "-m", "hbn"])

    assert result.exit_code == 2
    assert "profile 'sub_100keV' does not include hbn" in result.stderr
