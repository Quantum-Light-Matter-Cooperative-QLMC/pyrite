"""Focused Click-contract tests for ``cxr remote``."""

from __future__ import annotations

import click
import pytest

from cxr_mc import remote
from cxr_mc._remote import lifecycle, viewer
from tests.cli_helpers import assert_clean_result, invoke

REMOTE_COMMANDS = (
    "scan",
    "rebrem",
    "reline",
    "submit",
    "start",
    "attach",
    "jobs",
    "status",
    "logs",
    "stop",
    "reap",
    "pull",
    "profile",
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


def test_start_click_defaults_and_zero_meanings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote.command, ["submit", "hopg", "--workers", "0", "--headless"])

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


def test_hidden_remote_aliases_remain_callable():
    for alias in ("start", "check"):
        result = invoke(remote.command, [alias, "--help"])
        assert_clean_result(result)

    root_help = invoke(remote.command, ["--help"])
    command_lines = {
        line.split()[0]
        for line in root_help.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.lstrip().startswith("-")
    }
    assert {"submit", "validate"}.issubset(command_lines)
    assert command_lines.isdisjoint({"start", "check"})


@pytest.mark.parametrize(
    "argv, option",
    [
        (["start", "hopg", "--workers", "-1"], "--workers"),
        (["start", "hopg", "--chunk-minutes", "-1"], "--chunk-minutes"),
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
        (["start"], "needs material"),
        (["start", "hopg", "--all"], "--all does not take"),
        (["scan", "hopg", "--quick", "--grid"], "drop --grid"),
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


@pytest.mark.parametrize("command_name", ["scan", "rebrem", "reline", "submit"])
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

    # ``rebrem``/``reline`` support --dry-run, which returns before the
    # (mocked, always-disconnected) viewer.attach call; legacy ``scan`` always
    # attaches. Canonical ``submit`` defaults to attach+pull, so use its
    # explicit detached mode while this test isolates fidelity dispatch.
    argv = [command_name, "hopg", "--fidelity", "survey", "--no-sync"]
    if command_name in ("rebrem", "reline"):
        argv.append("--dry-run")
    elif command_name == "submit":
        argv.append("--headless")

    result = invoke(remote.command, argv)

    if command_name == "scan":
        assert result.exit_code == 0
        assert "skipping automatic pull" in result.stderr
    else:
        assert_clean_result(result)
    assert calls[0]["fidelity"] == "survey"


def test_submit_profile_option_dispatches_catalog_profile(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

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
        ["submit", "hopg", "--profile", "sub_100keV", "--headless"],
    )

    assert_clean_result(result)
    assert calls[0]["catalog_profile"] == "sub_100keV"


def test_submit_profile_with_membership_defaults_materials_when_none_given(monkeypatch):
    """A profile naming its own campaign materials is enough to run `submit
    --profile NAME` with no MATERIAL/--all -- naming it directly still works
    too (see test_submit_profile_option_dispatches_catalog_profile)."""
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return ("hopg", "mose2")

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "job",
    )

    result = invoke(remote.command, ["submit", "--profile", "sub_100keV", "--headless"])

    assert_clean_result(result)
    assert calls[0][0] == ["hopg", "mose2"]
    assert calls[0][1]["catalog_profile"] == "sub_100keV"


def test_submit_profile_without_membership_still_needs_material_or_all(monkeypatch):
    """A profile with no explicit membership (implicit all-in-use) has no
    narrower list to default to, so bare `submit --profile NAME` keeps
    requiring --all/-A or an explicit MATERIAL -- unchanged from before."""
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())

    result = invoke(remote.command, ["submit", "--profile", "sub_100keV"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "needs material name(s), or use --all" in result.stderr


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
    assert with_all.exit_code == 2
    assert "drop --all" in with_all.stderr

    with_selector = invoke(remote.command, ["pull", "hopg@sub_100keV", "--profile", "sub_100keV"])
    assert with_selector.exit_code == 2
    assert "drop the @PROFILE selector" in with_selector.stderr


def test_pull_profile_without_membership_needs_explicit_materials(monkeypatch):
    import cxr_mc.materials as materials_pkg

    class _FakeCatalog:
        profile_names = ("standard", "sub_100keV")

        def profile_materials(self, _name):
            return None

    monkeypatch.setattr(materials_pkg, "CATALOG", _FakeCatalog())

    result = invoke(remote.command, ["pull", "--profile", "sub_100keV"])

    assert result.exit_code == 2
    assert "no explicit material membership" in result.stderr
