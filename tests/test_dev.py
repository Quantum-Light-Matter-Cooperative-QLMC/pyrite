"""Tests for the repository developer command runner."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def dev_module():
    path = Path(__file__).parents[1] / "scripts" / "dev.py"
    spec = importlib.util.spec_from_file_location("cxr_mc_dev", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acp_up_stops_bridges_when_interrupted(dev_module, monkeypatch) -> None:
    class Process:
        def poll(self):
            return None

    stopped = []
    monkeypatch.setattr(dev_module, "start_acp_servers", lambda: [Process(), Process()])
    monkeypatch.setattr(dev_module, "stop_acp_servers", lambda: stopped.append(True))
    monkeypatch.setattr(
        dev_module.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    dev_module.cmd_acp_up(None)

    assert stopped == [True]


def test_notebook_commands_only_target_the_legacy_validation_notebook(
    dev_module,
) -> None:
    assert dev_module.iter_notebooks() == [
        dev_module.ROOT / "checks" / "cxr_analysis_feranchuk.ipynb"
    ]


def test_nbqa_passes_ruff_subcommand_as_one_shell_command(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_nbqa(Namespace())

    notebook = str(dev_module.ROOT / "checks" / "cxr_analysis_feranchuk.ipynb")
    assert calls == [("-m", "nbqa", "ruff check", notebook, "--nbqa-shell")]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cmd_lint", ("-m", "ruff", "check", ".")),
        ("cmd_format", ("-m", "ruff", "format", ".")),
        ("cmd_typecheck", ("-m", "pyright")),
        ("cmd_precommit", ("-m", "pre_commit", "run", "--all-files")),
    ],
)
def test_quality_commands_use_canonical_invocations(
    dev_module, monkeypatch, command: str, expected: tuple[str, ...]
) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    getattr(dev_module, command)(Namespace())

    assert calls == [expected]


def test_test_forwards_pytest_selectors_and_arguments(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    args = dev_module.build_parser().parse_args(
        ["test", "tests/test_dev.py", "-k", "forward", "-vv"]
    )
    args.func(args)

    assert calls == [("-m", "pytest", "tests/test_dev.py", "-k", "forward", "-vv")]


def test_test_forwards_pytest_arguments_when_option_comes_first(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "cmd_test", lambda args: calls.append(args))

    dev_module.main(["test", "-k", "forward", "-vv"])

    assert calls[0].pytest_args == ["-k", "forward", "-vv"]


def test_verify_runs_checks_in_required_order(dev_module, monkeypatch) -> None:
    calls = []
    for name in ("cmd_check_skills", "cmd_lint", "cmd_typecheck", "cmd_test"):
        monkeypatch.setattr(dev_module, name, lambda _args, name=name: calls.append(name))

    dev_module.cmd_verify(Namespace(pytest_args=[]))

    assert calls == ["cmd_check_skills", "cmd_lint", "cmd_typecheck", "cmd_test"]


def test_smoke_forwards_material_and_output_directory(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    args = dev_module.build_parser().parse_args(
        ["smoke", "--material", "hopg", "--output-dir", "/tmp/cxr-mc-smoke"]
    )
    args.func(args)

    assert calls == [
        (
            str(dev_module.ROOT / "scripts" / "smoke.py"),
            "--material",
            "hopg",
            "--output-dir",
            "/tmp/cxr-mc-smoke",
        )
    ]


def test_repo_map_groups_vendor_specific_directories_as_agent_tooling(dev_module, capsys) -> None:
    dev_module.cmd_repo_map(Namespace())

    output = capsys.readouterr().out
    assert "Agent tooling:\n  .agents/\n  .claude/\n  .codex/" in output
