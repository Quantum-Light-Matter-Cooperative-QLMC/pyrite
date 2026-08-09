"""Tests for the repository developer command runner."""

from __future__ import annotations

from argparse import Namespace

import pytest

from cxr_mc import _dev


@pytest.fixture
def dev_module():
    return _dev


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
        ("cmd_typecheck", ("-m", "ty", "check")),
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


@pytest.mark.parametrize(
    ("linkcheck", "builder"),
    [(False, "html"), (True, "linkcheck")],
)
def test_docs_cleans_generated_trees_and_runs_strict_build(
    dev_module, monkeypatch, linkcheck: bool, builder: str
) -> None:
    removed = []
    calls = []
    monkeypatch.setattr(dev_module, "_remove_path", removed.append)
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_docs(Namespace(linkcheck=linkcheck))

    docs = dev_module.ROOT / "docs"
    assert removed == [docs / "_autosummary", docs / "_build"]
    assert calls == [
        (
            "-m",
            "sphinx",
            "-E",
            "-a",
            "-W",
            "--keep-going",
            "-b",
            builder,
            str(docs),
            str(docs / "_build" / builder),
        )
    ]


def test_docs_parser_exposes_offline_and_linkcheck_modes(dev_module) -> None:
    offline = dev_module.build_parser().parse_args(["docs"])
    online = dev_module.build_parser().parse_args(["docs", "--linkcheck"])

    assert offline.func is dev_module.cmd_docs
    assert offline.linkcheck is False
    assert online.func is dev_module.cmd_docs
    assert online.linkcheck is True


def test_test_forwards_pytest_selectors_and_arguments(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    args = dev_module.build_parser().parse_args(
        ["test", "tests/dev/test_dev.py", "-k", "forward", "-vv"]
    )
    args.func(args)

    assert calls == [("-m", "pytest", "tests/dev/test_dev.py", "-k", "forward", "-vv")]


def test_test_forwards_pytest_arguments_when_option_comes_first(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "cmd_test", lambda args: calls.append(args))

    dev_module.main(["test", "-k", "forward", "-vv"])

    assert calls[0].pytest_args == ["-k", "forward", "-vv"]


def test_main_strips_leading_numba_flag_before_pytest_args(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "cmd_test", lambda args: calls.append(args))

    dev_module.main(["test", "--numba", "--cov"])

    assert calls[0].numba is True
    assert calls[0].pytest_args == ["--cov"]


def test_test_numba_sets_disable_jit_env_for_pytest_subprocess(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        dev_module, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    dev_module.cmd_test(Namespace(numba=True, pytest_args=["--cov"]))

    assert calls == [(("-m", "pytest", "--cov"), {"extra_env": {"NUMBA_DISABLE_JIT": "1"}})]


def test_test_without_numba_omits_extra_env_kwarg(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_test(Namespace(numba=False, pytest_args=["-k", "forward"]))

    assert calls == [("-m", "pytest", "-k", "forward")]


def test_run_merges_extra_env_into_subprocess_environment(dev_module, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        dev_module.subprocess,
        "run",
        lambda args, cwd, check, env: captured.update(args=args, env=env),
    )
    monkeypatch.setenv("EXISTING_VAR", "kept")

    dev_module.run("-m", "pytest", extra_env={"NUMBA_DISABLE_JIT": "1"})

    assert captured["env"]["NUMBA_DISABLE_JIT"] == "1"
    assert captured["env"]["EXISTING_VAR"] == "kept"


def test_domain_suites_partition_every_test_module_once(dev_module) -> None:
    all_tests = set((dev_module.ROOT / "tests").rglob("test_*.py"))
    selected = [
        path
        for suite in ("core", "cli", "apps", "packaging")
        for path in dev_module.test_files_for_suite(suite)
    ]

    assert set(selected) == all_tests
    assert len(selected) == len(set(selected))


def test_test_suite_forwards_stable_paths_and_pytest_arguments(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    args = dev_module.build_parser().parse_args(["test-suite", "integration", "-k", "export"])
    args.func(args)

    assert calls == [
        (
            "-m",
            "pytest",
            *[f"tests/{name}" for name in dev_module.INTEGRATION_TESTS],
            "-k",
            "export",
        )
    ]


def test_verify_runs_checks_in_required_order(dev_module, monkeypatch) -> None:
    calls = []
    for name in (
        "cmd_check_skills",
        "cmd_imports",
        "cmd_repo_map",
        "cmd_lint",
        "cmd_typecheck",
        "cmd_test",
    ):
        monkeypatch.setattr(dev_module, name, lambda _args, name=name: calls.append(name))

    dev_module.cmd_verify(Namespace(pytest_args=[]))

    assert calls == [
        "cmd_check_skills",
        "cmd_imports",
        "cmd_repo_map",
        "cmd_lint",
        "cmd_typecheck",
        "cmd_test",
    ]


def test_smoke_forwards_material_and_output_directory(dev_module, monkeypatch) -> None:
    from cxr_mc.devtools import smoke

    calls = []
    monkeypatch.setattr(smoke, "main", lambda args: calls.append(args) or 0)

    args = dev_module.build_parser().parse_args(
        ["smoke", "--material", "hopg", "--output-dir", "/tmp/cxr-mc-smoke"]
    )
    args.func(args)

    assert calls == [["--material", "hopg", "--output-dir", "/tmp/cxr-mc-smoke"]]


def test_package_smoke_uses_importable_devtool(dev_module, monkeypatch) -> None:
    from cxr_mc.devtools import package_smoke

    calls = []
    monkeypatch.setattr(package_smoke, "main", lambda: calls.append(True))

    dev_module.cmd_package_smoke(Namespace())

    assert calls == [True]


def test_repo_map_groups_present_vendor_directories_as_agent_tooling(dev_module, capsys) -> None:
    dev_module.cmd_repo_map(Namespace())

    output = capsys.readouterr().out
    assert (
        "Agent tooling:\n  .agents/\n  .claude/\n  agentdocs/\n\nCanonical commands:"
        in output
    )


def test_repo_map_write_and_check_delegate_to_importable_generator(
    dev_module, monkeypatch
) -> None:
    from cxr_mc.devtools import repo_map

    calls = []
    monkeypatch.setattr(
        repo_map,
        "write_or_check",
        lambda *, root, check: calls.append((root, check)) or True,
    )

    dev_module.cmd_repo_map(Namespace(write=True, check=False))
    dev_module.cmd_repo_map(Namespace(write=False, check=True))

    assert calls == [(dev_module.ROOT, False), (dev_module.ROOT, True)]


def test_repo_map_check_reports_stale_document(dev_module, monkeypatch, capsys) -> None:
    from cxr_mc.devtools import repo_map

    monkeypatch.setattr(repo_map, "write_or_check", lambda **_: False)

    with pytest.raises(SystemExit, match="1"):
        dev_module.cmd_repo_map(Namespace(write=False, check=True))

    assert "cxr-dev repo-map --write" in capsys.readouterr().err
