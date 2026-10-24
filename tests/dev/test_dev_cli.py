"""Tests for the repository developer command runner."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from pyrite.devtools import dev_cli


@pytest.fixture
def dev_module():
    return dev_cli


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
    assert dev_module.iter_notebooks() == [dev_module.ROOT / "checks" / "feranchuk_analysis.ipynb"]


def test_nbqa_passes_ruff_subcommand_as_one_shell_command(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_nbqa(Namespace())

    notebook = str(dev_module.ROOT / "checks" / "feranchuk_analysis.ipynb")
    assert calls == [("-m", "nbqa", "ruff check", notebook, "--nbqa-shell")]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cmd_lint", ("-m", "ruff", "check", ".")),
        ("cmd_format", ("-m", "ruff", "format", ".")),
        ("cmd_typecheck", ("-m", "ty", "check", "--exit-zero-on-warning")),
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
    monkeypatch.setattr(
        "pyrite.devtools.docs_paths.check_doc_paths", lambda root: calls.append(("paths", root))
    )
    monkeypatch.setattr(
        dev_module, "cmd_cli_reference", lambda args: calls.append(("cli-reference", args.check))
    )
    monkeypatch.setattr(dev_module, "run_uv", lambda *args, **kwargs: calls.append((args, kwargs)))

    dev_module.cmd_docs(Namespace(linkcheck=linkcheck))

    docs = dev_module.ROOT / "docs"
    assert removed == [docs / "_autosummary", docs / "_build"]
    assert calls == [
        ("paths", dev_module.ROOT),
        ("cli-reference", True),
        (
            (
                "--group",
                "docs",
                "sphinx-build",
                "-E",
                "-a",
                "-W",
                "--keep-going",
                "-b",
                builder,
                str(docs),
                str(docs / "_build" / builder),
            ),
            {"extra_env": {"PYRITE_MC_BACKEND": "cpu"}},
        ),
    ]


def test_docs_rejects_stale_cli_before_cleaning_or_building(dev_module, monkeypatch) -> None:
    monkeypatch.setattr("pyrite.devtools.docs_paths.check_doc_paths", lambda root: None)

    def stale(args):
        assert args.check is True
        raise SystemExit(2)

    monkeypatch.setattr(dev_module, "cmd_cli_reference", stale)
    removed = []
    built = []
    monkeypatch.setattr(dev_module, "_remove_path", removed.append)
    monkeypatch.setattr(dev_module, "run_uv", lambda *args, **kwargs: built.append(args))
    with pytest.raises(SystemExit, match="2"):
        dev_module.cmd_docs(Namespace(linkcheck=False))
    assert removed == []
    assert built == []


def test_docs_parser_exposes_offline_and_linkcheck_modes(dev_module) -> None:
    offline = dev_module.build_parser().parse_args(["docs"])
    online = dev_module.build_parser().parse_args(["docs", "--linkcheck"])

    assert offline.func is dev_module.cmd_docs
    assert offline.linkcheck is False
    assert online.func is dev_module.cmd_docs
    assert online.linkcheck is True


def test_perf_parser_delegates_remaining_arguments(dev_module) -> None:
    args = dev_module.build_parser().parse_args(["perf", "standard", "--perf-interval", "2"])

    assert args.func is dev_module.cmd_perf
    assert args.command_args == ["standard", "--perf-interval", "2"]


@pytest.mark.parametrize("command", ["performance", "energy-grid"])
def test_relocated_click_help_is_forwarded(dev_module, command: str, capsys) -> None:
    dev_module.main([command, "--help"])

    captured = capsys.readouterr()
    assert captured.out.startswith(f"Usage: pyrite-dev {command} ")
    assert captured.err == ""


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
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "1")
    monkeypatch.setattr(dev_module, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    dev_module.cmd_test(Namespace(numba=True, pytest_args=["--cov"]))

    assert calls == [(("-m", "pytest", "--cov"), {"extra_env": {"NUMBA_DISABLE_JIT": "1"}})]


def test_test_without_numba_omits_extra_env_kwarg(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "1")
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_test(Namespace(numba=False, pytest_args=["-k", "forward"]))

    assert calls == [("-m", "pytest", "-k", "forward")]


@pytest.mark.parametrize(
    ("argv", "numba", "pytest_args"),
    [
        (["test", "--slow", "-k", "x"], False, ["-k", "x"]),
        (["test", "--slow", "--numba", "--cov"], True, ["--cov"]),
        (["test", "--numba", "--slow"], True, []),
    ],
)
def test_main_strips_leading_slow_flag_before_pytest_args(
    dev_module, monkeypatch, argv, numba, pytest_args
) -> None:
    calls = []
    monkeypatch.setattr(dev_module, "cmd_test", lambda args: calls.append(args))
    monkeypatch.setattr(dev_module, "cmd_verify", lambda args: calls.append(args))

    dev_module.main(argv)

    assert calls[0].slow is True
    assert calls[0].numba is numba
    assert calls[0].pytest_args == pytest_args


def test_test_slow_sets_opt_in_env_for_pytest_subprocess(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "1")
    monkeypatch.setattr(dev_module, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    dev_module.cmd_test(Namespace(numba=False, slow=True, pytest_args=["-x"]))

    assert calls == [(("-m", "pytest", "-x"), {"extra_env": {"PYRITE_SLOW_TESTS": "1"}})]


@pytest.mark.parametrize(
    "argv", [["test-suite", "--slow", "integration"], ["test-suite", "integration", "--slow"]]
)
def test_test_suite_slow_sets_opt_in_env(dev_module, monkeypatch, argv) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "1")
    monkeypatch.setattr(dev_module, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    dev_module.main(argv)

    paths = [f"tests/{name}" for name in dev_module.INTEGRATION_TESTS]
    assert calls == [(("-m", "pytest", *paths), {"extra_env": {"PYRITE_SLOW_TESTS": "1"}})]


def test_full_suite_test_run_adds_memory_bounded_xdist_workers(dev_module, monkeypatch) -> None:
    monkeypatch.delenv("PYRITE_TEST_WORKERS", raising=False)
    monkeypatch.setattr(dev_module.os, "cpu_count", lambda: 16)
    available = SimpleNamespace(available=7 * dev_module.TEST_WORKER_MEMORY_BYTES // 2)
    monkeypatch.setattr("psutil.virtual_memory", lambda: available)

    assert dev_module.parallel_pytest_args(["-k", "forward"]) == [
        "-n",
        "3",
        "--dist",
        "worksteal",
        "-k",
        "forward",
    ]


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["tests/dev/test_dev_cli.py"],
        ["tests/dev/test_dev_cli.py::test_run_merges_extra_env_into_subprocess_environment"],
        ["-n", "2"],
        ["-n4"],
        ["--numprocesses=2"],
        ["-p", "no:xdist"],
    ],
)
def test_targeted_or_explicit_test_runs_keep_their_worker_choice(
    dev_module, monkeypatch, pytest_args
) -> None:
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "4")

    assert dev_module.parallel_pytest_args(pytest_args) == pytest_args


@pytest.mark.parametrize(("override", "expected"), [("1", []), ("0", []), ("5", ["-n", "5"])])
def test_test_workers_env_override(dev_module, monkeypatch, override, expected) -> None:
    monkeypatch.setenv("PYRITE_TEST_WORKERS", override)

    assert dev_module.parallel_pytest_args([])[:2] == expected


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
        for suite in dev_module.PRIMARY_TEST_SUITE_NAMES
        for path in dev_module.test_files_for_suite(suite)
    ]

    assert set(selected) == all_tests
    assert len(selected) == len(set(selected))


def test_ci_runs_every_primary_suite_once(dev_module) -> None:
    import yaml

    workflow = yaml.safe_load((dev_module.ROOT / ".github/workflows/ci.yml").read_text())
    matrix = workflow["jobs"]["focused"]["strategy"]["matrix"]

    assert set(matrix["suite"]) == set(dev_module.PRIMARY_TEST_SUITE_NAMES) - {"quantum"}
    assert len(matrix["suite"]) == len(set(matrix["suite"]))
    assert matrix["python"] == ["3.13", "3.14"]
    quantum = workflow["jobs"]["quantum"]["strategy"]["matrix"]
    assert quantum["python"] == matrix["python"]


@pytest.mark.parametrize(
    ("base", "expected"),
    [("main", True), ("issue-680", True), ("issue-681-execution", True), ("release", False)],
)
def test_ci_accepts_stacked_issue_pr_bases(dev_module, base, expected) -> None:
    from fnmatch import fnmatchcase

    import yaml

    workflow = yaml.load(
        (dev_module.ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    patterns = workflow["on"]["pull_request"]["branches"]
    assert any(fnmatchcase(base, pattern) for pattern in patterns) is expected


def test_formatter_regenerates_and_commits_cli_reference(dev_module) -> None:
    import yaml

    workflow = yaml.safe_load((dev_module.ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["format"]["steps"]
    reference = next(step for step in steps if step.get("name") == "Regenerate CLI reference")
    commit = next(
        step
        for step in steps
        if step.get("name") == "Commit formatting and generated-reference fixes"
    )

    assert reference["run"] == "uv run --locked pyrite-dev cli-reference --write"
    assert "'docs/repo-design/cli/cli-reference*.md'" in commit["run"]
    assert "style: apply Ruff formatting and regenerate CLI reference" in commit["run"]


@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("montecarlo/test_transport_per_electron.py", "transport"),
        ("montecarlo/test_coupled_radiative_runner.py", "transport-cascades"),
        ("montecarlo/test_shell_secondary_transport.py", "transport-cascades"),
        ("montecarlo/test_pair_production.py", "transport-cascades"),
        ("montecarlo/test_shell_partition.py", "transport-collisions"),
        ("montecarlo/test_straggling_surface.py", "transport-collisions"),
        ("montecarlo/test_trajectory_capture.py", "transport-runner"),
        ("montecarlo/test_gpu_oom_retry.py", "transport-runner"),
        ("montecarlo/test_brem_events.py", "spectrum-brem"),
        ("montecarlo/test_positron_brems.py", "spectrum-brem"),
        ("montecarlo/test_beam_energy_spread_grid.py", "spectrum"),
        ("montecarlo/test_coherent_emission.py", "spectrum-coherent"),
        ("montecarlo/test_temporal_profile.py", "spectrum-coherent"),
        ("montecarlo/test_transition_radiation.py", "spectrum"),
        ("montecarlo/test_channeling_handoff.py", "channeling"),
        ("montecarlo/test_channeling_replay.py", "channeling"),
        ("montecarlo/test_channeling_emission.py", "quantum"),
        ("montecarlo/test_channeling_density.py", "quantum"),
        ("montecarlo/test_channeling_axial.py", "quantum"),
        ("montecarlo/test_channeling_axial_dissipator.py", "quantum"),
        ("montecarlo/test_channeling_axial_emission.py", "quantum"),
        ("montecarlo/test_channeling_states.py", "quantum"),
        ("montecarlo/test_channeling_vortex.py", "quantum"),
        ("materials/test_channeling_profile.py", "quantum"),
        ("energy-grid/test_coherent_line_grid.py", "energy-grid"),
        ("materials/test_crystallography.py", "materials"),
        ("xsgen/test_sbethe.py", "xsgen"),
        ("checkpoint/test_io.py", "core"),
        ("energy-grid/test_cli.py", "cli"),
        ("montecarlo/test_output_noise.py", "cli"),
        ("materials/test_validation_background.py", "apps"),
        ("montecarlo/test_exports.py", "packaging"),
    ],
)
def test_domain_suite_ownership(dev_module, path, owner) -> None:
    assert dev_module.ROOT / "tests" / path in dev_module.test_files_for_suite(owner)


def test_new_modules_join_their_domain_or_core(dev_module, tmp_path) -> None:
    for relative in ("montecarlo/test_new_transport.py", "test_new_shared.py"):
        path = tmp_path / "tests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert dev_module.test_files_for_suite("transport", tmp_path) == [
        tmp_path / "tests/montecarlo/test_new_transport.py"
    ]
    assert dev_module.test_files_for_suite("core", tmp_path) == [
        tmp_path / "tests/test_new_shared.py"
    ]


def test_empty_suite_never_falls_back_to_all_tests(dev_module, monkeypatch) -> None:
    monkeypatch.setattr(dev_module, "test_files_for_suite", lambda name: [])
    calls = []
    monkeypatch.setattr(dev_module, "_run_pytest", lambda *args: calls.append(args))

    with pytest.raises(dev_module.AgentToolingError, match="contains no test modules"):
        dev_module.cmd_test_suite(Namespace(suite="transport"))

    assert calls == []


def test_test_suite_forwards_stable_paths_and_pytest_arguments(dev_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "1")
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    args = dev_module.build_parser().parse_args(["test-suite", "integration", "-k", "export"])
    args.func(args)

    assert calls == [
        (
            "-m",
            "pytest",
            "-k",
            "export",
            *[f"tests/{name}" for name in dev_module.INTEGRATION_TESTS],
        )
    ]


@pytest.mark.parametrize("suite", dev_cli.TEST_SUITE_NAMES)
def test_test_suite_uses_memory_bounded_workers(dev_module, monkeypatch, suite) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "3")
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_test_suite(Namespace(suite=suite, pytest_args=[]))

    paths = [
        str(path.relative_to(dev_module.ROOT)) for path in dev_module.test_files_for_suite(suite)
    ]
    assert calls == [("-m", "pytest", "-n", "3", "--dist", "worksteal", *paths)]


@pytest.mark.parametrize("pytest_args", [["-n", "2"], ["-p", "no:xdist"]])
def test_test_suite_respects_explicit_worker_choice(dev_module, monkeypatch, pytest_args) -> None:
    calls = []
    monkeypatch.setenv("PYRITE_TEST_WORKERS", "3")
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))

    dev_module.cmd_test_suite(Namespace(suite="integration", pytest_args=pytest_args))

    paths = [f"tests/{name}" for name in dev_module.INTEGRATION_TESTS]
    assert calls == [("-m", "pytest", *pytest_args, *paths)]


@pytest.mark.parametrize("argv", [["verify"], ["verify", "--skip-tests"]])
def test_verify_runs_static_checks_without_tests(dev_module, monkeypatch, argv) -> None:
    calls = []
    for name in (
        "cmd_imports",
        "cmd_docs",
        "cmd_lint",
        "cmd_typecheck",
        "cmd_check_lines",
        "cmd_complexity",
    ):
        monkeypatch.setattr(dev_module, name, lambda _args, name=name: calls.append(name))
    monkeypatch.setattr(dev_module, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(dev_module, "cmd_test", lambda _args: pytest.fail("tests ran"))

    dev_module.main(argv)

    assert calls == [
        "cmd_typecheck",
        "cmd_lint",
        ("-m", "ruff", "format", "--check", "."),
        "cmd_check_lines",
        "cmd_complexity",
        "cmd_docs",
        "cmd_imports",
    ]


@pytest.mark.parametrize("args", [["-k", "example"], ["--slow"], ["--numba"], ["tests"]])
def test_verify_rejects_pytest_arguments(dev_module, monkeypatch, args) -> None:
    monkeypatch.setattr(dev_module, "cmd_verify", lambda _args: pytest.fail("verify ran"))

    with pytest.raises(SystemExit, match="2"):
        dev_module.main(["verify", *args])


def test_verify_help_explains_static_gate_without_running_checks(
    dev_module, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(dev_module, "cmd_verify", lambda _args: pytest.fail("verify ran"))

    with pytest.raises(SystemExit, match="0"):
        dev_module.main(["verify", "--help"])

    output = capsys.readouterr().out
    assert "no tests run" in output
    assert "--skip-tests" not in output


def test_check_lines_reports_oversized_source_path(dev_module, monkeypatch, tmp_path) -> None:
    package = tmp_path / "src/pyrite"
    package.mkdir(parents=True)
    (package / "oversized.py").write_text("# line\n" * 1_251)
    monkeypatch.setattr(dev_module, "ROOT", tmp_path)

    with pytest.raises(dev_module.AgentToolingError, match=r"oversized.py: 1251"):
        dev_module.main(["check-lines"])


def test_source_line_budget_accepts_limit_and_ignores_non_python(tmp_path) -> None:
    from pyrite.devtools.source_checks import oversized_source_modules

    (tmp_path / "at_limit.py").write_text("# line\n" * 1_250)
    (tmp_path / "large.txt").write_text("text\n" * 2_000)

    assert oversized_source_modules(tmp_path) == {}


def test_smoke_forwards_material_and_output_directory(dev_module, monkeypatch) -> None:
    from pyrite.devtools import smoke

    calls = []
    monkeypatch.setattr(smoke, "main", lambda args: calls.append(args) or 0)

    args = dev_module.build_parser().parse_args(
        ["smoke", "--material", "hopg", "--output-dir", "/tmp/pyrite-smoke"]
    )
    args.func(args)

    assert calls == [["--material", "hopg", "--output-dir", "/tmp/pyrite-smoke"]]


def test_package_smoke_uses_importable_devtool(dev_module, monkeypatch) -> None:
    from pyrite.devtools import package_smoke

    calls = []
    monkeypatch.setattr(package_smoke, "main", lambda: calls.append(True))

    dev_module.cmd_package_smoke(Namespace())

    assert calls == [True]


def test_repo_map_groups_present_vendor_directories_as_agent_tooling(dev_module, capsys) -> None:
    dev_module.cmd_repo_map(Namespace())

    output = capsys.readouterr().out
    assert "Agent tooling:\n  .agents/\n  .claude/\n  agentdocs/\n\nCanonical commands:" in output


def test_repo_map_write_and_check_delegate_to_importable_generator(dev_module, monkeypatch) -> None:
    from pyrite.devtools import repo_map

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
    from pyrite.devtools import repo_map

    monkeypatch.setattr(repo_map, "write_or_check", lambda **_: False)

    with pytest.raises(SystemExit, match="1"):
        dev_module.cmd_repo_map(Namespace(write=False, check=True))

    assert "pyrite-dev repo-map --write" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "module_name", "target_name"),
    [
        ("cmd_cli_reference", "cli_reference", "cli-reference.md"),
        ("cmd_cli_deprecations", "cli_deprecations", "cli-deprecations.md"),
    ],
)
def test_cli_doc_commands_use_canonical_targets(
    dev_module, monkeypatch, command: str, module_name: str, target_name: str
) -> None:
    module = __import__(f"pyrite.devtools.{module_name}", fromlist=["main"])
    calls = []
    monkeypatch.setattr(module, "main", lambda args: calls.append(args) or 0)

    getattr(dev_module, command)(Namespace(check=True))

    assert calls == [
        [
            "--check",
            str(dev_module.ROOT / "docs" / "repo-design" / "cli" / target_name),
        ]
    ]


@pytest.mark.parametrize("command", ["cli-reference", "cli-deprecations"])
def test_cli_doc_parsers_accept_explicit_write_mode(dev_module, command: str) -> None:
    args = dev_module.build_parser().parse_args([command, "--write"])

    assert args.write is True
    assert args.check is False


def _fake_checkout(path):
    (path / "src" / "pyrite").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return path


def test_repo_root_follows_the_invoking_worktree_not_pyrite_home(monkeypatch, tmp_path) -> None:
    from pyrite.devtools import repo_root

    main = _fake_checkout(tmp_path / "main")
    worktree = _fake_checkout(main / ".worktrees" / "issue-1-task")
    (worktree / "tests").mkdir()
    monkeypatch.setenv("PYRITE_HOME", str(main))
    monkeypatch.chdir(worktree / "tests")

    assert repo_root() == worktree.resolve()


def test_repo_root_outside_a_checkout_uses_this_source_tree(tmp_path) -> None:
    from pathlib import Path

    from pyrite.devtools import repo_root

    assert repo_root(tmp_path) == Path(dev_cli.__file__).resolve().parents[3]


def test_dev_cli_root_is_the_checkout_holding_its_tests() -> None:
    from pathlib import Path

    assert dev_cli.ROOT == Path(__file__).resolve().parents[2]
