"""Small cross-platform developer command runner for cxr_mc.

This keeps agents and humans out of shell one-liner hell on Windows.
Exposed as the ``cxr-dev`` console script (see ``[project.scripts]``); run via:

    uv run cxr-dev <command>

Commands:
    acp-up     start the Claude and Codex ACP WebSocket bridges
    acp-down   stop bridges started by acp-up
    repo-map   print a compact repo tree and the canonical commands
    docs       clean, strict Sphinx build (optional external link check)
    lint       run Ruff over the repository
    format     run Ruff formatter
    typecheck  run ty
    precommit  run all pre-commit hooks
    nbqa       lint notebooks with nbQA + Ruff
    nbstrip    strip notebook outputs in-place
    test       run pytest, forwarding selectors and arguments
               (--numba disables JIT via NUMBA_DISABLE_JIT=1, must precede
               other forwarded args, to measure @njit bodies under --cov)
    test-suite run one stable core/CLI/app/packaging/integration test suite
    package-smoke build and install clean wheel/editable environments
    smoke      exercise checkpoint loading and plotting
    sync-skills mirror .agents/skills into .claude/skills
    check-skills validate the canonical skills and exact mirror
    bootstrap  configure per-clone local git state (TODO.md merge driver)
    verify     check skills, lint, type check, and test
    cli-deprecations  write or --check docs/cli-deprecations.md
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cxr_mc.apps._acp import ACP_SERVERS, start_acp_servers, stop_acp_servers
from cxr_mc.paths import workspace_root

from ._compat import warn_legacy_command

ROOT = workspace_root()
AGENT_SKILLS_DIR = ROOT / ".agents" / "skills"
CLAUDE_SKILLS_DIR = ROOT / ".claude" / "skills"
LEGACY_NOTEBOOK = ROOT / "checks" / "cxr_analysis_feranchuk.ipynb"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

TEST_SUITE_PATTERNS = {
    "packaging": (
        "dev/test_agent_tooling.py",
        "cli/test_contract.py",
        "cli/test_reference.py",
        "detectors/test_package.py",
        "dev/test_dev.py",
        "materials/test_materials_package.py",
        "montecarlo/test_exports.py",
        "plots/test_exports.py",
        "results/test_exports.py",
    ),
    "apps": (
        "notebooks/altair/test_*.py",
        "notebooks/analysis_app/test_app.py",
        "notebooks/analysis_app/test_analyze.py",
        "notebooks/analysis_app/test_export.py",
        "check/test_app.py",
        "plots/test_material_comparison.py",
        "notebooks/test_design.py",
        "plots/test_plotly_trajectories.py",
        "plots/test_render_*.py",
        "notebooks/test_scan_app.py",
        "notebooks/test_trace_app.py",
        "plots/test_trajectories.py",
        "materials/test_validation_background.py",
        "notebooks/test_viewer.py",
    ),
    "cli": (
        "checkpoint/test_archive.py",
        "cli/test_blaze.py",
        "materials/test_catalog_startup_errors.py",
        "check/test_config.py",
        "checkpoint/test_cli.py",
        "checkpoint/test_gc.py",
        "cli/test_*.py",
        "energy-grid/test_cli.py",
        "scan/test_local_dashboard.py",
        "montecarlo/test_output_noise.py",
        "remote/test_remote.py",
        "remote/test_click.py",
        "scan/test_scan_*.py",
        "checkpoint/test_slim.py",
    ),
}

INTEGRATION_TESTS = (
    "notebooks/analysis_app/test_app.py",
    "cli/test_contract.py",
    "materials/test_materials_package.py",
    "montecarlo/test_exports.py",
    "remote/test_remote.py",
    "results/test_exports.py",
    "scan/test_run.py",
    "scan/test_sweep.py",
)
TEST_SUITE_NAMES = ("core", "cli", "apps", "packaging", "integration")


class AgentToolingError(RuntimeError):
    """Raised when the portable skill tree is malformed or out of sync."""


def run(*args: str, cwd: Path = ROOT, extra_env: dict[str, str] | None = None) -> None:
    env = {**os.environ, **extra_env} if extra_env else None
    subprocess.run([sys.executable, *args], cwd=cwd, check=True, env=env)


def run_uv(*args: str, cwd: Path = ROOT) -> None:
    """Run a locked project tool, provisioning its explicit dependency group."""
    subprocess.run(["uv", "run", *args], cwd=cwd, check=True)


def cmd_acp_up(_: argparse.Namespace) -> None:
    processes = start_acp_servers()
    print("ACP bridges running:")
    for name, (_host, port) in ACP_SERVERS.items():
        print(f"  {name}: ws://localhost:{port}")
    print("Press Ctrl-C to stop both bridges, or run acp-down from another terminal.")
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_acp_servers()


def cmd_acp_down(_: argparse.Namespace) -> None:
    stop_acp_servers()


def iter_notebooks() -> list[Path]:
    return [LEGACY_NOTEBOOK] if LEGACY_NOTEBOOK.is_file() else []


def cmd_repo_map(args: argparse.Namespace) -> None:
    from cxr_mc.devtools import repo_map

    write = getattr(args, "write", False)
    check = getattr(args, "check", False)
    if write or check:
        current = repo_map.write_or_check(root=ROOT, check=check)
        if not current:
            print(
                "repository map dependency graph changed; "
                "regenerate with `cxr-dev repo-map --write`",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    interesting = [
        "src/cxr_mc",
        "tests",
        "checks",
        "docs",
        "dev",
        "scripts",
    ]
    print("cxr_mc repo map")
    for rel in interesting:
        path = ROOT / rel
        if not path.exists():
            continue
        if path.is_dir():
            print(f"{rel}/")
            entries = []
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    entries.append(f"  {child.name}/")
                else:
                    entries.append(f"  {child.name}")
            for line in entries[:40]:
                print(line)
            if len(entries) > 40:
                print("  ...")
            print()
        else:
            print(rel)
    print("Agent tooling:")
    for rel in [".agents", ".claude", "agentdocs"]:
        if (ROOT / rel).exists():
            print(f"  {rel}/")
    print()
    print("Canonical commands:")
    for line in [
        "uv run cxr-dev acp-up",
        "uv run cxr-dev acp-down",
        "uv run cxr-dev lint",
        "uv run cxr-dev format",
        "uv run cxr-dev typecheck",
        "uv run cxr-dev precommit",
        "uv run cxr-dev test",
        "uv run cxr-dev test-suite core",
        "uv run cxr-dev package-smoke",
        "uv run cxr-dev smoke --material hopg --output-dir /tmp/cxr-mc-smoke",
        "uv run cxr-dev sync-skills",
        "uv run cxr-dev check-skills",
        "uv run cxr-dev bootstrap",
        "uv run cxr-dev verify",
        "uv run cxr-dev nbqa",
        "uv run cxr-dev nbstrip",
    ]:
        print(f"  {line}")


def cmd_lint(_: argparse.Namespace) -> None:
    run("-m", "ruff", "check", ".")


def cmd_format(_: argparse.Namespace) -> None:
    run("-m", "ruff", "format", ".")


def cmd_typecheck(_: argparse.Namespace) -> None:
    run("-m", "ty", "check")


def cmd_precommit(_: argparse.Namespace) -> None:
    run("-m", "pre_commit", "run", "--all-files")


def cmd_nbqa(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run(
        "-m",
        "nbqa",
        "ruff check",
        *[str(p) for p in notebooks],
        "--nbqa-shell",
    )


def cmd_nbstrip(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run("-m", "nbstripout", *[str(p) for p in notebooks])


def cmd_test(args: argparse.Namespace) -> None:
    pytest_args = getattr(args, "pytest_args", [])
    if getattr(args, "numba", False):
        run("-m", "pytest", *pytest_args, extra_env={"NUMBA_DISABLE_JIT": "1"})
    else:
        run("-m", "pytest", *pytest_args)


def test_files_for_suite(name: str, root: Path = ROOT) -> list[Path]:
    """Return deterministic repository test paths for one documented suite.

    Four domain suites partition every ``tests/test_*.py`` module exactly once.
    ``integration`` intentionally samples public boundaries across domains; the
    full ``cxr-dev test`` gate remains unchanged.
    """
    tests_dir = root / "tests"
    files = sorted(tests_dir.rglob("test_*.py"))
    if name == "integration":
        selected = [tests_dir / filename for filename in INTEGRATION_TESTS]
        missing = [path.name for path in selected if not path.is_file()]
        if missing:
            raise AgentToolingError("missing integration tests: " + ", ".join(missing))
        return selected
    if name not in {"core", *TEST_SUITE_PATTERNS}:
        raise AgentToolingError(f"unknown test suite {name!r}")

    selected = []
    for path in files:
        owner = "core"
        for candidate, patterns in TEST_SUITE_PATTERNS.items():
            relative = path.relative_to(tests_dir).as_posix()
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
                owner = candidate
                break
        if owner == name:
            selected.append(path)
    return selected


def cmd_test_suite(args: argparse.Namespace) -> None:
    paths = [str(path.relative_to(ROOT)) for path in test_files_for_suite(args.suite)]
    run("-m", "pytest", *paths, *getattr(args, "pytest_args", []))


def cmd_smoke(args: argparse.Namespace) -> None:
    from cxr_mc.devtools.smoke import main

    status = main(["--material", args.material, "--output-dir", args.output_dir])
    if status:
        raise SystemExit(status)


def cmd_package_smoke(_: argparse.Namespace) -> None:
    from cxr_mc.devtools.package_smoke import main

    main()


def _frontmatter_fields(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentToolingError(f"{path}: cannot read UTF-8 text: {exc}") from exc

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise AgentToolingError(f"{path}: missing or unterminated YAML frontmatter")

    frontmatter_text = parts[1]
    try:
        fields = yaml.safe_load(frontmatter_text)
        if not isinstance(fields, dict):
            raise ValueError("Frontmatter is not a dictionary")
        return {str(k): str(v) for k, v in fields.items()}
    except Exception as exc:
        raise AgentToolingError(f"{path}: malformed YAML frontmatter: {exc}") from exc


def validate_skill_file(path: Path) -> None:
    """Validate the required portable frontmatter for one skill."""
    fields = _frontmatter_fields(path)
    for field in ("name", "description"):
        if field not in fields:
            raise AgentToolingError(f"{path}: missing {field} in YAML frontmatter")
    name = fields["name"]
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise AgentToolingError(f"{path}: invalid skill name {name!r}")
    if name != path.parent.name:
        raise AgentToolingError(
            f"{path}: skill name {name!r} must match directory {path.parent.name!r}"
        )
    if not fields["description"].startswith("Use when"):
        raise AgentToolingError(f"{path}: description must start with 'Use when'")


def _relative_files(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise AgentToolingError(f"missing skill directory: {root}")
    return {path.relative_to(root): path for path in sorted(root.rglob("*")) if path.is_file()}


def _validated_canonical_files(root: Path) -> dict[Path, Path]:
    files = _relative_files(root)
    skill_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not skill_dirs:
        raise AgentToolingError(f"no skill directories found under {root}")
    for skill_dir in skill_dirs:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            raise AgentToolingError(f"{skill_dir}: missing SKILL.md")
        validate_skill_file(skill_file)
    return files


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def check_skill_trees(canonical: Path = AGENT_SKILLS_DIR, mirror: Path = CLAUDE_SKILLS_DIR) -> None:
    """Validate canonical skills and require an exact byte-for-byte mirror."""
    canonical_files = _validated_canonical_files(canonical)
    mirror_files = _relative_files(mirror)

    canonical_names = set(canonical_files)
    mirror_names = set(mirror_files)
    missing = sorted(canonical_names - mirror_names)
    extra = sorted(mirror_names - canonical_names)
    different = sorted(
        rel
        for rel in canonical_names & mirror_names
        if canonical_files[rel].read_bytes() != mirror_files[rel].read_bytes()
        or mirror_files[rel].is_symlink()
    )
    errors = []
    if missing:
        errors.append("missing from mirror: " + ", ".join(map(str, missing)))
    if extra:
        errors.append("unexpected mirror files: " + ", ".join(map(str, extra)))
    if different:
        errors.append("byte-different mirror files: " + ", ".join(map(str, different)))
    if errors:
        raise AgentToolingError("; ".join(errors))


def sync_skill_trees(canonical: Path = AGENT_SKILLS_DIR, mirror: Path = CLAUDE_SKILLS_DIR) -> None:
    """Replace the generated mirror with ordinary copies of canonical files."""
    _validated_canonical_files(canonical)
    staged = mirror.with_name(f".{mirror.name}.tmp")
    backup = mirror.with_name(f".{mirror.name}.backup")

    for path in (staged, backup):
        _remove_path(path)

    shutil.copytree(canonical, staged, symlinks=False)
    check_skill_trees(canonical, staged)

    had_mirror = mirror.exists() or mirror.is_symlink()
    if had_mirror:
        mirror.replace(backup)
    try:
        staged.replace(mirror)
    except OSError:
        if had_mirror and not mirror.exists():
            backup.replace(mirror)
        raise
    finally:
        _remove_path(staged)
    _remove_path(backup)


def cmd_sync_skills(_: argparse.Namespace) -> None:
    sync_skill_trees()
    check_skill_trees()
    print(
        f"Synchronized {AGENT_SKILLS_DIR.relative_to(ROOT)} -> {CLAUDE_SKILLS_DIR.relative_to(ROOT)}"
    )


def cmd_check_skills(_: argparse.Namespace) -> None:
    check_skill_trees()
    print("Skill mirror is valid and synchronized.")


TODO_MERGE_DRIVER = "merge.ours.driver"


def _git_config_get(key: str) -> str | None:
    result = subprocess.run(
        ["git", "config", "--local", "--get", key],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def todo_merge_driver_configured() -> bool:
    """Whether the `ours` merge driver referenced by `.gitattributes` exists.

    `.gitattributes` maps `TODO.md merge=ours`, but the driver definition lives
    in local git config and cannot be committed. Without it, git silently falls
    back to a normal 3-way merge and reintroduces TODO.md conflicts.
    """
    return _git_config_get(TODO_MERGE_DRIVER) == "true"


def cmd_bootstrap(_: argparse.Namespace) -> None:
    """Install per-clone local git state. Idempotent; safe to re-run."""
    if todo_merge_driver_configured():
        print(f"{TODO_MERGE_DRIVER}=true already configured.")
        return
    subprocess.run(
        ["git", "config", "--local", TODO_MERGE_DRIVER, "true"],
        cwd=ROOT,
        check=True,
    )
    print(
        f"Configured {TODO_MERGE_DRIVER}=true; conflicting TODO.md hunks now "
        "resolve to the current branch's copy on merge/rebase."
    )


def cmd_imports(_: argparse.Namespace) -> None:
    subprocess.run(["lint-imports", "--no-cache"], cwd=ROOT, check=True)


def cmd_docs(args: argparse.Namespace) -> None:
    """Build all maintained and generated documentation from clean state."""
    docs_dir = ROOT / "docs"
    _remove_path(docs_dir / "_autosummary")
    _remove_path(docs_dir / "_build")
    builder = "linkcheck" if getattr(args, "linkcheck", False) else "html"
    run_uv(
        "--group",
        "docs",
        "sphinx-build",
        "-E",
        "-a",
        "-W",
        "--keep-going",
        "-b",
        builder,
        str(docs_dir),
        str(docs_dir / "_build" / builder),
    )


def cmd_verify(args: argparse.Namespace) -> None:
    cmd_check_skills(args)
    if not todo_merge_driver_configured():
        print(
            "warning: TODO.md merge driver not configured; "
            "run `uv run cxr-dev bootstrap` (see agentdocs/README.md).",
            file=sys.stderr,
        )
    cmd_imports(args)
    cmd_repo_map(argparse.Namespace(check=True, write=False))
    cmd_lint(args)
    cmd_typecheck(args)
    cmd_test(args)


def cmd_regen_golden(args: argparse.Namespace) -> None:
    from cxr_mc.energy_grid.golden import regen

    raise SystemExit(regen(check=getattr(args, "check", False)))


def cmd_cli_deprecations(args: argparse.Namespace) -> None:
    from cxr_mc.devtools.cli_deprecations import main

    target = ROOT / "docs" / "cli-deprecations.md"
    mode = "--check" if getattr(args, "check", False) else "--write"
    status = main([mode, str(target)])
    if status:
        raise SystemExit(status)


def build_parser(prog_name: str = "pyrite-dev") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog_name)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("acp-up", cmd_acp_up),
        ("acp-down", cmd_acp_down),
        ("lint", cmd_lint),
        ("format", cmd_format),
        ("typecheck", cmd_typecheck),
        ("precommit", cmd_precommit),
        ("nbqa", cmd_nbqa),
        ("nbstrip", cmd_nbstrip),
        ("sync-skills", cmd_sync_skills),
        ("check-skills", cmd_check_skills),
        ("bootstrap", cmd_bootstrap),
        ("package-smoke", cmd_package_smoke),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    repo_map = sub.add_parser(
        "repo-map",
        help="print repository inventory or update its generated dependency graph",
    )
    repo_map_mode = repo_map.add_mutually_exclusive_group()
    repo_map_mode.add_argument(
        "--write",
        action="store_true",
        help="replace the generated dependency region in docs/repo_map.md",
    )
    repo_map_mode.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated dependency region is stale",
    )
    repo_map.set_defaults(func=cmd_repo_map)
    docs = sub.add_parser(
        "docs",
        help="clean and strictly build the Sphinx documentation",
    )
    docs.add_argument(
        "--linkcheck",
        action="store_true",
        help="check external links (requires network access; excluded from offline gates)",
    )
    docs.set_defaults(func=cmd_docs)
    test = sub.add_parser("test")
    test.add_argument(
        "--numba",
        action="store_true",
        help=(
            "set NUMBA_DISABLE_JIT=1 so @njit bodies (transport.py, geometry.py, "
            "groove.py) are traced by coverage instead of running compiled; "
            "must precede pytest_args, e.g. `cxr-dev test --numba --cov`"
        ),
    )
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test.set_defaults(func=cmd_test)
    test_suite = sub.add_parser("test-suite")
    test_suite.add_argument("suite", choices=TEST_SUITE_NAMES)
    test_suite.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test_suite.set_defaults(func=cmd_test_suite)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--material", default="hopg")
    smoke.add_argument("--output-dir", default="smoke_out")
    smoke.set_defaults(func=cmd_smoke)
    verify = sub.add_parser("verify")
    verify.add_argument("pytest_args", nargs=argparse.REMAINDER)
    verify.set_defaults(func=cmd_verify)
    regen_golden = sub.add_parser("regen-golden")
    regen_golden.add_argument("--check", action="store_true")
    regen_golden.set_defaults(func=cmd_regen_golden)
    cli_deprecations = sub.add_parser("cli-deprecations")
    cli_deprecations.add_argument("--check", action="store_true")
    cli_deprecations.set_defaults(func=cmd_cli_deprecations)
    return ap


def main(argv: list[str] | None = None, *, prog_name: str = "pyrite-dev") -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {"test", "verify"}:
        command = raw_args[0]
        func = cmd_test if command == "test" else cmd_verify
        rest = raw_args[1:]
        numba = bool(rest) and rest[0] == "--numba"
        pytest_args = rest[1:] if numba else rest
        func(argparse.Namespace(command=command, numba=numba, pytest_args=pytest_args))
        return
    args = build_parser(prog_name).parse_args(raw_args)
    args.func(args)


def legacy_main(argv: list[str] | None = None) -> None:
    """Run the retained ``cxr-dev`` compatibility executable."""
    warn_legacy_command("cxr-dev", "pyrite-dev")
    main(argv, prog_name="cxr-dev")


if __name__ == "__main__":
    main()
