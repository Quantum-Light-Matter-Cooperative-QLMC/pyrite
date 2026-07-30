"""Small cross-platform developer command runner for cxr_mc.

This keeps agents and humans out of shell one-liner hell on Windows.
Exposed as the ``cxr-dev`` console script (see ``[project.scripts]``); run via:

    uv run cxr-dev <command>

Commands:
    acp-up     start the Claude and Codex ACP WebSocket bridges
    acp-down   stop bridges started by acp-up
    repo-map   print a compact repo tree and the canonical commands
    lint       run Ruff over the repository
    format     run Ruff formatter
    typecheck  run ty
    precommit  run all pre-commit hooks
    nbqa       lint notebooks with nbQA + Ruff
    nbstrip    strip notebook outputs in-place
    test       run pytest, forwarding selectors and arguments
    smoke      exercise checkpoint loading and plotting
    sync-skills mirror .agents/skills into .claude/skills
    check-skills validate the canonical skills and exact mirror
    bootstrap  configure per-clone local git state (TODO.md merge driver)
    verify     check skills, lint, type check, and test
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cxr_mc._acp import ACP_SERVERS, start_acp_servers, stop_acp_servers

ROOT = Path(__file__).resolve().parents[2]
AGENT_SKILLS_DIR = ROOT / ".agents" / "skills"
CLAUDE_SKILLS_DIR = ROOT / ".claude" / "skills"
LEGACY_NOTEBOOK = ROOT / "checks" / "cxr_analysis_feranchuk.ipynb"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class AgentToolingError(RuntimeError):
    """Raised when the portable skill tree is malformed or out of sync."""


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


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


def cmd_repo_map(_: argparse.Namespace) -> None:
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
    for rel in [".agents", ".claude"]:
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
    run("-m", "pytest", *getattr(args, "pytest_args", []))


def cmd_smoke(args: argparse.Namespace) -> None:
    run(
        str(ROOT / "scripts" / "smoke.py"),
        "--material",
        args.material,
        "--output-dir",
        args.output_dir,
    )


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


def cmd_verify(args: argparse.Namespace) -> None:
    cmd_check_skills(args)
    if not todo_merge_driver_configured():
        print(
            "warning: TODO.md merge driver not configured; "
            "run `uv run cxr-dev bootstrap` (see tasks/README.md).",
            file=sys.stderr,
        )
    cmd_lint(args)
    cmd_typecheck(args)
    cmd_test(args)


def cmd_regen_golden(args: argparse.Namespace) -> None:
    from cxr_mc.line_grid.golden import regen

    raise SystemExit(regen(check=getattr(args, "check", False)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cxr-dev")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("acp-up", cmd_acp_up),
        ("acp-down", cmd_acp_down),
        ("repo-map", cmd_repo_map),
        ("lint", cmd_lint),
        ("format", cmd_format),
        ("typecheck", cmd_typecheck),
        ("precommit", cmd_precommit),
        ("nbqa", cmd_nbqa),
        ("nbstrip", cmd_nbstrip),
        ("sync-skills", cmd_sync_skills),
        ("check-skills", cmd_check_skills),
        ("bootstrap", cmd_bootstrap),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    test = sub.add_parser("test")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test.set_defaults(func=cmd_test)
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
    return ap


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {"test", "verify"}:
        command = raw_args[0]
        func = cmd_test if command == "test" else cmd_verify
        func(argparse.Namespace(command=command, pytest_args=raw_args[1:]))
        return
    args = build_parser().parse_args(raw_args)
    args.func(args)


if __name__ == "__main__":
    main()
