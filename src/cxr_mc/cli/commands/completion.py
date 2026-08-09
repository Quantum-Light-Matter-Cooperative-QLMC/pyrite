"""Install or remove cxr shell tab-completion in a shell rc/config file."""

from __future__ import annotations

import os
from pathlib import Path

import click

from .._core import CLIError, LazyGroup, emit_result, run

PROG_NAME = "cxr"
COMPLETE_VAR = f"_{PROG_NAME.upper()}_COMPLETE"

# Click's shell_completion module natively speaks these three protocols only;
# anything else (PowerShell, tcsh, ksh, ...) needs its own ShellComplete
# subclass, not just another entry here.
_SHELL_SOURCE_MODE = {"bash": "bash_source", "zsh": "zsh_source", "fish": "fish_source"}
SHELL_CHOICE = click.Choice(tuple(_SHELL_SOURCE_MODE), case_sensitive=True)
_MANAGED_START = "# >>> cxr shell completion >>>"
_MANAGED_END = "# <<< cxr shell completion <<<"
_LEGACY_COMMENT = "# cxr shell completion"


def _completion_line(shell: str) -> str:
    mode = _SHELL_SOURCE_MODE[shell]
    if shell == "fish":
        return f"{COMPLETE_VAR}={mode} {PROG_NAME} | source"
    return f'eval "$({COMPLETE_VAR}={mode} {PROG_NAME})"'


def _default_rc_file(shell: str) -> Path:
    if shell == "zsh":
        zdotdir = os.environ.get("ZDOTDIR")
        base = Path(zdotdir) if zdotdir else Path.home()
        return base / ".zshrc"
    if shell == "fish":
        return Path.home() / ".config" / "fish" / "config.fish"
    return Path.home() / ".bashrc"


def _detect_shell() -> str | None:
    name = Path(os.environ.get("SHELL", "")).name
    return name if name in _SHELL_SOURCE_MODE else None


def _resolve_target(shell: str | None, rc_file: Path | None) -> tuple[str, Path]:
    if shell is None:
        shell = _detect_shell()
        if shell is None:
            raise click.UsageError(
                "could not detect shell from $SHELL; pass --shell explicitly "
                f"({', '.join(_SHELL_SOURCE_MODE)})"
            )
    return shell, rc_file if rc_file is not None else _default_rc_file(shell)


def _managed_block(shell: str) -> str:
    return f"{_MANAGED_START}\n{_completion_line(shell)}\n{_MANAGED_END}\n"


def _without_managed_completion(existing: str, line: str) -> tuple[str, bool]:
    """Remove exact current or legacy managed blocks for one shell."""
    lines = existing.splitlines(keepends=True)
    kept: list[str] = []
    removed = False
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if (
            current == _MANAGED_START
            and index + 2 < len(lines)
            and lines[index + 1].strip() == line
            and lines[index + 2].strip() == _MANAGED_END
        ):
            removed = True
            index += 3
            continue
        if (
            current == _LEGACY_COMMENT
            and index + 1 < len(lines)
            and lines[index + 1].strip() == line
        ):
            removed = True
            index += 2
            continue
        kept.append(lines[index])
        index += 1
    return "".join(kept), removed


def _target_options(function):
    function = click.option(
        "--dry-run",
        is_flag=True,
        help="Print what would change without writing.",
    )(function)
    function = click.option(
        "--rc-file",
        "rc_file",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Rc/config file to edit. Defaults to the shell's standard location.",
    )(function)
    return click.option(
        "--shell",
        type=SHELL_CHOICE,
        default=None,
        help="Target shell. Defaults to detecting from $SHELL.",
    )(function)


@click.command(
    "install",
    help=(
        "Append cxr tab-completion setup to a shell rc/config file.\n\n"
        "Idempotent: rerunning skips a file that already contains the line. "
        "With no --shell, detects from $SHELL."
    ),
)
@_target_options
def install_command(shell: str | None, rc_file: Path | None, dry_run: bool) -> None:
    """Append cxr tab-completion setup to a shell rc/config file.

    \b
    Examples:
      cxr completion install
      cxr completion install --shell zsh --dry-run
      cxr completion install --shell fish
    """
    shell, target = _resolve_target(shell, rc_file)
    line = _completion_line(shell)

    try:
        existing = target.read_text() if target.exists() else ""
    except OSError as exc:
        raise CLIError(f"could not read {target}: {exc}") from None

    if any(candidate.strip() == line for candidate in existing.splitlines()):
        emit_result(f"cxr completion already installed in {target}")
        return

    if dry_run:
        emit_result(f"Would append to {target}:\n{_managed_block(shell).rstrip()}")
        return

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        with target.open("a") as handle:
            handle.write(f"{separator}{_managed_block(shell)}")
    except OSError as exc:
        raise CLIError(f"could not write {target}: {exc}") from None

    emit_result(f"Installed cxr completion in {target}\nRestart your shell or run: source {target}")


@click.command(
    "remove",
    help=(
        "Remove cxr tab-completion setup from a shell rc/config file.\n\n"
        "Idempotent: only exact cxr-managed current or legacy blocks are removed. "
        "With no --shell, detects from $SHELL."
    ),
)
@_target_options
def remove_command(shell: str | None, rc_file: Path | None, dry_run: bool) -> None:
    """Remove cxr-managed tab-completion setup from a shell rc/config file."""
    shell, target = _resolve_target(shell, rc_file)
    try:
        existing = target.read_text() if target.exists() else ""
    except OSError as exc:
        raise CLIError(f"could not read {target}: {exc}") from None

    proposed, removed = _without_managed_completion(existing, _completion_line(shell))
    if not removed:
        emit_result(f"cxr completion not installed in {target}")
        return
    if dry_run:
        emit_result(f"Would remove cxr completion from {target}")
        return
    try:
        target.write_text(proposed)
    except OSError as exc:
        raise CLIError(f"could not write {target}: {exc}") from None
    emit_result(f"Removed cxr completion from {target}")


@click.group(
    "completion",
    cls=LazyGroup,
    lazy_commands={
        "install": "cxr_mc.cli.commands.completion.install_command",
        "remove": "cxr_mc.cli.commands.completion.remove_command",
    },
    lazy_help={
        "install": "Append cxr tab-completion setup to a shell rc/config file.",
        "remove": "Remove cxr tab-completion setup from a shell rc/config file.",
    },
    no_args_is_help=True,
)
def command() -> None:
    """Manage cxr shell tab-completion.

    \b
    Example:
      cxr completion install
      cxr completion install --shell zsh --dry-run
    """


def main(argv=None):
    return run(command, argv, prog_name="cxr-completion")


if __name__ == "__main__":
    raise SystemExit(main())
