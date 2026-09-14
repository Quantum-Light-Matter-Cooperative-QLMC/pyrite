"""Install or remove PyRITE shell tab-completion in a shell rc/config file."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
from pathlib import Path

import click
from click.shell_completion import get_completion_class

from ...console.output import CLIError, emit_result, run
from ...paths import atomic_write_text, user_data_dir
from .._groups import LazyGroup

PROG_NAME = "pyrite"
COMPLETE_VAR = f"_{PROG_NAME.upper()}_COMPLETE"

# Click's shell_completion module natively speaks these three protocols only;
# anything else (PowerShell, tcsh, ksh, ...) needs its own ShellComplete
# subclass, not just another entry here.
_SHELL_SOURCE_MODE = {"bash": "bash_source", "zsh": "zsh_source", "fish": "fish_source"}
SHELL_CHOICE = click.Choice(tuple(_SHELL_SOURCE_MODE), case_sensitive=True)
_MANAGED_START = "# >>> pyrite shell completion >>>"
_MANAGED_END = "# <<< pyrite shell completion <<<"


def _completion_line(shell: str, prog_name: str = PROG_NAME) -> str:
    """Return the dynamic-source line used by older PyRITE installations."""
    mode = _SHELL_SOURCE_MODE[shell]
    complete_var = f"_{prog_name.upper()}_COMPLETE"
    if shell == "fish":
        return f"{complete_var}={mode} {prog_name} | source"
    return f'eval "$({complete_var}={mode} {prog_name})"'


def _completion_file(shell: str) -> Path:
    suffix = {"bash": "bash", "zsh": "zsh", "fish": "fish"}[shell]
    return user_data_dir() / "completions" / f"pyrite.{suffix}"


def _generated_completion(shell: str) -> str:
    completion_class = get_completion_class(shell)
    if completion_class is None:  # pragma: no cover - choices and Click stay in lockstep
        raise CLIError(f"Click does not support {shell} completion")

    # Import lazily: this command is itself lazy-loaded by the root CLI.
    from pyrite.cli import command as root_command

    source = completion_class(root_command, {}, PROG_NAME, COMPLETE_VAR).source()
    return source if source.endswith("\n") else f"{source}\n"


def _source_line(completion_file: Path) -> str:
    return f". {shlex.quote(str(completion_file))}"


def _require_persistent_executable() -> None:
    """Reject an entry point available only through the active project venv."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    executable = shutil.which(PROG_NAME)
    if not virtual_env or not executable:
        return

    environment = Path(virtual_env).resolve()
    try:
        Path(executable).resolve().relative_to(environment)
    except ValueError:
        return

    persistent_path = os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if not _is_within(Path(entry), environment)
    )
    if shutil.which(PROG_NAME, path=persistent_path) is None:
        raise click.UsageError(
            "pyrite is available only inside the active project environment, so completion "
            "would not survive a new shell. Install the persistent command with "
            "'uv tool install .' (or 'uv tool install --editable .'), run "
            "'uv tool update-shell', restart the shell, then run this command again."
        )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except (OSError, ValueError):
        return False
    return True


def _write_shell_config(path: Path, content: str) -> None:
    """Atomically update an rc file without replacing a dotfile-manager symlink."""
    target = path.resolve() if path.is_symlink() else path
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    atomic_write_text(target, content)
    if mode is not None:
        target.chmod(mode)


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


def _managed_block(shell: str, completion_file: Path) -> str:
    lines = [_MANAGED_START]
    if shell == "zsh":
        lines.extend(
            (
                "if (( ! $+functions[compdef] )); then",
                "  autoload -Uz compinit && compinit",
                "fi",
            )
        )
    lines.extend((_source_line(completion_file), _MANAGED_END))
    return "\n".join(lines) + "\n"


def _without_managed_completion(
    existing: str, shell: str, completion_file: Path
) -> tuple[str, bool]:
    """Remove exact current managed blocks for one shell."""
    lines = existing.splitlines(keepends=True)
    kept: list[str] = []
    removed = False
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        dynamic_line = _completion_line(shell)
        legacy_blocks = {(_MANAGED_START, _MANAGED_END, dynamic_line)}
        current_block = _managed_block(shell, completion_file).splitlines()
        if lines[index : index + len(current_block)] and [
            item.strip() for item in lines[index : index + len(current_block)]
        ] == [item.strip() for item in current_block]:
            removed = True
            index += len(current_block)
            continue
        if index + 2 < len(lines) and any(
            current == start
            and lines[index + 1].strip() == candidate
            and lines[index + 2].strip() == end
            for start, end, candidate in legacy_blocks
        ):
            removed = True
            index += 3
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
        "--completion-file",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Generated script path. Defaults to PyRITE's user-data directory.",
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
        "Generate a pyrite completion script and source it from a shell rc/config file.\n\n"
        "The bare pyrite executable must remain on PATH across shell sessions; "
        "for uv installations, use 'uv tool install' rather than project-only 'uv run'. "
        "Idempotent: rerunning refreshes the generated script without duplicating the rc block. "
        "With no --shell, detects from $SHELL."
    ),
)
@_target_options
def install_command(
    shell: str | None,
    rc_file: Path | None,
    completion_file: Path | None,
    dry_run: bool,
) -> None:
    """Generate and persist pyrite tab-completion setup.

    \b
    Examples:
      pyrite config completion install
      pyrite config completion install --shell zsh --dry-run
      pyrite config completion install --shell fish
    """
    shell, target = _resolve_target(shell, rc_file)
    _require_persistent_executable()
    script_path = completion_file or _completion_file(shell)
    script = _generated_completion(shell)
    block = _managed_block(shell, script_path)

    try:
        existing = target.read_text() if target.exists() else ""
    except OSError as exc:
        raise CLIError(f"could not read {target}: {exc}") from None

    proposed, removed = _without_managed_completion(existing, shell, script_path)
    separator = "" if not proposed or proposed.endswith("\n") else "\n"
    proposed = f"{proposed}{separator}{block}"

    try:
        current_script = script_path.read_text() if script_path.exists() else None
    except OSError as exc:
        raise CLIError(f"could not read {script_path}: {exc}") from None

    if proposed == existing and current_script == script:
        emit_result(f"pyrite completion already installed in {target}")
        return

    if dry_run:
        action = "refresh" if removed else "install"
        emit_result(
            f"Would {action} pyrite completion script {script_path}\n"
            f"Would update {target}:\n{block.rstrip()}"
        )
        return

    try:
        atomic_write_text(script_path, script)
        _write_shell_config(target, proposed)
    except OSError as exc:
        raise CLIError(f"could not write completion setup: {exc}") from None

    emit_result(
        f"Installed pyrite completion script {script_path}\n"
        f"Updated {target}\nRestart your shell to load it."
    )


@click.command(
    "remove",
    help=(
        "Remove pyrite tab-completion setup and its generated script.\n\n"
        "Idempotent: exact PyRITE-managed blocks are removed. "
        "With no --shell, detects from $SHELL."
    ),
)
@_target_options
def remove_command(
    shell: str | None,
    rc_file: Path | None,
    completion_file: Path | None,
    dry_run: bool,
) -> None:
    """Remove PyRITE-managed tab-completion setup from a shell rc/config file."""
    shell, target = _resolve_target(shell, rc_file)
    script_path = completion_file or _completion_file(shell)
    try:
        existing = target.read_text() if target.exists() else ""
    except OSError as exc:
        raise CLIError(f"could not read {target}: {exc}") from None

    proposed, removed = _without_managed_completion(existing, shell, script_path)
    script_exists = script_path.exists()
    if not removed and not script_exists:
        emit_result(f"pyrite completion not installed in {target}")
        return
    if dry_run:
        emit_result(
            f"Would remove pyrite completion from {target}\n"
            f"Would remove generated script {script_path}"
        )
        return
    try:
        if removed:
            _write_shell_config(target, proposed)
        script_path.unlink(missing_ok=True)
    except OSError as exc:
        raise CLIError(f"could not remove completion setup: {exc}") from None
    emit_result(f"Removed pyrite completion from {target} and {script_path}")


@click.group(
    "completion",
    cls=LazyGroup,
    lazy_commands={
        "install": "pyrite.cli.commands.completion.install_command",
        "remove": "pyrite.cli.commands.completion.remove_command",
    },
    lazy_help={
        "install": "Generate and persist pyrite shell tab-completion.",
        "remove": "Remove pyrite tab-completion setup and its generated script.",
    },
    no_args_is_help=True,
)
def command() -> None:
    """Manage pyrite shell tab-completion.

    \b
    Example:
      pyrite config completion install
      pyrite config completion install --shell zsh --dry-run
    """


def main(argv=None):
    return run(command, argv, prog_name="pyrite-completion")


if __name__ == "__main__":
    raise SystemExit(main())
