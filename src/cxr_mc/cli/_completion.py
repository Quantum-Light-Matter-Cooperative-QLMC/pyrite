"""Fast, failure-silent shell-completion providers for the Click CLI.

Completion must stay safe to invoke from an interactive shell: local providers
perform bounded, non-recursive reads; remote job discovery uses non-interactive
SSH with both connection and subprocess timeouts.  Every provider returns an
empty list when its backing data is unavailable.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import click
from click.shell_completion import CompletionItem

from .. import DATA_DIR

MAX_LOCAL_CANDIDATES = 200
MAX_REMOTE_CANDIDATES = 100
REMOTE_COMPLETION_TIMEOUT_SECONDS = 1.5

_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
_ARCHIVE_CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"

Completion = Callable[[object, object, str], list[CompletionItem]]


class MaterialKey(click.ParamType):
    """Offline material-catalog key validated without scientific imports."""

    name = "material"

    def convert(self, value, param, ctx):
        key = str(value)
        keys = _material_keys()
        if key not in keys:
            self.fail(
                f"{key!r} is not a configured material; choose one shown by shell completion",
                param,
                ctx,
            )
        return key


MATERIAL = MaterialKey()


def _items(values: Iterable[str], incomplete: str) -> list[CompletionItem]:
    """Return safe, sorted prefix matches."""
    return [
        CompletionItem(value)
        for value in sorted(set(values))
        if value.startswith(incomplete) and _SAFE_TOKEN_RE.fullmatch(value)
    ]


@lru_cache(maxsize=1)
def _material_keys() -> tuple[str, ...]:
    """Read catalog keys without importing scientific material modules."""
    try:
        with (DATA_DIR / "materials.toml").open("rb") as source:
            materials = tomllib.load(source).get("materials", {})
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    if not isinstance(materials, dict):
        return ()
    return tuple(key for key in materials if _SAFE_TOKEN_RE.fullmatch(key))


@lru_cache(maxsize=1)
def _profile_keys() -> tuple[str, ...]:
    """Read catalog ``[profiles.*]`` keys without scientific imports."""
    try:
        with (DATA_DIR / "materials.toml").open("rb") as source:
            profiles = tomllib.load(source).get("profiles", {})
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    if not isinstance(profiles, dict):
        return ()
    return tuple(key for key in profiles if _SAFE_TOKEN_RE.fullmatch(key))


def complete_profile(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete one catalog ``[profiles.*]`` name."""
    del ctx, param
    return _items(_profile_keys(), incomplete)


def complete_material(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete one offline material-catalog key."""
    del ctx, param
    return _items(_material_keys(), incomplete)


def complete_material_csv(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete current token in a comma-separated material list."""
    del ctx, param
    prefix, separator, fragment = incomplete.rpartition(",")
    selected = set(prefix.split(",")) if separator else set()
    rendered_prefix = f"{prefix}," if separator else ""
    return [
        CompletionItem(f"{rendered_prefix}{item.value}")
        for item in _items((key for key in _material_keys() if key not in selected), fragment)
    ]


def complete_remote_checkpoint_stem(
    ctx: object,
    param: object,
    incomplete: str,
) -> list[CompletionItem]:
    """Complete predictable remote checkpoint stems without network access."""
    del ctx, param
    keys = _material_keys()
    return _items(
        (stem for key in keys for stem in (key, f"{key}_quick", f"{key}_blazed")),
        incomplete,
    )


def _safe_file_stems(directory: Path, *, suffix: str = ".pkl") -> list[str]:
    """Return at most ``MAX_LOCAL_CANDIDATES`` regular-file stems."""
    try:
        entries = os.scandir(directory)
    except OSError:
        return []
    values: list[str] = []
    try:
        with entries:
            for entry in entries:
                if len(values) >= MAX_LOCAL_CANDIDATES:
                    break
                if (
                    entry.name.endswith(suffix)
                    and entry.is_file(follow_symlinks=False)
                    and _SAFE_TOKEN_RE.fullmatch(entry.name[: -len(suffix)])
                ):
                    values.append(entry.name[: -len(suffix)])
    except OSError:
        return []
    return values


def _safe_checkpoint_stems(directory: Path) -> list[str]:
    """Return bounded component-directory and legacy checkpoint stems."""
    values = _safe_file_stems(directory)
    try:
        entries = os.scandir(directory)
    except OSError:
        return values
    try:
        with entries:
            for entry in entries:
                if len(values) >= MAX_LOCAL_CANDIDATES:
                    break
                if (
                    entry.is_dir(follow_symlinks=False)
                    and _SAFE_TOKEN_RE.fullmatch(entry.name)
                    and (Path(entry.path) / "line.pkl").is_file()
                ):
                    values.append(entry.name)
    except OSError:
        pass
    return values


def _checkpoint_root(ctx: object) -> Path:
    params = getattr(ctx, "params", {})
    configured = params.get("checkpoint_dir") if isinstance(params, dict) else None
    return Path(configured) if configured else _DEFAULT_CHECKPOINT_ROOT


def complete_checkpoint_stem(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete active checkpoint stems, honoring ``--checkpoint-dir``."""
    del param
    return _items(_safe_checkpoint_stems(_checkpoint_root(ctx)), incomplete)


def complete_archive_stem(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete repo-anchored active stems used by archive commands."""
    del ctx, param
    return _items(_safe_checkpoint_stems(_ARCHIVE_CHECKPOINT_ROOT), incomplete)


def complete_checkpoint(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete active checkpoint paths for commands such as ``cxr slim``."""
    del param
    typed = Path(incomplete)
    if typed.parent != Path("."):
        directory = typed.parent
        prefix = typed.name
        rendered_parent = f"{typed.parent}{os.sep}"
    else:
        directory = _checkpoint_root(ctx)
        prefix = typed.name
        rendered_parent = f"{directory}{os.sep}"
    return [
        CompletionItem(
            f"{rendered_parent}{stem}"
            if (directory / stem).is_dir()
            else f"{rendered_parent}{stem}.pkl"
        )
        for stem in sorted(set(_safe_checkpoint_stems(directory)))
        if stem.startswith(prefix)
    ]


def complete_archive_label(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete labels from the local checkpoint archive shelf."""
    del ctx, param
    archive_dir = _ARCHIVE_CHECKPOINT_ROOT / "archive"
    return _items(_safe_file_stems(archive_dir), incomplete)


def _query_remote_job_ids() -> Sequence[str]:
    """Read recent remote job-directory names with bounded, prompt-free SSH."""
    from .._remote import config

    jobs_dir = config.remote_path(config.JOBS_SUBDIR)
    remote_command = (
        f"JOBS={config.shell_word(jobs_dir)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'find "$JOBS" -mindepth 1 -maxdepth 1 -type d -printf "%f\\n" '
        f"| sort -r | head -n {MAX_REMOTE_CANDIDATES}"
    )
    result = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ConnectTimeout=1",
            "-o",
            "LogLevel=ERROR",
            config.remote_host(),
            remote_command,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=REMOTE_COMPLETION_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        return ()
    return result.stdout.splitlines()[:MAX_REMOTE_CANDIDATES]


def complete_job_id(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
    """Complete remote job IDs; return nothing on config, SSH, or timeout failure."""
    del ctx, param
    try:
        values = _query_remote_job_ids()
    except (OSError, subprocess.SubprocessError, SystemExit, ValueError):
        return []
    return _items(values, incomplete)


def choice_completer(values: Iterable[object]) -> Completion:
    """Build a Click callback for finite values not represented by ``Choice``."""
    choices = tuple(str(value) for value in values)

    def complete(ctx: object, param: object, incomplete: str) -> list[CompletionItem]:
        del ctx, param
        return _items(choices, incomplete)

    return complete


__all__ = [
    "choice_completer",
    "complete_archive_label",
    "complete_archive_stem",
    "complete_checkpoint",
    "complete_checkpoint_stem",
    "complete_job_id",
    "complete_material",
    "complete_material_csv",
    "complete_profile",
]
