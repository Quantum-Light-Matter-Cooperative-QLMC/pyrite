"""Local compute-performance artifact lifecycle."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import click

from .._core import CLIError, DeprecatingGroup, confirm_destructive, emit_result, hidden_alias

_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _profile_name(ctx, param, value):
    del ctx, param
    if value is None:
        return None
    if _PROFILE_RE.fullmatch(value) is None:
        raise click.BadParameter(
            "expected letters, digits, underscores, or hyphens; must start with a letter or digit"
        )
    return value


def _profile_names(ctx, param, values):
    return tuple(_profile_name(ctx, param, value) for value in values)


def _profile_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())


def _files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(item.relative_to(path)), stat.st_mtime_ns, stat.st_size)
        for item in _files(path)
        if (stat := item.stat())
    )


def analyze(name: str, performance_dir: Path, sample_period: float) -> int:
    """Analyze one named performance profile and emit its artifact summary."""
    from cxr_mc.performance_analysis import (
        PerformanceAnalysisError,
        analyze_performance_profile,
    )

    try:
        result = analyze_performance_profile(
            name,
            performance_dir,
            sample_period=sample_period,
        )
    except (OSError, PerformanceAnalysisError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(
        f"analyzed {result['sessions']} sessions ({result['intervals']} intervals) "
        f"-> {result['analysis_root']}"
    )
    if result["incomplete_sessions"]:
        click.echo(
            f"warning: {result['incomplete_sessions']} incomplete session(s)",
            err=True,
        )
    return 0


@click.group("performance", cls=DeprecatingGroup, no_args_is_help=True)
def command():
    """List, analyze, or delete local compute-performance artifacts."""


@command.command("list")
@click.argument("profiles", nargs=-1, callback=_profile_names)
@click.option(
    "--performance-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("performance-profiles"),
    show_default=True,
)
def list_command(profiles, performance_dir):
    """List local performance profiles with artifact counts and sizes."""
    available = {path.name: path for path in _profile_dirs(performance_dir)}
    selected = list(profiles) if profiles else sorted(available)
    if not selected:
        emit_result("(no local performance profiles)")
        return 0
    missing = [name for name in selected if name not in available]
    if missing:
        raise CLIError(f"no local performance profile: {', '.join(missing)}")
    for name in selected:
        files = _files(available[name])
        size = sum(path.stat().st_size for path in files)
        emit_result(f"{name}: {len(files)} artifact(s), {size} bytes")
    return 0


@command.command("analyze")
@click.argument("name", callback=_profile_name)
@click.option(
    "--performance-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("performance-profiles"),
    show_default=True,
    help="Directory containing NAME's local or pulled NDJSON logs.",
)
@click.option(
    "--sample-period",
    type=click.FloatRange(min=0, min_open=True),
    default=5.0,
    show_default=True,
    metavar="SECONDS",
    help="Expected sampling period; intervals over twice this value are gaps.",
)
def analyze_command(name, performance_dir, sample_period):
    """Analyze NAME's logs into CSV, Markdown, and PNG artifacts."""
    return analyze(name, performance_dir, sample_period)


@command.command("rm")
@click.argument("profiles", nargs=-1, callback=_profile_names)
@click.option("--all", "all_profiles", is_flag=True, help="Select every local profile.")
@click.option("-y", "--yes", is_flag=True, help="Delete exact previewed profile directories.")
@click.option(
    "--performance-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("performance-profiles"),
    show_default=True,
)
def rm_command(profiles, all_profiles, yes, performance_dir):
    """Delete explicitly selected local performance profiles; preview by default."""
    if all_profiles and profiles:
        raise click.UsageError("performance rm --all does not take PROFILE names")
    if not all_profiles and not profiles:
        raise click.UsageError("performance rm needs PROFILE name(s), or use --all")
    available = {path.name: path for path in _profile_dirs(performance_dir)}
    selected = sorted(available) if all_profiles else list(profiles)
    missing = [name for name in selected if name not in available]
    if missing:
        raise CLIError(f"no local performance profile: {', '.join(missing)}")
    snapshots = {name: _signature(available[name]) for name in selected}
    if not selected:
        emit_result("(nothing to prune)")
        return 0
    emit_result("would delete local performance profiles:")
    for name in selected:
        emit_result(f"  {available[name]} ({len(snapshots[name])} artifact(s))")
    if not confirm_destructive(yes, "Delete these local performance profiles?"):
        return 0
    for name in selected:
        path = available[name]
        if not path.is_dir() or path.is_symlink() or _signature(path) != snapshots[name]:
            raise CLIError(f"refusing to prune: performance profile changed: {path}")
    for name in selected:
        shutil.rmtree(available[name])
    try:
        os.rmdir(performance_dir)
    except OSError:
        pass
    emit_result(f"deleted {len(selected)} local performance profile(s)")
    return 0


hidden_alias(command, rm_command, "prune")
