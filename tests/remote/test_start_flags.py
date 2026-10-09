"""Unit tests for ``remote run`` flag-compatibility resolution."""

import re

import click
import pytest

from pyrite.cli.commands import remote as remote_cli

PROFILE = "standard"

# Click defaults for every ``remote run`` option, so each test states only the
# flags it exercises.
DEFAULT_FLAGS = {
    "catalog_profile": None,
    "material": None,
    "quick": False,
    "workers": None,
    "parallel_materials": None,
    "chunk_minutes": None,
    "perf": False,
    "performance_repetitions": 1,
    "performance_interval": 5.0,
    "spec_chunk": None,
    "brem_chunk": None,
    "nsys": False,
    "py_spy": False,
    "cpu": False,
    "cpu_only": False,
    "no_cache": False,
    "recompute": False,
    "no_sync": False,
    "dry_run": False,
    "headless": False,
    "no_pull": False,
    "grid": False,
    "drop_wide_brem": False,
    "downcast": False,
    "level9": False,
}


def plan(**overrides):
    flags = remote_cli._StartFlags(**{**DEFAULT_FLAGS, "catalog_profile": PROFILE, **overrides})
    return remote_cli._plan_start(flags, PROFILE)


def test_start_flags_covers_every_run_option():
    assert set(DEFAULT_FLAGS) == {parameter.name for parameter in remote_cli.start_command.params}


def test_run_options_keep_their_documented_order():
    assert [parameter.name for parameter in remote_cli.start_command.params] == list(DEFAULT_FLAGS)


def test_plain_run_profiles_nothing_and_chunks_by_default():
    resolved = plan()

    assert resolved.performance_profile is None
    assert resolved.chunk_minutes == 10.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"perf": True},
        {"nsys": True, "material": "hopg"},
        {"cpu": True},
        {"cpu_only": True},
        {"perf": True, "performance_repetitions": 2},
    ],
)
def test_profiling_modes_instrument_the_profile(overrides):
    assert plan(**overrides).performance_profile == PROFILE


@pytest.mark.parametrize(
    "overrides",
    [
        {"perf": True, "performance_repetitions": 2},
        {"nsys": True, "material": "hopg"},
        {"cpu": True},
        {"cpu_only": True},
    ],
)
def test_profiling_modes_default_to_one_monolithic_job(overrides):
    assert plan(**overrides).chunk_minutes == 0.0


def test_perf_alone_keeps_the_ordinary_chunk_default():
    assert plan(perf=True).chunk_minutes == 10.0


def test_explicit_chunk_minutes_is_never_overridden():
    assert plan(chunk_minutes=7.5).chunk_minutes == 7.5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpu": True, "cpu_only": True}, "--cpu and --cpu-only are mutually exclusive"),
        (
            {"no_cache": True, "recompute": True},
            "--no-cache and --recompute are mutually exclusive",
        ),
        ({"cpu_only": True, "nsys": True}, "--cpu-only cannot be combined with --nsys"),
        (
            {"cpu_only": True, "performance_repetitions": 2},
            "--cpu-only cannot be combined with --perf-reps",
        ),
        (
            {"cpu_only": True, "performance_interval": 2.0},
            "--cpu-only cannot be combined with --perf-interval",
        ),
        (
            {"cpu_only": True, "spec_chunk": 4},
            "--cpu-only cannot be combined with GPU chunk pins",
        ),
        (
            {"cpu_only": True, "brem_chunk": 4},
            "--cpu-only cannot be combined with GPU chunk pins",
        ),
        ({"performance_repetitions": 2}, "--perf-reps requires --perf"),
        ({"performance_interval": 2.0}, "--perf-interval requires --perf"),
        ({"spec_chunk": 4}, "--spec-chunk requires --perf"),
        ({"brem_chunk": 4}, "--brem-chunk requires --perf"),
        ({"nsys": True}, "--nsys requires one explicit -m/--material"),
        (
            {"perf": True, "performance_repetitions": 2, "chunk_minutes": 1.0},
            "--perf-reps requires --chunk-minutes 0",
        ),
        (
            {"perf": True, "performance_repetitions": 2, "parallel_materials": 2},
            "--perf-reps requires one material process per GPU",
        ),
        (
            {"nsys": True, "material": "hopg", "chunk_minutes": 1.0},
            "--nsys requires --chunk-minutes 0",
        ),
        (
            {"nsys": True, "material": "hopg", "perf": True, "performance_repetitions": 2},
            "--nsys requires --perf-reps 1",
        ),
        (
            {"nsys": True, "material": "hopg", "parallel_materials": 2},
            "--nsys requires one material process per GPU",
        ),
        ({"cpu": True, "chunk_minutes": 1.0}, "--cpu requires --chunk-minutes 0"),
        ({"cpu_only": True, "chunk_minutes": 1.0}, "--cpu-only requires --chunk-minutes 0"),
        (
            {"parallel_materials": 2, "chunk_minutes": 1.0},
            "--parallel-materials requires --chunk-minutes 0",
        ),
        ({"quick": True, "grid": True}, "quick checkpoints aren't grid-filterable"),
        ({"headless": True, "no_pull": True}, "--headless cannot be combined with --no-pull"),
    ],
)
def test_incompatible_flag_combinations_are_usage_errors(overrides, message):
    with pytest.raises(click.UsageError, match=re.escape(message)):
        plan(**overrides)


def test_earlier_rules_win_when_several_are_violated():
    # Mutual exclusion is checked before chunking, so --cpu/--cpu-only reports
    # first even though --chunk-minutes 1 is also invalid for both.
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        plan(cpu=True, cpu_only=True, chunk_minutes=1.0)
