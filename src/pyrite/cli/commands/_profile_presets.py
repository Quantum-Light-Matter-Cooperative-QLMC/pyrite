"""``profile set|remove`` options that attach or detach named presets (ADR-0021)."""

import functools

import click

from pyrite._config_presets import KINDS
from pyrite.campaign import config_presets
from pyrite.cli import _completion

_COMPLETE = {
    "physics": _completion.complete_physics_preset,
    "numerics": _completion.complete_numerics_preset,
}


def attach_options(function):
    """Add ``--physics NAME``/``--numerics NAME`` as one ``presets`` mapping argument."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        names = {kind: kwargs.pop(f"{kind.table}_preset") for kind in KINDS}
        presets = {kind: name for kind, name in names.items() if name is not None}
        return function(*args, presets=presets, **kwargs)

    for kind in reversed(KINDS):
        wrapper = click.option(
            f"--{kind.table}",
            f"{kind.table}_preset",
            metavar="NAME",
            shell_complete=_COMPLETE[kind.table],
            help=f"Attach a named {kind.table} preset; the profile's own fields stay overrides.",
        )(wrapper)
    return wrapper


def detach_options(function):
    """Add ``--physics``/``--numerics`` flags as one ``detach`` kinds argument."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        detach = [kind for kind in KINDS if kwargs.pop(f"detach_{kind.table}")]
        return function(*args, detach=detach, **kwargs)

    for kind in reversed(KINDS):
        wrapper = click.option(
            f"--{kind.table}",
            f"detach_{kind.table}",
            is_flag=True,
            help=f"Detach the {kind.table} preset; the profile's own fields remain.",
        )(wrapper)
    return wrapper


def tables(kinds):
    """Return the table names of ``kinds`` for confirmation prompts."""
    return [kind.table for kind in kinds]


def detach(document, name, kinds):
    """Detach ``kinds`` presets from profile ``name``; return ``{table: preset}``."""
    return {kind.table: config_presets.detach_preset(document, kind, name) for kind in kinds}


def detached_message(detached):
    """Summarize :func:`detach` results for the command's done message."""
    return "".join(f"; {table}: detached preset {preset}" for table, preset in detached.items())
