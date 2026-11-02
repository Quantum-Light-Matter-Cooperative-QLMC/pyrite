"""Named machine-specific remote targets; never part of scientific identity."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import asdict, replace

from ..console import config as store
from . import config
from .config import RemoteTarget

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_FIELDS = ("host", "remote_dir", "uv", "gpu_vendor", "partition", "nodelist", "gres", "description")


def validate_name(name: str) -> str:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(
            "target name must start with a letter and contain up to 64 letters, digits, _, . or -"
        )
    return name


def _remote_table(document):
    remote = document.get("remote", {})
    if not isinstance(remote, dict):
        raise store.ConfigError("[remote] must be a table")
    return remote


def _target_table():
    table = _remote_table(store._read_store()).get("targets", {})
    if not isinstance(table, dict):
        raise store.ConfigError("remote.targets must be a table")
    return table


def _load_target(name, values):
    validate_name(name)
    if not isinstance(values, dict) or set(values) - set(_FIELDS):
        raise store.ConfigError(f"invalid remote.targets.{name}: unknown fields or non-table")
    try:
        return RemoteTarget(**(_fallback_values(values) | values), name=name)
    except (TypeError, ValueError) as exc:
        raise store.ConfigError(f"invalid remote.targets.{name}: {exc}") from exc


def named_targets() -> dict[str, RemoteTarget]:
    return {name: _load_target(name, values) for name, values in _target_table().items()}


def selected_name() -> str | None:
    value = os.environ.get("PYRITE_REMOTE_TARGET")
    if value is None:
        value = _remote_table(store._read_store()).get("default_target")
    if value is not None:
        validate_name(value)
    return value


def resolve_target(target: str | RemoteTarget | None = None) -> RemoteTarget:
    """Resolve explicit value/name > selected name > environment > flat keys > defaults.

    Stored named fields override per-field environment variables; omitted fields
    fall back to environment, flat keys, then defaults. Unknown names raise ``ValueError``.
    """
    if isinstance(target, RemoteTarget):
        return target
    if target is not None and not isinstance(target, str):
        raise TypeError("target must be a name, RemoteTarget or None")
    name = target if target is not None else selected_name()
    if name is not None:
        validate_name(name)
        try:
            return _load_target(name, _target_table()[name])
        except KeyError as exc:
            raise ValueError(
                f"unknown remote target {name!r}; use pyrite remote target list"
            ) from exc
    return RemoteTarget(host=config._legacy_remote_host(), **_fallback_values())


def _fallback_values(exclude=()):
    return {
        "remote_dir": os.environ.get("PYRITE_REMOTE_DIR", config.REMOTE_DIR),
        "uv": os.environ.get("PYRITE_REMOTE_UV", config.REMOTE_UV),
        **{
            field: store.resolve(f"remote.{field}", getattr(config, override)).value
            for field, override in (
                ("gpu_vendor", "REMOTE_GPU_VENDOR"),
                ("partition", "SLURM_PARTITION"),
                ("nodelist", "SLURM_NODELIST"),
                ("gres", "SLURM_GRES"),
            )
            if field not in exclude
        },
    }


def save_target(name: str, values: dict, *, edit: bool = False) -> RemoteTarget:
    validate_name(name)
    targets = named_targets()
    if edit != (name in targets):
        raise ValueError(f"target {name!r} {'does not exist' if edit else 'already exists'}")
    target = (
        replace(targets[name], **values)
        if edit
        else RemoteTarget(**(_fallback_values(values) | values), name=name)
    )
    document = store._load_document()
    table = document.setdefault("remote", {}).setdefault("targets", {})
    table[name] = (dict(table[name]) | values) if edit else values
    store._write_document(document)
    return target


def default_target(name: str) -> None:
    resolve_target(name)
    document = store._load_document()
    document.setdefault("remote", {})["default_target"] = name
    store._write_document(document)


def remove_target(name: str) -> None:
    resolve_target(name)
    document = store._load_document()
    remote = document["remote"]
    del remote["targets"][name]
    if remote.get("default_target") == name:
        del remote["default_target"]
    store._write_document(document)


def remember_job(jobid: str) -> None:
    """Persist target provenance locally, so job commands need no host search."""
    target = config._selected_target()
    if target is None:
        return
    document = store._load_document()
    records = document.setdefault("remote_jobs", {}).setdefault(jobid, {})
    key = target.name or "_explicit"
    records[key] = {k: v for k, v in asdict(target).items() if v is not None}
    store._write_document(document)


def job_target(jobid: str) -> RemoteTarget | None:
    table = store._read_store().get("remote_jobs", {})
    if not isinstance(table, dict):
        raise store.ConfigError("remote_jobs must be a table")
    records = table.get(jobid, {})
    if not isinstance(records, dict):
        raise store.ConfigError(f"remote_jobs.{jobid} must be a table")
    if len(records) > 1:
        raise ValueError(f"job {jobid!r} exists on multiple targets; specify --target")
    return RemoteTarget(**next(iter(records.values()))) if records else None


@contextmanager
def selected(target=None, *, jobid=None):
    if target is None and jobid is not None and os.environ.get("PYRITE_REMOTE_TARGET") is None:
        target = job_target(jobid)
    if target is None:
        yield
    else:
        with config.use_target(target):
            yield
