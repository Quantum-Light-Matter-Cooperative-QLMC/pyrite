"""Opt-in HDF5 artifacts of a case's electron-transport result.

A run that requests trajectory capture hands the exact
:func:`~pyrite.montecarlo.transport.simulate_trajectories` mapping its case
transported -- before the spectrum phase consumes it -- to
:meth:`TrajectoryCapture.write`. Nothing is re-transported, no random draw is
consumed, and the mapping is never mutated, so spectra are identical with and
without capture. Capture is off unless a caller passes a
:class:`TrajectoryCapture`; ordinary runs never import :mod:`h5py` here.

Layout (``schema_version`` 1)::

    /                   attrs: format, schema_version, complete, created_utc,
                               pyrite_version, case_name, E0_keV, seed,
                               case_sha256, parameter_sha256, fields (JSON),
                               segment_count
    /case               attrs: json -- the resolved case mapping
    /provenance         attrs: json -- run-level identity (stem, profile, ...)
    /settings           attrs: json -- resolved transport controls
                        E_cut_by_electrons  (Ne_transport,) keV
    /transport/<field>  one node per returned key, insertion order kept

Each ``/transport`` node carries a ``pyrite_kind`` attribute so readback
restores the original Python structure: ``array`` datasets keep dtype, shape,
and row order; ``dict``/``list``/``tuple`` groups nest; scalars keep their
Python or NumPy type; ``none`` marks ``None``. A second key naming the SAME
array object (``E_keV`` is ``E_start_keV``) is stored as an HDF5 hard link, so
compatibility aliases cost no extra space. Known per-field units ride on each
dataset's ``units`` attribute; see ``docs/physics/beam-transport/
transport-outputs.md`` for field semantics.

Completion is atomic: the file is written to ``<name>.partial`` and renamed
onto its final path only after ``complete`` is set and the file is closed, so
an interrupted write can never be read as a finished artifact.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._backend import _to_cpu, is_device_array

FORMAT = "pyrite.transport-trajectories"
SCHEMA_VERSION = 1
PARTIAL_SUFFIX = ".partial"

# Device arrays are downloaded in bounded row blocks rather than as one host
# copy of the whole segment table.
_DEVICE_BLOCK_BYTES = 64 * 1024 * 1024

# Units of known transport fields. Anything else falls back to its suffix
# (_ang, _keV, _eV, _mm, _rad); identifiers, codes, counts, and unit vectors
# are dimensionless and carry "1".
_EXPLICIT_UNITS = {
    "r_mid": "angstrom",
    "v_hat": "1",
    "hard_secondary_v_hat": "1",
    "hard_radiative_direction": "1",
    "hard_radiative_target_momentum_eV_c": "eV/c",
    "hard_radiative_Z": "1",
    "initial_v_hat": "1",
    "elec_id": "1",
    "electron_id": "1",
    "vacuum_elec_id": "1",
    "layer": "1",
    "flight_id": "1",
    "substep_id": "1",
    "event_kind": "1",
    "track_id": "1",
    "parent_id": "1",
    "generation": "1",
    "hard_channel": "1",
}
_SUFFIX_UNITS = (
    ("_ang", "angstrom"),
    ("_keV", "keV"),
    ("_eV", "eV"),
    ("_mm", "mm"),
    ("_rad", "rad"),
)


class TrajectoryArtifactError(RuntimeError):
    """A trajectory artifact is missing, incomplete, foreign, or mismatched."""


class TrajectoryArtifactExistsError(TrajectoryArtifactError, FileExistsError):
    """A complete artifact already occupies a path this run would write."""


def field_units(name: str) -> str | None:
    """Unit string for a top-level transport field, or ``None`` when unknown."""
    if name in _EXPLICIT_UNITS:
        return _EXPLICIT_UNITS[name]
    if name.startswith("n_") or name in {"Ne", "n_layers"}:
        return "1"
    for suffix, unit in _SUFFIX_UNITS:
        if name.endswith(suffix):
            return unit
    return None


def case_key(case: Mapping[str, Any]) -> str:
    """Resume key of one case: its configuration name and incident energy."""
    return json.dumps([str(case["name"]), float(case["E0_keV"])])


def artifact_relpath(case: Mapping[str, Any]) -> Path:
    """Deterministic artifact path of ``case`` relative to a capture root.

    Configuration names contain spaces, ``/``, and ``=``, so the directory is a
    readable slug plus a short digest of the exact name (two names that slug
    alike never share a directory); the file names the incident energy.
    """
    name = str(case["name"])
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "case"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    energy = format(float(case["E0_keV"]), ".12g")
    return Path(f"{slug[:80]}-{digest}") / f"E0_{energy}keV.h5"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_device_array(value):
        return _jsonable(_to_cpu(value))
    return value


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, default=repr)


def case_digest(case: Mapping[str, Any]) -> str:
    """SHA-256 of the resolved case's canonical JSON; identifies its physics."""
    return hashlib.sha256(_dumps(case).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryCapture:
    """Where and how a run writes per-case transport artifacts.

    Picklable, so it crosses into transport worker processes with the case.

    Parameters
    ----------
    root
        Directory holding this run's artifacts; each case writes
        ``root / artifact_relpath(case)``.
    overwrite
        Replace a complete artifact already at a case's path. False raises
        :class:`TrajectoryArtifactExistsError` instead.
    provenance
        JSON-safe run-level identity recorded in every artifact. A
        ``parameter_sha256`` entry also takes part in stale-artifact detection.
    """

    root: str
    overwrite: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def path_for(self, case: Mapping[str, Any]) -> Path:
        """Final artifact path of ``case``."""
        return Path(self.root) / artifact_relpath(case)

    def write(
        self,
        case: Mapping[str, Any],
        transport: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write ``case``'s transport result atomically; return its path."""
        return write_trajectory_artifact(
            self.path_for(case),
            transport,
            case=case,
            settings=settings,
            provenance=self.provenance,
            overwrite=self.overwrite,
        )


def _pyrite_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyrite-xray")
    except Exception:
        return "unknown"


def _write_array(group, name: str, value: Any, units: str | None):
    if is_device_array(value):
        dtype = np.dtype(value.dtype)
        dataset = group.create_dataset(name, shape=value.shape, dtype=dtype)
        if value.size:
            row_bytes = max(1, dtype.itemsize * int(np.prod(value.shape[1:], dtype=np.int64)))
            block = max(1, _DEVICE_BLOCK_BYTES // row_bytes)
            for start in range(0, value.shape[0], block):
                stop = min(start + block, value.shape[0])
                dataset[start:stop] = _to_cpu(value[start:stop])
    else:
        dataset = group.create_dataset(name, data=value)
    dataset.attrs["pyrite_kind"] = "array"
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def _write_node(group, name: str, value: Any, links: dict[int, str], units: str | None) -> None:
    if isinstance(value, np.ndarray) or is_device_array(value):
        # Aliases share one array object; link instead of duplicating rows.
        target = links.get(id(value))
        if target is not None:
            group[name] = group.file[target]
            return
        dataset = _write_array(group, name, value, units)
        links[id(value)] = dataset.name
        return
    if isinstance(value, Mapping):
        sub = group.create_group(name, track_order=True)
        sub.attrs["pyrite_kind"] = "dict"
        for key, item in value.items():
            if not isinstance(key, str) or not key or "/" in key or key in {".", ".."}:
                raise TypeError(f"trajectory field {sub.name!r} has unsupported key {key!r}")
            _write_node(sub, key, item, links, None)
        return
    if isinstance(value, (list, tuple)):
        sub = group.create_group(name, track_order=True)
        sub.attrs["pyrite_kind"] = "tuple" if isinstance(value, tuple) else "list"
        sub.attrs["length"] = len(value)
        for index, item in enumerate(value):
            _write_node(sub, str(index), item, links, None)
        return
    if value is None:
        dataset = group.create_dataset(name, data=np.zeros(0, dtype=np.uint8))
        dataset.attrs["pyrite_kind"] = "none"
        return
    if isinstance(value, np.generic):
        dataset = group.create_dataset(name, data=value)
        dataset.attrs["pyrite_kind"] = "numpy_scalar"
    elif isinstance(value, bool):
        dataset = group.create_dataset(name, data=np.bool_(value))
        dataset.attrs["pyrite_kind"] = "bool"
    elif isinstance(value, int):
        dataset = group.create_dataset(name, data=np.int64(value))
        dataset.attrs["pyrite_kind"] = "int"
    elif isinstance(value, float):
        dataset = group.create_dataset(name, data=np.float64(value))
        dataset.attrs["pyrite_kind"] = "float"
    elif isinstance(value, str):
        dataset = group.create_dataset(name, data=value)
        dataset.attrs["pyrite_kind"] = "str"
    else:
        raise TypeError(
            f"trajectory field {group.name}/{name} has unsupported type {type(value).__name__}"
        )
    if units is not None:
        dataset.attrs["units"] = units


def _segment_count(transport: Mapping[str, Any]) -> int:
    rows = transport.get("electron_id", transport.get("elec_id"))
    return 0 if rows is None else int(rows.shape[0])


def write_trajectory_artifact(
    path: str | os.PathLike[str],
    transport: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    settings: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one complete transport result to ``path`` atomically.

    Parameters
    ----------
    path
        Final artifact path; parent directories are created.
    transport
        Mapping returned by ``simulate_trajectories``. Host arrays are written
        in place; device-resident arrays are downloaded in bounded row blocks.
        Not mutated.
    case
        Resolved case mapping that produced ``transport``.
    settings
        Resolved transport controls. An ``E_cut_by_electrons`` array is stored
        as a dataset; everything else as JSON.
    provenance
        JSON-safe run-level identity.
    overwrite
        Replace an existing complete artifact at ``path``.

    Returns
    -------
    Path
        The final artifact path.

    Raises
    ------
    TrajectoryArtifactExistsError
        If ``path`` exists and ``overwrite`` is false.
    TypeError
        If ``transport`` holds a value the schema cannot represent.
    """
    import h5py

    final = Path(path)
    if final.exists() and not overwrite:
        raise TrajectoryArtifactExistsError(f"trajectory artifact already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + PARTIAL_SUFFIX)
    settings = dict(settings or {})
    cutoffs = settings.pop("E_cut_by_electrons", None)
    try:
        with h5py.File(partial, "w", track_order=True) as handle:
            handle.attrs["format"] = FORMAT
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["complete"] = False
            handle.attrs["created_utc"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            handle.attrs["pyrite_version"] = _pyrite_version()
            handle.attrs["case_name"] = str(case["name"])
            handle.attrs["E0_keV"] = float(case["E0_keV"])
            handle.attrs["seed"] = int(case.get("seed", 0))
            handle.attrs["case_sha256"] = case_digest(case)
            handle.attrs["fields"] = json.dumps(list(transport))
            handle.attrs["segment_count"] = _segment_count(transport)
            handle.create_group("case").attrs["json"] = _dumps(case)
            handle.create_group("provenance").attrs["json"] = _dumps(provenance or {})
            handle.attrs["parameter_sha256"] = str((provenance or {}).get("parameter_sha256", ""))
            settings_group = handle.create_group("settings")
            settings_group.attrs["json"] = _dumps(settings)
            if cutoffs is not None:
                _write_array(settings_group, "E_cut_by_electrons", np.asarray(cutoffs), "keV")
            group = handle.create_group("transport", track_order=True)
            group.attrs["pyrite_kind"] = "dict"
            links: dict[int, str] = {}
            for name, value in transport.items():
                _write_node(group, name, value, links, field_units(name))
            handle.attrs["complete"] = True
        os.replace(partial, final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return final


def _read_node(node) -> Any:
    kind = node.attrs.get("pyrite_kind", "array")
    if kind == "dict":
        return {key: _read_node(node[key]) for key in node}
    if kind in {"list", "tuple"}:
        items = [_read_node(node[str(index)]) for index in range(int(node.attrs["length"]))]
        return tuple(items) if kind == "tuple" else items
    if kind == "none":
        return None
    if kind == "array":
        return node[()]
    value = node[()]
    if kind == "numpy_scalar":
        return value
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    raise TrajectoryArtifactError(f"unknown trajectory node kind {kind!r} at {node.name}")


def _check_header(handle, path: Path) -> None:
    if handle.attrs.get("format") != FORMAT:
        raise TrajectoryArtifactError(f"not a PyRITE trajectory artifact: {path}")
    version = int(handle.attrs.get("schema_version", 0))
    if version > SCHEMA_VERSION or version < 1:
        raise TrajectoryArtifactError(
            f"unsupported trajectory schema_version {version} in {path}; "
            f"this PyRITE reads 1..{SCHEMA_VERSION}"
        )
    if not bool(handle.attrs.get("complete", False)):
        raise TrajectoryArtifactError(f"incomplete trajectory artifact: {path}")


@dataclass(frozen=True)
class TrajectoryArtifact:
    """One reopened trajectory artifact.

    ``transport`` reproduces the ``simulate_trajectories`` mapping (NumPy
    arrays on the host); ``units`` maps each top-level field to its unit where
    known.
    """

    path: Path
    schema_version: int
    case: dict[str, Any]
    provenance: dict[str, Any]
    settings: dict[str, Any]
    transport: dict[str, Any]
    units: dict[str, str]
    attrs: dict[str, Any]


def _header_attrs(handle) -> dict[str, Any]:
    attrs = {}
    for key, value in handle.attrs.items():
        attrs[key] = value.item() if isinstance(value, np.generic) else value
    attrs["fields"] = json.loads(attrs.get("fields", "[]"))
    return attrs


def read_trajectory_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a complete artifact's root attributes without loading arrays."""
    import h5py

    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise TrajectoryArtifactError(f"unreadable trajectory artifact {path}: {error}") from error
    with handle:
        _check_header(handle, path)
        return _header_attrs(handle)


def read_trajectory_artifact(path: str | os.PathLike[str]) -> TrajectoryArtifact:
    """Reopen a complete trajectory artifact.

    Raises
    ------
    TrajectoryArtifactError
        If the file is not a complete artifact of a supported schema version.
    """
    import h5py

    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise TrajectoryArtifactError(f"unreadable trajectory artifact {path}: {error}") from error
    with handle:
        _check_header(handle, path)
        settings = json.loads(handle["settings"].attrs["json"])
        if "E_cut_by_electrons" in handle["settings"]:
            settings["E_cut_by_electrons"] = handle["settings/E_cut_by_electrons"][()]
        group = handle["transport"]
        units = {
            key: str(group[key].attrs["units"]) for key in group if "units" in group[key].attrs
        }
        return TrajectoryArtifact(
            path=path,
            schema_version=int(handle.attrs["schema_version"]),
            case=json.loads(handle["case"].attrs["json"]),
            provenance=json.loads(handle["provenance"].attrs["json"]),
            settings=settings,
            transport=_read_node(group),
            units=units,
            attrs=_header_attrs(handle),
        )


@dataclass(frozen=True)
class CapturePlan:
    """Outcome of :func:`preflight_capture` for one run."""

    to_write: int
    replaced: int
    kept: int
    missing_cached: int


def preflight_capture(
    capture: TrajectoryCapture,
    cases: Iterable[Mapping[str, Any]],
    todo: Iterable[Mapping[str, Any]],
) -> CapturePlan:
    """Check existing artifacts before any case transports.

    ``cases`` is the whole requested set; ``todo`` the subset this run will
    transport (the rest were resumed or replayed from a cache and are never
    re-transported, so they cannot gain an artifact). Stale ``.partial`` files
    of to-run cases are removed.

    Raises
    ------
    TrajectoryArtifactExistsError
        A to-run case already has a complete artifact and ``overwrite`` is off.
    TrajectoryArtifactError
        An existing file is not a complete artifact, or a cached case's file
        records a different resolved case or run parameter digest.
    """
    todo_keys = {case_key(case) for case in todo}
    expected_parameters = capture.provenance.get("parameter_sha256")
    to_write = replaced = kept = missing_cached = 0
    existing_to_run = []
    for case in cases:
        key = case_key(case)
        path = capture.path_for(case)
        will_run = key in todo_keys
        if will_run:
            to_write += 1
            path.with_name(path.name + PARTIAL_SUFFIX).unlink(missing_ok=True)
        if not path.exists():
            if not will_run:
                missing_cached += 1
            continue
        header = read_trajectory_header(path)
        stored_parameters = header.get("parameter_sha256") or None
        mismatched = header.get("case_sha256") != case_digest(case) or (
            expected_parameters is not None
            and stored_parameters is not None
            and stored_parameters != expected_parameters
        )
        if will_run:
            if not capture.overwrite:
                existing_to_run.append(path)
            replaced += 1
        elif mismatched:
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} records different physics than the requested "
                "case; remove it or choose another trajectory directory"
            )
        else:
            kept += 1
    if existing_to_run:
        shown = ", ".join(str(path) for path in existing_to_run[:3])
        more = "" if len(existing_to_run) <= 3 else f" (+{len(existing_to_run) - 3} more)"
        raise TrajectoryArtifactExistsError(
            f"{len(existing_to_run)} trajectory artifact(s) already exist for cases this run "
            f"will transport: {shown}{more}; pass --overwrite-trajectories to replace them"
        )
    return CapturePlan(
        to_write=to_write, replaced=replaced, kept=kept, missing_cached=missing_cached
    )
