"""Component-aware checkpoint storage.

Active checkpoints live at ``<root>/<stem>/{line,brem,characteristic}.h5``. Callers keep
using the historical merged ``{name: {E0: record}}`` in-memory shape; this
module splits records on write and merges them on read.  A legacy
``<root>/<stem>.pkl`` remains readable, so migration happens on the next save.
"""

import copy
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from . import _checkpoint_io

COMPONENTS = ("line", "brem", "characteristic")
_HDF5_SUFFIX = ".h5"
_LEGACY_SUFFIX = ".pkl"
_BREM_KEYS = frozenset({"brem", "E_grid_brem", "brem_wide"})
_BREM_CASE_KEYS = frozenset({"Ne_brem", "E_grid_brem", "brem_chunk", "brem_file"})
_CHARACTERISTIC_KEYS = frozenset({"spec_characteristic"})


def checkpoint_dir(stem: str, root: str | os.PathLike[str]) -> Path:
    return Path(root) / stem


def legacy_path(stem: str, root: str | os.PathLike[str]) -> Path:
    return Path(root) / f"{stem}.pkl"


def component_path(stem: str, component: str, root: str | os.PathLike[str]) -> Path:
    if component not in COMPONENTS:
        raise ValueError(f"unknown checkpoint component: {component!r}")
    return checkpoint_dir(stem, root) / f"{component}{_HDF5_SUFFIX}"


def _legacy_component_path(stem: str, component: str, root: str | os.PathLike[str]) -> Path:
    return component_path(stem, component, root).with_suffix(_LEGACY_SUFFIX)


def component_read_path(stem: str, component: str, root: str | os.PathLike[str]) -> Path:
    """Return the preferred new component path, falling back to legacy ``.pkl``."""
    path = component_path(stem, component, root)
    return path if path.is_file() else _legacy_component_path(stem, component, root)


def manifest_path(stem: str, root: str | os.PathLike[str]) -> Path:
    return checkpoint_dir(stem, root) / "meta.json"


def checkpoint_exists(stem: str, root: str | os.PathLike[str]) -> bool:
    return (
        component_read_path(stem, "line", root).is_file()
        or legacy_path(stem, root).is_file()
        or has_parts(stem, root)
    )


def discover(root: str | os.PathLike[str]) -> list[str]:
    """Return active checkpoint stems, including legacy monoliths."""
    base = Path(root)
    if not base.is_dir():
        return []
    stems = {
        path.name
        for path in base.iterdir()
        if path.is_dir()
        and (
            component_read_path(path.name, "line", base).is_file()
            or any((path / "parts").glob("*.h5"))
            or any((path / "parts").glob("*.pkl"))
        )
    }
    stems.update(path.stem for path in base.glob("*.pkl") if not path.name.endswith(".slim.pkl"))
    return sorted(stems)


def _component_store(results: dict, component: str) -> dict:
    out = {}
    for name, by_energy in results.items():
        selected = {}
        for energy, record in by_energy.items():
            if component == "line":
                selected[energy] = {
                    key: value
                    for key, value in record.items()
                    if key not in _BREM_KEYS and key not in _CHARACTERISTIC_KEYS
                }
            elif component == "brem":
                selected[energy] = {
                    key: value
                    for key, value in record.items()
                    if key in _BREM_KEYS or key in {"case", "E_grid", "scale"}
                }
            else:
                characteristic = record.get("spec_characteristic")
                if characteristic is None:
                    continue
                selected[energy] = {
                    key: value
                    for key, value in record.items()
                    if key in _CHARACTERISTIC_KEYS or key in {"case", "E_grid", "scale"}
                }
            selected[energy]["case"] = copy.deepcopy(record["case"])
        if selected:
            out[name] = selected
    return out


def parts_dir(stem: str, root: str | os.PathLike[str]) -> Path:
    """Directory holding per-config crash-safety shards during a live sweep.

    ``run_sweep`` writes one shard per finished config here instead of
    re-serializing the whole growing store on every save (which was O(N^2) in
    total bytes across a sweep).  Shards are the intermediate only: a normal or
    budget-stopped run consolidates them into the authoritative
    ``{line,brem,characteristic}.h5`` monoliths and clears this directory, so downstream tools
    (slim/prune/archive/remote) still see the historical layout.  A hard crash
    leaves shards behind; :func:`load_parts` recovers them on resume.
    """
    return checkpoint_dir(stem, root) / "parts"


def _part_path(stem: str, root: str | os.PathLike[str], name: str) -> Path:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return parts_dir(stem, root) / f"{digest}{_HDF5_SUFFIX}"


def _part_paths(directory: Path) -> list[Path]:
    """Return one path per shard, preferring a new-format name over `.pkl`."""
    paths = {path.stem: path for path in directory.glob(f"*{_LEGACY_SUFFIX}")}
    paths.update({path.stem: path for path in directory.glob(f"*{_HDF5_SUFFIX}")})
    return [paths[stem] for stem in sorted(paths)]


def save_part(
    stem: str,
    root: str | os.PathLike[str],
    name: str,
    record_map: dict,
) -> None:
    """Write one config's records (``{E0: record}``) as an immutable shard.

    Records never change once a config completes, so each shard is written
    exactly once -- the whole per-sweep write cost is O(N), not O(N^2).
    """
    _atomic_dump(_part_path(stem, root, name), {name: record_map})


def load_parts(stem: str, root: str | os.PathLike[str]) -> dict:
    """Merge every crash-safety shard back into a ``{name: {E0: record}}`` store."""
    directory = parts_dir(stem, root)
    if not directory.is_dir():
        return {}
    merged: dict = {}
    for shard in _part_paths(directory):
        merged.update(_checkpoint_io.load(str(shard)))
    return merged


def has_parts(stem: str, root: str | os.PathLike[str]) -> bool:
    directory = parts_dir(stem, root)
    return directory.is_dir() and bool(_part_paths(directory))


def clear_parts(stem: str, root: str | os.PathLike[str]) -> None:
    shutil.rmtree(parts_dir(stem, root), ignore_errors=True)


def parts_signature(stem: str, root: str | os.PathLike[str]) -> tuple:
    """Newest-shard stat, so manifest/load caches invalidate as shards land."""
    directory = parts_dir(stem, root)
    if not directory.is_dir():
        return ()
    shards = _part_paths(directory)
    if not shards:
        return ()
    newest = max(shards, key=lambda p: p.stat().st_mtime_ns)
    stat = newest.stat()
    return ((str(newest.resolve()), stat.st_mtime_ns, stat.st_size),)


def _atomic_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    _checkpoint_io.dump(value, str(tmp))
    os.replace(tmp, path)


def save(
    stem: str,
    root: str | os.PathLike[str],
    results: dict,
    *,
    components: tuple[str, ...] = COMPONENTS,
) -> None:
    """Atomically replace selected component artifacts.

    Components land in background, characteristic, line order. Since line is
    the authoritative record index, interruption cannot expose a new line
    record without its matching independently stored companions.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown checkpoint component(s): {sorted(unknown)}")
    old = legacy_path(stem, root)
    if old.is_file():
        legacy = None
        for component in COMPONENTS:
            if (
                component not in components
                and not component_read_path(stem, component, root).is_file()
            ):
                legacy = _checkpoint_io.load(str(old)) if legacy is None else legacy
                _atomic_dump(
                    component_path(stem, component, root),
                    _component_store(legacy, component),
                )
    order = tuple(
        component for component in ("brem", "characteristic", "line") if component in components
    )
    for component in order:
        _atomic_dump(component_path(stem, component, root), _component_store(results, component))


def _merge(line: dict, brem: dict, characteristic: dict) -> dict:
    merged = {}
    for name, by_energy in line.items():
        merged[name] = {}
        for energy, line_record in by_energy.items():
            record = dict(line_record)
            record["case"] = copy.deepcopy(line_record["case"])
            brem_record = brem.get(name, {}).get(energy)
            if brem_record is not None:
                for key in _BREM_KEYS:
                    if key in brem_record:
                        record[key] = brem_record[key]
                for key in _BREM_CASE_KEYS:
                    if key in brem_record.get("case", {}):
                        record["case"][key] = brem_record["case"][key]
                wide_grid = record.get("E_grid_brem")
                wide = record.get("brem_wide")
                if wide_grid is not None and wide is not None:
                    record["brem"] = np.interp(
                        np.asarray(record["E_grid"], float),
                        np.asarray(wide_grid, float),
                        np.asarray(wide, float),
                    )
            characteristic_record = characteristic.get(name, {}).get(energy)
            # Line artifacts omit characteristic radiation; attach it as its own
            # array. A legacy line artifact that still carries a co-located
            # component was already separated on load, so keep that one.
            if record.get("spec_characteristic") is None and characteristic_record is not None:
                characteristic_grid = characteristic_record.get("E_grid")
                if characteristic_grid is not None and not np.array_equal(
                    np.asarray(record.get("E_grid")), np.asarray(characteristic_grid)
                ):
                    raise ValueError(
                        f"characteristic checkpoint grid does not match line grid for {name!r} "
                        f"at {energy!r} keV"
                    )
                component = characteristic_record.get("spec_characteristic")
                if component is not None:
                    record["spec_characteristic"] = component
            merged[name][energy] = record
    return merged


def load(stem: str, root: str | os.PathLike[str]) -> dict:
    line_path = component_read_path(stem, "line", root)
    if line_path.is_file():
        line = _checkpoint_io.load(str(line_path))
        legacy = None
        brem_path = component_read_path(stem, "brem", root)
        if brem_path.is_file():
            brem = _checkpoint_io.load(str(brem_path))
        elif legacy_path(stem, root).is_file():
            legacy = _checkpoint_io.load(str(legacy_path(stem, root)))
            brem = _component_store(legacy, "brem")
        else:
            brem = {}
        characteristic_path = component_read_path(stem, "characteristic", root)
        if characteristic_path.is_file():
            characteristic = _checkpoint_io.load(str(characteristic_path))
        elif legacy_path(stem, root).is_file():
            legacy = _checkpoint_io.load(str(legacy_path(stem, root))) if legacy is None else legacy
            characteristic = _component_store(legacy, "characteristic")
        else:
            characteristic = {}
        merged = _merge(line, brem, characteristic)
        return _overlay(merged, load_parts(stem, root))
    old = legacy_path(stem, root)
    if old.is_file():
        return _overlay(_checkpoint_io.load(str(old)), load_parts(stem, root))
    if has_parts(stem, root):
        return load_parts(stem, root)
    raise FileNotFoundError(line_path)


def _overlay(base: dict, newer: dict) -> dict:
    """Return the record union with ``newer`` winning per ``(name, energy)``."""
    merged = {name: dict(by_energy) for name, by_energy in base.items()}
    for name, by_energy in newer.items():
        merged.setdefault(name, {}).update(by_energy)
    return merged


def signature(stem: str, root: str | os.PathLike[str]) -> tuple:
    """Stable cache key covering every active component or legacy monolith."""
    paths = [component_read_path(stem, component, root) for component in COMPONENTS]
    present = [path for path in paths if path.is_file()]
    if not present:
        old = legacy_path(stem, root)
        present = [old] if old.is_file() else []
    component_signature = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in present
    )
    return component_signature + parts_signature(stem, root)


# ---- per-case content-addressable store (CAS) --------------------------------
# A shared per-material store of single-case blobs, addressed by
# :func:`pyrite.campaign.profiles.case_content_key` and sharded git-style by the first two
# hex chars of the key (256 buckets -> bounded directory sizes). One blob holds
# one case's raw transport ``out`` dict, so a case computed by any profile can be
# replayed (``store_result``) by any other profile whose case hashes equal. The
# blobs live under the per-material directory alongside its component store; the
# 2-hex shard names never collide with the named component files or ``meta.json``
# and are invisible to :func:`discover`.


def _validate_content_key(content_key: str) -> str:
    if not isinstance(content_key, str) or len(content_key) != 64:
        raise ValueError("content key must be a 64-character SHA-256 hex digest")
    try:
        int(content_key, 16)
    except ValueError as exc:
        raise ValueError("content key must be a 64-character SHA-256 hex digest") from exc
    return content_key.lower()


def cas_blob_path(material: str, content_key: str, root: str | os.PathLike[str]) -> Path:
    """Sharded blob path ``<root>/<material>/<first2hex>/<content_key>.h5``."""
    content_key = _validate_content_key(content_key)
    return Path(root) / material / content_key[:2] / f"{content_key}{_HDF5_SUFFIX}"


def _cas_read_path(material: str, content_key: str, root: str | os.PathLike[str]) -> Path:
    path = cas_blob_path(material, content_key, root)
    return path if path.is_file() else path.with_suffix(_LEGACY_SUFFIX)


def cas_contains(material: str, content_key: str, root: str | os.PathLike[str]) -> bool:
    """Whether a case blob for ``content_key`` exists -- a single ``stat``, no load."""
    return _cas_read_path(material, content_key, root).is_file()


def cas_load(material: str, content_key: str, root: str | os.PathLike[str]) -> dict:
    """Load one case's stored transport ``out`` dict from the CAS."""
    return _checkpoint_io.load(str(_cas_read_path(material, content_key, root)))


def cas_save(
    material: str, content_key: str, root: str | os.PathLike[str], payload: object
) -> None:
    """Atomically write one case's transport ``out`` dict into the CAS.

    Write-once-by-content: the same key always names the same physics, so the
    atomic temp-file + ``os.replace`` (:func:`_atomic_dump`) makes concurrent
    same-material writers touch distinct blobs and never clobber a monolith.
    """
    _atomic_dump(cas_blob_path(material, content_key, root), payload)
