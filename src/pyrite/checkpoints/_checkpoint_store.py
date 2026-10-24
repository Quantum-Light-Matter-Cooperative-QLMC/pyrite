"""Component-aware checkpoint storage.

Active checkpoints live at ``<root>/<stem>/{line,brem,characteristic}.h5``,
plus ``channeling.h5`` once any record carries ``spec_channeling`` (ADR-0018 P5)
and ``transition.h5`` once any carries ``spec_transition`` (#103); stores
without them keep the historical three-file layout. Callers keep
using the historical merged ``{name: {E0: record}}`` in-memory shape; this
module splits records on write and merges them on read.  A legacy
``<root>/<stem>.pkl`` remains readable, so migration happens on the next save.

Validation: characteristic-radiation
"""

import copy
import hashlib
import os
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import _checkpoint_io

COMPONENTS = ("line", "brem", "characteristic", "channeling", "transition")
#: Components whose artifact is written only when non-empty or already present.
OPTIONAL_COMPONENTS = frozenset({"channeling", "transition"})
_HDF5_SUFFIX = ".h5"
_LEGACY_SUFFIX = ".pkl"
_BREM_KEYS = frozenset({"brem", "E_grid_brem", "brem_wide"})
_BREM_CASE_KEYS = frozenset({"Ne_brem", "E_grid_brem", "brem_chunk", "brem_file"})
#: Line-grid component key carried by each separately stored component.
_LINE_COMPONENT_KEY = {
    "characteristic": "spec_characteristic",
    "channeling": "spec_channeling",
    "transition": "spec_transition",
}
_SHARED_KEYS = frozenset({"case", "E_grid", "scale", "adaptive_sampling"})


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
                    if key not in _BREM_KEYS and key not in _LINE_COMPONENT_KEY.values()
                }
            elif component == "brem":
                selected[energy] = {
                    key: value
                    for key, value in record.items()
                    if key in _BREM_KEYS or key in _SHARED_KEYS
                }
            else:
                component_key = _LINE_COMPONENT_KEY[component]
                if record.get(component_key) is None:
                    continue
                selected[energy] = {
                    key: value
                    for key, value in record.items()
                    if key == component_key or key in _SHARED_KEYS
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

    Components land in background, characteristic, channeling, transition, line order.
    Since line is the authoritative record index, interruption cannot expose a
    new line record without its matching independently stored companions. An
    optional component (:data:`OPTIONAL_COMPONENTS`) is skipped while it is
    empty and absent on disk.
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
                _dump_component(stem, root, component, _component_store(legacy, component))
    order = tuple(
        component
        for component in ("brem", "characteristic", "channeling", "transition", "line")
        if component in components
    )
    for component in order:
        _dump_component(stem, root, component, _component_store(results, component))


def _dump_component(stem: str, root: str | os.PathLike[str], component: str, store: dict) -> None:
    path = component_path(stem, component, root)
    if component in OPTIONAL_COMPONENTS and not store and not path.is_file():
        return
    _atomic_dump(path, store)


def _attach_line_component(
    record: dict, component: str, stored: dict | None, *, name: str, energy: object
) -> None:
    """Attach one separately stored line-grid component to its line record.

    Line artifacts omit these components; each is attached as its own array.
    A legacy line artifact that still carries a co-located component was
    already separated on load, so that one is kept.
    """
    key = _LINE_COMPONENT_KEY[component]
    if record.get(key) is not None or stored is None:
        return
    grid = stored.get("E_grid")
    if grid is not None and not np.array_equal(np.asarray(record.get("E_grid")), np.asarray(grid)):
        raise ValueError(
            f"{component} checkpoint grid does not match line grid for {name!r} at {energy!r} keV"
        )
    value = stored.get(key)
    if value is not None:
        record[key] = value


def _merge(cxr: dict, brem: dict, companions: dict[str, dict]) -> dict:
    merged = {}
    for name, by_energy in cxr.items():
        merged[name] = {}
        for energy, cxr_record in by_energy.items():
            record = dict(cxr_record)
            record["case"] = copy.deepcopy(cxr_record["case"])
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
            for component, store in companions.items():
                _attach_line_component(
                    record, component, store.get(name, {}).get(energy), name=name, energy=energy
                )
            merged[name][energy] = record
    return merged


def load(stem: str, root: str | os.PathLike[str]) -> dict:
    line_path = component_read_path(stem, "line", root)
    if line_path.is_file():
        cxr = _checkpoint_io.load(str(line_path))
        legacy = None
        brem_path = component_read_path(stem, "brem", root)
        if brem_path.is_file():
            brem = _checkpoint_io.load(str(brem_path))
        elif legacy_path(stem, root).is_file():
            legacy = _checkpoint_io.load(str(legacy_path(stem, root)))
            brem = _component_store(legacy, "brem")
        else:
            brem = {}
        companions = {}
        for component in ("characteristic", "channeling", "transition"):
            path = component_read_path(stem, component, root)
            if path.is_file():
                companions[component] = _checkpoint_io.load(str(path))
            elif legacy_path(stem, root).is_file():
                if legacy is None:
                    legacy = _checkpoint_io.load(str(legacy_path(stem, root)))
                companions[component] = _component_store(legacy, component)
            else:
                companions[component] = {}
        merged = _merge(cxr, brem, companions)
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


def load_selected(
    stem: str,
    root: str | os.PathLike[str],
    cases: set[tuple[str, float]],
    *,
    components: tuple[str, ...] = COMPONENTS,
) -> dict:
    """Load selected cases and emission components, with shard precedence.

    ``line`` is always the authoritative record index. Requested companions
    attach to those records using the same grid validation as :func:`load`.
    Unselected companions are absent, including in crash-recovery shards.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown checkpoint component(s): {sorted(unknown)}")
    if not cases:
        return {}
    if not checkpoint_exists(stem, root):
        raise FileNotFoundError(component_read_path(stem, "line", root))
    old = legacy_path(stem, root)

    def component_records(component):
        path = component_read_path(stem, component, root)
        if path.is_file():
            return _component_store(_checkpoint_io.load_selected(str(path), cases), component)
        if old.is_file():
            return _component_store(_checkpoint_io.load_selected(str(old), cases), component)
        return {}

    line = component_records("line")
    brem = component_records("brem") if "brem" in components else {}
    companions = {
        component: component_records(component)
        for component in _LINE_COMPONENT_KEY
        if component in components
    }
    merged = _merge(line, brem, companions)
    # Shard filenames identify a whole config; select its energies before decoding.
    for name in sorted({name for name, _ in cases}):
        path = _part_path(stem, root, name)
        if not path.is_file():
            path = path.with_suffix(_LEGACY_SUFFIX)
        if path.is_file():
            shard = _checkpoint_io.load_selected(str(path), cases)
            selected_line = _component_store(shard, "line")
            selected_brem = _component_store(shard, "brem") if "brem" in components else {}
            selected_companions = {
                component: _component_store(shard, component) for component in companions
            }
            merged = _overlay(merged, _merge(selected_line, selected_brem, selected_companions))
    return merged


def load_metadata(stem: str, root: str | os.PathLike[str]) -> dict:
    """Recover the authoritative case index and overlay crash-safety shards."""
    path = component_read_path(stem, "line", root)
    if not path.is_file():
        path = legacy_path(stem, root)
    records = _checkpoint_io.load_metadata(str(path)) if path.is_file() else {}
    for shard in _part_paths(parts_dir(stem, root)):
        records = _overlay(records, _checkpoint_io.load_metadata(str(shard)))
    return records


def save_streaming(stem: str, root: str | os.PathLike[str], metadata: dict) -> None:
    """Stage every component before publication, loading at most one config.

    Shards remain recovery authority until the caller publishes its manifest.
    No source component is replaced while a staged component is still reading it.
    """
    order = ("brem", "characteristic", "channeling", "transition", "line")
    staged = []
    directory = checkpoint_dir(stem, root)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            writers = {}
            for component in order:
                target = component_path(stem, component, root)
                temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                staged.append((temporary, target, component))
                writers[component] = stack.enter_context(
                    _checkpoint_io.config_writer(list(metadata), str(temporary))
                )
            for name, energies in metadata.items():
                keys = {(name, energy) for energy in energies}
                records = load_selected(stem, root, keys)
                for component, writer in writers.items():
                    writer.add(name, _component_store(records, component).get(name, {}))
                del records
        for temporary, target, component in staged:
            if (
                component in OPTIONAL_COMPONENTS
                and not writers[component].keys
                and not target.is_file()
            ):
                temporary.unlink()
            else:
                os.replace(temporary, target)
    finally:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)
