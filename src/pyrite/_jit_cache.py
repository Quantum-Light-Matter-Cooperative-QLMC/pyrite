"""PyRITE-managed JIT compilation caches for Numba and CuPy.

Numba writes ``@njit(cache=True)`` overloads under ``NUMBA_CACHE_DIR``; CuPy
writes NVRTC output under ``CUPY_CACHE_DIR``. Both caches are native: this
module never compiles and never builds a binary layer on top of them.

:func:`apply_managed_cache_env` points *unset* (or empty) cache variables at
:func:`pyrite.paths.cache_dir` subdirectories (``~/.cache/pyrite/numba`` and
``~/.cache/pyrite/cupy`` on Linux). ``import pyrite`` calls it, as do the
``pyrite`` CLI and remote ``python -m`` shims, before any PyRITE kernel module
is decorated. Numba reads ``NUMBA_CACHE_DIR`` into ``numba.config.CACHE_DIR``
when it is first imported, so when Numba is already loaded the config value is
set too; each dispatcher consults ``config.CACHE_DIR`` when it enables caching,
so this is early enough for every PyRITE kernel. CuPy reads ``CUPY_CACHE_DIR``
lazily at each kernel compile. A non-empty user value always wins. The
in-tree ``__pycache__`` Numba location and CuPy's ``~/.cupy`` default are
legacy locations: still reported and (for in-tree PyRITE entries) clearable.

The inspection half (:func:`cache_locations`, :func:`numba_index_entries`,
:func:`clear_plan`) reads cache files without importing CuPy; Numba is imported
only to decode its own index format. The runtime half
(:class:`NumbaCompileRecorder`, :func:`numba_dispatcher_stats`) attributes
compilation and cache hits/misses inside one process for diagnostics.
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from .paths import cache_dir

NUMBA_ENV = "NUMBA_CACHE_DIR"
CUPY_ENV = "CUPY_CACHE_DIR"
_ENV = {"numba": NUMBA_ENV, "cupy": CUPY_ENV}
_PACKAGE_ROOT = Path(__file__).resolve().parent
_NUMBA_SUFFIXES = (".nbi", ".nbc")


def managed_dir(backend: str) -> Path:
    """Return PyRITE's managed cache directory for ``numba`` or ``cupy``."""
    if backend not in _ENV:
        raise ValueError(f"unknown JIT cache backend {backend!r}; expected numba or cupy")
    return cache_dir() / backend


def apply_managed_cache_env(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Point unset JIT cache variables at PyRITE's managed directories.

    Call before any PyRITE kernel module is decorated. Existing non-empty
    values are kept. When writing ``os.environ`` and Numba is already imported
    (its config froze the old value), ``numba.config.CACHE_DIR`` is updated too;
    Numba is never imported here. Returns the variables this call set.
    """
    env = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for backend, name in _ENV.items():
        if env.get(name):
            continue
        applied[name] = env[name] = str(managed_dir(backend))
    numba_config = sys.modules.get("numba.core.config")
    if env is os.environ and NUMBA_ENV in applied and numba_config is not None:
        if not getattr(numba_config, "CACHE_DIR", ""):
            setattr(numba_config, "CACHE_DIR", applied[NUMBA_ENV])  # noqa: B010
    return applied


# --------------------------------------------------------------------------- locations


@dataclass(frozen=True)
class CacheLocation:
    """One directory a JIT backend reads and writes compiled kernels in.

    ``source`` is ``managed`` (PyRITE's directory), ``environment`` (a user's
    variable), ``in-tree`` (legacy: Numba's ``__pycache__`` beside PyRITE sources,
    used before managed defaults or when the variable is unset) or ``default`` (CuPy's own
    home-directory default). ``owned`` means every file under ``path`` is
    PyRITE's to delete; otherwise only PyRITE-attributable entries are.
    """

    backend: str
    path: Path
    source: str
    owned: bool
    active: bool


def _cupy_default_dir() -> Path:
    return Path.home() / ".cupy" / "kernel_cache"


def cache_locations(environ: MutableMapping[str, str] | None = None) -> list[CacheLocation]:
    """Return active Numba/CuPy cache directories plus Numba's in-tree fallback."""
    env = os.environ if environ is None else environ
    locations = []
    for backend, name in _ENV.items():
        value = env.get(name)
        managed = managed_dir(backend)
        if value:
            path = Path(value).expanduser()
            source = "managed" if path.resolve() == managed.resolve() else "environment"
        elif backend == "numba":
            path, source = _PACKAGE_ROOT, "in-tree"
        else:
            path, source = _cupy_default_dir(), "default"
        locations.append(CacheLocation(backend, path, source, source == "managed", True))
        if backend == "numba" and source != "in-tree":
            locations.append(CacheLocation("numba", _PACKAGE_ROOT, "in-tree", False, False))
    return locations


def _package_dirs() -> list[Path]:
    return sorted(
        {path.parent for path in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts}
    )


def numba_cache_subpath(source_dir: Path) -> str:
    """Return Numba's ``NUMBA_CACHE_DIR`` subdirectory name for one source directory.

    Mirrors ``numba.core.caching._CacheLocator.get_suitable_cache_subpath``;
    a test pins the two together.
    """
    subpath = os.path.abspath(source_dir)
    return "_".join([os.path.split(subpath)[-1], hashlib.sha1(subpath.encode()).hexdigest()])


def _family(source_dir: Path) -> str:
    relative = source_dir.relative_to(_PACKAGE_ROOT)
    return ".".join(("pyrite", *relative.parts))


def _numba_dirs(location: CacheLocation) -> Iterator[tuple[str, Path, Path]]:
    """Yield ``(family, source_dir, cache_dir)`` for each PyRITE package."""
    for source_dir in _package_dirs():
        if location.source == "in-tree":
            cache = source_dir / "__pycache__"
        else:
            cache = location.path / numba_cache_subpath(source_dir)
        yield _family(source_dir), source_dir, cache


def _numba_files(cache: Path) -> list[Path]:
    if not cache.is_dir():
        return []
    return sorted(p for p in cache.iterdir() if p.is_file() and p.suffix in _NUMBA_SUFFIXES)


@dataclass
class FamilyUsage:
    """Cached kernels of one PyRITE package (kernel family) in one location."""

    family: str
    files: int = 0
    bytes: int = 0
    indexes: int = 0
    current: int = 0
    stale: int = 0
    overloads: int = 0
    paths: list[Path] = field(default_factory=list)


def _source_stamps(source_dir: Path) -> set[object]:
    """Return every source stamp Numba may have recorded for ``source_dir``.

    Current Numba (0.68) stamps the SHA-256 digest of the source bytes; older releases
    stamp ``(st_mtime, st_size)``. Both forms are accepted.
    """
    stamps: set[object] = set()
    for source in source_dir.glob("*.py"):
        stat = source.stat()
        stamps.add((stat.st_mtime, stat.st_size))
        stamps.add(hashlib.sha256(source.read_bytes()).digest())
    return stamps


def numba_index_entries(index: Path) -> tuple[object, int] | None:
    """Return ``(source_stamp, overload_count)`` of one current-version index.

    ``None`` means the index is unreadable or was written by another Numba
    version, so Numba itself would ignore it (a guaranteed miss).
    """
    import pickle

    import numba

    try:
        with index.open("rb") as stream:
            version = pickle.load(stream)
            if version != numba.__version__:
                return None
            stamp, overloads = pickle.loads(stream.read())
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError, ImportError):
        return None
    return stamp, len(overloads)


def numba_usage(location: CacheLocation) -> list[FamilyUsage]:
    """Summarise PyRITE's Numba cache entries per kernel family in ``location``.

    An index is ``current`` when its recorded source stamp matches a source
    file of its package (content digest, or mtime/size for old Numba) and its
    Numba version matches: its signatures will be
    cache hits. Anything else is ``stale`` and will recompile (a miss).
    """
    usage = []
    for family, source_dir, cache in _numba_dirs(location):
        files = _numba_files(cache)
        if not files:
            continue
        entry = FamilyUsage(family, files=len(files), paths=files)
        entry.bytes = sum(p.stat().st_size for p in files)
        stamps = _source_stamps(source_dir)
        for index in (p for p in files if p.suffix == ".nbi"):
            entry.indexes += 1
            loaded = numba_index_entries(index)
            if loaded is not None and loaded[0] in stamps:
                entry.current += 1
                entry.overloads += loaded[1]
            else:
                entry.stale += 1
        usage.append(entry)
    return usage


def _tree_size(path: Path) -> tuple[int, int]:
    files = size = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return files, size


def location_summary(location: CacheLocation) -> dict:
    """Return a JSON-ready status record for one cache location."""
    record: dict = {
        "backend": location.backend,
        "path": str(location.path),
        "source": location.source,
        "active": location.active,
        "owned": location.owned,
        "exists": location.path.is_dir(),
    }
    if location.backend == "numba":
        families = numba_usage(location)
        record["families"] = [
            {
                "family": f.family,
                "files": f.files,
                "bytes": f.bytes,
                "indexes": f.indexes,
                "current": f.current,
                "stale": f.stale,
                "overloads": f.overloads,
            }
            for f in families
        ]
        record["pyrite_files"] = sum(f.files for f in families)
        record["pyrite_bytes"] = sum(f.bytes for f in families)
    if location.source != "in-tree":
        record["files"], record["bytes"] = _tree_size(location.path)
    return record


# --------------------------------------------------------------------------- clearing


@dataclass(frozen=True)
class ClearTarget:
    """One path ``cache clear`` would delete, or a location it leaves alone."""

    backend: str
    path: Path
    kind: str  # "tree", "file" or "skip"
    reason: str = ""


def clear_plan(locations: list[CacheLocation]) -> list[ClearTarget]:
    """Return exactly the PyRITE-owned paths to delete in ``locations``.

    Managed directories are removed whole. A user-chosen Numba directory and the
    in-tree caches lose only PyRITE package entries. A user or default CuPy
    directory is never touched: its hash-named files cannot be attributed.
    """
    targets = []
    for location in locations:
        if location.owned:
            if location.path.is_dir():
                targets.append(ClearTarget(location.backend, location.path, "tree"))
        elif location.backend == "numba":
            for _, _, cache in _numba_dirs(location):
                if location.source == "in-tree":
                    targets.extend(ClearTarget("numba", p, "file") for p in _numba_files(cache))
                elif cache.is_dir():
                    targets.append(ClearTarget("numba", cache, "tree"))
        elif location.path.is_dir():
            targets.append(
                ClearTarget(
                    location.backend,
                    location.path,
                    "skip",
                    "not PyRITE-owned; remove manually if no other CuPy user needs it",
                )
            )
    return targets


def execute_clear(targets: list[ClearTarget]) -> list[ClearTarget]:
    """Delete non-skip targets; return those that were removed."""
    import shutil

    removed = []
    for target in targets:
        if target.kind == "tree" and target.path.is_dir():
            shutil.rmtree(target.path)
        elif target.kind == "file" and target.path.is_file():
            target.path.unlink()
        else:
            continue
        removed.append(target)
    return removed


# --------------------------------------------------------------------------- runtime diagnostics


def _module_source_digest(module_name: str) -> str:
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if not path:
        return f"{module_name}:<no source>"
    return f"{module_name}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def stabilize_closure_cache(closure, captured, key: str) -> None:
    """Key a ``cache=True`` closure's disk cache on ``key`` instead of its cells.

    Numba keys a closure's cache on its pickled closure cells. A captured Numba
    dispatcher pickles by value with lazily drawn random UUIDs (its own and
    those of every dispatcher its globals reach), so the closure would get a new
    key, and recompile, in every process. This replaces the cells' pickle in the
    index key with ``key`` plus SHA-256 digests of the source of ``captured``'s
    module and of every module defining a dispatcher ``captured`` calls, so an
    edit to any of them still invalidates the entry. The closure's own bytecode,
    signature, target and CPU stay in the key as usual.

    ``key`` must name everything else that specializes ``captured`` (for a
    factory, its arguments): equal keys must mean equal generated code. Call
    before the closure first compiles.
    """
    from numba.core.caching import FunctionCache
    from numba.core.dispatcher import Dispatcher

    func = captured.py_func
    modules = {func.__module__}
    for name in func.__code__.co_names:
        value = func.__globals__.get(name)
        if isinstance(value, Dispatcher):
            modules.add(value.py_func.__module__)
    digest = hashlib.sha256(
        "\n".join([key, *(_module_source_digest(m) for m in sorted(modules))]).encode()
    ).hexdigest()

    class _StableKeyCache(FunctionCache):
        def _index_key(self, sig, codegen):
            codebytes = self._py_func.__code__.co_code
            return (sig, codegen.magic_tuple(), (hashlib.sha256(codebytes).hexdigest(), digest))

    closure._cache = _StableKeyCache(closure.py_func)


def _kernel_name(dispatcher) -> str:
    func = dispatcher.py_func
    return f"{func.__module__}.{func.__qualname__}"


class NumbaCompileRecorder:
    """Record Numba compilations (cache misses) inside a ``with`` block.

    Uses Numba's ``numba:compile`` event, which fires only when a signature is
    neither already loaded nor found in the disk cache. Nested compilations
    (a kernel compiling its callees) are attributed exclusively: ``self_s``
    excludes time spent compiling callees, so per-kernel ``self_s`` values sum
    to the total compile time without double counting.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._stack: list[list[float]] = []
        self._listener = None

    def __enter__(self) -> NumbaCompileRecorder:
        from numba.core import event as numba_event

        recorder = self

        class _Listener(numba_event.Listener):
            def on_start(self, event) -> None:
                recorder._stack.append([time.perf_counter(), 0.0])

            def on_end(self, event) -> None:
                start, children = recorder._stack.pop()
                elapsed = time.perf_counter() - start
                if recorder._stack:
                    recorder._stack[-1][1] += elapsed
                data = event.data
                recorder.records.append(
                    {
                        "kernel": _kernel_name(data["dispatcher"]),
                        "signature": str(tuple(data["args"])),
                        "inclusive_s": elapsed,
                        "self_s": elapsed - children,
                        "nested": bool(recorder._stack),
                    }
                )

        self._listener = _Listener()
        numba_event.register("numba:compile", self._listener)
        return self

    def __exit__(self, *exc) -> None:
        from numba.core import event

        event.unregister("numba:compile", self._listener)

    @property
    def total_s(self) -> float:
        """Wall time spent in top-level compilations."""
        return sum(r["inclusive_s"] for r in self.records if not r["nested"])

    def by_kernel(self) -> list[dict]:
        """Aggregate records per kernel, largest exclusive compile time first."""
        totals: dict[str, dict] = {}
        for record in self.records:
            entry = totals.setdefault(
                record["kernel"], {"kernel": record["kernel"], "compiles": 0, "self_s": 0.0}
            )
            entry["compiles"] += 1
            entry["self_s"] += record["self_s"]
        return sorted(totals.values(), key=lambda e: e["self_s"], reverse=True)


def numba_dispatcher_stats(prefix: str = "pyrite") -> list[dict]:
    """Return disk-cache hits/misses of every live Numba dispatcher under ``prefix``.

    Counts are cumulative for this process. ``cached`` is false for kernels
    compiled without ``cache=True``; all of their compilations are misses.
    Scans the garbage collector, so closure-generated kernels are included;
    meant for diagnostics, not hot paths. Returns ``[]`` if Numba is not loaded.
    """
    if "numba" not in sys.modules:
        return []
    from numba.core.caching import NullCache
    from numba.core.dispatcher import Dispatcher

    stats = []
    for obj in gc.get_objects():
        if not issubclass(type(obj), Dispatcher):  # never touch dead weak proxies
            continue
        module = getattr(obj.py_func, "__module__", "") or ""
        if module != prefix and not module.startswith(prefix + "."):
            continue
        hits = sum(obj._cache_hits.values())
        misses = sum(obj._cache_misses.values())
        if not (hits or misses):
            continue
        stats.append(
            {
                "kernel": _kernel_name(obj),
                "cached": not isinstance(obj._cache, NullCache),
                "signatures": len(obj.overloads),
                "hits": hits,
                "misses": misses,
            }
        )
    return sorted(stats, key=lambda s: (s["kernel"], s["signatures"]))


class CupyCompileRecorder:
    """Record CuPy kernel compilation and disk-cache hits inside a ``with`` block.

    Wraps CuPy's private ``cupy.cuda.compiler`` entry points for diagnostics
    only: ``_compile_with_cache_cuda`` (reached once per kernel and process,
    after CuPy's in-memory caches miss), its disk-cache ``load`` (a hit returns
    a payload), and ``compile_using_nvrtc`` (the actual NVRTC compile). Restores
    them on exit. A no-op when those names are absent.
    """

    def __init__(self) -> None:
        self.lookups = 0
        self.disk_hits = 0
        self.nvrtc_compiles = 0
        self.nvrtc_s = 0.0
        self.compile_with_cache_s = 0.0
        self._restore: list[tuple[object, str, object]] = []

    def _wrap(self, owner, name: str, make) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return
        self._restore.append((owner, name, original))
        setattr(owner, name, make(original))

    def __enter__(self) -> CupyCompileRecorder:
        from cupy.cuda import compiler

        recorder = self

        def timed(attribute: str, counter: str | None):
            def make(original):
                def wrapper(*args, **kwargs):
                    start = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        elapsed = time.perf_counter() - start
                        setattr(recorder, attribute, getattr(recorder, attribute) + elapsed)
                        if counter is not None:
                            setattr(recorder, counter, getattr(recorder, counter) + 1)

                return wrapper

            return make

        def counted_load(original):
            def load(*args, **kwargs):
                payload = original(*args, **kwargs)
                recorder.disk_hits += payload is not None
                return payload

            return load

        self._wrap(compiler, "_compile_with_cache_cuda", timed("compile_with_cache_s", "lookups"))
        self._wrap(compiler, "compile_using_nvrtc", timed("nvrtc_s", "nvrtc_compiles"))
        backend = getattr(compiler, "_kernel_cache_backend", None)
        if backend is not None:
            self._wrap(backend, "load", counted_load)
        return self

    def __exit__(self, *exc) -> None:
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()

    def summary(self) -> dict:
        """Return counts and seconds; misses are lookups without a disk hit."""
        return {
            "lookups": self.lookups,
            "disk_hits": self.disk_hits,
            "disk_misses": self.lookups - self.disk_hits,
            "nvrtc_compiles": self.nvrtc_compiles,
            "nvrtc_s": self.nvrtc_s,
            "compile_with_cache_s": self.compile_with_cache_s,
        }
