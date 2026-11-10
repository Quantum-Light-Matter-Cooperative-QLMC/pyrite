"""Managed JIT cache policy, inspection, clearing, and native Numba invalidation."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pyrite import _jit_cache


@pytest.fixture
def managed(monkeypatch, tmp_path):
    root = tmp_path / "user-cache"
    monkeypatch.setattr(_jit_cache, "cache_dir", lambda: root)
    return root


def test_apply_sets_only_unset_variables(managed):
    env = {"CUPY_CACHE_DIR": "/user/cupy", "NUMBA_CACHE_DIR": ""}
    applied = _jit_cache.apply_managed_cache_env(env)
    assert applied == {"NUMBA_CACHE_DIR": str(managed / "numba")}
    assert env == {"CUPY_CACHE_DIR": "/user/cupy", "NUMBA_CACHE_DIR": str(managed / "numba")}
    assert _jit_cache.apply_managed_cache_env(env) == {}


def test_locations_classify_sources(managed, tmp_path):
    env = {"NUMBA_CACHE_DIR": str(tmp_path / "mine")}
    by_source = {(loc.backend, loc.source): loc for loc in _jit_cache.cache_locations(env)}
    assert by_source["numba", "environment"].active and not by_source["numba", "environment"].owned
    assert not by_source["numba", "in-tree"].active
    assert by_source["cupy", "default"].path == Path.home() / ".cupy" / "kernel_cache"
    _jit_cache.apply_managed_cache_env(env)
    managed_cupy = {(loc.backend, loc.source): loc for loc in _jit_cache.cache_locations(env)}
    assert managed_cupy["cupy", "managed"].owned


def test_subpath_matches_numba(tmp_path):
    from numba.core.caching import _CacheLocator

    source = tmp_path / "pkg" / "kernels.py"
    assert _jit_cache.numba_cache_subpath(source.parent) == (
        _CacheLocator.get_suitable_cache_subpath(str(source))
    )


def test_clear_plan_removes_only_pyrite_owned_paths(managed, tmp_path):
    user_numba = tmp_path / "shared-numba"
    pyrite_dir = user_numba / _jit_cache.numba_cache_subpath(_jit_cache._PACKAGE_ROOT)
    foreign = user_numba / "other_project_0123"
    for directory in (pyrite_dir, foreign, managed / "cupy", tmp_path / "user-cupy"):
        directory.mkdir(parents=True)
        (directory / "k.nbi").write_bytes(b"x")
    env = {"NUMBA_CACHE_DIR": str(user_numba)}
    plan = _jit_cache.clear_plan(_jit_cache.cache_locations(env))
    deletions = {t.path for t in plan if t.kind != "skip"}
    assert pyrite_dir in deletions and foreign not in deletions

    env = {"NUMBA_CACHE_DIR": str(user_numba), "CUPY_CACHE_DIR": str(tmp_path / "user-cupy")}
    plan = _jit_cache.clear_plan(_jit_cache.cache_locations(env))
    assert [t.kind for t in plan if t.backend == "cupy"] == ["skip"]

    _jit_cache.apply_managed_cache_env(env := {})
    plan = _jit_cache.clear_plan([loc for loc in _jit_cache.cache_locations(env) if loc.owned])
    assert {t.path for t in plan} == {managed / "cupy"}
    _jit_cache.execute_clear(plan)
    assert not (managed / "cupy").exists() and foreign.exists() and pyrite_dir.exists()


_PROBE = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import kernels
    kernels.scale(1.0)
    closure = kernels.make(2.0)
    closure(1.0)
    from pyrite._jit_cache import numba_dispatcher_stats
    print(json.dumps({s["kernel"]: [s["hits"], s["misses"]]
                      for s in numba_dispatcher_stats(prefix="kernels")}))
    """
)


def _probe(module_dir: Path, cache: Path) -> dict:
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache)}
    env.pop("NUMBA_DISABLE_JIT", None)
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(module_dir)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_native_numba_cache_hits_and_invalidates_across_processes(tmp_path):
    """Same source + cache dir hits; source edit or another directory misses.

    Numba stamps sources by content, so an mtime-only change (a remote sync)
    still hits.

    Closure kernels are keyed by their cell contents, so a closure created
    with the same values hits in a fresh process.
    """
    module_dir = tmp_path / "src"
    module_dir.mkdir()
    source = module_dir / "kernels.py"
    body = textwrap.dedent(
        """
        from numba import njit

        @njit(cache=True)
        def scale(x):
            return {factor} * x

        def make(k):
            @njit(cache=True)
            def closure(x):
                return k * x
            return closure
        """
    )
    source.write_text(body.format(factor=2.0))
    cache = tmp_path / "cache"
    miss = {"kernels.scale": [0, 1], "kernels.make.<locals>.closure": [0, 1]}
    hit = {"kernels.scale": [1, 0], "kernels.make.<locals>.closure": [1, 0]}
    assert _probe(module_dir, cache) == miss
    assert _probe(module_dir, cache) == hit

    index = next(cache.rglob("kernels.scale-*.nbi"))
    stamp, overloads = _jit_cache.numba_index_entries(index)
    assert stamp == hashlib.sha256(source.read_bytes()).digest() and overloads == 1
    stat = source.stat()

    assert _probe(module_dir, tmp_path / "other-cache") == miss
    source.write_text(body.format(factor=3.0))
    os.utime(source, (stat.st_atime, stat.st_mtime + 10))
    assert _probe(module_dir, cache) == miss
    assert _jit_cache.numba_index_entries(index)[0] == hashlib.sha256(source.read_bytes()).digest()

    # Touching the source without changing its bytes keeps the content stamp.
    os.utime(source, (stat.st_atime, stat.st_mtime + 20))
    assert _probe(module_dir, cache) == hit


def test_cli_entry_respects_overrides_after_import_defaults(tmp_path):
    script = textwrap.dedent(
        """
        import json, os, sys
        import pyrite, pyrite.cli
        before = {k: os.environ.get(k) for k in ("NUMBA_CACHE_DIR", "CUPY_CACHE_DIR")}
        loaded = {m: m in sys.modules for m in ("numba", "cupy")}
        try:
            pyrite.cli.main(["cache", "status", "-o", "json"])
        except SystemExit:
            pass
        after = {k: os.environ.get(k) for k in ("NUMBA_CACHE_DIR", "CUPY_CACHE_DIR")}
        print(json.dumps({"before": before, "after": after, "loaded": loaded,
                          "cupy": "cupy" in sys.modules}), file=sys.stderr)
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in {"NUMBA_CACHE_DIR", "CUPY_CACHE_DIR"}}
    env["CUPY_CACHE_DIR"] = str(tmp_path / "mine")
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    report = json.loads(done.stderr.strip().splitlines()[-1])
    managed_numba = str(tmp_path / "xdg" / "pyrite" / "numba")
    assert report["before"] == {
        "NUMBA_CACHE_DIR": managed_numba,
        "CUPY_CACHE_DIR": str(tmp_path / "mine"),
    }
    assert report["loaded"] == {"numba": False, "cupy": False}
    assert report["after"]["CUPY_CACHE_DIR"] == str(tmp_path / "mine")
    assert Path(report["after"]["NUMBA_CACHE_DIR"]).parts[-2:] == ("pyrite", "numba")
    assert not report["cupy"]
    status = json.loads(done.stdout)
    assert status["schema"] == "pyrite.cache.status.v1"
    sources = {(loc["backend"], loc["source"]) for loc in status["payload"]["locations"]}
    assert sources == {("numba", "managed"), ("numba", "in-tree"), ("cupy", "environment")}


_CORE_KEY = textwrap.dedent(
    """
    import sys
    from pyrite.montecarlo.transport.cores import make_cpu_transport_core
    core = make_cpu_transport_core(radiative=sys.argv[1] == "radiative")
    key = core._cache._index_key((), core.targetctx.codegen())
    print(key[2][1])
    """
)


def test_transport_core_cache_key_is_stable_across_processes():
    """Regression (#910): ``core`` captured ``body`` and missed in every process."""

    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_JIT"}

    def key(variant: str) -> str:
        done = subprocess.run(
            [sys.executable, "-c", _CORE_KEY, variant], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    first = key("default")
    assert first == key("default")
    assert first != key("radiative")


def test_transport_import_survives_disabled_jit():
    """Regression (#344): ``NUMBA_DISABLE_JIT`` closures are plain functions."""
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1"}
    done = subprocess.run(
        [sys.executable, "-c", "import pyrite.montecarlo.transport.core_specializations"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="needs compiled Numba dispatchers"
)
def test_transport_core_digest_reaches_module_attribute_callees():
    """Regression (#991): ``_cutoff.*``/``_moments.*`` edits invalidate the core.

    Numba does not re-check a disk-cached callee when its caller loads, so the
    stable key must digest modules ``body`` reaches through ``module.attr``
    calls (``_cutoff.lut_cutoff_bound``), not only through direct globals.
    """
    from numba.core.dispatcher import Dispatcher

    from pyrite.montecarlo.transport.cores import make_cpu_transport_core

    core = make_cpu_transport_core(radiative=True)
    (body,) = [
        cell.cell_contents
        for cell in core.py_func.__closure__
        if isinstance(cell.cell_contents, Dispatcher)
    ]
    modules = _jit_cache._dispatch_modules(body)
    prefix = "pyrite.montecarlo.transport."
    for leaf in ("cores", "stopping", "lut", "_jit_cutoff_bound", "_jit_moment_lookup"):
        assert prefix + leaf in modules
    assert all(m.startswith("pyrite.") for m in modules)


_IMPORT_PROBE = textwrap.dedent(
    """
    import json, os, sys
    pre = sys.argv[1] == "numba-first"
    if pre:
        import numba
    import pyrite
    from numba.core import config
    print(json.dumps({
        "numba": os.environ.get("NUMBA_CACHE_DIR"),
        "cupy": os.environ.get("CUPY_CACHE_DIR"),
        "config": config.CACHE_DIR if pre else None,
        "cupy_imported": "cupy" in sys.modules,
    }))
    """
)


def _import_probe(mode: str, cache_root: Path, **overrides: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("NUMBA_CACHE_DIR", "CUPY_CACHE_DIR")}
    env["XDG_CACHE_HOME"] = str(cache_root)
    env.update(overrides)
    done = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, mode], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("mode", ["plain", "numba-first"])
def test_import_pyrite_sets_managed_defaults(tmp_path, mode):
    out = _import_probe(mode, tmp_path)
    assert out["numba"] == str(tmp_path / "pyrite" / "numba")
    assert out["cupy"] == str(tmp_path / "pyrite" / "cupy")
    assert out["config"] in (None, out["numba"])
    assert not out["cupy_imported"]


@pytest.mark.parametrize("mode", ["plain", "numba-first"])
def test_import_pyrite_keeps_user_cache_dirs(tmp_path, mode):
    mine = {"NUMBA_CACHE_DIR": str(tmp_path / "n"), "CUPY_CACHE_DIR": str(tmp_path / "c")}
    out = _import_probe(mode, tmp_path / "xdg", **mine)
    assert (out["numba"], out["cupy"]) == (mine["NUMBA_CACHE_DIR"], mine["CUPY_CACHE_DIR"])
    assert out["config"] in (None, mine["NUMBA_CACHE_DIR"])


def _fake_cupy(monkeypatch, *, nvrtc_names):
    """Install a fake ``cupy.cuda.compiler`` mirroring CuPy's call structure."""
    import types

    compiler = types.ModuleType("cupy.cuda.compiler")
    store: dict[str, bytes] = {}

    class Backend:
        def load(self, name):
            return store.get(name)

        def save(self, name, payload, source):
            store[name] = payload

    compiler._kernel_cache_backend = Backend()
    internal = nvrtc_names[0]

    def nvrtc(*args, **kwargs):
        return b"cubin"

    def compile_with_cache(name):
        payload = compiler._kernel_cache_backend.load(name)
        if payload is None:
            getattr(compiler, internal)()
            compiler._kernel_cache_backend.save(name, b"cubin", "")
        return name

    for name in nvrtc_names:
        setattr(compiler, name, nvrtc)
    if len(nvrtc_names) == 2:  # CuPy 14: the public name wraps the private one
        compiler.compile_using_nvrtc = lambda: compiler._compile_using_nvrtc_no_warning()
    compiler._compile_with_cache_cuda = compile_with_cache
    cuda = types.ModuleType("cupy.cuda")
    cuda.compiler = compiler
    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.compiler", compiler)
    return compiler


@pytest.mark.parametrize(
    "names", [("_compile_using_nvrtc_no_warning", "compile_using_nvrtc"), ("compile_using_nvrtc",)]
)
def test_cupy_recorder_counts_nvrtc_compiles_and_disk_hits(monkeypatch, names):
    compiler = _fake_cupy(monkeypatch, nvrtc_names=names)
    originals = {n: getattr(compiler, n) for n in names}
    with _jit_cache.CupyCompileRecorder() as cold:
        compiler._compile_with_cache_cuda("k1")
        compiler._compile_with_cache_cuda("k2")
    with _jit_cache.CupyCompileRecorder() as warm:
        compiler._compile_with_cache_cuda("k1")
    assert (cold.lookups, cold.disk_hits, cold.nvrtc_compiles) == (2, 0, 2)
    assert (warm.lookups, warm.disk_hits, warm.nvrtc_compiles) == (1, 1, 0)
    assert {n: getattr(compiler, n) for n in names} == originals


def test_native_cache_separates_precision_signatures(tmp_path):
    """FP32 and FP64 must miss independently, then both hit in a fresh process."""
    source = tmp_path / "kernels.py"
    source.write_text(
        "from numba import njit\n@njit(cache=True)\ndef scale(x):\n    return x + x\n"
    )
    script = """
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import kernels
from pyrite._jit_cache import numba_dispatcher_stats
x = getattr(np, sys.argv[2])(1.25)
assert kernels.scale(x) == 2.5
print(json.dumps(numba_dispatcher_stats(prefix="kernels")))
"""
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache")}
    env.pop("NUMBA_DISABLE_JIT", None)
    for dtype, hits, misses in (
        ("float64", 0, 1),
        ("float32", 0, 1),
        ("float64", 1, 0),
        ("float32", 1, 0),
    ):
        done = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), dtype],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        (record,) = json.loads(done.stdout)
        assert (record["hits"], record["misses"]) == (hits, misses)
