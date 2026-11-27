"""``pixi.lock`` never installs one Python distribution from both conda and PyPI (#1462).

A conda Python package and a PyPI wheel of the same distribution would both
land in ``site-packages``; whichever writes last wins, so the environment no
longer matches its lock. pixi skips a PyPI requirement that a conda package
already satisfies only when its conda-to-PyPI map names that package, which is
why the ``svml`` feature maps the Anaconda ``pkgs/main`` channel explicitly.
"""

import re
import tomllib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "pixi.lock"

#: conda build strings of Python distributions (``py313h...``, ``pyhd8ed1ab_0``).
_PY_BUILD = re.compile(r"(^|_)py(\d+|h)")
#: conda name -> PyPI distribution name, where the two differ.
_CONDA_TO_PYPI = {"cupy-core": "cupy"}
#: Different distributions of one import package (CuPy wheels per toolkit).
_PYPI_ALIASES = {"cupy-cuda12x": "cupy", "cupy-cuda13x": "cupy"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _conda_python_dists(lock: dict) -> dict[str, str]:
    """conda URL -> normalized PyPI name, for packages that are Python dists."""
    dists = {}
    for package in lock["packages"]:
        url = package.get("conda")
        if url is None:
            continue
        filename = url.rsplit("/", 1)[-1].removesuffix(".conda").removesuffix(".tar.bz2")
        name, _version, build = filename.rsplit("-", 2)
        if _PY_BUILD.search(build):
            dists[url] = _normalize(_CONDA_TO_PYPI.get(name, name))
    return dists


def _pypi_name(lock_index: dict, url: str) -> str:
    name = _normalize(lock_index[url]["name"])
    return _PYPI_ALIASES.get(name, name)


@pytest.mark.skipif(not LOCK.exists(), reason="no pixi.lock in this checkout")
def test_no_distribution_installed_from_both_conda_and_pypi():
    lock = yaml.safe_load(LOCK.read_text())
    conda = _conda_python_dists(lock)
    pypi_index = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
    clashes = []
    for env, spec in lock["environments"].items():
        for platform, entries in spec["packages"].items():
            conda_names = {conda[e["conda"]] for e in entries if e.get("conda") in conda}
            pypi_names = {_pypi_name(pypi_index, e["pypi"]) for e in entries if "pypi" in e}
            clashes += [f"{env}/{platform}: {n}" for n in sorted(conda_names & pypi_names)]
    assert not clashes, "installed from both conda and PyPI:\n" + "\n".join(clashes)


_RUNTIME_PEERS = {
    "runtime-cpu": "default",
    "runtime-nvidia": "nvidia",
    "runtime-amd": "amd",
    "runtime-intel": "intel",
}


@pytest.mark.parametrize(("runtime", "peer"), _RUNTIME_PEERS.items())
def test_runtime_manifest_is_lean(runtime, peer):
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    environments = manifest["tool"]["pixi"]["environments"]
    spec = environments[runtime]
    features = set(spec["features"])
    assert features == set(environments[peer]["features"]) - set(manifest["dependency-groups"])
    assert not features.intersection({"apps", "plots"})
    assert spec["solve-group"] == environments[peer]["solve-group"]
    assert not spec.get("no-default-feature")


@pytest.mark.skipif(not LOCK.exists(), reason="no pixi.lock in this checkout")
@pytest.mark.parametrize(("runtime", "peer"), _RUNTIME_PEERS.items())
def test_runtime_lock_is_strict_subset_of_developer_peer(runtime, peer):
    lock = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    runtime_rows = lock["environments"][runtime]["packages"]
    peer_rows = lock["environments"][peer]["packages"]
    assert runtime_rows.keys() == peer_rows.keys()
    for platform, entries in runtime_rows.items():
        # Exact locked locations, not just names: compare like-for-like builds.
        runtime_packages = {tuple(sorted(entry.items())) for entry in entries}
        peer_packages = {tuple(sorted(entry.items())) for entry in peer_rows[platform]}
        assert runtime_packages < peer_packages, f"{runtime}/{platform} differs from {peer}"
