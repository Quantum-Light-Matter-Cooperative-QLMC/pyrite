"""Build and install pyrite-xray from wheel and editable source in clean uv venvs."""

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from pyrite.devtools import repo_root

ROOT = repo_root()
EXPECTED_EXTRAS = {"amd", "external-db", "intel", "nvidia"}
PROJECT_DOCUMENT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT_VERSION = PROJECT_DOCUMENT["project"]["version"]
NOTEBOOK_DEPENDENCIES = PROJECT_DOCUMENT["dependency-groups"]["notebooks"]


def _run(*args: str, cwd: Path) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PYRITE_MC_BACKEND"] = "cpu"
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _script(venv: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _inspect_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = archive.read(metadata_name).decode()
        entry_points = archive.read(entry_points_name).decode()

    assert metadata_name.startswith("pyrite_xray-")
    assert "Name: pyrite-xray" in metadata.splitlines()
    assert f"Version: {PROJECT_VERSION}" in metadata.splitlines()
    assert "pyrite/__init__.py" in names
    assert "pyrite/data/catalog/catalog.toml" in names
    assert "pyrite/data/catalog/profiles/standard.toml" in names
    assert "pyrite/apps/analysis_app.py" in names
    assert "pyrite/apps/compare_app.py" in names
    assert "pyrite/apps/pixel_app.py" in names
    assert "pyrite/apps/_design.css" in names
    assert "pyrite/apps/trace_app.py" in names
    assert "pyrite/apps/validation_app.py" in names
    assert "pyrite/apps/validation_defaults.json" in names
    assert (
        "pyrite/validation/reference_data/external_brem/v1/zhai_fig3b_25kev_1mm_brem.csv" in names
    )
    assert "pyrite = pyrite.cli:main" in entry_points
    assert "pyrite-dev = pyrite._dev:main" in entry_points
    extras = {
        line.removeprefix("Provides-Extra: ")
        for line in metadata.splitlines()
        if line.startswith("Provides-Extra: ")
    }
    assert extras == EXPECTED_EXTRAS
    assert all(name.startswith(("pyrite/", "pyrite_xray-")) for name in names)


def _probe_install(uv: str, source: Path, root: Path, label: str) -> None:
    venv = root / label
    _run(uv, "venv", "--python", sys.executable, str(venv), cwd=root)
    _run(uv, "pip", "install", "--python", str(_python(venv)), str(source), cwd=root)
    _run(
        str(_python(venv)),
        "-c",
        (
            "from importlib import util; from importlib.metadata import distribution; "
            "from pathlib import Path; "
            "import pyrite; dist = distribution('pyrite-xray'); "
            "assert dist.metadata['Name'] == 'pyrite-xray'; "
            "assert dist.version == pyrite.__version__; "
            "assert (pyrite.DATA_DIR / 'catalog' / 'catalog.toml').is_file(); "
            "from pyrite.materials import CATALOG; "
            "from pyrite.runs.scan import resolve_profile_materials; "
            "standard = CATALOG.profile_materials('standard'); "
            "assert standard is not None and 'hopg' in standard; "
            "assert resolve_profile_materials('standard') == list(standard); "
            "assert not (Path.cwd() / 'mats_to_sim.toml').exists(); "
            "assert util.find_spec('cxr_mc') is None; "
            "assert util.find_spec('pyrite_xray') is None"
        ),
        cwd=root,
    )
    _run(
        str(_python(venv)),
        "-c",
        (
            "import pickle; from pyrite.campaign.sweep import Sweep; "
            "payload = pickle.dumps(Sweep(material='hopg')); "
            "assert b'pyrite.campaign.sweep' in payload; "
            "assert type(pickle.loads(payload)) is Sweep"
        ),
        cwd=root,
    )
    pyrite = str(_script(venv, "pyrite"))
    env_home = root / f"{label}-workspace"
    env_home.mkdir()
    _run(pyrite, "--help", cwd=root)
    _run(pyrite, "run", "--help", cwd=root)
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(_python(venv)),
        *NOTEBOOK_DEPENDENCIES,
        cwd=root,
    )
    _run(pyrite, "app", "analysis", "launch", "--smoke", cwd=root)
    _run(pyrite, "app", "pixels", "launch", "--smoke", cwd=root)
    _run(pyrite, "app", "compare", "launch", "--smoke", cwd=root)
    _run(pyrite, "app", "viewer", "launch", "--smoke", cwd=root)
    _run(
        str(_python(venv)),
        "-c",
        (
            "from pathlib import Path; import pyrite, pyrite.runs.run as run; "
            "site = Path(pyrite.__file__).resolve().parent; "
            "assert not Path(run.DEFAULT_CHECKPOINT_DIR).resolve().is_relative_to(site)"
        ),
        cwd=env_home,
    )
    _run(str(_script(venv, "pyrite-dev")), "--help", cwd=root)


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found")
    with tempfile.TemporaryDirectory(prefix="pyrite-xray-package-") as tmp:
        work = Path(tmp)
        dist = work / "dist"
        _run(uv, "build", "--wheel", "--out-dir", str(dist), cwd=ROOT)
        wheel = next(dist.glob("pyrite_xray-*.whl"))
        _inspect_wheel(wheel)
        _probe_install(uv, wheel, work, "wheel-venv")
        _probe_install(uv, ROOT, work, "editable-venv")
    print("pyrite-xray wheel and editable installs expose only canonical PyRITE identities")


if __name__ == "__main__":
    main()
