"""Build and install pyrite-mc from wheel and editable source in clean pip venvs.

The venvs use only the standard library ``venv`` and ``pip``: the same path a
user takes with ``pip install pyrite-mc``, independent of the pixi workspace.
"""

import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from pyrite.devtools import repo_root

ROOT = repo_root()
PROJECT_DOCUMENT = tomllib.loads((ROOT / "pyproject.toml").read_text())
#: The wheel must publish exactly the extras ``pyproject.toml`` declares.
EXPECTED_EXTRAS = set(PROJECT_DOCUMENT["project"]["optional-dependencies"])
PROJECT_VERSION = PROJECT_DOCUMENT["project"]["version"]


def _run(*args: str, cwd: Path) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PYRITE_MC_BACKEND"] = "cpu"
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _run_fails_with_hint(*args: str, cwd: Path, hint: str) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PYRITE_MC_BACKEND"] = "cpu"
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"{args!r} succeeded without its optional extra"
    assert hint in output, f"{args!r} did not name {hint!r}:\n{output}"


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

    assert metadata_name.startswith("pyrite_mc-")
    assert "Name: pyrite-mc" in metadata.splitlines()
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
    assert all(name.startswith(("pyrite/", "pyrite_mc-")) for name in names)


def _pip(venv: Path, *args: str, cwd: Path) -> None:
    _run(str(_python(venv)), "-m", "pip", "--disable-pip-version-check", *args, cwd=cwd)


def _probe_install(source: Path, root: Path, label: str) -> None:
    venv = root / label
    _run(sys.executable, "-m", "venv", str(venv), cwd=root)
    _pip(venv, "install", str(source), cwd=root)
    _run(
        str(_python(venv)),
        "-c",
        (
            "from importlib import util; from importlib.metadata import distribution; "
            "from pathlib import Path; "
            "import pyrite; dist = distribution('pyrite-mc'); "
            "assert dist.metadata['Name'] == 'pyrite-mc'; "
            "assert dist.version == pyrite.__version__; "
            "assert (pyrite.DATA_DIR / 'catalog' / 'catalog.toml').is_file(); "
            "from pyrite.materials import CATALOG; "
            "from pyrite.runs.scan import resolve_profile_materials; "
            "standard = CATALOG.profile_materials('standard'); "
            "assert standard is not None and 'hopg' in standard; "
            "assert resolve_profile_materials('standard') == list(standard); "
            "assert not (Path.cwd() / 'mats_to_sim.toml').exists(); "
            "assert util.find_spec('cxr_mc') is None; "
            "assert util.find_spec('pyrite_mc') is None"
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
    # Base install: app launches and plot backends must fail with an install hint.
    _run_fails_with_hint(
        pyrite, "app", "analysis", "launch", "--smoke", cwd=root, hint="pyrite-mc[apps]"
    )
    _run_fails_with_hint(
        str(_python(venv)), "-c", "import pyrite.plots.mpl", cwd=root, hint="pyrite-mc[plots]"
    )
    _pip(venv, "install", f"pyrite-mc[apps] @ {source.as_uri()}", cwd=root)
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
    with tempfile.TemporaryDirectory(prefix="pyrite-mc-package-") as tmp:
        work = Path(tmp)
        dist = work / "dist"
        _run(sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(ROOT), cwd=ROOT)
        wheel = next(dist.glob("pyrite_mc-*.whl"))
        _inspect_wheel(wheel)
        _probe_install(wheel, work, "wheel-venv")
        _probe_install(ROOT, work, "editable-venv")
    print("pyrite-mc wheel and editable installs expose only canonical PyRITE identities")


if __name__ == "__main__":
    main()
