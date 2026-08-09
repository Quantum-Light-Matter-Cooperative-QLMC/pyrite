"""Build and install cxr-mc from wheel and editable source in clean uv venvs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXTRAS = {"amd", "external-db", "intel", "nvidia"}


def _run(*args: str, cwd: Path) -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
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

    assert "cxr_mc/__init__.py" in names
    assert "cxr_mc/data/materials.toml" in names
    assert "cxr_mc/apps/analysis_app.py" in names
    assert "cxr_mc/apps/trace_app.py" in names
    assert "cxr_mc/apps/validation_app.py" in names
    assert "cxr_mc/apps/validation_defaults.json" in names
    assert "cxr_mc/apps/reference_data/external_brem/v1/zhai_fig3b_25kev_1mm_brem.csv" in names
    assert "cxr = cxr_mc.cli:main" in entry_points
    assert "cxr-dev = cxr_mc._dev:main" in entry_points
    extras = {
        line.removeprefix("Provides-Extra: ")
        for line in metadata.splitlines()
        if line.startswith("Provides-Extra: ")
    }
    assert extras == EXPECTED_EXTRAS
    assert all(name.startswith(("cxr_mc/", "cxr_mc-")) for name in names)


def _probe_install(uv: str, source: Path, root: Path, label: str) -> None:
    venv = root / label
    _run(uv, "venv", "--python", sys.executable, str(venv), cwd=root)
    _run(uv, "pip", "install", "--python", str(_python(venv)), str(source), cwd=root)
    _run(
        str(_python(venv)),
        "-c",
        "import cxr_mc; assert (cxr_mc.DATA_DIR / 'materials.toml').is_file()",
        cwd=root,
    )
    cxr = str(_script(venv, "cxr"))
    env_home = root / f"{label}-workspace"
    env_home.mkdir()
    _run(cxr, "--help", cwd=root)
    _run(cxr, "run", "--help", cwd=root)
    _run(cxr, "app", "analysis", "launch", "--smoke", cwd=root)
    _run(cxr, "app", "viewer", "launch", "--smoke", cwd=root)
    _run(
        str(_python(venv)),
        "-c",
        (
            "from pathlib import Path; import cxr_mc, cxr_mc.runs.run as run; "
            "site = Path(cxr_mc.__file__).resolve().parent; "
            "assert not Path(run.DEFAULT_CHECKPOINT_DIR).resolve().is_relative_to(site)"
        ),
        cwd=env_home,
    )
    _run(str(_script(venv, "cxr-dev")), "--help", cwd=root)


def main() -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found")
    with tempfile.TemporaryDirectory(prefix="cxr-mc-package-") as tmp:
        work = Path(tmp)
        dist = work / "dist"
        _run(uv, "build", "--wheel", "--out-dir", str(dist), cwd=ROOT)
        wheel = next(dist.glob("cxr_mc-*.whl"))
        _inspect_wheel(wheel)
        _probe_install(uv, wheel, work, "wheel-venv")
        _probe_install(uv, ROOT, work, "editable-venv")
    print("wheel and editable installs preserve imports, data, extras, cxr, and cxr-dev")


if __name__ == "__main__":
    main()
