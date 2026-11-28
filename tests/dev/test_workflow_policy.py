"""Regression checks for CI coverage and release ordering."""

import ast
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_quantum_filter_covers_direct_imports_and_build_inputs():
    changes = _workflow("ci.yml")["jobs"]["changes"]
    filters = next(
        step["with"]["filters"] for step in changes["steps"] if step.get("id") == "filter"
    )
    patterns = yaml.safe_load(filters)["quantum"]
    required = {
        "src/pyrite/quantum/__init__.py",
        "src/pyrite/montecarlo/runner/axial.py",
        "src/pyrite/devtools/test_suites.py",
        "src/pyrite/devtools/suite_inventory.py",
        "pyproject.toml",
        "pixi.lock",
        ".github/actions/setup-pyrite/action.yml",
    }
    for file in (ROOT / "tests/montecarlo/quantum").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
            for name in names:
                if name == "pyrite" or name.startswith("pyrite."):
                    path = ROOT / "src" / name.replace(".", "/")
                    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
                        if candidate.is_file():
                            required.add(candidate.relative_to(ROOT).as_posix())
    missing = sorted(path for path in required if not any(fnmatchcase(path, p) for p in patterns))
    assert not missing


def test_release_publication_depends_on_verified_build():
    jobs = _workflow("release.yml")["jobs"]
    assert "needs" not in jobs["build"]
    assert jobs["release"]["needs"] == "build"
    assert set(jobs["publish"]["needs"]) == {"build", "release"}
    steps = [step.get("name") for step in jobs["build"]["steps"]]
    assert steps.index("Audit distribution contents") < steps.index("Upload distributions")
    assert steps.index("Clean-install smoke of the built wheel and sdist") < steps.index(
        "Upload distributions"
    )
    for job in (jobs["publish"], jobs["release"], _workflow("ci.yml")["jobs"]["format"]):
        for step in job["steps"]:
            action = step.get("uses", "")
            if action and not action.startswith("./"):
                ref = action.rsplit("@", 1)[-1]
                assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)
