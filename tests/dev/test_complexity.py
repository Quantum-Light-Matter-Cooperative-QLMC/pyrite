"""Tests for the source complexity report and high-CC ratchet."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from pyrite.devtools import complexity, dev_cli


def _routine(name: str, cc: int, indent: str = "") -> str:
    """Return a def whose Radon CC is exactly ``cc`` (one per ``if`` plus one)."""
    body = "".join(f"{indent}    if x == {i}:\n{indent}        pass\n" for i in range(cc - 1))
    return f"{indent}def {name}(x):\n{body}{indent}    return x\n"


def _checkout(root: Path, modules: dict[str, str]) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    package = root / "src" / "pyrite"
    package.mkdir(parents=True)
    for name, text in modules.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _check(root: Path) -> int:
    return complexity.run(root, check=True)


def _rewrite(root: Path, name: str, text: str) -> None:
    (root / "src" / "pyrite" / name).write_text(text)


@pytest.fixture
def repo(tmp_path):
    root = _checkout(
        tmp_path,
        {"hot.py": _routine("hot", 25) + _routine("cool", 3), "other.py": _routine("small", 2)},
    )
    assert complexity.run(root, write=True) == 0
    return root


def test_routines_cover_methods_closures_nested_classes_and_repeats() -> None:
    source = (
        "class A:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return 1\n"
        "    @x.setter\n"
        "    def x(self, value):\n"
        "        pass\n"
        "    class B:\n"
        "        async def b(self):\n"
        "            if self:\n"
        "                pass\n"
        "def outer(y):\n"
        "    def inner():\n"
        "        return y if y else 0\n"
        "    class K:\n"
        "        def m(self):\n"
        "            pass\n"
        "    return inner\n"
    )
    _, visitor, _ = complexity.radon_modules()

    routines = complexity.routines_in_source(source, "m.py", complexity._radon_cc(visitor))

    assert [(r.name, r.cc) for r in routines] == [
        ("A.x", 1),
        ("A.x#2", 1),
        ("A.B.b", 2),
        ("outer", 1),
        ("outer.inner", 2),
        ("outer.K.m", 1),
    ]
    assert len({(r.lineno, r.col) for r in routines}) == len(routines)


def test_scan_skips_excluded_trees_and_reports_parse_failures(tmp_path) -> None:
    root = _checkout(
        tmp_path,
        {
            "ok.py": _routine("ok", 2),
            "broken.py": "def broken(:\n",
            "vendor/third.py": _routine("third", 30),
            "__pycache__/cached.py": _routine("cached", 30),
        },
    )

    result = complexity.scan(root)

    assert [r.name for r in result.routines] == ["ok"]
    assert [f.path for f in result.failures] == ["src/pyrite/broken.py"]
    assert "SyntaxError" in result.failures[0].message
    with pytest.raises(complexity.ComplexityToolError, match="parse failures"):
        complexity.run(root, write=True)


def test_baseline_records_only_hotspots_with_tool_provenance(repo) -> None:
    baseline = json.loads((repo / complexity.BASELINE_PATH).read_text())

    assert baseline["tool"] == {"radon": complexity.radon_modules()[0]}
    assert baseline["threshold"] == complexity.CC_THRESHOLD
    assert baseline["routines"] == [{"path": "src/pyrite/hot.py", "name": "hot", "cc": 25}]


def test_unchanged_debt_passes(repo, capsys) -> None:
    assert _check(repo) == 0
    assert "1 unchanged hotspots" in capsys.readouterr().out


def test_new_and_worsened_hotspots_fail(repo, capsys) -> None:
    _rewrite(repo, "hot.py", _routine("hot", 26) + _routine("cool", 21))

    assert _check(repo) == 1

    out = capsys.readouterr().out
    assert "new hotspot src/pyrite/hot.py:" in out and "cool CC 21 > 20" in out
    assert "worsened hotspot src/pyrite/hot.py:1 hot CC 26 (baseline 25)" in out


def test_improved_and_resolved_debt_pass_and_write_drops_it(repo, capsys) -> None:
    _rewrite(repo, "hot.py", _routine("hot", 22))
    assert _check(repo) == 0
    assert "lock in with --write" in capsys.readouterr().out

    _rewrite(repo, "hot.py", _routine("hot", 5))
    assert _check(repo) == 0
    assert complexity.run(repo, write=True) == 0

    assert "removed src/pyrite/hot.py::hot (was CC 25)" in capsys.readouterr().out
    baseline = json.loads((repo / complexity.BASELINE_PATH).read_text())
    assert baseline["routines"] == []


def test_moved_or_renamed_hotspot_fails_with_its_origin(repo, capsys) -> None:
    _rewrite(repo, "hot.py", _routine("cool", 3))
    _rewrite(repo, "other.py", _routine("hot", 25))

    assert _check(repo) == 1

    out = capsys.readouterr().out
    assert "new hotspot src/pyrite/other.py:1 hot CC 25" in out
    assert "possibly moved/renamed from src/pyrite/hot.py::hot" in out


def test_write_keeps_exception_notes_for_surviving_routines(repo) -> None:
    path = repo / complexity.BASELINE_PATH
    baseline = json.loads(path.read_text())
    baseline["routines"][0]["exception"] = "kernel; needs parity evidence"
    path.write_text(json.dumps(baseline))
    _rewrite(repo, "hot.py", _routine("hot", 24))

    complexity.run(repo, write=True)

    assert json.loads(path.read_text())["routines"] == [
        {
            "path": "src/pyrite/hot.py",
            "name": "hot",
            "cc": 24,
            "exception": "kernel; needs parity evidence",
        }
    ]


def test_radon_version_drift_and_parse_failures_fail_the_check(repo, capsys) -> None:
    path = repo / complexity.BASELINE_PATH
    baseline = json.loads(path.read_text())
    baseline["tool"]["radon"] = "0.0"
    path.write_text(json.dumps(baseline))
    _rewrite(repo, "other.py", "def broken(:\n")

    assert _check(repo) == 1

    out = capsys.readouterr().out
    assert "parse failure: src/pyrite/other.py" in out
    assert "baseline was produced with radon 0.0" in out


def test_missing_baseline_and_test_scope_ratchet_are_actionable(tmp_path) -> None:
    root = _checkout(tmp_path, {"ok.py": _routine("ok", 2)})
    (root / "tests").mkdir()

    with pytest.raises(complexity.ComplexityToolError, match="complexity --write"):
        _check(root)
    with pytest.raises(complexity.ComplexityToolError, match="report-only"):
        complexity.run(root, scope="tests", check=True)


def test_missing_radon_names_the_pinned_group(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "radon", None)

    with pytest.raises(complexity.ComplexityToolError, match="`lint` dependency group"):
        complexity.radon_modules()


def test_report_json_carries_provenance_ranks_and_missing_scc(repo, monkeypatch, capsys) -> None:
    monkeypatch.delenv(complexity.SCC_ENV, raising=False)
    monkeypatch.setattr(complexity.shutil, "which", lambda _name: None)

    assert complexity.run(repo, as_json=True) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["provenance"]["scope"] == "src/pyrite"
    assert report["inventory"]["modules"] == 2
    assert "scc not found" in report["inventory"]["scc"]["error"]
    assert report["cc"]["ranks"] == {"A": 2, "B": 0, "C": 0, "D": 1, "E": 0, "F": 0}
    assert report["cc"]["hotspots"] == ["src/pyrite/hot.py:1 hot CC 25"]


def test_scc_inventory_reads_python_totals(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "scc"
    payload = json.dumps(
        [{"Name": "Python", "Count": 2, "Lines": 9, "Code": 7, "Comment": 1, "Blank": 1}]
    )
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo "scc version 9.9.9"; exit 0; fi\n'
        f"echo '{payload}'\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv(complexity.SCC_ENV, str(fake))

    inventory = complexity.scc_inventory(tmp_path)

    assert inventory["version"] == "9.9.9"
    assert (inventory["count"], inventory["code"], inventory["complexity"]) == (2, 7, 0)


def test_dev_cli_maps_tool_errors_to_exit_messages(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise complexity.ComplexityToolError("radon is not installed")

    monkeypatch.setattr(complexity, "run", fail)

    with pytest.raises(SystemExit, match="complexity: radon is not installed"):
        dev_cli.cmd_complexity(Namespace(check=True))


def test_dev_cli_parser_forwards_complexity_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(complexity, "run", lambda root, **kwargs: calls.append(kwargs) or 0)

    dev_cli.main(["complexity", "--scope", "tests", "--top", "3", "--churn", "50"])

    assert calls == [
        {
            "scope": "tests",
            "check": False,
            "write": False,
            "as_json": False,
            "top": 3,
            "churn": 50,
        }
    ]
