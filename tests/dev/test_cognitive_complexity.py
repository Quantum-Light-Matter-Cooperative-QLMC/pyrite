"""Cognitive reporting is advisory and preserves analyzer scope differences."""

import json

import pytest

from pyrite.devtools import complexity


def test_cognitive_locations_and_nested_scopes(tmp_path, capsys):
    package = tmp_path / "src" / "pyrite"
    package.mkdir(parents=True)
    (package / "sample.py").write_text(
        "class A:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return 1\n"
        "    @x.setter\n"
        "    def x(self, v):\n"
        "        if v: pass\n"
        "def outer(x):\n"
        "    def inner():\n"
        "        if x: pass\n"
        "    return inner\n"
    )
    assert complexity.run(tmp_path, metric="cognitive", as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert "cc" not in report
    cognitive = report["cognitive"]
    assert cognitive["matched_radon"] == 3
    assert cognitive["unmatched_radon"] == 1
    assert cognitive["unmatched_cognitive"] == 0
    assert [row["radon_match"] for row in cognitive["per_routine"]] == ["A.x", "A.x#2", "outer"]


def test_cognitive_modes_do_not_change_ratchet(tmp_path):
    with pytest.raises(complexity.ComplexityToolError, match="report-only"):
        complexity.run(tmp_path, metric="cognitive", check=True)
    with pytest.raises(complexity.ComplexityToolError, match="report-only"):
        complexity.run(tmp_path, suggest=True, write=True)
    with pytest.raises(complexity.ComplexityToolError, match="requires"):
        complexity.run(tmp_path, metric="cyclomatic", suggest=True)
