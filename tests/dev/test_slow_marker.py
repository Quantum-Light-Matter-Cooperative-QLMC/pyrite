"""The opt-in ``slow`` tier skips by default and runs on explicit request (#381)."""

import pytest

pytest_plugins = ["pytester"]

_TESTS = """
import pytest

@pytest.mark.slow
def test_heavy():
    pass

@pytest.mark.slow
@pytest.mark.parametrize("n", [0, 1])
def test_heavy_param(n):
    pass

def test_light():
    pass
"""


@pytest.fixture
def suite(pytester, monkeypatch):
    monkeypatch.delenv("PYRITE_SLOW_TESTS", raising=False)
    pytester.makeini("[pytest]\nmarkers =\n    slow: heavy test\n")
    pytester.makeconftest(
        "from tests.helpers.slow_marker import (  # noqa: F401\n"
        "    pytest_collection_modifyitems,\n"
        "    pytest_terminal_summary,\n"
        ")\n"
    )
    pytester.makepyfile(test_tier=_TESTS)
    return pytester


def test_default_run_skips_slow_tests_and_says_how_to_enable_them(suite) -> None:
    result = suite.runpytest_inprocess("-rs")

    result.assert_outcomes(passed=1, skipped=3)
    result.stdout.fnmatch_lines(
        ["3 slow tests skipped; run them with `pyrite-dev test --slow` or PYRITE_SLOW_TESTS=1"]
    )
    result.stdout.fnmatch_lines(
        ["SKIPPED * slow test: run with `pyrite-dev test --slow` or PYRITE_SLOW_TESTS=1*"]
    )


def test_env_var_runs_slow_tests(suite, monkeypatch) -> None:
    monkeypatch.setenv("PYRITE_SLOW_TESTS", "1")

    suite.runpytest_inprocess().assert_outcomes(passed=4)


@pytest.mark.parametrize(
    ("markexpr", "outcomes"),
    [
        ("slow", {"passed": 3}),
        ("slow or not slow", {"passed": 4}),
        ("not slow", {"passed": 1, "deselected": 3}),
    ],
)
def test_marker_expression_naming_slow_decides_selection(suite, markexpr, outcomes) -> None:
    result = suite.runpytest_inprocess("-m", markexpr)

    result.assert_outcomes(**outcomes)


@pytest.mark.parametrize(
    ("node", "passed"),
    [("test_tier.py::test_heavy", 1), ("test_tier.py::test_heavy_param", 2)],
)
def test_naming_a_slow_test_by_node_id_runs_it(suite, node, passed) -> None:
    suite.runpytest_inprocess(node).assert_outcomes(passed=passed)


def test_naming_one_parametrization_runs_only_it(suite) -> None:
    result = suite.runpytest_inprocess("test_tier.py::test_heavy_param[1]", "test_tier.py")

    result.assert_outcomes(passed=2, skipped=2)
