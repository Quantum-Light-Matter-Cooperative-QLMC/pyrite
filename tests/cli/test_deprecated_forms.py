"""Every deprecated argument form still runs, warns once and keeps its exit code (#1078)."""

import json

import pytest

from pyrite.cli import command
from pyrite.cli._deprecations import DEPRECATED_FORMS
from tests.helpers.cli import invoke

# One invocation per registered form; "campaign" is a user-layer profile.
_INVOCATIONS = {
    "numerics show PROFILE": ["numerics", "show", "campaign"],
    "numerics set PROFILE": ["numerics", "set", "campaign", "--cxr-histories", "2000"],
    "numerics reset PROFILE": ["numerics", "reset", "campaign", "cxr-histories"],
    "numerics validate PROFILE": ["numerics", "validate", "campaign"],
    "physics show PROFILE": ["physics", "show", "campaign"],
    "physics set PROFILE": ["physics", "set", "campaign", "--straggling"],
    "physics reset PROFILE": ["physics", "reset", "campaign", "straggling"],
    "physics validate PROFILE": ["physics", "validate", "campaign"],
    "profile numerics reset PHYSICS-FIELD": [
        "profile",
        "numerics",
        "reset",
        "campaign",
        "straggling",
    ],
}


@pytest.fixture
def campaign():
    created = invoke(command, ["profile", "create", "campaign", "--from", "quickstart"])
    assert created.exit_code == 0, created.output


def test_every_registered_form_is_exercised():
    assert _INVOCATIONS.keys() == DEPRECATED_FORMS.keys()


@pytest.mark.parametrize("form", sorted(_INVOCATIONS))
def test_deprecated_form_runs_and_warns_once(campaign, form):
    result = invoke(command, _INVOCATIONS[form])

    assert result.exit_code == 0, result.output
    warnings = [line for line in result.stderr.splitlines() if "is deprecated" in line]
    entry = DEPRECATED_FORMS[form]
    prefix = (
        f"warning: 'pyrite {entry.path}' is deprecated and will be removed in "
        f"{entry.remove_in}; use 'pyrite profile "
    )
    assert len(warnings) == 1
    assert warnings[0].startswith(prefix), warnings[0]


@pytest.mark.parametrize("noun", ["numerics", "physics"])
def test_deprecated_show_keeps_json_stdout_clean(campaign, noun):
    result = invoke(command, [noun, "show", "campaign", "-o", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema"] == f"pyrite.{noun}.show"
    assert "is deprecated" in result.stderr


@pytest.mark.parametrize("noun", ["numerics", "physics"])
def test_unknown_name_keeps_runtime_exit_code_without_warning(noun):
    result = invoke(command, [noun, "show", "no-such-profile-or-preset"])

    assert result.exit_code == 1
    assert "is deprecated" not in result.stderr
