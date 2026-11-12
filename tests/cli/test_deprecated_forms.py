"""The 0.32 argument forms are removed in 0.34; profile commands remain explicit."""

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


def test_no_argument_forms_remain_deprecated():
    assert not DEPRECATED_FORMS


@pytest.mark.parametrize("form", sorted(_INVOCATIONS))
def test_removed_form_rejects_profile(campaign, form):
    result = invoke(command, _INVOCATIONS[form])
    assert result.exit_code in (1, 2), result.output
    assert "is deprecated" not in result.stderr


@pytest.mark.parametrize("noun", ["numerics", "physics"])
def test_unknown_name_keeps_runtime_exit_code_without_warning(noun):
    result = invoke(command, [noun, "show", "no-such-profile-or-preset"])

    assert result.exit_code == 1
    assert "is deprecated" not in result.stderr


@pytest.mark.parametrize(
    "noun, child",
    [
        ("numerics", "preset"),
        ("physics", "preset"),
        ("numerics", "adaptive-sampling"),
        ("numerics", "cxr-grid"),
    ],
)
def test_removed_subgroups(noun, child):
    result = invoke(command, [noun, child, "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.stderr


@pytest.mark.parametrize(
    "flag",
    [
        "--straggling",
        "--inelastic-model",
        "--inelastic-cutoff-ev",
        "--secondary-threshold-ev",
        "--elastic-model",
        "--bremsstrahlung-model",
        "--radiative-model",
        "--radiative-cutoff-ev",
        "--pair-production-model",
        "--positron-transport",
        "--atomic-electron-deflection",
        "--transport-mode",
        "--mixed-c1",
        "--mixed-c2",
        "--shell-ionization-model",
        "--channeling-model",
        "--channeling-plane",
        "--channeling-admission",
    ],
)
def test_profile_numerics_rejects_physics_flags(campaign, flag):
    result = invoke(command, ["profile", "numerics", "set", "campaign", flag])
    assert result.exit_code == 2
    assert "No such option" in result.stderr
