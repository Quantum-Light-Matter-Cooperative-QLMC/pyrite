"""``pyrite profile line-grid`` against an isolated copy of the bundled catalog."""

import json

import pytest

from pyrite.campaign.config import material_sweep
from pyrite.cli import _catalog_io
from pyrite.cli import command as root_command
from pyrite.materials.catalog import load_material_catalog
from tests.helpers.cli import assert_clean_result, assert_table_row, invoke
from tests.helpers.user_catalog import copy_full_catalog

_FULL_POLICY = {
    "bandwidth": "resonance-population",
    "resolution": "resonance-local",
    "quadrature": "bin-mean",
}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "catalog"
    copy_full_catalog(root)
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", root)
    return root


def _profile_file(catalog, name):
    return (catalog / "profiles" / f"{name}.toml").read_text()


def _stored(catalog, name):
    return load_material_catalog(catalog, profile=name).profile_line_grid_policy(name)


def _line_grid(*args):
    return invoke(root_command, ["profile", "line-grid", *args])


def test_help_lists_selectors_defaults_combinations_and_identity():
    group = invoke(root_command, ["profile", "line-grid", "--help"])
    assert_clean_result(group)
    for text in (
        "kinematic-ceiling | resonance-population",
        "sinc-nyquist | resonance-local",
        "node | bin-mean",
        "resonance-local requires bandwidth resonance-population and quadrature",
        "identity",
    ):
        assert text in " ".join(group.stdout.split())

    setting = invoke(root_command, ["profile", "line-grid", "set", "--help"])
    assert_clean_result(setting)
    flat = " ".join(setting.stdout.split())
    for default in ("kinematic-ceiling", "sinc-nyquist", "node"):
        assert f"Default {default}." in flat

    root = invoke(root_command, ["profile", "--help"])
    assert_clean_result(root)
    assert "--observation-angle" not in root.stdout
    assert "line-grid" in root.stdout


def test_show_reports_explicit_effective_and_grid_source(catalog):
    shown = _line_grid("show", "high_energy")
    assert_clean_result(shown)
    assert "bandwidth: resonance-population (profile); explicit: resonance-population" in (
        shown.stdout
    )
    assert "automatic resolution for every case" in shown.stdout

    default = _line_grid("show", "sub_100keV", "-o", "json")
    assert_clean_result(default)
    envelope = json.loads(default.stdout)
    assert envelope["schema"] == "cxr.profile.line-grid.show"
    payload = envelope["payload"]
    assert payload["explicit"] is None
    assert {row["key"]: (row["effective"], row["source"]) for row in payload["fields"]} == {
        "bandwidth": ("kinematic-ceiling", "built-in default"),
        "resolution": ("sinc-nyquist", "built-in default"),
        "quadrature": ("node", "built-in default"),
    }
    assert payload["grid_source"]["automatic_for_every_case"] is False

    profile_json = invoke(root_command, ["profile", "show", "high_energy", "-o", "json"])
    assert_clean_result(profile_json)
    assert json.loads(profile_json.stdout)["payload"]["line_grid_policy"] == _FULL_POLICY
    profile_text = invoke(root_command, ["profile", "show", "sub_100keV"])
    assert "Line-grid policy (profile line-grid)" in profile_text.stdout
    assert_table_row(profile_text.stdout, "policy", "none (explicit, stored or built-in grids)")


def test_set_dry_run_write_reload_and_case_policy(catalog):
    before = _profile_file(catalog, "sub_100keV")
    dry_run = _line_grid("set", "sub_100keV", "--quadrature", "bin-mean", "--dry-run")
    assert_clean_result(dry_run)
    assert '+quadrature = "bin-mean"' in dry_run.stdout
    assert _profile_file(catalog, "sub_100keV") == before

    written = _line_grid("set", "sub_100keV", "--quadrature", "bin-mean")
    assert_clean_result(
        written, stdout="updated line-grid policy for profile sub_100keV: quadrature=bin-mean\n"
    )
    assert _stored(catalog, "sub_100keV") == {"quadrature": "bin-mean"}

    again = _line_grid("set", "sub_100keV", "--quadrature", "bin-mean")
    assert_clean_result(again, stdout="profile sub_100keV: line-grid policy already set\n")

    full = _line_grid(
        "set",
        "sub_100keV",
        "--bandwidth",
        "resonance-population",
        "--resolution",
        "resonance-local",
    )
    assert_clean_result(full)
    assert _stored(catalog, "sub_100keV") == _FULL_POLICY


def test_invalid_combinations_are_usage_errors_before_writing(catalog):
    before = _profile_file(catalog, "sub_100keV")

    local = _line_grid("set", "sub_100keV", "--resolution", "resonance-local")
    assert local.exit_code == 2
    assert "needs the resonance-population bandwidth" in local.stderr
    assert "--bandwidth resonance-population --quadrature bin-mean" in local.stderr

    node = _line_grid(
        "set",
        "sub_100keV",
        "--bandwidth",
        "resonance-population",
        "--resolution",
        "resonance-local",
        "--quadrature",
        "node",
    )
    assert node.exit_code == 2
    assert "needs bin-mean quadrature" in node.stderr

    unknown = _line_grid("set", "sub_100keV", "--bandwidth", "coverage-0.95")
    assert unknown.exit_code == 2

    empty = _line_grid("set", "sub_100keV")
    assert empty.exit_code == 2
    assert _profile_file(catalog, "sub_100keV") == before


def test_coherent_profile_refuses_any_policy(catalog):
    before = _profile_file(catalog, "hopg_short")
    refused = _line_grid("set", "hopg_short", "--bandwidth", "kinematic-ceiling")
    assert refused.exit_code == 2
    assert "both emission route does not support" in refused.stderr
    assert _profile_file(catalog, "hopg_short") == before


def test_coherent_profile_accepts_a_windowed_policy(catalog):
    """#350: feature windows resolve the coherent route, so the editor allows them."""
    path = catalog / "profiles" / "hopg_short.toml"
    path.write_text(path.read_text() + "\n[line_grid_policy]\nwindows = true\n")

    result = _line_grid("set", "hopg_short", "--bandwidth", "kinematic-ceiling")

    assert result.exit_code == 0, result.stderr
    # Editing a selector keeps the TOML-only window opt-in.
    assert _stored(catalog, "hopg_short") == {"windows": True, "bandwidth": "kinematic-ceiling"}


def test_reset_partial_conflict_named_and_whole(catalog):
    conflict = _line_grid("reset", "high_energy", "bandwidth", "-y")
    assert conflict.exit_code == 2
    assert "reset resolution too" in conflict.stderr

    named = _line_grid("reset", "high_energy", "resolution", "-y")
    assert_clean_result(named, stdout="reset line-grid resolution for profile high_energy\n")
    assert _stored(catalog, "high_energy") == {
        "bandwidth": "resonance-population",
        "quadrature": "bin-mean",
    }

    dry_run = _line_grid("reset", "high_energy", "--dry-run")
    assert_clean_result(dry_run)
    assert "line_grid_policy]" in dry_run.stdout
    assert '-bandwidth = "resonance-population"' in dry_run.stdout

    whole = _line_grid("reset", "high_energy", "-y")
    assert_clean_result(whole, stdout="removed line-grid policy for profile high_energy\n")
    assert "line_grid_policy" not in _profile_file(catalog, "high_energy")
    assert _stored(catalog, "high_energy") is None

    nothing = _line_grid("reset", "high_energy")
    assert_clean_result(nothing, stdout="profile high_energy: nothing to reset\n")


def test_standard_edit_prompts_and_clone_carries_policy(catalog):
    declined = invoke(
        root_command,
        ["profile", "line-grid", "set", "standard", "--quadrature", "bin-mean"],
        input="n\n",
    )
    assert declined.exit_code == 1
    assert "line_grid_policy" not in _profile_file(catalog, "standard")

    cloned = invoke(root_command, ["profile", "create", "he_copy", "--from", "high_energy"])
    assert_clean_result(cloned)
    assert _stored(catalog, "he_copy") == _FULL_POLICY

    fresh = invoke(root_command, ["profile", "create", "fresh"])
    assert_clean_result(fresh)
    assert _stored(catalog, "fresh") is None


def test_unknown_profile_is_a_runtime_error(catalog):
    missing = _line_grid("show", "nope")
    assert missing.exit_code == 1
    assert "unknown profile: nope" in missing.stderr


def test_profile_policy_reaches_the_sweep(catalog, monkeypatch):
    written = _line_grid("set", "sub_100keV", "--quadrature", "bin-mean")
    assert_clean_result(written)
    monkeypatch.setenv("PYRITE_CATALOG", str(catalog))
    load_material_catalog.cache_clear()
    sweep = material_sweep("hopg", catalog_profile="sub_100keV")
    assert sweep.line_grid_policy == {"quadrature": "bin-mean"}


def test_emission_and_numerics_edits_respect_an_existing_policy(catalog):
    before = _profile_file(catalog, "high_energy")
    coherent = invoke(root_command, ["profile", "set", "high_energy", "--emission", "coherent"])
    assert coherent.exit_code == 1
    assert "coherent emission route does not support" in coherent.stderr

    added = invoke(root_command, ["profile", "add", "high_energy", "--coherent"])
    assert added.exit_code == 1

    substeps = invoke(
        root_command,
        [
            "profile",
            "numerics",
            "set",
            "high_energy",
            "--energy-model",
            "midpoint",
            "--maximum-fractional-energy-loss",
            "0.02",
        ],
    )
    assert substeps.exit_code == 1
    assert "bin-mean quadrature is incompatible with a positive max_dE_frac" in substeps.stderr
    assert _profile_file(catalog, "high_energy") == before
