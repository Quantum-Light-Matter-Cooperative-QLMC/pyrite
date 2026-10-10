"""`pyrite profile` writes the user layer; `pyrite-dev profile` the bundled demos (#403)."""

import json

from pyrite._catalog_layout import bundled_catalog, source_files
from pyrite.cli.commands import profile
from pyrite.devtools import dev_cli
from tests.helpers.cli import assert_clean_result, invoke


def _bundled_snapshot():
    return {relative: file.read_bytes() for relative, file in source_files(bundled_catalog())}


def test_user_profile_edits_never_touch_the_installed_catalog(_isolate_user_catalog):
    before = {
        path.relative_to(bundled_catalog()): path.read_bytes()
        for path in bundled_catalog().rglob("*.toml")
    }

    created = invoke(profile.command, ["create", "mine", "--from", "quickstart"])
    edited = invoke(profile.command, ["set", "mine", "--energy", "40"])
    listed = invoke(profile.command, ["list", "-o", "json"])

    assert_clean_result(
        created, stdout="created profile mine from quickstart (inherited: detector)\n"
    )
    assert edited.exit_code == 0
    assert (
        "energy_keV = {values = [40.0]}"
        in (_isolate_user_catalog / "profiles" / "mine.toml").read_text()
    )
    sources = {
        row["name"]: row["source"] for row in json.loads(listed.stdout)["payload"]["profiles"]
    }
    assert sources["mine"] == "user" and sources["quickstart"] == "bundled"
    after = {
        path.relative_to(bundled_catalog()): path.read_bytes()
        for path in bundled_catalog().rglob("*.toml")
    }
    assert after == before


def test_editing_a_bundled_demo_points_at_create_from():
    result = invoke(profile.command, ["set", "quickstart", "--energy", "40"])

    assert result.exit_code == 1
    assert "bundled demo and is read-only" in result.stderr
    assert "pyrite profile create NAME --from quickstart" in result.stderr


def test_pyrite_dev_profile_targets_the_bundled_catalog(monkeypatch, capsys):
    calls = []

    def fake_run(command, argv, *, prog_name):
        from pyrite._catalog_layout import is_layered, selected_catalog

        calls.append((argv, prog_name, selected_catalog(), is_layered(bundled_catalog())))

    monkeypatch.setenv("PYRITE_CATALOG", "/elsewhere")
    monkeypatch.setattr(dev_cli, "_run_relocated_click", fake_run)

    dev_cli.main(["profile", "list"])

    assert calls == [(["list"], "pyrite-dev profile", bundled_catalog().resolve(), False)]
    import os

    assert os.environ["PYRITE_CATALOG"] == "/elsewhere"
