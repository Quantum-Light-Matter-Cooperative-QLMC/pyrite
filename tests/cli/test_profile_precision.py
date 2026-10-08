import json

import tomlkit

from pyrite.cli import command as root_command
from pyrite.cli.commands import profile
from tests.cli.test_profile import _CATALOG, _catalog
from tests.helpers.cli import assert_clean_result, invoke

_POLICY = [
    "--target-rse",
    "0.1",
    "--min-electrons",
    "200",
    "--max-electrons",
    "2000",
    "--block-electrons",
    "20",
]


def _precision(catalog):
    return tomlkit.parse(catalog.read_text())["profiles"]["sub_100keV"].get("precision")


def test_precision_show_reports_fixed_counts_without_a_policy(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert_clean_result(
        shown, stdout="[sub_100keV precision] fixed electron counts (no adaptive policy)\n"
    )
    machine = invoke(root_command, ["profile", "precision", "show", "sub_100keV", "-o", "json"])
    assert_clean_result(machine)
    envelope = json.loads(machine.stdout)
    assert envelope["schema"] == "cxr.profile.precision.show"
    assert envelope["payload"] == {
        "profile": "sub_100keV",
        "mode": "fixed",
        "explicit": None,
        "effective": None,
    }


def test_precision_set_show_and_reset_round_trip(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    dry_run = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY, "--dry-run"])
    assert_clean_result(dry_run)
    assert "+target_rse = 0.1" in dry_run.stdout
    assert catalog.read_text() == _CATALOG

    written = invoke(
        profile.command,
        [
            "precision",
            "set",
            "sub_100keV",
            *_POLICY,
            "--observable",
            "line",
            "--observable",
            "brem",
            "--band-ev",
            "100,900",
        ],
    )
    assert_clean_result(written, stdout="updated precision for profile sub_100keV\n")
    assert _precision(catalog).unwrap() == {
        "target_rse": 0.1,
        "min_electrons": 200,
        "max_electrons": 2000,
        "block_electrons": 20,
        "observables": ["line", "brem"],
        "band_eV": [100.0, 900.0],
    }

    updated = invoke(
        profile.command, ["precision", "set", "sub_100keV", "--max-electron-share", "0.02"]
    )
    assert_clean_result(updated)
    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert_clean_result(shown)
    assert "  target-rse: 0.1\n" in shown.stdout
    assert "  observable: line,brem\n" in shown.stdout
    assert "  max-electron-share: 0.02\n" in shown.stdout
    assert "  min-effective-electrons: 100 (default)\n" in shown.stdout
    assert "  pilot-electrons: none (default)\n" in shown.stdout
    machine = json.loads(
        invoke(profile.command, ["precision", "show", "sub_100keV", "-o", "json"]).stdout
    )
    assert machine["payload"]["mode"] == "adaptive"
    assert machine["payload"]["effective"]["stability_blocks"] == 3
    assert "stability_blocks" not in machine["payload"]["explicit"]
    summary = invoke(profile.command, ["show", "sub_100keV"])
    assert "electron counts: adaptive, target RSE 0.1, 200-2000 electrons" in summary.stdout

    field = invoke(profile.command, ["precision", "reset", "sub_100keV", "max-electron-share"])
    assert_clean_result(field, stdout="reset precision fields for profile sub_100keV\n")
    assert "max_electron_share" not in _precision(catalog)
    required = invoke(profile.command, ["precision", "reset", "sub_100keV", "target-rse"])
    assert required.exit_code == 1
    assert "reset without fields" in required.stderr

    removed = invoke(profile.command, ["precision", "reset", "sub_100keV"])
    assert_clean_result(removed, stdout="removed precision policy for profile sub_100keV\n")
    assert _precision(catalog) is None
    again = invoke(profile.command, ["precision", "reset", "sub_100keV"])
    assert_clean_result(again, stdout="profile sub_100keV: nothing to reset\n")


def test_precision_set_validates_at_the_boundary(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    partial = invoke(profile.command, ["precision", "set", "sub_100keV", "--target-rse", "0.1"])
    assert partial.exit_code == 2
    assert "--min-electrons, --max-electrons, --block-electrons" in partial.stderr

    band = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY, "--band-ev", "9,1"])
    assert band.exit_code == 2
    assert "START,STOP" in band.stderr

    blocks = [*_POLICY[:-1], "30"]
    misaligned = invoke(profile.command, ["precision", "set", "sub_100keV", *blocks])
    assert misaligned.exit_code == 1
    assert "multiples of block_electrons" in misaligned.stderr

    nothing = invoke(profile.command, ["precision", "set", "sub_100keV"])
    assert nothing.exit_code == 2
    assert catalog.read_text() == _CATALOG


def test_precision_set_refuses_fixed_counts_and_coherent_profiles(tmp_path, monkeypatch):
    fixed = _CATALOG.replace(
        "[profiles.sub_100keV]\n", "[profiles.sub_100keV]\nn_electrons = { values = [450] }\n"
    )
    _catalog(tmp_path, monkeypatch, fixed)
    refused = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "pyrite profile numerics reset sub_100keV" in refused.stderr

    coherent = _CATALOG.replace(
        "[profiles.sub_100keV]\n", '[profiles.sub_100keV]\nemission = "coherent"\n'
    )
    _catalog(tmp_path, monkeypatch, coherent)
    refused = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "incoherent emission only" in refused.stderr


def test_precision_standard_mutation_requires_confirmation(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    declined = invoke(profile.command, ["precision", "set", "standard", *_POLICY], input="n\n")
    assert declined.exit_code == 1
    assert catalog.read_text() == _CATALOG
    accepted = invoke(profile.command, ["precision", "set", "standard", *_POLICY, "--yes"])
    assert_clean_result(accepted)
