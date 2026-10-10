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


def test_precision_show_reports_the_default_policy_without_a_table(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert_clean_result(shown)
    assert shown.stdout.startswith("[sub_100keV precision] adaptive (default policy)\n")
    assert "  target-rse: 0.05 (default)\n" in shown.stdout
    assert "  max-electrons: 20000 (default)\n" in shown.stdout
    machine = invoke(root_command, ["profile", "precision", "show", "sub_100keV", "-o", "json"])
    assert_clean_result(machine)
    envelope = json.loads(machine.stdout)
    assert envelope["schema"] == "cxr.profile.precision.show"
    payload = envelope["payload"]
    assert (payload["mode"], payload["source"], payload["explicit"]) == (
        "adaptive",
        "default",
        None,
    )
    assert payload["effective"]["min_electrons"] == 200
    summary = invoke(profile.command, ["show", "sub_100keV"])
    assert "electron counts: adaptive (default), target RSE 0.05, 200-20000 electrons" in (
        summary.stdout
    )


def test_precision_show_names_why_a_profile_keeps_fixed_counts(tmp_path, monkeypatch):
    for extra, reason in (
        ("n_electrons = { values = [450] }\n", "the profile sets fixed electron counts"),
        ('emission = "both"\n', "emission is 'both'"),
    ):
        text = _CATALOG.replace("[profiles.sub_100keV]\n", f"[profiles.sub_100keV]\n{extra}")
        _catalog(tmp_path, monkeypatch, text)
        shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
        assert_clean_result(shown)
        assert shown.stdout.startswith("[sub_100keV precision] fixed electron counts: ")
        assert reason in shown.stdout
        machine = json.loads(
            invoke(profile.command, ["precision", "show", "sub_100keV", "-o", "json"]).stdout
        )
        assert machine["payload"]["mode"] == "fixed"
        assert reason in machine["payload"]["fixed_reason"]


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
    assert shown.stdout.startswith("[sub_100keV precision] adaptive (profile policy)\n")
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
    assert_clean_result(
        field,
        stdout=(
            "reset precision fields for profile sub_100keV\n"
            "profile sub_100keV: adaptive (profile policy)\n"
        ),
    )
    assert "max_electron_share" not in _precision(catalog)
    required = invoke(profile.command, ["precision", "reset", "sub_100keV", "target-rse"])
    assert required.exit_code == 1
    assert "reset without fields" in required.stderr

    removed = invoke(profile.command, ["precision", "reset", "sub_100keV"])
    assert_clean_result(
        removed,
        stdout=(
            "removed precision policy for profile sub_100keV\n"
            "profile sub_100keV: adaptive (default policy)\n"
        ),
    )
    assert _precision(catalog) is None
    again = invoke(profile.command, ["precision", "reset", "sub_100keV"])
    assert_clean_result(again, stdout="profile sub_100keV: nothing to reset\n")


def test_precision_set_validates_at_the_boundary(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    band = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY, "--band-ev", "9,1"])
    assert band.exit_code == 2
    assert "START,STOP" in band.stderr

    blocks = [*_POLICY[:-1], "30"]
    misaligned = invoke(profile.command, ["precision", "set", "sub_100keV", *blocks])
    assert misaligned.exit_code == 1
    assert "multiples of block_electrons" in misaligned.stderr

    nothing = invoke(profile.command, ["precision", "set", "sub_100keV"])
    assert nothing.exit_code == 2
    assert "pyrite profile precision enable sub_100keV" in nothing.stderr
    assert catalog.read_text() == _CATALOG


def test_precision_set_refuses_fixed_counts_and_coherent_profiles(tmp_path, monkeypatch):
    fixed = _CATALOG.replace(
        "[profiles.sub_100keV]\n", "[profiles.sub_100keV]\nn_electrons = { values = [450] }\n"
    )
    _catalog(tmp_path, monkeypatch, fixed)
    refused = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "line-trials 450" in refused.stderr
    assert "pyrite profile precision enable sub_100keV" in refused.stderr
    assert "line-electrons" not in refused.stderr

    coherent = _CATALOG.replace(
        "[profiles.sub_100keV]\n", '[profiles.sub_100keV]\nemission = "coherent"\n'
    )
    _catalog(tmp_path, monkeypatch, coherent)
    refused = invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "adaptive is incoherent-only" in refused.stderr


def test_precision_standard_mutation_requires_confirmation(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    declined = invoke(profile.command, ["precision", "set", "standard", *_POLICY], input="n\n")
    assert declined.exit_code == 1
    assert catalog.read_text() == _CATALOG
    accepted = invoke(profile.command, ["precision", "set", "standard", *_POLICY, "--yes"])
    assert_clean_result(accepted)


def _fixed(counts="n_electrons = { values = [500] }\nn_electrons_brem = { values = [100] }\n"):
    return _CATALOG.replace("[profiles.sub_100keV]\n", f"[profiles.sub_100keV]\n{counts}")


def _profile(catalog):
    return tomlkit.parse(catalog.read_text())["profiles"]["sub_100keV"]


def test_first_partial_set_starts_from_the_default_policy(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    first = invoke(profile.command, ["precision", "set", "sub_100keV", "--observable", "line"])
    assert_clean_result(first, stdout="updated precision for profile sub_100keV\n")
    assert _precision(catalog).unwrap() == {
        "target_rse": 0.05,
        "min_electrons": 200,
        "max_electrons": 20000,
        "block_electrons": 100,
        "observables": ["line"],
    }

    retained = invoke(profile.command, ["precision", "set", "sub_100keV", "--target-rse", "0.1"])
    assert_clean_result(retained)
    assert _precision(catalog).unwrap()["observables"] == ["line"]
    assert _precision(catalog).unwrap()["target_rse"] == 0.1


def test_enable_replaces_fixed_counts_atomically(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _fixed())

    preview = invoke(profile.command, ["precision", "enable", "sub_100keV", "--dry-run"])
    assert_clean_result(preview)
    assert "-n_electrons = { values = [500] }" in preview.stdout
    assert catalog.read_text() == _fixed()

    enabled = invoke(profile.command, ["precision", "enable", "sub_100keV", "--observable", "line"])
    assert_clean_result(
        enabled,
        stdout=(
            "enabled adaptive precision for profile sub_100keV\n"
            "  removed fixed counts: line-trials 500, brem-trials 100\n"
            "profile sub_100keV: adaptive (profile policy)\n"
        ),
    )
    row = _profile(catalog)
    assert "n_electrons" not in row and "n_electrons_brem" not in row
    assert row["precision"].unwrap()["observables"] == ["line"]
    assert row["precision"].unwrap()["max_electrons"] == 20000

    again = invoke(profile.command, ["precision", "enable", "sub_100keV"])
    assert_clean_result(again, stdout="profile sub_100keV: already adaptive; nothing to change\n")


def test_enable_without_options_selects_the_default_policy(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _fixed())

    enabled = invoke(profile.command, ["precision", "enable", "sub_100keV"])
    assert_clean_result(enabled)
    assert enabled.stdout.endswith("profile sub_100keV: adaptive (default policy)\n")
    assert "precision" not in _profile(catalog)


def test_enable_refuses_unsupported_profiles_without_writing(tmp_path, monkeypatch):
    for extra, reason in (
        ('emission = "coherent"\n', "adaptive is incoherent-only"),
        ("secondary_threshold_eV = 1000.0\n", "particle cascades"),
        ("positron_transport = true\n", "positron transport"),
    ):
        text = _fixed(f"n_electrons = {{ values = [500] }}\n{extra}")
        catalog = _catalog(tmp_path, monkeypatch, text)
        refused = invoke(profile.command, ["precision", "enable", "sub_100keV"])
        assert refused.exit_code == 1
        assert reason in refused.stderr
        assert catalog.read_text() == text

    detector = _CATALOG.replace(
        "\n[materials.hopg]",
        "\n[profiles.sub_100keV.physical_detector]\ndistance_mm = 300.0\n\n[materials.hopg]",
    )
    catalog = _catalog(tmp_path, monkeypatch, detector)
    refused = invoke(profile.command, ["precision", "set", "sub_100keV", "--target-rse", "0.1"])
    assert refused.exit_code == 1
    assert "physical detector" in refused.stderr
    assert catalog.read_text() == detector

    override = _CATALOG.replace(
        "\n[materials.hopg]",
        "\n[profiles.sub_100keV.overrides.hopg]\nn_electrons = { values = [900] }\n"
        "\n[materials.hopg]",
    )
    catalog = _catalog(tmp_path, monkeypatch, override)
    refused = invoke(profile.command, ["precision", "enable", "sub_100keV"])
    assert refused.exit_code == 1
    assert "[overrides.hopg] sets line-trials" in refused.stderr
    assert catalog.read_text() == override
    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert "adaptive unsupported: [overrides.hopg] sets line-trials" in shown.stdout


def test_disable_selects_fixed_counts_with_defaults_or_flags(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])

    preview = invoke(profile.command, ["precision", "disable", "sub_100keV", "--dry-run"])
    assert_clean_result(preview)
    assert "+n_electrons = {values = [300]}" in preview.stdout
    assert "precision" in _profile(catalog)

    disabled = invoke(profile.command, ["precision", "disable", "sub_100keV"])
    assert_clean_result(
        disabled,
        stdout=(
            "selected fixed electron counts for profile sub_100keV\n"
            "  removed precision policy\n"
            "  counts: line-trials 300, brem-trials 150\n"
        ),
    )
    row = _profile(catalog)
    assert "precision" not in row
    assert (row["n_electrons"]["values"], row["n_electrons_brem"]["values"]) == ([300], [150])
    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert "  counts: line-trials 300, brem-trials 150\n" in shown.stdout
    assert "to switch to adaptive: pyrite profile precision enable sub_100keV" in shown.stdout

    changed = invoke(
        profile.command, ["precision", "disable", "sub_100keV", "--line-trials", "800"]
    )
    assert_clean_result(changed)
    assert _profile(catalog)["n_electrons"]["values"] == [800]
    assert _profile(catalog)["n_electrons_brem"]["values"] == [150]

    same = invoke(profile.command, ["precision", "disable", "sub_100keV"])
    assert_clean_result(
        same, stdout="profile sub_100keV: already uses fixed counts; nothing to change\n"
    )


def test_disable_keeps_count_sweeps_and_prompts_only_on_a_terminal(tmp_path, monkeypatch):
    from pyrite.cli.commands import _profile_precision

    grid = "n_electrons = { values = [100, 200] }\n"
    catalog = _catalog(tmp_path, monkeypatch, _fixed(grid))
    monkeypatch.setattr(_profile_precision, "_interactive", lambda: True)

    prompted = invoke(profile.command, ["precision", "disable", "sub_100keV"], input="75\n")
    assert prompted.exit_code == 0
    assert "brem-trials [150]" in prompted.stderr
    assert "line-trials" not in prompted.stderr
    row = _profile(catalog)
    assert row["n_electrons"]["values"] == [100, 200]
    assert row["n_electrons_brem"]["values"] == [75]

    _catalog(tmp_path, monkeypatch)
    bypassed = invoke(
        profile.command,
        ["precision", "disable", "sub_100keV", "--line-trials", "40", "--brem-trials", "20"],
    )
    assert bypassed.exit_code == 0
    assert "[" not in bypassed.stderr


def test_numerics_counts_on_an_adaptive_profile_point_at_disable(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(profile.command, ["precision", "set", "sub_100keV", *_POLICY])
    before = catalog.read_text()

    refused = invoke(profile.command, ["numerics", "set", "sub_100keV", "--line-trials", "500"])
    assert refused.exit_code == 1
    assert "pyrite profile precision disable sub_100keV" in refused.stderr
    assert catalog.read_text() == before


def test_show_reports_line_grid_independently_and_next_steps(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert "line grid: explicit or stored grids first" in shown.stdout
    assert "to select fixed counts: pyrite profile precision disable sub_100keV" in shown.stdout
    machine = json.loads(
        invoke(profile.command, ["precision", "show", "sub_100keV", "-o", "json"]).stdout
    )["payload"]
    assert machine["fixed_counts"] == {}
    assert machine["blockers"] == []
    assert machine["line_grid_automatic_for_every_case"] is False

    _catalog(
        tmp_path,
        monkeypatch,
        _fixed().replace(
            "\n[materials.hopg]",
            '\n[profiles.sub_100keV.line_grid_policy]\nbandwidth = "resonance-population"\n'
            "\n[materials.hopg]",
        ),
    )
    machine = json.loads(
        invoke(profile.command, ["precision", "show", "sub_100keV", "-o", "json"]).stdout
    )["payload"]
    assert machine["fixed_counts"] == {"line_trials": [500], "brem_trials": [100]}
    assert machine["line_grid_automatic_for_every_case"] is True
    enabled = invoke(profile.command, ["precision", "enable", "sub_100keV"])
    assert_clean_result(enabled)
    shown = invoke(profile.command, ["precision", "show", "sub_100keV"])
    assert "line grid: automatic for every case (line-grid policy)" in shown.stdout
