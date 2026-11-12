import json

import pytest
import tomlkit

from pyrite.cli import command as root_command
from pyrite.cli.commands import profile
from pyrite.cli.commands.calculation import profile_numerics_command as numerics_command
from tests.cli.test_profile import _CATALOG, _catalog
from tests.helpers.cli import assert_clean_result, assert_row, assert_table_row, invoke

_POLICY = [
    "--target-rse",
    "0.1",
    "--min-histories",
    "200",
    "--max-histories",
    "2000",
    "--block-histories",
    "20",
]


def _precision(catalog):
    return tomlkit.parse(catalog.read_text())["profiles"]["sub_100keV"].get("adaptive_sampling")


def test_independent_brem_targets_roundtrip_and_reset(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    result = invoke(
        numerics_command,
        [
            "adaptive-sampling",
            "set",
            "sub_100keV",
            *_POLICY,
            "--brem-target-rse",
            "0.2",
            "--brem-min-histories",
            "100",
            "--brem-max-histories",
            "1000",
        ],
    )
    assert_clean_result(result)
    assert _precision(catalog)["brem_target_rse"] == 0.2
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV", "-o", "json"])
    assert_clean_result(shown)
    effective = json.loads(shown.stdout)["payload"]["effective"]
    assert effective["brem_min_histories"] == 100
    assert effective["brem_max_histories"] == 1000
    original = catalog.read_text()
    invalid = invoke(
        numerics_command,
        [
            "adaptive-sampling",
            "set",
            "sub_100keV",
            "--brem-max-histories",
            "2020",
        ],
    )
    assert invalid.exit_code != 0
    assert "must not exceed max_histories" in invalid.output
    assert catalog.read_text() == original
    reset = invoke(
        numerics_command,
        [
            "adaptive-sampling",
            "reset",
            "sub_100keV",
            "brem-target-rse",
            "brem-min-histories",
            "brem-max-histories",
        ],
    )
    assert_clean_result(reset)
    assert not any(key.startswith("brem_") for key in _precision(catalog))


def test_precision_show_reports_the_default_policy_without_a_table(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert_clean_result(shown)
    assert shown.stdout.startswith(
        "[sub_100keV adaptive-sampling]  mode: adaptive (default policy)\n"
    )
    assert_row(shown.stdout, "target-rse", "0.05", "default")
    assert_row(shown.stdout, "max-histories", "20000", "default")
    machine = invoke(
        root_command,
        ["profile", "numerics", "adaptive-sampling", "show", "sub_100keV", "-o", "json"],
    )
    assert_clean_result(machine)
    envelope = json.loads(machine.stdout)
    assert envelope["schema"] == "cxr.profile.adaptive_sampling.show"
    payload = envelope["payload"]
    assert (payload["mode"], payload["source"], payload["explicit"]) == (
        "adaptive",
        "default",
        None,
    )
    assert payload["effective"]["min_histories"] == 200
    summary = invoke(profile.command, ["show", "sub_100keV"])
    assert_table_row(summary.stdout, "mode", "adaptive (default policy)")
    assert_table_row(summary.stdout, "target-rse", "0.05")
    assert_table_row(summary.stdout, "min-histories", "200")
    assert_table_row(summary.stdout, "max-histories", "20000")


def test_precision_show_names_why_a_profile_keeps_fixed_counts(tmp_path, monkeypatch):
    for extra, reason in (
        ("cxr_histories = { values = [450] }\n", "the profile sets fixed history counts"),
        ('emission = "both"\n', "emission is 'both'"),
    ):
        text = _CATALOG.replace("[profiles.sub_100keV]\n", f"[profiles.sub_100keV]\n{extra}")
        _catalog(tmp_path, monkeypatch, text)
        shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
        assert_clean_result(shown)
        assert shown.stdout.startswith("[sub_100keV adaptive-sampling]  mode: fixed (")
        assert reason in shown.stdout
        machine = json.loads(
            invoke(
                numerics_command, ["adaptive-sampling", "show", "sub_100keV", "-o", "json"]
            ).stdout
        )
        assert machine["payload"]["mode"] == "fixed"
        assert reason in machine["payload"]["fixed_reason"]


def test_precision_set_show_and_reset_round_trip(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    dry_run = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY, "--dry-run"]
    )
    assert_clean_result(dry_run)
    assert "+target_rse = 0.1" in dry_run.stdout
    assert catalog.read_text() == _CATALOG

    written = invoke(
        numerics_command,
        [
            "adaptive-sampling",
            "set",
            "sub_100keV",
            *_POLICY,
            "--observable",
            "cxr",
            "--observable",
            "brem",
            "--band-ev",
            "100,900",
        ],
    )
    assert_clean_result(written, stdout="updated adaptive sampling for profile sub_100keV\n")
    assert _precision(catalog).unwrap() == {
        "target_rse": 0.1,
        "min_histories": 200,
        "max_histories": 2000,
        "block_histories": 20,
        "observables": ["cxr", "brem"],
        "band_eV": [100.0, 900.0],
    }

    updated = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", "--max-history-share", "0.02"]
    )
    assert_clean_result(updated)
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert_clean_result(shown)
    assert shown.stdout.startswith(
        "[sub_100keV adaptive-sampling]  mode: adaptive (profile policy)\n"
    )
    assert_row(shown.stdout, "target-rse", "0.1", "profile")
    assert_row(shown.stdout, "observable", "cxr,brem", "profile")
    assert_row(shown.stdout, "max-history-share", "0.02", "profile")
    assert_row(shown.stdout, "min-effective-histories", "100", "default")
    assert_row(shown.stdout, "pilot-histories", "none", "default")
    machine = json.loads(
        invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV", "-o", "json"]).stdout
    )
    assert machine["payload"]["mode"] == "adaptive"
    assert machine["payload"]["effective"]["stability_blocks"] == 3
    assert "stability_blocks" not in machine["payload"]["explicit"]
    summary = invoke(profile.command, ["show", "sub_100keV"])
    assert_table_row(summary.stdout, "mode", "adaptive (profile policy)")
    assert_table_row(summary.stdout, "target-rse", "0.1")
    assert_table_row(summary.stdout, "max-histories", "2000")

    field = invoke(
        numerics_command, ["adaptive-sampling", "reset", "sub_100keV", "max-history-share"]
    )
    assert_clean_result(
        field,
        stdout=(
            "reset adaptive-sampling fields for profile sub_100keV\n"
            "profile sub_100keV: adaptive (profile policy)\n"
        ),
    )
    assert "max_history_share" not in _precision(catalog)
    required = invoke(numerics_command, ["adaptive-sampling", "reset", "sub_100keV", "target-rse"])
    assert required.exit_code == 1
    assert "reset without fields" in required.stderr

    removed = invoke(numerics_command, ["adaptive-sampling", "reset", "sub_100keV"])
    assert_clean_result(
        removed,
        stdout=(
            "removed adaptive-sampling policy for profile sub_100keV\n"
            "profile sub_100keV: adaptive (default policy)\n"
        ),
    )
    assert _precision(catalog) is None
    again = invoke(numerics_command, ["adaptive-sampling", "reset", "sub_100keV"])
    assert_clean_result(again, stdout="profile sub_100keV: nothing to reset\n")


def test_precision_set_validates_at_the_boundary(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    band = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY, "--band-ev", "9,1"]
    )
    assert band.exit_code == 2
    assert "START,STOP" in band.stderr

    blocks = [*_POLICY[:-1], "30"]
    misaligned = invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV", *blocks])
    assert misaligned.exit_code == 1
    assert "multiples of block_histories" in misaligned.stderr

    nothing = invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV"])
    assert nothing.exit_code == 2
    assert "pyrite profile numerics adaptive-sampling enable sub_100keV" in nothing.stderr
    assert catalog.read_text() == _CATALOG


def test_precision_set_refuses_fixed_counts_and_coherent_profiles(tmp_path, monkeypatch):
    fixed = _CATALOG.replace(
        "[profiles.sub_100keV]\n", "[profiles.sub_100keV]\ncxr_histories = { values = [450] }\n"
    )
    _catalog(tmp_path, monkeypatch, fixed)
    refused = invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "cxr-histories 450" in refused.stderr
    assert "pyrite profile numerics adaptive-sampling enable sub_100keV" in refused.stderr
    assert "line-electrons" not in refused.stderr

    coherent = _CATALOG.replace(
        "[profiles.sub_100keV]\n", '[profiles.sub_100keV]\nemission = "coherent"\n'
    )
    _catalog(tmp_path, monkeypatch, coherent)
    refused = invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY])
    assert refused.exit_code == 1
    assert "adaptive is incoherent-only" in refused.stderr


def test_precision_standard_mutation_requires_confirmation(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    declined = invoke(
        numerics_command, ["adaptive-sampling", "set", "standard", *_POLICY], input="n\n"
    )
    assert declined.exit_code == 1
    assert catalog.read_text() == _CATALOG
    accepted = invoke(numerics_command, ["adaptive-sampling", "set", "standard", *_POLICY, "--yes"])
    assert_clean_result(accepted)


def _fixed(counts="cxr_histories = { values = [500] }\nbrem_histories = { values = [100] }\n"):
    return _CATALOG.replace("[profiles.sub_100keV]\n", f"[profiles.sub_100keV]\n{counts}")


def _profile(catalog):
    return tomlkit.parse(catalog.read_text())["profiles"]["sub_100keV"]


def test_first_partial_set_starts_from_the_default_policy(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    first = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", "--observable", "cxr"]
    )
    assert_clean_result(first, stdout="updated adaptive sampling for profile sub_100keV\n")
    assert _precision(catalog).unwrap() == {
        "target_rse": 0.05,
        "min_histories": 200,
        "max_histories": 20000,
        "block_histories": 100,
        "observables": ["cxr"],
    }

    retained = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", "--target-rse", "0.1"]
    )
    assert_clean_result(retained)
    assert _precision(catalog).unwrap()["observables"] == ["cxr"]
    assert _precision(catalog).unwrap()["target_rse"] == 0.1


def test_enable_replaces_fixed_counts_atomically(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _fixed())

    preview = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV", "--dry-run"])
    assert_clean_result(preview)
    assert "-cxr_histories = { values = [500] }" in preview.stdout
    assert catalog.read_text() == _fixed()

    enabled = invoke(
        numerics_command, ["adaptive-sampling", "enable", "sub_100keV", "--observable", "cxr"]
    )
    assert_clean_result(
        enabled,
        stdout=(
            "enabled adaptive sampling for profile sub_100keV\n"
            "  removed fixed counts: cxr-histories 500, brem-histories 100\n"
            "profile sub_100keV: adaptive (profile policy)\n"
        ),
    )
    row = _profile(catalog)
    assert "cxr_histories" not in row and "brem_histories" not in row
    assert row["adaptive_sampling"].unwrap()["observables"] == ["cxr"]
    assert row["adaptive_sampling"].unwrap()["max_histories"] == 20000

    again = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV"])
    assert_clean_result(again, stdout="profile sub_100keV: already adaptive; nothing to change\n")


def test_enable_without_options_selects_the_default_policy(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch, _fixed())

    enabled = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV"])
    assert_clean_result(enabled)
    assert enabled.stdout.endswith("profile sub_100keV: adaptive (default policy)\n")
    assert "adaptive_sampling" not in _profile(catalog)


def test_enable_refuses_unsupported_profiles_without_writing(tmp_path, monkeypatch):
    for extra, reason in (
        ('emission = "coherent"\n', "adaptive is incoherent-only"),
        ("secondary_threshold_eV = 1000.0\n", "particle cascades"),
        ("positron_transport = true\n", "positron transport"),
    ):
        text = _fixed(f"cxr_histories = {{ values = [500] }}\n{extra}")
        catalog = _catalog(tmp_path, monkeypatch, text)
        refused = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV"])
        assert refused.exit_code == 1
        assert reason in refused.stderr
        assert catalog.read_text() == text

    detector = _CATALOG.replace(
        "\n[materials.hopg]",
        "\n[profiles.sub_100keV.physical_detector]\ndistance_mm = 300.0\n\n[materials.hopg]",
    )
    catalog = _catalog(tmp_path, monkeypatch, detector)
    refused = invoke(
        numerics_command, ["adaptive-sampling", "set", "sub_100keV", "--target-rse", "0.1"]
    )
    assert refused.exit_code == 1
    assert "physical detector" in refused.stderr
    assert catalog.read_text() == detector

    override = _CATALOG.replace(
        "\n[materials.hopg]",
        "\n[profiles.sub_100keV.overrides.hopg]\ncxr_histories = { values = [900] }\n"
        "\n[materials.hopg]",
    )
    catalog = _catalog(tmp_path, monkeypatch, override)
    refused = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV"])
    assert refused.exit_code == 1
    assert "[overrides.hopg] sets cxr-histories" in refused.stderr
    assert catalog.read_text() == override
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert "adaptive unsupported: [overrides.hopg] sets cxr-histories" in shown.stdout


def test_disable_selects_fixed_counts_with_defaults_or_flags(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY])

    preview = invoke(numerics_command, ["adaptive-sampling", "disable", "sub_100keV", "--dry-run"])
    assert_clean_result(preview)
    assert "+cxr_histories = {values = [300]}" in preview.stdout
    assert "adaptive_sampling" in _profile(catalog)

    disabled = invoke(numerics_command, ["adaptive-sampling", "disable", "sub_100keV"])
    assert_clean_result(
        disabled,
        stdout=(
            "selected fixed history counts for profile sub_100keV\n"
            "  removed adaptive-sampling policy\n"
            "  counts: cxr-histories 300, brem-histories 150\n"
        ),
    )
    row = _profile(catalog)
    assert "adaptive_sampling" not in row
    assert (row["cxr_histories"]["values"], row["brem_histories"]["values"]) == ([300], [150])
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert_row(shown.stdout, "cxr-histories", "300", "profile")
    assert_row(shown.stdout, "brem-histories", "150", "profile")
    assert (
        "to switch to adaptive: pyrite profile numerics adaptive-sampling enable sub_100keV"
        in shown.stdout
    )

    changed = invoke(
        numerics_command, ["adaptive-sampling", "disable", "sub_100keV", "--cxr-histories", "800"]
    )
    assert_clean_result(changed)
    assert _profile(catalog)["cxr_histories"]["values"] == [800]
    assert _profile(catalog)["brem_histories"]["values"] == [150]

    same = invoke(numerics_command, ["adaptive-sampling", "disable", "sub_100keV"])
    assert_clean_result(
        same, stdout="profile sub_100keV: already uses fixed counts; nothing to change\n"
    )


def test_disable_keeps_count_sweeps_and_prompts_only_on_a_terminal(tmp_path, monkeypatch):
    from pyrite.cli.commands import _profile_precision

    grid = "cxr_histories = { values = [100, 200] }\n"
    catalog = _catalog(tmp_path, monkeypatch, _fixed(grid))
    monkeypatch.setattr(_profile_precision, "_interactive", lambda: True)

    prompted = invoke(
        numerics_command, ["adaptive-sampling", "disable", "sub_100keV"], input="75\n"
    )
    assert prompted.exit_code == 0
    assert "brem-histories [150]" in prompted.stderr
    assert "cxr-histories" not in prompted.stderr
    row = _profile(catalog)
    assert row["cxr_histories"]["values"] == [100, 200]
    assert row["brem_histories"]["values"] == [75]

    _catalog(tmp_path, monkeypatch)
    bypassed = invoke(
        numerics_command,
        [
            "adaptive-sampling",
            "disable",
            "sub_100keV",
            "--cxr-histories",
            "40",
            "--brem-histories",
            "20",
        ],
    )
    assert bypassed.exit_code == 0
    assert "[" not in bypassed.stderr


def test_numerics_counts_on_an_adaptive_profile_point_at_disable(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    invoke(numerics_command, ["adaptive-sampling", "set", "sub_100keV", *_POLICY])
    before = catalog.read_text()

    refused = invoke(profile.command, ["numerics", "set", "sub_100keV", "--cxr-histories", "500"])
    assert refused.exit_code == 1
    assert "pyrite profile numerics adaptive-sampling disable sub_100keV" in refused.stderr
    assert catalog.read_text() == before


def test_show_reports_line_grid_independently_and_next_steps(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert "CXR grid: explicit or stored grids first" in shown.stdout
    assert (
        "to select fixed counts: pyrite profile numerics adaptive-sampling disable sub_100keV"
        in shown.stdout
    )
    machine = json.loads(
        invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV", "-o", "json"]).stdout
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
        invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV", "-o", "json"]).stdout
    )["payload"]
    assert machine["fixed_counts"] == {"cxr_histories": [500], "brem_histories": [100]}
    assert machine["line_grid_automatic_for_every_case"] is True
    enabled = invoke(numerics_command, ["adaptive-sampling", "enable", "sub_100keV"])
    assert_clean_result(enabled)
    shown = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert "CXR grid: automatic for every case (cxr-grid policy)" in shown.stdout


@pytest.mark.parametrize(
    ("argv", "old", "new"),
    [
        *(
            (argv, f"--{kind}-{suffix}", f"--{kind}-histories")
            for argv in (
                ["profile", "numerics", "adaptive-sampling", "disable", "p"],
                ["profile", "numerics", "set", "p"],
                ["profile", "set", "p"],
                ["profile", "add", "p"],
                ["profile", "create", "p"],
            )
            for kind in ("cxr", "brem")
            for suffix in ("trials", "electrons")
        ),
        *(
            (["profile", "numerics", "adaptive-sampling", verb, "p"], f"--{old}", f"--{new}")
            for verb in ("set", "enable")
            for old, new in (
                ("min-electrons", "min-histories"),
                ("max-electrons", "max-histories"),
                ("block-electrons", "block-histories"),
                ("pilot-electrons", "pilot-histories"),
                ("max-electron-share", "max-history-share"),
                ("min-effective-electrons", "min-effective-histories"),
                ("brem-min-electrons", "brem-min-histories"),
                ("brem-max-electrons", "brem-max-histories"),
            )
        ),
        (["checkpoint", "recompute", "cxr", "m"], "--cxr-trials", "--cxr-histories"),
        (["checkpoint", "recompute", "brem", "m"], "--brem-trials", "--brem-histories"),
        (["app", "validation", "export"], "--brem-trials", "--brem-histories"),
        (["app", "validation", "reproduce"], "--brem-trials", "--brem-histories"),
    ],
)
def test_removed_budget_flags_are_usage_errors_naming_the_replacement(argv, old, new):
    result = invoke(root_command, [*argv, old, "5"])
    assert result.exit_code == 2
    assert f"{old} was removed; use {new}" in result.stderr
    assert "Traceback" not in result.stderr


def test_removed_precision_command_names_adaptive_sampling():
    result = invoke(root_command, ["numerics", "precision", "show", "p"])
    assert result.exit_code == 2
    assert "numerics precision' was removed; use '" in result.stderr
    assert result.stderr.rstrip().endswith("numerics adaptive-sampling'")


@pytest.mark.parametrize(
    ("line", "key", "replacement"),
    [
        ("cxr_electrons = { values = [450] }", "cxr_electrons", "cxr_histories"),
        ("precision = { target_rse = 0.1 }", "precision", "adaptive_sampling"),
    ],
)
def test_removed_electron_catalog_keys_name_histories(
    tmp_path, monkeypatch, line, key, replacement
):
    stale = _CATALOG.replace("[profiles.sub_100keV]\n", f"[profiles.sub_100keV]\n{line}\n")
    _catalog(tmp_path, monkeypatch, stale)
    result = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert result.exit_code == 1
    assert f"{key}: catalog key was removed; rename it to {replacement!r}" in result.stderr


def test_removed_budget_catalog_key_is_a_clear_runtime_error(tmp_path, monkeypatch):
    stale = _CATALOG.replace(
        "[profiles.sub_100keV]\n", "[profiles.sub_100keV]\ncxr_trials = { values = [450] }\n"
    )
    _catalog(tmp_path, monkeypatch, stale)
    result = invoke(numerics_command, ["adaptive-sampling", "show", "sub_100keV"])
    assert result.exit_code == 1
    assert "cxr_trials: catalog key was removed; rename it to 'cxr_histories'" in result.stderr
    assert "Traceback" not in result.stderr
