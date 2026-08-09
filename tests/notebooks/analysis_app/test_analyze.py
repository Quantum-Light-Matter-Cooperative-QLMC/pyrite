"""``cxr analyze`` -- launches src/cxr_mc/apps/analysis_app.py via marimo run/edit with
a chosen initial material. The initial-material resolution has to be a pure,
unit-testable helper (:func:`analyze.initial_material`) because marimo apps
can't be driven live in this environment; these tests exercise that helper and
the CLI arg handling directly, without spawning marimo."""

import json
import os
import sys

import pytest
from click.testing import CliRunner

from cxr_mc.apps import analyze
from cxr_mc.materials import CATALOG


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    # keep the env-var fallback transport from polluting the pure-precedence
    # tests below (CXR_ANALYZE_INITIAL sits between cli-arg and persisted-default
    # in precedence, so an ambient value would silently win over "persisted").
    monkeypatch.delenv("CXR_ANALYZE_INITIAL", raising=False)


# --- initial_material precedence ---------------------------------------


def test_cli_arg_wins_over_persisted_default():
    assert analyze.initial_material({"material": "wse2"}, "hopg") == "wse2"


def test_persisted_default_used_when_no_cli_arg():
    assert analyze.initial_material({}, "wse2") == "wse2"
    assert analyze.initial_material({"material": None}, "wse2") == "wse2"


def test_hopg_fallback_when_neither_given():
    assert analyze.initial_material({}, None) == "hopg"


def test_material_menu_marks_only_configured_checkpoint_stems_available(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for stem in ("silicon", "quick_only_quick", "unknown"):
        (checkpoints / f"{stem}.pkl").touch()

    menu = analyze.material_menu(
        checkpoints,
        materials=("hopg", "silicon", "quick_only"),
        labels={"hopg": "HOPG", "silicon": "Silicon", "quick_only": "Quick"},
    )

    # Available materials (checkpoint exists) sort first; each group is
    # alphabetical by label. Only "silicon" is available here.
    assert menu == (
        {"value": "silicon", "label": "Silicon", "disabled": False},
        {"value": "hopg", "label": "HOPG", "disabled": True},
        {"value": "quick_only", "label": "Quick", "disabled": True},
    )


def test_material_menu_sorts_available_before_unavailable_then_alphabetically(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for stem in ("zeta", "alpha"):
        (checkpoints / f"{stem}.pkl").touch()

    menu = analyze.material_menu(
        checkpoints,
        materials=("beta", "alpha", "zeta", "delta"),
        labels={"beta": "beta", "alpha": "Alpha", "zeta": "zeta", "delta": "Delta"},
    )

    assert menu == (
        {"value": "alpha", "label": "Alpha", "disabled": False},
        {"value": "zeta", "label": "zeta", "disabled": False},
        {"value": "beta", "label": "beta", "disabled": True},
        {"value": "delta", "label": "Delta", "disabled": True},
    )


def test_material_menu_label_sort_is_case_insensitive(tmp_path):
    # Raw string comparison would put "Banana" (capital B) before "apple"
    # (ASCII uppercase < lowercase); casefold sorting must not do that.
    menu = analyze.material_menu(
        tmp_path,
        materials=("b_mat", "a_mat"),
        labels={"b_mat": "Banana", "a_mat": "apple"},
    )

    assert tuple(row["value"] for row in menu) == ("a_mat", "b_mat")


def test_material_menu_defaults_to_catalog_materials_and_spec_labels(tmp_path):
    menu = analyze.material_menu(tmp_path)

    assert {row["value"] for row in menu} == set(CATALOG.material_keys)
    assert all(row["disabled"] for row in menu)  # tmp_path has no checkpoints
    assert {row["value"]: row["label"] for row in menu} == {
        key: CATALOG.material(key).label for key in CATALOG.material_keys
    }
    labels = [row["label"] for row in menu]
    assert labels == sorted(labels, key=str.casefold)


def test_checkpoint_stem_flat_is_material_blazed_is_suffixed():
    assert analyze.checkpoint_stem("hopg", "flat") == "hopg"
    assert analyze.checkpoint_stem("hopg", "blazed") == "hopg_blazed"


def _write_variant(
    checkpoint_dir,
    stem,
    *,
    material,
    catalog_profile="standard",
    variant=None,
    fidelity="full",
    digest="abcdef0123456789",
):
    stem_dir = checkpoint_dir / stem
    stem_dir.mkdir()
    (stem_dir / "line.pkl").touch()
    (stem_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset_identity": {
                    "material": material,
                    "fidelity": fidelity,
                    "variant": variant,
                    "catalog_profile": catalog_profile,
                    "parameter_sha256": digest,
                }
            }
        )
    )


def test_profile_menu_lists_standard_and_sidecar_resolved_variant(tmp_path):
    (tmp_path / "hbn.pkl").touch()
    _write_variant(
        tmp_path,
        "hbn--full-6a7c899190fc",
        material="hbn",
        catalog_profile="hopg_hbn_microtrain_200fs",
    )

    menu = analyze.profile_menu("hbn", tmp_path)

    assert menu == (
        {"value": "hbn", "label": "Standard", "disabled": False},
        {
            "value": "hbn--full-6a7c899190fc",
            "label": "hopg_hbn_microtrain_200fs (abcdef)",
            "disabled": False,
        },
    )


def test_material_and_profile_menu_browse_new_at_stem(tmp_path):
    """A named-profile checkpoint written under the @-stem scheme
    (`<material>@<catalog_profile>-<digest>`) is browsable in `cxr app
    analysis`: material_menu marks the material available and profile_menu
    lists the variant labeled by its catalog_profile. Sidecar-driven, so it
    behaves identically to the legacy `--` path -- this guards the milestone's
    binding acceptance against any future stem-text parsing creeping in."""
    _write_variant(
        tmp_path, "hbn@sub_100keV-6a7c899190fc", material="hbn", catalog_profile="sub_100keV"
    )

    material_rows = {row["value"]: row for row in analyze.material_menu(tmp_path)}
    assert material_rows["hbn"]["disabled"] is False  # available via @-stem sidecar

    assert analyze.profile_menu("hbn", tmp_path) == (
        {"value": "hbn@sub_100keV-6a7c899190fc", "label": "sub_100keV (abcdef)", "disabled": False},
    )


def test_profile_menu_ignores_stem_belonging_to_a_different_material(tmp_path):
    _write_variant(tmp_path, "hopg--full-4303954822d6", material="hopg")

    assert analyze.profile_menu("hbn", tmp_path) == ()


def test_profile_menu_ignores_blazed_stem_and_stems_without_a_sidecar(tmp_path):
    (tmp_path / "hbn_blazed.pkl").touch()
    (tmp_path / "unknown.pkl").touch()

    assert analyze.profile_menu("hbn", tmp_path) == ()


def test_profile_menu_orders_newest_variant_first_when_no_standard(tmp_path):
    _write_variant(tmp_path, "hbn--full-aaaaaaaaaaaa", material="hbn", digest="aaaaaa000000")
    older = tmp_path / "hbn--full-aaaaaaaaaaaa" / "meta.json"
    os.utime(older, (1_000_000, 1_000_000))
    _write_variant(tmp_path, "hbn--full-bbbbbbbbbbbb", material="hbn", digest="bbbbbb000000")
    newer = tmp_path / "hbn--full-bbbbbbbbbbbb" / "meta.json"
    os.utime(newer, (2_000_000, 2_000_000))

    menu = analyze.profile_menu("hbn", tmp_path)

    assert [row["value"] for row in menu] == ["hbn--full-bbbbbbbbbbbb", "hbn--full-aaaaaaaaaaaa"]


def test_profile_menu_labels_quick_variant_by_variant_not_catalog_profile(tmp_path):
    _write_variant(tmp_path, "hbn_quick", material="hbn", variant="quick", fidelity="full")

    menu = analyze.profile_menu("hbn", tmp_path)

    assert menu == ({"value": "hbn_quick", "label": "quick (abcdef)", "disabled": False},)


def test_profile_menu_empty_for_missing_material_or_dir(tmp_path):
    assert analyze.profile_menu("", tmp_path) == ()
    assert analyze.profile_menu("hbn", tmp_path / "does-not-exist") == ()


@pytest.mark.parametrize(
    ("helper", "run_name", "args"),
    [
        (analyze.load_analysis_checkpoint, "load_checkpoint", ()),
        (analyze.analysis_checkpoint_manifest, "checkpoint_manifest", ()),
        (analyze.cached_analysis, "cached_material_analysis", (lambda value: value, "key")),
    ],
)
def test_analysis_reads_skip_partially_pulled_pickle(
    helper, run_name, args, monkeypatch, capsys, tmp_path
):
    from cxr_mc.runs import run

    def incomplete(*_args):
        raise EOFError("Compressed file ended before the end-of-stream marker")

    monkeypatch.setattr(run, run_name, incomplete)

    assert helper("hopg", *args, checkpoint_dir=tmp_path) is None
    assert "temporarily unreadable" in capsys.readouterr().err


def test_analysis_read_preserves_complete_checkpoint(monkeypatch, tmp_path):
    from cxr_mc.runs import run

    expected = {"hopg": {30.0: {"case": {}}}}
    monkeypatch.setattr(run, "load_checkpoint", lambda material, root: (material, root, expected))

    assert analyze.load_analysis_checkpoint("hopg", tmp_path) == ("hopg", tmp_path, expected)


def test_load_analysis_checkpoint_skips_truncated_gzip(tmp_path, capsys):
    from cxr_mc.checkpoints import _checkpoint_io

    checkpoint = tmp_path / "hopg.pkl"
    _checkpoint_io.dump({"hopg": {}}, str(checkpoint))
    payload = checkpoint.read_bytes()
    checkpoint.write_bytes(payload[: len(payload) // 2])

    assert analyze.load_analysis_checkpoint("hopg", tmp_path) is None
    assert "retry after the pull finishes" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("present", "flat_disabled", "blazed_disabled"),
    [
        ((), True, True),  # neither checkpoint
        (("hopg",), False, True),  # only flat
        (("hopg_blazed",), True, False),  # only blazed
        (("hopg", "hopg_blazed"), False, False),  # both
    ],
)
def test_face_menu_disabled_flags_track_present_checkpoints(
    tmp_path, present, flat_disabled, blazed_disabled
):
    for stem in present:
        (tmp_path / f"{stem}.pkl").touch()

    menu = analyze.face_menu("hopg", tmp_path)

    # Flat first (preferred default), then blazed; labels are stable.
    assert menu == (
        {"value": "flat", "label": "Flat", "disabled": flat_disabled},
        {"value": "blazed", "label": "Blazed", "disabled": blazed_disabled},
    )


def test_face_menu_default_reuses_select_initial_material_flat_preferred(tmp_path):
    # Both present -> flat preferred; only blazed present -> blazed defaults.
    (tmp_path / "hopg.pkl").touch()
    (tmp_path / "hopg_blazed.pkl").touch()
    both = analyze.face_menu("hopg", tmp_path)
    assert analyze.select_initial_material(None, both) == "flat"

    (tmp_path / "hopg.pkl").unlink()
    blazed_only = analyze.face_menu("hopg", tmp_path)
    assert analyze.select_initial_material(None, blazed_only) == "blazed"


def test_select_initial_material_falls_back_to_first_available():
    menu = (
        {"value": "hopg", "label": "HOPG", "disabled": True},
        {"value": "silicon", "label": "Silicon", "disabled": False},
    )

    assert analyze.select_initial_material("hopg", menu) == "silicon"
    assert analyze.select_initial_material("silicon", menu) == "silicon"
    assert analyze.select_initial_material(None, menu) == "silicon"
    assert (
        analyze.select_initial_material(
            None, ({"value": "hopg", "label": "HOPG", "disabled": True},)
        )
        is None
    )


def test_select_initial_material_works_with_material_menu_ordering(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "silicon.pkl").touch()

    menu = analyze.material_menu(
        checkpoints,
        materials=("hopg", "silicon"),
        labels={"hopg": "HOPG", "silicon": "Silicon"},
    )

    assert analyze.select_initial_material(None, menu) == "silicon"
    assert analyze.select_initial_material("hopg", menu) == "silicon"
    assert analyze.select_initial_material("silicon", menu) == "silicon"


def test_env_var_fallback_used_between_cli_and_persisted(monkeypatch):
    monkeypatch.setenv("CXR_ANALYZE_INITIAL", "diamond")
    assert analyze.initial_material({}, "hopg") == "diamond"  # env beats persisted
    assert analyze.initial_material({"material": "wse2"}, "hopg") == "wse2"  # cli beats env


# --- get_default_material / set_default_material round-trip ------------


def test_default_material_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    assert analyze.get_default_material() is None  # never written yet
    analyze.set_default_material("mose2")
    assert analyze.get_default_material() == "mose2"


def test_default_material_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / "does-not-exist")
    assert analyze.get_default_material() is None


def test_default_material_empty_file_returns_none(tmp_path, monkeypatch):
    f = tmp_path / ".cxr-analyze-default"
    f.write_text("   \n")
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", f)
    assert analyze.get_default_material() is None


# --- CLI arg parsing / errors --------------------------------------------


def _invoke(argv=()):
    return CliRunner().invoke(analyze.command, list(argv), catch_exceptions=False)


def test_default_flag_without_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    result = _invoke(["-d"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--save-default requires MATERIAL" in result.stderr


def test_unknown_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    result = _invoke(["not-a-real-material"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "not a configured material" in result.stderr


def test_default_flag_persists_and_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    result = _invoke(["-d", "wse2"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert analyze.get_default_material() == "wse2"
    assert launched == {"material": "wse2", "edit": False, "watch": False}


def test_no_args_uses_persisted_default(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    analyze.set_default_material("hbn")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    result = _invoke()
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {"material": "hbn", "edit": False, "watch": False}


def test_acp_flag_starts_analysis_with_bridge_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )

    result = _invoke(["--acp"])
    assert result.exit_code == 0
    assert result.stderr == ""

    assert launched == {"material": "hopg", "edit": False, "watch": False, "acp": True}


def test_material_arg_is_transient_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    monkeypatch.setattr(analyze, "_launch", lambda material, **kw: None)
    result = _invoke(["wse2"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert analyze.get_default_material() is None  # not persisted, no -d


# --- argv construction (no marimo spawned) -------------------------------


def test_command_run_default():
    cmd = analyze._command("hopg")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "marimo", "run"]
    assert "--watch" not in cmd
    assert "--port" not in cmd
    assert cmd[-4:] == [analyze.NOTEBOOK, "--", "--material", "hopg"]


def test_headless_flag_is_not_supported():
    result = _invoke(["--headless"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "No such option '--headless'" in result.stderr


def test_smoke_command_executes_analysis_app_to_a_temporary_html_file(tmp_path):
    output = tmp_path / "analysis.html"

    cmd = analyze._smoke_command("hopg", output)

    assert cmd[:5] == [sys.executable, "-m", "marimo", "export", "html"]
    assert cmd[5:] == [
        analyze.NOTEBOOK,
        "--output",
        str(output),
        "--force",
        "--",
        "--material",
        "hopg",
    ]


def test_command_tunnel_uses_fixed_marimo_port():
    command = analyze._command("hopg", tunnel=True)
    assert command[3:7] == ["run", "--port", "2718", analyze.NOTEBOOK]


def test_tunnel_launch_prints_forwarding_instructions_without_running_marimo(monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(
        analyze.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    analyze._launch("hopg", tunnel=True)

    output = capsys.readouterr().out
    assert "ssh -L 2718:127.0.0.1:2718 <your-pi-ssh-host>" in output
    assert "http://127.0.0.1:2718" in output
    assert len(launched) == 1


def test_tunnel_flag_forwards_to_analysis_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze, "_launch", lambda material, **kw: launched.update(material=material, **kw)
    )
    result = _invoke(["--tunnel"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {"material": "hopg", "edit": False, "watch": False, "tunnel": True}


def test_command_edit():
    cmd = analyze._command("wse2", edit=True)
    assert cmd[3] == "edit"
    assert "--watch" not in cmd


def test_command_no_token_passes_marimo_flag():
    command = analyze._command("hopg", no_token=True)
    assert "--no-token" in command
    assert command.index("--no-token") < command.index(analyze.NOTEBOOK)


def test_command_no_token_omitted_by_default():
    command = analyze._command("hopg")
    assert "--no-token" not in command


def test_no_token_flag_forwards_to_analysis_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze, "_launch", lambda material, **kw: launched.update(material=material, **kw)
    )
    result = _invoke(["--no-token"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {
        "material": "hopg",
        "edit": False,
        "watch": False,
        "no_token": True,
    }


def test_command_watch_combines_with_run_and_edit():
    run_cmd = analyze._command("hopg", watch=True)
    assert run_cmd[3] == "run"
    assert run_cmd[4] == "--watch"
    assert run_cmd[5] == analyze.NOTEBOOK

    edit_cmd = analyze._command("hopg", edit=True, watch=True)
    assert edit_cmd[3] == "edit"
    assert edit_cmd[4] == "--watch"
    assert edit_cmd[5] == analyze.NOTEBOOK


# --- emission view selector (analysis-app) ------------------------------


def _emission_records(*, coherent=False):
    import numpy as np

    record = {"case": {"E0_keV": 30.0}, "spec": np.array([1.0, 2.0, 3.0])}
    if coherent:
        record["spec_coherent"] = np.array([10.0, 20.0, 30.0])
    return {"hopg@30": {30.0: record}}


def test_emission_menu_incoherent_only_when_no_spec_coherent():
    rows = analyze.emission_menu(_emission_records(coherent=False))
    by_value = {row["value"]: row for row in rows}
    assert by_value["incoherent"]["disabled"] is False
    assert by_value["coherent"]["disabled"] is True


def test_emission_menu_enables_coherent_when_spec_coherent_present():
    rows = analyze.emission_menu(_emission_records(coherent=True))
    by_value = {row["value"]: row for row in rows}
    assert by_value["incoherent"]["disabled"] is False
    assert by_value["coherent"]["disabled"] is False


def test_emission_menu_both_disabled_for_empty_results():
    rows = analyze.emission_menu({})
    assert all(row["disabled"] for row in rows)


def test_emission_menu_both_enabled_only_with_both_spectra():
    incoherent_only = {
        row["value"]: row for row in analyze.emission_menu(_emission_records(coherent=False))
    }
    assert incoherent_only["both"]["disabled"] is True

    both_present = {
        row["value"]: row for row in analyze.emission_menu(_emission_records(coherent=True))
    }
    assert both_present["both"]["disabled"] is False


def test_apply_emission_both_is_identity():
    results = _emission_records(coherent=True)
    assert analyze.apply_emission(results, "both") is results


def test_pick_spectrum_routes_incoherent_and_coherent():
    import numpy as np

    record = next(iter(_emission_records(coherent=True)["hopg@30"].values()))
    np.testing.assert_array_equal(analyze.pick_spectrum(record, "incoherent"), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(analyze.pick_spectrum(record, "coherent"), [10.0, 20.0, 30.0])


def test_pick_spectrum_falls_back_to_spec_without_spec_coherent():
    import numpy as np

    record = next(iter(_emission_records(coherent=False)["hopg@30"].values()))
    # coherent requested but none stored -> the incoherent spec, never a KeyError
    np.testing.assert_array_equal(analyze.pick_spectrum(record, "coherent"), [1.0, 2.0, 3.0])


def test_apply_emission_incoherent_is_identity():
    results = _emission_records(coherent=True)
    assert analyze.apply_emission(results, "incoherent") is results


def test_apply_emission_coherent_swaps_spec_without_mutating_source():
    import numpy as np

    results = _emission_records(coherent=True)
    original = results["hopg@30"][30.0]["spec"].copy()

    picked = analyze.apply_emission(results, "coherent")

    np.testing.assert_array_equal(picked["hopg@30"][30.0]["spec"], [10.0, 20.0, 30.0])
    # source checkpoint record is left intact (shallow copy, arrays shared)
    np.testing.assert_array_equal(results["hopg@30"][30.0]["spec"], original)
    assert picked["hopg@30"][30.0] is not results["hopg@30"][30.0]
