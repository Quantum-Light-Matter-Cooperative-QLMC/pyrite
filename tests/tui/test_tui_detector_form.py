"""UI-independent detector editor model: parsing, staging, parity and no-write paths."""

import pytest

from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite.cli import command
from pyrite.cli.commands.detector import typed_create
from pyrite.tui.detector_form import (
    FIELDS,
    DetectorSession,
    FormError,
    fields_for,
    form_values,
    parse_field,
)
from tests.helpers.cli import assert_clean_result, invoke


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_fields_mirror_typed_cli_options():
    group = typed_create
    for kind in ("scalar", "planar"):
        options = {
            param.name: param
            for param in group.commands[kind].params
            if param.name not in ("name", "source", "label", "dry_run")
        }
        specs = {spec.key: spec for spec in fields_for(kind)}
        assert set(specs) == set(options)
        for key, spec in specs.items():
            assert spec.flag in options[key].opts
            if key != "pitch_mm":  # the CLI checks positivity in a callback
                assert spec.type.to_info_dict() == options[key].type.to_info_dict()


def test_stored_rows_round_trip_through_editor_text():
    rows = read_raw(bundled_catalog())["detectors"]
    for name, row in rows.items():
        values = form_values(row)
        for spec in FIELDS:
            if spec.key in values:
                assert parse_field(spec.key, values[spec.key]) is not None, (name, spec.key)


@pytest.mark.parametrize(
    ("key", "text", "message"),
    [
        ("observation_angle_deg", "181", "180"),
        ("solid_angle_sr", "0", "0"),
        ("shape", "256", "2 numbers"),
        ("pitch_mm", "0.05 nan", "finite"),
        ("measured_edges_ev", "-1,5", "nonnegative"),
        ("timepix_samples", "1.5", "integer"),
    ],
)
def test_field_errors_name_the_field_and_domain(key, text, message):
    with pytest.raises(FormError, match=message) as info:
        parse_field(key, text)
    assert key in info.value.fields


def test_blank_is_unset_and_pairs_accept_commas():
    assert parse_field("distance_mm", "  ") is None
    assert parse_field("offset_mm", "1, -2") == (1.0, -2.0)


def test_cross_field_errors_collect_without_staging():
    session = DetectorSession(bundled_catalog())
    with pytest.raises(FormError) as info:
        session.stage("create", "bad", {"distance_mm": "-1", "shape": "3"}, kind="planar")
    assert set(info.value.fields) == {"distance_mm", "shape"}
    with pytest.raises(FormError, match="go together"):
        session.stage(
            "create", "bad", {"distance_mm": "10", "measured_range_ev": "0 10"}, kind="planar"
        )
    with pytest.raises(FormError, match="timepix3"):
        session.stage(
            "create",
            "bad",
            {"distance_mm": "10", "response": "ideal", "timepix_bias_v": "50"},
            kind="planar",
        )


def _replay_as(staged, name):
    """Run the staged edit's CLI equivalent on ``name``; return both stored rows."""
    for args in staged.cli_commands():
        assert_clean_result(invoke(command, [name if arg == staged.name else arg for arg in args]))
    rows = read_raw(bundled_catalog())["detectors"]
    return rows[staged.name], rows[name]


@pytest.mark.parametrize(
    ("mode", "kind", "values"),
    [
        ("create", "scalar", {"observation_angle_deg": "91", "solid_angle_sr": "0.01"}),
        (
            "create",
            "planar",
            {
                "label": "lab chip",
                "distance_mm": "250",
                "polar_deg": "45",
                "pitch_mm": "0.11 0.11",
                "response": "timepix3",
                "timepix_bias_v": "100",
                "measured_edges_ev": "1000,2000,4000",
                "realization": "poisson",
                "exposure_s": "2",
                "realization_seed": "7",
            },
        ),
    ],
)
def test_create_serializes_like_cli(_isolate_user_catalog, mode, kind, values):
    session = DetectorSession(bundled_catalog())
    staged = session.stage(mode, "editor", values, kind=kind)
    session.save(staged)
    editor, batch = _replay_as(staged, f"batch_{staged.mode}")
    assert editor == batch


def test_clone_and_edit_patch_only_changed_fields_like_cli(_isolate_user_catalog):
    session = DetectorSession(bundled_catalog())
    initial = session.form("pixelated_reference")
    changed = {**initial, "distance_mm": "500", "measured_bin_width_ev": "50"}
    staged = session.stage(
        "clone", "editor", changed, source="pixelated_reference", initial=initial
    )
    assert "--measured-range-ev" in staged.cli_commands()[1]
    session.save(staged)
    editor, batch = _replay_as(staged, f"batch_{staged.mode}")
    assert editor == batch
    assert editor["distance_mm"] == 500
    assert editor["shape"] == [256, 256]

    session.reload()  # the CLI replay changed the catalog after this session read it
    initial = session.form("editor")
    staged = session.stage(
        "edit", "editor", {**initial, "realization": "poisson", "realization_seed": "3"},
        initial=initial,
    )  # fmt: skip
    session.save(staged)
    editor, batch = _replay_as(staged, "batch_clone")
    assert editor == batch
    assert editor["acquisition"]["seed"] == 3


def test_typing_validation_failure_cancel_and_readonly_never_write(_isolate_user_catalog):
    before = snapshot(_isolate_user_catalog)
    session = DetectorSession(bundled_catalog())
    staged = session.stage("create", "unsaved", {"observation_angle_deg": "90"}, kind="scalar")
    assert "+[detectors.unsaved]" in staged.diff().replace(" ", "")
    assert "E_grid_line" not in staged.diff()  # canonical vocabulary, no legacy noise
    with pytest.raises(FormError, match="read-only"):
        initial = session.form("reference")
        session.stage("edit", "reference", {**initial, "label": "x"}, initial=initial)
    with pytest.raises(FormError, match="no changes"):
        session.stage(
            "edit", "reference", session.form("reference"), initial=session.form("reference")
        )
    with pytest.raises(FormError, match="already exists"):
        session.stage("create", "Reference", {"observation_angle_deg": "1"}, kind="scalar")
    assert snapshot(_isolate_user_catalog) == before


def test_clearing_a_field_points_to_reset(_isolate_user_catalog):
    session = DetectorSession(bundled_catalog())
    session.save(session.stage("clone", "mine", {}, source="pixelated_reference"))
    initial = session.form("mine")
    with pytest.raises(FormError, match="detector reset") as info:
        session.stage("edit", "mine", {**initial, "exposure_s": ""}, initial=initial)
    assert info.value.fields == {"exposure_s": "clearing is not supported"}


def test_stale_draft_is_rejected_and_keeps_newer_catalog(_isolate_user_catalog):
    session = DetectorSession(bundled_catalog())
    staged = session.stage("create", "stale", {"observation_angle_deg": "90"}, kind="scalar")
    assert_clean_result(
        invoke(command, ["detector", "create", "scalar", "newer", "--observation-angle", "1"])
    )
    newer = snapshot(_isolate_user_catalog)
    with pytest.raises(FormError, match="changed while editing"):
        session.save(staged)
    assert snapshot(_isolate_user_catalog) == newer


def test_external_catalog_is_edited_in_place(tmp_path):
    from tests.helpers.user_catalog import copy_full_catalog

    selected = copy_full_catalog(tmp_path / "catalog")
    session = DetectorSession(selected)
    assert {entry.source for entry in session.entries()} == {"catalog"}
    initial = session.form("reference")
    session.save(
        session.stage("edit", "reference", {**initial, "solid_angle_sr": "0.002"}, initial=initial)
    )
    assert read_raw(selected)["detectors"]["reference"]["solid_angle_sr"] == 0.002
