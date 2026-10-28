"""Configuration transactions and widget adapters, without simulation work."""

import io
import json
import shlex

import numpy as np
import pytest
from click.testing import CliRunner

from pyrite._catalog_layout import bundled_catalog, read_raw
from pyrite.apps.configuration_controls import calculation_defaults, field_values
from pyrite.apps.configuration_exports import available_components, export_record
from pyrite.campaign.configuration_editor import ConfigurationEditor, run_command
from pyrite.catalog_edit import CatalogDraft


@pytest.fixture
def editor(_isolate_user_catalog):
    return ConfigurationEditor.open(bundled_catalog(), "quickstart").create(
        "notebook", "quickstart"
    )


def test_unchanged_catalog_has_no_vocabulary_only_diff():
    assert CatalogDraft.open(bundled_catalog()).diff() == ""


def test_load_clone_preserves_resolved_instruments_and_leaves_source_untouched(editor):
    source = ConfigurationEditor.open(bundled_catalog(), "quickstart")
    original = source.validate()
    cloned = editor.validate()
    assert cloned.profile_beam("notebook") == original.profile_beam("quickstart")
    assert cloned.profile_detector_set("notebook") == original.profile_detector_set("quickstart")
    assert cloned.profile_numerics("notebook") == original.profile_numerics("quickstart")
    assert "notebook" not in read_raw(bundled_catalog())["profiles"]
    assert editor.row["energy_keV"] == source.row["energy_keV"]


def test_create_defaults_follow_profile_lifecycle(editor):
    from pyrite.campaign.profile_lifecycle import create_profile

    expected = CatalogDraft.open(bundled_catalog())
    create_profile(
        expected.document,
        "new",
        None,
        updates={},
        beam_name=None,
        detector_name=None,
        transport_updates={},
        materials=None,
    )
    created = editor.create("new").patch({"detector": "reference"})
    expected.document["profiles"]["new"]["detector"] = "reference"
    assert created.row == expected.document["profiles"]["new"]
    assert not created.errors()


def test_beam_copy_does_not_change_named_source_or_other_profiles(editor):
    before = editor.draft.document["beams"]["example"].unwrap()
    updated = editor.beam(
        "private-beam", {**before, "bunch_charge_pc": 2.0, "bunch_length_fs": 100.0}
    )
    catalog = updated.validate()
    assert catalog.profile_beam("notebook")["bunch_charge_pc"] == 2.0
    assert updated.draft.document["beams"]["example"].unwrap() == before
    assert updated.draft.document["profiles"]["quickstart"]["beam"] == "example"
    assert editor.row["beam"] == "example"
    with pytest.raises(ValueError, match="already exists"):
        updated.beam("EXAMPLE", {})


def test_geometry_physics_and_numerics_use_existing_catalog_validation(editor):
    updated = editor.patch(
        {
            "energy_keV": {"values": [40.0, 50.0]},
            "thickness_ang": {"values": [10000.0]},
            "tilt_deg": {"values": [30.0]},
            "emission": "both",
            "n_electrons": {"values": [10]},
            "n_electrons_brem": {"values": [5]},
            "energy_model": "midpoint",
            "straggling": True,
        }
    )
    catalog = updated.validate()
    assert catalog.material("hopg").scan.energy_keV.tolist() == [40.0, 50.0]
    assert catalog.material("hopg").scan.tilt_deg.tolist() == [30.0]
    assert catalog.profile_numerics("notebook")["straggling"] is True
    assert catalog.profile_emission("notebook") == "both"
    assert catalog.profile_numerics("notebook")["n_electrons"] == (10,)
    assert "cxr_trials" in updated.draft.serialize()


def test_scalar_and_pixel_slots_preserve_shared_definitions(editor):
    scalar = dict(editor.draft.document["detectors"]["reference"].unwrap())
    scalar.pop("label", None)
    scalar["observation_angle_deg"] = 70.0
    changed = editor.detector("default", values=scalar)
    assert (
        changed.validate().profile_detector_set("notebook")["default"]["observation_angle_deg"]
        == 70.0
    )
    assert editor.draft.document["detectors"]["reference"]["observation_angle_deg"] == 90.0
    collection = changed.detector(
        "camera", action="add", values="pixelated_reference", migrate=True
    )
    assert set(collection.validate().profile_detector_set("notebook")) == {"default", "camera"}
    duplicate = collection.detector("camera2", action="duplicate", source="camera")
    assert duplicate.row["detectors"]["camera2"] == "pixelated_reference"
    removed = duplicate.detector("camera", action="remove")
    assert "camera" not in removed.validate().profile_detector_set("notebook")
    with pytest.raises(ValueError, match="at least one"):
        changed.detector("default", action="remove")
    with pytest.raises(ValueError, match="explicitly convert"):
        changed.detector("camera", action="add", values="pixelated_reference")


def test_create_named_pixel_and_attach(editor):
    values = editor.draft.document["detectors"]["pixelated_reference"].unwrap()
    values.pop("label", None)
    created = editor.named_detector("private-camera", values).detector(
        "default", values="private-camera"
    )
    resolved = created.validate().profile_detector_set("notebook")["default"]
    assert resolved["distance_mm"] == 400.0
    assert resolved["shape"] == (256, 256)


@pytest.mark.parametrize("value", [{}, None, ""])
def test_clearing_scalar_geometry_cannot_silently_restore_defaults(editor, value):
    with pytest.raises(ValueError, match="explicit geometry"):
        editor.detector("default", values=value)
    assert editor.row["detector"] == "reference"


@pytest.mark.parametrize(
    "values,match",
    [
        ({"energy_keV": {"values": [-1.0]}}, "energy_keV"),
        ({"materials": ["missing"]}, "missing"),
        ({"beam": "missing"}, "beam"),
        ({"elastic_model": "invented"}, "elastic_model"),
        ({"n_electrons": {"values": [0]}}, "n_electrons"),
        ({"detector": "missing"}, "detector"),
    ],
)
def test_invalid_configuration_cannot_save(editor, values, match):
    invalid = editor.patch(values)
    assert any(match in error for error in invalid.errors())
    with pytest.raises(ValueError):
        invalid.save()
    assert "notebook" not in read_raw(bundled_catalog())["profiles"]


def test_incompatible_pixel_scalar_configuration_is_not_replaced_with_default(editor):
    values = editor.draft.document["detectors"]["pixelated_reference"].unwrap()
    values.pop("label", None)
    values["solid_angle_sr"] = 0.1
    invalid = editor.detector("default", values=values, migrate=True)
    assert invalid.errors()
    assert invalid.row["detectors"]["default"]["solid_angle_sr"] == 0.1
    with pytest.raises(ValueError):
        invalid.save()


def test_bundled_objects_are_protected_before_staging():
    editor = ConfigurationEditor.open(bundled_catalog(), "quickstart")
    with pytest.raises(ValueError, match="read-only"):
        editor.patch({"emission": "both"})
    with pytest.raises(ValueError, match="read-only"):
        editor.beam("private", {})


def test_explicit_adaptive_precision_rejects_pixel_and_coherent_modes(editor):
    from pyrite._precision import DEFAULT_PRECISION

    adaptive = editor.patch(
        {
            "precision": {
                key: value
                for key, value in DEFAULT_PRECISION.to_dict().items()
                if value is not None
            }
        }
    )
    assert not adaptive.errors()
    pixel = adaptive.detector("default", values="pixelated_reference")
    assert any("fixed trial counts" in error for error in pixel.errors())
    assert adaptive.patch({"emission": "coherent"}).errors()
    assert adaptive.patch({"n_electrons": {"values": [20]}}).errors()


def test_save_reload_round_trip(editor):
    saved = editor.patch({"n_electrons": {"values": [10]}}).save()
    assert saved.draft.diff() == ""
    reopened = ConfigurationEditor.open(bundled_catalog(), "notebook")
    assert reopened.row == saved.row
    assert json.loads(reopened.resolved_json())["profile"] == "notebook"


def test_save_catalog_button_persists_staged_objects(editor, monkeypatch, _isolate_user_catalog):
    import gc
    import weakref
    from types import SimpleNamespace

    import marimo as mo

    from pyrite.apps.configuration_app import app
    from pyrite.apps.configuration_sections import transact
    from pyrite.materials.catalog import load_material_catalog

    edited = editor.patch({"energy_keV": {"values": [42.0]}}).beam(
        "notebook-beam", {"transverse_fwhm_mm": 0.5, "bunch_charge_pc": 2.0}
    )
    state = [(edited, "Changes staged")]
    buttons = []
    callback_globals = []
    original_button = mo.ui.button

    def capture_button(*args, **kwargs):
        button = original_button(*args, **kwargs)
        if kwargs.get("label") == "Save catalog changes":
            buttons.append(weakref.ref(button))
            callback_globals.append(kwargs["on_click"].__globals__)
        return button

    def apply(operation, message):
        transact(
            lambda: state[0], lambda updated: state.__setitem__(0, updated), operation, message
        )

    monkeypatch.setattr(mo.ui, "button", capture_button)
    outputs, definitions = app.run(
        defs={
            "section": SimpleNamespace(value="Review and Save"),
            "editor": edited,
            "notice": "Changes staged",
            "apply": apply,
        }
    )
    profile_file = _isolate_user_catalog / "profiles" / "notebook.toml"
    beam_file = _isolate_user_catalog / "beams" / "notebook-beam.toml"
    assert not profile_file.exists()
    assert not beam_file.exists()
    assert outputs and definitions
    # The live kernel discards cell temporaries after displaying HTML.
    for cell in app._graph.cells.values():
        for name in cell.temporaries - cell.closed_over_temporaries:
            callback_globals[0].pop(name, None)
    del outputs
    gc.collect()
    assert len(buttons) == 1
    button = buttons[0]()
    assert button is not None, "Save button must stay alive to receive browser clicks"
    button._update(1)
    assert state[0][1] == "Catalog changes saved"
    assert profile_file.is_file() and beam_file.is_file()
    reloaded = load_material_catalog(edited.draft.path, profile="notebook")
    assert reloaded.material("hopg").scan.energy_keV.tolist() == [42.0]
    assert reloaded.profile_beam("notebook")["bunch_charge_pc"] == 2.0
    assert state[0][0].draft.diff() == ""
    from pyrite.cli import command

    listed = CliRunner().invoke(command, ["profile", "list"])
    assert listed.exit_code == 0, listed.output
    assert "notebook:" in listed.output


@pytest.mark.parametrize("kind", ["geometry", "physics", "numerics"])
def test_configuration_layout_retains_forms_for_browser_events(editor, monkeypatch, kind):
    import gc
    import weakref

    import marimo as mo

    from pyrite.apps import configuration_sections as sections

    forms = []
    original = sections.fields_form

    def capture(*args, **kwargs):
        form = original(*args, **kwargs)
        forms.append(weakref.ref(form))
        return form

    monkeypatch.setattr(sections, "fields_form", capture)
    if kind == "geometry":
        rendered = sections.geometry(mo, editor, lambda *_: None, "thickness_ang")
    else:
        rendered = sections.calculation(mo, editor, kind, False, lambda *_: None)
    gc.collect()
    assert rendered is not None
    assert forms and all(form() is not None for form in forms), (
        "Rendered forms must retain event handlers"
    )


def test_create_then_save_remains_interactive_after_cell_cleanup(
    monkeypatch, _isolate_user_catalog
):
    import gc
    from copy import deepcopy
    from types import SimpleNamespace

    from pyrite.apps.configuration_app import app
    from pyrite.apps.configuration_sections import transact
    from pyrite.cli import command

    state = [(ConfigurationEditor.open(bundled_catalog(), "quickstart"), "Catalog opened")]
    # The JSON inspection preview is independent of widget events and saving.
    monkeypatch.setattr(ConfigurationEditor, "resolved_json", lambda self: "{}")

    def apply(operation, message):
        transact(
            lambda: state[0], lambda updated: state.__setitem__(0, updated), operation, message
        )

    def render(section):
        outputs, definitions = app.run(
            defs={
                "section": SimpleNamespace(value=section),
                "editor": state[0][0],
                "notice": state[0][1],
                "apply": apply,
            }
        )
        glbls = definitions["open_form"]._on_change.__globals__
        for cell in app._graph.cells.values():
            for name in cell.temporaries - cell.closed_over_temporaries:
                glbls.pop(name, None)
        del outputs
        gc.collect()
        return definitions

    overview = render("Overview")
    create = overview["configuration_view"]._live_children[1]
    values = deepcopy(create.element._value_frontend)
    values["name"] = "test_ui"
    values["source"] = ["quickstart"]
    create._update(values)
    assert state[0][0].profile == "test_ui"
    assert not (_isolate_user_catalog / "profiles" / "test_ui.toml").exists()
    review = render("Review and Save")
    review["save_catalog_button"]._update(1)
    assert state[0][1] == "Catalog changes saved"
    assert (_isolate_user_catalog / "profiles" / "test_ui.toml").is_file()
    listed = CliRunner().invoke(command, ["profile", "list"])
    assert listed.exit_code == 0, listed.output
    assert "test_ui:" in listed.output


def test_stale_edits_cannot_overwrite_newer_profile(editor):
    editor.save()
    reopened = ConfigurationEditor.open(bundled_catalog(), "notebook")
    newer = reopened.patch({"n_electrons": {"values": [20]}}).save()
    with pytest.raises(ValueError, match="changed while editing"):
        reopened.patch({"emission": "both"}).save()
    assert newer.validate().profile_numerics("notebook")["n_electrons"] == (20,)


def test_generated_run_command_keeps_execution_options_out_of_profile(editor):
    original = editor.draft.serialize()
    argv = shlex.split(
        run_command(
            editor,
            checkpoint_dir="my outputs/checkpoints",
            trajectories=True,
            trajectory_dir="my outputs/histories",
            trajectory_storage="throughput",
            backend="cpu",
            workers=0,
        )
    )
    assert argv[:4] == ["env", "PYRITE_MC_BACKEND=cpu", "pyrite", "--catalog"]
    assert argv[4] == str(editor.draft.path.resolve())
    assert "--trajectories=my outputs/histories" in argv
    assert argv[-2:] == ["--workers", "0"]
    assert "--trajectories" not in run_command(editor)
    assert editor.draft.serialize() == original
    with pytest.raises(ValueError, match="trajectory_dir"):
        run_command(editor, trajectories=True)


def test_widget_decode_preserves_optional_settings_and_numeric_types():
    assert field_values(
        {
            "energy_keV.values": {"explicit": True, "value": "30, 40"},
            "straggling": {"explicit": True, "value": False},
            "mosaic_nodes": {"explicit": False, "value": 5},
        },
        {"energy_keV.values": [30.0], "straggling": False, "mosaic_nodes": 5},
    ) == {
        "energy_keV": {"values": [30.0, 40.0]},
        "straggling": False,
    }
    assert calculation_defaults("physics")["emission"] == "incoherent"
    with pytest.raises(ValueError, match="energy_keV.values"):
        field_values(
            {"energy_keV.values": {"explicit": True, "value": "bad"}}, {"energy_keV.values": [30.0]}
        )


@pytest.mark.parametrize("selected", ["thickness_ang", "thickness_layers"])
@pytest.mark.parametrize("initial", ["other_mode", "selected_scalar"])
def test_thickness_selector_applies_only_selected_representation(editor, selected, initial):
    import marimo as mo

    from pyrite.apps.configuration_controls import fields_form

    other = "thickness_layers" if selected == "thickness_ang" else "thickness_ang"
    suggestions = {"thickness_ang": {"values": [20000.0]}, "thickness_layers": {"values": [3]}}
    submitted = []
    current = (
        {other: suggestions[other]}
        if initial == "other_mode"
        else {selected: suggestions[selected]["values"][0]}
    )
    form = fields_form(
        mo,
        current,
        suggestions,
        lambda build: submitted.append(build()),
        title="Apply thickness",
        thickness_key=selected,
    )
    form._on_change(form.element.value)
    assert submitted == [{selected: suggestions[selected]}]
    updated = editor.patch({"materials": ["mos2"], other: None, **submitted[0]})
    assert other not in updated.row
    assert not updated.errors()


def test_material_and_scene_values_submit_together(editor):
    import marimo as mo

    from pyrite.apps.configuration_controls import fields_form

    submitted = []
    form = fields_form(
        mo,
        {"energy_keV": {"values": [30.0]}},
        {},
        lambda build: submitted.append(build()),
        title="Apply scene and materials",
        extra_controls={"materials": mo.ui.multiselect(["hopg", "mos2"], value=["hopg"])},
    )
    values = form.element.value
    values["materials"] = ["mos2"]
    values["parameter_0"]["value"] = "40"
    form._on_change(values)
    assert submitted == [{"energy_keV": {"values": [40.0]}, "materials": ["mos2"]}]
    assert not editor.patch(submitted[0]).errors()


def test_selected_source_exports_retain_distinct_axes_and_metadata():
    record = {
        "E_grid": np.array([1.0, 2.0]),
        "spec": np.array([3.0, 4.0]),
        "spec_characteristic": np.array([5.0, 6.0]),
        "E_grid_brem": np.array([7.0, 8.0, 9.0]),
        "brem_wide": np.array([10.0, 11.0, 12.0]),
        "case": {"E0_keV": 30.0},
    }
    assert set(available_components(record)) == {"spec", "spec_characteristic", "brem_wide"}
    with np.load(
        io.BytesIO(export_record(record, ["spec_characteristic", "brem_wide"], "npz"))
    ) as data:
        assert set(data.files) == {
            "E_grid",
            "E_grid_brem",
            "spec_characteristic",
            "brem_wide",
            "metadata_json",
        }
        np.testing.assert_array_equal(data["spec_characteristic"], record["spec_characteristic"])
    assert "spec_characteristic" not in export_record(record, ["spec"], "csv").decode()
    with pytest.raises(ValueError, match="available"):
        export_record(record, ["pxr"], "npz")


def test_cli_launcher_uses_existing_marimo_launcher(monkeypatch):
    from pyrite.cli import command
    from pyrite.cli.commands import app_views

    calls = []
    monkeypatch.setattr(app_views, "_launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner = CliRunner()
    result = runner.invoke(command, ["app", "configuration", "launch", "--smoke"])
    assert result.exit_code == 0, result.output
    assert calls[0][0][0].notebook_name == "configuration_app.py"
    assert calls[0][1]["smoke"] is True
    assert runner.invoke(command, ["app", "configuration", "--help"]).exit_code == 0


@pytest.mark.parametrize(
    "section",
    [
        "Overview",
        "Scene and Materials",
        "Beam",
        "Detectors",
        "Physics",
        "Numerics",
        "Outputs and Exports",
        "Review and Save",
    ],
)
def test_every_notebook_section_executes_without_simulation(section, monkeypatch):
    from types import SimpleNamespace

    from pyrite.apps.configuration_app import app

    def forbidden(*args, **kwargs):
        pytest.fail("configuration widgets must not start simulation or save")

    monkeypatch.setattr(CatalogDraft, "save", forbidden)
    outputs, definitions = app.run(defs={"section": SimpleNamespace(value=section)})
    assert outputs
    assert definitions["editor"].profile == "standard"
    if section in {"Physics", "Numerics"}:
        assert f"Show advanced {section.lower()}" in definitions["advanced"].text
    else:
        assert definitions["advanced"] is None


def test_physics_preset_and_overrides_apply_together(editor, monkeypatch):
    import marimo as mo

    from pyrite.apps import configuration_sections as sections

    forms = []
    current = [editor]
    original = sections.fields_form

    def capture(*args, **kwargs):
        form = original(*args, **kwargs)
        forms.append(form)
        return form

    def apply(operation, _message):
        current[0] = operation(current[0])

    monkeypatch.setattr(sections, "fields_form", capture)
    rendered = sections.calculation(mo, editor, "physics", False, apply)
    assert len(forms) == 1
    assert "Attach physics preset" not in rendered.text
    values = forms[0].element.value
    values["physics"] = "electron-standard"
    values["parameter_0"] = {"explicit": True, "value": "both"}
    forms[0]._on_change(values)
    assert current[0].row["physics"] == "electron-standard"
    assert current[0].row["emission"] == "both"
    assert not current[0].errors()
    values["physics"] = ""
    forms[0]._on_change(values)
    assert "physics" not in current[0].row
    assert current[0].row["emission"] == "both"


@pytest.mark.slow
def test_saved_notebook_profile_reaches_existing_pipeline_and_preserves_identity(
    editor, monkeypatch
):
    from pyrite.campaign import config
    from pyrite.campaign.profiles import case_content_key
    from pyrite.campaign.sweep import build_cases
    from pyrite.materials.catalog import load_material_catalog

    saved = editor.save()

    def cases(profile):
        catalog = load_material_catalog(saved.draft.path, profile=profile)
        monkeypatch.setattr(config, "_catalog", lambda _: catalog)
        sweep = config.material_sweep("hopg", catalog_profile=profile)
        return build_cases(
            sweep,
            n_electrons=10,
            n_electrons_brem=5,
            inelastic_model="continuous",
            bremsstrahlung_model="eedl",
            radiative_model="uncoupled",
            check_kinematic_validity=False,
        )

    original, configured = cases("quickstart"), cases("notebook")
    assert len(configured) == len(original) == 1
    assert case_content_key(configured[0]) == case_content_key(original[0])
    saved.patch({"energy_keV": {"values": [40.0]}}).save()
    changed = cases("notebook")
    assert changed[0]["E0_keV"] == 40.0
    assert case_content_key(changed[0]) != case_content_key(configured[0])
