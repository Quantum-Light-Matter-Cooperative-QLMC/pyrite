"""Interactive catalog configuration; explicit saves and no automatic simulations."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full", app_title="PyRITE · Configuration", html_head_file="../_assets/branding/head.html"
)


@app.cell
def _():
    import marimo as mo

    from pyrite._catalog_layout import selected_catalog
    from pyrite.apps._design import page_title, style_sheet
    from pyrite.apps.configuration_controls import beam_template, detector_template, fields_form
    from pyrite.apps.configuration_sections import (
        calculation,
        detector_membership,
        geometry,
        overview,
        transact,
    )
    from pyrite.campaign.configuration_editor import ConfigurationEditor, plain, run_command
    from pyrite.campaign.profile_detectors import profile_detector_entries
    from pyrite.console.outputs import output_dir

    mo.vstack(
        [
            style_sheet(mo),
            page_title(mo, "Configuration", "Build, validate and save simulation profiles"),
        ]
    )
    return (
        ConfigurationEditor,
        beam_template,
        calculation,
        detector_membership,
        detector_template,
        fields_form,
        geometry,
        mo,
        output_dir,
        overview,
        plain,
        profile_detector_entries,
        run_command,
        selected_catalog,
        transact,
    )


@app.cell
def _(ConfigurationEditor, mo, selected_catalog):
    from pyrite.catalog_edit import CatalogDraft

    try:
        _path = selected_catalog()
        _profiles = CatalogDraft.open(_path).document["profiles"]
        _profile = "standard" if "standard" in _profiles else next(iter(_profiles))
        _initial = ConfigurationEditor.open(_path, _profile)
        _notice = "Catalog opened; no changes saved"
    except (OSError, ValueError, KeyError, StopIteration) as _error:
        _initial, _notice = None, f"Cannot open selected catalog: {_error}"
    get_state, set_state = mo.state((_initial, _notice))
    return get_state, set_state


@app.cell
def _(ConfigurationEditor, get_state, mo, selected_catalog, set_state):
    def _open(value):
        if value is None:
            return
        _previous, _ = get_state()
        try:
            if _previous is not None and _previous.draft.diff() and not value["discard"]:
                raise ValueError("Check discard unsaved changes before opening another catalog")
            _loaded = ConfigurationEditor.open(value["path"], value["profile"])
            set_state((_loaded, "Catalog opened"))
        except (OSError, ValueError, KeyError) as error:
            set_state((_previous, f"Open rejected: {error}"))

    open_form = mo.ui.dictionary(
        {
            "path": mo.ui.text(
                value=str(selected_catalog()),
                label="Catalog file or directory on notebook host",
                full_width=True,
            ),
            "profile": mo.ui.text(value="standard", label="Profile to open"),
            "discard": mo.ui.checkbox(label="Discard unsaved changes when opening"),
        }
    ).form(submit_button_label="Open catalog", on_change=_open)
    open_form
    return


@app.cell
def _(get_state, mo, set_state, transact):
    editor, notice = get_state()
    mo.stop(editor is None, mo.callout(mo.md(notice), kind="warn"))

    def apply(operation, message):
        transact(get_state, set_state, operation, message)

    _errors = editor.errors()
    mo.vstack(
        [
            mo.md(notice),
            *([mo.callout(mo.md("\n\n".join(_errors)), kind="warn")] if _errors else []),
        ]
    )
    return apply, editor


@app.cell
def _(mo):
    section = mo.ui.radio(
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
        value="Overview",
        inline=True,
        label="Configuration section",
    )
    advanced = mo.ui.checkbox(label="Show advanced physics and numerics")
    mo.vstack([section, advanced])
    return advanced, section


@app.cell
def _(advanced, apply, calculation, editor, geometry, mo, overview, section):
    mo.stop(section.value not in {"Overview", "Scene and Materials", "Physics", "Numerics"})
    if section.value == "Overview":
        _content = overview(mo, editor, apply)
    elif section.value == "Scene and Materials":
        _content = geometry(mo, editor, apply)
    else:
        _content = calculation(mo, editor, section.value.lower(), advanced.value, apply)
    _content
    return


@app.cell
def _(editor, mo, section):
    mo.stop(section.value != "Scene and Materials")
    override_material = mo.ui.dropdown(
        list(editor.draft.document["materials"]), label="Optional material-specific override"
    )
    override_material
    return (override_material,)


@app.cell
def _(apply, editor, fields_form, mo, override_material, plain):
    mo.stop(not override_material.value)
    _overrides = plain(editor.row.get("overrides", {}))
    _current = _overrides.get(override_material.value, {})
    _suggestions = {
        "thickness_ang": {"values": [20000.0]},
        "energy_keV": {"values": [30.0]},
        "tilt_deg": {"values": [45.0]},
        "tilt_azim_deg": {"values": [180.0]},
    }

    def _submit(build):
        def _operation(e):
            _updated = plain(e.row.get("overrides", {}))
            _values = build()
            if _values:
                _updated[override_material.value] = _values
            else:
                _updated.pop(override_material.value, None)
            return e.patch({"overrides": _updated or None})

        apply(_operation, "Material override staged")

    fields_form(mo, _current, _suggestions, _submit, title="Apply material-specific ranges")
    return


@app.cell
def _(editor, mo, plain, section):
    mo.stop(section.value != "Beam")
    _row = plain(editor.row)
    _beam = _row.get("beam")
    beam_source = mo.ui.dropdown(
        {
            "Profile inline beam / built-in defaults": "",
            **{name: name for name in editor.draft.document.get("beams", {})},
        },
        value=_beam if isinstance(_beam, str) else "Profile inline beam / built-in defaults",
        label="Load named beam",
    )
    beam_mode = mo.ui.dropdown(
        ["Keep source fields", "Gaussian spot", "Twiss", "GDF"],
        value="Keep source fields",
        label="Source/transverse template",
    )
    bunch_mode = mo.ui.dropdown(
        ["Point bunch", "Bunch duration", "gaussian", "microtrain", "compressed"],
        value="Point bunch",
        label="Longitudinal template (when replacing source fields)",
    )
    beam_name = mo.ui.text(label="New private named beam (existing names are protected)")
    attach_beam = mo.ui.run_button(label="Attach selected existing beam without edits")
    mo.vstack(
        [
            mo.md(
                "Energy belongs to **Scene and Materials**, in keV. Beam normalization uses "
                "**Hz** and **pC**; transverse FWHM **mm**, duration **fs**, RMS energy spread "
                "is fractional. Creating a private copy leaves every other profile unchanged. "
                "GDF requires a host file path and origin **m**; time **s** and screen **m** "
                "selectors are mutually exclusive. No GDF sampling runs here."
            ),
            beam_source,
            beam_mode,
            bunch_mode,
            beam_name,
            attach_beam,
        ]
    )
    return attach_beam, beam_mode, beam_name, beam_source, bunch_mode


@app.cell
def _(apply, attach_beam, beam_source, mo):
    mo.stop(not attach_beam.value or not beam_source.value)
    apply(lambda e: e.patch({"beam": beam_source.value}), "Existing beam attached")
    return


@app.cell
def _(
    apply,
    beam_mode,
    beam_name,
    beam_source,
    beam_template,
    bunch_mode,
    editor,
    fields_form,
    mo,
    plain,
):
    if beam_source.value:
        _source = plain(editor.draft.document["beams"][beam_source.value])
    else:
        _source = editor.row.get("beam", {})
        _source = plain(_source) if isinstance(_source, dict) else {}
    _source = {key: value for key, value in _source.items() if key != "label"}
    _template = beam_template(
        "Gaussian spot" if beam_mode.value == "Keep source fields" else beam_mode.value,
        bunch_mode.value,
    )
    if beam_mode.value != "Keep source fields":
        _source = {
            key: value
            for key, value in _template.items()
            if key
            in {
                "rep_rate_hz",
                "bunch_charge_pc",
                "source",
                "gdf_path",
                "gdf_z_origin_m",
                "transverse_fwhm_mm",
                "transverse",
                "longitudinal",
                "bunch_length_fs",
                "long_shape",
            }
        }

    def _submit(build):
        apply(
            lambda e: e.beam(beam_name.value, build()), "Private beam created and attached in draft"
        )

    fields_form(mo, _source, _template, _submit, title="Create private beam and attach")
    return


@app.cell
def _(apply, detector_membership, editor, mo, profile_detector_entries, section):
    mo.stop(section.value != "Detectors")
    _entries = profile_detector_entries(editor.row)
    detector_id = mo.ui.dropdown(
        list(_entries), value=next(iter(_entries)), label="Edit detector ID"
    )
    detector_kind = mo.ui.dropdown(
        ["Keep current geometry", "New scalar", "New pixel"],
        value="Keep current geometry",
        label="Geometry template (explicit replacement)",
    )
    detector_name = mo.ui.text(label="Optional new named detector; blank = local inline copy")
    mo.vstack([detector_membership(mo, editor, apply), detector_id, detector_kind, detector_name])
    return detector_id, detector_kind, detector_name


@app.cell
def _(
    apply,
    detector_id,
    detector_kind,
    detector_name,
    detector_template,
    editor,
    fields_form,
    mo,
    plain,
    profile_detector_entries,
):
    _entry = profile_detector_entries(editor.row)[detector_id.value]
    _source = plain(
        editor.draft.document["detectors"][_entry] if isinstance(_entry, str) else _entry
    )
    _source = {key: value for key, value in _source.items() if key != "label"}
    _pixel = detector_kind.value == "New pixel" or (
        detector_kind.value == "Keep current geometry" and "distance_mm" in _source
    )
    _template = detector_template(_pixel)
    if detector_kind.value != "Keep current geometry":
        _source = {
            key: value
            for key, value in _template.items()
            if key not in {"response", "acquisition", "scorer"}
        }

    def _submit(build):
        def _operation(e):
            _values = build()
            if detector_name.value:
                return e.named_detector(detector_name.value, _values).detector(
                    detector_id.value, values=detector_name.value
                )
            return e.detector(detector_id.value, values=_values)

        apply(_operation, "Detector edited in draft; named source unchanged")

    mo.vstack(
        [
            mo.md(
                "Scalar acceptance: observation/full polar span **deg**, solid angle **sr**. "
                "Pixel geometry: distance/offset/pitch **mm**, shape **pixel counts**; polar, "
                "azimuth and roll **deg**. Pixel acceptance derives from geometry. Response "
                "thickness **µm**, bias **V**; acquisition exposure **s**, thresholds/bins **eV**. "
                "Optional sections must be enabled together with their required fields."
            ),
            fields_form(
                mo, _source, _template, _submit, title="Apply detector geometry and scoring"
            ),
        ]
    )
    return


@app.cell
def _(mo, output_dir):
    execution = mo.ui.dictionary(
        {
            "checkpoint_dir": mo.ui.text(
                label="Checkpoint directory; blank = normal workspace default"
            ),
            "trajectories": mo.ui.checkbox(label="Record transported electron histories (HDF5)"),
            "trajectory_dir": mo.ui.text(
                value=str(output_dir("trajectories")), label="Trajectory capture directory"
            ),
            "trajectory_storage": mo.ui.dropdown(
                ["compact", "throughput"], value="compact", label="HDF5 capture storage"
            ),
            "backend": mo.ui.dropdown(
                ["inherit", "auto", "cpu", "cuda", "rocm", "sycl"],
                value="inherit",
                label="Run backend (new process)",
            ),
        }
    )
    return (execution,)


@app.cell
def _(execution, mo, section):
    mo.stop(section.value != "Outputs and Exports")
    mo.vstack(
        [
            mo.md(
                "**Execution/persistence** choices below generate a command; they are not profile "
                "fields and do not enter checkpoint identity. Full runs retain the normal "
                "component checkpoints and configured detector observations. Histories are "
                "opt-in HDF5 captures; cached cases are not retransported. Use an explicit "
                "`--recompute` run if captures are needed for cached cases. Use "
                "`pyrite trajectories export` afterward for VTK. No simulation starts here. "
                "Independent PXR/CBS retention and arbitrary score/persistence subsets are not "
                "supported by profiles; source-array download filtering below is postprocessing. "
                "Pixel measurements and responses remain available through the Pixels app."
            ),
            execution,
        ]
    )
    return


@app.cell
def _(editor, execution, mo, run_command, section):
    mo.stop(section.value != "Outputs and Exports")
    try:
        _command = run_command(editor, **execution.value)
        _content = mo.vstack(
            [
                mo.md(
                    "Save the profile first, then run explicitly:\n```bash\n" + _command + "\n```"
                ),
                mo.download(
                    (_command + "\n").encode(),
                    filename="run-pyrite.sh",
                    label="Download run command",
                ),
            ]
        )
    except ValueError as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


@app.cell
def _(mo, output_dir, section):
    mo.stop(section.value != "Outputs and Exports")
    stored = mo.ui.dictionary(
        {
            "root": mo.ui.text(
                value=str(output_dir("checkpoints")), label="Existing checkpoint directory"
            ),
            "stem": mo.ui.text(label="Exact saved checkpoint stem (including identity variant)"),
        }
    ).form(submit_button_label="Load saved source arrays for export")
    stored
    return (stored,)


@app.cell
def _(mo, stored):
    from pyrite.storage import load_checkpoint

    mo.stop(stored.value is None)
    try:
        _store = load_checkpoint(stored.value["stem"], stored.value["root"])
        saved_records = {
            f"{name} / {energy:g} keV": record
            for name, energies in _store.items()
            for energy, record in energies.items()
        }
        mo.stop(not saved_records, mo.md("No saved records found; check the stem and directory."))
        record_picker = mo.ui.dropdown(
            list(saved_records), value=next(iter(saved_records)), label="Saved record"
        )
        _content = record_picker
    except (ValueError, OSError) as _error:
        saved_records, record_picker = {}, None
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return record_picker, saved_records


@app.cell
def _(mo, record_picker, saved_records):
    from pyrite.apps.configuration_exports import available_components, export_record

    mo.stop(record_picker is None)
    saved_record = saved_records[record_picker.value]
    export_components = mo.ui.multiselect(
        list(available_components(saved_record)), label="Stored source components to export"
    )
    export_format = mo.ui.dropdown(["npz", "csv"], value="npz", label="Download format")
    export_stem = mo.ui.text(value="spectrum", label="Download filename stem")
    export_metadata = mo.ui.checkbox(value=True, label="Include case metadata")
    mo.vstack(
        [
            mo.md(
                "Downloads preserve stored source values and grids; no detector response, "
                "flux normalization or physical-process selection is applied."
            ),
            export_components,
            export_format,
            export_stem,
            export_metadata,
        ]
    )
    return (
        export_components,
        export_format,
        export_metadata,
        export_record,
        export_stem,
        saved_record,
    )


@app.cell
def _(
    export_components, export_format, export_metadata, export_record, export_stem, mo, saved_record
):
    mo.stop(not export_components.value)
    try:
        _data = export_record(
            saved_record,
            export_components.value,
            export_format.value,
            metadata=export_metadata.value,
        )
        _content = mo.download(
            _data,
            filename=f"{export_stem.value}.{export_format.value}",
            label="Download selected source arrays",
        )
    except ValueError as _error:
        _content = mo.callout(mo.md(str(_error)), kind="warn")
    _content
    return


@app.cell
def _(apply, editor, mo, section):
    mo.stop(section.value != "Review and Save")
    _errors = editor.errors()
    _status = mo.callout(
        mo.md(
            "\n\n".join(_errors)
            if _errors
            else "Configuration valid; review the proposed changes before saving."
        ),
        kind="warn" if _errors else "success",
    )
    _save = mo.ui.button(
        label="Save catalog changes",
        disabled=bool(_errors),
        on_click=lambda _: apply(lambda e: e.save(), "Catalog changes saved"),
    )
    _downloads = [
        mo.download(
            editor.draft.serialize().encode(),
            filename="catalog-snapshot.toml",
            label="Download catalog TOML snapshot",
        )
    ]
    if not _errors:
        _json = editor.resolved_json()
        _downloads.append(
            mo.download(
                _json.encode(),
                filename="resolved-configuration.json",
                label="Download resolved inspection JSON",
            )
        )
        _review = mo.accordion(
            {
                "Resolved beam, detectors, numerics and material geometry": mo.md(
                    "```json\n" + _json + "\n```"
                )
            }
        )
    else:
        _review = mo.md("Correct the fields above; saving is disabled.")
    mo.vstack(
        [
            _status,
            _review,
            mo.accordion(
                {
                    "Proposed catalog diff": mo.md("```diff\n" + editor.draft.diff() + "\n```"),
                    "Canonical catalog TOML": mo.md(
                        "```toml\n" + editor.draft.serialize() + "\n```"
                    ),
                }
            ),
            mo.md(
                "**Save** writes every staged catalog object through existing catalog persistence. "
                "Concurrent edits are rejected; reopen and reapply. TOML snapshots retain asset "
                "paths and immutable grid references: transfer their referenced assets as well. "
                "Resolved JSON is an inspection report, not an executable configuration format."
            ),
            _save,
            *_downloads,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
