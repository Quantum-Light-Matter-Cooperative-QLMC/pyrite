"""Configuration section rendering and callbacks; no transport work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pyrite._config_presets import NUMERICS, PHYSICS
from pyrite.campaign.configuration_editor import ConfigurationEditor, plain
from pyrite.campaign.profile_detectors import profile_detector_entries

from .configuration_controls import calculation_defaults, fields_form, parameter_list


def transact(get_state: Callable, set_state: Callable, operation: Callable, message: str) -> None:
    """Keep rejected operations recoverable, with the previous draft intact."""
    editor, _ = get_state()
    try:
        updated = operation(editor)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        set_state((editor, f"Change rejected: {exc}"))
    else:
        set_state((updated, message))


def overview(mo: Any, editor: ConfigurationEditor, apply: Callable) -> Any:
    profiles = list(editor.draft.document["profiles"])
    selector = mo.ui.dropdown(
        profiles,
        value=editor.profile,
        label="Profile (unsaved edits are retained)",
        on_change=lambda name: apply(lambda current: current.select(name), "Profile selected"),
    )
    create = parameter_list(
        mo,
        {
            "name": mo.ui.text(label="New profile name"),
            "source": mo.ui.dropdown(
                {"New from packaged standard ranges": "", **{name: name for name in profiles}},
                value=editor.profile,
                label="Duplicate source, or create new",
            ),
        },
        title="New profile",
    ).form(
        submit_button_label="Create unsaved profile",
        on_change=lambda value: (
            value is not None
            and apply(
                lambda current: current.create(value["name"], value["source"] or None),
                "Profile created in draft; review and save explicitly",
            )
        ),
    )
    return mo.vstack(
        [
            selector,
            create,
            mo.md(
                "Bundled objects are read-only. Duplicate a profile to edit it. Cloning follows "
                "`profile create --from`: material overrides are not copied. All edits remain "
                "in memory until **Save catalog changes**."
            ),
        ]
    )


def geometry(mo: Any, editor: ConfigurationEditor, apply: Callable, thickness_key: str) -> Any:
    keys = ("energy_keV", "thickness_ang", "thickness_layers", "tilt_deg", "tilt_azim_deg")
    row = plain(editor.row)
    current = {key: row[key] for key in keys if key in row}
    suggestions = {
        "energy_keV": {"values": [30.0]},
        "thickness_ang": {"values": [20000.0]},
        "thickness_layers": {"values": [3]},
        "tilt_deg": {"values": [45.0]},
        "tilt_azim_deg": {"values": [180.0]},
    }

    def update(build: Callable) -> None:
        apply(lambda e: e.patch({**dict.fromkeys(keys), **build()}), "Scene and materials staged")

    members = mo.ui.multiselect(
        list(editor.draft.document["materials"]),
        value=list(row.get("materials", editor.draft.document["materials"])),
        label="Target materials (catalog geometry/cut/stack)",
    )
    return mo.vstack(
        [
            mo.md(
                "Sweep values are comma separated; units: kinetic energy **keV**, thickness "
                "**Å** (10,000 Å = 1 µm), layer count, incidence/azimuth **deg**. "
                "Choose thickness in Å or layers, as supported by the target. "
                "Catalog material cuts, stacks and ordered filters are preserved; arbitrary "
                "Scene bodies are not profile fields. Material overrides can supersede ranges; "
                "inspect the resolved review before saving."
            ),
            fields_form(
                mo,
                current,
                suggestions,
                update,
                title="Apply scene and materials",
                thickness_key=thickness_key,
                extra_controls={"materials": members},
                heading="Scene and materials",
            ),
        ]
    )


def calculation(
    mo: Any, editor: ConfigurationEditor, kind: str, advanced: bool, apply: Callable
) -> Any:
    policy = PHYSICS if kind == "physics" else NUMERICS
    row = plain(editor.row)
    keys = (*policy.keys, "E_grid_line", "E_grid_brem") if kind == "numerics" else policy.keys
    current = {key: row[key] for key in keys if key in row}
    suggestions = calculation_defaults(kind)
    if not advanced:
        common = (
            {"emission", "straggling", "elastic_model", "inelastic_model", "transition_radiation"}
            if kind == "physics"
            else {"n_electrons", "n_electrons_brem", "energy_model", "max_dE_frac", "mosaic_nodes"}
        )
        suggestions = {key: value for key, value in suggestions.items() if key in common}
    shown = set(current) | set(suggestions)
    presets = list(editor.draft.document.get(kind, {}))
    preset = mo.ui.dropdown(
        {"Inherit without a named preset": "", **{name: name for name in presets}},
        value=row.get(kind, "Inherit without a named preset"),
        label=f"Named {kind} preset",
    )
    if kind == "numerics":
        preset = preset.form(
            submit_button_label="Attach numerics preset (explicit overrides remain)",
            on_change=lambda value: (
                value is not None
                and apply(lambda e: e.patch({kind: value or None}), f"{kind} reference staged")
            ),
        )

    def update(build: Callable) -> None:
        def operation(e: ConfigurationEditor) -> ConfigurationEditor:
            values = build()
            if kind in values:
                values[kind] = values[kind] or None
            return e.patch({**dict.fromkeys(shown), **values})

        apply(operation, f"{kind} values staged")

    return mo.vstack(
        [
            mo.md(
                "**Override previous** writes an override. Unchecked restores preset/default "
                "resolution; displayed suggestions are not new defaults. Coupled settings are "
                "validated together. Fixed trial counts and adaptive precision cannot mix. "
                "PXR/CBS are a joint line scorer; emission selects incoherent/coherent scoring. "
                "These controls affect computation, not export filtering."
            ),
            *([preset] if kind == "numerics" else []),
            fields_form(
                mo,
                current,
                suggestions,
                update,
                title=f"Apply {kind} settings",
                extra_controls={"physics": preset} if kind == "physics" else None,
                heading="Physics" if kind == "physics" else "Parameters",
            ),
        ]
    )


def detector_membership(mo: Any, editor: ConfigurationEditor, apply: Callable) -> Any:
    entries = profile_detector_entries(editor.row)
    named = list(editor.draft.document.get("detectors", {}))
    form = parameter_list(
        mo,
        {
            "action": mo.ui.dropdown(
                ["add", "duplicate", "remove", "set", "migrate"], value="add", label="Action"
            ),
            "id": mo.ui.text(label="Detector ID (new ID for add/duplicate)"),
            "source": mo.ui.dropdown(
                list(entries), value=next(iter(entries)), label="Duplicate source ID"
            ),
            "reference": mo.ui.dropdown(named, label="Named detector (add/set)"),
            "migrate": mo.ui.checkbox(label="Explicitly convert legacy slots to an ID collection"),
        },
        title="Detector membership",
    ).form(
        submit_button_label="Apply detector membership",
        on_change=lambda value: (
            value is not None
            and apply(
                lambda e: e.detector(
                    value["id"],
                    action=value["action"],
                    values=value["reference"],
                    source=value["source"],
                    migrate=value["migrate"],
                ),
                "Detector membership staged",
            )
        ),
    )
    return mo.vstack(
        [
            mo.md(
                "Named objects are shared. Editing geometry below makes a profile-local copy; "
                "it never modifies the named detector. Adding IDs to legacy profiles requires "
                "explicit conversion, which can change the observation/checkpoint identity. "
                "The last explicit detector cannot be removed."
            ),
            mo.ui.table(
                [
                    {"ID": key, "reference": value if isinstance(value, str) else "inline"}
                    for key, value in entries.items()
                ],
                selection=None,
            ),
            form,
        ]
    )
