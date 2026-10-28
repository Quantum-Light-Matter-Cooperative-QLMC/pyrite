"""Typed widget adapters for catalog values; domain parsers remain authoritative."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from html import escape
from typing import Any, Literal, get_args, get_origin

from pyrite._config_presets import EMISSION_VALUES, NUMERICS, PHYSICS
from pyrite._cxr_vocabulary import CATALOG_FIELDS
from pyrite._numerics import CxrReflections, Numerics, Physics
from pyrite.campaign.configuration_editor import plain


def label(key: str) -> str:
    """Keep canonical names and their unit suffixes visible."""
    reverse = {value: name for name, value in CATALOG_FIELDS.items()}
    return " / ".join(
        reverse.get(part, part).replace("_", " ") for part in key.split(".") if part != "values"
    )


def parameter_list(
    mo: Any, elements: Mapping[str, Any], *, title: str | None = "Parameters"
) -> Any:
    """A vertical form layout with named values and no JSON-tree presentation."""
    heading = f"<h3>{escape(title)}</h3>" if title else ""
    rows = "".join(f"<div>{{{key}}}</div>" for key in elements)
    return mo.ui.batch(
        mo.Html(
            heading + '<div style="display:flex;flex-direction:column;gap:1rem">' + rows + "</div>"
        ),
        elements,
    )


def choices() -> dict[str, list[Any]]:
    result = {"emission": list(EMISSION_VALUES)}
    for cls in (Physics, Numerics, CxrReflections):
        for field in fields(cls):
            name, annotation = field.name, field.type
            types = (annotation, *get_args(annotation))
            values = [
                value for item in types if get_origin(item) is Literal for value in get_args(item)
            ]
            if values:
                result[name] = values
    return result


CHOICES = choices()


def calculation_defaults(kind: str) -> dict[str, Any]:
    """Suggestions from typed API defaults; unchecked fields retain inheritance."""
    policy = PHYSICS if kind == "physics" else NUMERICS
    values = {**plain(Physics()), **plain(Numerics()), **plain(CxrReflections())}
    values["emission"] = Physics().emission_policy
    values.update(
        n_electrons={"values": [Numerics().cxr_trials]},
        n_electrons_brem={"values": [Numerics().brem_trials]},
    )
    values.update(
        inelastic_cutoff_eV=50.0,
        secondary_threshold_eV=1000.0,
        radiative_cutoff_eV=1000.0,
        pair_production_model="penelope-2024",
        channeling_plane=[0, 0, 1],
        transition_lattice_origin_ang=[0.0, 0.0, 0.0],
        channeling_zone_max_ang=1000.0,
        max_reflections=10,
    )
    suggestions = {key: values[key] for key in policy.scalar_keys if values.get(key) is not None}
    if kind == "numerics":
        from pyrite._precision import DEFAULT_PRECISION

        suggestions["precision"] = DEFAULT_PRECISION.to_dict()
        suggestions["line_grid_policy"] = {
            "bandwidth": "kinematic-ceiling",
            "resolution": "sinc-nyquist",
            "quadrature": "node",
            "windows": False,
        }
        suggestions["E_grid_line"] = {"values": [1000.0, 2000.0, 3000.0]}
        suggestions["E_grid_brem"] = {"arange": {"start": 1000.0, "stop": 30000.0, "step": 25.0}}
    return suggestions


def beam_template(mode: str, longitudinal: str) -> dict[str, Any]:
    from pyrite.campaign.sweep import BeamSpec

    defaults = BeamSpec()
    values: dict[str, Any] = {
        "rep_rate_hz": defaults.rep_rate_hz,
        "bunch_charge_pc": defaults.bunch_charge_pc,
    }
    if mode == "GDF":
        return {
            **values,
            "source": "gpt_gdf",
            "gdf_path": "",
            "gdf_z_origin_m": 0.0,
            "gdf_time_s": 0.0,
            "gdf_time_tolerance_s": 1e-15,
            "gdf_screen_position_m": 0.0,
            "gdf_screen_tolerance_m": 1e-9,
            "gdf_normalization": "pyrite_current",
            "gdf_shape_only": False,
        }
    if mode == "Twiss":
        values["transverse"] = {
            "normalized_emittance_x_mm_mrad": 1.0,
            "normalized_emittance_y_mm_mrad": 1.0,
            "beta_twiss_x_m": 1.0,
            "beta_twiss_y_m": 1.0,
            "alpha_twiss_x": 0.0,
            "alpha_twiss_y": 0.0,
        }
    else:
        values.update(transverse_fwhm_mm=defaults.transverse_fwhm_x_mm, energy_spread_frac=0.01)
    if longitudinal == "Bunch duration":
        values.update(bunch_length_fs=100.0, long_shape="gaussian")
    elif longitudinal in {"gaussian", "microtrain", "compressed"}:
        policy: dict[str, Any] = {"kind": longitudinal}
        if longitudinal != "compressed":
            policy["envelope_rms_fs"] = 100.0
        if longitudinal != "gaussian":
            policy.update(
                retained_coherence=0.9,
                spacing_periods=1,
                modulation_depth=1.0,
                timing_jitter_fs=0.0,
            )
        values["longitudinal"] = policy
    return values


def detector_template(pixel: bool) -> dict[str, Any]:
    """Explicit creation suggestions, never a fallback for an invalid detector."""
    from pyrite.detectors.spec import Detector

    if not pixel:
        detector = Detector()
        return {
            key: getattr(detector, key)
            for key in ("observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr")
        }
    return {
        "distance_mm": 400.0,
        "polar_deg": 90.0,
        "azimuth_deg": 0.0,
        "roll_deg": 0.0,
        "offset_mm": [0.0, 0.0],
        "shape": [256, 256],
        "pitch_mm": [0.055, 0.055],
        "scorer": {"reconstruction": "nearest_tile", "angular_shape": [2, 2]},
        "response": {"kind": "ideal", "thickness_um": 300.0, "bias_v": 100.0},
        "acquisition": {
            "exposure_s": 1.0,
            "measured_min_eV": 0.0,
            "measured_max_eV": 30000.0,
            "measured_bin_width_eV": 100.0,
            "hit_threshold_eV": 0.0,
            "mode": "expected",
            "seed": 0,
        },
    }


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    fields = {}
    for name, value in values.items():
        if value is None:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            fields.update(_flatten(value, path))
        else:
            fields[path] = value
    return fields


def _widget(mo: Any, path: str, value: Any) -> Any:
    key = path.rsplit(".", 1)[-1]
    options = CHOICES.get(key)
    if options and value in options:
        return mo.ui.dropdown(options, value=value, label=label(path))
    if isinstance(value, bool):
        return mo.ui.checkbox(value=value, label=label(path))
    if isinstance(value, (int, float)):
        return mo.ui.number(
            value=value, step=1 if isinstance(value, int) else None, label=label(path)
        )
    if isinstance(value, list):
        return mo.ui.text(
            value=", ".join(str(item) for item in value), label=label(path) + " (comma separated)"
        )
    return mo.ui.text(value=str(value), label=label(path), full_width=True)


def _decode(value: Any, template: Any) -> Any:
    if isinstance(template, list):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("list must contain at least one value")
        converter = type(template[0]) if template else float
        return [converter(part) for part in parts]
    return value


def field_values(submitted: Mapping, templates: Mapping) -> dict[str, Any]:
    """Decode typed controls into ordinary existing catalog values."""
    result: dict[str, Any] = {}
    for path, setting in submitted.items():
        if not setting["explicit"]:
            continue
        try:
            value = _decode(setting["value"], templates[path])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        parts = path.split(".")
        parent = result
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
        parent[parts[-1]] = value
    return result


def fields_form(
    mo: Any,
    current: Mapping,
    suggestions: Mapping,
    submit: Callable,
    *,
    title: str,
    thickness_key: str | None = None,
    extra_controls: Mapping[str, Any] | None = None,
    heading: str = "Parameters",
) -> Any:
    """Optional typed controls. Unchecked means absent, preserving defaults/presets.

    Existing fields always remain visible, including nested supported settings
    for which there is no creation suggestion.
    """
    current, suggestions = plain(current), plain(suggestions)
    if thickness_key is not None:
        thickness_keys = {"thickness_ang", "thickness_layers"}
        if thickness_key not in thickness_keys:
            raise ValueError("Choose thickness in angstroms or layers")
        had_thickness = bool(thickness_keys.intersection(current))
        for other in thickness_keys - {thickness_key}:
            current.pop(other, None)
            suggestions.pop(other, None)
        if had_thickness and thickness_key not in current:
            current[thickness_key] = suggestions[thickness_key]
        if thickness_key in current and not isinstance(current[thickness_key], Mapping):
            current[thickness_key] = {"values": [current[thickness_key]]}
    present = _flatten(current)
    templates = {**_flatten(suggestions), **present}
    paths = list(templates)
    widgets = {
        f"parameter_{index}": mo.ui.batch(
            mo.Html(
                '<div style="display:flex;align-items:center;gap:0.75rem">'
                '<div>{value}</div><div style="white-space:nowrap">{explicit}</div></div>'
            ),
            {
                "value": _widget(mo, path, value),
                "explicit": mo.ui.checkbox(value=path in present, label="Override previous"),
            },
        )
        for index, (path, value) in enumerate(templates.items())
    }
    extra_controls = extra_controls or {}
    widgets = {**extra_controls, **widgets}

    def changed(value: Any) -> None:
        if value is not None:
            submit(
                lambda: {
                    **field_values(
                        {path: value[f"parameter_{index}"] for index, path in enumerate(paths)},
                        templates,
                    ),
                    **{key: value[key] for key in extra_controls},
                }
            )

    return parameter_list(mo, widgets, title=heading).form(
        submit_button_label=title,
        on_change=changed,
        bordered=True,
        show_clear_button=False,
    )
