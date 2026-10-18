"""Detector catalog construction and patches shared by interactive and batch UIs."""

import re
from copy import deepcopy

import tomlkit

SCALAR_FIELDS = ("observation_angle_deg", "polar_acceptance_deg", "solid_angle_sr")


def patch_detector(target, scalar, pixel):
    """Apply already typed values; domain constructors validate the resulting draft."""
    geometry, scorer, response, acquisition = pixel
    if any(pixel):
        if scalar or any(key in target for key in SCALAR_FIELDS):
            raise ValueError("pixel geometry cannot be combined with scalar acceptance")
        if "distance_mm" not in target and "distance_mm" not in geometry:
            raise ValueError("creating pixel geometry requires --distance-mm")
        merge_pixel_detector(
            target, geometry=geometry, scorer=scorer, response=response, acquisition=acquisition
        )
    elif scalar and "distance_mm" in target:
        raise ValueError("pixel detector acceptance is derived from its geometry")
    target.update(scalar)


def create_detector(document, name, *, source=None, label=None, scalar=None, pixel=None, kind=None):
    """Stage a named detector/clone without I/O; reject conflicting geometry kinds."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"invalid detector name {name!r}; use letters, digits, '.', '_', '-'")
    rows = document.get("detectors", {})
    if any(key.casefold() == name.casefold() for key in rows):
        raise ValueError(f"detector {name!r} already exists; choose a new name")
    if source is not None and source not in rows:
        raise ValueError(f"unknown detector: {source}; use pyrite detector list")
    target = deepcopy(rows[source]) if source is not None else tomlkit.table()
    scalar = scalar or {}
    pixel = pixel or ({}, {}, {}, {})
    if not scalar and not any(pixel) and source is None:
        raise ValueError("provide at least one detector-geometry option")
    patch_detector(target, scalar, pixel)
    actual = "planar" if "distance_mm" in target else "scalar"
    if kind is not None and actual != kind:
        raise ValueError(
            f"--from {source} is {actual}; use detector create {actual} or create clone"
        )
    if label is not None:
        target["label"] = label
    if "detectors" not in document:
        document["detectors"] = tomlkit.table()
    document["detectors"][name] = target
    return target


_ACQUISITION_AXIS_KEYS = (
    "reporting_bins",
    "measured_edges_eV",
    "measured_min_eV",
    "measured_max_eV",
    "measured_bin_width_eV",
)


def _toml_value(value):
    if isinstance(value, dict):
        table = tomlkit.table()
        for key, item in value.items():
            table[key] = _toml_value(item)
        return table
    return list(value) if isinstance(value, tuple) else value


def merge_pixel_detector(table, *, geometry, scorer=None, response=None, acquisition=None):
    """Merge supplied pixel fields, replacing incompatible response/bin sections."""
    for key, value in geometry.items():
        table[key] = _toml_value(value)
    if scorer:
        section = table.setdefault("scorer", tomlkit.table())
        for key, value in scorer.items():
            section[key] = _toml_value(value)
    if response:
        current = table.get("response")
        kind = response.get("kind")
        parameters = {key: value for key, value in response.items() if key != "kind"}
        effective_kind = kind if kind is not None else (current or {}).get("kind")
        if parameters and effective_kind != "timepix3":
            raise ValueError("Timepix3 response options require --response timepix3")
        if kind is not None and (current is None or current.get("kind") != kind):
            table["response"] = _toml_value(dict(response))
        else:
            for key, value in parameters.items():
                current[key] = _toml_value(value)
    if acquisition:
        section = table.setdefault("acquisition", tomlkit.table())
        if any(key in acquisition for key in _ACQUISITION_AXIS_KEYS):
            for key in _ACQUISITION_AXIS_KEYS:
                section.pop(key, None)
        if acquisition.get("mode") == "expected":
            section.pop("seed", None)
        for key, value in acquisition.items():
            section[key] = _toml_value(value)


def acquisition_changes(values):
    """Typed acquisition option values to TOML keys; reject conflicting reporting bins."""
    edges = values["measured_edges_ev"]
    uniform = values["measured_range_ev"], values["measured_bin_width_ev"]
    native = values.get("reporting_bins")
    if native is not None and (edges is not None or any(item is not None for item in uniform)):
        raise ValueError("--reporting-bins native conflicts with explicit measured bins")
    if edges is not None and any(item is not None for item in uniform):
        raise ValueError(
            "use --measured-edges-ev or --measured-range-ev with --measured-bin-width-ev, not both"
        )
    if (uniform[0] is None) != (uniform[1] is None):
        raise ValueError("--measured-range-ev and --measured-bin-width-ev go together")
    if edges is not None and any(
        right <= left for left, right in zip(edges, edges[1:], strict=False)
    ):
        raise ValueError("--measured-edges-ev must be strictly increasing")
    if edges is not None and len(edges) < 2:
        raise ValueError("--measured-edges-ev needs at least two edges")
    if values["realization"] == "expected" and values["realization_seed"] is not None:
        raise ValueError("--realization-seed applies only to --realization poisson")
    changes = {}
    if native is not None:
        changes["reporting_bins"] = native
    if values["exposure_s"] is not None:
        changes["exposure_s"] = values["exposure_s"]
    if edges is not None:
        changes["measured_edges_eV"] = list(edges)
    if uniform[0] is not None:
        minimum, maximum = uniform[0]
        changes.update(
            measured_min_eV=minimum, measured_max_eV=maximum, measured_bin_width_eV=uniform[1]
        )
    if values["hit_threshold_ev"] is not None:
        changes["hit_threshold_eV"] = values["hit_threshold_ev"]
    if values["realization"] is not None:
        changes["mode"] = values["realization"]
    if values["realization_seed"] is not None:
        changes["seed"] = values["realization_seed"]
    return changes


def pixel_changes(values):
    """Split option-keyed typed values into (geometry, scorer, response, acquisition).

    ``values`` uses the ``pyrite detector`` option names (``None`` when unset), so
    the CLI and the terminal editor stage identical patches.
    """
    geometry = {
        key: values[key]
        for key in (
            "distance_mm",
            "polar_deg",
            "azimuth_deg",
            "roll_deg",
            "offset_mm",
            "shape",
            "pitch_mm",
        )
        if values[key] is not None
    }
    scorer = {
        key: values[key] for key in ("reconstruction", "angular_shape") if values[key] is not None
    }
    response = {
        key: values[option]
        for key, option in (
            ("kind", "response"),
            ("thickness_um", "timepix_thickness_um"),
            ("bias_v", "timepix_bias_v"),
            ("dE_mc", "timepix_input_bin_ev"),
            ("dE_out", "timepix_output_bin_ev"),
            ("n_mc", "timepix_samples"),
            ("seed", "timepix_seed"),
        )
        if values[option] is not None
    }
    acquisition = acquisition_changes(values)
    return geometry, scorer, response, acquisition
