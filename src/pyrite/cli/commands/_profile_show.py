"""Human profile summary, grouped by the CLI's settings owners."""

from pyrite._energy_grid_encoding import encode_energy_grid
from pyrite._numerics import PHYSICS_FIELDS
from pyrite.campaign import _profile_document, profile_cxr_grid, profile_detectors
from pyrite.cli import _catalog_io
from pyrite.console.output import emit_result
from pyrite.console.tables import display, emit_settings, emit_table
from pyrite.materials._catalog_decode import resolve_grid_descriptor
from pyrite.materials._schema import DEFAULT_BREM_GRID


def _display(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:g}"
    return display(value)


def _rows(values, prefix=""):
    """Flatten nested instrument settings without printing Python dictionaries."""
    for key, value in values.items():
        label = prefix + key.replace("_", "-")
        if isinstance(value, dict):
            yield from _rows(value, label + ".")
        else:
            yield label, _display(value)


def _table(title, rows):
    emit_result("")
    emit_table(("setting", "value"), list(rows) or [("settings", "none")], title=title, indent=2)


def _grid_summary(value, *, brem=False):
    grid = resolve_grid_descriptor(value)
    encoded = encode_energy_grid(grid)
    if brem and isinstance(encoded, tuple):
        start, _stop, step = encoded
        lower = "material floor" if start <= 0 else f"max({start:g} eV, material floor)"
        return f"uniform {step:g} eV; start = {lower}; stop = case beam energy + step"
    coordinates = (
        _display(grid.tolist())
        if grid.size <= 6
        else f"{grid.size} nodes, {grid[0]:g}..{grid[-1]:g}"
    )
    return f"{coordinates} eV (exact grid)"


def energy_grid_rows(document, target):
    """Report declared sources and case semantics without resolving trajectories."""
    sources = profile_cxr_grid.cxr_grid_sources(document, target)
    policy = profile_cxr_grid.profile_cxr_grid_values(target)
    refs = _profile_document.energy_grid_refs(target)
    if policy:
        line = "automatic for every case (profile policy)"
    elif "E_grid_line" in target:
        line = _grid_summary(target["E_grid_line"]) + " (profile)"
    elif sources["environment"]:
        line = "automatic (environment policy)"
    else:
        line = "automatic for uncovered energies (built-in)"
    rows = [("line default", line)]
    if "E_grid_brem" in target:
        brem = _grid_summary(target["E_grid_brem"], brem=True) + " (profile)"
    else:
        start, stop, step = DEFAULT_BREM_GRID
        brem = _grid_summary({"arange": {"start": start, "stop": stop, "step": step}}, brem=True)
        brem += " (built-in)"
    rows.append(("brem default", brem))
    if policy and "E_grid_line" in target:
        rows.append(
            ("line explicit", _grid_summary(target["E_grid_line"]) + " (inactive: profile policy)")
        )
    for material, override in _profile_document.profile_overrides(target).items():
        for key, name in (("E_grid_line", "line"), ("E_grid_brem", "brem")):
            if key not in override:
                continue
            suffix = " (material override)"
            if name == "line" and policy:
                suffix = " (inactive: profile policy)"
            elif material in refs:
                suffix = " (superseded by stored artifact)"
            rows.append(
                (f"{material}.{name}", _grid_summary(override[key], brem=name == "brem") + suffix)
            )
    for material, digest in refs.items():
        rows.append((f"{material}.artifact", digest))
    if refs:
        rows.append(
            (
                "artifact precedence",
                "supersedes inline brem grids; line uses profile policy"
                if policy
                else "supersedes inline grids; stores brem and covered line energies",
            )
        )
    if sources["shared_energy_grids"]:
        suffix = " (inactive: profile policy)" if policy else " (covered energies only)"
        rows.append(("shared line rows", ", ".join(sources["shared_energy_grids"]) + suffix))
    if sources["environment"]:
        rows.append(
            (
                "line environment",
                ", ".join(sources["environment"]) + "; automatic ahead of stored rows",
            )
        )
    return rows


def _emission_label(payload):
    """Explicit emission, else the physics preset's, else the built-in default."""
    if payload["emission"]:
        return payload["emission"]
    preset = payload["presets"]["physics"]
    if preset:
        return f"{payload['effective_emission']} (preset {preset})"
    return "incoherent (default)"


def _preset_rows(payload):
    return [(table, name or "none") for table, name in payload["presets"].items()]


def _filter_summary(row):
    if "ref" in row:
        return f"preset {row['ref']}"
    return (
        f"{row.get('material')}, {row['thickness_mm']:g} mm, {_display(row.get('size_mm', []))} mm"
    )


def _emit_targets(payload):
    if payload.get("targets"):
        _table(
            "Targets; profile geometry above is unused for these materials (pyrite target)",
            [(material, str(name)) for material, name in payload["targets"].items()],
        )


_NUMERICS_GROUP_LABELS = {
    "convergence": "reflections and mosaic",
    "channeling": "channeling (quantum zone)",
}


def _numerics_group_rows(group, numerics_names, adaptive_active):
    """SETTING/VALUE/SOURCE rows for one group; channeling collapses to its model when off."""
    fields = group["fields"]
    if group["name"] == "channeling" and any(
        row["key"] == "channeling_model" and row["effective"] == "off" for row in fields
    ):
        fields = [row for row in fields if row["key"] == "channeling_model"]
    return [
        (
            numerics_names[row["key"]],
            "unused (adaptive)"
            if group["name"] == "sampling" and adaptive_active
            else _display(row["effective"]),
            row["source"],
            row["key"] in PHYSICS_FIELDS,
        )
        for row in fields
    ]


def _emit_calculation_groups(resolution, numerics_names, adaptive_active):
    """Numerics and Physics groups, each titled with the command that owns it."""
    for group in resolution.groups():
        label = _NUMERICS_GROUP_LABELS.get(group["name"], group["name"])
        rows = _numerics_group_rows(group, numerics_names, adaptive_active)
        for physics, owner in ((False, "Numerics"), (True, "Physics")):
            selected = [row[:3] for row in rows if row[3] is physics]
            if selected:
                emit_result("")
                emit_settings(f"{owner} / {label} (profile {owner.lower()})", selected)


def emit_show(payload, target, resolution, numerics_names, grid_rows):
    emit_result(f"[{payload['name']}]")
    ranges = [
        (row["name"], f"[{_catalog_io.display(row['values'])}]")
        for row in payload["ranges"]
        if row["catalog_key"] in _profile_document.RANGES.values()
    ]
    units = {"thickness": "Angstrom", "energy": "keV", "polar": "deg", "azimuth": "deg"}
    ranges = [(label, f"{value} {units[label]}") for label, value in ranges]
    # Layer-count grids have no CLI setter but remain useful sweep axes.
    if "thickness_layers" in target:
        ranges.append(
            (
                "thickness-layers",
                _display(_profile_document.range_values(target, "thickness_layers")),
            )
        )
    materials = payload["materials"]
    _table(
        "Sweep / membership / emission (profile set|add|remove)",
        [
            *ranges,
            (
                "materials",
                "all catalog materials (implicit)"
                if materials is None
                else ", ".join(materials) or "(none)",
            ),
            ("emission", _emission_label(payload)),
            ("temporal-profile", "on" if payload["temporal_profile"] else "off (default)"),
        ],
    )

    _emit_targets(payload)

    _table(
        "Named presets (profile set --physics/--numerics; pyrite physics|numerics)",
        _preset_rows(payload),
    )

    if payload["beam_ref"] is not None:
        beam_rows = [("reference", f"{payload['beam_ref']} (named reference; pyrite beam show)")]
    elif payload["beam"] is not None:
        beam_rows = list(_rows(payload["beam"]))
    else:
        beam_rows = [("beam", "none (default)")]
    _table("Beam (profile set --beam; pyrite beam)", beam_rows)

    for detector_id, entry in payload["detectors"].items():
        rows = [("kind", entry["kind"]), ("reference", entry["reference"] or "inline / built-in")]
        for key, label, unit in profile_detectors.ACTIVE_DETECTOR_FIELDS:
            value = entry["acceptance"][key]
            rows.append((label, "unspecified" if value is None else f"{value:g} {unit}"))
        if entry["kind"] == "pixel":
            defaults = {
                "polar_deg": 90.0,
                "azimuth_deg": 0.0,
                "roll_deg": 0.0,
                "offset_mm": (0.0, 0.0),
                "shape": (256, 256),
                "pitch_mm": (0.055, 0.055),
            }
            rows.extend(_rows({**defaults, **entry["settings"]}))
        owner = (
            "profile detector; pyrite detector"
            if entry["kind"] == "pixel"
            else "profile set --detector; pyrite detector"
        )
        _table(f"Detector {detector_id} ({owner})", rows)

    filters = payload["filters"]
    _table(
        "Filters (profile filter)",
        [
            (
                f"{index}. {row.get('name') or '(unnamed)'}",
                _filter_summary(row),
            )
            for index, row in enumerate(filters, start=1)
        ]
        or [("filters", "none")],
    )

    adaptive_active = payload["adaptive_sampling"]["effective"] is not None
    _emit_calculation_groups(resolution, numerics_names, adaptive_active)

    adaptive = payload["adaptive_sampling"]
    if adaptive["effective"] is None:
        adaptive_rows = [("mode", f"fixed ({adaptive['fixed_reason']})")]
    else:
        adaptive_rows = [
            ("mode", f"adaptive ({adaptive['source']} policy)"),
            *list(_rows(adaptive["effective"])),
        ]
    _table("Adaptive sampling (profile numerics adaptive-sampling)", adaptive_rows)

    policy = payload["line_grid_policy"]
    _table(
        "CXR-grid policy (profile numerics cxr-grid)",
        [("source", "automatic for every case"), *list(_rows(policy))]
        if policy
        else [("policy", "none (explicit, stored or built-in grids)")],
    )

    _table("Energy grids (profile defaults / numerics energy-grid)", grid_rows)
    non_grid_overrides = {
        material: [key for key in labels if key not in {"E_grid_line", "E_grid_brem"}]
        for material, labels in payload["overrides"].items()
    }
    _table(
        "Material overrides (non-grid catalog TOML)",
        [(material, ", ".join(labels)) for material, labels in non_grid_overrides.items() if labels]
        or [("overrides", "none")],
    )
