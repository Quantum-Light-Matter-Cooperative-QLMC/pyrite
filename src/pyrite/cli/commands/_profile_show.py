"""Human profile summary, grouped by the CLI's settings owners."""

import shutil
import textwrap

import click

from pyrite._energy_grid_encoding import encode_energy_grid
from pyrite.campaign import profile_edit
from pyrite.cli import _catalog_io
from pyrite.console.output import emit_result
from pyrite.materials._catalog_decode import resolve_grid_descriptor
from pyrite.materials._schema import DEFAULT_BREM_GRID


def _display(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_display(item) for item in value) + "]"
    return str(value)


def _rows(values, prefix=""):
    """Flatten nested instrument settings without printing Python dictionaries."""
    for key, value in values.items():
        label = prefix + key.replace("_", "-")
        if isinstance(value, dict):
            yield from _rows(value, label + ".")
        else:
            yield label, _display(value)


def _table(title, rows):
    rows = list(rows) or [("settings", "none")]
    width = max([len("Setting"), *(len(label) for label, _value in rows)])
    emit_result(f"\n{title}")
    emit_result(f"  {'Setting':<{width}}  Value")
    emit_result(f"  {'-' * width}  {'-' * 5}")
    context = click.get_current_context(silent=True)
    wide = context is not None and context.meta.get("pyrite.output_format") == "wide"
    columns = max(60, shutil.get_terminal_size().columns)
    for label, value in rows:
        lines = (
            [value]
            if wide
            else textwrap.wrap(
                value,
                width=max(16, columns - width - 4),
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
        for index, line in enumerate(lines):
            emit_result(f"  {label if index == 0 else '':<{width}}  {line}")


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
    sources = profile_edit.cxr_grid_sources(document, target)
    policy = profile_edit.profile_cxr_grid_values(target)
    refs = profile_edit.energy_grid_refs(target)
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
    for material, override in profile_edit.profile_overrides(target).items():
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


def emit_show(payload, target, resolution, numerics_names, grid_rows):
    emit_result(f"[{payload['name']}]")
    ranges = [
        (row["name"], f"[{_catalog_io.display(row['values'])}]")
        for row in payload["ranges"]
        if row["catalog_key"] in profile_edit.RANGES.values()
    ]
    units = {"thickness": "Angstrom", "energy": "keV", "polar": "deg", "azimuth": "deg"}
    ranges = [(label, f"{value} {units[label]}") for label, value in ranges]
    # Layer-count grids have no CLI setter but remain useful sweep axes.
    if "thickness_layers" in target:
        ranges.append(
            ("thickness-layers", _display(profile_edit.range_values(target, "thickness_layers")))
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
            ("emission", payload["emission"] or "incoherent (default)"),
            ("temporal-profile", "on" if payload["temporal_profile"] else "off (default)"),
        ],
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
        for key, label, unit in profile_edit.ACTIVE_DETECTOR_FIELDS:
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
            "profile physical-detector; pyrite detector"
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
                f"{row.get('material')}, {row['thickness_mm']:g} mm, {_display(row.get('size_mm', []))} mm",
            )
            for index, row in enumerate(filters, start=1)
        ]
        or [("filters", "none")],
    )

    for group in resolution.groups():
        rows = []
        for row in group["fields"]:
            label = numerics_names[row["key"]]
            if group["name"] == "sampling" and payload["precision"]["effective"] is not None:
                value = "adaptive (profile precision)"
            else:
                source = "default" if row["source"] == "fidelity" else row["source"]
                value = f"{_display(row['effective'])} ({source})"
            rows.append((label, value))
        _table(f"Numerics / {group['name']} (profile numerics)", rows)

    precision = payload["precision"]
    if precision["effective"] is None:
        precision_rows = [("mode", f"fixed ({precision['fixed_reason']})")]
    else:
        precision_rows = [
            ("mode", f"adaptive ({precision['source']} policy)"),
            *list(_rows(precision["effective"])),
        ]
    _table("Precision (profile precision)", precision_rows)

    policy = payload["line_grid_policy"]
    _table(
        "CXR-grid policy (profile cxr-grid)",
        [("source", "automatic for every case"), *list(_rows(policy))]
        if policy
        else [("policy", "none (explicit, stored or built-in grids)")],
    )

    _table("Energy grids (profile defaults / material energy-grid)", grid_rows)
    non_grid_overrides = {
        material: [key for key in labels if key not in {"E_grid_line", "E_grid_brem"}]
        for material, labels in payload["overrides"].items()
    }
    _table(
        "Material overrides (non-grid catalog TOML)",
        [(material, ", ".join(labels)) for material, labels in non_grid_overrides.items() if labels]
        or [("overrides", "none")],
    )
