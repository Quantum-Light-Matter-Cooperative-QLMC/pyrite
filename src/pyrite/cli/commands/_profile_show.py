"""Human profile summary, grouped by the CLI's settings owners."""

import shutil
import textwrap

import click

from pyrite.campaign import profile_edit
from pyrite.cli import _catalog_io
from pyrite.console.output import emit_result


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


def emit_show(payload, target, resolution, numerics_names):
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
        "Line-grid policy (profile line-grid)",
        [("source", "automatic for every case"), *list(_rows(policy))]
        if policy
        else [("policy", "none (explicit, stored or built-in grids)")],
    )

    grids = []
    for key in ("E_grid_line", "E_grid_brem"):
        if key in target:
            grids.extend(_rows(target[key], key + "."))
    refs = payload["energy_grid_refs"]
    _table(
        "Energy grids (material energy-grid)",
        [
            (
                "storage",
                "stored per-material artifacts"
                if refs
                else "inline (E_grid_brem + material overrides)",
            ),
            *grids,
            *[(material, digest) for material, digest in refs.items()],
        ],
    )
    _table(
        "Material overrides (catalog TOML)",
        [(material, ", ".join(labels)) for material, labels in payload["overrides"].items()]
        or [("overrides", "none")],
    )
