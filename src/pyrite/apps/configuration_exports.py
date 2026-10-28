"""Explicit postprocessing downloads of existing stored source arrays."""

import csv
import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyrite._spectral_components import LINE_COMPONENT_KEYS
from pyrite.campaign.configuration_editor import plain


def available_components(record: Mapping[str, Any]) -> dict[str, str]:
    """Stored components only; do not manufacture independent PXR/CBS channels."""
    axes = dict.fromkeys(("spec", "spec_coherent", *LINE_COMPONENT_KEYS, "brem"), "E_grid")
    axes["brem_wide"] = "E_grid_brem"
    return {
        key: axis for key, axis in axes.items() if record.get(key) is not None and axis in record
    }


def export_record(
    record: Mapping[str, Any], selected: Sequence[str], format: str, *, metadata: bool = True
) -> bytes:
    """Export an explicit source-array subset, preserving separate grids and units.

    No detector response or bin conversion is applied. NPZ embeds each original
    axis and optional JSON provenance; CSV uses long rows with the source key.
    """
    axes = available_components(record)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(key not in axes for key in selected)
    ):
        raise ValueError("outputs: select distinct available stored components")
    arrays = {}
    for key in selected:
        axis = axes[key]
        energy = np.asarray(record[axis])
        density = np.asarray(record[key])
        if energy.ndim != 1 or density.shape != energy.shape:
            raise ValueError(f"{key}: export requires matching one-dimensional node arrays")
        arrays[axis] = energy
        arrays[key] = density
    info = {
        "components": list(selected),
        "representation": "stored source arrays; no detector response applied",
        "energy_unit": "eV",
        "case": plain(record.get("case", {})),
        "scale": plain(record.get("scale")),
    }
    if format == "npz":
        output = io.BytesIO()
        if metadata:
            arrays["metadata_json"] = np.asarray(json.dumps(info))
        np.savez_compressed(output, **arrays)
        return output.getvalue()
    if format != "csv":
        raise ValueError("format: choose npz or csv")
    text = io.StringIO()
    if metadata:
        text.write("# " + json.dumps(info) + "\n")
    writer = csv.writer(text)
    writer.writerow(("component", "energy_eV", "stored_value"))
    for key in selected:
        writer.writerows((key, x, y) for x, y in zip(arrays[axes[key]], arrays[key], strict=True))
    return text.getvalue().encode()
