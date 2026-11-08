"""Compile a Si axial depth budget from merged fibre-source arms and fate samples (#993).

Usage::

    python checks/axial_depth_calibration.py SPEC.json OUT.json

``SPEC.json``::

    {"case_key": ..., "model_id": ..., "evidence_id": ...,
     "checkpoints_ang": [...], "span_ang": 15.0,
     "bands_eV": {"whole": [800, 8000], "line": [2800, 4000]},
     "arms": {"control": "ctl.json", "all_on": ..., "return_flux": ...,
              "unresolved": ..., "matching_surface": ..., "grid_supercell": ...,
              "source_model": ...},
     "fates": {"control": "fates-ctl.json", "all_on": "fates-all.json"}}

Arms are ``checks/axial_fiber_source.py merge`` outputs run with
``checkpoints_ang``; fates are ``checks/axial_fiber_fates.py merge`` outputs.
One evidence record is compiled per band (observable key ``<band>``). The
output keeps every interval's components, the audit numbers and a
batch-doubling ratio of the control standard error as the normality check.

Validation: cr-axial-depth-calibration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pyrite.montecarlo.channeling.axial_depth_calibration import (
    ArmBatches,
    batch_mean_se,
    calibrate_axial_depth,
)
from pyrite.montecarlo.channeling.axial_outgoing_launch import AxialHandoffRecord


def arm_batches(merged: dict, band: tuple[float, float]) -> ArmBatches:
    """Per-chunk mean band yields (photons/sr per incident) at each recorded thickness."""
    energies = np.asarray(merged["photon_energy_eV"])
    keep = (energies >= band[0]) & (energies <= band[1])
    chunks = merged["chunks"]
    thickness = tuple(chunks[0]["thickness_ang"])
    if any(tuple(c["thickness_ang"]) != thickness for c in chunks):
        raise ValueError("chunks record different thicknesses")
    counts = np.array([c["count"] for c in chunks])
    sums = np.array([c["thickness_spectrum_sum"] for c in chunks])[:, :, keep]
    yields = np.trapezoid(sums, energies[keep], axis=-1) / counts[:, None]
    return ArmBatches(thickness, tuple(c["start"] for c in chunks), counts, yields[:, :, None])


def fate_records(merged: dict) -> list[AxialHandoffRecord]:
    return [AxialHandoffRecord(int(i), fate, float(d)) for i, fate, d in merged["records"]]


def doubling_ratio(arm: ArmBatches, thickness: float) -> float:
    """SE from pairs of adjacent chunks over SE from single chunks (about 1 if normal)."""
    values, counts = arm.at(thickness)[:, 0], arm.counts
    even = values.size - values.size % 2
    paired = (values[:even:2] * counts[:even:2] + values[1:even:2] * counts[1:even:2]) / (
        counts[:even:2] + counts[1:even:2]
    )
    _, se1 = batch_mean_se(values[:even], counts[:even])
    _, se2 = batch_mean_se(paired, counts[:even:2] + counts[1:even:2])
    return float(se2 / se1)


def main(spec_path: str, out_path: str) -> None:
    base = Path(spec_path).parent
    spec = json.loads(Path(spec_path).read_text())
    merged = {k: json.loads((base / v).read_text()) for k, v in spec["arms"].items()}
    fates = {k: json.loads((base / v).read_text()) for k, v in spec["fates"].items()}
    depth = float(merged["control"]["settings"]["thickness_ang"])
    if fates["control"]["settings"]["thickness_ang"] != depth:
        raise ValueError("fate sample depth differs from the control slab")
    out = {"spec": spec, "settings": {k: m["settings"] for k, m in merged.items()}, "bands": {}}
    for name, band in spec["bands_eV"].items():
        arms = {k: arm_batches(m, tuple(band)) for k, m in merged.items()}
        evidence, report = calibrate_axial_depth(
            arms,
            fate_records(fates["control"]),
            fate_records(fates["all_on"]),
            checkpoints_ang=[float(c) for c in spec["checkpoints_ang"]],
            span_ang=float(spec.get("span_ang", 15.0)),
            sampled_max_depth_ang=depth,
            case_key=spec["case_key"],
            observable_key=f"on-axis/{name}/{band[0]:g}-{band[1]:g}eV",
            model_id=spec["model_id"],
            evidence_id=f"{spec['evidence_id']}/{name}",
        )
        report["observable_key"] = evidence.observable_key
        report["doubling_ratio"] = {
            f"{c:g}": doubling_ratio(arms["control"], float(c)) for c in spec["checkpoints_ang"]
        }
        out["bands"][name] = report
        for row in report["intervals"]:
            print(
                f"{name} ({row['start_ang']:g}, {row['stop_ang']:g}] A: "
                f"eps = {row['relative_error_upper']:.4f}  "
                f"R={row['return_upper']:.3g} U={row['unresolved_upper']:.3g} "
                f"G={row['grid_supercell_upper']:.3g} S={row['source_model_upper']:.3g} "
                f"L={row['reference_lower']:.3g}"
            )
        print(f"{name}: joint confidence {report['simultaneous_confidence']:.3f}")
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
