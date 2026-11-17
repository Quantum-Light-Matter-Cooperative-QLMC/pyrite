"""Attribute selected #1254 CXR outliers on the original #1252 capture checkout.

Copy this diagnostic into that checkout's checks/ directory and run with its
original catalog overlay, CPU environment and workspace/table selection:
``uv run python checks/adaptive_tail_attribution_1254.py``. It uses the
existing #1252 profile resolver and counter-addressed primary offsets. Eight
single-primary runs, not a sweep. Requires out-cpu-1000/ captures; writes
attribution-1254.json. Exact captured-score agreement is mandatory.

Validation: adaptive-sample-size-stopping
"""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.adaptive import _case_geometry, _cxr_band
from pyrite.montecarlo.runner.emission_adapters import EmissionBatch, select_emission
from pyrite.montecarlo.spectrum.cxr._attribution import merge_cxr_attribution

h = runpy.run_path("checks/adaptive_block_size_1252.py")
results = []
for material, ids in [("hopg", [0, 40211, 53961, 893438]), ("hbn", [0, 150453, 261777, 270761])]:
    case, p, *_ = h["profile_case"](material, 1000, 15, 48000)
    c = {**dict(case), "Ne": 1, "Ne_brem": 1}
    c.pop("adaptive_precision", None)
    with np.load(f"out-cpu-1000/{material}-1000-15-b48000.values.npz") as data:
        values = data["cxr"]
    emission = select_emission(c)
    grid, _, _, nh, groove = _case_geometry(c)
    lo, hi = p.band_eV or _cxr_band(c, grid)
    for primary in ids:
        tp = runner._transport_case(
            c, transport_core="per-electron", engine_options={"_electron_start": primary}
        )
        segs = tp["segs"]
        audit = {"start_eV": lo, "stop_eV": hi, "n_electrons": 1, "attribute": {"top": 5}}
        emission.cxr.evaluate(
            EmissionBatch(
                segs,
                np.array([lo, hi]),
                c,
                nh,
                c.get("abs_layers"),
                groove,
                electron_limit=1,
                truncation_audit=audit,
            ),
            "spec",
        )
        attr = merge_cxr_attribution(audit["attribution"], 5)
        top = attr["lines_by_mass"]
        row = int(top["segment"][0])
        got = float(audit["electron_mass"][0])
        ref = float(values[primary])
        assert got == ref, (material, primary, got, ref)
        result = {
            "material": material,
            "primary_id_zero_based": primary,
            "capture_value": ref,
            "replayed_value": got,
            "relative_difference": got / ref - 1,
            "channeling": c.get("channeling"),
            "rows": len(segs["E_keV"]),
            "segments": {
                k: np.asarray(segs[k]).tolist()
                for k in [
                    "r_mid",
                    "v_hat",
                    "L_ang",
                    "E_start_keV",
                    "E_end_keV",
                    "event_kind",
                    "hard_channel",
                ]
            },
            "dominant_segment": row,
            "dominant_E_start_keV": float(segs["E_start_keV"][row]),
            "dominant_E_end_keV": float(segs["E_end_keV"][row]),
            "dominant_midpoint_ang": np.asarray(segs["r_mid"][row]).tolist(),
            "dominant_direction": np.asarray(segs["v_hat"][row]).tolist(),
            "dominant_event_kind": int(segs["event_kind"][row]),
            "dominant_hard_channel": int(segs["hard_channel"][row]),
            "top_lines": {k: np.asarray(v).tolist() for k, v in top.items()},
        }
        results.append(result)
        print(json.dumps(result), flush=True)
Path("attribution-1254.json").write_text(
    json.dumps(
        {
            "code_sha": Path("CODE_SHA").read_text().strip(),
            "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Exact primary replay on original capture checkout, same catalog overlay and workspace; simulation source unmodified. Diagnostic file added separately.",
            "cases": results,
        },
        indent=2,
    )
    + "\n"
)
