"""Read-time radiation component diagnostics; no response or population scaling."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from .._spectral_components import cxr_spectrum


def two_beam_record(record: Mapping[str, Any]) -> bool:
    """Whether the line slot is a joint-field correction rather than bare CXR."""
    mode = record.get("case", {}).get("transition_radiation", {})
    return (
        mode == "two-beam"
        if isinstance(mode, str)
        else bool(mode and mode.get("model") == "two-beam")
    )


def radiation_band_metrics(
    record: Mapping[str, Any], band_eV: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Integrate stored densities in photons/(sr incident electron).

    Interpolate band endpoints; never extrapolate outside the stored line grid.
    Signed estimates remain signed. No current, charge, acceptance or detector
    factor is applied. The two-beam line slot is a signed joint correction;
    removing its TR baseline does not construct a DTR-off spectrum.
    """
    energy = np.asarray(record["E_grid"], dtype=float)
    if energy.size < 2 or not np.all(np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
        raise ValueError("radiation diagnostics require a finite increasing grid with >=2 nodes")
    lo, hi = (energy[0], energy[-1]) if band_eV is None else band_eV
    if not np.isfinite([lo, hi]).all() or not energy[0] <= lo < hi <= energy[-1]:
        raise ValueError("band must lie inside the stored line grid with lower < upper")
    nodes = np.r_[lo, energy[(energy > lo) & (energy < hi)], hi]
    parts = {"line": record["spec"]}
    for name in ("transition", "characteristic", "channeling"):
        if record.get(f"spec_{name}") is not None:
            parts[name] = record[f"spec_{name}"]
    parts["combined"] = cxr_spectrum(record)
    yields = {
        name: float(np.trapezoid(np.interp(nodes, energy, v), nodes)) for name, v in parts.items()
    }
    joint = two_beam_record(record)
    case = record.get("case", {})
    charge = case.get("bunch_charge_pc")
    return {
        "band_eV": (float(lo), float(hi)),
        "basis": "TR baseline + signed joint-field correction"
        if joint
        else "PXR/CBS + stored additive components",
        "units": "photons/(sr incident electron)",
        "yields": yields,
        "transition_fraction": yields.get("transition", 0) / yields["combined"]
        if yields["combined"]
        else None,
        "negative_transition_bins": int(np.sum(np.asarray(parts.get("transition", [])) < 0)),
        "physical_electrons": None if charge is None else float(charge) * 1e-12 / elementary_charge,
        "sampled_histories": case.get("Ne_brem"),
    }
