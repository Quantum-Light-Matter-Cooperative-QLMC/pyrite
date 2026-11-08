"""Si axial on-axis photon source of the fibred absorbing model (#992). Research only.

Builds Si one-cell fibre channels (continuum TDS strings + electronic kicks)
with ``checks/axial_outgoing_maps.py`` and runs
:func:`pyrite.montecarlo.channeling.axial_fiber_regression.fiber_on_axis_source`
in incident chunks that merge exactly (sums of per-primary estimates).

Usage::

    python checks/axial_fiber_source.py chunk '<json kwargs>' START COUNT OUT.json
    python checks/axial_fiber_source.py merge OUT.json PART.json [PART.json ...]

Kwargs: axis, energy_eV, thickness_ang, step_ang, max_order, outgoing_order_max,
target_eV, electronic, boundary ("max", "none" or a float in eV), photon grid
(e_min_eV, e_max_eV, n_energy), absorption (bool), seed, max_working_bytes
(None keeps the library budget), tilt_deg and tilt_azimuth_deg (incidence angle
to the axis and its azimuth from the profile's first reciprocal axis; the
entrance plane wave sits at order n with carrier k_perp - n g folded into
[-g/2, g/2)). Photons are scored along the crystal axis; at these tilts
(< 0.3 deg) the beam-direction offset is far inside 1/gamma.

Validation: cr-axial-fiber-regression
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

_MAPS = runpy.run_path(str(Path(__file__).with_name("axial_outgoing_maps.py")))

DEFAULTS = {
    "axis": [1, 1, 1],
    "energy_eV": 2.99e6,
    "thickness_ang": 11000.0,
    "step_ang": 15.0,
    "max_order": 9,
    "outgoing_order_max": 20,
    "target_eV": None,
    "electronic": True,
    "boundary": "max",
    "e_min_eV": 800.0,
    "e_max_eV": 8000.0,
    "n_energy": 145,
    "absorption": True,
    "seed": 0,
    "max_working_bytes": None,
    "tilt_deg": 0.0,
    "tilt_azimuth_deg": 0.0,
}


def _extents(value):
    return value if isinstance(value, int) else tuple(value)


def build(options: dict):
    """Channels, entry, hbar_v, info and the resolved run settings."""
    o = {**DEFAULTS, **options}
    from pyrite.materials.channeling.axial import axial_profile

    extrema = axial_profile("silicon", tuple(o["axis"])).extrema()
    target = o["target_eV"]
    if target is None:  # provisional numerical cut rule of the escape note
        target = extrema[1] + (extrema[1] - extrema[0])
    channels, entry, hbar_v, info = _MAPS["material_fiber_channels"](
        _extents(o["max_order"]),
        float(target),
        electronic=bool(o["electronic"]),
        outgoing_order_max=_extents(o["outgoing_order_max"]),
        tds="strings",
        axis=tuple(o["axis"]),
        energy_eV=float(o["energy_eV"]),
    )
    carrier, entry = _tilted_entry(o, info, channels, entry)
    boundary = o["boundary"]
    if boundary == "max":
        boundary = info["potential_extrema_eV"][1]
    elif boundary == "none":
        boundary = None
    o.update(target_eV=float(target), boundary_eV=boundary, carrier_inv_ang=list(carrier))
    return channels, entry, hbar_v, info, o


def _tilted_entry(o: dict, info: dict, channels, entry):
    """Carrier and entrance order vector of a plane wave tilted by ``tilt_deg``."""
    theta = np.deg2rad(float(o["tilt_deg"]))
    if theta == 0.0:
        return (0.0, 0.0), entry
    from pyrite.materials.channeling.axial import axial_profile, axial_states
    from pyrite.montecarlo.channeling import longitudinal_kinematics

    phi = np.deg2rad(float(o["tilt_azimuth_deg"]))
    k_perp = info["k_inv_ang"] * np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
    g = np.asarray(channels.reciprocal_inv_ang, dtype=float)
    order = np.floor(k_perp / g + 0.5).astype(int)
    carrier = k_perp - order * g
    profile = axial_profile("silicon", tuple(o["axis"]))
    gamma, _ = longitudinal_kinematics(float(o["energy_eV"]))
    window = _extents(o["max_order"])
    orders = axial_states(profile, gamma, (0.0, 0.0), window).orders
    tilted = np.all(orders == order, axis=1).astype(complex)
    if tilted.sum() != 1:
        raise ValueError("tilted entrance order lies outside the order window")
    return (float(carrier[0]), float(carrier[1])), tilted


def chunk(options: dict, start: int, count: int) -> dict:
    """One merge-ready chunk: sums of per-primary spectra and retained weights."""
    from pyrite.materials.attenuation import linear_attenuation_inv_mm
    from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_on_axis_source

    channels, entry, hbar_v, info, o = build(options)
    depths = np.linspace(
        0.0, o["thickness_ang"], int(round(o["thickness_ang"] / o["step_ang"])) + 1
    )
    energies = np.linspace(o["e_min_eV"], o["e_max_eV"], int(o["n_energy"]))
    extra = {}
    if o["max_working_bytes"] is not None:
        extra["max_working_bytes"] = int(o["max_working_bytes"])
    if o["absorption"]:  # normal incidence along the axis: path T - z
        extra.update(
            absorption_inv_ang=linear_attenuation_inv_mm("silicon", energies) * 1e-7,
            exit_path_ang=depths[-1] - depths,
        )
    clock = time.perf_counter()
    source = fiber_on_axis_source(
        channels,
        entry,
        kinetic_energy_eV=float(o["energy_eV"]),
        hbar_v_eV_ang=hbar_v,
        depths_ang=depths,
        photon_energy_eV=energies,
        incident_ids=range(start, start + count),
        seed=int(o["seed"]),
        boundary_eV=o["boundary_eV"],
        carrier_inv_ang=tuple(o["carrier_inv_ang"]),
        search_step_ang=200.0,
        **extra,
    )
    seconds = time.perf_counter() - clock
    n = len(source.events)
    sum_sq = source.sem**2 * n * (n - 1) + n * source.spectrum**2
    kinds = {}
    for event in source.events:
        kinds[event.kind] = kinds.get(event.kind, 0.0) + event.weight
    return {
        "status": "research-only; fibred absorbing model, on-axis, retained subspace only",
        "settings": {k: v for k, v in o.items()},
        "info": {k: v for k, v in info.items() if k != "events"},
        "start": start,
        "count": n,
        "seconds": seconds,
        "photon_energy_eV": energies.tolist(),
        "depths_ang": [float(depths[0]), float(depths[-1]), int(depths.size)],
        "spectrum_sum": (source.spectrum * n).tolist(),
        "spectrum_sum_sq": sum_sq.tolist(),
        "trace_sum": (source.correlation.trace * n).tolist(),
        "terminal_weight_sum": kinds,
        "kick_jumps_sum": int(sum(e.kick_jumps for e in source.events)),
        "tds_jumps_sum": int(sum(e.lattice_jumps for e in source.events)),
    }


def merge(parts: list[dict]) -> dict:
    """Combine chunks exactly; mean, SEM and peak summary per eV of photon energy."""
    first = parts[0]
    for part in parts[1:]:
        if part["settings"] != first["settings"]:
            raise ValueError("chunks were run with different settings")
    starts = sorted((p["start"], p["count"]) for p in parts)
    for (a, n), (b, _) in zip(starts, starts[1:], strict=False):
        if a + n != b:
            raise ValueError("chunks overlap or leave a gap")
    n = sum(p["count"] for p in parts)
    total = np.sum([p["spectrum_sum"] for p in parts], axis=0)
    squares = np.sum([p["spectrum_sum_sq"] for p in parts], axis=0)
    mean = total / n
    sem = np.sqrt(np.maximum(squares / n - mean**2, 0.0) / (n - 1))
    energies = np.asarray(first["photon_energy_eV"])
    kinds: dict[str, float] = {}
    for p in parts:
        for kind, weight in p["terminal_weight_sum"].items():
            kinds[kind] = kinds.get(kind, 0.0) + weight
    return {
        "status": first["status"],
        "settings": first["settings"],
        "info": first["info"],
        "incidents": n,
        "cpu_seconds": float(sum(p["seconds"] for p in parts)),
        "photon_energy_eV": energies.tolist(),
        "spectrum_per_eV_sr": mean.tolist(),
        "sem_per_eV_sr": sem.tolist(),
        "integrated_per_sr": float(np.trapezoid(mean, energies)),
        "integrated_sem_per_sr": _batch_sem(parts, energies),
        "retained_weight": (np.sum([p["trace_sum"] for p in parts], axis=0) / n).tolist()[-1],
        "terminal_fraction": {k: v / n for k, v in kinds.items()},
        "mean_kick_jumps": sum(p["kick_jumps_sum"] for p in parts) / n,
        "mean_tds_jumps": sum(p["tds_jumps_sum"] for p in parts) / n,
        "peaks": peaks(energies, mean, sem),
    }


def _batch_sem(parts: list[dict], energies) -> float | None:
    """Count-weighted batch-means SEM of the integrated yield across chunks."""
    if len(parts) < 2:
        return None
    counts = np.array([p["count"] for p in parts], dtype=float)
    values = np.array([np.trapezoid(p["spectrum_sum"], energies) for p in parts]) / counts
    mean = np.average(values, weights=counts)
    variance = np.sum(counts * (values - mean) ** 2) / (len(parts) - 1) / counts.sum()
    return float(np.sqrt(variance))


def peaks(energies, mean, sem, min_z: float = 4.0) -> list[dict]:
    """Local maxima significant at ``min_z`` with FWHM by linear interpolation."""
    found = []
    for i in range(1, len(mean) - 1):
        if not (mean[i] >= mean[i - 1] and mean[i] > mean[i + 1] and mean[i] > min_z * sem[i]):
            continue
        half = 0.5 * mean[i]
        lo = i
        while lo > 0 and mean[lo] > half:
            lo -= 1
        hi = i
        while hi < len(mean) - 1 and mean[hi] > half:
            hi += 1
        left = np.interp(half, [mean[lo], mean[lo + 1]], [energies[lo], energies[lo + 1]])
        right = np.interp(half, [mean[hi], mean[hi - 1]], [energies[hi], energies[hi - 1]])
        found.append(
            {
                "energy_eV": float(energies[i]),
                "height_per_eV_sr": float(mean[i]),
                "z": float(mean[i] / sem[i]) if sem[i] > 0 else None,
                "fwhm_eV": float(right - left) if mean[lo] <= half and mean[hi] <= half else None,
            }
        )
    return found


WATSON = Path(__file__).parents[1] / "tests/data/channeling/watson-koehler-1982-si-axial.json"


def compare(merged: dict, *, resolution_fwhm_eV: float = 200.0, draws: int = 400) -> dict:
    """Peaks of the detector-smeared source vs Watson & Koehler (1982) Table II.

    The mean spectrum is convolved with a Gaussian of the Si(Li) resolution
    (200 eV FWHM). Each measured forward-equivalent peak is matched to the
    largest smeared maximum within +-25% of it; position and FWHM spreads come
    from ``draws`` resamplings of the per-bin SEM (bins treated as independent).
    """
    o = merged["settings"]
    axis = "".join(str(int(v)) for v in o["axis"])
    rows = json.loads(WATSON.read_text())["axes"].get(axis, {}).get("rows", [])
    row = min(rows, key=lambda r: abs(r["kinetic_energy_MeV"] * 1e6 - o["energy_eV"]), default=None)
    if row is None or abs(row["kinetic_energy_MeV"] * 1e6 - o["energy_eV"]) > 1e4:
        return {"measured": None}
    e = np.asarray(merged["photon_energy_eV"])
    mean = np.asarray(merged["spectrum_per_eV_sr"])
    sem = np.asarray(merged["sem_per_eV_sr"])
    sigma = resolution_fwhm_eV / (2 * np.sqrt(2 * np.log(2)))
    kernel = np.exp(-0.5 * ((e[:, None] - e[None, :]) / sigma) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    samples = [kernel @ mean] + [
        kernel @ (mean + sem * rng.standard_normal(e.size)) for _ in range(draws)
    ]
    out = []
    for index, measured in enumerate(row["peaks_keV"]):
        if measured is None:
            continue
        window = (e > 0.75 * measured * 1e3) & (e < 1.25 * measured * 1e3)
        found = []
        for curve in samples:
            c = np.where(window, curve, -np.inf)
            i = int(np.argmax(c))
            interior = 0 < i < e.size - 1 and window[i - 1] and window[i + 1]
            if not interior or not (curve[i] > curve[i - 1] and curve[i] > curve[i + 1]):
                found.append((np.nan, np.nan))
                continue
            half = 0.5 * curve[i]
            lo, hi = i, i
            while lo > 0 and curve[lo] > half:
                lo -= 1
            while hi < e.size - 1 and curve[hi] > half:
                hi += 1
            left = np.interp(half, [curve[lo], curve[lo + 1]], [e[lo], e[lo + 1]])
            right = np.interp(half, [curve[hi], curve[hi - 1]], [e[hi], e[hi - 1]])
            found.append((e[i], right - left))
        found = np.asarray(found)
        model = found[0]
        spread = np.nanstd(found[1:], axis=0)
        out.append(
            {
                "peak": index,
                "measured_keV": measured,
                "model_keV": None if np.isnan(model[0]) else float(model[0] / 1e3),
                "model_sd_keV": float(spread[0] / 1e3),
                "relative": None if np.isnan(model[0]) else float(model[0] / 1e3 / measured - 1),
                "smeared_fwhm_keV": None if np.isnan(model[1]) else float(model[1] / 1e3),
                "found_fraction": float(np.mean(~np.isnan(found[1:, 0]))),
            }
        )
    return {
        "measured": {"source": "Watson & Koehler, PRB 25, 3079 (1982), Table II", **row},
        "resolution_fwhm_eV": resolution_fwhm_eV,
        "peaks": out,
    }


def main(argv: list[str]) -> None:
    if argv[0] == "chunk":
        options, start, count, out = json.loads(argv[1]), int(argv[2]), int(argv[3]), argv[4]
        Path(out).write_text(json.dumps(chunk(options, start, count)))
    elif argv[0] == "merge":
        parts = [json.loads(Path(p).read_text()) for p in argv[2:]]
        merged = merge(parts)
        merged["comparison"] = compare(merged)
        Path(argv[1]).write_text(json.dumps(merged, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
