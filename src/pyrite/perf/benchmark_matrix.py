"""Engine-neutral parameter matrix for opt-in cross-engine benchmark studies.

The matrix freezes absolute scene inputs before any engine runs. Relative
thicknesses ``t/R`` are converted once, from one pinned independent CSDA-range
tabulation, so every engine receives the same surface-normal thickness instead
of its own computed range. Without a pinned table the thickness stays an
explicit pending entry; nothing is estimated from an engine's stopping model.
"""

import csv
import hashlib
import io
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

RANGE_COLUMNS = ("kinetic_energy_MeV", "csda_range_g_cm2")
TERMINAL_FRACTIONS = ("transmitted", "backscattered", "side_exited", "stopped")
SOURCE_BANDS_KEV = (("photons_10_50kev", 10.0, 50.0), ("photons_50_100kev", 50.0, 100.0))
_FWHM_PER_RMS = 2.0 * math.sqrt(2.0 * math.log(2.0))
_ELECTRON_REST_KEV = 510.99895


def parse_range_table(text: str) -> dict[str, Any]:
    """Parse a two-column CSDA-range CSV and return arrays plus its SHA-256.

    The header must be exactly ``kinetic_energy_MeV,csda_range_g_cm2``. Both
    columns must be finite, positive and strictly increasing; the digest covers
    the exact UTF-8 bytes so a re-exported table cannot silently replace a pin.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(cell.strip() for cell in rows[0]) != RANGE_COLUMNS:
        raise ValueError(f"range table header must be {','.join(RANGE_COLUMNS)}")
    values = np.asarray([[float(cell) for cell in row] for row in rows[1:] if row], dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError("range table needs at least two two-column rows")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("range table entries must be finite and positive")
    if np.any(np.diff(values, axis=0) <= 0.0):
        raise ValueError("range table energies and ranges must increase strictly")
    return {
        "energy_MeV": values[:, 0],
        "range_g_cm2": values[:, 1],
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def csda_range_g_cm2(table: Mapping[str, Any], energy_keV: float) -> float:
    """Log-log interpolate a pinned CSDA range; extrapolation is rejected."""
    energy = np.asarray(table["energy_MeV"], dtype=float)
    ranges = np.asarray(table["range_g_cm2"], dtype=float)
    e_mev = float(energy_keV) / 1000.0
    if not energy[0] <= e_mev <= energy[-1]:
        raise ValueError(f"{energy_keV} keV lies outside the pinned range table")
    return float(np.exp(np.interp(np.log(e_mev), np.log(energy), np.log(ranges))))


def case_id(scene: Mapping[str, Any]) -> str:
    """Return the stable id built only from frozen physical inputs and variant."""
    parts = [
        str(scene["material"]),
        f"{_number(scene['energy_keV'])}keV",
        (
            f"tr{_number(scene['t_over_r'])}"
            if scene.get("t_over_r") is not None
            else f"{_number(scene['thickness_um'])}um"
        ),
        f"{_number(scene.get('angle_deg', 0))}deg",
    ]
    if scene.get("variant", "core") != "core":
        parts.append(str(scene["variant"]))
    return "-".join(parts).lower()


def gaussian_fwhm_mm(rms_mm: float) -> float:
    """Convert a declared RMS spot to the FWHM PyRITE's spot option takes."""
    return _FWHM_PER_RMS * float(rms_mm)


def waist_normalized_emittance_mm_mrad(
    sigma_x_mm: float, sigma_xp_mrad: float, energy_keV: float
) -> tuple[float, float]:
    """Return ``(eps_n [mm mrad], beta [m])`` for an uncorrelated round waist.

    At a waist ``eps = sigma_x * sigma_x'`` and ``beta = sigma_x / sigma_x'``;
    the normalized emittance multiplies by ``beta_rel * gamma`` at the beam energy.
    """
    if sigma_x_mm <= 0.0 or sigma_xp_mrad <= 0.0 or energy_keV <= 0.0:
        raise ValueError("waist moments and energy must be positive")
    gamma = 1.0 + float(energy_keV) / _ELECTRON_REST_KEV
    beta_gamma = math.sqrt(gamma * gamma - 1.0)
    return sigma_x_mm * sigma_xp_mrad * beta_gamma, sigma_x_mm / sigma_xp_mrad


def observable_applicability(scene: Mapping[str, Any]) -> dict[str, str]:
    """Classify each declared observable as headline, diagnostic or excluded.

    Bands at or above the beam endpoint are excluded, and transmitted-energy
    moments are diagnostics in the stopping-dominated regime ``t/R >= 1``
    where transmission is expected to be unresolved.
    """
    endpoint = float(scene["energy_keV"])
    out = {name: "headline" for name in TERMINAL_FRACTIONS}
    out["side_exited"] = "headline" if scene.get("width_mm", 10.0) < 10.0 else "diagnostic"
    stopping = scene.get("t_over_r") is not None and float(scene["t_over_r"]) >= 1.0
    out["transmitted_energy_keV_per_primary"] = "diagnostic" if stopping else "headline"
    out["photons_ge10kev"] = "headline" if endpoint > 10.0 else "excluded"
    for name, low, _high in SOURCE_BANDS_KEV:
        out[name] = "headline" if endpoint > low else "excluded"
    out["photons_ge100kev"] = "headline" if endpoint > 100.0 else "excluded"
    return out


def expand_matrix(
    spec: Mapping[str, Any], ranges: Mapping[str, Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Expand the frozen core product and declared sparse extensions to cases.

    ``core.t_over_r_cap_by_energy_keV`` lowers every core ``t/R`` above the cap
    at that energy to the cap, so they merge into one case.

    ``ranges`` maps a material key to a :func:`parse_range_table` result. A
    material without one keeps ``thickness_um = None`` and status
    ``pending-range-source`` rather than a guessed dimension.
    """
    materials = {m["key"]: m for m in spec["materials"]}
    core = spec["core"]
    caps = {float(e): cap for e, cap in core.get("t_over_r_cap_by_energy_keV", {}).items()}
    scenes: list[dict[str, Any]] = [
        {"material": key, "energy_keV": e, "t_over_r": tr, "variant": "core"}
        for key in core["materials"]
        for e in core["energies_keV"]
        for tr in sorted({min(t, caps.get(float(e), t)) for t in core["t_over_r"]})
    ]
    scenes.extend(dict(extension) for extension in spec.get("extensions", []))
    cases = []
    seen: set[str] = set()
    for scene in scenes:
        material = materials[scene["material"]]
        resolved = {
            "angle_deg": 0,
            "width_mm": spec["defaults"]["width_mm"],
            "beam": dict(spec["defaults"]["beam"]),
            "detector": None,
            "physics": dict(spec["defaults"]["physics"]),
            **scene,
            "density_g_cm3": material["density_g_cm3"],
        }
        for key in ("beam", "physics"):
            resolved[key] = {**spec["defaults"][key], **scene.get(key, {})}
        resolved["id"] = case_id(resolved)
        if resolved["id"] in seen:
            raise ValueError(f"duplicate matrix case id {resolved['id']}")
        seen.add(resolved["id"])
        table = (ranges or {}).get(scene["material"])
        resolved.update(_thickness(resolved, table))
        resolved["observables"] = observable_applicability(resolved)
        cases.append(resolved)
    return cases


def coverage_table(cases: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Group case ids by each varied parameter so gaps stay visible."""
    table: dict[str, dict[str, list[str]]] = {}
    for case in cases:
        for axis, value in [
            ("material", case["material"]),
            ("energy_keV", _number(case["energy_keV"])),
            (
                "t_over_r",
                "absolute" if case.get("t_over_r") is None else _number(case["t_over_r"]),
            ),
            ("angle_deg", _number(case["angle_deg"])),
            ("variant", case.get("variant", "core")),
        ]:
            table.setdefault(axis, {}).setdefault(str(value), []).append(case["id"])
    return table


def _thickness(scene: Mapping[str, Any], table: Mapping[str, Any] | None) -> dict[str, Any]:
    if scene.get("t_over_r") is None:
        # Historical anchors keep their absolute dimensions; no t/R is claimed.
        return {"range_source": None, "status": "frozen-absolute"}
    if table is None:
        return {"thickness_um": None, "range_source": None, "status": "pending-range-source"}
    r_g_cm2 = csda_range_g_cm2(table, scene["energy_keV"])
    # Surface-normal thickness; oblique paths are reported, never rescaled.
    thickness_um = float(scene["t_over_r"]) * r_g_cm2 / float(scene["density_g_cm3"]) * 1.0e4
    return {
        "thickness_um": thickness_um,
        "csda_range_g_cm2": r_g_cm2,
        "range_source": table["sha256"],
        "status": "frozen",
    }


def _number(value: Any) -> str:
    return f"{float(value):g}"
