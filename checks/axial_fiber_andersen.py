"""Compare merged fibre-source runs with Andersen et al. (1982) Si data (#992). Research only.

Andersen, Bonderup, Laegsgaard, Marsh & Sorensen, Nucl. Instrum. Methods 194,
209 (1982); data in ``tests/data/channeling/andersen-1982-si-axial.json``.

* Fig. 8 line energies: peaks of the model spectrum smeared with the 300 eV
  detector resolution, converted to the rest frame with E_R = E / (2 gamma)
  (the paper's Eq. 6 convention).
* Table 2 widths and Fig. 11 line areas: the unsmeared model spectrum is fitted
  around 2p-1s with the finite-thickness Lorentzian of Eq. (35),
  ``|1 - exp(-(l / D hbar c)(Gamma/2 + i(E - E0)))|^2 / ((E - E0)^2 + Gamma^2/4)``
  (D = 1/(1 - beta)), area-normalized, plus a linear background. Gamma is the
  lifetime FWHM in the paper's notation; the area is photons/sr/electron.

Usage::

    python checks/axial_fiber_andersen.py MERGED.json [MERGED.json ...]

Validation: cr-axial-fiber-regression
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parents[1] / "tests/data/channeling/andersen-1982-si-axial.json"
HBARC_EV_ANG = 1973.269804
M_E_EV = 510998.95


def _kinematics(kinetic_eV: float) -> tuple[float, float]:
    gamma = 1.0 + kinetic_eV / M_E_EV
    return gamma, float(np.sqrt(1.0 - 1.0 / gamma**2))


def _axis(settings: dict) -> str:
    uvw = sorted(abs(int(v)) for v in settings["axis"])
    return {(0, 0, 1): "100", (0, 1, 1): "110", (1, 1, 1): "111"}[tuple(uvw)]


def _smear(e, y, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    kernel = np.exp(-0.5 * ((e[:, None] - e[None, :]) / sigma) ** 2)
    return (kernel / kernel.sum(axis=1, keepdims=True)) @ y


def _peak_in(e, y, lo, hi):
    inside = (e > lo) & (e < hi)
    c = np.where(inside, y, -np.inf)
    i = int(np.argmax(c))
    if not (0 < i < e.size - 1 and inside[i - 1] and inside[i + 1]):
        return None
    if not (y[i] > y[i - 1] and y[i] > y[i + 1]):
        return None
    a, b, d = y[i - 1], y[i], y[i + 1]  # parabolic refinement on the uniform grid
    step = e[1] - e[0]
    return float(e[i] + 0.5 * step * (a - d) / (a - 2 * b + d))


def line_energies(merged: dict, data: dict, fwhm_eV: float = 300.0, draws: int = 200) -> list:
    """Model rest-frame peak energies at each matching Fig. 8 point."""
    o = merged["settings"]
    axis = _axis(o)
    gamma, _ = _kinematics(o["energy_eV"])
    d = data["line_energies_fig8"]["d_angstrom"][axis]
    e = np.asarray(merged["photon_energy_eV"])
    mean = np.asarray(merged["spectrum_per_eV_sr"])
    sem = np.asarray(merged["sem_per_eV_sr"])
    rng = np.random.default_rng(0)
    curves = [_smear(e, mean, fwhm_eV)] + [
        _smear(e, mean + sem * rng.standard_normal(e.size), fwhm_eV) for _ in range(draws)
    ]
    out = []
    for point in data["line_energies_fig8"]["points"]:
        if point["axis"] != axis or abs(point["gamma_over_d"] * d - gamma) > 0.1:
            continue
        # measured lab energy at the model's gamma (E_R scales with gamma/d)
        lab = 2 * gamma * point["rest_frame_eV"]
        found = [_peak_in(e, c, 0.85 * lab, 1.15 * lab) for c in curves]
        spread = [f for f in found[1:] if f is not None]
        model = found[0]
        out.append(
            {
                **point,
                "model_gamma_over_d": gamma / d,
                "model_rest_frame_eV": None if model is None else model / (2 * gamma),
                "model_sd_eV": float(np.std(spread) / (2 * gamma)) if len(spread) > 1 else None,
                "relative": None if model is None else model / lab - 1.0,
                "found_fraction": len(spread) / draws,
            }
        )
    return out


def line_shape(e, e0: float, width: float, length_ang: float, doppler: float):
    """Area-normalized Eq. (35) profile on ``e`` (eV); ``width`` = Gamma (eV)."""
    scale = length_ang / (doppler * HBARC_EV_ANG)
    grid = np.linspace(e0 - 60 * width, e0 + 60 * width, 24001)

    def raw(x):
        z = scale * (0.5 * width + 1j * (x - e0))
        return np.abs(1.0 - np.exp(-z)) ** 2 / ((x - e0) ** 2 + 0.25 * width**2)

    return raw(np.asarray(e, dtype=float)) / np.trapezoid(raw(grid), grid)


def fit_line(merged: dict, guess_eV: float, half_window: float = 0.22) -> dict:
    """Eq. (35) line + linear background fitted to the unsmeared model spectrum."""
    from scipy.optimize import curve_fit

    o = merged["settings"]
    gamma, beta = _kinematics(o["energy_eV"])
    doppler = 1.0 / (1.0 - beta)
    e = np.asarray(merged["photon_energy_eV"])
    y = np.asarray(merged["spectrum_per_eV_sr"])
    s = np.asarray(merged["sem_per_eV_sr"])
    keep = (e > guess_eV * (1 - half_window)) & (e < guess_eV * (1 + half_window))
    e, y, s = e[keep], y[keep], np.maximum(s[keep], 1e-3 * np.max(np.abs(y[keep])))
    length = float(o["thickness_ang"])

    def model(x, area, e0, width, b0, b1):
        return area * line_shape(x, e0, abs(width), length, doppler) + b0 + b1 * (x - guess_eV)

    peak = float(np.max(y) - np.min(y))
    p0 = [peak * 600.0, guess_eV, 600.0, float(np.min(y)), 0.0]
    popt, pcov = curve_fit(model, e, y, p0=p0, sigma=s, absolute_sigma=True, maxfev=20000)
    err = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((model(e, *popt) - y) / s) ** 2))
    return {
        "guess_eV": guess_eV,
        "window_eV": [float(e[0]), float(e[-1])],
        "e0_eV": float(popt[1]),
        "e0_err_eV": float(err[1]),
        "gamma_eV": float(abs(popt[2])),
        "gamma_err_eV": float(err[2]),
        "area_per_sr": float(popt[0]),
        "area_err_per_sr": float(err[0]),
        "thickness_width_eV": float(doppler * 2 * np.pi * HBARC_EV_ANG / length),
        "chi2_per_dof": chi2 / max(e.size - 5, 1),
    }


def _two_p_one_s_guess(merged: dict, data: dict) -> float:
    """Lab 2p-1s energy at the run's gamma from a linear fit to Fig. 8 (same axis)."""
    o = merged["settings"]
    axis = _axis(o)
    gamma, _ = _kinematics(o["energy_eV"])
    d = data["line_energies_fig8"]["d_angstrom"]
    pts = [p for p in data["line_energies_fig8"]["points"] if p["transition"] == "2p-1s"]
    x = np.array([p["gamma_over_d"] for p in pts])
    yv = np.array([p["rest_frame_eV"] for p in pts])
    slope, offset = np.polyfit(x, yv, 1)
    rest = slope * gamma / d[axis] + offset
    e = np.asarray(merged["photon_energy_eV"])
    smeared = _smear(e, np.asarray(merged["spectrum_per_eV_sr"]), 300.0)
    found = _peak_in(e, smeared, 0.85 * 2 * gamma * rest, 1.15 * 2 * gamma * rest)
    return found if found is not None else 2 * gamma * rest


def compare(merged: dict, data: dict) -> dict:
    o = merged["settings"]
    axis = _axis(o)
    out = {
        "settings": {
            k: o.get(k)
            for k in (
                "axis",
                "energy_eV",
                "thickness_ang",
                "tilt_deg",
                "absorption",
                "correlated_partners",
            )
        },
        "correlation": merged.get("info", {}).get("correlation"),
        "incidents": merged["incidents"],
        "line_energies": line_energies(merged, data),
    }
    two_p = fit_line(merged, _two_p_one_s_guess(merged, data))
    out["fit_2p1s"] = two_p
    thickness_um = o["thickness_ang"] / 1e4
    for row in data["line_widths_table2"]["rows"]:
        if row["axis"] == axis and abs(o["energy_eV"] - 4e6) < 1e5:
            if abs(row["thickness_um"] - thickness_um) < 0.05:
                out["width_table2"] = {
                    **row,
                    "model_gamma_eV": two_p["gamma_eV"],
                    "relative": two_p["gamma_eV"] / row["gamma_exp_eV"] - 1.0,
                }
    fig = data["line_yield_fig11"]
    if (
        axis == fig["axis"]
        and abs(o["energy_eV"] - fig["kinetic_energy_MeV"] * 1e6) < 1e5
        and abs(thickness_um - fig["thickness_um"]) < 0.02
    ):
        pts = np.asarray(fig["points_deg_counts"], dtype=float)
        tilt = float(o.get("tilt_deg", 0.0))
        order = np.argsort(pts[:, 0])
        left = np.interp(-tilt, pts[order, 0], pts[order, 1])
        right = np.interp(tilt, pts[order, 0], pts[order, 1])
        k = fig["counts_per_photon_per_sr_per_electron"]
        out["yield_fig11"] = {
            "tilt_deg": tilt,
            "model_counts": two_p["area_per_sr"] * k,
            "model_counts_err": two_p["area_err_per_sr"] * k,
            "measured_counts_mean_pm": 0.5 * (left + right),
            "measured_counts_minus_plus": [float(left), float(right)],
        }
    return out


def main(argv: list[str]) -> None:
    data = json.loads(DATA.read_text())
    results = [compare(json.loads(Path(p).read_text()), data) for p in argv]
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
