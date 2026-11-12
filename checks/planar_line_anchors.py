"""Planar channeling-radiation absolute yields and line widths against measurements (#1057).

Datasets (transcribed with provenance under ``tests/data/channeling/``):

* Genz et al., Phys. Rev. B 53, 8922 (1996): natural diamond (110) and (111),
  5.2 and 9.0 MeV, 13-55 um, on-axis Si(Li) of 2.6e-6 sr. Table I absolute
  line yields [photons/(e sr)], Table IV line widths (FWHM), Table VII
  centroids. ``genz-1996-diamond-planar.json``.
* Kephart et al., Phys. Rev. B 40, 4249 (1989): Si (110) and (100),
  16.92 MeV. Integrated-exponential fits ``A (1 - exp(-Z/L))`` of the
  absorption-corrected line fluxes and their slopes at ``Z = 0``.
  ``kephart-1989-si-planar-yields.json``.
* Azadegan et al., Phys. Rev. B 74, 045209 (2006): diamond (110) 1->0,
  42.5 um, observed FWHM including the stated CdTe resolution.
  ``azadegan-2006-diamond110-widths.json``.
* Klein et al., Phys. Rev. B 31, 68 (1985): natural IIa diamond (110),
  23 um, 16.9 and 30.5 MeV, Table III Lorentzian FWHM of the
  detector-convolved lines (Ge 2 keV / 1 keV FWHM); the authors attribute
  their excess widths to crystal defects.
  ``klein-1985-diamond-planar-widths.json``.

Model: :func:`~pyrite.montecarlo.channeling.channeling_spectrum` with the TDS
plus electronic kernel at 293.15 K and the declared 300 eV collapse margin,
entry divergence normal to the plane, the stated energy spread and detector
cone, and self-absorption. ``channeling_spectrum`` takes one depth-independent
in-plane divergence, so in-plane angular diffusion is added here by linearity:
the slab is cut into depth bins, and each bin's emission is the difference of
two spectra (to the bin's two faces, same solver history, photon absorption
through the full slab via ``layers``) evaluated at that bin's in-plane RMS
angle ``sigma_y(z)``. Three ``sigma_y(z)`` models bracket the in-plane
spread:

* ``kernel`` (default): the out-of-plane angle of an electron diffuses with
  the planar-uniform decoherence rate ``Lambda(s) = Gamma(Delta = s / p)``
  (``decoherence_rate`` on a flat electron density, the
  ``detector_acceptance`` model of ``checks/kephart_occupation.py``), TDS and
  electronic parts weighted by the bound-state densities
  (``bound_rate_weights``). Its characteristic function
  ``exp(-z Lambda(s))`` is matched to a Gaussian at its ``e^{-1/2}`` point:
  ``z Lambda(1 / sigma_ms) = 1/2``; ``sigma_y^2 = sigma_0^2 + sigma_ms^2``.
  Coulomb tails make the growth sub-diffusive, as Azadegan observed.
* ``none``: ``sigma_y = sigma_0`` (entry divergence only).
* ``amorphous``: Highland projected angle for random incidence
  (PDG: ``13.6 MeV / (beta c p) sqrt(x/X0) (1 + 0.038 ln(x/X0))``), an upper
  bracket; channeled electrons scatter less in plane (Azadegan Table II:
  ~0.55x at 42.5 um; Kephart Sec. V E: < 1.6x of 2.5 sqrt(z/um) mrad).

Observables per measured line, from the bound -> bound part of the spectrum
(Genz and Azadegan fit bound lines separately from free -> bound
Gaussians): the window integral [photons/(e sr)], the FWHM of the peak in the
window, and the centroid. The window spans the line's Bloch-block centres and
reaches half way to the nearest other bound line (at most +-25%). Neighbour
lines inside the window are reported as a 4 pi yield fraction
(``contamination``).

Runs: each case is a few minutes at the default grid; the full set and the
grid/bin convergence arms belong on ``pyrite remote``::

    PYRITE_MC_BACKEND=cpu uv run python checks/planar_line_anchors.py \
        --datasets genz kephart azadegan klein --out planar_anchors.json

Validation: cr-line-shape, cr-yield-assembly
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pyrite.materials.attenuation import _resolve_composition
from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import (
    DensityMatrixSolver,
    channeling_spectrum,
    decoherence_rate,
    electronic_kernel,
    longitudinal_kinematics,
    tds_kernel,
)
from pyrite.montecarlo.channeling import emission as _emission

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "channeling"
TEMPERATURE_K = 293.15
SAMPLE_ANG = 500.0
REUSE_HISTORY = True
_KEPHART = runpy.run_path(str(Path(__file__).with_name("kephart_occupation.py")))
bound_rate_weights = _KEPHART["bound_rate_weights"]
# Radiation lengths [cm] (PDG): diamond 42.70 g/cm^2 / 3.515 g/cm^3, Si 9.37 cm.
RADIATION_LENGTH_CM = {"diamond": 12.15, "silicon": 9.37}


@dataclass(frozen=True)
class Case:
    """One crystal, plane, energy and thickness with its measured lines."""

    key: str
    dataset: str
    material: str
    hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    thicknesses_um: tuple[float, ...]
    divergence_rad: float
    relative_energy_spread: float
    aperture_rad: float
    detector_fwhm_eV: float
    absorption: bool
    transitions: tuple[tuple[int, int], ...]


def _genz_cases():
    data = json.loads((DATA / "genz-1996-diamond-planar.json").read_text())
    aperture = float(np.sqrt(data["detector"]["solid_angle_sr"] / np.pi))
    cases = []
    for hkl, transitions in (((1, 1, 0), ((1, 0),)), ((1, 1, 1), ((2, 1), (1, 0), (3, 2)))):
        for mev in data["beam"]["kinetic_energy_MeV"]:
            for z in data["crystals"]["thickness_um"]:
                cases.append(
                    Case(
                        f"genz_{''.join(map(str, hkl))}_{mev:g}_{z}",
                        "genz",
                        "diamond",
                        hkl,
                        mev * 1e6,
                        (float(z),),
                        data["beam"]["divergence_mrad"] * 1e-3,
                        data["beam"]["relative_energy_spread"],
                        aperture,
                        0.0,
                        True,
                        transitions if mev > 6 else transitions[:1],
                    )
                )
    return cases


def _kephart_cases():
    data = json.loads((DATA / "kephart-1989-si-planar-yields.json").read_text())
    thicknesses = (1.0, 2.8, 5.0, 7.8, 10.0, 19.0, 40.0, 80.0, 120.0)
    out = []
    for hkl, transitions in (((1, 1, 0), ((1, 0), (2, 1), (3, 0))), ((1, 0, 0), ((1, 0),))):
        out.append(
            Case(
                f"kephart_{''.join(map(str, hkl))}_16.92",
                "kephart",
                "silicon",
                hkl,
                data["kinetic_energy_MeV"] * 1e6,
                thicknesses,
                data["beam"]["divergence_mrad"] * 1e-3,
                0.0,
                0.0,
                0.0,
                False,  # Kephart's fluxes are absorption-corrected
                transitions,
            )
        )
    return out


def _azadegan_cases():
    data = json.loads((DATA / "azadegan-2006-diamond110-widths.json").read_text())
    return [
        Case(
            f"azadegan_110_{row['kinetic_energy_MeV']:g}_42.5",
            "azadegan",
            "diamond",
            (1, 1, 0),
            row["kinetic_energy_MeV"] * 1e6,
            (42.5,),
            data["beam"]["divergence_mrad"] * 1e-3,
            data["beam"]["relative_energy_spread"],
            0.0,
            row["detector_fwhm_keV"] * 1e3,
            True,
            ((1, 0),),
        )
        for row in data["rows_42p5um"]
    ]


def _klein_cases():
    data = json.loads((DATA / "klein-1985-diamond-planar-widths.json").read_text())
    fwhm_to_rms = 1.0 / 2.3548200450309493
    out = []
    for mev, transitions in ((16.9, ((1, 0),)), (30.5, ((1, 0), (2, 1), (3, 2)))):
        divergence = np.mean(np.atleast_1d(data["beam"]["divergence_fwhm_mrad"][f"{mev:g}"]))
        out.append(
            Case(
                f"klein_110_{mev:g}_23",
                "klein",
                "diamond",
                (1, 1, 0),
                mev * 1e6,
                (float(data["crystal"]["thickness_um"]),),
                divergence * 1e-3 * fwhm_to_rms,
                data["beam"]["energy_spread_fraction_max"] * fwhm_to_rms,
                0.0,
                data["detector"]["resolution_fwhm_keV"][f"{mev:g}"] * 1e3,
                True,
                transitions,
            )
        )
    return out


DATASETS = {
    "genz": _genz_cases,
    "kephart": _kephart_cases,
    "azadegan": _azadegan_cases,
    "klein": _klein_cases,
}


def inplane_rms(model, profile, material, energy_eV, weights, sigma0, z_ang):
    """In-plane RMS electron angle [rad] at depths ``z_ang`` for one ``sigma_y`` model."""
    z_ang = np.asarray(z_ang, dtype=float)
    gamma, beta = longitudinal_kinematics(energy_eV)
    if model == "none":
        return np.full(z_ang.shape, sigma0)
    if model == "amorphous":
        x = np.maximum(z_ang, 1.0) * 1e-8 / RADIATION_LENGTH_CM[material]
        pv_MeV = gamma * beta**2 * M_E_EV / 1e6
        theta = 13.6 / pv_MeV * np.sqrt(x) * np.maximum(1.0 + 0.038 * np.log(x), 0.0)
        return np.hypot(sigma0, theta)
    if model != "kernel":
        raise ValueError(f"unknown in-plane model {model!r}")
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    s = np.geomspace(1.0, 1e5, 800)
    flat = replace(profile, electron_density=profile.electron_density[:1].copy())
    parts = decoherence_rate(flat, energy_eV, s / p_inv_ang)
    rate = np.maximum.accumulate(weights[0] * parts.tds + weights[1] * parts.electronic)
    out = np.empty_like(z_ang)
    for i, z in enumerate(z_ang):
        if z <= 0.0:
            out[i] = sigma0
            continue
        target = 0.5 / z
        s_c = np.interp(target, rate, s) if target <= rate[-1] else s[-1]
        out[i] = np.hypot(sigma0, 1.0 / s_c)
    return out


def _bin_edges_ang(thickness_um, n_bins, extra_um=()):
    """Depth-bin faces on the estimator sample grid; quadratic spacing resolves early growth."""
    edges = thickness_um * np.linspace(0.0, 1.0, n_bins + 1) ** 2
    edges = np.unique(np.concatenate([edges, np.asarray(extra_um, dtype=float)]))
    edges = edges[edges <= thickness_um + 1e-9]
    snapped = np.unique(np.round(edges * 1e4 / SAMPLE_ANG) * SAMPLE_ANG)
    return snapped


@contextmanager
def _prefix_history(total_ang):
    """Reuse one depth history for every slab face ``<= total_ang`` (speed only).

    ``channeling_spectrum`` re-solves ``rho(z)`` from the entrance for each
    thickness. With faces on the ``SAMPLE_ANG`` grid its chained segments are
    exactly ``SAMPLE_ANG`` long, so the history to a shallower face is a prefix
    of the history to ``total_ang``. ``--no-history-reuse`` disables this and
    re-solves every face (used once to confirm identical output).
    """
    if not REUSE_HISTORY:
        yield
        return
    original = _emission._depth_history
    cache = {}

    def prefix(solver, lines, thickness_ang, dz_ang, sample_ang, *args, **kwargs):
        key = (id(solver), dz_ang, sample_ang, args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = original(solver, lines, total_ang, dz_ang, sample_ang, *args, **kwargs)
        full = cache[key]
        n = int(round(thickness_ang / sample_ang))
        if full.interval_weight is not None or not np.isclose(full.z_ang[n], thickness_ang):
            return original(solver, lines, thickness_ang, dz_ang, sample_ang, *args, **kwargs)
        return replace(
            full, z_ang=full.z_ang[: n + 1], rates=full.rates[: n + 1], bound=full.bound[: n + 1]
        )

    _emission._depth_history = prefix
    try:
        yield
    finally:
        _emission._depth_history = original


def binned_spectrum(solver, case, edges_ang, sigma_y, grid, dz_ang, n_divergence):
    """Bound->bound and capture spectra of each depth bin, absorbed through the full slab."""
    total = edges_ang[-1]
    layers = (
        [(0.0, total, _resolve_composition(solver.profile.crystal))] if case.absorption else None
    )
    kwargs = dict(
        divergence_rad=case.divergence_rad,
        energy_spread_eV=case.relative_energy_spread * case.kinetic_energy_eV,
        aperture_rad=case.aperture_rad,
        dz_ang=dz_ang,
        sample_ang=SAMPLE_ANG,
        absorption=case.absorption,
        layers=layers,
        n_divergence=n_divergence,
    )
    bound_bins, capture_bins, last = [], [], None
    with _prefix_history(total):
        for lo, hi, sig in zip(edges_ang[:-1], edges_ang[1:], sigma_y, strict=True):
            hi_result = channeling_spectrum(solver, hi, grid, 0.0, divergence_y_rad=sig, **kwargs)
            bound = hi_result.spectrum - hi_result.capture_spectrum
            capture = hi_result.capture_spectrum.copy()
            if lo > 0.0:
                lo_result = channeling_spectrum(
                    solver, lo, grid, 0.0, divergence_y_rad=sig, **kwargs
                )
                bound = bound - (lo_result.spectrum - lo_result.capture_spectrum)
                capture = capture - lo_result.capture_spectrum
            bound_bins.append(bound)
            capture_bins.append(capture)
            last = hi_result
    return np.asarray(bound_bins), np.asarray(capture_bins), last


def _gaussian_smear(grid, spectrum, fwhm_eV):
    if fwhm_eV <= 0.0:
        return spectrum
    sigma = fwhm_eV / 2.3548200450309493
    diff = grid[:, None] - grid[None, :]
    kernel = np.exp(-0.5 * (diff / sigma) ** 2)
    kernel /= kernel.sum(axis=0, keepdims=True)
    return kernel @ spectrum


def line_window(lines, line_yield, transition):
    """Window [eV] for one transition and the 4 pi yield fraction of other lines inside it."""
    centre = lines.forward_energy_eV()
    bound = lines.kind == 0
    target = bound & (lines.upper == transition[0]) & (lines.lower == transition[1])
    if not np.any(target):
        return None
    lo, hi = centre[target].min(), centre[target].max()
    reference = line_yield[target].sum()
    others = bound & ~target & (line_yield > 0.02 * reference)
    below = centre[others & (centre < lo)]
    above = centre[others & (centre > hi)]
    left = max(0.75 * lo, 0.5 * (lo + below.max())) if below.size else 0.75 * lo
    right = min(1.25 * hi, 0.5 * (hi + above.min())) if above.size else 1.25 * hi
    inside = ~target & (centre >= left) & (centre <= right)
    return left, right, float(line_yield[inside].sum() / reference)


def line_metrics(grid, spectrum, window):
    """Integral, centroid, peak and FWHM of ``spectrum`` inside ``window``."""
    left, right = window
    mask = (grid >= left) & (grid <= right)
    e, y = grid[mask], spectrum[mask]
    area = float(np.trapezoid(y, e))
    peak = int(np.argmax(y))
    half = 0.5 * y[peak]
    i = peak
    while i > 0 and y[i] > half:
        i -= 1
    j = peak
    while j < y.size - 1 and y[j] > half:
        j += 1
    lo = np.interp(half, [y[i], y[i + 1]], [e[i], e[i + 1]]) if y[i] <= half else e[0]
    hi = np.interp(half, [y[j], y[j - 1]], [e[j], e[j - 1]]) if y[j] <= half else e[-1]
    return {
        "yield_per_sr": area,
        "centroid_eV": float(np.trapezoid(y * e, e) / area) if area > 0 else float("nan"),
        "peak_eV": float(e[peak]),
        "fwhm_eV": float(hi - lo),
        "fwhm_clipped": bool(y[i] > half or y[j] > half),
    }


def run_case(case, *, cells, orders, dz_ang, n_bins, models, n_grid, n_divergence, tds_scale=1.0):
    """Spectrum observables of one case for every in-plane model."""
    start = time.perf_counter()
    profile = planar_profile(case.material, case.hkl, TEMPERATURE_K)
    energy = case.kinetic_energy_eV
    tds = tds_kernel(profile, energy, n_cells=cells, max_order=orders)
    if tds_scale != 1.0:
        # Sensitivity arm only: a uniform factor on the Einstein TDS kernel
        # (loss and jumps alike) stands in for a correlation enhancement C.
        tds = replace(
            tds,
            momentum_kernel=tds_scale * tds.momentum_kernel,
            diagonal_coefficients=tds_scale * tds.diagonal_coefficients,
        )
    el = electronic_kernel(profile, energy, n_cells=cells, max_order=orders)
    solver = DensityMatrixSolver(profile, energy, tds + el)
    weights = bound_rate_weights(solver, (tds, el), case.divergence_rad)
    weights = (weights[0] * tds_scale, weights[1])  # in-plane diffusion scales alike
    probe = channeling_spectrum(solver, SAMPLE_ANG, np.array([1.0]), 0.0, dz_ang=dz_ang)
    centres = probe.lines.forward_energy_eV()[probe.lines.kind == 0]
    grid = np.linspace(0.35 * centres.min(), 1.6 * centres.max(), n_grid)
    thickness = max(case.thicknesses_um)
    edges = _bin_edges_ang(thickness, n_bins, case.thicknesses_um if not case.absorption else ())
    mids = 0.5 * (edges[:-1] + edges[1:])
    out = {
        "case": case.key,
        "dataset": case.dataset,
        "material": case.material,
        "hkl": list(case.hkl),
        "kinetic_energy_MeV": energy / 1e6,
        "barrier_eV": solver.barrier_eV,
        "tds_scale": tds_scale,
        "divergence_rad": case.divergence_rad,
        "y_rate_weights": list(weights),
        "edges_um": (edges / 1e4).tolist(),
        "models": {},
    }
    for model in models:
        sigma = inplane_rms(
            model, profile, case.material, energy, weights, case.divergence_rad, mids
        )
        bound_bins, capture_bins, last = binned_spectrum(
            solver, case, edges, sigma, grid, dz_ang, n_divergence
        )
        per_thickness = []
        targets = case.thicknesses_um if not case.absorption else (thickness,)
        for z_um in targets:
            n_in = int(np.searchsorted(edges, z_um * 1e4 + 1e-6) - 1)
            bound = _gaussian_smear(grid, bound_bins[:n_in].sum(axis=0), case.detector_fwhm_eV)
            capture = capture_bins[:n_in].sum(axis=0)
            row = {"thickness_um": float(edges[n_in] / 1e4), "lines": {}}
            for transition in case.transitions:
                window = line_window(last.lines, last.line_yield, transition)
                if window is None:
                    row["lines"][f"{transition[0]}-{transition[1]}"] = None
                    continue
                metrics = line_metrics(grid, bound, window[:2])
                metrics["with_capture_yield_per_sr"] = line_metrics(
                    grid, bound + _gaussian_smear(grid, capture, case.detector_fwhm_eV), window[:2]
                )["yield_per_sr"]
                metrics["window_eV"] = [float(window[0]), float(window[1])]
                metrics["contamination"] = window[2]
                row["lines"][f"{transition[0]}-{transition[1]}"] = metrics
            per_thickness.append(row)
        out["models"][model] = {
            "sigma_y_mrad": (sigma * 1e3).tolist(),
            "thicknesses": per_thickness,
        }
    out["seconds"] = time.perf_counter() - start
    return out


def _revision():
    if os.environ.get("PYRITE_SOURCE_REVISION"):
        return os.environ["PYRITE_SOURCE_REVISION"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _measured(rows, hkl, transition, mev):
    for row in rows:
        if tuple(row["plane"]) == tuple(hkl) and row["transition"] == list(transition):
            if np.isclose(row["kinetic_energy_MeV"], mev):
                return row
    return None


def report(paths, out):
    """Model/measurement tables from stored run JSONs (default grid, unscaled arms)."""
    genz = json.loads((DATA / "genz-1996-diamond-planar.json").read_text())
    kephart = json.loads((DATA / "kephart-1989-si-planar-yields.json").read_text())
    azadegan = json.loads((DATA / "azadegan-2006-diamond110-widths.json").read_text())
    klein = json.loads((DATA / "klein-1985-diamond-planar-widths.json").read_text())
    thick = genz["crystals"]["thickness_um"]
    rows = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        settings = record["settings"]
        arm = {
            k: settings.get(k)
            for k in ("cells", "orders", "dz", "bins", "divergence_scale", "tds_scale")
        }
        for case in record["cases"]:
            mev = case["kinetic_energy_MeV"]
            for model, result in case["models"].items():
                for t in result["thicknesses"]:
                    for name, m in t["lines"].items():
                        if m is None:
                            continue
                        transition = [int(v) for v in name.split("-")]
                        base = {
                            "source": Path(path).name,
                            "arm": arm,
                            "case": case["case"],
                            "model": model,
                            "line": name,
                            "thickness_um": t["thickness_um"],
                            "model_yield": m["yield_per_sr"],
                            "model_yield_with_capture": m["with_capture_yield_per_sr"],
                            "model_fwhm_keV": m["fwhm_eV"] / 1e3,
                            "model_centroid_keV": m["centroid_eV"] / 1e3,
                            "contamination": m["contamination"],
                            "fwhm_clipped": m["fwhm_clipped"],
                        }
                        if case["dataset"] == "genz":
                            i = thick.index(int(round(t["thickness_um"])))
                            for key, table in (
                                ("yield", "yields_table1"),
                                ("fwhm", "widths_table4"),
                                ("centroid", "energies_table7"),
                            ):
                                meas = _measured(genz[table]["rows"], case["hkl"], transition, mev)
                                if meas is None:
                                    continue
                                scale = 1e-3 if key == "yield" else 1.0
                                base[f"measured_{key}"] = meas["measured"][i] * scale
                                base[f"measured_{key}_err"] = meas["uncertainty"][i] * scale
                                if meas.get("theory"):
                                    base[f"genz_theory_{key}"] = meas["theory"][i] * scale
                        elif case["dataset"] == "kephart":
                            for fit in kephart["integrated_exponential_fits"]["rows"]:
                                if fit["plane"] == case["hkl"] and fit["transition"] == transition:
                                    z = t["thickness_um"]
                                    base["measured_yield"] = fit["amplitude"] * (
                                        1 - np.exp(-z / fit["length_um"])
                                    )
                                    base["measured_slope"] = fit["slope_text"]
                                    base["model_slope_estimate"] = m["yield_per_sr"] / z
                        elif case["dataset"] == "klein":
                            for kl in klein["widths_table3"]["rows"]:
                                if (
                                    kl["plane"] == case["hkl"]
                                    and kl["transition"] == transition
                                    and np.isclose(kl["kinetic_energy_MeV"], mev)
                                ):
                                    base["measured_fwhm"] = kl["measured"]
                                    base["measured_fwhm_err"] = kl["uncertainty"]
                                    base["measured_centroid"] = kl["energy_keV"]
                                    base["klein_calc_1"] = kl["calc_1"]
                        else:
                            for az in azadegan["rows_42p5um"]:
                                if np.isclose(az["kinetic_energy_MeV"], mev):
                                    base["measured_fwhm"] = az["gamma_exp_keV"]
                                    base["measured_centroid"] = az["E_exp_keV"]
                                    base["azadegan_theta_ms_ch_mrad"] = az["theta_ms_ch_mrad"]
                                    base["model_sigma_y_exit_mrad"] = result["sigma_y_mrad"][-1]
                        for key in ("yield", "fwhm", "centroid"):
                            model_key = {
                                "yield": "model_yield",
                                "fwhm": "model_fwhm_keV",
                                "centroid": "model_centroid_keV",
                            }[key]
                            if base.get(f"measured_{key}"):
                                base[f"ratio_{key}"] = base[model_key] / base[f"measured_{key}"]
                        rows.append(base)
    for row in rows:
        ratios = " ".join(
            f"{k[6:]}={row[k]:.3f}"
            for k in ("ratio_yield", "ratio_fwhm", "ratio_centroid")
            if k in row
        )
        print(
            f"{row['case']:22s} {row['model']:9s} {row['line']:4s} {row['thickness_um']:6.1f}um "
            f"Y={row['model_yield']:.3g} W={row['model_fwhm_keV']:.3f} E={row['model_centroid_keV']:.3f} "
            f"{ratios} {row['arm']}"
        )
    Path(out).write_text(
        json.dumps({"check": "planar_line_anchors report", "rows": rows}, indent=1)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--cases", nargs="*", help="Case keys (substring match); default all")
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--orders", type=int, default=16)
    parser.add_argument("--dz", type=float, default=100.0, help="solver step [angstrom]")
    parser.add_argument("--bins", type=int, default=6, help="depth bins for in-plane diffusion")
    parser.add_argument("--models", nargs="+", default=["kernel", "none", "amorphous"])
    parser.add_argument("--grid", type=int, default=2400, help="photon-energy grid points")
    parser.add_argument("--n-divergence", type=int, default=15)
    parser.add_argument(
        "--divergence-scale", type=float, default=1.0, help="multiply the stated beam divergence"
    )
    parser.add_argument(
        "--tds-scale", type=float, default=1.0, help="sensitivity: multiply the TDS kernel"
    )
    parser.add_argument("--no-history-reuse", action="store_true")
    parser.add_argument("--report", nargs="+", help="tabulate stored run JSONs instead of running")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.report:
        report(args.report, args.out)
        return
    global REUSE_HISTORY
    REUSE_HISTORY = not args.no_history_reuse
    cases = [c for name in args.datasets for c in DATASETS[name]()]
    if args.cases:
        cases = [c for c in cases if any(token in c.key for token in args.cases)]
    cases = [replace(c, divergence_rad=c.divergence_rad * args.divergence_scale) for c in cases]
    rows = []
    for case in cases:
        row = run_case(
            case,
            cells=args.cells,
            orders=args.orders,
            dz_ang=args.dz,
            n_bins=args.bins,
            models=args.models,
            n_grid=args.grid,
            n_divergence=args.n_divergence,
            tds_scale=args.tds_scale,
        )
        rows.append(row)
        kernel = row["models"][args.models[0]]["thicknesses"]
        summary = {
            f"{t['thickness_um']:g}um {name}": (
                None
                if m is None
                else f"Y={m['yield_per_sr']:.3g}/sr FWHM={m['fwhm_eV']:.0f}eV E={m['centroid_eV'] / 1e3:.2f}keV"
            )
            for t in kernel
            for name, m in t["lines"].items()
        }
        print(case.key, f"{row['seconds']:.0f}s", json.dumps(summary), flush=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "check": "planar_line_anchors",
                    "revision": _revision(),
                    "host": platform.node(),
                    "backend": os.environ.get("PYRITE_MC_BACKEND", ""),
                    "temperature_K": TEMPERATURE_K,
                    "settings": vars(args),
                    "cases": rows,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
