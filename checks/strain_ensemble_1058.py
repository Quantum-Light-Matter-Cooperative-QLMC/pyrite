"""Stage-0 quasi-static strain and tilt ensembles for CBS, PXR and planar CR (#1058).

Research reference, no gate and no production change. Every number comes
from existing surfaces; the only new step is averaging them over a frozen
acoustic (or static) distribution of local plane tilt and lattice parameter:

* CBS: the classical forward resonance ``E = hbar c beta g sin(psi) / (1 - beta)``
  of one planar reflection, convolved with the beam divergence and with the
  arcsine distribution of a sinusoidal tilt of amplitude ``a``. The coherent
  sideband picture (wave along the beam, weights ``J_m(g u0)**2``) is checked
  against it through the second moment.
* PXR: ``pyrite.materials.emission.photon_energies`` (exact recoil root) for
  the tilt lever; the d-spacing lever is exact (``E`` is linear in ``g``).
  ``chi_g`` and ``absorption_length_ang`` give the extinction/absorption
  ratio that decides whether strain can relieve extinction.
* Planar CR: ``channeling_spectrum`` line yields (TDS + electronic kernels,
  1 um, no absorption) versus entrance tilt, then averaged over the arcsine
  distribution with Gauss--Chebyshev nodes. Isotropic lattice strain is
  applied by scaling the catalog cell of a private ``CRYSTALS`` copy for the
  duration of one profile build (research only).

Run (CPU, a few minutes per planar case)::

    pixi run python checks/strain_ensemble_1058.py --out scratch/strain-1058.json

``--skip-planar`` runs the analytic parts only (seconds).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.special import jv

from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    absorption_length_ang,
    chi_g,
)
from pyrite.materials.emission import photon_energies

# Representative shear-wave speeds [m/s] (tilt waves); handbook values, +-10 %.
SHEAR_SPEED = {"silicon": 5.84e3, "diamond": 12.8e3, "quartz": 3.8e3}
TILT_AMPLITUDES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
R_E_ANG = 2.8179403262e-5


def kinematics(kinetic_eV):
    gamma = 1.0 + kinetic_eV / M_E_EV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return gamma, beta


def arcsine_gauss_profile(x, a, sigma):
    """Arcsine(amplitude ``a``) convolved with a unit Gaussian of RMS ``sigma`` on grid ``x``."""
    nodes = a * np.cos((2 * np.arange(1, 65) - 1) * np.pi / 128)  # Gauss-Chebyshev
    prof = np.exp(-0.5 * ((x[:, None] - nodes[None, :]) / sigma) ** 2).mean(axis=1)
    return prof / (np.sqrt(2 * np.pi) * sigma)


def cbs_case(material, d_ang, kinetic_eV, psi_rad, divergence_rad, frequency_hz=1.0e9):
    """Forward planar CBS lever: line shift, envelope and sideband check."""
    gamma, beta = kinematics(kinetic_eV)
    g = 2 * np.pi / d_ang
    scale = HBARC_EV_ANG * beta * g / (1.0 - beta)  # eV per rad of tilt
    e0 = scale * np.sin(psi_rad)
    x = np.linspace(-6, 6, 4001) * (divergence_rad + 3e-3)
    ref = arcsine_gauss_profile(x, 0.0, divergence_rad).max()
    rows = []
    q = 2 * np.pi * frequency_hz / SHEAR_SPEED[material] * 1e-10  # 1/angstrom
    spacing = HBARC_EV_ANG * beta * q / (1.0 - beta)
    for a in TILT_AMPLITUDES:
        prof = arcsine_gauss_profile(x, a, divergence_rad)
        u0 = a / q  # displacement amplitude [angstrom] of a wave along the beam
        z = g * u0
        m = np.arange(-int(z) - 40, int(z) + 41)
        w = jv(m, z) ** 2
        rows.append(
            {
                "tilt_amplitude_rad": a,
                "shift_over_E_at_peak_tilt": a / np.tan(psi_rad),
                "peak_height_ratio": float(prof.max() / ref),
                "rms_width_over_E": float(np.hypot(divergence_rad, a / np.sqrt(2)) / psi_rad),
                "sideband_g_u0": float(z),
                "sideband_weight_sum": float(w.sum()),
                "sideband_rms_eV": float(np.sqrt((w * (m * spacing) ** 2).sum())),
                "arcsine_rms_eV": float(scale * a / np.sqrt(2)),
            }
        )
    return {
        "material": material,
        "d_ang": d_ang,
        "kinetic_MeV": kinetic_eV / 1e6,
        "psi_mrad": psi_rad * 1e3,
        "divergence_mrad": divergence_rad * 1e3,
        "line_eV": float(e0),
        "divergence_fwhm_over_E": float(2.3548 * divergence_rad / psi_rad),
        "acoustic_frequency_GHz": frequency_hz / 1e9,
        "sideband_spacing_eV": float(spacing),
        "rows": rows,
    }


def pxr_case(material, kinetic_eV, theta_obs_deg, hkl, base_tilts_deg=(15.0, 45.0, 75.0)):
    """PXR tilt lever from the exact recoil root; d-spacing lever is exact (-eps).

    At zero sample tilt the default cut puts ``g`` along the beam, so the beam
    meets the planes at ``theta_B = 90 deg - tilt``; the classical root then
    gives ``d ln E / d tilt = -cot(theta_B)``.
    """
    rows = []
    for base in base_tilts_deg:
        tilts = base + np.array([-0.5e-3, 0.5e-3]) * 180 / np.pi  # 1 mrad central step
        e = photon_energies(
            material, kinetic_eV, tilt_deg=tilts, theta_obs_deg=theta_obs_deg, hkl=hkl
        )[0]
        rows.append(
            {
                "sample_tilt_deg": base,
                "theta_B_deg": 90.0 - base,
                "line_eV": float(e.mean()),
                "dlnE_dtilt_per_rad": float((e[1] - e[0]) / e.mean() / 1e-3),
                "minus_cot_theta_B": float(-1.0 / np.tan(np.radians(90.0 - base))),
            }
        )
    return {
        "material": material,
        "hkl": list(hkl),
        "kinetic_MeV": kinetic_eV / 1e6,
        "theta_obs_deg": theta_obs_deg,
        "rows": rows,
        "dlnE_dstrain": -1.0,
    }


def extinction_rows(material, element, hkl, energies_eV):
    """Symmetric-Bragg extinction depth vs absorption length (sigma polarisation)."""
    info = CRYSTALS[material]
    a = float(info["lattice"]["a"])
    d = a / np.sqrt(sum(h * h for h in hkl))
    n = len(info["basis"]) / float(info["V_cell"])
    rows = []
    for energy in energies_eV:
        lam = 2 * np.pi * HBARC_EV_ANG / energy
        s = lam / (2 * d)
        if s >= 1.0:
            continue
        chi = abs(complex(chi_g(material, hkl, energy)))
        l_ext = lam * s / (2 * np.pi * chi)  # 1/e amplitude depth at the Bragg centre
        l_abs = float(absorption_length_ang(element, energy, n))
        darwin = 2 * chi / np.sin(2 * np.arcsin(s))
        rows.append(
            {
                "photon_eV": float(energy),
                "theta_B_deg": float(np.degrees(np.arcsin(s))),
                "extinction_um": float(l_ext / 1e4),
                "absorption_um": float(l_abs / 1e4),
                "abs_over_ext": float(l_abs / l_ext),
                "darwin_width_urad": float(darwin * 1e6),
            }
        )
    return {"material": material, "hkl": list(hkl), "rows": rows}


@contextmanager
def strained(material, eps):
    """Scale the catalog cell isotropically by ``1 + eps`` (research only; restored on exit)."""
    original = CRYSTALS[material]
    lattice = dict(original["lattice"])
    for key in ("a", "b", "c"):
        lattice[key] = lattice[key] * (1.0 + eps)
    CRYSTALS[material] = {
        **original,
        "lattice": MappingProxyType(lattice),
        "V_cell": float(original["V_cell"]) * (1.0 + eps) ** 3,
    }
    try:
        yield
    finally:
        CRYSTALS[material] = original


def _solver(material, hkl, kinetic_eV, eps=0.0):
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel

    with strained(material, eps):
        profile = planar_profile(material, hkl)
    kernel = tds_kernel(profile, kinetic_eV) + electronic_kernel(profile, kinetic_eV)
    return profile, DensityMatrixSolver(profile, kinetic_eV, kernel)


def _well_depth(profile):
    x = np.linspace(0.0, profile.period_ang, 2001)
    m = np.arange(profile.potential_eV.size)
    u = 2 * np.real(
        profile.potential_eV[None, 1:]
        * np.exp(2j * np.pi * m[None, 1:] * x[:, None] / profile.period_ang)
    ).sum(axis=1)
    return float(u.max() - u.min())


def _line_table(solver, thickness_ang, incidence, divergence):
    from pyrite.montecarlo.channeling import channeling_spectrum

    r = channeling_spectrum(
        solver,
        thickness_ang,
        np.array([1.0]),
        0.0,
        incidence_rad=incidence,
        divergence_rad=divergence,
        absorption=False,
        include_capture=False,
    )
    lines, y = r.lines, r.line_yield
    centre = lines.forward_energy_eV()
    out = {}
    for up, lo, e, w, kind in zip(lines.upper, lines.lower, centre, y, lines.kind, strict=True):
        if kind != 0:
            continue
        key = f"{int(up)}-{int(lo)}"
        acc = out.setdefault(key, [0.0, 0.0])
        acc[0] += float(w)
        acc[1] += float(w * e)
    return {
        k: {"yield": v[0], "centroid_eV": v[1] / v[0] if v[0] > 0 else None} for k, v in out.items()
    }


def planar_case(material, hkl, kinetic_eV, divergence_rad, thickness_um, strains, n_nodes=4):
    """Line yields vs entrance tilt, arcsine tilt ensembles, and isotropic-strain shifts."""
    start = time.perf_counter()
    profile, solver = _solver(material, hkl, kinetic_eV)
    gamma, beta = kinematics(kinetic_eV)
    depth = _well_depth(profile)
    psi_c = float(np.sqrt(2 * depth / (gamma * beta**2 * M_E_EV)))
    fractions = (0.1, 0.3, 1.0)
    cheb = np.cos((2 * np.arange(1, n_nodes + 1) - 1) * np.pi / (4 * n_nodes))  # half-range nodes
    tilts = sorted({0.0} | {float(f * psi_c * c) for f in fractions for c in cheb})
    thickness = thickness_um * 1e4
    scan = {t: _line_table(solver, thickness, t, divergence_rad) for t in tilts}
    keys = sorted(scan[0.0], key=lambda k: -scan[0.0][k]["yield"])[:4]
    ensembles = []
    for f in fractions:
        nodes = [float(f * psi_c * c) for c in cheb]
        avg = {k: float(np.mean([scan[t][k]["yield"] for t in nodes])) for k in keys}
        ensembles.append(
            {
                "amplitude_over_psi_c": f,
                "amplitude_mrad": f * psi_c * 1e3,
                "yield_ratio_to_aligned": {k: avg[k] / scan[0.0][k]["yield"] for k in keys},
            }
        )
    shifts = {}
    for eps in strains:
        _, s_eps = _solver(material, hkl, kinetic_eV, eps)
        table = _line_table(s_eps, thickness, 0.0, divergence_rad)
        shifts[f"{eps:g}"] = {
            k: (table[k]["centroid_eV"] / scan[0.0][k]["centroid_eV"] - 1.0) / eps
            for k in keys
            if k in table and table[k]["centroid_eV"]
        }
    return {
        "material": material,
        "hkl": list(hkl),
        "kinetic_MeV": kinetic_eV / 1e6,
        "thickness_um": thickness_um,
        "divergence_mrad": divergence_rad * 1e3,
        "well_depth_eV": depth,
        "psi_c_mrad": psi_c * 1e3,
        "lines": keys,
        "aligned": {k: scan[0.0][k] for k in keys},
        "tilt_scan": [
            {"tilt_mrad": t * 1e3, **{k: scan[t][k]["yield"] for k in keys}} for t in tilts
        ],
        "ensembles": ensembles,
        "dlnE_dstrain": shifts,
        "seconds": time.perf_counter() - start,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-planar", action="store_true")
    args = parser.parse_args(argv)
    record = {
        "issue": 1058,
        "cbs": [
            cbs_case("silicon", 5.4309 / np.sqrt(8), 16.92e6, 5e-3, 0.2e-3),
            cbs_case("diamond", 3.5668 / np.sqrt(8), 9.0e6, 5e-3, 0.3e-3),
            cbs_case("quartz", 3.343, 20.0e6, 1.9e-3, 0.5e-3, frequency_hz=1.024e9),
        ],
        "pxr": [
            pxr_case("silicon", 16.92e6, 90.0, (2, 2, 0)),
            pxr_case("diamond", 9.0e6, 90.0, (2, 2, 0)),
        ],
        "extinction": [
            extinction_rows("silicon", "Si", (1, 1, 1), (2.5e3, 8e3, 17.5e3, 30e3)),
            extinction_rows("silicon", "Si", (2, 2, 0), (8e3, 17.5e3, 30e3)),
            extinction_rows("diamond", "C", (2, 2, 0), (3e3, 8e3, 17.5e3, 30e3)),
        ],
    }
    if not args.skip_planar:
        record["planar"] = [
            planar_case("silicon", (1, 1, 0), 16.92e6, 0.2e-3, 1.0, (1e-3,)),
            planar_case("diamond", (1, 1, 0), 9.0e6, 0.3e-3, 1.0, (1e-3,)),
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(json.dumps(record, indent=1, default=float))


if __name__ == "__main__":
    main()
