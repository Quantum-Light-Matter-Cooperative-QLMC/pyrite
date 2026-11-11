"""In-crystal survival of the exact vacuum line-recoil shift (#527 gate).

Compares Huang et al., Nat. Photon. 17, 224 (2023), SI Table S4 (graphite and
hBN (00m) lines from 10.7-14.5 keV electrons, m = 2 and 4) with the straight
single-electron exact vacuum root ``materials/recoil.py::recoil_resonance_eV``
at the paper's fitted tilts. The fit-independent observable is the harmonic
defect ``2 E2 - E4``: the classical root is linear in ``g``, so it is exactly
zero for every tilt and is unchanged by a linear EDS gain error; a calibration
offset ``a`` enters it as ``+a``.

It then sizes the in-medium corrections the vacuum root omits, at the Fig. 1c
(graphite, 11.7 keV, 130 deg) and Fig. 2e (hBN, 14.0 keV, 119 deg) geometries:

- electron mean inner potential ``U0`` (``cr-mean-inner-potential``): both
  states gain ``-U0`` of kinetic energy, so the root is evaluated at ``T - U0``;
- photon refraction: first-order exit-face Snell root change. Tangential
  continuity gives ``k_eff = omega (n + delta grad L)``, ``grad L = -e/(e.n)``
  with ``e`` the outward exit-face normal (``xray-in-medium-resonance``), so the
  denominator gains ``delta beta u.e/(e.n)``. ``delta = 1 - Re n`` comes from
  ``materials/crystal.py::refractive_index``. A backward photon leaves through
  the entrance face, a forward photon through the back face;
- electron refraction at the (001) entrance face (tangential momentum kept);
- Bloch dispersion: per-coupling two-beam branch shifts
  ``-d/2 + sign(d) sqrt(d^2/4 + U_h^2)``, ``d = E(k) - E(k + h)``, of the initial
  and final free-like states (``U_h^2/d`` far from a Bragg condition, at most
  ``|U_h|`` at one; one initial coupling here is within 0.5 eV of Bragg),
  ``U_h = (hbar^2 / 2 m0) 4 pi F_e(h) / V`` with Mott-Bethe ``F_e``, summed over
  low-order graphite reflections; the line moves by their difference.
  ILLUSTRATIVE ONLY: it is one arbitrary in-plane crystal azimuth (x_c along
  y x c; Huang does not report the rotation about [001]), and it depends on the
  reflection cutoff. Over azimuth the free-like branch shift of the defect has a
  median of ~0.4-0.5 eV and reaches several eV near a strong Bragg condition.
  What licenses neglecting it is the two-beam sum rule: the excitation-weighted
  branch centroid is unshifted, so at EDS resolution the line broadens rather
  than moves. It is therefore excluded from the gate.

    uv run python checks/recoil_in_crystal_527.py

Exit 0 when the mean measured-minus-predicted defect lies within twice its
combined statistical and calibration uncertainty, the classical value lies
more than ten such uncertainties away, and the mean-inner-potential,
exit-face photon refraction and entrance-refraction changes of the defect are
each below 1 eV.

Validation: line-recoil-in-crystal
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pyrite.materials.atomic import cromer_mann_f0
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV, refractive_index
from pyrite.materials.recoil import recoil_resonance_eV, recoil_shift_fraction
from pyrite.montecarlo.channeling.entry import mean_inner_potential_eV

DATA = Path(__file__).resolve().parents[1] / "tests/data/recoil/huang-2023-table-s4.json"
BEAM = np.array([0.0, 0.0, 1.0])
A0_ANG = 0.529177210544
H2M_EV_ANG2 = HBARC_EV_ANG**2 / (2.0 * M_E_EV)


def geometry(theta_til, phi_til, theta_obs):
    """Unit [001] axis and photon direction in the lab frame (paper Eq. 5)."""
    t, p, o = np.radians([theta_til, phi_til, theta_obs])
    axis = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    return axis, np.array([np.sin(o), 0.0, np.cos(o)])


def lines(kinetic_eV, c_ang, axis, normal, orders=(2, 4)):
    """(classical, exact) photon energies for (00m); PyRITE sign g = +m 2 pi/c axis."""
    out = []
    for m in orders:
        g = m * 2.0 * np.pi / c_ang * axis
        exact = float(recoil_resonance_eV(kinetic_eV, BEAM, g, normal))
        shift = float(recoil_shift_fraction(kinetic_eV, BEAM, g, normal))
        out.append((exact / (1.0 + shift), exact))
    return out


def table_comparison(data):
    sets = data["sets"]
    rows = []
    for name, kinetic_keV, e2, e4, theta_til, phi_til in data["rows"]:
        info = sets[name]
        axis, normal = geometry(theta_til, phi_til, info["theta_obs_deg"])
        (c2, q2), (c4, q4) = lines(kinetic_keV * 1e3, info["c_ang"], axis, normal)
        rows.append((name, kinetic_keV, e2, e4, c2, q2, c4, q4))
    return rows


def medium_corrections(key, c_ang, kinetic_eV, theta_til, phi_til, theta_obs):
    """Per-line (U0, photon, entry-bend) shifts in eV for orders 2 and 4."""
    axis, normal = geometry(theta_til, phi_til, theta_obs)
    u0 = mean_inner_potential_eV(key)
    gamma = 1.0 + kinetic_eV / M_E_EV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    u_n = float(BEAM @ normal)
    p0 = np.sqrt(kinetic_eV * (kinetic_eV + 2 * M_E_EV))
    p1 = np.sqrt((kinetic_eV - u0) * (kinetic_eV - u0 + 2 * M_E_EV))
    sin_in = np.sqrt(1.0 - float(BEAM @ axis) ** 2)
    bend = np.arcsin(sin_in * p0 / p1) - np.arcsin(sin_in)  # toward the normal
    # cross(cross(axis, z), z) / sin_in points away from the tilted normal and
    # bend < 0, so +sin(bend) turns the direction toward the normal.
    bent = np.cos(bend) * BEAM + np.sin(bend) * np.cross(np.cross(axis, BEAM), BEAM) / sin_in
    bent /= np.linalg.norm(bent)
    out = {}
    for m in (2, 4):
        g = m * 2.0 * np.pi / c_ang * axis
        e0 = float(recoil_resonance_eV(kinetic_eV, BEAM, g, normal))
        e_mip = float(recoil_resonance_eV(kinetic_eV - u0, BEAM, g, normal)) - e0
        delta = 1.0 - complex(refractive_index(key, np.array([e0]))[0]).real
        exit_normal = axis if float(normal @ axis) > 0.0 else -axis  # outward exit face
        u_e = float(BEAM @ exit_normal)
        e_phot = -e0 * delta * beta * u_e / float(exit_normal @ normal) / (1.0 - beta * u_n)
        e_bend = float(recoil_resonance_eV(kinetic_eV, bent, g, normal)) - e0
        out[m] = (e0, u0, e_mip, delta, e_phot, e_bend)
    return out


def graphite_bloch(kinetic_eV, theta_til, phi_til, theta_obs, c_ang=6.71, a_ang=2.464):
    """Second-order Bloch shift of the (002)/(004) lines for AB graphite."""
    lattice = np.array([[a_ang, 0, 0], [-a_ang / 2, a_ang * np.sqrt(3) / 2, 0], [0, 0, c_ang]])
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    volume = abs(np.linalg.det(lattice))
    basis = ((0, 0, 0), (0, 0, 0.5), (1 / 3, 2 / 3, 0), (2 / 3, 1 / 3, 0.5))
    axis, normal = geometry(theta_til, phi_til, theta_obs)
    xc = np.cross([0.0, 1.0, 0.0], axis)
    xc /= np.linalg.norm(xc)
    rot = np.column_stack([xc, np.cross(axis, xc), axis])
    gamma = 1.0 + kinetic_eV / M_E_EV
    k_i = np.sqrt(kinetic_eV * (kinetic_eV + 2 * M_E_EV)) / HBARC_EV_ANG * BEAM
    couplings = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for ell in range(-4, 5):
                if (h, k, ell) == (0, 0, 0):
                    continue
                vec = rot @ (h * recip[0] + k * recip[1] + ell * recip[2])
                size = np.linalg.norm(vec)
                f_x = float(np.asarray(cromer_mann_f0("C", np.array([size]))).ravel()[0])
                f_e = 2.0 / A0_ANG * (6.0 - f_x) / size**2
                phase = sum(np.exp(2j * np.pi * (h * x + k * y + ell * z)) for x, y, z in basis)
                couplings.append((vec, size, abs(H2M_EV_ANG2 * 4 * np.pi * f_e * phase / volume)))

    def shift(kv):
        # Two-beam branch shift per coupling: -d/2 + sign(d) sqrt(d^2/4 + U^2), d = E(k) - E(k+h).
        # Equals U^2/d far from a Bragg condition and is bounded by |U| at one.
        total = 0.0
        for v, s, u in couplings:
            d = -(H2M_EV_ANG2 / gamma) * (2 * kv @ v + s * s)
            total += -0.5 * d + np.sign(d) * np.sqrt(0.25 * d * d + u * u)
        return total

    out = {}
    for m in (2, 4):
        g = m * 2.0 * np.pi / c_ang * axis
        photon = float(recoil_resonance_eV(kinetic_eV, BEAM, g, normal))
        k_f = k_i - g - photon / HBARC_EV_ANG * normal
        out[m] = shift(k_i) - shift(k_f)
    largest = max(u for _, _, u in couplings)
    return out, largest


def main() -> int:
    data = json.loads(DATA.read_text())
    rows = table_comparison(data)
    print("Huang 2023 SI Table S4: measured vs straight exact vacuum root (eV)")
    print(
        f"{'set':7s} {'T keV':>5s} {'E2':>8s} {'cl':>7s} {'q':>7s} {'E4':>8s} {'cl':>7s} {'q':>7s}  2E2-E4 meas  q"
    )
    meas, pred = [], []
    for name, t, e2, e4, c2, q2, c4, q4 in rows:
        meas.append(2 * e2 - e4)
        pred.append(2 * q2 - q4)
        print(
            f"{name:7s} {t:5.1f} {e2:8.2f} {c2:7.1f} {q2:7.1f} {e4:8.2f} {c4:7.1f} {q4:7.1f}  {meas[-1]:11.2f} {pred[-1]:6.2f}"
        )
    meas, pred = np.array(meas), np.array(pred)
    resid = meas - pred
    sem = resid.std(ddof=1) / np.sqrt(resid.size)
    cal = data["detector"]["calibration_uncertainty_eV"]
    sigma = float(np.hypot(sem, cal))
    print(
        f"\nharmonic defect: measured {meas.mean():.2f} eV, exact recoil {pred.mean():.2f} eV, classical 0\n"
        f"residual {resid.mean():+.2f} +- {sigma:.2f} eV (sem {sem:.2f}, calibration {cal}); "
        f"classical excluded at {meas.mean() / sigma:.0f} sigma"
    )

    worst = 0.0
    print("\nin-medium corrections (eV)")
    for key, c_ang, t, th, ph, obs in (
        ("hopg", 6.71, 11.7e3, 12.45, 0.4, 130.0),
        ("hbn", 6.62, 14.0e3, 12.15, 78.8, 119.0),
    ):
        corr = medium_corrections(key, c_ang, t, th, ph, obs)
        for m, (e0, u0, e_mip, delta, e_phot, e_bend) in corr.items():
            print(
                f"{key} m={m}: line {e0:7.1f}; U0 {u0:+.2f} -> {e_mip:+.3f}; "
                f"delta {delta:.2e} -> {e_phot:+.3f}; entry bend -> {e_bend:+.3f}"
            )
        for index in (2, 4, 5):
            defect = 2 * corr[2][index] - corr[4][index]
            worst = max(worst, abs(defect))
    bloch, largest = graphite_bloch(11.7e3, 12.45, 0.4, 130.0)
    defect = 2 * bloch[2] - bloch[4]
    print(
        f"graphite Bloch, one arbitrary azimuth, not gated (largest |U_h| {largest:.2f} eV): "
        f"line m=2 {bloch[2]:+.3f}, m=4 {bloch[4]:+.3f}, defect {defect:+.3f}"
    )
    print(f"largest gated in-medium change of 2E2-E4: {worst:.3f} eV")

    ok = abs(resid.mean()) <= 2 * sigma and meas.mean() > 10 * sigma and worst < 1.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
