"""Exact second-order PXR/CB elements in production, and their size (#1134).

Production ``A_PXR``/``A_CBS`` use ``_quantum_line_factors``. This is a
closed form of Shi et al., Nat. Commun. 15, 7803 (2024), SI Eqs.
(S12)-(S15). It keeps the exact intermediate-state denominators and the
convective transition currents ``(p + p')/(E + E')``. This check evaluates
those elements independently, as momentum vectors and on-shell energies, and
confirms that the closed form is algebraically the same. It then sizes the
correction on line yields against the Debye-Waller and f' sensitivity.
Units: eV and Angstrom.

Checks (fail -> exit 1):
1. Closed form = independent vector elements: relative error <= 1e-8 over
   random kinematics at 10 keV-1 GeV, both elements and both orderings.
2. hbar -> 0 identity: h = 0 reproduces the classical production elements
   ``v.(k+g)(g.e+k.e) - k^2 v.e`` and
   ``-({g;e} + (v.e){k;g}/(v.g))/(gamma v.g)`` to <= 1e-12.
3. Size: summed Si (Shi Fig. 2 geometry, 30 nm [001], theta = 100 deg) line
   yield at the exact root, quantum over classical elements, exceeds the
   +-10 % B and +-10 % f' sensitivity (in quadrature) for 30 keV n_z = 2 and
   10 keV n_z = 3, and is below 0.5 % at 5 MeV. Other sub-100 keV Si groups
   are reported; some fall below that floor. The HOPG/hBN (002)/(004) lines at the Huang 2023
   geometries are reported (about -2 %, comparable to that floor).

Run:  uv run python checks/quantum_cb_elements_1134.py

Validation: cbs-amplitude
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import pyrite.materials.atomic as atomic
from pyrite.materials.crystal import CRYSTALS, U_g, chi_g, reciprocal_g_vector
from pyrite.montecarlo.spectrum.cxr._recoil import _quantum_line_factors

ROOT = Path(__file__).resolve().parents[1]
HBARC = 1973.269804
MC2 = 510998.95
H_RED = HBARC / MC2
B_CAT = {"silicon": 0.4632, "hopg": 1.30, "hbn": 3.45}


def _energy(p):
    return np.sqrt(p @ p + MC2**2)


def _shift(p, d, e_p, e_pd):
    """E(p + d) - E(p) without cancellation."""
    return (2 * (p @ d) + d @ d) / (e_pd + e_p)


def vector_elements(gamma, v, g, k, e, hbarc=HBARC):
    """(PXR numerator, f_cbs) from momenta; production sign, lattice transfer -hbar g."""
    p_i = gamma * MC2 * v
    a, b = hbarc * g, hbarc * k
    p_1, p_2, p_f = p_i - a, p_i - b, p_i - a - b
    e_i, e_1, e_2, e_f = (_energy(p) for p in (p_i, p_1, p_2, p_f))
    j_fi = (p_i + p_f) / (e_i + e_f)
    j_f1 = (p_1 + p_f) / (e_1 + e_f)
    j_2i = (p_i + p_2) / (e_i + e_2)
    kg = k + g
    pxr = (j_fi @ kg) * (kg @ e) - (k @ k) * (j_fi @ e)
    cbs = MC2 * (-(j_f1 @ e) / _shift(p_i, -a, e_i, e_1) + (j_2i @ e) / _shift(p_2, -a, e_2, e_f))
    return pxr, cbs


def _factors(h, gamma, v, g, k, e):
    return _quantum_line_factors(h, gamma, v @ g, k @ v, k @ g, g @ g, k @ k, v @ e, g @ e, k @ e)


def check_closed_form(rng):
    worst = 0.0
    for T in (1e4, 3e4, 1e5, 1e6, 1e9):
        gamma = 1 + T / MC2
        beta = np.sqrt(1 - 1 / gamma**2)
        for _ in range(200):
            v = rng.normal(size=3)
            v *= beta / np.linalg.norm(v)
            g, n = rng.normal(size=3) * 3, rng.normal(size=3)
            n /= np.linalg.norm(n)
            k = (0.05 + rng.random()) * n
            e = np.cross(n, rng.normal(size=3))
            e /= np.linalg.norm(e)
            ref = vector_elements(gamma, v, g, k, e)
            got = _factors(H_RED, gamma, v, g, k, e)
            scale = [abs(x) for x in _factors(0.0, gamma, v, g, k, e)]
            worst = max(worst, *(abs(a - b) / s for a, b, s in zip(got, ref, scale, strict=True)))
    print(f"1. closed form vs vector elements: max relative error {worst:.1e}")
    return worst <= 1e-8  # the vector reference itself cancels to ~1e-9 at 1 GeV


def check_classical(rng):
    worst = 0.0
    for _ in range(500):
        gamma = 1 + 10 * rng.random()
        v = rng.normal(size=3)
        v *= np.sqrt(1 - 1 / gamma**2) / np.linalg.norm(v)
        g, k, e = (rng.normal(size=3) for _ in range(3))
        vdg, kv, kg = v @ g, k @ v, k @ g
        pxr0 = (vdg + kv) * (g @ e + k @ e) - (k @ k) * (v @ e)
        cbs0 = -((g @ e - vdg * (v @ e)) + (v @ e) * (kg - kv * vdg) / vdg) / (gamma * vdg)
        pxr, cbs = _factors(0.0, gamma, v, g, k, e)
        worst = max(worst, abs(pxr / pxr0 - 1), abs(cbs / cbs0 - 1))
    print(f"2. hbar -> 0 identity: max relative deviation {worst:.1e}")
    return worst <= 1e-12


def _photon(theta_deg):
    th = np.radians(theta_deg)
    return np.array([np.sin(th), 0.0, np.cos(th)])


def _rotation(theta_deg, phi_deg):
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    c = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    axis = np.cross([0.0, 0.0, 1.0], c)
    s = np.linalg.norm(axis)
    if s < 1e-14:
        return np.eye(3)
    x = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + x + x @ x * ((1 - c[2]) / s**2)


def line_yield(crystal, hkl, T, n_hat, rot, B):
    """(omega sum_pol |A|^2) with quantum and with classical elements at the exact root."""
    g = rot @ reciprocal_g_vector(hkl, CRYSTALS[crystal]["lattice"])[0]
    gamma = 1 + T / MC2
    v = np.array([0.0, 0.0, np.sqrt(1 - 1 / gamma**2)])
    p_i = gamma * MC2 * v

    def mismatch(w):
        return _energy(p_i - HBARC * g - w * n_hat) + w - gamma * MC2

    if mismatch(1e-6) * mismatch(0.999999 * T) > 0:
        return None
    hw = brentq(mismatch, 1e-6, 0.999999 * T, xtol=1e-10)
    k = hw / HBARC * n_hat
    chi = complex(chi_g(crystal, hkl, hw, B, True))
    u = complex(U_g(crystal, hkl, hw, B)) / MC2
    e1 = np.cross(n_hat, v) / np.linalg.norm(np.cross(n_hat, v))
    det = (k + g) @ (k + g) - k @ k
    out = []
    for h in (H_RED, 0.0):
        y = 0.0
        for e in (e1, np.cross(n_hat, e1)):
            pxr, cbs = _factors(h, gamma, v, g, k, e)
            y += abs(chi * pxr / det + u * cbs) ** 2
        out.append(hw * y)
    return out


def _si_rows(nz, hmax=8):
    rows = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            par = {h % 2, k % 2, nz % 2}
            if len(par) == 1 and (par == {1} or (h + k + nz) % 4 == 0):
                rows.append((h, k, nz))
    return rows


def _yields(case, B, f_scale):
    orig = atomic.henke_dispersion

    def scaled(el, E, *a, **kw):
        fp, fpp = orig(el, E, *a, **kw)
        return np.asarray(fp) * f_scale, fpp

    atomic.henke_dispersion = scaled
    try:
        crystal, T, rows, n_hat, rot = case
        ys = [line_yield(crystal, r, T, n_hat, rot, B) for r in rows]
    finally:
        atomic.henke_dispersion = orig
    ys = np.array([y for y in ys if y is not None and y[1] > 0])
    return ys[:, 0].sum(), ys[:, 1].sum()


def check_size():
    huang = json.loads((ROOT / "tests/data/recoil/huang-2023-table-s4.json").read_text())
    cases = {
        f"Si {T / 1e3:g} keV n_z={nz}": ("silicon", T, _si_rows(nz), _photon(100.0), np.eye(3))
        for T in (1e4, 3e4, 1e5, 5e6)
        for nz in (2, 3)
    }
    for idx in (3, 11):  # graphite 11.7 keV 130 deg; hBN 14.0 keV 119 deg
        s, T, _, _, tilt, phi = huang["rows"][idx]
        st = huang["sets"][s]
        crystal = "hopg" if st["material"] == "graphite" else "hbn"
        rows = [(0, 0, 2), (0, 0, 4)]
        cases[f"{crystal} {T} keV (002)+(004)"] = (
            crystal,
            T * 1e3,
            rows,
            _photon(st["theta_obs_deg"]),
            _rotation(tilt, phi),
        )
    ok = True
    print("3. line yield: quantum/classical elements vs +-10 % B and f' sensitivity")
    for label, case in cases.items():
        B = B_CAT[case[0]]
        qm, cl = _yields(case, B, 1.0)
        effect = qm / cl - 1
        db = _yields(case, 1.1 * B, 1.0)[1] / cl - 1
        df = _yields(case, B, 1.1)[1] / cl - 1
        floor = np.hypot(db, df)
        print(f"  {label}: {effect:+.4f}  (B {db:+.4f}, f' {df:+.4f}, floor {floor:.4f})")
        if label in ("Si 30 keV n_z=2", "Si 10 keV n_z=3"):
            ok &= abs(effect) > floor
        if label.startswith("Si 5000 "):
            ok &= abs(effect) < 5e-3
    return ok


def main():
    rng = np.random.default_rng(1134)
    results = [check_closed_form(rng), check_classical(rng), check_size()]
    if not all(results):
        print("FAIL")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
