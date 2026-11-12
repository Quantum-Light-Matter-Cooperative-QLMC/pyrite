"""Independent per-reflection Shi 2024 matrix elements against PyRITE (#1103).

Evaluates the plane-wave PXR and CB elements of Shi et al., Nat. Commun. 15,
7803 (2024), SI Eqs. (S12)-(S17), one reflection at a time, and compares them
with PyRITE's ``A_PXR``/``A_CBS`` on Shi's Fig. 2 geometry (straight 30 nm Si
[001], theta = 100 deg). Convention q_i + g = q_f + k, g = n 2 pi/a (PyRITE
rows are -n). Units eV and Angstrom. With the common factor
-e sqrt(hbar / 2 eps0 omega V) removed, second-order perturbation theory gives

  M_P = (j_fi/c) . [k^2 e - K (K . e)] chi_g / (K^2 - k^2),   K = k - g
  M_C = e phi_g [ (j_f1/c) . e / (E_i - E_1) + (j_2i/c) . e / (E_f - E_2) ]

with intermediate momenta q_1 = q_i + g, q_2 = q_i - k. ``current="gordon"``
uses the convective transition current j_{q'q}/c = hbar c (q + q')/(E + E');
``current="shi"`` uses the paper's stated u_q ~ u_qi, j/c = beta_i z.
Shi's inputs: Salvat Yukawa potential and form factor (SI Eq. S3), no
anomalous dispersion, no Debye-Waller. Line height per row is
omega sum_sigma |M|^2 up to one constant per electron energy, fixed by
calibrating against PyRITE with PyRITE's own couplings.

Checks (fail -> exit 1):
1. Classical limit: at 5 MeV the model with PyRITE's |chi_g|, |U_g| matches
   PyRITE row by row (PXR scale spread <= 5 %, CB/PXR and interference/PXR
   ratios within 5 % of PyRITE).
2. Attribution (n_z = -2, axis-unambiguous): with Shi's inputs, Shi's CL
   peak matches the model's PXR-only classical rate (0.6-1.5) and not the
   PXR+CB one; Shi's QM/CL area ratio matches the model's
   (PXR+CB, exact roots) / (PXR-only, classical) ratio within 1.5x.

Run:  uv run python checks/shi_2024_matrix_elements_1103.py

Validation: shi-2024-fig2-intensity-attribution
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pyrite.materials.crystal import U_g, chi_g
from pyrite.montecarlo.spectrum import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import _recoil

ROOT = Path(__file__).resolve().parents[1]
SHI = json.loads((ROOT / "tests/data/recoil/shi-2024-fig2.json").read_text())["curves"]
_spec = importlib.util.spec_from_file_location("tr527", ROOT / "checks/transverse_recoil_527.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

HBARC = 1973.269804  # eV Angstrom
MC2 = 510998.95
ALPHA = 1 / 137.035999084
R_E = 2.8179403262e-5  # Angstrom
A_BOHR = 0.529177210903
A_SI = 5.431
Z_SI = 14
YUKAWA_C = (0.5160, 0.4840)  # Shi SI S1 (Salvat 1987)
YUKAWA_MU = (5.8492 / A_BOHR, 1.1732 / A_BOHR)
BASIS = A_SI * np.array(
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
)
V_CELL = A_SI**3
N_HAT = tr._photon(100.0)
E1_POL = np.array([0.0, 1.0, 0.0])
E2_POL = np.cross(N_HAT, E1_POL) / np.linalg.norm(np.cross(N_HAT, E1_POL))
HB = _recoil._RECOIL_HBARC_EV_ANG


def _energy(q):
    return np.sqrt((HBARC * np.linalg.norm(q)) ** 2 + MC2**2)


def _yukawa(g2):
    """Atomic form factor and |e| phi_atom [eV Angstrom^3] of the Yukawa fit."""
    f = Z_SI * sum(c * m * m / (m * m + g2) for c, m in zip(YUKAWA_C, YUKAWA_MU, strict=True))
    phi = (
        4
        * np.pi
        * ALPHA
        * HBARC
        * Z_SI
        * sum(c / (m * m + g2) for c, m in zip(YUKAWA_C, YUKAWA_MU, strict=True))
    )
    return f, phi


def shi_row(T_eV, n, current="gordon", classical=False, couplings=None):
    """Return per-row ``omega |M|^2`` components (P, C, I, total) and ``hbar omega``."""
    E_i = T_eV + MC2
    q_i = np.array([0.0, 0.0, np.sqrt(E_i**2 - MC2**2) / HBARC])
    beta = HBARC * q_i / E_i
    g = np.asarray(n, float) * 2 * np.pi / A_SI
    if classical:
        hw = -HBARC * (beta @ g) / (1.0 - beta @ N_HAT)
    else:
        mismatch = lambda w: _energy(q_i + g - w / HBARC * N_HAT) + w - E_i  # noqa: E731
        if mismatch(1e-6) * mismatch(0.999999 * T_eV) > 0:
            return None
        hw = brentq(mismatch, 1e-6, 0.999999 * T_eV, xtol=1e-10)
    if hw <= 0:
        return None
    k = hw / HBARC * N_HAT
    q_f = q_i + g - k
    phase = np.exp(-1j * BASIS @ g).sum()
    f, phi = _yukawa(g @ g)
    chi = -R_E * (2 * np.pi * HBARC / hw) ** 2 / (np.pi * V_CELL) * f * phase
    e_phi = -phi * phase / V_CELL  # electron charge e < 0
    if couplings is not None:  # external magnitudes, model phases
        chi_abs, u_abs = couplings(n, hw)
        chi, e_phi = chi / abs(chi) * chi_abs, e_phi / abs(e_phi) * u_abs
    # classical virtual-photon momentum has K_z = omega/v; quantum K = q_i - q_f
    K = np.array([k[0] - g[0], k[1] - g[1], hw / HBARC / beta[2]]) if classical else q_i - q_f
    q_1, q_2 = q_i + g, q_i - k
    E_f, E_1, E_2 = _energy(q_f), _energy(q_1), _energy(q_2)
    if current == "shi":
        j_fi = j_f1 = j_2i = beta
    else:
        j_fi = HBARC * (q_i + q_f) / (E_i + E_f)
        j_f1 = HBARC * (q_1 + q_f) / (E_1 + E_f)
        j_2i = HBARC * (q_i + q_2) / (E_i + E_2)
    kk = k @ k
    P = C = X = 0.0
    for e in (E1_POL, E2_POL):
        m_p = (j_fi @ (kk * e - K * (K @ e))) * chi / (K @ K - kk)
        m_c = e_phi * ((j_f1 @ e) / (E_i - E_1) + (j_2i @ e) / (E_f - E_2))
        P, C, X = P + abs(m_p) ** 2, C + abs(m_c) ** 2, X + 2 * (m_p * np.conj(m_c)).real
    return dict(hw=hw, P=hw * P, C=hw * C, I=hw * X, T=hw * (P + C + X))


def _pyrite(T_keV, rows, grid, classical, components=True):
    _recoil._RECOIL_HBARC_EV_ANG = 0.0 if classical else HB
    try:
        out = mc_cxr_spectrum(
            tr._straight(T_keV, 300.0),
            grid,
            "silicon",
            rows,
            n_hat=N_HAT,
            beam_uvw=(0, 0, 1),
            B_ang2=tr.B_SI,
            components=components,
        )
    finally:
        _recoil._RECOIL_HBARC_EV_ANG = HB
    return [np.asarray(a) for a in out] if components else np.asarray(out)


def _pyrite_couplings(n, hw):
    hkl = tuple(-x for x in n)
    return abs(chi_g("silicon", hkl, hw, tr.B_SI, True)), abs(U_g("silicon", hkl, hw, tr.B_SI))


def calibrate(T_keV, nz):
    """Per-row model / PyRITE ratios of P, C, I under the classical hook."""
    ratios = []
    for r in tr._si_rows(nz, 4):
        n = tuple(-x for x in r)
        m = shi_row(T_keV * 1e3, n, "gordon", classical=True, couplings=_pyrite_couplings)
        if m is None:
            continue
        _, root, _ = tr._roots("silicon", [r], T_keV, N_HAT)
        grid = np.arange(root[0] - 10.0, root[0] + 10.0, 0.01)
        t, p, c = _pyrite(T_keV, [r], grid, classical=True)
        i = t.argmax()
        if p[i] <= 0 or c[i] <= 0:
            continue
        ratios.append((m["P"] / p[i], m["C"] / c[i], m["I"] / (t[i] - p[i] - c[i])))
    return np.array(ratios)


def _model_lines(T_keV, nz, scale, current, classical, part):
    out = []
    for r in tr._si_rows(nz, 8):
        m = shi_row(T_keV * 1e3, tuple(-x for x in r), current, classical)
        if m is not None:
            out.append((m["hw"], m[part] / scale))
    return np.array(out)


def main():
    failures = []
    print("1. Classical limit: model with PyRITE couplings / PyRITE, per row (|h|,|k| <= 4)")
    scale = {}
    for T in (30.0, 300.0, 5000.0):
        for nz in (2, 3):
            r = calibrate(T, nz)
            scale[(T, nz)] = float(np.median(r[:, 0]))
            c_rel, i_rel = r[:, 1] / r[:, 0], r[:, 2] / r[:, 0]
            spread = r[:, 0].max() / r[:, 0].min()
            print(
                f"  T={T:6.0f} keV n_z=-{nz}: scale {scale[(T, nz)]:.4e} (spread {spread:.3f}); "
                f"CB/PXR model/PyRITE {np.median(c_rel):.3f} [{c_rel.min():.3f}, {c_rel.max():.3f}]; "
                f"interference {np.median(i_rel):.3f} [{i_rel.min():.3f}, {i_rel.max():.3f}]"
            )
            if T == 5000.0 and (
                spread > 1.05 or np.abs(c_rel - 1).max() > 0.05 or np.abs(i_rel - 1).max() > 0.05
            ):
                failures.append(f"5 MeV n_z=-{nz} classical-limit mismatch")
    print(
        "   (sub-MeV CB/PXR > 1: the model keeps exact quantum CB denominators and transition currents)"
    )

    print("\n2. Shi's inputs (Yukawa, no f', no Debye-Waller) against Shi Fig. 2 (digitized)")
    print(
        "  T, n_z | Shi CL / model CL PXR-only, PXR+CB | Shi QM/CL area | model QM(PXR+CB)/CL(PXR) gordon, shi-current | QM/CL, CB on both sides"
    )
    for T in (30.0, 300.0, 5000.0):
        for nz in (2, 3):
            s = scale[(T, nz)]
            cl_p = _model_lines(T, nz, s, "gordon", True, "P")[:, 1].sum()
            cl_t = _model_lines(T, nz, s, "gordon", True, "T")[:, 1].sum()
            qm_g = _model_lines(T, nz, s, "gordon", False, "T")[:, 1].sum()
            qm_s = _model_lines(T, nz, s, "shi", False, "T")[:, 1].sum()
            cl, qm = SHI[f"{T:g} keV CL nz=-{nz}"], SHI[f"{T:g} keV QM nz=-{nz}"]
            shi_ratio = qm["area_per_sr"] / cl["area_per_sr"]
            print(
                f"  {T:6.0f} -{nz} | {cl['peak'] / cl_p:.2f}, {cl['peak'] / cl_t:.2f} | {shi_ratio:.2f} | "
                f"{qm_g / cl_p:.2f}, {qm_s / cl_p:.2f} | {qm_g / cl_t:.2f}"
            )
            if nz == 2 and T < 5000.0:
                if not 0.6 <= cl["peak"] / cl_p <= 1.5 or cl["peak"] / cl_t > 0.5:
                    failures.append(f"{T} keV: Shi CL is not the PXR-only classical rate")
                if not 1 / 1.5 <= (qm_g / cl_p) / shi_ratio <= 1.5:
                    failures.append(f"{T} keV: QM/CL area ratio not reproduced")

    print(
        "\n3. PyRITE production: exact-recoil PXR+CBS area / classical-hook PXR-only area (|h|,|k| <= 8)"
    )
    for T in (30.0, 300.0, 5000.0):
        for nz in (2, 3):
            cl, qm = SHI[f"{T:g} keV CL nz=-{nz}"], SHI[f"{T:g} keV QM nz=-{nz}"]
            grid = np.arange(
                min(cl["E_eV"][0], qm["E_eV"][0]), max(cl["E_eV"][-1], qm["E_eV"][-1]), 0.5
            )
            rows = tr._si_rows(nz, 8)
            _, pxr, _ = _pyrite(T, rows, grid, classical=True)
            total = _pyrite(T, rows, grid, classical=False, components=False)
            ratio = total.sum() / pxr.sum()
            shi_ratio = qm["area_per_sr"] / cl["area_per_sr"]
            print(
                f"  {T:6.0f} -{nz}: PyRITE {ratio:.2f}, Shi {shi_ratio:.2f}, PyRITE/Shi {ratio / shi_ratio:.2f}"
            )

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print(
        "\nPASS: classical limit reproduced; Shi Fig. 2 CL is the PXR-only classical rate, QM carries CB"
    )


if __name__ == "__main__":
    main()
