"""Regime tables for longitudinal coherence and recoil in channeling radiation (#629).

Prints the numbers quoted in
``docs/research/physics/longitudinal-coherence-recoil.md``:

1. PINEM Talbot lengths and the quantum harmonic-bunching cap: coherent
   (pair-term) emission at photon energy ``hbar*omega`` needs the
   single-electron coherent comb to span ``hbar*omega``; the closed form
   ``|J_l(4|g| sin(2 pi l z/z_T))|`` is maximized over drift ``z`` and compared
   with the classical ``max_X |J_l(l X)|``.
2. The single-electron pair factor ``exp(-(hbar omega)^2/(8 sigma^2))`` of a
   transform-limited Gaussian wavepacket whose coherent rms energy width is
   bounded by an ordinary beam's total energy spread.
3. Recoil-exact planar channeling lines for the lowest bound-bound transition
   at ``k = 0`` in Si(110) and diamond(110), compared with the no-recoil
   Doppler line ``cr-doppler-line-energy``. Levels keep the solver's
   ``U_0 = 0`` reference; a vacuum reference moves each shift by <= 0.5 %.

    PYRITE_MC_BACKEND=cpu uv run python checks/longitudinal_coherence_regimes.py

Exit 0 when the internal limits hold (forward fixed-level root equals the
Baier--Katkov form to 1e-5; quantum bunching never exceeds the classical
maximum). Under a second.

Validation: longitudinal-quantum-bunching, cr-recoil-line-energy
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import jv

from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.channeling.bloch import (
    HBAR2_2M_EV_ANG2,
    line_energy_eV,
    longitudinal_kinematics,
    transverse_states,
)
from pyrite.quantum import _longitudinal as lc


def _talbot_table() -> None:
    print("PINEM Talbot length z_T (hbar*Omega = 1.55 eV)")
    for kinetic in (30e3, 200e3, 300e3, 5e6):
        print(
            f"  {kinetic / 1e3:>7.0f} keV  z_T = {lc.talbot_length_ang(kinetic, 1.55) * 1e-7:.4g} mm"
        )


def _bunching_cap() -> bool:
    ok = True
    print("\nMax harmonic bunching |b_l| over drift (hbar*Omega = 1.55 eV)")
    print("  photon    l     |g|=10     |g|=100    |g|=300   classical max_X|J_l(lX)|")
    for photon in (15.5, 50.0, 277.0, 1500.0):
        ell = round(photon / 1.55)
        x = np.linspace(0.0, 3.0 * ell, 60001)
        classical = float(np.max(np.abs(jv(ell, x))))
        row = []
        for coupling in (10.0, 100.0, 300.0):
            # |sin| <= 1: the drift maximum of J_l(4|g| sin y) is max over x <= 4|g|.
            quantum = float(np.max(np.abs(jv(ell, x[x <= 4 * coupling]))))
            ok &= quantum <= classical * (1 + 1e-9) or 4 * coupling > ell
            row.append(quantum)
        print(
            f"  {photon:>6.1f} {ell:>5d}  "
            + "  ".join(f"{v:9.2e}" if v > 0 else "  <1e-300" for v in row)
            + f"   {classical:9.3f}"
        )
    return ok


def _ordinary_beam_factor() -> None:
    print(
        "\nSingle-electron pair factor |F1|^2 = exp(-(hbar w)^2/(4 sigma^2)), sigma <= total spread"
    )
    for sigma in (0.3, 0.7, 1.5):
        values = [np.exp(-(photon**2) / (4 * sigma**2)) for photon in (1.0, 5.0, 50.0)]
        print(
            f"  sigma={sigma:3.1f} eV  1 eV: {values[0]:.2e}  5 eV: {values[1]:.2e}  50 eV: {values[2]:.1e}"
        )


def _mean_transverse_kinetic(states, index: int) -> float:
    q = states.k_inv_ang + states.orders * states.profile.g_inv_ang
    weights = np.abs(states.coefficients[:, index]) ** 2
    return float(np.sum(weights * HBAR2_2M_EV_ANG2 / states.gamma * q**2))


def _channeling_table() -> bool:
    ok = True
    print("\nRecoil-exact lowest bound-bound CR line at k = 0 (forward unless noted)")
    print(
        "  crystal        T[MeV]  dE[eV]  <T>_f  w_D[keV]   exact/w_D-1   BK-only   "
        "<T>_f share  aniso(1/gamma)"
    )
    for crystal in ("silicon", "diamond"):
        profile = planar_profile(crystal, (1, 1, 0), 293.15)
        for kinetic in (5e6, 20e6, 54e6):
            gamma, beta = longitudinal_kinematics(kinetic)
            states = transverse_states(profile, gamma, 0.0)
            bound = np.flatnonzero(states.energies_eV < states.barrier_eV)
            if bound.size < 2:
                print(f"  {crystal}(110) {kinetic / 1e6:6.0f}  <2 bound states")
                continue
            eps_f, eps_i = states.energies_eV[bound[0]], states.energies_eV[bound[1]]
            t_f = _mean_transverse_kinetic(states, int(bound[0]))
            total = kinetic + M_E_EV
            doppler = float(line_energy_eV(eps_i - eps_f, kinetic, 0.0))
            exact = float(lc.channeling_recoil_line_energy_eV(kinetic, eps_i, eps_f, t_f, 0.0, 0.0))
            fixed = float(lc.channeling_recoil_line_energy_eV(kinetic, eps_i, eps_f, 0.0, 0.0, 0.0))
            ok &= abs(fixed / (doppler / (1 + doppler / total)) - 1) < 1e-5
            theta = 1.0 / gamma
            normal = lc.channeling_recoil_line_energy_eV(kinetic, eps_i, eps_f, t_f, theta, 0.0)
            in_plane = lc.channeling_recoil_line_energy_eV(
                kinetic, eps_i, eps_f, t_f, theta, np.pi / 2
            )
            shift, bk = exact / doppler - 1, fixed / doppler - 1
            print(
                f"  {crystal + '(110)':<14} {kinetic / 1e6:6.0f}  {eps_i - eps_f:6.2f}  {t_f:5.2f}"
                f"  {doppler / 1e3:8.3f}   {shift:11.3e}  {bk:9.2e}   {(shift - bk) / shift:9.2f}"
                f"   {float(normal / in_plane - 1):11.2e}"
            )
    return ok


def main() -> int:
    _talbot_table()
    ok = _bunching_cap()
    _ordinary_beam_factor()
    ok &= _channeling_table()
    print("\nlimits:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
