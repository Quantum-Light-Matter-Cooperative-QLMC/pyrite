"""Size of the soft inelastic angular deflection omitted by the shell mode.

The shell soft/hard mode deflects the primary only in hard inelastic
collisions (``W > W_c``). PENELOPE also folds the deflection of soft
inelastic collisions into its random hinge through the transport mean free
paths of the soft angular DCS (PENELOPE-2024 Eqs. 4.101-4.118). PyRITE has no
hinge: every elastic collision is simulated, and PXR/CBS read each segment's
velocity direction. This script sizes what the omission leaves out, from the
same closed shell model and closure the transport uses:

* ``1/lambda_in,1^(s)`` and ``1/lambda_in,2^(s)`` (Eqs. 4.115-4.116) from the
  soft distant longitudinal DCS (Eqs. 4.101-4.106, recoil density
  ``1/[Q(Q+2mc^2)]`` on ``[Q_-, Q'_k]``) and the soft close DCS (Eqs.
  4.107-4.112, ``Q = W``). Distant transverse losses do not deflect
  (footnote to Eq. 4.103). Each channel's weight is its closed soft
  ``sigma^(0)`` from ``catalog_shell_partition``, converted to a path rate
  with the transport's own normalization ``S_stp/sigma^(1)``;
* the same for the hard window, which the transport already samples;
* the elastic ``1/lambda_el,1`` of the default ``elastic_model="mott"``:
  ``sum_i n_i sigma_tr,i`` from the NIST SRD 64 transport cross sections
  (screened Rutherford with the Joy alpha for an element without a table, as
  transport does), and ``1/lambda_el`` from the Browning totals;
* the conduction-band share of the soft angular rate, since that channel
  carries the total-IMFP deficit of ``penelope-shell-rate-closure``.

Derived columns: ``soft/el`` is the fractional increase of the angular
diffusion rate (``<theta^2>`` per unit path) that the omission removes;
``theta_row`` is the small-angle rms soft deflection accumulated over one
elastic mean free path, ``sqrt(2 lambda_el/lambda_in,1^(s))``.

Validation: shell-soft-hard-transport

Run:
  uv run python checks/soft_inelastic_deflection.py
  uv run python checks/soft_inelastic_deflection.py --output REPORT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from pyrite.materials import CATALOG
from pyrite.montecarlo.transport.inelastic import _qmin_ev
from pyrite.montecarlo.transport.scattering import _load_mott_transport
from pyrite.montecarlo.transport.shell_gos import _MC2_EV, _PREF_EV_CM2, _kinematics
from pyrite.montecarlo.transport.shell_partition import catalog_shell_partition
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

MATERIALS = ("silicon", "sio2", "mos2")
ENERGIES_KEV = (5.0, 10.0, 20.0, 50.0, 100.0)
CUTOFFS_EV = (30.0, 50.0, 200.0)
TWO_MC2 = 2.0 * _MC2_EV


def _composition(key):
    if key in CATALOG.crystals:
        return CATALOG.crystal(key).composition
    return CATALOG.media[key].composition


def _cp(kinetic_eV):
    return np.sqrt(kinetic_eV * (kinetic_eV + TWO_MC2))


def _angular_weights(mu):
    """``(2 mu, 6 (mu - mu^2))``: the Eq. 4.115 and 4.116 integrand factors."""
    return np.array([2.0 * mu, 6.0 * (mu - mu * mu)])


def _distant_factors(energy_eV, osc):
    """Mean ``(2mu, 6(mu-mu^2))`` of a distant longitudinal event, or None.

    Recoil density ``1/[Q(Q+2mc^2)]`` on ``[Q_-, Q'_k]`` (Eqs. 3.83, 3.104);
    ``mu(Q)`` from Eq. 4.101 with ``p_k`` at the (modified) resonance, Eq. 4.102.
    """
    u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
    if u > 0.0:
        full = 3.0 * w - 2.0 * u
        w_mod, q_top = (
            (w, u) if energy_eV > full else ((energy_eV + 2.0 * u) / 3.0, u * energy_eV / full)
        )
    else:
        w_mod, q_top = w, w
    if not w_mod < energy_eV:
        return None
    q_low = float(_qmin_ev(energy_eV, w_mod))
    if not q_low < q_top:
        return None
    cp, cpk = _cp(energy_eV), _cp(energy_eV - w_mod)

    def mu_of(t):  # t = ln Q
        q = np.exp(t)
        return (q * (q + TWO_MC2) - (cp - cpk) ** 2) / (4.0 * cp * cpk)

    def density(t):  # dQ / [Q (Q + 2mc^2)] with dQ = Q dt
        return 1.0 / (np.exp(t) + TWO_MC2)

    a, b = np.log(q_low), np.log(q_top)
    norm = quad(density, a, b, epsabs=0.0, epsrel=1e-11, limit=200)[0]
    out = np.array(
        [
            quad(
                lambda t, j=j: density(t) * _angular_weights(mu_of(t))[j],
                a,
                b,
                epsabs=0.0,
                epsrel=1e-10,
                limit=200,
            )[0]
            for j in range(2)
        ]
    )
    return out / norm


def _close_moments(energy_eV, osc, lower, upper):
    """``pref f int F^(-)/W^2 * (1, 2mu, 6(mu-mu^2)) dW`` over ``[lower, upper]``.

    Eqs. 3.86-3.87 with ``E' = E + U``; ``mu(W)`` from Eq. 3.134 (``Q = W``).
    """
    if not lower < upper:
        return np.zeros(3)
    _, beta2 = _kinematics(energy_eV)
    pref = _PREF_EV_CM2 / beta2 * osc.strength
    a = (energy_eV / (energy_eV + _MC2_EV)) ** 2
    prime = energy_eV + osc.ionization_energy_eV

    def dcs(w):
        x = w / (prime - w)
        return (1.0 + x * x - (1.0 - a) * x + a * (w / prime) ** 2) / (w * w)

    def mu(w):
        rest = energy_eV - w
        cos2 = rest / energy_eV * (energy_eV + TWO_MC2) / (rest + TWO_MC2)
        return 0.5 * (1.0 - np.sqrt(cos2))

    def integral(fn):
        # ln W substitution: the 1/W^2 weight spans many decades.
        return quad(
            lambda t: fn(np.exp(t)) * np.exp(t),
            np.log(lower),
            np.log(upper),
            epsabs=0.0,
            epsrel=1e-10,
            limit=400,
        )[0]

    return pref * np.array(
        [
            integral(dcs),
            integral(lambda w: dcs(w) * 2.0 * mu(w)),
            integral(lambda w: dcs(w) * 6.0 * (mu(w) - mu(w) ** 2)),
        ]
    )


def inelastic_transport_rates(key, energy_eV, cutoff_eV):
    """Soft and hard ``(1/lambda_1, 1/lambda_2)`` [1/Angstrom] and diagnostics."""
    material = catalog_shell_oscillators(key)
    part = catalog_shell_partition(key, energy_eV, cutoff_eV)
    scale = part.closure.scale
    table = resolve_catalog_table(key).arrays()
    stopping = float(
        np.exp(
            np.interp(
                np.log(energy_eV),
                np.log(table["stopping_energy_eV"]),
                np.log(table["stopping_eV_per_angstrom"]),
            )
        )
    )
    total1 = part.soft.total[1] + part.hard.total[1]
    per_ang = stopping / total1  # 1/(Angstrom cm^2), the transport's normalization
    rates = {"soft": np.zeros(2), "hard": np.zeros(2)}
    band_soft = np.zeros(2)
    close_check = []
    for i, osc in enumerate(material.oscillators):
        factors = _distant_factors(energy_eV, osc)
        u = osc.ionization_energy_eV
        w_max = 0.5 * (energy_eV + u)
        q_close = u if u > 0.0 else osc.resonance_energy_eV
        for window, moments, lo, hi in (
            ("soft", part.soft, q_close, min(w_max, cutoff_eV)),
            ("hard", part.hard, max(q_close, cutoff_eV), w_max),
        ):
            add = np.zeros(2)
            if factors is not None:
                add += moments.distant_longitudinal[i, 0] * factors
            close = _close_moments(energy_eV, osc, lo, hi) * scale[i]
            if moments.close[i, 0] > 0.0:
                close_check.append(close[0] / moments.close[i, 0] - 1.0)
            add += close[1:]
            rates[window] += add * per_ang
            if window == "soft" and osc.atomic_number == 0:
                band_soft += add * per_ang
    return {
        "soft": rates["soft"],
        "hard": rates["hard"],
        "band_soft_fraction": float(band_soft[0] / rates["soft"][0]) if rates["soft"][0] else 0.0,
        "close_sigma0_max_rel_error": float(np.max(np.abs(close_check))) if close_check else 0.0,
    }


def elastic_rates(key, energy_eV):
    """``(1/lambda_el, 1/lambda_el,1)`` [1/Angstrom] of the default Mott model."""
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.montecarlo.transport.scattering import _alpha_sr_joy, _sigma_browning_cm2

    total = transport = 0.0
    for element, n_per_ang3 in _composition(key):
        z = float(TRANSPORT_ELEMENTS[element]["Z"])
        n_cm3 = n_per_ang3 * 1e24
        sigma = float(_sigma_browning_cm2(z, np.array([energy_eV / 1e3]))[0])
        try:
            e_tab, sig_tr = _load_mott_transport(element)
            tr = float(np.exp(np.interp(np.log(energy_eV), np.log(e_tab), np.log(sig_tr))))
        except FileNotFoundError:
            # Transport falls back to screened-Rutherford angles with the Joy
            # alpha on the Browning total: <1-cos> = 2a[(1+a)ln(1+1/a) - 1].
            alpha = _alpha_sr_joy(z, energy_eV / 1e3)
            tr = sigma * 2.0 * alpha * ((1.0 + alpha) * np.log1p(1.0 / alpha) - 1.0)
        total += n_cm3 * sigma
        transport += n_cm3 * tr
    return total * 1e-8, transport * 1e-8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = []
    header = (
        f"{'material':8} {'E keV':>6} {'W_c':>5} {'lam_el A':>9} {'lam_el1 A':>10} "
        f"{'lam_s1 A':>10} {'soft/el':>8} {'hard/el':>8} {'band%':>6} {'th_row mrad':>11}"
    )
    print(header)
    for key in MATERIALS:
        for energy_keV in ENERGIES_KEV:
            inv_el, inv_el1 = elastic_rates(key, energy_keV * 1e3)
            for cutoff in CUTOFFS_EV:
                r = inelastic_transport_rates(key, energy_keV * 1e3, cutoff)
                soft1, hard1 = r["soft"][0], r["hard"][0]
                theta_row = np.sqrt(2.0 * soft1 / inv_el)
                row = {
                    "material": key,
                    "E_keV": energy_keV,
                    "cutoff_eV": cutoff,
                    "lambda_el_ang": 1.0 / inv_el,
                    "lambda_el1_ang": 1.0 / inv_el1,
                    "lambda_in1_soft_ang": 1.0 / soft1,
                    "lambda_in2_soft_ang": 1.0 / r["soft"][1],
                    "lambda_in1_hard_ang": 1.0 / hard1 if hard1 else None,
                    "soft_over_elastic": soft1 / inv_el1,
                    "hard_over_elastic": hard1 / inv_el1,
                    "band_share_of_soft": r["band_soft_fraction"],
                    "theta_row_rms_mrad": 1e3 * theta_row,
                    "close_sigma0_max_rel_error": r["close_sigma0_max_rel_error"],
                }
                rows.append(row)
                print(
                    f"{key:8} {energy_keV:6.0f} {cutoff:5.0f} {row['lambda_el_ang']:9.1f} "
                    f"{row['lambda_el1_ang']:10.1f} {row['lambda_in1_soft_ang']:10.3g} "
                    f"{row['soft_over_elastic']:8.4f} {row['hard_over_elastic']:8.4f} "
                    f"{100 * row['band_share_of_soft']:6.1f} {row['theta_row_rms_mrad']:11.3f}"
                )
    worst = max(r["close_sigma0_max_rel_error"] for r in rows)
    print(f"close sigma^(0) quadrature vs shell_gos: max rel error {worst:.2e}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
