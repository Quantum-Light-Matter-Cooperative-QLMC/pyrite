"""Creation-level bremsstrahlung emission angles, PyRITE against Geant4 (#341).

Usage:
  emission_angles.py pyrite CASE NE FIRST_BATCH OUT_NPZ
  emission_angles.py compare CASE GEANT4_LOG.gz PYRITE.npz [PYRITE.npz ...]

Diagnostic for the source difference reported by the thick-target check: it
compares the angle between each hard photon and its parent electron at
creation, conditioned on the pre-emission electron energy T and on k/T, so
transport only enters through which (T, k) are populated. Geant4 lines are the
#182 ``EBREM`` records of the 10^5-event pilot logs (post-step electron
direction, which PenBrem leaves unchanged; T = T_after + k). PyRITE uses the
realized hard photons (k > 1 keV) of the ``pyrite_bench`` electron stack.
Report only; no gate.
"""

import gzip
import sys

import numpy as np
from cases import CASES, MM

from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.events import EVENT_HARD_RADIATIVE

T_BINS = (0.1, 0.3, 0.6, 1.0)  # fractions of E0
K_BINS = (0.05, 0.2, 0.5, 0.9)  # k / T


def run_pyrite(case, ne, first_batch, out):
    from pyrite_bench import BATCH, _transport_kwargs

    element, _, thickness_mm, E0_MeV = CASES[case]
    kwargs, composition = _transport_kwargs(element)
    k, T, cos = [], [], []
    for b, start in enumerate(range(0, ne, BATCH), start=first_batch):
        r = simulate_trajectories(
            E0_keV=E0_MeV * 1e3,
            Ne=min(BATCH, ne - start),
            composition=composition,
            thickness_ang=thickness_mm * MM,
            seed=34500 + b,
            **kwargs,
        )
        kind = np.asarray(r["event_kind"])
        hard = np.flatnonzero(kind == EVENT_HARD_RADIATIVE)
        n_hat = np.asarray(r["hard_radiative_direction"])[hard]
        k.append(np.asarray(r["hard_radiative_k_eV"])[hard])
        T.append(np.asarray(r["E_end_keV"])[hard] * 1e3)  # hard rows end pre-emission
        cos.append(np.sum(n_hat * np.asarray(r["v_hat"])[hard], axis=1))
    np.savez_compressed(
        out, k_eV=np.concatenate(k), T_eV=np.concatenate(T), cos=np.concatenate(cos)
    )


def _geant4(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            i = line.find("EBREM ")
            if i >= 0:
                rows.append(line[i:].split()[2:10])
    a = np.asarray(rows, dtype=float)
    k = a[:, 1]
    return k, a[:, 0] + k, np.sum(a[:, 2:5] * a[:, 5:8], axis=1)


def _stats(cos):
    theta = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    n = cos.size
    return dict(
        n=int(n),
        mean_cos=float(cos.mean()),
        mean_cos_se=float(cos.std(ddof=1) / np.sqrt(n)),
        median_deg=float(np.median(theta)),
        backward=float(np.mean(cos < 0)),
    )


def compare(case, log, npzs):
    E0 = CASES[case][3] * 1e6
    g = _geant4(log)
    p = [np.load(x) for x in npzs]
    p = tuple(np.concatenate([d[key] for d in p]) for key in ("k_eV", "T_eV", "cos"))
    report = []
    for t_lo, t_hi in zip(T_BINS[:-1], T_BINS[1:], strict=True):
        for k_lo, k_hi in zip(K_BINS[:-1], K_BINS[1:], strict=True):
            row = dict(T_over_E0=(t_lo, t_hi), k_over_T=(k_lo, k_hi))
            for name, (k, T, cos) in (("geant4", g), ("pyrite", p)):
                sel = (T >= t_lo * E0) & (T < t_hi * E0) & (k >= k_lo * T) & (k < k_hi * T)
                row[name] = _stats(cos[sel]) if sel.sum() > 30 else None
            report.append(row)
            if row["geant4"] and row["pyrite"]:
                G, P = row["geant4"], row["pyrite"]
                z = (P["mean_cos"] - G["mean_cos"]) / np.hypot(P["mean_cos_se"], G["mean_cos_se"])
                print(
                    f"{case} T/E0 {t_lo:.1f}-{t_hi:.1f} k/T {k_lo:.2f}-{k_hi:.2f}: "
                    f"<cos> G {G['mean_cos']:.3f} P {P['mean_cos']:.3f} (z {z:+.1f}); "
                    f"median G {G['median_deg']:.1f} P {P['median_deg']:.1f} deg; "
                    f"backward G {G['backward']:.3f} P {P['backward']:.3f}; n G {G['n']} P {P['n']}"
                )
    return report


def main(argv):
    if argv[0] == "pyrite":
        run_pyrite(argv[1], int(argv[2]), int(argv[3]), argv[4])
    else:
        compare(argv[1], argv[2], argv[3:])


if __name__ == "__main__":
    main(sys.argv[1:])
