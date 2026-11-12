"""Transverse-recoil intensity prototype and observability table for #527.

Part 1 compares PyRITE's classical per-reflection PXR on a straight
segment (30 nm Si [001], theta = 100 deg) with Shi et al., Nat. Commun. 15,
7803 (2024), Fig. 2 (visual peak reads, not digitized). Each reflection's
classical spectrum is shifted rigidly to its exact recoil root
(``recoil_resonance_eV``) to form the prototype "QM" spectrum. The shift
keeps each row's integral and shape; it does not model Shi's quantum matrix
element or the recoil Jacobian.

Part 2 repeats the construction on transported segments and folds in a
Gaussian detector (0, 5 eV WDS/microcalorimeter-like, 70 eV EDS-like) for:
- the Fig. 2 large-angle geometry (30 and 300 keV, 30 nm Si, 100 deg);
- the SI S10 forward geometry (100 keV, 100 nm Si, 2.5 deg);
- a Huang-like vdW case (HOPG, 11.7 keV, 85 nm, 130 deg).
The per-row shift uses the nominal beam direction and energy. Scattered
flights keep their classical line position from ``mc_cxr_spectrum``; only
the recoil correction is nominal. Backgrounds (brem, characteristic, forward
CBS) are not included.

Production lines sit on the exact root since #1099, so the classical
spectra here are taken under the ``hbar -> 0`` hook
(``_recoil._RECOIL_HBARC_EV_ANG = 0``) that keeps the classical root, width
and phase bit for bit. Shi heights come from the vector-digitized Fig. 2
(``tests/data/recoil/shi-2024-fig2.json``, #1103). Shi's CL curve is the
PXR-only classical rate, so Part 1 also reports PyRITE's classical PXR
component (``checks/shi_2024_matrix_elements_1103.py``).

Run:  uv run python checks/transverse_recoil_527.py [electrons]

Validation: quantum-recoil-observability
"""

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.materials.recoil import recoil_resonance_eV, recoil_shift_fraction
from pyrite.montecarlo.spectrum import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import _recoil
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories

BEAM = np.array([0.0, 0.0, 1.0])
SI = [("Si", 0.04994)]
HOPG = [("C", 0.1136)]
B_SI = 0.4632  # Si Debye-Waller B at room temperature [Ang^2]
B_HOPG = 0.8
DETECTORS_EV = (0.0, 5.0, 70.0)

# Shi Fig. 2b/d/f classical (CL) peak heights, photons/(eV sr electron):
# (n_z=-2, n_z=-3), from the vector-digitized figure (#1103).
_FIG2 = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/data/recoil/shi-2024-fig2.json").read_text()
)["curves"]
SHI_CL = {
    T: tuple(_FIG2[f"{T:g} keV CL nz=-{nz}"]["peak"] for nz in (2, 3))
    for T in (30.0, 300.0, 5000.0)
}


@contextlib.contextmanager
def _classical_lines():
    """Classical production lines: the ``hbar -> 0`` recoil hook (#1099)."""
    saved = _recoil._RECOIL_HBARC_EV_ANG
    _recoil._RECOIL_HBARC_EV_ANG = 0.0
    try:
        yield
    finally:
        _recoil._RECOIL_HBARC_EV_ANG = saved


def _si_allowed(h, k, nz):
    if len({h % 2, k % 2, nz % 2}) != 1:
        return False
    return h % 2 == 1 or (h + k + nz) % 4 == 0


def _si_rows(nz, hmax=8):
    span = range(-hmax, hmax + 1)
    return [(h, k, nz) for h in span for k in span if _si_allowed(h, k, nz)]


def _straight(T_keV, L_ang):
    return {
        "r_mid": np.array([[0.0, 0.0, 0.5 * L_ang]]),
        "v_hat": BEAM[None, :].copy(),
        "L_ang": np.array([L_ang]),
        "E_keV": np.array([T_keV]),
        "t_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": L_ang,
    }


def _photon(theta_deg):
    th = np.radians(theta_deg)
    return np.array([np.sin(th), 0.0, np.cos(th)])


def _roots(crystal, rows, T_keV, n_hat):
    lat = CRYSTALS[crystal]["lattice"]
    g = np.array([reciprocal_g_vector(r, lat)[0] for r in rows])
    exact = recoil_resonance_eV(T_keV * 1e3, BEAM, g, n_hat)
    classical = exact / (1.0 + recoil_shift_fraction(T_keV * 1e3, BEAM, g, n_hat))
    ok = np.isfinite(exact) & (exact > 0.0) & np.isfinite(classical)
    return exact, classical, ok


def _cl_qm(segments, grid, crystal, rows, B, n_hat, exact, classical, ok):
    """Classical sum and rigidly recoil-shifted sum over reflections."""
    cl = np.zeros_like(grid)
    qm = np.zeros_like(grid)
    per_row = []
    for row, e_x, e_c, keep in zip(rows, exact, classical, ok, strict=True):
        if not keep:
            continue
        with _classical_lines():
            s = np.asarray(
                mc_cxr_spectrum(
                    segments, grid, crystal, [row], n_hat=n_hat, beam_uvw=(0, 0, 1), B_ang2=B
                )
            )
        cl += s
        qm += np.interp(grid - (e_x - e_c), grid, s, left=0.0, right=0.0)
        per_row.append((float(s.max()), row, float(e_x)))
    return cl, qm, sorted(per_row, reverse=True)


def _gauss(grid, spectrum, fwhm):
    if fwhm <= 0.0:
        return spectrum
    sigma = fwhm / 2.3548
    dx = grid[1] - grid[0]
    half = int(4.0 * sigma / dx) + 1
    x = np.arange(-half, half + 1) * dx
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(spectrum, w / w.sum(), mode="same")


def part1_shi_fig2():
    print("Part 1: Shi Fig. 2 (straight 30 nm Si [001], theta = 100 deg)")
    n_hat = _photon(100.0)
    failures = []
    for T in (30.0, 300.0, 5000.0):
        heights = []
        for nz in (2, 3):
            rows = _si_rows(nz)
            exact, classical, ok = _roots("silicon", rows, T, n_hat)
            c = float(np.nanmax(classical[ok]))
            grid = np.arange(max(0.6 * c, float(np.nanmin(exact[ok])) - 60.0), c + 60.0, 0.25)
            cl, qm, per_row = _cl_qm(
                _straight(T, 300.0), grid, "silicon", rows, B_SI, n_hat, exact, classical, ok
            )
            peak = grid[cl.argmax()]
            heights.append(cl.max())
            with _classical_lines():
                _, pxr, _ = mc_cxr_spectrum(
                    _straight(T, 300.0),
                    grid,
                    "silicon",
                    [r for r, k in zip(rows, ok, strict=True) if k],
                    n_hat=n_hat,
                    beam_uvw=(0, 0, 1),
                    B_ang2=B_SI,
                    components=True,
                )
            pxr_ratio = float(np.max(pxr)) / SHI_CL[T][nz - 2]
            share = per_row[0][0] / cl.max()
            print(
                f"  T={T:6.0f} keV n_z=-{nz}: classical root {c:7.1f} eV, CL peak {peak:7.1f} eV "
                f"{cl.max():.3g} (Shi {SHI_CL[T][nz - 2]:.3g}, ratio {cl.max() / SHI_CL[T][nz - 2]:.2f}, "
                f"PXR-only {pxr_ratio:.2f}); "
                f"QM peak {grid[qm.argmax()]:7.1f} eV, QM/CL {qm.max() / cl.max():.2f}; "
                f"largest row {per_row[0][1]} at {per_row[0][2]:.1f} eV holds {share:.2f} of CL peak"
            )
            if abs(peak - c) > 2.0:
                failures.append(f"CL peak {peak} vs classical root {c} at {T} keV")
        if T == 5000.0:
            for got, ref in zip(heights, SHI_CL[T], strict=True):
                if abs(got / ref - 1.0) > 0.15:
                    failures.append(f"5 MeV CL height {got:.3g} vs Shi {ref:.3g}")
    return failures


CASES = (
    ("Si 30 keV, 30 nm, 100 deg", "silicon", SI, B_SI, 30.0, 300.0, 100.0, _si_rows(2)),
    ("Si 300 keV, 30 nm, 100 deg", "silicon", SI, B_SI, 300.0, 300.0, 100.0, _si_rows(2)),
    ("Si 100 keV, 100 nm, 2.5 deg", "silicon", SI, B_SI, 100.0, 1000.0, 2.5, _si_rows(1, 7)),
    ("HOPG 11.7 keV, 85 nm, 130 deg (004)", "hopg", HOPG, B_HOPG, 11.7, 850.0, 130.0, [(0, 0, 4)]),
)


def part2_observability(n_electrons):
    print(f"\nPart 2: transported observability ({n_electrons} electrons per case)")
    print(
        "  case | segments | detector FWHM | CL peak | QM peak | peak shift | "
        "centroid shift | FWHM CL / QM | QM/CL height"
    )
    for name, crystal, comp, B, T, L, theta, rows in CASES:
        t0 = time.time()
        n_hat = _photon(theta)
        exact, classical, ok = _roots(crystal, rows, T, n_hat)
        c = float(np.nanmax(classical[ok]))
        grid = np.arange(max(0.6 * c, float(np.nanmin(exact[ok])) - 150.0), c + 250.0, 0.5)
        transported = simulate_trajectories(
            E0_keV=T,
            Ne=n_electrons,
            thickness_ang=L,
            composition=comp,
            seed=11,
            E_cut_keV=min(5.0, 0.5 * T),
        )
        flights = transported["L_ang"].size / n_electrons
        for tag, segments in (("straight", _straight(T, L)), ("transported", transported)):
            cl, qm, _ = _cl_qm(segments, grid, crystal, rows, B, n_hat, exact, classical, ok)
            for fwhm in DETECTORS_EV:
                a, b = _gauss(grid, cl, fwhm), _gauss(grid, qm, fwhm)
                pa, pb = grid[a.argmax()], grid[b.argmax()]
                da = np.sum(grid * b) / b.sum() - np.sum(grid * a) / a.sum()
                wa, wb = np.ptp(grid[a >= 0.5 * a.max()]), np.ptp(grid[b >= 0.5 * b.max()])
                print(
                    f"  {name} | {tag} | {fwhm:4.0f} eV | "
                    f"{pa:7.1f} | {pb:7.1f} | {pb - pa:6.1f} | {da:6.1f} | "
                    f"{wa:5.1f} / {wb:5.1f} | {b.max() / a.max():.2f}"
                )
        print(f"  ({flights:.1f} transported flights per electron, {time.time() - t0:.0f} s)")


def main():
    n_electrons = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    failures = part1_shi_fig2()
    part2_observability(n_electrons)
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("\nPASS: classical positions within 2 eV of the root; 5 MeV heights within 15 % of Shi")


if __name__ == "__main__":
    main()
