"""Planar TDS interband rates against Andersen et al. (1983), 4 MeV electrons in Ni (#522).

Source: J. U. Andersen, E. Bonderup, E. Laegsgaard and A. H. Sorensen, Phys. Scr.
28, 308 (1983), doi:10.1088/0031-8949/28/3/008 (Zotero ``WCIUHZH7``). Their thermal
(Einstein, Doyle-Turner) interband probabilities ``W^T_{n n'}`` are the golden-rule
rates of our Pauli limit, ``DensityMatrixSolver.rate_matrix``. Transcribed by eye
from the 300 dpi page images, reading precision about 5% in ``W`` and 20 eV in
``Gamma``:

* Fig. 12: ``W^T_{0 n'}`` (Ni {111}, kappa = g/4, 300 K) for n' = 0..28;
* Fig. 9: interband-only 1-0 linewidth (short-dashed) versus rms vibration
  ``rho`` for {111} and {100};
* Fig. 8: measured 1-0 line energies (and the text: about 50% of the ground
  state's interband scattering goes to n' >= 8).

Model: ``planar_profile`` (independent atoms, no bonding charge) and
``tds_kernel`` only. Andersen's Fig. 12 and the short-dashed Fig. 9 curves are
thermal; they estimate electronic scattering at about 10% of interband
(Sec. 5.5), and Ni has no packaged conduction-band shell data. Ni also has no
packaged channeling thermal data, so Andersen's own ``rho`` (0.049/0.074/0.095 A
at 100/300/500 K, Fig. 7, T_D = 375 K) is injected as ``B = 8 pi^2 rho^2``
through a temporary thermal table; the package data are not changed.

Band index ``n'`` here sums destination Bloch blocks by energy order within each
block. Andersen's free-state labels (Fig. 12, n = 11) separate momentum
direction, so free-state bars of theirs correspond to adjacent pairs of ours;
the comparison uses the bound ground state, whose labels coincide.

The interband linewidth uses ``Gamma = (1 + beta) gamma^2 hbar v (W_0 + W_1)``
(Eqs. 3.9-3.10) at kappa = g/4. Andersen's curves average over kappa, include
a small vibration-correlation correction and the coherence factor of their
Eq. (4.19), which this estimate omits.

CPU, a few minutes locally::

    pixi run python checks/andersen1983_interband.py --out andersen.json

Diagnostic; always exits 0.

Validation: cr-lindblad-jump-operators
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import pyrite.materials.channeling.thermal as thermal

KINETIC_EV = 4.0e6
RHO_ANG = {100.0: 0.049, 300.0: 0.074, 500.0: 0.095}
# Fig. 12, n = 0 (300 dpi reading).
FIG12_W0 = [7.8, 2.2, 0.55, 0.38, 0.47, 0.55, 0.56, 0.54, 0.51, 0.46, 0.41, 0.36, 0.32, 0.29,
            0.25, 0.22, 0.18, 0.16, 0.135, 0.12, 0.09, 0.08, 0.07, 0.06, 0.055, 0.05, 0.045,
            0.04, 0.04]  # fmt: skip
# Fig. 9 short-dashed (interband only) [eV] at rho = 0.049, 0.074, 0.095 A.
FIG9_INTERBAND_EV = {"111": [330.0, 520.0, 655.0], "100": [290.0, 440.0, 545.0]}
# Fig. 8 measured E_{1-0} [keV] at the same rho.
FIG8_E10_KEV = {"111": [4.75, 4.45, 4.15], "100": [4.25, 3.95, 3.65]}
HBARC_EV_ANG = 1973.269804


def _inject_thermal(directory: Path) -> None:
    table = Path(thermal.THERMAL_DATA_PATH).read_text(encoding="utf-8")
    b = [8.0 * np.pi**2 * r**2 for r in RHO_ANG.values()]
    table += (
        "\n[ni.Ni]\natomic_weight = 58.6934\n"
        f"temperatures_K = {list(RHO_ANG)}\nB_ang2 = {b}\n"
        'source = "Andersen et al., Phys. Scr. 28, 308 (1983), Fig. 7"\n'
        'doi = "10.1088/0031-8949/28/3/008"\n'
    )
    path = directory / "thermal.toml"
    path.write_text(table, encoding="utf-8")
    thermal.THERMAL_DATA_PATH = path
    thermal._load.cache_clear()


def case(hkl, temperature_K: float, max_order: int) -> dict:
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling import (
        DensityMatrixSolver,
        longitudinal_kinematics,
        tds_kernel,
    )

    profile = planar_profile("ni", hkl, temperature_K, bonding_charge=False)
    low, barrier = profile.extrema()
    gamma, beta = longitudinal_kinematics(KINETIC_EV)
    solver = DensityMatrixSolver(
        profile,
        KINETIC_EV,
        tds_kernel(profile, KINETIC_EV, n_cells=4, max_order=max_order),
        margin_eV=1e6,
    )
    energies = solver.basis.energies
    rates, loss = solver.rate_matrix()
    index = np.argwhere(solver.basis.subspace)
    block = 1  # kappa = g / 4
    out = {
        "well_eV": barrier - low,
        "bound_states": int(np.sum(energies[block] < barrier)),
        "E10_keV": float((1 + beta) * gamma**2 * (energies[block][1] - energies[block][0]) / 1e3),
    }
    for n in (0, 1):
        column = int(np.flatnonzero((index[:, 0] == block) & (index[:, 1] == n))[0])
        to_band = np.zeros(energies.shape[1])
        np.add.at(to_band, index[:, 1], rates[:, column])
        to_band *= 1e4  # 1/angstrom -> 1/um
        total = float(loss[column] * 1e4)
        out[f"n{n}"] = {
            "W_total_per_um": total,
            "sum_rule": float(to_band.sum() / total),
            "W_inter_per_um": total - float(to_band[n]),
            "W_to_band_per_um": [float(w) for w in to_band[:29]],
            "inter_fraction_to_n_ge_8": float(to_band[8:].sum() / (total - to_band[n])),
        }
    out["Gamma10_interband_eV"] = float(
        (1 + beta) * gamma**2 * HBARC_EV_ANG * beta
        * (out["n0"]["W_inter_per_um"] + out["n1"]["W_inter_per_um"]) / 1e4
    )  # fmt: skip
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-order", type=int, default=15, help="Andersen use 31 plane waves")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = {"source": "Andersen et al., Phys. Scr. 28, 308 (1983)", "cases": {}}
    with tempfile.TemporaryDirectory() as tmp:
        _inject_thermal(Path(tmp))
        for plane, hkl in (("111", (1, 1, 1)), ("100", (1, 0, 0))):
            for j, (temperature, rho) in enumerate(RHO_ANG.items()):
                row = case(hkl, temperature, args.max_order)
                row["rho_ang"] = rho
                row["fig9_interband_eV"] = FIG9_INTERBAND_EV[plane][j]
                row["fig8_E10_keV"] = FIG8_E10_KEV[plane][j]
                if plane == "111" and temperature == 300.0:
                    ours = np.array(row["n0"]["W_to_band_per_um"])
                    ref = np.array(FIG12_W0)
                    row["fig12_ratio"] = [float(x) for x in ours / ref]
                report["cases"][f"{plane}-{int(temperature)}K"] = row
                print(
                    f"{{{plane}}} rho={rho:.3f}: E10 {row['E10_keV']:.2f} keV "
                    f"(Fig. 8 {row['fig8_E10_keV']:.2f}); Gamma_inter {row['Gamma10_interband_eV']:.0f} eV "
                    f"(Fig. 9 {row['fig9_interband_eV']:.0f}); W0 inter {row['n0']['W_inter_per_um']:.2f}/um, "
                    f"n'>=8 share {row['n0']['inter_fraction_to_n_ge_8']:.2f}, sum rule {row['n0']['sum_rule']:.3f}",
                    flush=True,
                )
    ratio = np.array(report["cases"]["111-300K"]["fig12_ratio"])
    print(
        f"Fig. 12 n=0 ratio ours/Andersen: median {np.median(ratio):.3f}, range {ratio.min():.2f}-{ratio.max():.2f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
