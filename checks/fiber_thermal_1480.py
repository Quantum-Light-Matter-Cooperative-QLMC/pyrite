#!/usr/bin/env python3
"""Research-only #1480: measure CBS subtraction shell weights on the CXR domain.

Uses the same B per hkl as the paired frozen diagnostic. The shell sum is
orientation independent in the existing isotropic Born kernel. This is NOT
an emitted spectrum or a corrected transport calculation. With --kernel,
report pointwise subtraction densities at the requested T, theta and energies.
No production selector, table cache, or model marker is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from fiber_nonbasal_cxr_1480 import b_effective, families

from pyrite.materials.catalog import load_material_catalog
from pyrite.montecarlo.spectrum.brem_lattice_share import (
    LatticeShareKernel,
    _reflection_shell_weights,
    lattice_share_density,
    reflection_shell_weights,
)


def measure(args):
    cat = load_material_catalog().crystal(args.crystal)
    fams = families(args.crystal, args.g_max)
    full = [hkl for fam in fams for hkl in fam["members"]]
    basal = [hkl for hkl in full if hkl[:2] == (0, 0)]
    kernel = LatticeShareKernel.load(args.kernel) if args.kernel else None
    record = {
        "crystal": args.crystal,
        "g_max": args.g_max,
        "thermal": (
            "packaged graphite tensor, exact catalog basal B33"
            if args.crystal == "hopg"
            else "provisional scalar B33"
        ),
        "provisional": args.crystal != "hopg",
        "arms": {},
    }
    for name, hkls, tensor in (
        ("basal", basal, False),
        ("full_B33", full, False),
        ("full_tensor", full, True),
    ):
        if tensor:
            g, weights = _reflection_shell_weights(
                args.crystal,
                hkls,
                lambda hkl: b_effective(args.crystal, hkl, cat.B_ang2),
                cat.composition,
            )
        else:
            g, weights = reflection_shell_weights(args.crystal, hkls, cat.B_ang2, cat.composition)
        w = sum(weights.values(), np.zeros_like(g))
        out = {
            "reflections": len(hkls),
            "shells": len(g),
            "sum_U2_eV2": float(w.sum()),
            "g_invang": g.tolist(),
            "weights_eV2": w.tolist(),
        }
        if kernel is not None:
            if not kernel.T_keV[0] <= args.energy <= kernel.T_keV[-1]:
                raise ValueError("electron energy outside kernel domain")
            if g.size and (g[0] < kernel.g_invang[0] or g[-1] > kernel.g_invang[-1]):
                raise ValueError("reflection domain outside kernel domain")
            density = lattice_share_density(kernel, g, w, args.energy, args.photons)
            out["share_density_per_eV_sr_ang"] = [
                float(np.interp(args.theta, kernel.theta_deg, col)) for col in density.T
            ]
        record["arms"][name] = out
    if kernel is not None:
        record.update(
            energy_keV=args.energy,
            theta_deg=args.theta,
            photons_eV=args.photons,
            kernel=str(args.kernel),
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crystal", choices=("hopg", "hbn"))
    parser.add_argument("--g-max", type=float, default=10.0)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--energy", type=float, default=30.0)
    parser.add_argument("--theta", type=float, default=119.0)
    parser.add_argument("--photons", nargs="+", type=float, default=[500, 1036, 1500, 2000])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0 < args.g_max <= 4 * np.pi * 6 or not 0 <= args.theta <= 180:
        parser.error("g-max must be inside the form-factor domain and theta in [0, 180]")
    if args.energy <= 0 or any(not 0 < k <= args.energy * 1000 for k in args.photons):
        parser.error("require positive energies and photons below the kinetic energy")
    record = measure(args)
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n")
    for name, arm in record["arms"].items():
        print(
            f"{name:>14}: {arm['reflections']:5d} reflections, "
            f"sum |U_g|^2 = {arm['sum_U2_eV2']:.8g} eV^2"
        )
        if "share_density_per_eV_sr_ang" in arm:
            print("  subtraction densities:", arm["share_density_per_eV_sr_ang"])
    if record["provisional"]:
        print("WARNING: hBN lacks complete ADPs; full_tensor is scalar B33 only")


if __name__ == "__main__":
    main()
