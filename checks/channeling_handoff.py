"""Separability and angle scale of the planar channeling handoff (#506).

The zone handoff (``montecarlo/channeling/handoff.py``) samples a
dechanneled electron's transverse energy from the depth-integrated free flux,
independently of its handoff depth. This check measures that assumption at
the production grid (Si(110), 293.15 K, 5 MeV, aligned beam without
divergence, ``H = 16``, 16 Bloch cells, ``dz = 250`` angstrom, 20 um) by
chaining constant-energy solves over 5 um chunks. It reports the total
variation distance between each chunk's ``E_perp`` marginal and the
whole-depth marginal, the overflow share, and the mean free handoff angle in
units of the planar critical angle ``psi_c = sqrt(2 U_depth / (p v))``, the
scale the declared 300 eV margin sets.

    PYRITE_MC_BACKEND=cpu uv run python checks/channeling_handoff.py

Exit 0 when the largest chunk distance is at most ``--tolerance`` (0.1).

Validation: cr-zone-handoff-accounting
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energy-eV", type=float, default=5e6)
    parser.add_argument("--orders", type=int, default=16)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--dz", type=float, default=250.0)
    parser.add_argument("--chunk-um", type=float, default=5.0)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args(argv)

    energy = args.energy_eV
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    grid = {"n_cells": args.cells, "max_order": args.orders}
    kernel = tds_kernel(profile, energy, **grid) + electronic_kernel(profile, energy, **grid)
    solver = DensityMatrixSolver(profile, energy, kernel)
    edges = np.linspace(solver.cutoff_eV, float(solver.basis.energies.max()), 65)
    rho, entry = None, 0.0
    marginals, free, overflow = [], 0.0, 0.0
    for _ in range(args.chunks):
        result = solver.propagate(
            args.chunk_um * 1e4,
            args.dz,
            energy_edges_eV=edges,
            initial=None if rho is None else (rho, entry),
        )
        rho, entry = result.final_density, result.entry_flux
        marginals.append(result.free_flux_by_energy.copy())
        free += float(result.free_flux[-1] + result.reprojection_flux[-1])
        overflow += float(result.overflow_flux[-1])
    whole = np.sum(marginals, axis=0)
    whole_p = whole / whole.sum()
    distances = [0.5 * float(np.abs(m / m.sum() - whole_p).sum()) for m in marginals]
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma, beta = solver.gamma0, solver.beta0
    pv = gamma * 510998.95 * beta**2
    minimum, barrier = profile.extrema()
    psi_c = np.sqrt(2.0 * (barrier - minimum) / pv)
    mean_psi = float(np.sum(whole_p * np.sqrt(2.0 * centres / pv)))
    print(f"chunk E_perp total-variation distances: {np.round(distances, 4).tolist()}")
    print(f"max distance {max(distances):.4f} (tolerance {args.tolerance})")
    print(
        f"free share {free:.4f}, overflow share {overflow:.4f}, trace {float(np.einsum('rnn->', rho).real):.4f}"
    )
    print(
        f"psi_c {psi_c * 1e3:.3f} mrad; mean free handoff psi {mean_psi * 1e3:.3f} mrad = {mean_psi / psi_c:.2f} psi_c"
    )
    return 0 if max(distances) <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
