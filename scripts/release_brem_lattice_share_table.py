"""Build PyRITE's universal lattice-coherent bremsstrahlung share kernel (#1250).

Maintainer-only. Evaluates :func:`pyrite.montecarlo.spectrum.brem_lattice_share.shell_kernel_x`
on the released node grid and writes one ``.npz`` (``pyrite.datasets.BREM_LATTICE_SHARE``,
``pyrite tables fetch brem-lattice-share``). The kernel depends on no crystal
or element, so one table serves every radiator; it changes only with the
production CBS amplitude, its recoil form, or this grid.

The work splits by electron-energy node so it can run on several hosts::

    release_brem_lattice_share_table.py part OUT_part0.npz --T-index 0:12 --processes 16
    release_brem_lattice_share_table.py merge OUT.npz part*.npz

Grid (P5a measurements on #1250): ``|g|`` log-spaced 0.25--75.4 1/Angstrom
(69 nodes; the Waasmaier--Kirfel cap of the reflection selector), ``T`` from
1 keV by factors of 1.22 to above 1 MeV (36 nodes), ``theta`` every 5 degrees,
``x`` in 210 bins on ``[0, 1.05]``. Leave-one-out interpolation on these
spacings: ``|g|`` per-bin <= 1 %, ``theta`` <= 0.7 % (2.7 % at 10 degrees),
``T`` total <= 0.4 % with log-total + linear-shape interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pyrite.montecarlo.runner.reflection_cutoff import F0_RANGE_INVANG
from pyrite.montecarlo.spectrum.brem_lattice_share import LatticeShareKernel, shell_kernel_x

GRID = dict(
    g_invang=np.geomspace(0.25, F0_RANGE_INVANG, 69),
    T_keV=1.22 ** np.arange(36),
    theta_deg=np.arange(0.0, 181.0, 5.0),
    x_edges=np.linspace(0.0, 1.05, 211),
    n_mu=4000,
    n_phi=48,
)


def _cell(job):
    # Jobs carry their grid values: Pool workers re-import this module
    # (forkserver/spawn), so module state is not a channel.
    i_T, i_th, T, theta_deg, grid = job
    th = np.deg2rad(theta_deg)
    rows = np.array(
        [
            shell_kernel_x(g, T, th, grid["x_edges"], n_mu=grid["n_mu"], n_phi=grid["n_phi"])
            for g in grid["g_invang"]
        ]
    )
    return i_T, i_th, rows


def _empty(grid):
    shape = (grid["g_invang"].size, grid["T_keV"].size, grid["theta_deg"].size)
    return np.full((*shape, grid["x_edges"].size - 1), np.nan)


def build_part(out: Path, T_index: slice, processes: int, grid=GRID) -> None:
    out = Path(out)
    idx = np.arange(grid["T_keV"].size)[T_index]
    mass = _empty(grid)
    jobs = [
        (i, j, float(grid["T_keV"][i]), float(grid["theta_deg"][j]), grid)
        for i in idx
        for j in range(grid["theta_deg"].size)
    ]
    t0 = time.time()
    with Pool(processes) as pool:
        for n, (i, j, rows) in enumerate(pool.imap_unordered(_cell, jobs), 1):
            mass[:, i, j] = rows
            if n % 37 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, mass=mass, T_index=idx)


def merge(out: Path, parts: list[Path], grid=GRID) -> None:
    out = Path(out)
    mass = _empty(grid)
    for p in parts:
        with np.load(p) as z:
            for i in z["T_index"]:
                if not np.isnan(mass[:, i]).all():
                    raise ValueError(f"T node {i} appears in more than one part")
                mass[:, i] = z["mass"][:, i]
    if np.isnan(mass).any():
        missing = np.flatnonzero(np.isnan(mass).any(axis=(0, 2, 3)))
        raise ValueError(f"missing T nodes: {missing}")
    kernel = LatticeShareKernel(
        grid["g_invang"],
        grid["T_keV"],
        grid["theta_deg"],
        grid["x_edges"],
        mass,
        grid["n_mu"],
        grid["n_phi"],
    )
    np.savez_compressed(out, **kernel.arrays())
    print(out, out.stat().st_size, "bytes sha256", hashlib.sha256(out.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("part")
    p.add_argument("out", type=Path)
    p.add_argument("--T-index", default=":", help="python slice over the T nodes, e.g. 0:12")
    p.add_argument("--processes", type=int, default=16)
    m = sub.add_parser("merge")
    m.add_argument("out", type=Path)
    m.add_argument("parts", type=Path, nargs="+")
    a = parser.parse_args()
    if a.cmd == "part":
        build_part(
            a.out, slice(*[int(s) if s else None for s in a.T_index.split(":")]), a.processes
        )
    else:
        merge(a.out, a.parts)


if __name__ == "__main__":
    main()
