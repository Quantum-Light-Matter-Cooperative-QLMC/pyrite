"""W <001> 1s/2p occupation evolution for sub-MeV axial CR (#585). Remote only.

Run on the remote box (CPU) with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=1 uv run python checks/channeling_submev_w001_occupation.py \
        --energy-kev 30 --fraction 0.1 --arm tds+electronic --step-ang 0.5

Each invocation runs one case and prints JSON; a SLURM array covers the grid.

The electron enters as a plane wave on a ``cells x cells`` supercell of strings.
Its carrier tilt is drawn from an isotropic 2D Gaussian divergence of width
``f psi_c``. That is the incoherent entry mixture of
``channeling_submev_w001_entry.py``, sampled by Monte Carlo over ``tilts``.
:class:`AxialMCWF` (``cr-axial-realspace-dissipator``) propagates each tilt
with the TDS and, optionally, electronic jumps of :func:`axial_dissipator`.
These are the MeV-calibrated kernels, applied here without a sub-MeV
calibration.

The observable (:class:`BlochBandObservable`) is exact on the supercell. A
state ``exp(iK.r) phi(r)`` with supercell-periodic ``phi`` holds plane waves
``K + q`` only, ``q`` on the supercell reciprocal lattice. These fall on
``cells**2`` crystal wavevectors ``k = K + g r / cells`` (folded to the first
zone). At each ``k`` the state is projected on the Bloch bands of
:func:`axial_states`. The record gives, against depth:

* the bound 1s (band 0) and 2p (bands 1-2) populations per incident electron,
  with the surviving fraction and jump counts;
* the 2p -> 1s photon rate from the vertical (same-``k``) velocity-form dipole
  and gap of each band at each ``k``, and its running depth integral. The
  doublet bands add incoherently; at entry the omitted inter-band coherence is
  at most 0.26 % (``channeling_submev_w001_dispersion.py``).

Cutting the ``k = 0`` orbitals to one string cell is not a substitute below
1 MeV: the 2p band disperses by 2.6 eV at 30 keV, and an incommensurate
carrier is not periodic on the supercell (#585 record).

Not included: photon absorption and escape, longitudinal harmonics, stopping,
capture from outside the supercell, and any sub-MeV check of the TDS or
electronic kernels. Nothing here is a validated yield or a support verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2, M_E_EV
from pyrite.montecarlo.channeling import longitudinal_kinematics, states
from pyrite.montecarlo.channeling.axial_bloch import axial_states
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_emission import axial_line_catalog
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.dipole import dipole_rate_per_ang
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

TEMPERATURE_K = 293.15
#: Bandwidth used only to obtain the ``k = 0`` reference line below 300 keV [eV].
PROJECTOR_BANDWIDTH_EV = 4.0
CATALOG_ORDER = 10
#: Bloch solves kept per observable; electronic kicks make most tilts unique.
CACHE_LIMIT = 4096
ARMS = {"tds": ("tds",), "tds+electronic": ("tds", "electronic")}


def two_level_catalog(profile, kinetic_eV: float):
    """1s and the 2p doublet with their ``k = 0`` ``2p -> 1s`` line (record metadata)."""
    catalog = axial_line_catalog(
        profile, kinetic_eV, max_order=CATALOG_ORDER, max_bandwidth_eV=PROJECTOR_BANDWIDTH_EV
    )
    levels = catalog.levels[:2]
    if [len(level.states) for level in levels] != [1, 2]:
        raise ValueError("W <001> no longer has the 1s level and the 2p doublet")
    lines = tuple(line for line in catalog.lines if (line.initial, line.final) == (1, 0))
    return replace(catalog, levels=levels, lines=lines)


class BlochBandObservable:
    """Observable: bound 1s and 2p band populations and the ``2p -> 1s`` rate.

    Returns ``[P_1s, P_2p, dN/dz]`` for the normalized state
    ``exp(iK.r) phi(r)``; ``phi`` is periodic on ``grid``, a ``cells x cells``
    supercell whose origin is a string site.
    """

    def __init__(self, profile, gamma: float, grid: RealSpaceGrid, cells: int) -> None:
        if abs(grid.x[0]) > 1e-12 or abs(grid.y[0]) > 1e-12:
            raise ValueError("grid must start at the string origin")
        self.profile = profile
        self.gamma = gamma
        self.cells = cells
        self.g = np.asarray(profile.g_inv_ang, dtype=float)
        self.barrier = profile.extrema()[1]
        self.sqrt_area = np.sqrt(grid.extent_ang[0] * grid.extent_ang[1])
        self.order = CATALOG_ORDER
        # Integer supercell harmonics m = cells n + r of each FFT axis.
        self._m = [np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(int) for n in grid.shape]
        self._cache: dict = {}

    def _bands(self, k_red):
        key = tuple(np.round(k_red, 12))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        solved = axial_states(self.profile, self.gamma, tuple(k_red), max_order=self.order)
        c = solved.coefficients
        kvec = np.asarray(k_red) + solved.orders * self.g
        momentum = np.einsum("g,ga,gb->ab", c[:, 0].conj(), kvec, c[:, 1:3])
        gaps = solved.energies_eV[1:3] - solved.energies_eV[0]
        dipole = (
            (2.0 * HBAR2_2M_EV_ANG2 / self.gamma)
            * np.sqrt(np.sum(np.abs(momentum) ** 2, axis=0))
            / gaps
        )
        bound = solved.energies_eV[:3] < self.barrier
        rates = np.where(bound[1:], dipole_rate_per_ang(gaps, dipole, self.gamma), 0.0)
        o = self.order
        table = np.full((2 * o + 1, 2 * o + 1), -1)
        table[solved.orders[:, 0] + o, solved.orders[:, 1] + o] = np.arange(len(c))
        hit = (c[:, :3].conj().T, bound, rates, table)
        self._cache[key] = hit
        return hit

    def __call__(self, phi, tilt_inv_ang=(0.0, 0.0)):
        amplitude = np.fft.fft2(phi) * (self.sqrt_area / phi.size)  # on e^{i(K+q).r}/sqrt(A)
        tilt = np.asarray(tilt_inv_ang, dtype=float)
        out = np.zeros(3)
        o, cells = self.order, self.cells
        for rx in range(cells):
            sx = self._m[0] % cells == rx
            for ry in range(cells):
                sy = self._m[1] % cells == ry
                k = tilt + self.g * np.array([rx, ry]) / cells
                shift = np.rint(k / self.g).astype(int)
                conj_c, bound, rates, table = self._bands(k - shift * self.g)
                nx = (self._m[0][sx] - rx) // cells + shift[0] + o
                ny = (self._m[1][sy] - ry) // cells + shift[1] + o
                okx = (nx >= 0) & (nx <= 2 * o)
                oky = (ny >= 0) & (ny <= 2 * o)
                rows = table[np.ix_(nx[okx], ny[oky])]
                block = amplitude[np.ix_(np.flatnonzero(sx)[okx], np.flatnonzero(sy)[oky])]
                band = np.abs(conj_c[:, rows.ravel()] @ block.ravel()) ** 2
                band = np.where(bound, band, 0.0)
                out += (band[0], band[1] + band[2], rates @ band[1:])
        return out


def tilt_series(result) -> np.ndarray:
    """1s and 2p populations, 2p -> 1s rate and survival against depth for one tilt."""
    out = np.zeros((4, len(result.depths_ang)))
    out[:3] = np.asarray(result.observables, dtype=float).T
    out[3] = result.surviving
    return out


def cumulative_photons(depths, rates) -> np.ndarray:
    """Trapezoid depth integral of each row of ``rates`` (photons per electron)."""
    rates = np.atleast_2d(rates)
    steps = 0.5 * (rates[:, 1:] + rates[:, :-1]) * np.diff(depths)
    return np.concatenate([np.zeros((rates.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)


def plane_wave(grid: RealSpaceGrid):
    spacing = tuple(e / n for e, n in zip(grid.extent_ang, grid.shape, strict=True))
    phi = states.from_array(np.ones(grid.shape), spacing_ang=spacing)
    if abs(grid.norm(grid.sample(phi)) - 1.0) > 1e-9:
        raise ValueError("plane-wave entry is not normalized on the supercell")
    return phi


def run_case(args) -> dict:
    kinetic_eV = 1e3 * args.energy_kev
    profile = axial_profile("tungsten", (0, 0, 1), TEMPERATURE_K)
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    lo, barrier = profile.extrema()
    critical = float(np.sqrt(2.0 * (barrier - lo) / (gamma * beta**2 * M_E_EV)))
    catalog = two_level_catalog(profile, kinetic_eV)
    (line,) = catalog.lines
    cells = args.cells
    shape = (args.points * cells, args.points * cells)
    started = time.perf_counter()
    dissipator = axial_dissipator(
        profile, kinetic_eV, (cells, cells), shape, mechanisms=ARMS[args.arm]
    )
    stepper = AxialMCWF(dissipator, kinetic_eV, profile.on_grid((cells, cells), shape))
    observable = BlochBandObservable(profile, gamma, dissipator.grid, cells)
    entry = plane_wave(dissipator.grid)
    depths = np.arange(0.0, args.max_depth_ang + 0.5 * args.record_ang, args.record_ang)
    rng = np.random.default_rng(args.seed)
    sigma = args.fraction * critical
    per_tilt = []  # (tilts, 4, depths): 1s, 2p, rate, surviving
    jumps: dict[str, int] = {}
    for t in range(args.tilts):
        tilt = tuple(float(v) for v in rng.normal(0.0, sigma, 2)) if sigma > 0 else (0.0, 0.0)
        result = stepper.run(
            entry,
            depths,
            args.histories,
            seed=args.seed * 100_003 + t,
            step_ang=args.step_ang,
            tilt_rad=tilt,
            observable=observable,
        )
        per_tilt.append(tilt_series(result))
        for kind, count in result.jumps.items():
            jumps[kind] = jumps.get(kind, 0) + int(count)
    series = np.asarray(per_tilt)
    cumulative = cumulative_photons(depths, series[:, 2])
    mean = series.mean(axis=0)
    sem = series.std(axis=0, ddof=1) / np.sqrt(args.tilts) if args.tilts > 1 else 0 * mean
    cum_sem = (
        cumulative.std(axis=0, ddof=1) / np.sqrt(args.tilts)
        if args.tilts > 1
        else 0 * cumulative[0]
    )
    return {
        "kinetic_keV": args.energy_kev,
        "arm": args.arm,
        "observable": "bloch-band",
        "divergence_over_critical": args.fraction,
        "sigma_mrad": 1e3 * sigma,
        "cells": cells,
        "grid_shape": list(shape),
        "step_ang": args.step_ang,
        "tilts": args.tilts,
        "histories_per_tilt": args.histories,
        "seed": args.seed,
        "k0_gap_eV": float(line.delta_eV),
        "forward_k0_line_eV": float(catalog.photon_energy_eV(line)),
        "depths_ang": depths.tolist(),
        "population_1s": mean[0].tolist(),
        "population_1s_sem": sem[0].tolist(),
        "population_2p": mean[1].tolist(),
        "population_2p_sem": sem[1].tolist(),
        "rate_2p_1s_per_ang": mean[2].tolist(),
        "rate_2p_1s_per_ang_sem": sem[2].tolist(),
        "surviving": mean[3].tolist(),
        "cumulative_photons": cumulative.mean(axis=0).tolist(),
        "cumulative_photons_sem": cum_sem.tolist(),
        "jumps_per_electron": {k: v / (args.tilts * args.histories) for k, v in jumps.items()},
        "wall_s": time.perf_counter() - started,
    }


SUMMARY_DEPTHS_ANG = (100.0, 300.0, 1000.0)


def efold_depth(depths, values) -> float | None:
    """First depth where ``values`` falls to ``values[0] / e`` (linear interpolation)."""
    depths = np.asarray(depths, dtype=float)
    values = np.asarray(values, dtype=float)
    target = values[0] / np.e
    below = np.flatnonzero(values <= target)
    if values[0] <= 0.0 or below.size == 0:
        return None
    i = int(below[0])
    x0, x1, y0, y1 = depths[i - 1], depths[i], values[i - 1], values[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def summarize_case(case: dict) -> dict:
    """Derived quantities of one case record."""
    depths = np.asarray(case["depths_ang"])
    row = {
        key: case[key]
        for key in ("kinetic_keV", "arm", "divergence_over_critical", "step_ang", "cells")
    }
    row["points_per_cell"] = case["grid_shape"][0] // case["cells"]
    for level in ("1s", "2p"):
        row[f"population_{level}_entry"] = case[f"population_{level}"][0]
        row[f"population_{level}_entry_sem"] = case[f"population_{level}_sem"][0]
        row[f"efold_{level}_ang"] = efold_depth(depths, case[f"population_{level}"])
    for depth in SUMMARY_DEPTHS_ANG:
        i = int(np.argmin(np.abs(depths - depth)))
        if abs(depths[i] - depth) > 1e-6:
            continue
        row[f"photons_{depth:g}A"] = case["cumulative_photons"][i]
        row[f"photons_{depth:g}A_sem"] = case["cumulative_photons_sem"][i]
    row["surviving_end"] = case["surviving"][-1]
    row["jumps_per_electron"] = case["jumps_per_electron"]
    return row


def summarize(paths) -> dict:
    """Merge case records into one evidence record with a summary table."""
    cases, shas = [], set()
    for path in sorted(paths, key=lambda p: int(Path(p).stem.split("-")[-1])):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        cases.append(record["case"])
        shas.add(record["meta"].get("sha") or os.environ.get("PYRITE_SYNCED_SHA"))
    return {
        "meta": {
            "issue": 585,
            "material": "tungsten",
            "axis_uvw": [0, 0, 1],
            "shas": sorted(str(s) for s in shas),
        },
        "summary": [summarize_case(case) for case in cases],
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "summarize":
        json.dump(summarize(argv[1:]), sys.stdout, indent=1)
        print()
        return 0
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="`summarize CASE.json ...` merges case records into one evidence record.",
    )
    parser.add_argument("--energy-kev", type=float, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--arm", choices=sorted(ARMS), default="tds+electronic")
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--points", type=int, default=25, help="grid points per cell per axis")
    parser.add_argument("--step-ang", type=float, default=0.5)
    parser.add_argument("--record-ang", type=float, default=10.0)
    parser.add_argument("--max-depth-ang", type=float, default=500.0)
    parser.add_argument("--tilts", type=int, default=64)
    parser.add_argument("--histories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=585)
    args = parser.parse_args(argv)
    if (
        args.energy_kev <= 0
        or args.fraction < 0
        or min(args.cells, args.points, args.tilts, args.histories) < 1
        or args.step_ang <= 0
        or args.record_ang < args.step_ang
    ):
        parser.error("positive energy, sizes and step required; record >= step")
    json.dump(
        {
            "meta": {
                "issue": 585,
                "material": "tungsten",
                "axis_uvw": [0, 0, 1],
                "sha": os.environ.get("PYRITE_SYNCED_SHA"),
            },
            "case": run_case(args),
        },
        sys.stdout,
        indent=1,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
