"""Research diagnostic: MCWF return-flux deficit of an energy-cut handoff.

Heavy: run on the lab box, not locally (see the remote-gpu-jobs skill)::

    OMP_NUM_THREADS=1 uv run python checks/axial_mcwf_return_flux.py --out axial_return_flux.json

``--smoke`` runs a seconds-long TDS-only check of the plumbing.

The reference ensemble is the production MCWF unravelling (TDS and
electronic jumps, no handoff). The absorbing ensembles measure the spectral
projector P(E_m) of the grid Hamiltonian at the electron's carrier tilt at
entry and after every jump: outcome P continues with the projected state,
outcome F is handed off (absorbed). This is the unravelling of the absorbing
retained generator in docs/research/physics/axial-escape-instrument.md, up to
the coherent leakage of the no-jump loss between P and F. That leakage is
small only when the cut sits in a spectral gap wide against the loss matrix
elements; electronic kicks give each electron a generic tilt, so the cut is
placed per tilt in the widest gap within ``--gap-window-eV`` of the nominal
matching energy, and the gaps used are reported. The bound observable is the
weight below the cell maximum U_max. Chunks share seeds across models, so
their differences are paired. Research diagnostic only; not a dechanneling
prediction or native launch model.

Source: Dalibard, Castin & Molmer, PRL 68, 580 (1992) (MCWF unravelling);
Castin, Dalibard & Molmer, arXiv:0805.4002, Eq. (2); TDS and electronic jumps
of cr-axial-realspace-dissipator. Lengths in Å, energies in eV.

Validation: cr-axial-zone-handoff-accounting
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np


class GridProjector:
    """Spectral projectors of the MCWF grid Hamiltonian at a carrier tilt.

    The grid Hamiltonian is ``T(k_t + k_m) + U`` with ``U`` applied pointwise,
    i.e. a circular convolution of plane-wave orders. With ``cells`` periodic
    cells per axis it is block diagonal in the order residue modulo ``cells``.
    Eigendecompositions are cached per tilt (least recently used first out;
    each entry holds ``cells²`` dense blocks of ``(N/cells)²`` orders).
    Validation: cr-axial-zone-handoff-accounting
    """

    def __init__(self, potential_eV, kx, ky, cells, kinetic_scale_eV_ang2, cache_size=160):
        self.n = potential_eV.shape
        self.u = np.fft.fft2(potential_eV) / potential_eV.size
        self.kx, self.ky = np.asarray(kx), np.asarray(ky)
        self.cells = tuple(int(c) for c in cells)
        if any(n % c for n, c in zip(self.n, self.cells, strict=True)):
            raise ValueError("grid points per axis must be a multiple of the cell count")
        self.scale = float(kinetic_scale_eV_ang2)
        self._cache: OrderedDict = OrderedDict()
        self.cache_size = int(cache_size)
        mx = np.rint(np.fft.fftfreq(self.n[0]) * self.n[0]).astype(int)
        my = np.rint(np.fft.fftfreq(self.n[1]) * self.n[1]).astype(int)
        self.blocks = []
        for rx in range(self.cells[0]):
            for ry in range(self.cells[1]):
                ix = np.flatnonzero(mx % self.cells[0] == rx)
                iy = np.flatnonzero(my % self.cells[1] == ry)
                index = (np.repeat(ix, iy.size), np.tile(iy, ix.size))
                dx = (index[0][:, None] - index[0][None, :]) % self.n[0]
                dy = (index[1][:, None] - index[1][None, :]) % self.n[1]
                self.blocks.append((index, self.u[dx, dy]))

    def spectrum(self, tilt):
        key = (round(float(tilt[0]), 9), round(float(tilt[1]), 9))
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            out = []
            for index, coupling in self.blocks:
                k2 = (self.kx[index[0]] + tilt[0]) ** 2 + (self.ky[index[1]] + tilt[1]) ** 2
                values, vectors = np.linalg.eigh(coupling + np.diag(self.scale * k2))
                out.append((index, values, vectors))
            self._cache[key] = out
        return self._cache[key]

    def energies(self, tilt=(0.0, 0.0)) -> np.ndarray:
        return np.sort(np.concatenate([v for _, v, _ in self.spectrum(tilt)]))

    def gap_cut(self, tilt, target_eV, window_eV):
        """Choose the largest level-free interval centred within the cut window.

        The reported width is twice the distance to the nearest level, not
        necessarily the full enclosing gap. An empty/full retained spectrum
        uses the nominal cut and its one-sided level clearance.
        """
        if not np.isfinite(target_eV) or not np.isfinite(window_eV) or window_eV < 0:
            raise ValueError("matching energy and nonnegative cut window must be finite")
        e = self.energies(tilt)
        if target_eV <= e[0]:
            return float(target_eV), float(2 * (e[0] - target_eV))
        if target_eV >= e[-1]:
            return float(target_eV), float(2 * (target_eV - e[-1]))
        mid = 0.5 * (e[1:] + e[:-1])
        low = np.maximum(e[:-1], target_eV - window_eV)
        high = np.minimum(e[1:], target_eV + window_eV)
        cut = np.clip(mid, low, high)
        clearance = np.where(high >= low, np.minimum(cut - e[:-1], e[1:] - cut), -1.0)
        best = int(np.argmax(clearance))
        return float(cut[best]), float(2 * clearance[best])

    def split(self, phi, tilt, cut_eV):
        """Return ``(weight below cut, phi below, phi above)`` of a grid state."""
        a = np.fft.fft2(phi)
        below = np.zeros_like(a)
        for index, values, vectors in self.spectrum(tilt):
            keep = vectors[:, values < cut_eV]
            below[index] = keep @ (keep.conj().T @ a[index])
        total = float(np.sum(np.abs(a) ** 2))
        weight = float(np.sum(np.abs(below) ** 2)) / total if total > 0 else 0.0
        return weight, np.fft.ifft2(below), np.fft.ifft2(a - below)


def projective_hook(projector: GridProjector, cut_eV: float, seed: int, window_eV=None, gaps=None):
    """Measure P(cut) with per-electron streams; absorb outcome F.

    With ``window_eV`` the cut at each tilt is ``projector.gap_cut(tilt,
    cut_eV, window_eV)`` and each gap width is appended to ``gaps``.

    The first call per electron (entry) projects deterministically; the
    caller multiplies the ensemble by the entrance P weight, which removes
    entry sampling noise for a pure entry state. Streams come from
    ``SeedSequence((seed, 0x548, electron))``, disjoint from the MCWF Philox
    keys, so the measurement never perturbs jump draws.
    Validation: cr-axial-zone-handoff-accounting
    """
    streams: dict[int, np.random.Generator] = {}

    def hook(electron, phi, tilt):
        cut = cut_eV
        if window_eV is not None:
            cut, gap = projector.gap_cut(tilt, cut_eV, window_eV)
            if gaps is not None:
                gaps.append(gap)
        weight, below, _ = projector.split(phi, tilt, cut)
        rng = streams.get(electron)
        if rng is None:
            streams[electron] = np.random.default_rng(
                np.random.SeedSequence((seed, 0x548, electron))
            )
            return below if weight > 0 else None
        return below if rng.random() < weight else None

    return hook


def _setup(cells, shape, mechanisms, energy_eV):
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF

    profile = axial_profile("silicon", (0, 0, 1))
    dissipator = axial_dissipator(profile, energy_eV, cells, shape, mechanisms=mechanisms)
    potential = profile.on_grid(cells, shape)
    solver = AxialMCWF(dissipator, energy_eV, potential)
    projector = GridProjector(
        potential, dissipator.grid.kx, dissipator.grid.ky, cells, HBAR2_2M_EV_ANG2 / solver.gamma
    )
    return profile, solver, projector


def _job(args):
    cells, shape, mechanisms, energy, cut, window, u_max, depths, n, seed, step = args
    from pyrite.montecarlo.channeling import states

    _, solver, projector = _setup(cells, shape, mechanisms, energy)
    gaps: list[float] = []
    hook = None if cut is None else projective_hook(projector, cut, seed, window, gaps)

    def bound(phi, tilt):
        return np.array([projector.split(phi, tilt, u_max)[0]])

    # A 20 Å waist periodizes to the zero-order plane wave on any supercell here.
    entry = states.gaussian(waist_ang=20.0)
    scale = 1.0
    if cut is not None:
        sampled = solver.grid.sample(entry)
        scale = projector.split(sampled, (0.0, 0.0), projector.gap_cut((0.0, 0.0), cut, window)[0])[
            0
        ]
    result = solver.run(
        entry,
        depths,
        n,
        seed=seed,
        step_ang=step,
        observable=bound,
        jump_hook=hook,
    )
    bound_mean = scale * result.observables[:, 0]
    gap_stats = [min(gaps), float(np.median(gaps))] if gaps else None
    return cut, seed, bound_mean.tolist(), result.jumps, gap_stats


def _pooled_deficit(ref_x, x):
    """Ratio of pooled means ``1 - <x>/<ref_x>`` per depth, with a
    delete-one-chunk jackknife SEM (``None`` for a single chunk); 0 where the
    pooled reference exposure is 0."""

    def ratio(r, m):
        rs, ms = r.mean(0), m.mean(0)
        return np.where(rs > 0, 1.0 - ms / np.where(rs > 0, rs, 1.0), 0.0)

    full, n = ratio(ref_x, x), len(ref_x)
    if n < 2:
        return full, None
    keep = ~np.eye(n, dtype=bool)
    loo = np.array([ratio(ref_x[k], x[k]) for k in keep])
    return full, np.sqrt((n - 1) / n * np.sum((loo - loo.mean(0)) ** 2, 0)).tolist()


def _gap(energies, target):
    if not np.isfinite(target) or target <= energies[0] or target >= energies[-1]:
        raise ValueError("matching energy must lie inside the represented spectrum")
    return 0.5 * (energies[energies <= target].max() + energies[energies > target].min())


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--points-per-cell", type=int, default=16)
    parser.add_argument("--gap-window-eV", type=float, default=2.0)
    parser.add_argument(
        "--cut-eV",
        type=float,
        action="append",
        dest="cuts_eV",
        help="nominal matching energy; repeat to study selected cuts (default: barrier and one/two well depths above)",
    )
    parser.add_argument("--mechanisms", default="tds,electronic")
    parser.add_argument("--energy-eV", type=float, default=5e6)
    parser.add_argument("--depth-max-ang", type=float, default=3e4)
    parser.add_argument("--depth-step-ang", type=float, default=500.0)
    parser.add_argument("--step-ang", type=float, default=5.0)
    parser.add_argument("--electrons-per-chunk", type=int, default=128)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out")
    ns = parser.parse_args(argv)
    if ns.smoke:
        ns.cells, ns.points_per_cell, ns.mechanisms = 1, 12, "tds"
        ns.depth_max_ang, ns.depth_step_ang, ns.step_ang = 1000.0, 500.0, 10.0
        ns.electrons_per_chunk, ns.chunks, ns.workers = 8, 2, 1
    cells = (ns.cells, ns.cells)
    shape = (ns.cells * ns.points_per_cell,) * 2
    mechanisms = tuple(ns.mechanisms.split(","))
    profile, _, projector = _setup(cells, shape, mechanisms, ns.energy_eV)
    u_min, u_max = profile.extrema()
    energies = projector.energies()
    well = u_max - u_min
    targets = ns.cuts_eV if ns.cuts_eV is not None else (u_max, u_max + well, u_max + 2 * well)
    cuts = [_gap(energies, t) for t in targets]
    window = ns.gap_window_eV
    depths = np.arange(0.0, ns.depth_max_ang + 0.5 * ns.depth_step_ang, ns.depth_step_ang)
    jobs = [
        (
            cells,
            shape,
            mechanisms,
            ns.energy_eV,
            cut,
            window,
            u_max,
            depths.tolist(),
            ns.electrons_per_chunk,
            seed,
            ns.step_ang,
        )
        for seed in range(ns.chunks)
        for cut in [None, *cuts]
    ]
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        results = list(pool.map(_job, jobs))
    by = {}
    for cut, seed, bound, jumps, gap_stats in results:
        by.setdefault(cut, {})[seed] = (np.array(bound), jumps, gap_stats)
    seeds = sorted(by[None])
    reference = np.array([by[None][s][0] for s in seeds])

    def exposure(values):
        return np.concatenate(
            [
                np.zeros((len(values), 1)),
                np.cumsum(0.5 * (values[:, 1:] + values[:, :-1]) * np.diff(depths), axis=1),
            ],
            axis=1,
        )

    ref_x = exposure(reference)
    rows = []
    for cut in cuts:
        model = np.array([by[cut][s][0] for s in seeds])
        x = exposure(model)
        deficit, sem = _pooled_deficit(ref_x, x)
        gap_stats = [by[cut][s][2] for s in seeds if by[cut][s][2] is not None]
        rows.append(
            {
                "matching_energy_eV": float(cut),
                "bound_population": model.mean(0).tolist(),
                "bound_exposure_ang": x.mean(0).tolist(),
                "exposure_deficit": deficit.tolist(),
                "exposure_deficit_sem": sem,
                "absorbed": int(sum(by[cut][s][1].get("absorbed", 0) for s in seeds)),
                "min_gap_eV": min(g[0] for g in gap_stats) if gap_stats else None,
                "median_chunk_median_gap_eV": (
                    float(np.median([g[1] for g in gap_stats])) if gap_stats else None
                ),
            }
        )
    out = {
        "status": "research-only MCWF return-flux study; not a dechanneling prediction",
        "settings": {**vars(ns), "u_min_eV": u_min, "u_max_eV": u_max},
        "depths_ang": depths.tolist(),
        "reference_bound_population": reference.mean(0).tolist(),
        "reference_bound_exposure_ang": ref_x.mean(0).tolist(),
        "reference_jumps": {
            k: int(sum(by[None][s][1][k] for s in seeds)) for k in by[None][seeds[0]][1]
        },
        "rows": rows,
    }
    if ns.out:
        with open(ns.out, "w") as handle:
            json.dump(out, handle, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
