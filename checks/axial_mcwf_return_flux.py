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

``--loss-coupling`` additionally records bounded, hook-event-weighted
histograms of cuts, clearances, ||F W P||_2 and sector separation for the
production absorptive W [eV]. Their ratio is a diagnostic scale, not a
finite-depth leakage bound. No propagation or random draws are changed.

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

    def __init__(
        self,
        potential_eV,
        kx,
        ky,
        cells,
        kinetic_scale_eV_ang2,
        cache_size=160,
        *,
        absorptive_eV=None,
    ):
        self.n = potential_eV.shape
        self.u = np.fft.fft2(potential_eV) / potential_eV.size
        self.kx, self.ky = np.asarray(kx), np.asarray(ky)
        self.cells = tuple(int(c) for c in cells)
        if any(n % c for n, c in zip(self.n, self.cells, strict=True)):
            raise ValueError("grid points per axis must be a multiple of the cell count")
        self.scale = float(kinetic_scale_eV_ang2)
        self._cache: OrderedDict = OrderedDict()
        self.cache_size = int(cache_size)
        self._loss_cache: OrderedDict = OrderedDict()
        self.loss_blocks = []
        if absorptive_eV is not None:
            w = np.asarray(absorptive_eV, dtype=float)
            if w.shape != self.n or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("absorptive_eV must be a finite nonnegative grid array")
            for axis, (n, c) in enumerate(zip(self.n, self.cells, strict=True)):
                if not np.allclose(w, np.roll(w, n // c, axis), rtol=1e-12, atol=1e-12):
                    raise ValueError("absorptive_eV must be periodic on each crystal cell")
            loss_fourier = np.fft.fft2(w) / w.size
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
                if absorptive_eV is not None:
                    self.loss_blocks.append(loss_fourier[dx, dy])

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

    def cross_cut_loss(self, tilt, cut_eV):
        """Measure ||F W P||_2 [eV] for H_eff = H - i W at this tilt.

        The full represented spectrum sets the sector separation. Their ratio
        is a diagnostic scale, NOT a finite-depth or photon-error bound.
        Empty/full sectors have zero cross norm and no separation or ratio.
        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        if not self.loss_blocks or not np.isfinite(cut_eV):
            raise ValueError("finite cut and absorptive_eV are required")
        key = (round(float(tilt[0]), 9), round(float(tilt[1]), 9), float(cut_eV))
        if key in self._loss_cache:
            self._loss_cache.move_to_end(key)
            return dict(self._loss_cache[key])
        norm, retained_max, outgoing_min = 0.0, -np.inf, np.inf
        for (_, values, vectors), loss in zip(self.spectrum(tilt), self.loss_blocks, strict=True):
            keep = values < cut_eV
            if keep.any():
                retained_max = max(retained_max, float(values[keep].max()))
            if (~keep).any():
                outgoing_min = min(outgoing_min, float(values[~keep].min()))
            if keep.any() and (~keep).any():
                cross = vectors[:, ~keep].conj().T @ loss @ vectors[:, keep]
                norm = max(norm, float(np.linalg.svd(cross, compute_uv=False)[0]))
        separation = outgoing_min - retained_max
        result = {
            "loss_cross_norm_eV": norm,
            "sector_separation_eV": float(separation) if np.isfinite(separation) else None,
            "loss_to_separation_ratio": norm / separation if np.isfinite(separation) else None,
        }
        if len(self._loss_cache) >= self.cache_size:
            self._loss_cache.popitem(last=False)
        self._loss_cache[key] = result
        return dict(result)


class MatchingDiagnostics:
    """Bounded, hook-event-weighted histograms; no additional random draws.

    Counts have len(edges)+1 bins: below first edge, half-open interior bins,
    and at/above last edge. No histories or wavefunctions are retained.
    Validation: cr-axial-zone-handoff-accounting (unverified).
    """

    def __init__(self, target_eV, window_eV):
        positive_edges = np.r_[0.0, np.logspace(-12, 8, 81)]
        self.data = {}
        for name in (
            "selected_cut_eV",
            "clearance_eV",
            "loss_cross_norm_eV",
            "sector_separation_eV",
            "loss_to_separation_ratio",
        ):
            edges = (
                np.unique(np.linspace(target_eV - window_eV, target_eV + window_eV, 33))
                if name == "selected_cut_eV"
                else positive_edges
            )
            self.data[name] = {
                "edges": edges.tolist(),
                "counts": [0] * (len(edges) + 1),
                "min": None,
                "max": None,
                "missing": 0,
            }

    def observe(self, projector, tilt, cut, gap):
        values = {
            "selected_cut_eV": cut,
            "clearance_eV": gap / 2,
            **projector.cross_cut_loss(tilt, cut),
        }
        for name, value in values.items():
            row = self.data[name]
            if value is None:
                row["missing"] += 1
                continue
            row["counts"][int(np.searchsorted(row["edges"], value, side="right"))] += 1
            row["min"] = value if row["min"] is None else min(row["min"], value)
            row["max"] = value if row["max"] is None else max(row["max"], value)


def projective_hook(
    projector: GridProjector,
    cut_eV: float,
    seed: int,
    window_eV=None,
    gaps=None,
    diagnostics=None,
):
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
        if diagnostics is not None:
            if window_eV is None:
                gap = 2 * float(np.min(np.abs(projector.energies(tilt) - cut)))
            diagnostics.observe(projector, tilt, cut, gap)
        weight, below, _ = projector.split(phi, tilt, cut)
        rng = streams.get(electron)
        if rng is None:
            streams[electron] = np.random.default_rng(
                np.random.SeedSequence((seed, 0x548, electron))
            )
            return below if weight > 0 else None
        return below if rng.random() < weight else None

    return hook


def _setup(cells, shape, mechanisms, energy_eV, *, loss_coupling=False):
    from pyrite.materials.channeling.axial import axial_profile
    from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF

    profile = axial_profile("silicon", (0, 0, 1))
    dissipator = axial_dissipator(profile, energy_eV, cells, shape, mechanisms=mechanisms)
    potential = profile.on_grid(cells, shape)
    solver = AxialMCWF(dissipator, energy_eV, potential)
    projector = GridProjector(
        potential,
        dissipator.grid.kx,
        dissipator.grid.ky,
        cells,
        HBAR2_2M_EV_ANG2 / solver.gamma,
        absorptive_eV=dissipator.absorptive_eV if loss_coupling else None,
    )
    return profile, solver, projector


def _job(args):
    cells, shape, mechanisms, energy, cut, window, u_max, depths, n, seed, step, coupling = args[
        :12
    ]
    mode = args[12] if len(args) > 12 else "split"
    if mode not in ("split", "exact", "projected") or (mode == "projected" and cut is None):
        raise ValueError("invalid no-jump mode or missing projected cut")
    from pyrite.montecarlo.channeling import states

    _, solver, projector = _setup(
        cells, shape, mechanisms, energy, loss_coupling=coupling or mode != "split"
    )
    gaps: list[float] = []
    diagnostics = MatchingDiagnostics(cut, window) if coupling and cut is not None else None
    hook = None if cut is None else projective_hook(projector, cut, seed, window, gaps, diagnostics)

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
    kwargs = dict(seed=seed, step_ang=step, observable=bound, jump_hook=hook)
    if mode == "split":
        result = solver.run(entry, depths, n, **kwargs)
    else:
        from pyrite.montecarlo.channeling.axial_projection import run_spectral_mcwf

        result = run_spectral_mcwf(
            solver,
            projector,
            entry,
            depths,
            n,
            cut_eV=cut if mode == "projected" else None,
            window_eV=window,
            **kwargs,
        )
    bound_mean = scale * result.observables[:, 0]
    gap_stats = [min(gaps), float(np.median(gaps))] if gaps else None
    return (
        cut if mode == "split" else (mode, cut),
        seed,
        bound_mean.tolist(),
        result.jumps,
        gap_stats,
        (diagnostics.data if diagnostics is not None else None),
    )


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


def _paired_deficit_change(reference, hook, projected):
    """Projected-minus-hook pooled deficit with paired delete-one-chunk SEM.

    The common reference denominator is retained for every leave-one-out ratio.
    No confidence bound or physical leakage probability is inferred.
    """

    def ratio(r, h, p):
        denominator = r.mean(0)
        return np.where(
            denominator > 0,
            (h.mean(0) - p.mean(0)) / np.where(denominator > 0, denominator, 1.0),
            0.0,
        )

    full, n = ratio(reference, hook, projected), len(reference)
    if n < 2:
        return full, None
    keep = ~np.eye(n, dtype=bool)
    loo = np.array([ratio(reference[k], hook[k], projected[k]) for k in keep])
    sem = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(0)) ** 2, 0))
    return full, sem.tolist()


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--points-per-cell", type=int, default=16)
    parser.add_argument("--gap-window-eV", type=float, default=2.0)
    parser.add_argument(
        "--projected-no-jump",
        action="store_true",
        help="compare full-space and retained-space matrix-exponential no-jump propagation (CPU research)",
    )
    parser.add_argument(
        "--loss-coupling",
        action="store_true",
        help="record cut/clearance and cross-cut absorptive-loss histograms (extra dense algebra)",
    )
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
            ns.loss_coupling,
        )
        for seed in range(ns.chunks)
        for cut in [None, *cuts]
    ]
    if ns.projected_no_jump:
        base_jobs = list(jobs)
        jobs.extend((*job, "exact") for job in base_jobs)
        jobs.extend((*job, "projected") for job in base_jobs if job[4] is not None)
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        results = list(pool.map(_job, jobs))
    by = {}
    for cut, seed, bound, jumps, gap_stats, diagnostics in results:
        by.setdefault(cut, {})[seed] = (np.array(bound), jumps, gap_stats, diagnostics)
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

    def summarize(cut, key, reference_exposure):
        model = np.array([by[key][s][0] for s in seeds])
        x = exposure(model)
        deficit, sem = _pooled_deficit(reference_exposure, x)
        gap_stats = [by[key][s][2] for s in seeds if by[key][s][2] is not None]
        return {
            "matching_energy_eV": float(cut),
            "bound_population": model.mean(0).tolist(),
            "bound_exposure_ang": x.mean(0).tolist(),
            "bound_exposure_ang_by_chunk": [
                {"seed": s, "values": values.tolist()} for s, values in zip(seeds, x, strict=True)
            ],
            "exposure_deficit": deficit.tolist(),
            "exposure_deficit_sem": sem,
            "absorbed": int(sum(by[key][s][1].get("absorbed", 0) for s in seeds)),
            "jumps": {k: int(sum(by[key][s][1][k] for s in seeds)) for k in by[key][seeds[0]][1]},
            "min_gap_eV": min(g[0] for g in gap_stats) if gap_stats else None,
            "median_chunk_median_gap_eV": (
                float(np.median([g[1] for g in gap_stats])) if gap_stats else None
            ),
            "matching_diagnostics_by_chunk": [
                {"seed": s, "histograms": by[key][s][3]} for s in seeds
            ]
            if ns.loss_coupling
            else None,
        }

    rows = [summarize(cut, cut, ref_x) for cut in cuts]
    out = {
        "status": "research-only MCWF return-flux study; not a dechanneling prediction",
        "settings": {**vars(ns), "u_min_eV": u_min, "u_max_eV": u_max},
        "depths_ang": depths.tolist(),
        "reference_bound_population": reference.mean(0).tolist(),
        "reference_bound_exposure_ang": ref_x.mean(0).tolist(),
        "reference_bound_exposure_ang_by_chunk": [
            {"seed": s, "values": values.tolist()} for s, values in zip(seeds, ref_x, strict=True)
        ],
        "reference_jumps": {
            k: int(sum(by[None][s][1][k] for s in seeds)) for k in by[None][seeds[0]][1]
        },
        "rows": rows,
    }
    if ns.projected_no_jump:
        exact_reference = np.array([by[("exact", None)][s][0] for s in seeds])
        exact_x = exposure(exact_reference)
        changes = []
        for cut in cuts:
            hook = exposure(np.array([by[("exact", cut)][s][0] for s in seeds]))
            projected = exposure(np.array([by[("projected", cut)][s][0] for s in seeds]))
            change, sem = _paired_deficit_change(exact_x, hook, projected)
            changes.append(
                {
                    "matching_energy_eV": float(cut),
                    "projected_minus_hook_deficit": change.tolist(),
                    "sem": sem,
                }
            )
        out["spectral_comparison"] = {
            "model": "finite-grid-spectral-expm-v1",
            "jump_time_rule": "first step crossing; at most one jump per step",
            "reference_bound_population": exact_reference.mean(0).tolist(),
            "reference_bound_exposure_ang": exact_x.mean(0).tolist(),
            "reference_bound_exposure_ang_by_chunk": [
                {"seed": s, "values": values.tolist()}
                for s, values in zip(seeds, exact_x, strict=True)
            ],
            "reference_jumps": {
                k: int(sum(by[("exact", None)][s][1][k] for s in seeds))
                for k in by[("exact", None)][seeds[0]][1]
            },
            "hook_rows": [summarize(cut, ("exact", cut), exact_x) for cut in cuts],
            "projected_rows": [summarize(cut, ("projected", cut), exact_x) for cut in cuts],
            "changes": changes,
        }
    if ns.out:
        with open(ns.out, "w") as handle:
            json.dump(out, handle, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
