"""Fixed-N frozen production-kernel roll experiment (follow-up to #1527).

No profile, adaptive stopping, continuum, or grid-policy changes. Run with
an explicit common energy axis; all arms see the same retained transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

SECTOR = math.pi / 3
RNG_NAMESPACE = 0xF1527


@dataclass(frozen=True)
class RollPlan:
    """Slot per incident primary, and one roll in radians per slot."""

    labels: np.ndarray
    rolls: np.ndarray

    def occupancy(self):
        return np.bincount(self.labels, minlength=self.rolls.size)


def independent_rolls(ids, seed):
    """Counter-addressed SplitMix64 permutation, check-owned RNG namespace.

    Each global primary ID has a stable pseudorandom uniform draw; no state
    is consumed from transport. The seed is a replicate seed, not case RNG.
    This is a diagnostic RNG, not a proposed production RNG contract.
    """
    ids = np.asarray(ids, dtype=np.uint64)
    key = np.random.SeedSequence([seed, RNG_NAMESPACE, 0]).generate_state(
        1, dtype=np.uint64
    )[0]
    x = ids ^ key
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (SECTOR / 2**53)


def grouped_plan(n, strata, histories_per_roll, seed):
    """Independent stratified randomizations in fixed global-primary groups.

    Group size is strata * histories_per_roll, independent of runtime
    blocks and final N. Every slot has a jittered roll; a random permutation
    of strata assigns primary IDs cyclically, including in partial groups.
    Each primary is marginally uniform. Shared-roll scores are correlated;
    only independent orientation-seed replications estimate conditional MC
    uncertainty here, never the existing per-electron IID stopping rule.
    """
    if min(n, strata, histories_per_roll) < 1 or seed < 0:
        raise ValueError("positive counts and nonnegative seed required")
    group_size = strata * histories_per_roll
    ng = (n + group_size - 1) // group_size
    labels = np.empty(n, dtype=np.int64)
    rolls = np.empty(ng * strata, dtype=np.float64)
    for g in range(ng):
        rng = np.random.default_rng(np.random.SeedSequence([seed, RNG_NAMESPACE, 1, g]))
        start = g * group_size
        stop = min(n, start + group_size)
        slots = slice(g * strata, (g + 1) * strata)
        rolls[slots] = (np.arange(strata) + rng.random(strata)) * (SECTOR / strata)
        labels[start:stop] = g * strata + rng.permutation(strata)[
            np.arange(stop - start) % strata
        ]
    return RollPlan(labels, rolls)


def independent_plan(n, seed):
    ids = np.arange(n, dtype=np.int64)
    return RollPlan(ids, independent_rolls(ids, seed))


def midpoint_rolls(n, phase=0.0):
    """Phase is measured in units of node spacing, modulo one sector."""
    return ((np.arange(n) + 0.5 + phase) * (SECTOR / n)) % SECTOR


def row_partitions(ids, plan):
    """Gather each occupied row group once, retaining original row order.

    IDs name incident primaries, not secondary tracks. Zero-row histories
    still belong to plan.occupancy() and the global normalization.
    """
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("expected integer primary IDs")
    if ids.size and (ids.min() < 0 or ids.max() >= plan.labels.size):
        raise ValueError("primary IDs outside the retained incident population")
    slots = plan.labels[ids]
    order = np.argsort(slots, kind="stable")
    sorted_slots = slots[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(sorted_slots)) + 1, order.size]
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        if stop > start:
            yield int(sorted_slots[start]), order[start:stop]


def _sync():
    from pyrite._backend import BACKEND

    cp = getattr(BACKEND, "cp", None)
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()


def _host(value):
    from pyrite._backend import _to_cpu

    return np.asarray(_to_cpu(value))


def _rss_mib():
    import resource
    import sys

    scale = 2**20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def _device_pool_mib():
    from pyrite._backend import BACKEND

    cp = getattr(BACKEND, "cp", None)
    if cp is None:
        return None
    return cp.get_default_memory_pool().total_bytes() / 2**20


def _fingerprint(segs):
    from pyrite.montecarlo.spectrum.cxr._kernels import _SEG_ARRAYS

    h = hashlib.sha256()
    for key in sorted(set(_SEG_ARRAYS).intersection(segs)):
        arr = np.ascontiguousarray(_host(segs[key]))
        h.update(f"{key}:{arr.dtype.str}:{arr.shape}:".encode())
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def _case(args, energy):
    # Same public lowering as #1480 campaign and #1527 tail checks, with
    # explicit roll-independent bin axis. This is NOT the saved CLI profile.
    from pyrite import Scene
    from pyrite.beams import Beam, PointBunch, Spot
    from pyrite.campaign.lowering import lower_simulation
    from pyrite.campaign.resolution import resolve_simulation
    from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.geometry import Footprint, Slab

    scene = Scene(
        Beam(args.energy, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", args.thick_nm * 10, tilt_deg=args.tilt,
             tilt_azim_deg=args.tilt_azim, mosaic=args.mosaic, footprint=Footprint(5.0, 5.0)),
        Detector(observation_angle_deg=args.theta),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=args.histories, brem_histories=args.histories),
        cxr_reflections=CxrReflections(
            fiber=FiberAverage(cutoff_invang=args.cutoff, azimuth_count=args.reference,
                               reflection_batch_size=args.batch),
            mosaic_nodes=args.mosaic_nodes, mosaic_route=args.mosaic_route,
        ),
        energy_bins=EnergyBins(cxr=energy),
        seed=args.transport_seed,
    )
    physics = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
    return lower_simulation(resolve_simulation(scene, physics=physics, numerics=numerics))


class FrozenScorer:
    """Reuse the production fiber domain, table batching and CXR line seam.

    Unlike a production mode this only scores retained, complete populations.
    For every subset keep the original Ne, original IDs and flight metadata;
    its spectrum is additive and must NOT also be weighted by its occupancy.
    """

    def __init__(self, case, tp, energy):
        from pyrite.montecarlo import runner
        from pyrite.montecarlo.runner.electron_blocks import _electron_ids
        from pyrite.montecarlo.runner.fiber_emission import (
            FIBER_BATCH_CASE_KEY, FIBER_CASE_KEY, fiber_domain, validate_fiber_case,
        )

        validate_fiber_case(case)
        self.case, self.tp, self.energy = case, tp, energy
        self.ne = int(tp["Ne_lines"])
        self.segs = runner._segments_on_device(tp["segs"])
        self.ids = _electron_ids(self.segs)
        if int(self.segs["Ne"]) != self.ne:
            raise ValueError("check requires one equal-count retained line population")
        if self.ids.size != self.segs["L_ang"].size:
            raise ValueError("primary IDs do not align with trajectory rows")
        if self.ids.size and (self.ids.min() < 0 or self.ids.max() >= self.ne):
            raise ValueError("check requires global retained primary IDs in [0, Ne)")
        if self.segs.get("cone_split") is not None:
            raise ValueError("cone split is outside this diagnostic")
        self.domain = fiber_domain(case)
        self.lookup = self.domain._thermal_lookup()
        self.basal = [h for h in self.domain.hkls if h[:2] == (0, 0)]
        self.other = [h for h in self.domain.hkls if h[:2] != (0, 0)]
        self.base = {k: v for k, v in case.items()
                     if k not in (FIBER_CASE_KEY, FIBER_BATCH_CASE_KEY)}
        if self.base.get("beam_uvw") is None and self.base.get("surface_hkl") is None:
            self.base["surface_hkl"] = (0, 0, 1)
        self.phi0 = float(case.get("azimuth_rad", 0.0))
        self.batch = int(self.domain.model.reflection_batch_size)

    def subset(self, rows):
        from pyrite.montecarlo.runner.electron_blocks import _take
        from pyrite.montecarlo.spectrum.cxr._kernels import _SEG_ARRAYS

        out = dict(self.segs)
        for key in _SEG_ARRAYS:
            if key in out:
                out[key] = _take(out[key], rows)
        return out

    def parts(self, hkls, rolls, plan=None):
        from pyrite.montecarlo import runner
        from pyrite.montecarlo.spectrum.cxr._bin_queue import shared_bin_lines

        _sync()
        started = perf_counter()
        spec = np.zeros(self.energy.shape, dtype=np.float64)
        calls = row_rolls = 0
        if plan is None:
            partitions = ((i, None) for i in range(len(rolls)))
            weight = 1.0 / len(rolls)
        else:
            if plan.labels.size != self.ne:
                raise ValueError("roll assignment must cover every incident primary")
            partitions = row_partitions(self.ids, plan)
            weight = 1.0
        with shared_bin_lines() as sink:
            sink.weight = weight
            for slot, rows in partitions:
                if not hkls or not self.ids.size:
                    continue
                part = self.segs if rows is None else self.subset(rows)
                row_rolls += int(part["L_ang"].size)
                for start in range(0, len(hkls), self.batch):
                    value = runner._cxr_for_segments(
                        part, self.energy,
                        dict(self.base, hkl_list=hkls[start:start + self.batch],
                             B_ang2=self.lookup.__getitem__,
                             azimuth_rad=self.phi0 + float(rolls[slot])),
                        self.tp["n_hat"], self.case.get("abs_layers"), self.tp.get("groove"),
                        coherent=False, Ne=self.ne, table_cache=None,
                    )
                    spec += weight * _host(value)
                    calls += 1
            _sync()
            before_flush = perf_counter()
            spec += sink.result(spec.shape)
            _sync()
        if not np.isfinite(spec).all():
            raise RuntimeError("nonfinite spectrum")
        seconds = perf_counter() - started
        return spec, dict(
            seconds=seconds, pre_final_flush_s=before_flush - started,
            final_host_flush_s=seconds - (before_flush - started), calls=calls,
            segment_roll_evaluations=row_rolls, peak_process_rss_mib=_rss_mib(),
            device_pool_reserved_mib=_device_pool_mib(),
        )

    def production(self):
        from pyrite.montecarlo import runner

        _sync()
        start = perf_counter()
        value = runner._cxr_for_segments(
            self.segs, self.energy, self.case, self.tp["n_hat"],
            self.case.get("abs_layers"), self.tp.get("groove"), Ne=self.ne,
        )
        spec = _host(value)
        _sync()
        return spec, perf_counter() - start


def metrics(energy, width, spec, reference=None):
    mass = float(np.sum(spec) * width)
    out = {"yield": mass}
    if mass > 0:
        peak = int(np.argmax(spec))
        lo = hi = peak
        while lo > 0 and spec[lo] > spec[peak] / 2:
            lo -= 1
        while hi < spec.size - 1 and spec[hi] > spec[peak] / 2:
            hi += 1
        out.update(peak_eV=float(energy[peak]), peak_density=float(spec[peak]),
                   centroid_eV=float(np.sum(energy * spec) * width / mass),
                   fwhm_eV=float(energy[hi] - energy[lo]))
    if reference is not None:
        ref_mass = float(np.sum(np.abs(reference)) * width)
        ref_peak = float(np.max(np.abs(reference)))
        out.update(
            mass_l1=float(np.sum(np.abs(spec - reference)) * width / ref_mass)
            if ref_mass else None,
            peak_linf=float(np.max(np.abs(spec - reference)) / ref_peak) if ref_peak else None,
            relative_yield_error=float((np.sum(spec) - np.sum(reference)) * width / ref_mass)
            if ref_mass else None,
        )
    return out


def _band_errors(energy, width, spec, reference, bands):
    out = {}
    for lo, hi in bands:
        # Fractional overlap at band boundaries, with bin-mean density.
        overlap = np.maximum(0.0, np.minimum(energy + width / 2, hi)
                             - np.maximum(energy - width / 2, lo))
        ref = float(reference @ overlap)
        mass = float(spec @ overlap)
        out[f"{lo:g}:{hi:g}"] = dict(yield_=mass, reference=ref,
                                      relative_error=(mass - ref) / ref if ref else None)
    return out


def _parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--histories", type=int, default=512)
    p.add_argument("--energy", type=float, default=300)
    p.add_argument("--thick-nm", type=float, default=500)
    p.add_argument("--tilt", type=float, default=50)
    p.add_argument("--tilt-azim", type=float, default=180)
    p.add_argument("--theta", type=float, default=45)
    p.add_argument("--cutoff", type=float, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--mosaic", action="store_true", help="enable catalog target mosaicity")
    p.add_argument("--mosaic-nodes", type=int, default=5)
    p.add_argument("--mosaic-route", choices=("auto", "mc"), default="auto")
    p.add_argument("--energy-start", type=float, default=50)
    p.add_argument("--energy-max", type=float, default=13000)
    p.add_argument("--step", type=float, default=1)
    p.add_argument("--transport-seed", type=int, default=1)
    p.add_argument("--orientation-seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--quadratures", type=int, nargs="*", default=[32, 64, 128])
    p.add_argument("--reference", type=int, default=256)
    p.add_argument("--shifted", type=int, nargs="*", default=[32])
    p.add_argument("--strata", type=int, nargs="*", default=[32, 128])
    p.add_argument("--histories-per-roll", type=int, default=32)
    p.add_argument("--independent", action="store_true")
    p.add_argument("--max-independent-histories", type=int, default=1024)
    p.add_argument("--production-parity", action="store_true")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--band", type=float, nargs=2, action="append", default=[])
    p.add_argument("--transport-core", default="per-electron")
    p.add_argument("--output", type=Path, required=True, help="output stem for JSON/NPZ")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    counts = [a.histories, a.batch, a.mosaic_nodes, a.reference, a.histories_per_roll,
              a.max_independent_histories, *a.quadratures, *a.shifted, *a.strata]
    if min(counts) < 1 or a.transport_seed < 0 or min(a.orientation_seeds) < 0:
        p.error("positive counts and nonnegative seeds required")
    if len(set(a.orientation_seeds)) != len(a.orientation_seeds):
        p.error("orientation seeds must be distinct independent replicate keys")
    if a.reference < max([0, *a.quadratures, *a.shifted]):
        p.error("reference must be at least as fine as every quadrature")
    if a.independent and a.histories > a.max_independent_histories:
        p.error("independent scalar-table arm is expensive; explicitly raise its history cap")
    if a.mosaic and a.mosaic_route != "mc":
        p.error("full-fiber target mosaicity requires --mosaic-route mc")
    if not (a.step > 0 and 10 <= a.energy_start < a.energy_max <= a.energy * 1000):
        p.error("invalid common energy grid")
    if any(not a.energy_start <= lo < hi <= a.energy_max for lo, hi in a.band):
        p.error("bands must lie inside the requested common axis")
    return a


def main():
    args = _parse()
    from pyrite._backend import BACKEND
    from pyrite.montecarlo import runner
    energy = np.arange(args.energy_start, args.energy_max, args.step)
    if energy.size < 2:
        raise ValueError("need at least two photon grid nodes")
    case = _case(args, energy)
    t0 = perf_counter()
    tp = runner._transport_case(case, transport_core=args.transport_core)
    _sync()
    transport_s = perf_counter() - t0
    if not np.array_equal(np.asarray(tp["E_grid"]), energy):
        raise RuntimeError("lowering changed the requested common photon axis")
    scorer = FrozenScorer(case, tp, energy)
    if scorer.ne != args.histories:
        raise ValueError("realized incident count differs from requested fixed N")
    sha = os.environ.get("PYRITE_GIT_SHA")
    if not sha:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            sha = "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        status = None
    edges = np.r_[energy - args.step / 2, energy[-1] + args.step / 2]
    record = dict(
        git_sha=sha, git_status=status, settings=vars(args), case=case,
        backend=type(BACKEND).__name__, segment_dtype=str(scorer.segs["L_ang"].dtype),
        n_hat=_host(tp["n_hat"]).tolist(), transport_s=transport_s,
        transport_sha256=_fingerprint(scorer.segs), histories=scorer.ne,
        rows=scorer.ids.size, represented_primaries=int(np.unique(scorer.ids).size),
        basal_reflections=len(scorer.basal), nonbasal_reflections=len(scorer.other),
        rng_namespace=RNG_NAMESPACE, energy_edges_eV=[float(edges[0]), float(edges[-1])],
        arms={}, ensembles={},
    )
    if args.warmup:
        _, record["warmup"] = scorer.parts(scorer.other, [0.0])
    basal, record["basal"] = scorer.parts(scorer.basal, [0.0])
    spectra = {"E_eV": energy, "edges_eV": edges, "basal": basal}
    groups = {}

    def save_arm(name, rolls, plan=None, setup_s=0.0):
        nonbasal, arm = scorer.parts(scorer.other, rolls, plan)
        arm["roll_setup_s"] = setup_s
        arm["radiation_plus_setup_s"] = arm["seconds"] + setup_s + record["basal"]["seconds"]
        arm["history_roll_evaluations"] = scorer.ne * len(rolls) if plan is None else scorer.ne
        if plan is not None:
            occ = plan.occupancy()
            arm.update(roll_slots=int(occ.size), occupied_slots=int(np.count_nonzero(occ)),
                       max_occupancy=int(occ.max()))
        spectra[name + "_rolls_rad"] = np.asarray(rolls)
        if plan is not None:
            spectra[name + "_labels"] = plan.labels
            spectra[name + "_occupancy"] = occ
        spectra[name + "_nonbasal"] = nonbasal
        spectra[name + "_total"] = nonbasal + basal
        record["arms"][name] = arm
        print(json.dumps({"arm": name, "seconds": arm["seconds"],
                          "nonbasal_yield": float(nonbasal.sum() * args.step),
                          "calls": arm["calls"]}), flush=True)

    for n in sorted({*args.quadratures, args.reference}):
        save_arm(f"Q{n}", midpoint_rolls(n))
    for n in sorted(set(args.shifted)):
        save_arm(f"Q{n}_shift", midpoint_rolls(n, phase=0.5))
    for seed in args.orientation_seeds:
        for k in sorted(set(args.strata)):
            start = perf_counter()
            plan = grouped_plan(scorer.ne, k, args.histories_per_roll, seed)
            setup_s = perf_counter() - start
            name = f"S{k}_seed{seed}"
            save_arm(name, plan.rolls, plan, setup_s)
            groups.setdefault(f"S{k}", []).append(name)
        if args.independent:
            start = perf_counter()
            plan = independent_plan(scorer.ne, seed)
            setup_s = perf_counter() - start
            name = f"I_seed{seed}"
            save_arm(name, plan.rolls, plan, setup_s)
            groups.setdefault("I", []).append(name)
    ref_name = f"Q{args.reference}"
    ref = spectra[ref_name + "_nonbasal"]
    reference_total = ref + basal
    for name, arm in record["arms"].items():
        spec = spectra[name + "_nonbasal"]
        arm["nonbasal_metrics"] = metrics(energy, args.step, spec, ref)
        arm["total_metrics"] = metrics(energy, args.step, spec + basal, reference_total)
        arm["nonbasal_bands"] = _band_errors(energy, args.step, spec, ref, args.band)
    for label, names in groups.items():
        samples = np.stack([spectra[name + "_nonbasal"] for name in names])
        mean = samples.mean(axis=0)
        spectra[label + "_mean_nonbasal"] = mean
        spectra[label + "_mean_total"] = mean + basal
        elapsed = [record["arms"][name]["radiation_plus_setup_s"] for name in names]
        yields = samples.sum(axis=1) * args.step
        info = dict(replicates=names, mean_metrics=metrics(energy, args.step, mean, ref),
                    mean_wall_s=float(np.mean(elapsed)), yield_mean=float(yields.mean()),
                    bands=_band_errors(energy, args.step, mean, ref, args.band))
        if len(names) > 1:
            se = samples.std(axis=0, ddof=1) / math.sqrt(len(names))
            spectra[label + "_se_nonbasal"] = se
            info["yield_se"] = float(yields.std(ddof=1) / math.sqrt(len(names)))
            ref_yield = float(ref.sum() * args.step)
            info["conditional_yield_relative_fom"] = (
                ref_yield**2 / (float(yields.var(ddof=1)) * float(np.mean(elapsed)))
                if yields.var(ddof=1) > 0 and np.mean(elapsed) > 0 else None
            )
            for lo, hi in args.band:
                overlap = np.maximum(0.0, np.minimum(energy + args.step / 2, hi)
                                     - np.maximum(energy - args.step / 2, lo))
                band_yields = samples @ overlap
                info["bands"][f"{lo:g}:{hi:g}"]["se"] = float(
                    band_yields.std(ddof=1) / math.sqrt(len(names)))
        record["ensembles"][label] = info
    if args.production_parity:
        production, elapsed = scorer.production()
        spectra["production_total"] = production
        record["production_parity"] = dict(seconds=elapsed, **metrics(
            energy, args.step, reference_total, production))
    record["reference"] = {"arm": ref_name, "certified": False,
                           "note": "candidate reference; inspect successive Q and phase errors"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Append extensions rather than replacing a user's dotted run stem.
    with Path(str(args.output) + ".json").open("w") as handle:
        json.dump(record, handle, indent=2, default=str, allow_nan=False)
    np.savez_compressed(str(args.output) + ".npz", **spectra)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        axes[0].plot(energy, basal, label="basal")
        for name in (ref_name, "Q32", "Q32_shift"):
            if name + "_nonbasal" in spectra:
                axes[1].plot(energy, spectra[name + "_nonbasal"], label=name)
        for label in groups:
            mean = spectra[label + "_mean_nonbasal"]
            axes[1].plot(energy, mean, label=label + " mean")
            if label + "_se_nonbasal" in spectra:
                se = spectra[label + "_se_nonbasal"]
                axes[1].fill_between(energy, mean - se, mean + se, alpha=0.15)
        axes[0].set_ylabel("Basal density")
        axes[1].set_ylabel("Non-basal density")
        axes[1].set_xlabel("Photon energy [eV]")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(str(args.output) + ".png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
