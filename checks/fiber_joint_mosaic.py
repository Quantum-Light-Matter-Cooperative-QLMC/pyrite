"""Frozen-transport comparison of joint mosaic/roll sampling and GH mosaic.

The case seed is varied only in scoring, NOT transport, to isolate orientation
sampling variability. GH references retain the identical history rolls. K=1
samples one crystallite per primary; larger K averages K tilts at the same roll.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from fiber_randomized_rolls import FrozenScorer, _case, _host, _sync, metrics


def scalar_oracle(scorer, seed, draws):
    """One primary at a time with explicit scalar roll and tilt matrices."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.geometry import _small_tilt_R
    from pyrite.montecarlo.spectrum.cxr._fiber_history import HistoryRollSeed
    from pyrite.montecarlo.spectrum.cxr._mosaic_sampled import history_tilts

    ids = np.unique(scorer.ids)
    phi = HistoryRollSeed(seed).for_ids(ids)
    tilts = history_tilts(seed, ids, scorer.case["mosaic_mc_fwhm_rad"], draws)
    base = dict(scorer.base, mosaic_mc_nodes=draws)
    base.pop("mosaic_mc_sampling", None)
    basal, other = (np.zeros_like(scorer.energy, dtype=np.float64) for _ in range(2))
    for primary, roll, tilt in zip(ids, phi, tilts, strict=True):
        segs = scorer.subset(np.flatnonzero(scorer.ids == primary))
        orientations = [(_small_tilt_R(dx, dy), 1.0 / draws) for dx, dy in tilt]
        with patch("pyrite.montecarlo.geometry._mosaic_quadrature", return_value=orientations):
            for hkls, result in ((scorer.basal, basal), (scorer.other, other)):
                for start in range(0, len(hkls), scorer.batch):
                    part = dict(
                        base,
                        hkl_list=hkls[start : start + scorer.batch],
                        B_ang2=scorer.lookup.__getitem__,
                        azimuth_rad=scorer.phi0 + float(roll),
                    )
                    result += _host(
                        runner._cxr_for_segments(
                            segs,
                            scorer.energy,
                            part,
                            scorer.tp["n_hat"],
                            scorer.case.get("abs_layers"),
                            scorer.tp.get("groove"),
                            coherent=False,
                            Ne=scorer.ne,
                        )
                    )
    _sync()
    return basal, other


def score_arm(scorer, case):
    """Total CXR and separately scored basal; timing excludes diagnostic basal."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.fiber_emission import validate_fiber_case

    validate_fiber_case(case)
    _sync()
    started = perf_counter()
    spec = _host(
        runner._cxr_for_segments(
            scorer.segs,
            scorer.energy,
            case,
            scorer.tp["n_hat"],
            case.get("abs_layers"),
            scorer.tp.get("groove"),
            Ne=scorer.ne,
        )
    )
    _sync()
    seconds = perf_counter() - started
    base = dict(scorer.base, seed=case["seed"], mosaic_mc_nodes=case["mosaic_mc_nodes"])
    if "mosaic_mc_sampling" in case:
        base["mosaic_mc_sampling"] = case["mosaic_mc_sampling"]
    basal = np.zeros_like(spec, dtype=np.float64)
    for start in range(0, len(scorer.basal), scorer.batch):
        part = dict(
            base,
            hkl_list=scorer.basal[start : start + scorer.batch],
            B_ang2=scorer.lookup.__getitem__,
        )
        basal += _host(
            runner._cxr_for_segments(
                scorer.segs,
                scorer.energy,
                part,
                scorer.tp["n_hat"],
                case.get("abs_layers"),
                scorer.tp.get("groove"),
                Ne=scorer.ne,
            )
        )
    _sync()
    return basal, spec - basal, seconds


def main():
    from pyrite._backend import BACKEND, REAL
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--histories", type=int, default=512)
    p.add_argument("--energy", type=float, default=300)
    p.add_argument("--thick-nm", type=float, default=500)
    p.add_argument("--tilt", type=float, default=50)
    p.add_argument("--tilt-azim", type=float, default=180)
    p.add_argument("--theta", type=float, default=45)
    p.add_argument("--cutoff", type=float, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--transport-seed", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--draws", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--gh-nodes", type=int, nargs="+", default=[5])
    p.add_argument("--energy-start", type=float, default=50)
    p.add_argument("--energy-max", type=float, default=13000)
    p.add_argument("--step", type=float, default=1)
    p.add_argument("--scalar-parity", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.histories, a.batch, *a.draws, *a.gh_nodes) < 1 or min(a.seeds) < 0:
        p.error("positive counts and nonnegative seeds required")
    if a.scalar_parity and a.histories > 128:
        p.error("scalar parity is a small-population check (at most 128 primaries)")
    if not 10 <= a.energy_start < a.energy_max <= 1000 * a.energy or a.step <= 0:
        p.error("invalid photon grid")
    a.mosaic, a.mosaic_route, a.mosaic_nodes, a.reference = True, "mc", max(a.gh_nodes), 2
    grid = np.arange(a.energy_start, a.energy_max, a.step)
    if grid.size < 2:
        p.error("need at least two photon bins")
    case = _case(a, grid)
    model = FiberCXRModel.from_case_record(case["cxr_fiber"], reflection_batch_size=a.batch)
    history = replace(model, numerics=replace(model.numerics, roll_sampling="history"))
    _sync()
    began = perf_counter()
    tp = runner._transport_case(case, transport_core="per-electron")
    _sync()
    transport_s = perf_counter() - began
    scorer = FrozenScorer(case, tp, grid)
    base = dict(case, cxr_fiber=history.case_record())
    spectra = {"E_eV": grid}
    report = dict(
        settings=vars(a),
        backend=type(BACKEND).__name__,
        dtype=np.dtype(REAL).name,
        transport_s=transport_s,
        rows=int(scorer.ids.size),
        arms={},
        scalar_parity={},
    )
    # Warm the history route before measuring any arm.
    score_arm(scorer, dict(base, seed=a.seeds[0], mosaic_mc_nodes=1, mosaic_mc_sampling="history"))
    for seed in a.seeds:
        refs = {}
        for kind, counts in (("GH", a.gh_nodes), ("J", a.draws)):
            for count in counts:
                arm_case = dict(base, seed=seed, mosaic_mc_nodes=count)
                if kind == "J":
                    arm_case["mosaic_mc_sampling"] = "history"
                basal, other, seconds = score_arm(scorer, arm_case)
                name = f"{kind}{count}_seed{seed}"
                if kind == "GH" and count == max(a.gh_nodes):
                    refs = {"basal": basal, "nonbasal": other, "total": basal + other}
                record = {"seconds": seconds, "metrics": {}}
                for label, value in (
                    ("basal", basal),
                    ("nonbasal", other),
                    ("total", basal + other),
                ):
                    assert np.isfinite(value).all()
                    spectra[f"{name}_{label}"] = value
                    record["metrics"][label] = metrics(grid, a.step, value, refs.get(label))
                report["arms"][name] = record
                print(json.dumps({"arm": name, **record}, allow_nan=False), flush=True)
                if a.scalar_parity and kind == "J" and seed == a.seeds[0]:
                    expected = scalar_oracle(scorer, seed, count)
                    result = {}
                    for label, value, target in zip(
                        ("basal", "nonbasal"), (basal, other), expected, strict=True
                    ):
                        assert np.sum(np.abs(target)) > 0, label
                        m = metrics(grid, a.step, value, target)
                        result[label] = m
                        assert m["mass_l1"] <= 3e-4 and m["peak_linf"] <= 3e-4, (label, m)
                        spectra[f"{name}_{label}_scalar"] = target
                    report["scalar_parity"][name] = result
                    print(json.dumps({"scalar_parity": name, **result}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.output) + ".json").write_text(
        json.dumps(report, default=str, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(str(a.output) + ".npz", **spectra)


if __name__ == "__main__":
    main()
