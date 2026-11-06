"""Characteristic vs continuum convergence and nested-prefix probes (#1006).

Usage:
  characteristic_budget_1006.py run MATERIAL E0_KEV THICKNESS_ANG NE BLOCK OUT_NPZ [SEED [CORE]]
  characteristic_budget_1006.py prefix MATERIAL E0_KEV THICKNESS_ANG OUT_JSON [CORE]
  characteristic_budget_1006.py report OUT_JSON NPZ...

``run`` transports the lower-cutoff (brem) population in equal electron blocks
and scores ``spec_characteristic`` and ``brem_wide`` per block through the
production adapters, keeping every block's per-electron mean spectrum. Blocks
hold i.i.d. electrons, so block means are i.i.d.: their spread gives the
per-electron coefficient of variation without per-electron scoring. Timers
split transport and each scorer at synchronized boundaries.

``report`` reduces the block matrices: per observable, CV, relative standard
error vs N (with re-batching and prefix checks), electrons needed for a target
RSE, bootstrap intervals on ``(CV_char/CV_brem)^2``, and char/brem correlation.

``prefix`` compares fixed-count runs bit for bit, to test whether estimators
on nested prefixes of one population are independent of the deeper budget.
Engineering evidence; no physics claim.
"""

import json
import sys
import time
from dataclasses import replace

import numpy as np

#: Named characteristic windows (eV) reported when the mean spectrum has a line there.
LINE_WINDOWS_EV = {
    "silicon": {"Si K": (1600.0, 1950.0)},
    "tungsten": {"W M": (1500.0, 2600.0), "W L": (7000.0, 13000.0), "W K": (55000.0, 72000.0)},
}
BOOTSTRAPS = 2000
TARGET_RSE = 0.01


def _sync():
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def build_case(material, e0_kev, thickness_ang, ne_lines, ne_brem, seed=1):
    """Production-style coupled BremsLib midpoint case at the default cutoffs."""
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    numerics = pr.Numerics(
        cxr_trials=int(ne_lines),
        brem_trials=int(ne_brem),
        energy_model="midpoint",
        energy_bins=EnergyBins(
            cxr=np.arange(1000.0, min(e0_kev, 80.0) * 1e3 + 25.0, 50.0),
            brem=np.arange(1000.0, e0_kev * 1e3, 1000.0),
        ),
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
    )
    target = pr.geometry.Slab(
        material,
        thickness_ang=float(thickness_ang),
        tilt_deg=30.0,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=e0_kev, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        target,
        pr.Detector(),
        physics=physics,
    )
    case = api.build_case(scene, numerics)
    assert case.get("line_grid_policy") is None, "needs a final line grid"
    return replace(case, seed=int(seed))


class BlockRecorder:
    """Block consumer: one per-electron mean spectrum of each estimator per block."""

    def __init__(self, case, block_electrons):
        from pyrite.montecarlo.runner.artifacts import _case_geometry
        from pyrite.montecarlo.runner.emission_adapters import select_emission

        self.case = case
        self.block = int(block_electrons)
        self.ne = int(case["Ne_brem"])
        self.E_grid, self.E_brem, _, self.n_hat, self.groove = _case_geometry(case)
        self.emission = select_emission(case)
        self.emission.preflight(case, streaming=True)
        self.table_cache = {}
        self.char, self.brem, self.rows = [], [], []
        self.seconds = {"characteristic": 0.0, "continuum": 0.0, "upload": 0.0}

    def _timed(self, name, fn):
        from pyrite._backend import _to_cpu

        _sync()
        start = time.perf_counter()
        values = fn()
        _sync()
        self.seconds[name] += time.perf_counter() - start
        return np.asarray(_to_cpu(values), dtype=np.float64)

    def __call__(self, segs):
        from pyrite.montecarlo.runner import _segments_on_device
        from pyrite.montecarlo.runner.emission_adapters import EmissionBatch

        start = time.perf_counter()
        block = _segments_on_device(segs)
        _sync()
        self.seconds["upload"] += time.perf_counter() - start
        batch = EmissionBatch(
            block,
            self.E_grid,
            self.case,
            self.n_hat,
            self.case.get("abs_layers"),
            self.groove,
            electron_limit=self.ne,
            table_cache=self.table_cache,
        )
        char = self._timed(
            "characteristic",
            lambda: self.emission.characteristic.evaluate(batch, "spec_characteristic").values,
        )
        brem = self._timed(
            "continuum",
            lambda: (
                self.emission.continuum.evaluate(
                    replace(batch, energy_eV=self.E_brem), "brem_wide"
                ).values
            ),
        )
        # Scorers normalize by the whole population; rescale to this block's mean.
        scale = self.ne / self.block
        self.char.append(char * scale)
        self.brem.append(brem * scale)
        self.rows.append(int(segs["L_ang"].shape[0]))


def run(material, e0_kev, thickness_ang, ne, block, out_path, seed=1, core="cuda"):
    from pyrite.montecarlo.runner import _transport_case
    from pyrite.montecarlo.runner.oom import _ensure_pool_limit

    e0_kev, thickness_ang = float(e0_kev), float(thickness_ang)
    ne, block = int(ne), int(block)
    if ne % block:
        raise ValueError("NE must be a multiple of BLOCK: block means must be equal-sized")
    case = build_case(material, e0_kev, thickness_ang, 1, ne, int(seed))
    recorder = BlockRecorder(case, block)
    _ensure_pool_limit()
    start = time.perf_counter()
    transport = _transport_case(
        case,
        transport_core=core,
        block_electrons=block,
        block_consumer=recorder,
        keep_segments_on_device=True,
    )
    _sync()
    wall = time.perf_counter() - start
    counts = transport["segs"]
    np.savez_compressed(
        out_path,
        char=np.asarray(recorder.char),
        brem=np.asarray(recorder.brem),
        rows=np.asarray(recorder.rows),
        E_grid=np.asarray(recorder.E_grid),
        E_brem=np.asarray(recorder.E_brem),
        meta=json.dumps(
            {
                "material": material,
                "E0_keV": e0_kev,
                "thickness_ang": thickness_ang,
                "Ne": ne,
                "block": block,
                "seed": int(seed),
                "core": core,
                "E_cut_brem_keV": case.get("E_cut_brem_keV", 1.0),
                "wall_s": wall,
                "scoring_s": recorder.seconds,
                "transport_s": wall - sum(recorder.seconds.values()),
                "n_backscattered": int(counts["n_backscattered"]),
                "n_transmitted": int(counts.get("n_transmitted", 0)),
            }
        ),
    )
    print(json.dumps({"out": out_path, "wall_s": wall, "scoring_s": recorder.seconds}))


def _integrate(spectrum, energies):
    return spectrum @ np.gradient(energies)


def observables(data):
    """Per-block scalar observables: ``{name: (estimator, block values)}``."""
    meta = json.loads(str(data["meta"]))
    char, brem = data["char"], data["brem"]
    E_grid, E_brem = data["E_grid"], data["E_brem"]
    e0_ev = meta["E0_keV"] * 1e3
    out = {"char total": ("char", _integrate(char, E_grid))}
    mean_char = char.mean(axis=0)
    total = _integrate(mean_char, E_grid)
    for name, (lo, hi) in LINE_WINDOWS_EV.get(meta["material"], {}).items():
        mask = (E_grid >= lo) & (E_grid <= hi)
        if not mask.any():
            continue
        share = _integrate(mean_char[mask], E_grid[mask]) / total if total > 0 else 0.0
        if share < 1e-3:
            continue
        out[f"char {name} window"] = ("char", _integrate(char[:, mask], E_grid[mask]))
        peak = np.flatnonzero(mask)[np.argmax(mean_char[mask])]
        out[f"char {name} peak bin"] = ("char", char[:, peak])
    out["brem total"] = ("brem", _integrate(brem, E_brem))
    for label, fraction in (("0.1E0", 0.1), ("0.5E0", 0.5), ("0.9E0", 0.9)):
        index = int(np.argmin(np.abs(E_brem - fraction * e0_ev)))
        out[f"brem {label} bin"] = ("brem", brem[:, index])
    tip = E_brem >= 0.9 * e0_ev
    out["brem >=0.9E0 band"] = ("brem", _integrate(brem[:, tip], E_brem[tip]))
    return out


def _cv(values, block):
    """Per-electron coefficient of variation from equal-block means."""
    mean = values.mean()
    if mean == 0.0:
        return float("nan")
    return float(values.std(ddof=1) * np.sqrt(block) / abs(mean))


def _rebatched_cv(values, block, factor):
    k = values.size // factor
    merged = values[: k * factor].reshape(k, factor).mean(axis=1)
    return _cv(merged, block * factor)


def summarize(path, rng):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    block, n_blocks = meta["block"], data["char"].shape[0]
    found = observables(data)
    index = rng.integers(0, n_blocks, size=(BOOTSTRAPS, n_blocks))
    rows, draws = {}, {}
    for name, (estimator, values) in found.items():
        cv = _cv(values, block)
        resampled = values[index]
        means = resampled.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws[name] = resampled.std(axis=1, ddof=1) * np.sqrt(block) / np.abs(means)
        rows[name] = {
            "estimator": estimator,
            "mean": float(values.mean()),
            "cv": cv,
            "cv_ci95": [float(x) for x in np.nanpercentile(draws[name], [2.5, 97.5])],
            "cv_rebatched_x4": _rebatched_cv(values, block, 4) if n_blocks >= 12 else None,
            "cv_first_half": _cv(values[: n_blocks // 2], block),
            "cv_second_half": _cv(values[n_blocks // 2 :], block),
            "rse_at_run": cv / np.sqrt(meta["Ne"]),
            "max_block_share": float(values.max() / values.sum()) if values.sum() > 0 else None,
            "N_for_target": cv**2 / TARGET_RSE**2,
        }
    reference = "brem total"
    ratios = {}
    for name, row in rows.items():
        if row["estimator"] != "char":
            continue
        for brem_name in (n for n, r in rows.items() if r["estimator"] == "brem"):
            ratio = (draws[name] / draws[brem_name]) ** 2
            ratios[f"{name} / {brem_name}"] = {
                "ratio": (row["cv"] / rows[brem_name]["cv"]) ** 2,
                "ci95": [float(x) for x in np.nanpercentile(ratio, [2.5, 97.5])],
            }
    correlation = {
        name: float(np.corrcoef(found[name][1], found[reference][1])[0, 1])
        for name, row in rows.items()
        if row["estimator"] == "char"
    }
    return {
        "meta": meta,
        "n_blocks": n_blocks,
        "segments_per_electron": float(data["rows"].sum() / meta["Ne"]),
        "observables": rows,
        "n_ratio": ratios,
        "correlation_with_brem_total": correlation,
    }


def report(out_path, *paths):
    rng = np.random.default_rng(1006)
    summaries = [summarize(path, rng) for path in paths]
    with open(out_path, "w") as handle:
        json.dump(summaries, handle, indent=1)
    print("| case | N | seg/e | CV char total | CV brem total | (CVc/CVb)^2 [95%] | corr |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        m, o = s["meta"], s["observables"]
        ratio = s["n_ratio"]["char total / brem total"]
        print(
            f"| {m['material']} {m['thickness_ang'] / 1e4:g} um {m['E0_keV']:g} keV s{m['seed']} "
            f"| {m['Ne']} | {s['segments_per_electron']:.0f} | {o['char total']['cv']:.3f} "
            f"| {o['brem total']['cv']:.3f} | {ratio['ratio']:.2f} "
            f"[{ratio['ci95'][0]:.2f}, {ratio['ci95'][1]:.2f}] "
            f"| {s['correlation_with_brem_total']['char total']:.2f} |"
        )


def _score_fixed(material, e0_kev, thickness_ang, ne_lines, ne_brem, core, limits, block=None):
    """One transport at ``(ne_lines, ne_brem)``, scored on each ``limits`` prefix."""
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.runner import _segments_on_device, _transport_case
    from pyrite.montecarlo.runner.artifacts import _case_geometry
    from pyrite.montecarlo.runner.emission_adapters import EmissionBatch, select_emission

    case = build_case(material, e0_kev, thickness_ang, ne_lines, ne_brem)
    E_grid, E_brem, _, n_hat, groove = _case_geometry(case)
    emission = select_emission(case)
    totals = {}

    def score(segs):
        staged = _segments_on_device(segs)
        batch = EmissionBatch(staged, E_grid, case, n_hat, case.get("abs_layers"), groove)
        for name, limit in limits.items():
            view = replace(batch, electron_limit=limit)
            if name == "characteristic":
                values = emission.characteristic.evaluate(view, "spec_characteristic").values
            elif name == "brem":
                values = emission.continuum.evaluate(
                    replace(view, energy_eV=E_brem), "brem_wide"
                ).values
            else:
                values = emission.cxr.evaluate(view, "spec").values
            part = np.asarray(_to_cpu(values), dtype=np.float64)
            totals[name] = part if name not in totals else totals[name] + part

    if block is None:
        transport = _transport_case(case, transport_core=core, keep_segments_on_device=False)
        score(transport["segs"])
    else:
        _transport_case(
            case,
            transport_core=core,
            block_electrons=block,
            block_consumer=score,
            keep_segments_on_device=True,
        )
    return totals


def _bitwise(a, b):
    peak = max(float(np.max(np.abs(b), initial=0.0)), np.finfo(float).tiny)
    return {
        "bitwise": bool(a.shape == b.shape and np.array_equal(a, b)),
        "max_rel_to_peak": float(np.max(np.abs(a - b), initial=0.0) / peak),
    }


def prefix(material, e0_kev, thickness_ang, out_path, core="per-electron"):
    """Do nested estimators depend on how deep the lower-cutoff prefix runs?

    Populations ``N_brem=64 < N_char=128 < N_lines=256``. Today's runner can
    already express the proposed three-budget transport: it is the
    ``(Ne=256, Ne_brem=128)`` run, with brem read from its first 64 electrons.
    Each estimator is compared with the run where it is the deepest prefix.
    """
    e0_kev, thickness_ang = float(e0_kev), float(thickness_ang)
    n_brem, n_char, n_lines = 64, 128, 256
    nested_limits = {"brem": n_brem, "characteristic": n_char, "cxr": n_lines}
    args = (material, e0_kev, thickness_ang)
    nested = _score_fixed(*args, n_lines, n_char, core, nested_limits)
    brem_ref = _score_fixed(*args, n_lines, n_brem, core, {"brem": n_brem, "cxr": n_lines})
    char_ref = _score_fixed(*args, n_lines, n_char, core, {"characteristic": n_char})
    equal_ref = _score_fixed(*args, n_brem, n_brem, core, {"brem": n_brem})
    out = {
        "core": core,
        "populations": {"brem": n_brem, "characteristic": n_char, "lines": n_lines},
        "brem_vs_brem_deepest": _bitwise(nested["brem"], brem_ref["brem"]),
        "brem_vs_equal_population_run": _bitwise(nested["brem"], equal_ref["brem"]),
        "characteristic_vs_char_deepest": _bitwise(
            nested["characteristic"], char_ref["characteristic"]
        ),
        "cxr_vs_shallower_prefix": _bitwise(nested["cxr"], brem_ref["cxr"]),
    }
    if core != "lockstep":
        streamed = _score_fixed(*args, n_lines, n_char, core, nested_limits, block=48)
        for name in ("brem", "characteristic", "cxr"):
            out[f"{name}_streamed_block48_vs_joined"] = _bitwise(streamed[name], nested[name])
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "run":
        run(*sys.argv[2:])
    elif command == "prefix":
        prefix(*sys.argv[2:])
    elif command == "report":
        report(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
