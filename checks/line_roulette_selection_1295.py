"""Line-mass concentration of the selected set and predicted roulette gain (#1295).

Usage:
  line_roulette_selection_1295.py run MATERIAL E0_KEV TILT_DEG HISTORIES SEED OUT_STEM
  line_roulette_selection_1295.py predict OUT_STEM.npz [KEEP:FLOOR ...]
  line_roulette_selection_1295.py table OUT_DIR

``run`` resolves one ``PROFILE`` case (default ``cleo_sub_1281``, shipped
under ``XDG_CONFIG_HOME``) at a fixed population exactly as
``reflection_selection_subsample_1281.py`` does, and lets the automatic
reflection selection run on its electron prefix. At the production pass of the
selected set it times, synchronized: the analog production pass (every row,
every selected chunk), a mass-only screen of the same set (what a roulette
screen would cost: exact per-primary line mass, no grid reduction), and the
production pass on the rows of the primaries one roulette arm keeps
(``ARM``, default ``0.1:0.1``). It saves per-primary exact line masses
(``sum w pi / a_w`` over the whole selected set) and row counts to
``OUT_STEM.npz`` and the record to ``OUT_STEM.json``.

``predict`` evaluates, from those masses alone, each ``keep_fraction:floor``
arm of the ``exact-line-mass-two-stratum-v1`` policy (pilot 128, the case
seed's draws): the analog and rouletted variance of the summed line mass
(Eq. ``eq-roulette-variance``), the expected rouletted production cost
``t_screen + t_production * E[rows scored] / rows``, and the production-pass
figure-of-merit gain ``(Var_A T_A) / (Var_R T_R)``; transport and the
fixed prefix selection are excluded, so the whole-case gain is smaller.

Yield-level prediction only: per-bin variance can concentrate differently.
Remote GPU only. Engineering evidence for ``line-scoring-roulette``; no
accuracy verdict.
"""

import json
import os
import runpy
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("PROFILE", "cleo_sub_1281")
os.environ["PROBE_PRODUCTION"] = "1"
SUB = runpy.run_path(str(Path(__file__).with_name("reflection_selection_subsample_1281.py")))
ARM = tuple(float(v) for v in os.environ.get("ARM", "0.1:0.1").split(":"))
#: Forced device bin-mean reduction (``bin_reduction_override``), e.g. ``auto``
#: for near/far when the Fourier route cannot fit a case; empty keeps the default.
REDUCTION = os.environ.get("PROBE_REDUCTION", "")
PILOT = 128
DEFAULT_ARMS = [f"{k}:{f}" for k in (0.02, 0.05, 0.1, 0.25) for f in (0.05, 0.1, 0.25, 0.5)]


def _sync():
    SUB["_sync"]()


def _timed(fn):
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t0


def _chunks(case, grid):
    from pyrite.montecarlo.runner import reflection_cutoff as rc

    per_call = rc.selection_call_sizes(case, grid)
    layered = case.get("layer_radiators") is not None
    for layer, radiator in enumerate(case["layer_radiators"]) if layered else [(None, case)]:
        if radiator is None:
            continue
        hkls = radiator["hkl_list"]
        for i in range(0, len(hkls), per_call[layer]):
            yield rc._pilot_case(case, layer, hkls[i : i + per_call[layer]], None)


def _screen_masses(emission, batch, case, n):
    """Exact per-primary line mass of every selected chunk, no grid reduction."""
    from pyrite._backend import _to_cpu

    caller = batch.truncation_audit
    grid = np.asarray(_to_cpu(batch.energy_eV), dtype=float)
    band = (caller["start_eV"], caller["stop_eV"]) if caller else (grid[0], grid[-1])
    axis = np.array(band, dtype=float)
    mass = np.zeros(n)
    for pilot in _chunks(case, batch.energy_eV):
        audit = {"start_eV": band[0], "stop_eV": band[1], "n_electrons": n, "mass_only": True}
        with np.errstate(over="ignore", divide="ignore"):
            emission.cxr.evaluate(
                replace(batch, case=pilot, energy_eV=axis, truncation_audit=audit, table_cache={}),
                "spec",
            )
        if audit.get("electron_mass") is not None:
            mass += np.asarray(_to_cpu(audit["electron_mass"]), dtype=float)
    return mass


def arm_prediction(mass, rows, seed, keep, floor, pilot=PILOT):
    """Strata, expected cost share and added variance of one policy arm.

    Mirrors ``runner.line_roulette.LineScreen._select``.
    """
    from pyrite.montecarlo.runner.line_roulette import roulette_uniforms

    n = mass.size
    threshold = float(np.quantile(mass[:pilot], 1.0 - keep))
    ids = np.arange(n)
    low = (ids >= pilot) & (mass > 0.0) & (mass < threshold)
    p = np.where(low, floor, 1.0)
    kept = (mass > 0.0) & np.where(low, roulette_uniforms(seed, ids) < floor, True)
    scored = mass > 0.0
    added = float(((1.0 / p[low] - 1.0) * mass[low] ** 2).sum())
    return dict(
        keep_fraction=keep,
        floor=floor,
        threshold=threshold,
        primaries_at_floor=int(low.sum()),
        mass_share_at_floor=float(mass[low].sum() / mass.sum()),
        expected_row_share=float((rows * p)[scored].sum() / rows.sum()),
        realized_row_share=float(rows[kept].sum() / rows.sum()),
        added_variance=added,
        realized_estimate_ratio=float((mass[kept] / p[kept]).sum() / mass.sum()),
    )


def predict(mass, rows, seed, timing, arms):
    n = mass.size
    var_a = float(n * mass.var(ddof=1))
    order = np.sort(mass)[::-1]
    cumulative = np.cumsum(order) / order.sum()
    t_prod, t_screen = timing["production_s"], timing["screen_s"]
    out = dict(
        primaries=n,
        rows=int(rows.sum()),
        zero_mass=int(np.count_nonzero(mass == 0.0)),
        analog_variance=var_a,
        relative_se_analog=float(np.sqrt(var_a) / mass.sum()),
        mass_share_top={
            f: float(cumulative[max(int(f * n) - 1, 0)]) for f in (0.001, 0.01, 0.1, 0.5)
        },
        rows_vs_mass_corr=float(np.corrcoef(rows, mass)[0, 1]),
        screen_over_production=t_screen / t_prod,
        arms=[],
    )
    for keep, floor in arms:
        row = arm_prediction(mass, rows, seed, keep, floor)
        var_r = var_a + row["added_variance"]
        t_r = t_screen + t_prod * row["expected_row_share"]
        row.update(
            variance_ratio=var_r / var_a,
            production_time_ratio=t_r / t_prod,
            fom_gain_production=(var_a * t_prod) / (var_r * t_r),
        )
        out["arms"].append(row)
    out["best"] = max(out["arms"], key=lambda r: r["fom_gain_production"])
    return out


def instrument(state):
    from pyrite.montecarlo.runner import reflection_cutoff as rc

    batch_selection, production = rc._batch_selection, rc._selected_spectrum

    def captured(emission, batch, Ne_brem, histories):
        state.update(emission=emission, batch=batch)
        return batch_selection(emission, batch, Ne_brem, histories)

    def measured(case, evaluate, grid):
        from pyrite.montecarlo.runner.electron_blocks import _electron_ids
        from pyrite.montecarlo.runner.line_roulette import _rows_of

        emission, batch = state["emission"], state["batch"]
        n = int(batch.segments["Ne"] if batch.electron_limit is None else batch.electron_limit)
        spec, state["production_s"] = _timed(lambda: production(case, evaluate, grid))
        mass, state["screen_s"] = _timed(lambda: _screen_masses(emission, batch, case, n))
        ids = _electron_ids(batch.segments)
        rows = np.bincount(ids[ids < n], minlength=n)
        seed = int(batch.case["seed"])
        arm = arm_prediction(mass, rows, seed, *ARM)
        threshold = arm["threshold"]
        low = (np.arange(n) >= PILOT) & (mass > 0.0) & (mass < threshold)
        from pyrite.montecarlo.runner.line_roulette import roulette_uniforms

        kept = (mass > 0.0) & np.where(low, roulette_uniforms(seed, np.arange(n)) < ARM[1], True)
        subset = replace(batch, segments=_rows_of(batch.segments, np.flatnonzero(kept[ids])))

        def subset_evaluate(pilot):
            return emission.cxr.evaluate(replace(subset, case=pilot, table_cache={}), "spec").values

        sub_spec, state["subset_s"] = _timed(lambda: production(case, subset_evaluate, grid))
        state.update(
            mass=mass,
            rows=rows,
            seed=seed,
            subset_row_share=float(rows[kept].sum() / rows.sum()),
            spec_sum=float(np.sum(spec)),
            subset_spec_sum=float(np.sum(sub_spec)),
        )
        return spec

    rc._batch_selection, rc._selected_spectrum = captured, measured


def _reduction_scope():
    from contextlib import nullcontext

    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_reduction_override

    return bin_reduction_override(REDUCTION) if REDUCTION else nullcontext()


def run(material, e0_kev, tilt_deg, histories, seed, out_stem):
    from pyrite.montecarlo.runner.scheduling import run_cases

    stem = Path(out_stem)
    SUB["precision_override"](int(histories))
    spec, changes, n = SUB["fixed_case"](material, float(e0_kev), float(tilt_deg), histories, seed)
    state: dict = {}
    timings: list = []
    instrument(state)
    record = dict(
        issue=1295,
        code_sha=os.environ.get("PROBE_SHA"),
        profile=os.environ["PROFILE"],
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        seed_offset=int(seed),
        arm=list(ARM),
        reduction_override=REDUCTION or None,
        error=None,
    )
    case = replace(spec, **changes) if changes else spec
    t0 = time.perf_counter()
    out = None
    try:
        with _reduction_scope():
            out = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)[0]
    except Exception as exc:  # record failure as evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        print(record["traceback"], file=sys.stderr)
    record["wall_s"] = time.perf_counter() - t0
    record["timings"] = timings
    record["reflection_selection"] = (out or {}).get("reflection_selection")
    record["n_segments"] = None if out is None else out.get("n_segments")
    for key in ("production_s", "screen_s", "subset_s", "subset_row_share", "spec_sum"):
        record[key] = state.get(key)
    record["subset_spec_sum"] = state.get("subset_spec_sum")
    if "mass" in state:
        np.savez(stem.with_suffix(".npz"), mass=state["mass"], rows=state["rows"])
        timing = dict(production_s=state["production_s"], screen_s=state["screen_s"])
        record["seed"] = state["seed"]
        record["prediction"] = predict(
            state["mass"], state["rows"], state["seed"], timing, _parse(DEFAULT_ARMS)
        )
    elif record["error"] is None:
        record["error"] = "no selected production pass ran (case without reflection_selection?)"
    stem.with_suffix(".json").write_text(json.dumps(record, indent=1, default=SUB["_json_default"]))
    print(json.dumps(_summary(record), default=SUB["_json_default"]))


def _parse(arms):
    return [tuple(float(v) for v in text.split(":")) for text in arms]


def _summary(record):
    pred = record.get("prediction") or {}
    best = pred.get("best") or {}
    return dict(
        case=f"{record['material']} {record['E0_keV']:g}/{record['tilt_deg']:g} N={record['histories']}",
        error=record["error"],
        production_s=record.get("production_s"),
        screen_s=record.get("screen_s"),
        subset_s=record.get("subset_s"),
        subset_row_share=record.get("subset_row_share"),
        mass_share_top=pred.get("mass_share_top"),
        best=None
        if not best
        else {k: best[k] for k in ("keep_fraction", "floor", "fom_gain_production")},
    )


def predict_cli(npz, *arms):
    stem = Path(npz).with_suffix("")
    record = json.loads(stem.with_suffix(".json").read_text())
    data = np.load(npz)
    timing = dict(production_s=record["production_s"], screen_s=record["screen_s"])
    out = predict(data["mass"], data["rows"], record["seed"], timing, _parse(arms or DEFAULT_ARMS))
    print(json.dumps(out, indent=1))


def table(out_dir):
    for path in sorted(Path(out_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("issue") != 1295:
            continue
        print(json.dumps(_summary(record), default=str))
        for row in (record.get("prediction") or {}).get("arms", []):
            print(
                f"  keep={row['keep_fraction']:<5g} floor={row['floor']:<5g} "
                f"rows={row['expected_row_share']:.3f} var={row['variance_ratio']:.3f} "
                f"T={row['production_time_ratio']:.3f} FOM={row['fom_gain_production']:.2f}"
            )


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    {"run": run, "predict": predict_cli, "table": table}.get(command, lambda *_: sys.exit(__doc__))(
        *rest
    )
