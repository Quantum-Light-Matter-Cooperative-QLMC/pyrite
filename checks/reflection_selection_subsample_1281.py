"""Reflection selection on an electron prefix: cost and set stability (#1281).

Usage:
  reflection_selection_subsample_1281.py run MATERIAL E0_KEV TILT_DEG HISTORIES SEED BUDGETS OUT_JSON
  reflection_selection_subsample_1281.py table OUT_DIR

``run`` resolves one case of the ``PROFILE`` catalog profile (default
``cleo_sub_1281``; the profile, beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) as
``pyrite run`` does, fixes its population at ``HISTORIES`` (adaptive
``min = max``), offsets its seed by ``SEED``, and runs it once through
``run_cases`` (production serial path). Transport runs once. At the spectrum
phase every comma-separated budget in ``BUDGETS`` runs the measured selection
on its prefix: ``0`` the whole population, ``N`` or ``hN`` the first ``N``
electrons, ``rN`` (``N`` may be ``1e7``) the #1368 row-budget prefix, the
fewest electrons whose segment rows reach ``N``, at most the default
``SELECTION_HISTORIES``. Only the first arm also runs the production pass on
every history (unless ``PROBE_PRODUCTION=0``); later arms stop after
selecting, since their production pass would repeat the first. The record is
rewritten after every arm, so a timed-out or cancelled case keeps its
finished arms.

A case whose rows do not fit the device takes the streamed route (#1281,
``PrefixSelectingScorer``); there the arms select on the held prefix, which
holds the default ``SELECTION_HISTORIES`` electrons, so ``0`` and history
budgets above it read only that prefix. With ``PROBE_PRODUCTION=0`` a streamed
case stops after its arms. ``rows`` records the segment rows per electron of
the selection population: total, and cumulative at every power-of-two prefix
(#1368).

Recorded per arm: synchronized reference-selection and production-pass wall
time, per-radiator cutoff, reflection count, tail, convergence and shell
table, and a digest of the selected set; plus transport and spectrum splits.
``table`` compares each arm against the ``0`` (full) arm of its case.

Remote only (GPU, multi-minute selections). Engineering evidence for
``reflection-yield-tolerance-selector`` set stability; no accuracy verdict.
"""

import dataclasses
import hashlib
import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

PROFILE = os.environ.get("PROFILE", "cleo_sub_1281")
PRODUCTION = os.environ.get("PROBE_PRODUCTION", "1") != "0"
os.environ["PROFILE"] = PROFILE
HELPER = runpy.run_path(str(Path(__file__).with_name("adaptive_block_size_1252.py")))


def _sync():
    cupy = HELPER["CUPY"]
    if cupy is not None:
        cupy.cuda.Device().synchronize()


def _digest(hkls) -> str:
    rows = sorted(tuple(int(i) for i in hkl) for hkl in hkls)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def fixed_case(material, e0_kev, tilt_deg, histories, seed):
    """The profile case at a fixed population, its seed offset by ``seed``."""
    case, precision, _ = HELPER["profile_case"](material, e0_kev, tilt_deg, "auto")
    del precision
    fields = {f.name for f in dataclasses.fields(case)}
    changes = {}
    for name in ("seed", "base_seed"):
        if name in fields and seed:
            changes[name] = getattr(case, name) + int(seed)
            break
    else:
        if seed:
            raise SystemExit(f"case has no seed field among {sorted(fields)}")
    return case, changes, int(histories)


#: Unbounded history or row budget.
UNBOUNDED = 2**62


class ProbeStop(Exception):
    """A streamed case stops after its selection arms (``PROBE_PRODUCTION=0``)."""


def parse_budget(budget, cap):
    """``(histories, rows)`` of one ``BUDGETS`` entry (see the module docstring)."""
    text = str(budget).strip()
    if text.startswith("r"):
        return int(cap), int(float(text[1:]))
    n = int(float(text.lstrip("h")))
    return (n or UNBOUNDED), UNBOUNDED


def row_profile(blocks, cap):
    """Segment rows of the first ``cap`` electrons: total and power-of-two prefixes."""
    from pyrite._backend import _to_cpu, array_namespace

    counts = np.zeros(cap, dtype=np.int64)
    for block in blocks:
        ids = block["electron_id" if "electron_id" in block else "elec_id"]
        kept = ids[ids < cap]
        if kept.size:
            counts += np.asarray(_to_cpu(array_namespace(ids).bincount(kept, minlength=cap)))
    cumulative = np.cumsum(counts)
    prefixes = sorted({*(1 << i for i in range(cap.bit_length()) if 1 << i <= cap), cap})
    return dict(
        electrons=int(cap),
        rows=int(cumulative[-1]) if cap else 0,
        rows_per_electron=float(cumulative[-1] / cap) if cap else None,
        max_rows_per_electron=int(counts.max()) if cap else 0,
        cumulative={str(k): int(cumulative[k - 1]) for k in prefixes},
    )


def instrument(budgets, arms, flush=lambda: None, rows_out=None):
    """Run every budget at the spectrum phase's selection call."""
    from pyrite.montecarlo.runner import reflection_cutoff as rc

    original = rc.selected_batch_spectrum
    select = rc.select_case_reflections
    production = rc._selected_spectrum
    cap = rc.SELECTION_HISTORIES
    rows_out = {} if rows_out is None else rows_out
    # Streamed scorers hold the full history-cap prefix; arms pass their own rows.
    rc.SELECTION_ROWS = UNBOUNDED

    def run_arm(emission, batch, Ne_brem, budget, first, parts=None):
        timing = {"reference_s": 0.0, "production_s": 0.0}
        captured = []

        def timed_select(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            out = select(*args, **kwargs)
            _sync()
            timing["reference_s"] += time.perf_counter() - t0
            captured.append(out[1])
            return out

        def timed_production(case, evaluate, grid):
            if not (first and PRODUCTION):
                return np.zeros(np.shape(grid))
            _sync()
            t0 = time.perf_counter()
            out = production(case, evaluate, grid)
            _sync()
            timing["production_s"] += time.perf_counter() - t0
            return out

        histories, rows = parse_budget(budget, cap)
        rc.select_case_reflections, rc._selected_spectrum = timed_select, timed_production
        try:
            t0 = time.perf_counter()
            if parts is None:
                out = original(emission, batch, Ne_brem, histories=histories, rows=rows)
                spec = out[0]
            else:
                spec = None
                out = select_streamed(
                    emission, batch, Ne_brem, histories=histories, parts=parts, rows=rows
                )
            record = out[1]
            timing["wall_s"] = time.perf_counter() - t0
        finally:
            rc.select_case_reflections, rc._selected_spectrum = select, production
        outcomes = captured[-1] if captured else []
        arms.append(
            dict(
                budget=str(budget),
                histories=None if histories == UNBOUNDED else histories,
                row_budget=None if rows == UNBOUNDED else rows,
                **timing,
                record=record,
                digests=[_digest(o.hkl_list) for o in outcomes],
                spec_sum=(
                    float(np.sum(np.asarray(spec, dtype=float)))
                    if first and PRODUCTION and spec is not None
                    else None
                ),
            )
        )
        flush()
        return out

    def profile(blocks, Ne, Ne_brem):
        population = max(int(Ne), int(Ne if Ne_brem is None else Ne_brem))
        rows_out.update(row_profile(blocks, min(population, cap)))
        flush()

    def wrapped(emission, batch, Ne_brem, histories=None, rows=None):
        lines = batch.segments["Ne"] if batch.electron_limit is None else batch.electron_limit
        profile([batch.segments], lines, Ne_brem)
        result = None
        for index, budget in enumerate(budgets):
            out = run_arm(emission, batch, Ne_brem, budget, index == 0)
            if index == 0:
                result = out
        return result

    select_streamed = rc.select_batch_reflections

    def streamed(emission, batch, Ne_brem, histories=None, parts=None, rows=None):
        profile(parts(), batch.electron_limit, Ne_brem)
        result = None
        for index, budget in enumerate(budgets):
            out = run_arm(emission, batch, Ne_brem, budget, index == 0, parts=parts)
            if index == 0:
                result = out
        if not PRODUCTION:
            raise ProbeStop
        return result

    rc.selected_batch_spectrum = wrapped
    rc.select_batch_reflections = streamed


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run(material, e0_kev, tilt_deg, histories, seed, budgets, out_json):
    from pyrite.montecarlo.runner.scheduling import run_cases

    budget_list = [b.strip() for b in budgets.split(",")]
    precision_override(int(histories))
    spec, changes, n = fixed_case(material, float(e0_kev), float(tilt_deg), histories, seed)
    arms: list[dict] = []
    timings: list = []
    t0 = time.perf_counter()
    record = dict(
        issue=1281,
        code_sha=os.environ.get("PROBE_SHA"),
        profile=PROFILE,
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        seed_offset=int(seed),
        budgets=budget_list,
        production=PRODUCTION,
        complete=False,
        error=None,
        timings=timings,
        rows={},
        arms=arms,
    )

    def flush():
        record["wall_s"] = time.perf_counter() - t0
        Path(out_json).write_text(json.dumps(record, indent=1, default=_json_default))

    instrument(budget_list, arms, flush, record["rows"])
    case = dataclasses.replace(spec, **changes) if changes else spec
    out = None
    try:
        out = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)[0]
    except ProbeStop:
        record["stopped_after_selection"] = True
    except Exception as exc:  # record failure as evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
    record.update(
        complete=record["error"] is None,
        realized_electrons=((out or {}).get("adaptive_sampling") or {}).get("realized_electrons"),
        n_segments=None if out is None else out.get("n_segments"),
    )
    flush()
    print(json.dumps(summary(record), default=_json_default))


def precision_override(n):
    """Fix every case's adaptive population at ``n`` (``min = max``, no pilot)."""
    from pyrite.runs import scan

    resolved = scan._resolved_run

    def fixed(args, material):
        settings, sweep, identity, stem = resolved(args, material)
        # #1398: brem stops on its own prefix, which must not exceed max_histories.
        brem = (
            {"brem_min_histories": n, "brem_max_histories": n}
            if getattr(settings.precision, "brem_max_histories", None) is not None
            else {}
        )
        precision = dataclasses.replace(
            settings.precision, min_histories=n, max_histories=n, pilot_histories=None, **brem
        )
        return dataclasses.replace(settings, precision=precision), sweep, identity, stem

    scan._resolved_run = fixed


def summary(record):
    rows = []
    for arm in record["arms"]:
        radiators = (arm["record"] or {}).get("radiators", [])
        rows.append(
            dict(
                budget=arm["budget"],
                selection=(arm["record"] or {}).get("selection_histories"),
                reference_s=round(arm["reference_s"], 2),
                production_s=round(arm["production_s"], 2),
                cutoffs=[r["cutoff_invang"] for r in radiators],
                reflections=[r["reflections"] for r in radiators],
                tails=[r["tail_fraction"] for r in radiators],
                digests=arm["digests"],
            )
        )
    return dict(
        case=f"{record['material']} {record['E0_keV']:g} keV {record['tilt_deg']:g} deg "
        f"N={record['histories']} seed+{record['seed_offset']}",
        error=record["error"] or (None if record.get("complete", True) else "incomplete"),
        rows=record.get("rows"),
        arms=rows,
    )


def table(out_dir):
    for path in sorted(Path(out_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("issue") != 1281:
            continue
        s = summary(record)
        # Compare against the full arm, else the first (the streamed reference prefix).
        full = next((a for a in s["arms"] if str(a["budget"]) == "0"), None)
        full = full or (s["arms"][0] if s["arms"] else None)
        rows = s.get("rows") or {}
        print(s["case"], s["error"] or "", f"rows/electron={rows.get('rows_per_electron')}")
        for arm in s["arms"]:
            same = None if full is None else arm["digests"] == full["digests"]
            print(
                f"  budget={arm['budget']:>8} ref={arm['reference_s']:>8.1f}s "
                f"prod={arm['production_s']:>8.1f}s cutoff={arm['cutoffs']} "
                f"n={arm['reflections']} same_as_full={same} prefix={arm['selection']}"
            )


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    if command == "run":
        run(*rest)
    elif command == "table":
        table(*rest)
    else:
        raise SystemExit(__doc__)
