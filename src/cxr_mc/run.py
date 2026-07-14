"""
run.py
==========

Drive a sweep through ``montecarlo.run_cases`` with on-disk checkpointing
and per-group streaming feedback.

:func:`run_sweep` resumes from a per-material checkpoint
(``checkpoints/<material>.pkl``, skipping cached cases), stores each finished
case into ``results``, re-pickles that material's checkpoint at config
granularity (crash-safe), and invokes ``on_chunk(group_config_names)`` once a
whole **group**
has finished. By default a group is everything that shares
(material, thickness, polar tilt, finite footprint) -- i.e. the full azimuth
sweep at one tilt and footprint --
so the streamed plot/table waits until every azimuth is in and can collapse to
the best azimuth per energy (see plots.stream_chunk).

run_cases shows a tqdm bar over completed cases; it renders once at the top of
the cell, so as streamed plots/tables pile up below it scrolls out of view --
hence the explicit "tilt N/M" progress line printed with each chunk here, which
stays next to the latest output. With a GPU present the run is serial (one CUDA
context); see montecarlo.run_cases.
"""

import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

from . import _checkpoint_io
from .montecarlo import run_cases
from .results import store_result

# Anchored to the repo root (src/cxr_mc/run.py -> parents[2] = repo root) so
# checkpoint lookup works regardless of the notebook's kernel cwd.
_DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parents[2] / "checkpoints")


def checkpoint_path_for(material, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Path to the per-material results checkpoint run_sweep writes."""
    return os.path.join(checkpoint_dir, f"{material}.pkl")


def _checkpoint_save(checkpoint_path, results):
    """Atomically gzip-pickle ``results`` to ``checkpoint_path`` (see
    :mod:`cxr_mc._checkpoint_io`, TODO P2 #8): write a sibling ``.<pid>.tmp``
    then ``os.replace`` it into place. The replace is atomic on a single
    filesystem, so a crash/OOM mid-write never leaves a half-written ``.pkl``
    -- the old checkpoint survives intact and the run stays resumable.

    The pid in the temp name keeps two processes writing the *same* checkpoint
    from sharing one ``.tmp``: with a fixed name they clobber each other's temp
    and race on the rename -- the first ``os.replace`` consumes it, the second
    dies with ``FileNotFoundError``. (The remote runner also guards against
    concurrent same-material jobs, but this makes the save safe on its own.)
    Shared by :func:`run_sweep` (per-config crash-safe saves) and
    :func:`repair_checkpoint`."""
    tmp = f"{checkpoint_path}.{os.getpid()}.tmp"
    _checkpoint_io.dump(results, tmp)
    os.replace(tmp, checkpoint_path)


def load_checkpoint(material, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR):
    """Load a per-material results checkpoint (``checkpoints/<material>.pkl``)
    written by :func:`run_sweep`, WITHOUT re-running anything -- this is how the
    visualization app (``notebooks/analysis_app.py``) gets its ``results`` after the
    scan-runner app (``notebooks/scan_app.py``) has produced them. Returns the
    ``{name: {E0: record}}`` store (empty dict if the checkpoint is missing).

    Reconstruct the sweep's case list straight from it with
    ``cases = [r["case"] for r in results.records(results)]`` -- the records
    carry their own cases, so the visualization app needs no Sweep to filter/plot."""
    path = checkpoint_path_for(material, checkpoint_dir)
    if not os.path.exists(path):
        print(f"no checkpoint at {path} -- run `cxr scan {material}` first")
        return {}
    results = _checkpoint_io.load(path)
    n = sum(len(v) for v in results.values())
    print(f"loaded {n} {material} records from {path}")
    return results


def cases_from_results(results):
    """The flat case list backing a loaded ``results`` store (each record carries
    its own ``case``) -- pass to filter_results / plot_heatmaps / the trajectory
    grid so the visualization app never has to rebuild the Sweep."""
    return [rec["case"] for recs in results.values() for rec in recs.values()]


def _default_group_key(case):
    """Everything but azimuth and energy: the sweep at one tilt and footprint.

    Missing footprint fields represent the legacy laterally infinite slab, so
    older case dictionaries continue to group together.
    """
    return (
        case["crystal"],
        case["thickness_ang"],
        case["tilt_deg"],
        case.get("crystal_width_mm"),
        case.get("crystal_height_mm"),
    )


def run_sweep(
    cases,
    results,
    *,
    checkpoint_dir=_DEFAULT_CHECKPOINT_DIR,
    checkpoint_path=None,
    resume=True,
    max_workers=None,
    progress=True,
    group_key=None,
    on_chunk=None,
):
    """Run ``cases`` into ``results`` (mutated in place).

    cases : list of case dicts from sweep.build_cases.
    results : the dict to fill ({name: {E0: record}}).
    checkpoint_dir : directory for the per-material checkpoints. A sweep is
        single-material, so each writes ``<checkpoint_dir>/<material>.pkl``
        holding ONLY that material's configs -- different materials never share a
        pickle, and ``results`` can hold several materials in one kernel without
        them clobbering each other on disk. Created if missing; seeded once from
        the legacy combined ``cxr_run_checkpoint.pkl`` if that's still around.
    checkpoint_path : explicit override for the pickle path; None (default)
        derives the per-material path above. A rerun with resume=True skips every
        config already in the pickle.
    max_workers : forwarded to run_cases (None -> serial on GPU, ~3/4 cores on
        CPU); >1 on a GPU is coerced to serial there.
    progress : forwarded to run_cases (the per-case tqdm bar).
    group_key : case -> hashable. Configs sharing a key form one group;
        on_chunk fires once the WHOLE group has finished. Default groups by
        (material, thickness, polar tilt, finite footprint), so the azimuth
        sweep at one tilt and footprint is one group -- on_chunk gets every
        azimuth at once.
    on_chunk : optional callback(list_of_config_names) fired once per completed
        group, with all of that group's config names (cached + freshly run).
    """
    if group_key is None:
        group_key = _default_group_key

    # per-material checkpoint: a sweep is single-material, so name the pickle for
    # the crystal and keep them together in their own subdir.
    material = cases[0]["crystal"] if cases else "mixed"
    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_for(material, checkpoint_dir)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    def _crystal_of(rec_map):
        return next(iter(rec_map.values()))["case"]["crystal"]

    def _save():
        """Pickle just THIS material's configs -- ``results`` may also hold other
        materials run earlier in the same kernel, which belong in their own pkl."""
        subset = {n: results[n] for n in results if _crystal_of(results[n]) == material}
        _checkpoint_save(checkpoint_path, subset)

    if resume and os.path.exists(checkpoint_path):
        loaded = _checkpoint_io.load(checkpoint_path)
        results.update(loaded)
        print(
            f"resumed {sum(len(v) for v in loaded.values())} {material} cases from {checkpoint_path}"
        )

    todo = [c for c in cases if not (c["name"] in results and c["E0_keV"] in results[c["name"]])]
    print(f"{len(todo)} of {len(cases)} cases to run ({len(cases) - len(todo)} cached)")

    # all config names in each group (cached + to-run), in first-seen order
    group_names = defaultdict(list)
    seen = set()
    for c in cases:
        if c["name"] not in seen:
            seen.add(c["name"])
            group_names[group_key(c)].append(c["name"])

    # per group: how many of its configs still have unfinished energies;
    # per config: how many beam energies it still owes
    group_remaining = defaultdict(int)
    energies_remaining = {}
    for c in todo:
        if c["name"] not in energies_remaining:
            group_remaining[group_key(c)] += 1
        energies_remaining[c["name"]] = energies_remaining.get(c["name"], 0) + 1
    n_groups = len(group_remaining)
    progress_state = {"done": 0}

    def _cb(i, case, out):
        store_result(results, case, out)
        name = case["name"]
        energies_remaining[name] -= 1
        if energies_remaining[name] == 0:  # this config (all energies) is done
            _save()  # crash-safe at config granularity (this material's subset)
            g = group_key(case)
            group_remaining[g] -= 1
            if group_remaining[g] == 0 and on_chunk is not None:
                # the whole azimuth sweep at this tilt is in -> stream it, with a
                # progress line that stays next to the streamed plot/table (the
                # tqdm bar is up top and scrolls away)
                progress_state["done"] += 1
                print(
                    f"\n=== {case['thickness_ang'] / 1e4:g} um, tilt "
                    f"{case['tilt_deg']:g} deg done "
                    f"-- {progress_state['done']}/{n_groups} tilt-groups ==="
                )
                on_chunk(group_names[g])

    # Fully-cached tilt-groups have no cases left to run, so the callback above
    # never fires for them. On resume, replay them through on_chunk first (in
    # tilt order) so their best-azimuth plots/tables are redrawn too.
    if on_chunk is not None:
        for g in sorted(g for g in group_names if group_remaining.get(g, 0) == 0):
            rep = next(iter(results[group_names[g][0]].values()))["case"]
            print(
                f"\n=== cached: {rep['thickness_ang'] / 1e4:g} um, "
                f"tilt {rep['tilt_deg']:g} deg (already computed) ==="
            )
            on_chunk(group_names[g])

    t0 = time.perf_counter()
    run_cases(todo, max_workers=max_workers, progress=progress, callback=_cb)
    print(f"{len(todo)} cases in {time.perf_counter() - t0:.0f} s")


# ---- brem-only repair --------------------------------------------------------
def repair_brem_wide(results, only_nonfinite=True, progress=True, save_every=0, save_cb=None):
    """Regenerate ``brem_wide`` (and the line-grid ``brem``) for cached records
    using the CURRENT ``mc_brem_spectrum`` -- WITHOUT re-running the expensive
    line spectrum.

    Records computed before the brem-absorption fix have NaN ``brem_wide`` above
    the Henke ceiling (~30 keV) and at E=0, which the log plots drop (the
    full-range curve clips at 30 keV). This re-runs only the cheap bremsstrahlung
    transport (``Ne_brem`` electrons, same seed as the original case) per record
    and rewrites ``r["brem_wide"]`` / ``r["brem"]`` in place; the line ``spec`` is
    left untouched. Everything needed is already stored in ``r["case"]``.

    The transport + per-layer brem sum is delegated to the runner's
    ``_brem_for_case`` -- the SAME path a live sweep takes -- so a film-on-
    substrate / multilayer record is repaired with its full stacked brem
    (``layers=abs_layers``, per-layer Z^2 sum, ``brem_chunk`` honored) rather
    than being silently rewritten as single-slab brem.

    only_nonfinite : skip records whose ``brem_wide`` is already all-finite
        (default), so only the stale ones are touched. Set False to redo all.
    save_every / save_cb : if both set, call ``save_cb(results)`` every
        ``save_every`` repaired records (and once at the end) -- used by
        :func:`repair_checkpoint` to checkpoint progress so a crash/OOM doesn't
        lose everything; a re-run then resumes (only_nonfinite skips the saved ones).
    Returns the number of records repaired. Mutates ``results`` in place; re-pickle
    the checkpoint afterwards (or use :func:`repair_checkpoint`).
    """
    import numpy as np

    from .montecarlo import _brem_for_case

    todo = []
    for name in results:
        for r in results[name].values():
            bw = r.get("brem_wide")
            if only_nonfinite and bw is not None and np.isfinite(np.asarray(bw)).all():
                continue
            todo.append(r)
    if not todo:
        print("nothing to repair (all brem_wide already finite)")
        return 0
    print(f"repairing brem for {len(todo)} record(s)...")
    t0 = time.perf_counter()
    for k, r in enumerate(todo, 1):
        c = r["case"]
        if "E_grid_brem" in c:  # (start, stop, step) tuple
            E_brem = np.arange(*c["E_grid_brem"])
        elif r.get("E_grid_brem") is not None and np.asarray(r["E_grid_brem"]).size:
            E_brem = np.asarray(r["E_grid_brem"], float)
        else:  # legacy single-grid record
            step_b = c.get("brem_step_eV", 10.0)
            eg = np.asarray(r["E_grid"], float)
            E_brem = np.arange(eg[0], eg[-1] + step_b, step_b)
        brem_wide = _brem_for_case(c, E_brem)
        r["brem_wide"] = brem_wide
        r["E_grid_brem"] = E_brem
        r["brem"] = np.interp(np.asarray(r["E_grid"], float), E_brem, brem_wide)
        if save_cb is not None and save_every and (k % save_every == 0):
            save_cb(results)
            if progress:
                print(f"    ...saved progress ({k}/{len(todo)})")
        if progress and (k % 50 == 0 or k == len(todo)):
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
    if save_cb is not None:
        save_cb(results)  # final save
    return len(todo)


def repair_checkpoint(checkpoint_path, save_every=100, **kw):
    """Load a per-material checkpoint, repair its ``brem_wide`` (see
    :func:`repair_brem_wide`), and re-pickle it in place -- saving progress every
    ``save_every`` records (atomic temp+replace) so a crash/OOM is RESUMABLE: just
    call again and only_nonfinite picks up where it left off. Returns the repaired
    ``results`` dict (also usable directly in the notebook)."""
    if not os.path.exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    results = _checkpoint_io.load(checkpoint_path)

    save_cb = partial(_checkpoint_save, checkpoint_path)  # atomic; resumable on crash
    n = repair_brem_wide(results, save_every=save_every, save_cb=save_cb, **kw)
    print(f"re-saved {checkpoint_path}" if n else "checkpoint unchanged")
    return results
