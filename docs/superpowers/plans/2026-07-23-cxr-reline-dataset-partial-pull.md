# `cxr reline` + dataset-partial pull Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `cxr reline` (recompute ONLY the coherent line spectrum of existing checkpoints, mirror of `cxr rebrem`) plus dataset-partial pull that overwrites only the brem OR only the line portion of the local pickle in place.

**Architecture:** Mirror the existing brem-repair stack on the line side (`_lines_for_case` → `repair_line_spec` → `reline_checkpoint` → `cxr reline` → `start_reline_queue` → `_cli_reline`). Add a dataset projection (box-side, via `cxr slim --brem-only/--line-only`) + a local `merge_dataset` so a pull overwrites only one dataset's record keys, keeping the other intact. `rebrem`/`reline` remote auto-pulls become dataset-partial merges.

**Tech Stack:** Python, numpy, argparse, pytest. GPU Monte-Carlo transport (`cxr_mc.montecarlo`). SLURM remote runner (`cxr_mc._remote`).

## Global Constraints

- Run every test/uv command with the shared cache prefix: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test <path> -k <name>`.
- `_lines_for_case` is a pure factor-out of `_spectrum_case`'s already-validated line block — it must produce a bit-identical `spec` to a live sweep (same seed, same `mc_spectrum` args). No new physics, so no new derivation docstring / ledger row; preserve any existing `Validation:` markers moved with the code.
- Record dataset key groups live in ONE place (`cxr_mc/results/selection.py` constants) consumed by both projection and merge — never re-list them inline.
- Line seed is `case["seed"]` (brem seed is `case["seed"] + 1`).
- `E_grid` is the line grid; `brem` is always `brem_wide` interpolated onto the CURRENT `E_grid`. Any code that changes `E_grid` or `brem_wide`/`E_grid_brem` must re-derive `brem = np.interp(E_grid, E_grid_brem, brem_wide)`.
- Atomic saves only: `_checkpoint_save` + `_manifest_save` (temp+`os.replace`).
- Materials selection mirrors rebrem: positional `material` (nargs `*`) XOR `-a/--all`; remote side uses `_selected_materials(args, "material")`.

---

### Task 1: `_lines_for_case` — per-case line spectrum recompute

**Files:**
- Modify: `src/cxr_mc/montecarlo/runner.py` (add `_lines_for_case`; refactor `_spectrum_case`'s line block to call it, ~L457-508)
- Modify: `src/cxr_mc/montecarlo/__init__.py` (export `_lines_for_case` next to `_brem_for_case`)
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `simulate_trajectories`, `tilted_geometry`, `blazed_groove_spec`, `mc_spectrum`, `_segments_in_layer`, `_adaptive_chunk`, `_SPEC_CHUNK` (all already in runner.py).
- Produces: `runner._lines_for_case(case: dict, E_grid: np.ndarray) -> np.ndarray` (the line `spec` on `E_grid`), importable as `from cxr_mc.montecarlo import _lines_for_case`.

- [ ] **Step 1: Write the failing test** — a stacked/mosaic-free single-slab case relines to the same `spec` a full transport+spectrum produces, and delegates through `mc_spectrum` with the case's crystal args.

In `tests/test_run.py` (near `test_repair_brem_wide_delegates_stacked_case_to_runner`):

```python
def test_lines_for_case_matches_spectrum_case_single_slab():
    import numpy as np
    from cxr_mc.montecarlo import _lines_for_case, runner

    case = dict(
        E0_keV=30.0, Ne=200, Ne_brem=50, thickness_ang=1.0e4,
        composition={"Mo": 1, "S": 2}, crystal="mos2",
        hkl_list=[(1, 0, 0)], B_ang2=0.5, theta_obs_rad=np.deg2rad(90.0),
        seed=7, E_cut_lines_keV=5.0, E_cut_brem_keV=1.0,
    )
    E_grid = np.linspace(1000.0, 30000.0, 64)
    tp = runner._transport_case({**case, "E_grid": E_grid})
    spec_ref = runner._spectrum_case({**case, "E_grid": E_grid}, tp)["spec"]
    spec = _lines_for_case(case, E_grid)
    assert spec.shape == E_grid.shape
    np.testing.assert_allclose(spec, spec_ref, rtol=0, atol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k lines_for_case_matches`
Expected: FAIL — `ImportError: cannot import name '_lines_for_case'`.

- [ ] **Step 3: Add `_lines_for_case` and refactor the line block**

In `runner.py`, add after `_brem_for_case` (which ends ~L431). Factor the line-spectrum block out of `_spectrum_case` so both share ONE path:

```python
def _lines_for_segments(segs, E_grid, case, n_hat, abs_layers, groove):
    """Coherent line spectrum on ``E_grid`` from already-transported line
    segments ``segs``. Single slab (``layer_radiators`` absent) radiates from
    all segments via the case's scalar crystal keys; a multilayer stack sums
    each CRYSTALLINE layer's lines incoherently, every line self-absorbing
    through the whole stack. Pure move of _spectrum_case's line block, shared
    with :func:`_lines_for_case` so a line-only reline reproduces the SAME
    spectrum as a live sweep."""
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = case.get("spec_chunk") or _SPEC_CHUNK or _adaptive_chunk(E_grid.size)
    if radiators is None:
        return mc_spectrum(
            segs, E_grid,
            crystal=case["crystal"], hkl_list=case["hkl_list"], n_hat=n_hat,
            B_ang2=case["B_ang2"], composition=case["composition"],
            beam_uvw=case.get("beam_uvw"), surface_hkl=case.get("surface_hkl"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            recip_miscut_rad=case.get("recip_miscut_rad"),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk, layers=abs_layers, groove=groove, **mosaic_kw,
        )
    assert case.get("groove_spacing_ang") is None
    spec = np.zeros(E_grid.shape, dtype=float)
    for L, rad in enumerate(radiators):
        if rad is None:
            continue
        sL = _segments_in_layer(segs, L)
        if sL["L_ang"].size == 0:
            continue
        spec = spec + mc_spectrum(
            sL, E_grid,
            crystal=rad["crystal"], hkl_list=rad["hkl_list"], n_hat=n_hat,
            B_ang2=rad["B_ang2"], composition=abs_layers[L][2],
            beam_uvw=rad.get("beam_uvw"), surface_hkl=rad.get("surface_hkl"),
            azimuth_rad=rad.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
            recip_miscut_rad=rad.get("recip_miscut_rad", case.get("recip_miscut_rad")),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk, layers=abs_layers, **mosaic_kw,
        )
    return spec


def _lines_for_case(case, E_grid):
    """Regenerate a case's coherent line spectrum on ``E_grid`` from scratch:
    tilted geometry + optional groove, transport ``Ne`` electrons at ``seed``
    (the line seed, NOT ``seed + 1``), then the per-layer line spectrum via
    :func:`_lines_for_segments`. Returns ``spec``. The line half of run_case's
    transport + spectrum phases factored out so :func:`cxr_mc.run.repair_line_spec`
    (``cxr reline``) reuses the EXACT live-sweep line path -- multilayer,
    mosaic, groove and all -- rather than re-deriving it by hand."""
    abs_layers = case.get("abs_layers")
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )
    segs = simulate_trajectories(
        case["E0_keV"], case["Ne"], case["thickness_ang"],
        composition=case["composition"], E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=case["seed"], beam_dir=beam, layers=abs_layers,
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=tilt_polar_rad, tilt_azim_rad=tilt_azim_rad, groove=groove,
    )
    return _lines_for_segments(segs, E_grid, case, n_hat, abs_layers, groove)
```

Then replace the line block in `_spectrum_case` (the `if radiators is None: spec = mc_spectrum(...) else: ...` span, ~L452-508) with:

```python
    spec = _lines_for_segments(segs, E_grid, case, n_hat, abs_layers, tp.get("groove"))
```

(Delete the now-duplicated `radiators`/`mosaic_kw`/`spec_chunk` locals from `_spectrum_case`; they live in `_lines_for_segments` now. Leave the brem block below unchanged.)

- [ ] **Step 4: Export it** — in `src/cxr_mc/montecarlo/__init__.py`, add `_lines_for_case` wherever `_brem_for_case` is re-exported (same `from .runner import ...` line / `__all__`).

- [ ] **Step 5: Run the test + the existing spectrum/brem tests to prove no drift**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k "lines_for_case or repair_brem_wide"`
Expected: PASS (line factor-out is bit-identical; brem tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/montecarlo/runner.py src/cxr_mc/montecarlo/__init__.py tests/test_run.py
git commit -m "feat(runner): _lines_for_case factors the line spectrum path for reline"
```

---

### Task 2: `repair_line_spec` + `reline_checkpoint`

**Files:**
- Modify: `src/cxr_mc/run.py` (add both functions after `repair_checkpoint`, ~L560)
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `runner._lines_for_case` (Task 1); `config.material_sweep`; `sweep._line_grid_for_energy`; `_checkpoint_save`, `_manifest_save`, `_checkpoint_io.load`.
- Produces:
  - `run.repair_line_spec(results, material, only_stale=True, line_ne=None, line_step_eV=None, from_config=True, redo_all=False, save_every=0, save_cb=None, on_progress=None) -> int`
  - `run.reline_checkpoint(checkpoint_path, material, save_every=100, **kw) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
def _line_record(E0=30.0, ne=200):
    import numpy as np
    E_grid = np.linspace(1000.0, 30000.0, 32)
    E_brem = np.linspace(1000.0, 30500.0, 16)
    brem_wide = np.linspace(1.0, 0.1, 16)
    case = dict(
        E0_keV=E0, Ne=ne, Ne_brem=50, thickness_ang=1.0e4,
        composition={"Mo": 1, "S": 2}, crystal="mos2", hkl_list=[(1, 0, 0)],
        B_ang2=0.5, theta_obs_rad=np.deg2rad(90.0), seed=7,
    )
    return {
        "case": case, "E_grid": E_grid, "spec": np.ones(32),
        "brem_wide": brem_wide, "E_grid_brem": E_brem,
        "brem": np.interp(E_grid, E_brem, brem_wide),
    }


def test_repair_line_spec_rewrites_spec_and_reinterp_brem_keeps_brem_wide(monkeypatch):
    import numpy as np
    from cxr_mc import run
    monkeypatch.setattr(run.runner, "_lines_for_case",
                        lambda case, E_grid: np.full(E_grid.shape, 5.0))
    results = {"mos2@30": {30.0: _line_record()}}
    r = results["mos2@30"][30.0]
    brem_wide0 = r["brem_wide"].copy()
    n = run.repair_line_spec(results, material="mos2", line_ne=999, from_config=False)
    assert n == 1
    assert np.all(r["spec"] == 5.0)
    assert r["case"]["Ne"] == 999
    np.testing.assert_array_equal(r["brem_wide"], brem_wide0)          # brem untouched
    np.testing.assert_allclose(r["brem"],
                               np.interp(r["E_grid"], r["E_grid_brem"], r["brem_wide"]))


def test_repair_line_spec_skips_at_target(monkeypatch):
    import numpy as np
    from cxr_mc import run
    monkeypatch.setattr(run.runner, "_lines_for_case",
                        lambda case, E_grid: np.full(E_grid.shape, 5.0))
    results = {"mos2@30": {30.0: _line_record(ne=200)}}
    # same Ne, same grid, finite spec -> nothing to redo
    n = run.repair_line_spec(results, material="mos2", line_ne=200, from_config=False)
    assert n == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k repair_line_spec`
Expected: FAIL — `AttributeError: module 'cxr_mc.run' has no attribute 'repair_line_spec'`.

- [ ] **Step 3: Implement**

In `run.py` add (`import numpy as np` is already module-level; `from . import runner`-style access — use the same import style already present, e.g. `from .montecarlo import runner` at module top if not already imported, else `from .montecarlo import _lines_for_case` inside the function):

```python
def _target_line_grid(sweep, r, line_step_eV, from_config):
    """The line grid a reline should land on for record ``r``.
    ``line_step_eV`` -> uniform grid at that spacing over the record's current
    [E_grid[0], E_grid[-1]]. Else ``from_config`` -> the material's CURRENT
    E_grid_line_by_energy at this record's E0 (the bespoke-grid workflow), via
    the same lookup build_cases uses. Else the record's existing grid."""
    import numpy as np

    eg = np.asarray(r["E_grid"], float)
    if line_step_eV is not None:
        return np.arange(eg[0], eg[-1] + float(line_step_eV), float(line_step_eV))
    if from_config and sweep is not None:
        from .sweep import _line_grid_for_energy

        return np.asarray(_line_grid_for_energy(sweep, eg, float(r["case"]["E0_keV"])), float)
    return eg


def repair_line_spec(
    results,
    material,
    only_stale=True,
    line_ne=None,
    line_step_eV=None,
    from_config=True,
    redo_all=False,
    save_every=0,
    save_cb=None,
    on_progress=None,
):
    """Regenerate the coherent line ``spec`` for cached records with the CURRENT
    line path (:func:`cxr_mc.montecarlo._lines_for_case`) -- the mirror of
    :func:`repair_brem_wide`, WITHOUT recomputing the brem background.

    Target line grid per record (see :func:`_target_line_grid`): ``line_step_eV``
    (explicit uniform spacing), else ``from_config`` rebuilds it from
    ``material``'s current ``E_grid_line_by_energy`` at the record's E0 (edit
    materials.toml, ``cxr reline`` re-runs lines on the new grid), else the
    record's existing grid (pure ``line_ne`` bump). ``line_ne`` overrides
    ``case["Ne"]``.

    After relining, brem stays consistent: ``r["brem"]`` is re-interpolated from
    the RETAINED ``brem_wide`` onto the new ``E_grid`` (no brem transport). A
    record already at the target grid + Ne (and finite spec) is SKIPPED
    (``only_stale``), so a crashed/OOM run resumes; ``redo_all`` forces all.
    ``save_every``/``save_cb``/``on_progress`` have the SAME contract as
    :func:`repair_brem_wide`. Returns the number relined; mutates in place."""
    import time

    import numpy as np

    from .montecarlo import _lines_for_case

    sweep = None
    if from_config and line_step_eV is None:
        from .config import material_sweep

        try:
            sweep = material_sweep(material)
        except Exception:
            sweep = None  # unknown/derived stem -> fall back to each record's grid

    todo = []
    n_skipped = 0
    for name in results:
        for r in results[name].values():
            target = _target_line_grid(sweep, r, line_step_eV, from_config)
            eg = np.asarray(r["E_grid"], float)
            spec = r.get("spec")
            finite = spec is not None and np.isfinite(np.asarray(spec)).all()
            grid_ok = eg.shape == target.shape and np.allclose(eg, target)
            ne_ok = line_ne is None or int(r["case"].get("Ne", -1)) == int(line_ne)
            at_target = finite and grid_ok and ne_ok
            if only_stale and not redo_all and at_target:
                n_skipped += 1
                continue
            todo.append((r, target))
    if on_progress is not None:
        on_progress(0, len(todo), n_skipped)
    if not todo:
        print("nothing to reline (all records already at target line grid / Ne)")
        return 0
    print(f"relining {len(todo)} record(s)...")
    t0 = time.perf_counter()
    for k, (r, target) in enumerate(todo, 1):
        c = r["case"]
        if line_ne is not None:
            c["Ne"] = int(line_ne)
        r["spec"] = _lines_for_case(c, target)
        r["E_grid"] = target
        c["E_grid_line"] = (float(target[0]), float(target[-1]), len(target))
        bw = r.get("brem_wide")
        egb = r.get("E_grid_brem")
        if bw is not None and egb is not None:
            r["brem"] = np.interp(target, np.asarray(egb, float), np.asarray(bw, float))
        if on_progress is not None:
            on_progress(k, len(todo), n_skipped)
        if save_cb is not None and save_every and (k % save_every == 0):
            save_cb(results)
        if k % 50 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
    if save_cb is not None:
        save_cb(results)
    return len(todo)


def reline_checkpoint(checkpoint_path, material, save_every=100, **kw):
    """Load a per-material checkpoint, reline its line ``spec`` (see
    :func:`repair_line_spec`), and re-pickle it in place (atomic, resumable),
    saving every ``save_every`` records. Mirror of :func:`repair_checkpoint`."""
    if not os.path.exists(checkpoint_path):
        print(f"no such checkpoint: {checkpoint_path}")
        return {}
    results = _checkpoint_io.load(checkpoint_path)

    def save_cb(results):
        _checkpoint_save(checkpoint_path, results)
        _manifest_save(checkpoint_path, results)

    n = repair_line_spec(results, material=material, save_every=save_every, save_cb=save_cb, **kw)
    print(f"re-saved {checkpoint_path}" if n else "checkpoint unchanged")
    return results
```

Note: if `run.py` does not already expose `runner` as an attribute (the tests monkeypatch `run.runner._lines_for_case`), add `from .montecarlo import runner` at module top. Confirm the import style matches the file (it already imports `_checkpoint_io`, `os`, `time`).

- [ ] **Step 4: Run tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k repair_line_spec`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/run.py tests/test_run.py
git commit -m "feat(run): repair_line_spec + reline_checkpoint recompute line spec in place"
```

---

### Task 3: `cxr reline` CLI

**Files:**
- Create: `src/cxr_mc/reline.py`
- Modify: `src/cxr_mc/cli.py:61` (register `reline.add_subparser(sub)` next to `rebrem`)
- Test: `tests/test_run.py` (or a new `tests/test_reline.py`)

**Interfaces:**
- Consumes: `run.checkpoint_path_for`, `run.reline_checkpoint`; `scan._write_progress_record`.
- Produces: `reline.reline_checkpoints(...)`, `reline.add_subparser(sub)`, `reline._cli(args)`.

- [ ] **Step 1: Write the failing test** — `reline_checkpoints` dispatches per material and forwards flags.

```python
def test_reline_checkpoints_forwards_flags(monkeypatch, tmp_path):
    from cxr_mc import reline
    calls = []
    monkeypatch.setattr(reline, "_reline_one", None, raising=False)
    import cxr_mc.run as run
    monkeypatch.setattr(run, "reline_checkpoint",
                        lambda path, material, **kw: calls.append((material, kw)) or {})
    (tmp_path / "mos2.pkl").write_bytes(b"x")
    reline.reline_checkpoints(materials=["mos2"], checkpoint_dir=str(tmp_path),
                              line_ne=40000, line_step_eV=5.0, redo_all=True)
    assert calls == [("mos2", {"line_ne": 40000, "line_step_eV": 5.0,
                               "from_config": True, "redo_all": True, "save_every": 100})]
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k reline_checkpoints_forwards`
Expected: FAIL — `ModuleNotFoundError: No module named 'cxr_mc.reline'`.

- [ ] **Step 3: Create `src/cxr_mc/reline.py`** (structural mirror of `rebrem.py`):

```python
"""``cxr reline`` -- recompute checkpoint line spectra with new parameters.

Rewrites ONLY the line ``spec`` (and ``E_grid``) of existing per-material
checkpoints; the brem background (``brem_wide``/``E_grid_brem``) is untouched
(only re-interpolated onto the new line grid). Mirror of ``cxr rebrem``: use it
to re-run coherent lines on new bespoke ``E_grid_line_by_energy`` bounds, or to
bump the line electron count, without recomputing brem.

    cxr reline MoS2                     # re-run lines on the current-config grid
    cxr reline MoS2 --line-ne 40000     # bump line Ne, same grid
    cxr reline --all --line-step 5      # explicit 5 eV line grid, every checkpoint

Runs locally on ``checkpoints/``, on the GPU box via ``cxr remote reline``, or
by hand over ssh. Records already at the target grid + Ne are skipped, so a
crashed/OOM run resumes; ``--redo-all`` forces a full recompute.
"""

import argparse
from pathlib import Path


def reline_checkpoints(
    materials=None,
    checkpoint_dir="checkpoints",
    line_ne=None,
    line_step_eV=None,
    from_config=True,
    redo_all=False,
    save_every=100,
    progress_file=None,
):
    """Run :func:`cxr_mc.run.reline_checkpoint` over one or more materials.
    ``materials=None`` sweeps every ``*.pkl`` (excluding ``*.slim.pkl``).
    Returns ``{stem: results}``. ``progress_file`` (single material only) writes
    the same compact JSON progress record ``cxr rebrem`` does, feeding the remote
    status/attach dashboard."""
    from .run import checkpoint_path_for, reline_checkpoint

    ckpt_dir = Path(checkpoint_dir)
    if materials:
        paths = [checkpoint_path_for(m, str(ckpt_dir)) for m in materials]
    else:
        paths = sorted(str(p) for p in ckpt_dir.glob("*.pkl") if not p.name.endswith(".slim.pkl"))
        if not paths:
            print(f"no checkpoints in {ckpt_dir}")
            return {}
    if progress_file is not None and len(paths) != 1:
        raise SystemExit("progress_file needs exactly one material")
    out = {}
    for path in paths:
        stem = Path(path).stem
        print(f"== {stem} ==")
        kw = {}
        if progress_file is not None:
            from .scan import _write_progress_record

            latest = {"total_cases": 0, "cached_cases": 0, "completed_new_cases": 0}

            def _on_progress(done, todo_total, skipped, _latest=latest, _stem=stem):
                _latest.update(
                    total_cases=todo_total + skipped,
                    cached_cases=skipped,
                    completed_new_cases=done,
                )
                _write_progress_record(progress_file, material=_stem, state="running", **_latest)

            kw["on_progress"] = _on_progress
            _write_progress_record(progress_file, material=stem, state="running", **latest)
        try:
            out[stem] = reline_checkpoint(
                path,
                material=stem,
                save_every=save_every,
                line_ne=line_ne,
                line_step_eV=line_step_eV,
                from_config=from_config,
                redo_all=redo_all,
                **kw,
            )
        except BaseException:
            if progress_file is not None:
                _write_progress_record(progress_file, material=stem, state="failed", **latest)
            raise
        if progress_file is not None:
            _write_progress_record(progress_file, material=stem, state="done", **latest)
    return out


def _cli(args):
    """CLI handler -- returns None so the dict never reaches sys.exit."""
    if bool(args.material) == bool(args.all):
        raise SystemExit(
            "cxr reline: give one or more materials, or -a/--all for every checkpoint "
            "(exactly one of the two)"
        )
    reline_checkpoints(
        materials=args.material or None,
        checkpoint_dir=args.checkpoint_dir,
        line_ne=args.line_ne,
        line_step_eV=args.line_step,
        redo_all=args.redo_all,
        save_every=args.save_every,
        progress_file=args.progress_file,
    )


def add_subparser(sub):
    """Register the ``reline`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser(
        "reline",
        help="recompute ONLY the line spectra of existing checkpoints "
        "(new line grid / Ne); brem untouched",
    )
    ap.add_argument("material", nargs="*", help="material stems (checkpoints/<material>.pkl); or -a/--all")
    ap.add_argument("-a", "--all", action="store_true", help="recompute every *.pkl in the checkpoint dir")
    ap.add_argument("--line-ne", type=int, default=None, help="new line electron count (sweep default per material)")
    ap.add_argument(
        "--line-step",
        type=float,
        default=None,
        help="explicit uniform line-grid spacing [eV]; default rebuilds each "
        "record's grid from the material's current E_grid_line_by_energy config",
    )
    ap.add_argument("--redo-all", action="store_true", help="recompute every record even if already at target")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--save-every", type=int, default=100, help="re-pickle every N relined records (crash-safe)")
    ap.set_defaults(func=_cli)
    return ap
```

Remove the stray `monkeypatch.setattr(reline, "_reline_one", ...)` line from the test if you kept it — it was a placeholder; the real test only patches `run.reline_checkpoint`. Final test body:

```python
def test_reline_checkpoints_forwards_flags(monkeypatch, tmp_path):
    from cxr_mc import reline
    import cxr_mc.run as run
    calls = []
    monkeypatch.setattr(run, "reline_checkpoint",
                        lambda path, material, **kw: calls.append((material, kw)) or {})
    (tmp_path / "mos2.pkl").write_bytes(b"x")
    reline.reline_checkpoints(materials=["mos2"], checkpoint_dir=str(tmp_path),
                              line_ne=40000, line_step_eV=5.0, redo_all=True)
    assert calls == [("mos2", {"line_ne": 40000, "line_step_eV": 5.0,
                               "from_config": True, "redo_all": True, "save_every": 100})]
```

- [ ] **Step 4: Register the subcommand** — in `src/cxr_mc/cli.py`, add after the `rebrem.add_subparser(sub)` line (~L61):

```python
            reline.add_subparser(sub)
```

and add `reline` to the module's import block alongside `rebrem` (match the existing import style in `cli.py`).

- [ ] **Step 5: Run tests + smoke the parser**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_run.py -k reline_checkpoints_forwards`
Then: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr reline --help`
Expected: test PASS; `--help` shows the `reline` usage.

- [ ] **Step 6: Commit**

```bash
git add src/cxr_mc/reline.py src/cxr_mc/cli.py tests/test_run.py
git commit -m "feat(reline): cxr reline recomputes checkpoint line spectra with new params"
```

---

### Task 4: dataset projection + `cxr slim --brem-only/--line-only`

**Files:**
- Modify: `src/cxr_mc/results/selection.py` (dataset key constants + `project_dataset`)
- Modify: `src/cxr_mc/results/__init__.py` (re-export `project_dataset` if the package re-exports `slim_results`)
- Modify: `src/cxr_mc/slim.py` (add `--brem-only`/`--line-only`, thread through)
- Test: `tests/test_slim.py`

**Interfaces:**
- Consumes: `slim_results` (its `fields=` allow-list).
- Produces:
  - `results.selection.LINE_RECORD_KEYS = ("spec", "E_grid")`, `BREM_RECORD_KEYS = ("brem_wide", "brem", "E_grid_brem")`
  - `results.project_dataset(results, dataset) -> dict` (`dataset` in `{"line","brem"}`)
  - `slim_checkpoint(..., dataset=None)`; CLI `--brem-only`/`--line-only`.

- [ ] **Step 1: Write the failing test**

```python
def test_project_dataset_keeps_only_that_datasets_keys():
    from cxr_mc.results import project_dataset
    rec = {"case": {"E0_keV": 30.0}, "spec": [1.0], "E_grid": [1.0],
           "brem_wide": [2.0], "brem": [3.0], "E_grid_brem": [4.0]}
    results = {"n": {30.0: rec}}
    brem = project_dataset(results, "brem")["n"][30.0]
    line = project_dataset(results, "line")["n"][30.0]
    assert set(brem) == {"case", "brem_wide", "brem", "E_grid_brem"}
    assert set(line) == {"case", "spec", "E_grid"}
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_slim.py -k project_dataset`
Expected: FAIL — `ImportError: cannot import name 'project_dataset'`.

- [ ] **Step 3: Implement** — in `selection.py`, near `_WIDE_BREM_FIELDS`:

```python
LINE_RECORD_KEYS = ("spec", "E_grid")
BREM_RECORD_KEYS = ("brem_wide", "brem", "E_grid_brem")


def project_dataset(results, dataset):
    """Return a NEW results store carrying only ``dataset``'s record arrays
    (plus ``case``) per record -- the wire payload for a dataset-partial pull
    (``cxr slim --brem-only/--line-only``). ``dataset`` is ``"line"`` (keeps
    :data:`LINE_RECORD_KEYS`) or ``"brem"`` (:data:`BREM_RECORD_KEYS`).
    Delegates to :func:`slim_results`' ``fields`` allow-list so the drop logic
    lives in one place; ``case`` is always kept."""
    keys = {"line": LINE_RECORD_KEYS, "brem": BREM_RECORD_KEYS}.get(dataset)
    if keys is None:
        raise ValueError(f"dataset must be 'line' or 'brem', got {dataset!r}")
    return slim_results(results, fields=list(keys))
```

Re-export in `results/__init__.py` alongside `slim_results` (`from .selection import project_dataset, LINE_RECORD_KEYS, BREM_RECORD_KEYS`).

In `slim.py`, add a `dataset=None` parameter to `slim_checkpoint` (project BEFORE the other trims):

```python
def slim_checkpoint(in_path, out_path=None, *, grid=False, drop_wide_brem=False,
                    downcast=False, dataset=None, compresslevel=6, **constraints):
    ...
    material = _material_from_stem(in_path) if grid else None
    results = _checkpoint_io.load(in_path)
    if dataset is not None:
        from .results import project_dataset
        results = project_dataset(results, dataset)
    slim = slim_results(results, grid=material, drop_wide_brem=drop_wide_brem,
                        downcast=downcast, **constraints)
    ...
```

Add to `_cli`: `dataset="brem" if args.brem_only else ("line" if args.line_only else None)`, and register a mutually-exclusive group in `add_subparser`:

```python
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--brem-only", action="store_true",
                     help="keep only the brem arrays (brem_wide/brem/E_grid_brem) per record")
    grp.add_argument("--line-only", action="store_true",
                     help="keep only the line arrays (spec/E_grid) per record")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_slim.py -k project_dataset`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/results/selection.py src/cxr_mc/results/__init__.py src/cxr_mc/slim.py tests/test_slim.py
git commit -m "feat(slim): project_dataset + cxr slim --brem-only/--line-only"
```

---

### Task 5: `merge_dataset` — local dataset-partial merge

**Files:**
- Modify: `src/cxr_mc/results/selection.py` (add `merge_dataset`)
- Modify: `src/cxr_mc/results/__init__.py` (re-export `merge_dataset`)
- Test: `tests/test_slim.py`

**Interfaces:**
- Consumes: `LINE_RECORD_KEYS`, `BREM_RECORD_KEYS`.
- Produces: `results.merge_dataset(local, incoming, dataset, force=False) -> (n_merged, n_skipped)` — mutates `local` in place.

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_dataset_line_overwrites_spec_and_reinterps_brem():
    import numpy as np
    from cxr_mc.results import merge_dataset
    local = {"n": {30.0: {"case": {}, "spec": np.zeros(3), "E_grid": np.array([1.0, 2.0, 3.0]),
                          "brem_wide": np.array([10.0, 8.0, 6.0, 4.0]),
                          "E_grid_brem": np.array([1.0, 2.0, 3.0, 4.0]),
                          "brem": np.zeros(3)}}}
    incoming = {"n": {30.0: {"case": {}, "spec": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                             "E_grid": np.array([1.0, 1.5, 2.0, 2.5, 3.0])}}}
    merged, skipped = merge_dataset(local, incoming, "line")
    r = local["n"][30.0]
    assert (merged, skipped) == (1, 0)
    assert r["spec"].shape == (5,) and np.all(r["spec"] == 5.0)      # line overwritten
    np.testing.assert_array_equal(r["E_grid"], incoming["n"][30.0]["E_grid"])
    assert r["brem_wide"].shape == (4,)                              # brem_wide kept
    np.testing.assert_allclose(r["brem"], np.interp(r["E_grid"], r["E_grid_brem"], r["brem_wide"]))


def test_merge_dataset_skips_unmatched_unless_force():
    from cxr_mc.results import merge_dataset
    local = {"n": {30.0: {"case": {}, "spec": [0.0], "E_grid": [1.0]}}}
    incoming = {"n": {50.0: {"case": {}, "spec": [9.0], "E_grid": [1.0]}}}
    merged, skipped = merge_dataset(local, incoming, "line")
    assert (merged, skipped) == (0, 1)
    assert 50.0 not in local["n"]
    merged, skipped = merge_dataset(local, incoming, "line", force=True)
    assert (merged, skipped) == (1, 0)
    assert local["n"][50.0]["spec"] == [9.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_slim.py -k merge_dataset`
Expected: FAIL — `ImportError: cannot import name 'merge_dataset'`.

- [ ] **Step 3: Implement** — in `selection.py`:

```python
def merge_dataset(local, incoming, dataset, force=False):
    """Overwrite ONLY ``dataset``'s record keys in ``local`` from ``incoming``,
    matched by (config name, E0), leaving the other dataset's arrays intact.
    ``dataset`` is ``"line"`` or ``"brem"``. After copying, ``brem`` is always
    re-derived as ``interp(E_grid, E_grid_brem, brem_wide)`` so it stays
    consistent with whichever grid/brem_wide now holds (a line merge that
    changed ``E_grid`` re-interps the retained local brem_wide; a brem merge
    re-interps the fresh brem_wide onto the local line grid). Records in
    ``incoming`` absent from ``local`` are SKIPPED (reported) unless ``force``,
    which inserts them whole. Mutates ``local``; returns (n_merged, n_skipped)."""
    import numpy as np

    keys = {"line": LINE_RECORD_KEYS, "brem": BREM_RECORD_KEYS}.get(dataset)
    if keys is None:
        raise ValueError(f"dataset must be 'line' or 'brem', got {dataset!r}")
    n_merged = n_skipped = 0
    for name, by_E in incoming.items():
        for E0, inc in by_E.items():
            local_by_E = local.get(name)
            if local_by_E is None or E0 not in local_by_E:
                if force:
                    local.setdefault(name, {})[E0] = dict(inc)
                    n_merged += 1
                else:
                    n_skipped += 1
                continue
            r = local_by_E[E0]
            for k in keys:
                if k in inc:
                    r[k] = inc[k]
            bw, egb, eg = r.get("brem_wide"), r.get("E_grid_brem"), r.get("E_grid")
            if bw is not None and egb is not None and eg is not None:
                r["brem"] = np.interp(np.asarray(eg, float), np.asarray(egb, float),
                                      np.asarray(bw, float))
            n_merged += 1
    if n_skipped:
        print(f"merge_dataset: skipped {n_skipped} record(s) not present locally "
              f"(pass force=True to insert them)")
    return n_merged, n_skipped
```

Re-export in `results/__init__.py`.

- [ ] **Step 4: Run tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_slim.py -k merge_dataset`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/results/selection.py src/cxr_mc/results/__init__.py tests/test_slim.py
git commit -m "feat(results): merge_dataset overwrites one dataset's keys in place"
```

---

### Task 6: dataset-partial pull (`lifecycle.pull` + `cxr remote pull` flags)

**Files:**
- Modify: `src/cxr_mc/_remote/lifecycle.py` (`pull` gains `dataset=`, `force=`)
- Modify: `src/cxr_mc/_remote/cli.py` (`_cli_pull` + pull parser flags)
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: `results.merge_dataset`; `archive.archive_checkpoint`; `run._checkpoint_save`, `run._manifest_save`, `_checkpoint_io.load`; existing `transport._run`, `config`.
- Produces: `lifecycle.pull(stems, grid=False, drop_wide_brem=False, downcast=False, level9=False, no_sync=False, dataset=None, force=False)`.

- [ ] **Step 1: Write the failing test** — a `dataset` pull runs the box `slim --{dataset}-only`, then merges (not whole-file-replaces) and archives first.

```python
def test_pull_dataset_merges_and_archives(monkeypatch, tmp_path):
    import numpy as np
    from cxr_mc._remote import lifecycle
    from cxr_mc import _checkpoint_io

    # local checkpoint with a line record
    ckpt = tmp_path / "checkpoints" / "mos2.pkl"
    ckpt.parent.mkdir(parents=True)
    local = {"n": {30.0: {"case": {}, "spec": np.zeros(3), "E_grid": np.array([1.0, 2.0, 3.0]),
                          "brem_wide": np.array([1.0, 1.0, 1.0]), "E_grid_brem": np.array([1.0, 2.0, 3.0]),
                          "brem": np.zeros(3)}}}
    _checkpoint_io.dump(local, str(ckpt))
    # the "remote" sub-pickle (what scp would land): new spec only
    remote_tmp = tmp_path / "remote.pkl"
    _checkpoint_io.dump({"n": {30.0: {"case": {}, "spec": np.full(3, 7.0),
                                      "E_grid": np.array([1.0, 2.0, 3.0])}}}, str(remote_tmp))

    monkeypatch.setattr(lifecycle.config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle.transport, "sync_code", lambda: None)
    archived = []
    monkeypatch.setattr(lifecycle.archive, "archive_checkpoint",
                        lambda stem, **kw: archived.append(stem))
    # stub the box round-trip: ssh slim (no-op), scp copies our remote_tmp into place
    def fake_run(cmd):
        if cmd[0] == "scp":
            import shutil; shutil.copy(remote_tmp, cmd[-1])
    monkeypatch.setattr(lifecycle.transport, "_run", fake_run)

    lifecycle.pull(["mos2"], dataset="line")

    merged = _checkpoint_io.load(str(ckpt))["n"][30.0]
    assert np.all(merged["spec"] == 7.0)              # line overwritten
    assert archived == ["mos2"]                       # archived before merge
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k pull_dataset_merges`
Expected: FAIL — `pull()` has no `dataset` kwarg / whole-file path clobbers instead of merging.

- [ ] **Step 3: Implement** — in `lifecycle.py`, add imports at the point of use and a `dataset` branch at the TOP of `pull`'s per-stem loop (before the `use_slim`/whole-file branches). Add to the signature `dataset=None, force=False`. Insert, inside `for stem in stems:` `try:` block, a leading branch:

```python
            if dataset is not None:
                from .. import _checkpoint_io, archive
                from ..results import merge_dataset
                from ..run import _checkpoint_save, _manifest_save

                if not no_sync:
                    transport.sync_code()  # box projects with the same key groups
                remote_tmp = f"/tmp/{stem}.{dataset}.pkl"
                ckpt = f"{config.REMOTE_DIR}/checkpoints/{stem}.pkl"
                try:
                    transport._run([
                        "ssh", "-n", config.HOST,
                        f"cd {config.REMOTE_DIR} && {config.REMOTE_UV} run --no-sync cxr slim "
                        f"{ckpt} --{dataset}-only -o {remote_tmp}",
                    ])
                    incoming_local = dest / f".{stem}.{dataset}.incoming.pkl"
                    transport._run(["scp", f"{config.HOST}:{remote_tmp}", str(incoming_local)])
                finally:
                    transport._run(["ssh", "-n", config.HOST, f"rm -f {remote_tmp}"])
                if not local.exists():
                    print(f"warning: no local checkpoints/{stem}.pkl to merge into; skipping")
                    incoming_local.unlink(missing_ok=True)
                    continue
                archive.archive_checkpoint(stem, force=True)  # undoable via cxr restore
                base = _checkpoint_io.load(str(local))
                incoming = _checkpoint_io.load(str(incoming_local))
                incoming_local.unlink(missing_ok=True)
                n_merged, n_skipped = merge_dataset(base, incoming, dataset, force=force)
                _checkpoint_save(str(local), base)
                _manifest_save(str(local), base)
                print(f"merged {dataset} ({n_merged} rec, skipped {n_skipped}) "
                      f"-> checkpoints/{stem}.pkl")
                continue
```

(`local = dest / f"{stem}.pkl"` is already computed at the top of the loop; keep it above this branch. Confirm `archive` / `_checkpoint_io` / `run` are import-safe here — use the function-local imports shown to avoid any module cycle.)

In `_remote/cli.py` `_cli_pull`:

```python
def _cli_pull(args):
    dataset = "brem" if args.brem_only else ("line" if args.line_only else None)
    lifecycle.pull(
        _selected_materials(args, "material"),
        grid=(not args.full) and dataset is None,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        level9=args.level9,
        no_sync=args.no_sync,
        dataset=dataset,
        force=args.force,
    )
```

And in the pull parser (`p = sub.add_parser("pull", ...)`), add:

```python
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--brem-only", action="store_true",
                     help="merge ONLY the brem arrays into the local pickle (keep local line spectra)")
    grp.add_argument("--line-only", action="store_true",
                     help="merge ONLY the line spectra into the local pickle (keep local brem)")
    p.add_argument("--force", action="store_true",
                   help="with --brem-only/--line-only: insert records absent locally "
                   "(default: skip + warn)")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k pull_dataset_merges`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/_remote/lifecycle.py src/cxr_mc/_remote/cli.py tests/test_remote.py
git commit -m "feat(remote): dataset-partial pull merges one dataset into the local pickle"
```

---

### Task 7: remote reline queue (`cxr remote reline`)

**Files:**
- Modify: `src/cxr_mc/_remote/scripts.py` (`_reline_flags`, `_reline_queue_script`, `_reline_queue_metadata`)
- Modify: `src/cxr_mc/_remote/lifecycle.py` (`start_reline_queue`)
- Modify: `src/cxr_mc/_remote/cli.py` (`_cli_reline` + `reline` subparser)
- Modify: `src/cxr_mc/_remote/presentation.py` (`_mode_summary` recognises `kind: reline`, mirroring `rebrem`)
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: `scripts._slurm_batch_script`, `scripts._write_job_script_command`, `scripts._submit_slurm_command`, `transport`, `viewer.attach`, `state._completed_materials`, `lifecycle.pull(dataset="line")`, `_selected_materials`.
- Produces: `lifecycle.start_reline_queue(materials, line_ne=None, line_step_eV=None, redo_all=False, no_sync=False, dry_run=False) -> jobid`; `_cli_reline(args)`.

- [ ] **Step 1: Write the failing test** — the reline queue script runs `cxr reline` per material with the line flags + progress file, and its metadata is `kind: reline`.

```python
def test_reline_queue_script_and_metadata():
    from cxr_mc._remote import scripts
    s = scripts._reline_queue_script("J1", ["mos2", "w"], line_ne=40000,
                                     line_step_eV=None, redo_all=True)
    assert "cxr reline" in s and "--line-ne 40000" in s and "--redo-all" in s
    assert "--progress-file" in s
    meta = scripts._reline_queue_metadata("J1", ["mos2", "w"], 40000, None, True)
    assert "kind: reline" in meta and "line_ne: 40000" in meta
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k reline_queue_script`
Expected: FAIL — `AttributeError: module has no attribute '_reline_queue_script'`.

- [ ] **Step 3: Implement.** In `scripts.py`, after `_rebrem_queue_metadata` (~L272):

```python
def _reline_flags(line_ne, line_step_eV, redo_all):
    flags = ""
    if line_ne is not None:
        flags += f" --line-ne {int(line_ne)}"
    if line_step_eV is not None:
        flags += f" --line-step {float(line_step_eV):g}"
    if redo_all:
        flags += " --redo-all"
    return flags


def _reline_queue_script(jobid, materials, line_ne, line_step_eV, redo_all):
    """CXR payload for a line-only checkpoint recompute (``cxr reline``) in a
    SLURM allocation. One sequential reline per material; each writes the same
    per-material JSON progress record a scan/rebrem does, and the
    ``completed:``/``failed:`` markers match so ``state._completed_materials``
    drives the post-attach pull unchanged."""
    flags = _reline_flags(line_ne, line_step_eV, redo_all)
    mats = " ".join(materials)
    jobdir = f"{config.REMOTE_DIR}/{config.JOBS_SUBDIR}/{jobid}"
    return f"""JOBDIR="{jobdir}"
cd "{config.REMOTE_DIR}" || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{config.REMOTE_UV} sync >> "$JOBDIR/log" 2>&1 || {{ echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
mats=({mats})
total=${{#mats[@]}}
n=0
failures=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  if ! {config.REMOTE_UV} run --no-sync cxr reline "$m"{flags} \
    --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1
  then
    echo "WARNING: reline failed for $m; continuing" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    failures=$((failures + 1))
    echo "failed: $m" >> "$JOBDIR/log"
    continue
  fi
  echo "completed: $m" >> "$JOBDIR/log"
done
if [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


def _reline_queue_metadata(jobid, materials, line_ne, line_step_eV, redo_all):
    """Static metadata for a reline queue. ``kind: reline`` keys the Mode line."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {' '.join(materials)}",
            "quick: False",
            "kind: reline",
            f"line_ne: {line_ne}",
            f"line_step_eV: {line_step_eV}",
            f"redo_all: {bool(redo_all)}",
            "progress_dashboard: True",
            "",
        ]
    )
```

In `lifecycle.py`, add `start_reline_queue` immediately after `start_rebrem_queue` (mirror it exactly, swapping the script/metadata builders and the Mode/flags):

```python
def start_reline_queue(materials, line_ne=None, line_step_eV=None, redo_all=False,
                       no_sync=False, dry_run=False):
    """Submit a line-only checkpoint recompute (``cxr reline``) to SLURM.
    Reserves the same ``<material>.pkl`` stems as a sweep/rebrem. Returns the
    local job id."""
    transport._check_materials(materials)
    if not dry_run:
        _refuse_if_busy(materials, False)
    jobid = scripts._new_jobid()
    jobdir = f"{config.REMOTE_DIR}/{config.JOBS_SUBDIR}/{jobid}"
    stems = list(materials)
    payload = scripts._reline_queue_script(jobid, materials, line_ne, line_step_eV, redo_all)
    script = scripts._slurm_batch_script(
        jobid, payload, job_name=f"cxr-reline-{jobid}", reservation_stems=stems
    )
    upload = scripts._write_job_script_command(
        jobdir, scripts._reline_queue_metadata(jobid, materials, line_ne, line_step_eV, redo_all)
    )
    submit = scripts._submit_slurm_command(jobid, stems)
    if dry_run:
        print(f"# reline job {jobid}: {' '.join(materials)} "
              f"line_ne={line_ne} line_step={line_step_eV} redo_all={redo_all}")
        print(f"# --- ssh {config.HOST}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {config.HOST}: {submit}")
        return jobid
    if not no_sync:
        transport.sync_code()
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems)
    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + presentation._format_fields([
            ("SLURM", scheduler_id),
            ("Host", config.HOST),
            ("Materials", ", ".join(materials)),
            ("Mode", presentation._mode_summary(
                scripts._reline_queue_metadata(jobid, materials, line_ne, line_step_eV, redo_all))),
            ("Attach", f"cxr remote attach {jobid}"),
            ("Status", f"cxr remote status {jobid} -vv"),
            ("Logs", f"cxr remote logs {jobid} --follow"),
            ("Pull", f"cxr remote pull {' '.join(stems)} --line-only  (after completion)"),
        ])
    )
    return jobid
```

In `cli.py`, add `_cli_reline` next to `_cli_rebrem`:

```python
def _cli_reline(args):
    """Submit a line-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials(args, "material")
    jobid = lifecycle.start_reline_queue(
        materials, line_ne=args.line_ne, line_step_eV=args.line_step,
        redo_all=args.redo_all, no_sync=args.no_sync, dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    if not viewer.attach(jobid):
        print("reline is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM reline updated no checkpoints successfully; nothing to pull")
        return
    lifecycle.pull(completed, dataset="line")
```

Register the subparser after the rebrem block (mirror rb, ~L319):

```python
    rl = sub.add_parser(
        "reline",
        help="recompute line-only in the box's checkpoints (GPU), follow, and pull them back",
    )
    rl.add_argument("material", nargs="*", help="one or more crystal keys")
    rl.add_argument("-a", "--all", action="store_true", help="every material in mats_to_sim.toml")
    rl.add_argument("--line-ne", type=int, default=None, help="new line electron count")
    rl.add_argument("--line-step", type=float, default=None,
                    help="explicit uniform line-grid spacing [eV] (default: from config)")
    rl.add_argument("--redo-all", action="store_true",
                    help="recompute every record even if already at target")
    rl.add_argument("--no-sync", action="store_true", help="skip the code upload")
    rl.add_argument("--dry-run", action="store_true",
                    help="print the SLURM batch script + submission command, don't ssh")
    rl.set_defaults(func=_dispatch(_cli_reline))
```

In `presentation.py`, find `_mode_summary`'s `kind == "rebrem"` branch and add a sibling `reline` branch (render `line_ne`/`line_step` the way rebrem renders `ne_brem`/`step`). Read the function first; mirror its exact format string.

- [ ] **Step 4: Run tests to verify pass + a dry-run smoke**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k reline`
Then: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr remote reline mos2 --line-ne 40000 --dry-run`
Expected: test PASS; dry-run prints the batch script containing `cxr reline mos2 --line-ne 40000`.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/_remote/scripts.py src/cxr_mc/_remote/lifecycle.py src/cxr_mc/_remote/cli.py src/cxr_mc/_remote/presentation.py tests/test_remote.py
git commit -m "feat(remote): cxr remote reline submits line-only recompute and line-only pull"
```

---

### Task 8: rebrem auto-pull becomes dataset-partial (`dataset="brem"`)

**Files:**
- Modify: `src/cxr_mc/_remote/cli.py` (`_cli_rebrem` trailing pull)
- Test: `tests/test_remote.py`

**Interfaces:**
- Consumes: `lifecycle.pull(dataset="brem")`.

- [ ] **Step 1: Write the failing test** — `_cli_rebrem`'s pull now passes `dataset="brem"`.

```python
def test_cli_rebrem_pulls_brem_dataset(monkeypatch):
    from cxr_mc._remote import cli, lifecycle, viewer, state
    monkeypatch.setattr(lifecycle, "start_rebrem_queue", lambda *a, **k: "J1")
    monkeypatch.setattr(viewer, "attach", lambda jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: ["mos2"])
    captured = {}
    monkeypatch.setattr(lifecycle, "pull",
                        lambda stems, **kw: captured.update(stems=stems, kw=kw))
    args = type("A", (), dict(material=["mos2"], all=False, ne_brem=None, step=None,
                              redo_all=False, no_sync=False, dry_run=False,
                              remote_command="rebrem"))()
    cli._cli_rebrem(args)
    assert captured["stems"] == ["mos2"]
    assert captured["kw"].get("dataset") == "brem"
```

- [ ] **Step 2: Run to verify failure**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k cli_rebrem_pulls_brem`
Expected: FAIL — current call is `lifecycle.pull(completed)` with no `dataset`.

- [ ] **Step 3: Implement** — change the last line of `_cli_rebrem`:

```python
    # merge ONLY the fresh brem into the local pickle, preserving any local line spectra
    lifecycle.pull(completed, dataset="brem")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_remote.py -k cli_rebrem_pulls_brem`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cxr_mc/_remote/cli.py tests/test_remote.py
git commit -m "feat(remote): rebrem auto-pull merges only brem, preserving local line spectra"
```

---

### Task 9: docs — repo map, help text, verify sweep

**Files:**
- Modify: `docs/repo_map.md` (add reline module map row + dataset-partial pull pointer)
- Modify: `TODO.md` (this branch — note the reline/dataset-pull work landed)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a repo-map entry** for the reline stack, mirroring the rebrem row. Include:
  - `src/cxr_mc/reline.py` — `cxr reline` driver (mirror `rebrem.py`)
  - `run.repair_line_spec` / `run.reline_checkpoint` — in-place line recompute
  - `runner._lines_for_case` / `_lines_for_segments` — shared live-sweep line path
  - `results.project_dataset` / `results.merge_dataset` + `LINE_RECORD_KEYS`/`BREM_RECORD_KEYS` — dataset-partial transfer
  - `lifecycle.pull(dataset=...)` + `cxr remote reline`

- [ ] **Step 2: Update `TODO.md`** on this branch with the completed items (follow the branch TODO conventions in `AGENTS.md`).

- [ ] **Step 3: Run the full verification suite**

Run: `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify`
Expected: PASS (tests + lint + typecheck + format). Fix any lint/type nits the new code introduces (unused imports, line length).

- [ ] **Step 4: Commit**

```bash
git add docs/repo_map.md TODO.md
git commit -m "docs(reline): map the reline + dataset-partial-pull stack"
```

---

## Self-Review

**Spec coverage:**
- Spec A (line recompute core) → Tasks 1, 2. `_lines_for_case`, `repair_line_spec`, `reline_checkpoint`, re-interp brem, from-config + `--line-step` grid selection, resumable skip. ✓
- Spec B (`cxr reline` CLI) → Task 3. ✓
- Spec C (dataset projection + slim flags) → Task 4. ✓
- Spec D (local merge, skip+warn, force) → Task 5. ✓
- Spec E (dataset-partial pull, archive-before-merge, no digest fast-path) → Task 6. ✓
- Spec F (remote reline queue) → Task 7. ✓
- Spec G (rebrem auto-pull `dataset="brem"`, `cxr remote pull --brem-only/--line-only`) → Tasks 6 (pull flags) + 8 (rebrem). ✓
- Symmetry table, testing plan → Tasks 1-8 tests + Task 9 verify. ✓

**Placeholder scan:** No TBD/TODO in code steps; every step shows the code or exact command. The one stray placeholder line in Task 3's first draft test is explicitly corrected in Step 3.

**Type consistency:**
- `_lines_for_case(case, E_grid) -> spec` used identically in Task 2 (`run.repair_line_spec`) and monkeypatched in Task 2/3 tests.
- `repair_line_spec(results, material, ..., line_ne, line_step_eV, from_config, redo_all, save_every, save_cb, on_progress)` — same signature consumed by `reline_checkpoint` (Task 2) and `reline_checkpoints` (Task 3).
- `merge_dataset(local, incoming, dataset, force) -> (int, int)` — defined Task 5, consumed Task 6.
- `project_dataset(results, dataset)` — defined Task 4, consumed by `cxr slim` (Task 4) and indirectly by the box in Task 6.
- `lifecycle.pull(..., dataset=None, force=False)` — extended Task 6, consumed Tasks 6/7/8.
- Dataset key constants `LINE_RECORD_KEYS`/`BREM_RECORD_KEYS` — single definition (Task 4), consumed by `merge_dataset` (Task 5).
- `_reline_queue_script`/`_reline_queue_metadata`/`start_reline_queue`/`_cli_reline` — consistent names across Task 7.
