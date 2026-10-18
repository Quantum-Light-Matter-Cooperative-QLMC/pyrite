"""Pure checkpoint repair selection and target planning; no scoring or writes."""

from dataclasses import dataclass

import numpy as np

from .._energy_grid_encoding import decode_energy_grid, encode_energy_grid


def _validate_grid(grid, name, minimum=1):
    """Reject unusable coordinates before any selected record is changed."""
    if (
        grid.ndim != 1
        or grid.size < minimum
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(f"{name} must be a finite, strictly increasing one-dimensional grid")
    return grid


def _uniform_grid(start, stop, step):
    if not np.isfinite([start, stop, step]).all() or step <= 0 or stop <= start:
        raise ValueError("grid bounds must be finite, with stop > start and step > 0")
    return np.arange(start, stop, step)


def _stored_brem_grid(r):
    """The brem grid a cached record was computed on: the case's encoded grid,
    the record's stored array, or the legacy single-grid fallback."""
    c = r["case"]
    if "E_grid_brem" in c:
        return decode_energy_grid(c["E_grid_brem"])
    if r.get("E_grid_brem") is not None and np.asarray(r["E_grid_brem"]).size:
        return np.asarray(r["E_grid_brem"], float)
    # legacy single-grid record
    step_b = c.get("brem_step_eV", 10.0)
    eg = np.asarray(r["E_grid"], float)
    return np.arange(eg[0], eg[-1] + step_b, step_b)


@dataclass(frozen=True)
class _BremRequest:
    ne_brem: int | None
    brem_start_eV: float | None
    brem_stop_eV: float | None
    brem_step_eV: float | None
    target_nodes: np.ndarray | None
    fidelity: str | None

    @property
    def retune(self):
        return self.target_nodes is not None or any(
            value is not None
            for value in (
                self.ne_brem,
                self.brem_start_eV,
                self.brem_stop_eV,
                self.brem_step_eV,
                self.fidelity,
            )
        )


def _brem_at_target(r, request):
    bw = r.get("brem_wide")
    bw_array = np.asarray(bw) if bw is not None else np.asarray([])
    finite = bw_array.size > 0 and np.isfinite(bw_array).all() and np.any(bw_array != 0)
    if not finite or not request.retune:
        return finite
    c = r["case"]
    if request.ne_brem is not None and int(c.get("Ne_brem", -1)) != int(request.ne_brem):
        return False
    if request.fidelity is not None and c.get("brem_profile") != request.fidelity:
        return False
    return _brem_grid_matches(r, request)


def _brem_grid_matches(r, request):
    if request.target_nodes is not None:
        g = _stored_brem_grid(r)
        return g.shape == request.target_nodes.shape and np.array_equal(g, request.target_nodes)
    if not _brem_uniform_requested(request):
        return True
    g = _stored_brem_grid(r)
    # Preserve legacy step-only eligibility: matching the first interval is
    # enough, even when a graded grid's later intervals or extent differ.
    if request.brem_start_eV is None and request.brem_stop_eV is None:
        return g.size >= 2 and np.isclose(g[1] - g[0], float(request.brem_step_eV))
    target = _uniform_grid(*_brem_uniform_bounds(r, request))
    return g.shape == target.shape and np.allclose(g, target)


def _brem_uniform_requested(request):
    return any(
        value is not None
        for value in (
            request.brem_start_eV,
            request.brem_stop_eV,
            request.brem_step_eV,
        )
    )


def _brem_uniform_bounds(r, request):
    stored = _stored_brem_grid(r)
    if (not stored.size and request.brem_start_eV is None) or (
        stored.size < 2 and request.brem_step_eV is None
    ):
        raise ValueError("stored brem grid cannot supply the requested bounds or spacing")
    step = (
        float(request.brem_step_eV)
        if request.brem_step_eV is not None
        else float(stored[1] - stored[0])
    )
    start = float(request.brem_start_eV) if request.brem_start_eV is not None else float(stored[0])
    stop = (
        float(request.brem_stop_eV)
        if request.brem_stop_eV is not None
        else float(r["case"]["E0_keV"]) * 1e3 + step
    )
    return start, stop, step


def _brem_target(r, request):
    updates = {}
    if request.ne_brem is not None:
        updates["Ne_brem"] = int(request.ne_brem)
    if request.fidelity is not None:
        updates["brem_profile"] = request.fidelity
    if request.target_nodes is not None:
        # A graded target is already final: build_cases likewise passes a
        # nonuniform grid through for every beam energy instead of extending
        # it to E0, so the retuned record matches what a live sweep would get.
        E_brem = request.target_nodes
        updates["E_grid_brem"] = encode_energy_grid(E_brem)
    elif _brem_uniform_requested(request):
        bounds = _brem_uniform_bounds(r, request)
        E_brem = _uniform_grid(*bounds)
        updates["E_grid_brem"] = bounds
    else:
        E_brem = _stored_brem_grid(r)
    return _validate_grid(E_brem, "brem grid"), updates


def _plan_brem(results, request, only_nonfinite):
    todo = []
    skipped = 0
    if request.target_nodes is not None:
        _validate_grid(request.target_nodes, "brem_nodes", minimum=2)
    for records in results.values():
        for record in records.values():
            if only_nonfinite and _brem_at_target(record, request):
                skipped += 1
                continue
            grid, updates = _brem_target(record, request)
            todo.append((record, grid, updates))
    return todo, skipped


def _target_cxr_grid(
    sweep,
    r,
    line_step_eV,
    from_config,
    cxr_start_eV=None,
    line_stop_eV=None,
):
    """The line grid a reline should land on for record ``r``.

    ``line_step_eV`` -> uniform grid at that spacing over the record's current
    [E_grid[0], E_grid[-1]]. Else ``from_config`` -> the material's CURRENT
    E_grid_line_by_energy at this record's E0 (the bespoke-grid workflow), via
    the same lookup build_cases uses. Else the record's existing grid."""
    eg = np.asarray(r["E_grid"], float)
    if from_config and sweep is not None:
        from ..campaign.sweep import _cxr_grid_for_energy, sweep_crystal_params

        # Coordinates only: a returned policy payload describes a resolution the
        # runner would refine from its own trajectories, and reline pins the
        # explicit coordinates instead (it drops `case["line_grid_policy"]`).
        coordinates, _policy = _cxr_grid_for_energy(
            sweep, sweep_crystal_params(sweep), float(r["case"]["E0_keV"])
        )
        base = np.asarray(coordinates, float)
    else:
        base = eg
    if any(value is not None for value in (cxr_start_eV, line_stop_eV, line_step_eV)):
        from ..checkpoints.recompute_defaults import uniform_bounds

        start, stop, step = uniform_bounds(base)
        return _uniform_grid(
            start if cxr_start_eV is None else float(cxr_start_eV),
            stop if line_stop_eV is None else float(line_stop_eV),
            step if line_step_eV is None else float(line_step_eV),
        )
    return base


def _cxr_at_target(r, target, cxr_ne, fidelity):
    eg = np.asarray(r["E_grid"], float)
    spec = r.get("spec")
    finite = spec is not None and np.isfinite(np.asarray(spec)).all()
    if r["case"].get("channeling") is not None:
        channeling = r.get("spec_channeling")
        finite = finite and channeling is not None
        if channeling is not None:
            channeling = np.asarray(channeling)
            finite = finite and channeling.shape == eg.shape and np.isfinite(channeling).all()
    grid_ok = eg.shape == target.shape and np.allclose(eg, target)
    ne_ok = cxr_ne is None or int(r["case"].get("Ne", -1)) == int(cxr_ne)
    fidelity_ok = fidelity is None or r["case"].get("line_profile") == fidelity
    at_target = finite and grid_ok and ne_ok and fidelity_ok
    return at_target


def _plan_cxr(
    results,
    sweep,
    only_stale,
    redo_all,
    cxr_ne,
    fidelity,
    line_step_eV,
    from_config,
    cxr_start_eV,
    line_stop_eV,
):
    todo = []
    skipped = 0
    for records in results.values():
        for record in records.values():
            target = _validate_grid(
                _target_cxr_grid(
                    sweep,
                    record,
                    line_step_eV,
                    from_config,
                    cxr_start_eV,
                    line_stop_eV,
                ),
                "line grid",
            )
            if only_stale and not redo_all and _cxr_at_target(record, target, cxr_ne, fidelity):
                skipped += 1
                continue
            if (
                record.get("spec_channeling") is not None
                and record["case"].get("channeling") is None
            ):
                raise ValueError(
                    "cannot reline spec_channeling without its case channeling metadata"
                )
            updates = {}
            if cxr_ne is not None:
                updates["Ne"] = int(cxr_ne)
            if fidelity is not None:
                updates["line_profile"] = fidelity
            todo.append((record, target, updates))
    return todo, skipped
