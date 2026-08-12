"""Slice-E convergence matrices for energy-controlled electron transport.

Two independent questions, measured over a low-Z/high-Z x thin/thick x
5--300 keV case matrix.

Part A -- transport.  Does replacing the frozen (left-endpoint) propagation
rule with the slice-C midpoint rule move the physical observables?  The two
rules consume the same seed but diverge trajectory-by-trajectory, so the
comparison is statistical: exit fractions, path length, retained energy, and
transit clock are reported with their Monte Carlo standard errors, and the
frozen-minus-midpoint shift is quoted in units of that error.  The slice-B
per-flight diagnostics (fractional loss, clock error) give the step size that
drives any shift.

Part B -- radiation.  Fix the physical flights from one frozen run and refine
only the NUMERICAL sampling of the emission integral along each flight,
subdividing every flight into substeps of at most ``f`` fractional energy loss
for ``f`` = 2%, 1%, 0.5%.  The substep rows are fed to the production kernels
unchanged.  Substeps are integrated with the midpoint rule, and the reference
rung with it too, so the ladder measures the emission quadrature rather than
the clock error the phase table below reports separately.  What each column
means depends on the kernel's own coherence semantics:

- ``brem`` -- ``mc_brem_spectrum`` sums an independent path integral per row,
  so substep rows are a refined Riemann sum of the same per-flight integral.
  The ladder converges to the exact refined continuum.
- ``cxrC`` -- ``mc_spectrum(coherent=True)`` sums ONE global complex field over
  every row and squares at the end (``elec_id`` only masks rows at
  ``elec_id < Ne``; per-electron decoherence is emergent from the ``t0_ang``
  spread, not a grouping).  Substep rows therefore refine that global field
  exactly, so this ladder is the convergence of the coherent complex field.
- ``cxrCfz`` -- the same ladder integrated with the frozen rule against the same
  midpoint reference, so the pair isolates the propagation rule from the
  substep count.
- ``cxrI`` -- ``mc_spectrum(coherent=False)`` accumulates ``|A_j|^2 |Q_j|^2``
  per ROW.  Refining a flight into n substeps splits one emitter into n
  independent ones, each with a 1/n shorter coherent time and an n-times wider
  sinc: the reduction is NOT invariant, and this column is a CONTROL that
  measures the artificial decoherence, not a convergent quantity.  Its
  magnitude is the defect slice G has to remove; it is expected to stay flat
  or grow down the ladder while ``brem`` and ``cxrC`` fall.

The physical-flight-incoherent reduction (coherent within a flight, incoherent
across flights) is what slice G owes the kernels; ``--part c`` measures its
ladder exactly by calling the coherent kernel once per flight, which is
affordable only for one small case.

Convergence is the L1 and worst-bin change against a finer reference rung.

Part B also prints a PHASE CRITERION table, which is the operative slice-E
result: a coherent kernel weights each row by ``e^{i omega t_abs}``, so what
must converge is the absolute emission phase, and the fractional-loss ladder
does not control it.  The clock error accumulates over an electron's whole
history while ``|dE|/E`` is per-flight, and ages are 10^3--10^4 Ang, so under
the frozen rule the accumulated phase error is O(10--10^3 rad) at EVERY rung
and falls only first order in ``f``.  Under the midpoint rule it is 1e-3 to
0.2 rad at one row per flight.  The step control that matters for coherent CXR
is therefore the propagation rule, not the substep count.

Numbers from this script back
``docs/validation/beam-transport/energy-step-convergence.md`` and set the
accepted tolerances for slices F--H.

Validation: energy-step-convergence

Run:  uv run python checks/energy_step_convergence_matrix.py
      uv run python checks/energy_step_convergence_matrix.py --quick
      uv run python checks/energy_step_convergence_matrix.py --part c
      uv run python checks/energy_step_convergence_matrix.py --part d
"""

import argparse

import numpy as np

from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from pyrite.montecarlo.spectrum import mc_brem_spectrum, mc_spectrum
from pyrite.montecarlo.spectrum.diagnostics import (
    _flight_E_end_keV,
    _host,
    _stopping_keV_per_ang,
)
from pyrite.montecarlo.spectrum.lines import _observation_direction
from pyrite.montecarlo.transport import beta_from_keV, simulate_trajectories

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.06305)]

# (label, composition, E0_keV, thickness_ang).  Thicknesses bracket the CSDA
# range at each energy so every material sees a thin (mostly transmitting) and
# a thick (mostly stopping/backscattering) case.
CASES_A = [
    ("C    5 keV thin", CARBON, 5.0, 1.0e3),
    ("C    5 keV thick", CARBON, 5.0, 1.0e4),
    ("C   25 keV thin", CARBON, 25.0, 2.0e3),
    ("C   25 keV thick", CARBON, 25.0, 2.0e4),
    ("C  100 keV thin", CARBON, 100.0, 2.0e4),
    ("C  100 keV thick", CARBON, 100.0, 2.0e5),
    ("C  300 keV thick", CARBON, 300.0, 1.0e6),
    ("W    5 keV thin", TUNGSTEN, 5.0, 2.0e2),
    ("W    5 keV thick", TUNGSTEN, 5.0, 2.0e3),
    ("W   25 keV thin", TUNGSTEN, 25.0, 5.0e2),
    ("W   25 keV thick", TUNGSTEN, 25.0, 5.0e3),
    ("W  100 keV thin", TUNGSTEN, 100.0, 5.0e3),
    ("W  100 keV thick", TUNGSTEN, 100.0, 5.0e4),
    ("W  300 keV thick", TUNGSTEN, 300.0, 2.0e5),
]

# Radiation ladder runs the same geometry but fewer electrons: the measured
# quantity is a deterministic quadrature error at fixed flights, not a sampling
# error.  CXR needs the crystal to BE the transported material, so the hopg
# columns are carbon-only; tungsten contributes bremsstrahlung alone.
CASES_B = [
    ("C   25 keV thin", CARBON, 25.0, 2.0e3),
    ("C   25 keV thick", CARBON, 25.0, 2.0e4),
    ("C  100 keV thick", CARBON, 100.0, 2.0e5),
    ("W   25 keV thick", TUNGSTEN, 25.0, 5.0e3),
    ("W  100 keV thick", TUNGSTEN, 100.0, 5.0e4),
]
# Exact per-flight-incoherent ladder: one coherent kernel call per physical
# flight, so it is restricted to the smallest case and its own electron count.
CASE_C = ("C   25 keV thin", CARBON, 25.0, 2.0e3)
NE_C = 60

# Ladder rungs: None is the current production rule (one row per flight).
LADDER = [None, 2.0e-2, 1.0e-2, 5.0e-3]
LADDER_REFERENCE = 1.25e-3

BREM_GRID = np.linspace(1.0e3, 3.0e4, 501)
# hopg (0,0,2) at the kernel's DEFAULT 119 deg take-off, not the near-grazing
# ``n_hat=(1,0,0.01)`` of tests/montecarlo/test_coherent_emission.py.  At
# n_z = 0.01 the Beer--Lambert escape path is 100x the depth, so a flight's own
# escape factor swings by ~2x between its endpoints and merely SPLITTING rows
# (at frozen energy, where the coherent sum is an exact identity) moves the
# spectrum ~30%.  That geometry cannot resolve an energy-step effect; it is
# reported separately by ``--part d``.
CXR_THETA_OBS_RAD = np.deg2rad(119.0)
CXR_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "theta_obs_rad": CXR_THETA_OBS_RAD,
}
CXR_GRAZING_N_HAT = np.array([1.0, 0.0, 0.01])
# The line window is built per case: the resonance rides beta, so a window
# fixed for 25 keV would only catch the sinc tail at 100 keV.
CXR_HALF_WINDOW_EV = 400.0

NE_A = 4000
NE_B = 200
SEED = 7
# Below the 5 keV default so the 5 keV cases have a live cutoff, and low enough
# that soft-X-ray bremsstrahlung is not truncated at the source.
E_CUT_KEV = 1.0

# Substep energies stay above this floor: the Joy--Luo stopping law diverges as
# 1/E and the transport cutoff already keeps flight starts at or above E_cut.
_E_FLOOR_KEV = 0.1
# Bounds the refinement loop when a single cutoff-stopped flight loses most of
# its energy; reported when it binds so a capped rung is never read as converged.
_MAX_SUBSTEPS = 512


def _cxr_resonance_eV(E0_keV, g_vec, n_hat=None):
    """Forward-beam resonance energy for this beam energy and take-off [eV].

    ``E_res = hbar_c * (beta v.g) / (1 - beta v.n)`` at the incident direction
    +z, the kernel's own resonance condition evaluated before any scattering.
    """
    unit = _observation_direction(CXR_THETA_OBS_RAD, n_hat)
    v_hat = np.array([0.0, 0.0, 1.0])
    beta = float(beta_from_keV(np.asarray(E0_keV, dtype=float)))
    return HBARC_EV_ANG * beta * float(v_hat @ g_vec) / (1.0 - beta * float(v_hat @ unit))


def _cxr_grid(E0_keV, g_vec, n_hat=None):
    """1 eV grid centred on the forward-beam resonance for this beam energy."""
    E_res = _cxr_resonance_eV(E0_keV, g_vec, n_hat)
    return np.arange(max(1.0, E_res - CXR_HALF_WINDOW_EV), E_res + CXR_HALF_WINDOW_EV, 1.0)


def _advance(E_keV, walk_ang, composition, rule):
    """One substep of the CSDA propagation: ``(E_end, representative clock E)``.

    ``"frozen"`` is the left-endpoint rule.  ``"midpoint"`` mirrors the lockstep
    core: a predictor-corrector for the implicit midpoint rule
    ``E_end = E + (dE/ds)((E + E_end)/2) s``, with the same representative
    energy reused for ``int ds / beta(E(s)) = s / beta(E_mid)``.
    """
    if rule == "midpoint":
        E_pred = E_keV + _stopping_keV_per_ang(E_keV, composition) * walk_ang
        E_end = np.maximum(
            E_keV
            + walk_ang
            * _stopping_keV_per_ang(np.maximum(0.5 * (E_keV + E_pred), _E_FLOOR_KEV), composition),
            _E_FLOOR_KEV,
        )
        return E_end, np.maximum(0.5 * (E_keV + E_end), _E_FLOOR_KEV)
    if rule == "frozen":
        E_end = np.maximum(
            E_keV + _stopping_keV_per_ang(E_keV, composition) * walk_ang, _E_FLOOR_KEV
        )
        return E_end, E_keV
    raise ValueError(f"unknown rule {rule!r}")


def _flight_end_clock(rows, parent, composition, rule):
    """Per-flight end age under a rung's substep integration [Ang, c=1].

    ``rows["t_ang"]`` already carries the rule's accumulated start times, so
    only the final substep's own leg has to be added -- with that rule's
    representative clock energy, not the left endpoint.
    """
    last = np.nonzero(np.diff(parent, append=-1) != 0)[0]
    E_last = rows["E_keV"][last]
    walk = rows["L_ang"][last]
    _, clock_E = _advance(E_last, walk, composition, rule)
    return rows["t_ang"][last] + walk / beta_from_keV(clock_E)


def _accumulate_per_electron(delta, elec_id, t_ang):
    """Running sum of a per-flight quantity along each electron's own history.

    Every rung re-anchors each flight's clock to the parent transport run's
    ``t_ang``, so a rung's per-flight error does not accumulate on its own.
    The emission phase depends on the ABSOLUTE age, so the physically binding
    quantity is this cumulative sum along the trajectory.
    """
    order = np.lexsort((t_ang, elec_id))
    run = np.cumsum(delta[order])
    starts = np.searchsorted(elec_id[order], np.unique(elec_id), side="left")
    before = np.where(starts > 0, run[np.maximum(starts - 1, 0)], 0.0)
    base = np.repeat(before, np.diff(np.append(starts, run.size)))
    out = np.empty_like(run)
    out[order] = run - base
    return out


def _rel_l1(a, b):
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(b)))


def _rel_max_bin(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _last_row_per_electron(elec_id, t_ang):
    """Row indices of each electron's final flight, in ascending electron order."""
    order = np.lexsort((t_ang, elec_id))
    sorted_id = elec_id[order]
    ends = np.searchsorted(sorted_id, np.unique(sorted_id), side="right") - 1
    return order[ends]


def _transport_summary(segments, composition, E0_keV):
    """Physical observables plus their Monte Carlo standard errors."""
    Ne = int(segments["Ne"])
    elec_id = _host(segments["elec_id"]).astype(np.int64, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    t_start = _host(segments["t_ang"]).astype(float, copy=False)
    E_end = _flight_E_end_keV(segments, composition, None)
    t_end = segments.get("t_end_ang")
    if t_end is None:
        E_start = _host(segments["E_keV"]).astype(float, copy=False)
        t_end = t_start + length / beta_from_keV(E_start)
    else:
        t_end = _host(t_end).astype(float, copy=False)

    # Per-electron path length; electrons that missed the target contribute a
    # zero path and no final row, so both means stay over the full Ne.
    path = np.bincount(elec_id, weights=length, minlength=Ne)
    last = _last_row_per_electron(elec_id, t_start)
    retained = np.full(Ne, float(E0_keV))
    retained[elec_id[last]] = E_end[last]
    clock = np.zeros(Ne)
    clock[elec_id[last]] = t_end[last]

    summary = {}
    for key, count in (
        ("trans", segments["n_transmitted"]),
        ("back", segments["n_backscattered"]),
        ("side", segments["n_side_exited"]),
        ("stop", segments["n_cutoff_stopped"]),
    ):
        p = int(count) / Ne
        summary[key] = (p, float(np.sqrt(max(p * (1.0 - p), 0.0) / Ne)))
    for key, values in (("path", path), ("E_ret", retained), ("clock", clock)):
        summary[key] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(Ne)))
    return summary


def _subdivide(segments, composition, max_frac_loss, refine="both", rule="frozen"):
    """Split each physical flight into substeps of <= f fractional energy loss.

    ``rule`` selects the integrator applied along the substep chain: ``"frozen"``
    is the left-endpoint rule and ``"midpoint"`` mirrors the lockstep core's
    predictor-corrector, so the two can be compared on identical flights without
    the trajectory divergence that separates two full transport runs.

    Geometry is exact: the flight is a straight constant-direction ray, so a
    substep's midpoint and length follow from the parent's without any
    transport state.  Only the energy and clock sampling refines, which is the
    quadrature error under test.  ``max_frac_loss=None`` returns one row per
    flight, i.e. the current production input.

    Three refinement channels, so a coherent-field change can be attributed:

    - ``"both"`` -- energy and clock both refine.  This is the real ladder.
    - ``"energy"`` -- the emission energy refines, but each substep is placed
      on the PARENT flight's frozen constant-velocity parameterization, so the
      flight's total duration and every emission phase are unchanged.
    - ``"geometry"`` -- energy AND clock frozen at the parent's, i.e. pure row
      splitting with no physics change.  At constant velocity and amplitude the
      coherent sum of substeps is algebraically identical to the parent row
      (the Dirichlet-kernel sum ``sum_k e^{2iP s_k}`` times the substep
      ``sinc`` rebuilds the parent ``t_L sinc(P t_L / pi)``), so any residual
      here is the kernel's own per-row approximations -- chiefly the single
      escape factor taken at the row midpoint -- and is the FLOOR below which
      no energy-step number in the same column is meaningful.

    Each substep's time is set so the kernel's own midpoint convention
    (``t_abs = t_ang + L/(2 beta) + t0``) lands on the intended midpoint,
    keeping the emission time consistent with ``r_mid`` in every mode.
    """
    E_start = _host(segments["E_keV"]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    v_hat = _host(segments["v_hat"]).astype(float, copy=False)
    r_mid = _host(segments["r_mid"]).astype(float, copy=False)
    t_start = _host(segments["t_ang"]).astype(float, copy=False)
    n_rows = E_start.size

    if max_frac_loss is None:
        n_sub = np.ones(n_rows, dtype=np.int64)
        capped = 0
    else:
        loss = np.maximum(0.0, -_stopping_keV_per_ang(E_start, composition) * length / E_start)
        exact = np.maximum(1.0, np.ceil(loss / max_frac_loss))
        capped = int(np.count_nonzero(exact > _MAX_SUBSTEPS))
        n_sub = np.minimum(exact, _MAX_SUBSTEPS).astype(np.int64)

    offsets = np.concatenate(([0], np.cumsum(n_sub)[:-1]))
    total = int(n_sub.sum())
    parent = np.repeat(np.arange(n_rows), n_sub)
    within = np.arange(total) - np.repeat(offsets, n_sub)

    step_len = length / n_sub
    r_entry = r_mid - 0.5 * length[:, None] * v_hat
    sub_r_mid = r_entry[parent] + v_hat[parent] * ((within + 0.5) * step_len[parent])[:, None]

    sub_E = np.empty(total)
    sub_t = np.empty(total)
    E_cur = E_start.copy()
    t_cur = t_start.copy()
    for k in range(int(n_sub.max()) if n_rows else 0):
        active = np.nonzero(n_sub > k)[0]
        slot = offsets[active] + k
        E_here = E_cur[active]
        walk = step_len[active]
        sub_E[slot] = E_here
        sub_t[slot] = t_cur[active]
        E_next, clock_E = _advance(E_here, walk, composition, rule)
        t_cur[active] += walk / beta_from_keV(clock_E)
        E_cur[active] = E_next
    if refine in ("energy", "geometry"):
        if refine == "geometry":
            sub_E = E_start[parent]
        parent_beta = beta_from_keV(E_start[parent])
        half = 0.5 * step_len[parent]
        sub_t = (
            t_start[parent]
            + (within + 0.5) * step_len[parent] / parent_beta
            - half / beta_from_keV(sub_E)
        )
    elif refine != "both":
        raise ValueError(f"unknown refine mode {refine!r}")

    out = {
        key: value
        for key, value in segments.items()
        # Per-row arrays are rebuilt below; the end-state and alias fields are
        # dropped rather than left at the parent length, because every consumer
        # loops the owning registry with a presence guard and a stale array of
        # the wrong length silently desynchronizes under any row mask.
        if key
        not in (
            "r_mid",
            "v_hat",
            "L_ang",
            "E_keV",
            "E_start_keV",
            "E_end_keV",
            "t_ang",
            "t_start_ang",
            "t_end_ang",
            "t0_ang",
            "elec_id",
            "layer",
        )
    }
    out.update(
        {
            "r_mid": sub_r_mid,
            "v_hat": v_hat[parent],
            "L_ang": step_len[parent],
            "E_keV": sub_E,
            "E_start_keV": sub_E,
            "t_ang": sub_t,
            "t_start_ang": sub_t,
            "t0_ang": _host(segments["t0_ang"]).astype(float, copy=False)[parent],
            "elec_id": _host(segments["elec_id"]).astype(np.int64, copy=False)[parent],
            "layer": _host(segments["layer"]).astype(np.int64, copy=False)[parent],
        }
    )
    return out, parent, capped


def _radiation_reductions(rows, composition, cxr_grid, only=None):
    """Production-kernel reductions for one ladder rung.

    ``cxr_grid=None`` skips the CXR columns (the transported material is not
    the emitting crystal).  ``only`` restricts the reductions actually computed,
    which matters because each is a full kernel pass over every row.
    """
    wanted = {"brem", "cxr_coh", "cxr_incoh"} if only is None else set(only)
    values = {}
    with np.errstate(all="ignore"):
        if "brem" in wanted:
            values["brem"] = mc_brem_spectrum(rows, BREM_GRID, composition=composition)
        if cxr_grid is not None and "cxr_coh" in wanted:
            values["cxr_coh"] = mc_spectrum(
                rows, cxr_grid, composition=composition, coherent=True, **CXR_KWARGS
            )
        if cxr_grid is not None and "cxr_incoh" in wanted:
            values["cxr_incoh"] = mc_spectrum(rows, cxr_grid, composition=composition, **CXR_KWARGS)
    return values


def _flight_incoherent_cxr(rows, parent, composition, cxr_grid, Ne):
    """Coherent within a physical flight, incoherent across flights.

    The production kernel has no per-flight grouping, so this sums one kernel
    call per flight.  Each call is normalized by its own ``Ne``, so the single
    flight's field is rescaled back to an absolute yield before the incoherent
    sum and the true per-electron normalization is applied once at the end.
    """
    total = np.zeros(cxr_grid.size)
    for flight in np.unique(parent):
        mask = parent == flight
        one = {
            key: (
                value[mask]
                if isinstance(value, np.ndarray) and value.shape[:1] == parent.shape
                else value
            )
            for key, value in rows.items()
        }
        one["Ne"] = 1
        with np.errstate(all="ignore"):
            total += mc_spectrum(
                one, cxr_grid, composition=composition, coherent=True, **CXR_KWARGS
            )
    return total / Ne


_A_WIDTHS = (
    ("case", 17),
    ("model", 9),
    ("trans", 16),
    ("back", 16),
    ("side", 16),
    ("stop", 16),
    ("path/e", 12),
    ("E_ret", 10),
    ("clock", 12),
    ("loss p99", 9),
    ("clk p99", 9),
)
_B_WIDTHS = (
    ("case", 17),
    ("f", 8),
    ("rows", 9),
    ("brem L1", 10),
    ("brem max", 10),
    ("cxrC L1", 10),
    ("cxrC max", 10),
    ("cxrCfz L1", 10),
    ("cxrCfz max", 10),
    ("cxrI L1", 10),
    ("cxrI max", 10),
)
_P_WIDTHS = (
    ("case", 17),
    ("rule", 9),
    ("f", 8),
    ("dt p50", 10),
    ("dt p99", 10),
    ("cum dt p99", 11),
    ("cum dt max", 11),
    ("dphi p99", 10),
    ("cum dphi p99", 13),
    ("cum dphi max", 13),
)
_C_WIDTHS = (
    ("case", 17),
    ("f", 8),
    ("rows", 9),
    ("flights", 9),
    ("cxrF L1", 10),
    ("cxrF max", 10),
    ("cxrC L1", 10),
    ("cxrC max", 10),
)


def _fmt_pm(pair, scale=1.0, digits=4):
    value, error = pair
    return f"{value * scale:.{digits}f}+-{error * scale:.{digits}f}"


def _row(cols, widths):
    """Fixed-width row; downstream markdown injection slices by these widths."""
    for key, w in widths:
        if len(str(cols[key])) > w:
            raise ValueError(f"cell {key}={cols[key]!r} overflows width {w}")
    return " ".join(f"{cols[key]:>{w}}" for key, w in widths)


def part_a(Ne):
    print("=" * 118)
    print(f"Part A -- transport observables, frozen vs midpoint (Ne={Ne}, seed={SEED})")
    print("Exit fractions and means carry 1-sigma Monte Carlo errors; 'shift' rows")
    print("give (frozen - midpoint) in units of the combined error.")
    print("=" * 118)
    print(" ".join(f"{key:>{w}}" for key, w in _A_WIDTHS))
    for name, comp, E0, thickness in CASES_A:
        summaries = {}
        for model in ("frozen", "midpoint"):
            segments = simulate_trajectories(
                E0_keV=E0,
                Ne=Ne,
                thickness_ang=thickness,
                composition=comp,
                E_cut_keV=E_CUT_KEV,
                seed=SEED,
                transport_core="lockstep",
                energy_model=model,
                collect_diagnostics=True,
            )
            summary = _transport_summary(segments, comp, E0)
            diag = segments["transport_diagnostics"]
            summaries[model] = summary
            cols = {
                "case": name if model == "frozen" else "",
                "model": model,
                "trans": _fmt_pm(summary["trans"]),
                "back": _fmt_pm(summary["back"]),
                "side": _fmt_pm(summary["side"]),
                "stop": _fmt_pm(summary["stop"]),
                "path/e": f"{summary['path'][0]:.4g}",
                "E_ret": f"{summary['E_ret'][0]:.4g}",
                "clock": f"{summary['clock'][0]:.4g}",
                "loss p99": f"{diag['fractional_energy_loss']['p99']:.2e}",
                "clk p99": f"{diag['relative_clock_error_estimate']['p99']:.2e}",
            }
            print(_row(cols, _A_WIDTHS))

        shift = {"case": "", "model": "shift/sig"}
        for key in ("trans", "back", "side", "stop", "path", "E_ret", "clock"):
            (value_f, err_f), (value_m, err_m) = (
                summaries["frozen"][key],
                summaries["midpoint"][key],
            )
            sigma = np.hypot(err_f, err_m)
            text = "--" if sigma == 0.0 else f"{(value_f - value_m) / sigma:+.1f}"
            shift["path/e" if key == "path" else key] = text
        shift["loss p99"] = ""
        shift["clk p99"] = ""
        print(_row(shift, _A_WIDTHS))


def part_b(Ne):
    g002, g002_mag = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    print()
    print("=" * 130)
    print(f"Part B -- radiation refinement ladder at fixed physical flights (Ne={Ne}, seed={SEED})")
    print(
        f"hopg (0,0,2) |g| = {g002_mag:.4f} 1/Ang at {np.rad2deg(CXR_THETA_OBS_RAD):.0f} deg;"
        f" reference rung f={LADDER_REFERENCE:.3%}, midpoint rule"
    )
    print("f='flight' is the current production rule (one emission row per physical flight).")
    print("Substeps are integrated with the MIDPOINT rule, so the reference is not itself")
    print("mis-phased and the ladder measures the emission quadrature rather than the clock.")
    print("brem   = mc_brem_spectrum (incoherent, phase-free)")
    print("cxrC   = coherent CXR, midpoint substeps -- the real convergence question")
    print("cxrCfz = the SAME ladder integrated with the frozen rule, against the same")
    print("         midpoint reference, so cxrC vs cxrCfz isolates the propagation rule")
    print("cxrI   = row-incoherent CXR: a CONTROL. Row splitting is not invariant for that")
    print("         reduction, so it measures artificial decoherence, not convergence.")
    print("'floor' is the pure row-splitting residual at the reference rung (frozen energy")
    print("and clock), below which no number in the same column is meaningful.")
    print("=" * 130)
    print(" ".join(f"{key:>{w}}" for key, w in _B_WIDTHS))
    for name, comp, E0, thickness in CASES_B:
        segments = simulate_trajectories(
            E0_keV=E0,
            Ne=Ne,
            thickness_ang=thickness,
            composition=comp,
            E_cut_keV=E_CUT_KEV,
            seed=SEED,
            transport_core="lockstep",
        )
        cxr_grid = _cxr_grid(E0, g002) if comp is CARBON else None
        ref_rows, _, ref_capped = _subdivide(segments, comp, LADDER_REFERENCE, rule="midpoint")
        reference = _radiation_reductions(ref_rows, comp, cxr_grid)
        if ref_capped:
            print(f"  ! reference rung capped {ref_capped} flight(s) at {_MAX_SUBSTEPS} substeps")
        split_rows, _, _ = _subdivide(segments, comp, LADDER_REFERENCE, refine="geometry")
        split = _radiation_reductions(split_rows, comp, cxr_grid)
        base_rows, _, _ = _subdivide(segments, comp, None)
        base = _radiation_reductions(base_rows, comp, cxr_grid)

        for max_loss in LADDER:
            rows, _, capped = _subdivide(segments, comp, max_loss, rule="midpoint")
            values = _radiation_reductions(rows, comp, cxr_grid)
            frozen_rows, _, _ = _subdivide(segments, comp, max_loss, rule="frozen")
            frozen_values = _radiation_reductions(frozen_rows, comp, cxr_grid, only=("cxr_coh",))
            cols = {
                "case": name if max_loss is LADDER[0] else "",
                "f": "flight" if max_loss is None else f"{max_loss:.3%}",
                "rows": str(rows["L_ang"].size),
            }
            for tag, key, got in (
                ("brem", "brem", values),
                ("cxrC", "cxr_coh", values),
                ("cxrCfz", "cxr_coh", frozen_values),
                ("cxrI", "cxr_incoh", values),
            ):
                if key in got:
                    cols[f"{tag} L1"] = f"{_rel_l1(got[key], reference[key]):.2e}"
                    cols[f"{tag} max"] = f"{_rel_max_bin(got[key], reference[key]):.2e}"
                else:
                    cols[f"{tag} L1"] = cols[f"{tag} max"] = "--"
            print(_row(cols, _B_WIDTHS))
            if capped:
                print(f"  ! rung capped {capped} flight(s) at {_MAX_SUBSTEPS} substeps")

        floor = {"case": "", "f": "floor", "rows": ""}
        for tag, key in (
            ("brem", "brem"),
            ("cxrC", "cxr_coh"),
            ("cxrCfz", "cxr_coh"),
            ("cxrI", "cxr_incoh"),
        ):
            if key in base:
                floor[f"{tag} L1"] = f"{_rel_l1(split[key], base[key]):.2e}"
                floor[f"{tag} max"] = f"{_rel_max_bin(split[key], base[key]):.2e}"
            else:
                floor[f"{tag} L1"] = floor[f"{tag} max"] = "--"
        print(_row(floor, _B_WIDTHS))


def part_b_phase(Ne):
    """The emission-phase criterion, which the fractional-loss ladder is not.

    A coherent kernel weights each row by ``e^{i omega t_abs}``, so what has to
    converge is the ABSOLUTE emission phase, not the fractional energy loss.
    The two are only loosely related: ``|dE|/E`` is a per-flight quantity while
    the phase error accumulates over an electron's whole history, and the age
    itself is large (10^3--10^4 Ang), so a relative clock error of 1e-3 is
    already O(1) radian at a 1 keV line.

    Both propagation rules are run over the SAME parent flights, so this is the
    quadrature error of each rule with trajectory divergence removed.
    """
    g002, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    print()
    print("=" * 132)
    print(
        f"Part B (phase criterion) -- clock error vs the midpoint rule at"
        f" f={LADDER_REFERENCE:.3%} (Ne={Ne}, seed={SEED})"
    )
    print("dt   = per-flight end-age error [Ang, c=1];  cum dt = same, accumulated over")
    print("       each electron's flights, which is what an absolute emission phase sees.")
    print("dphi = cum dt * E_res / hbar_c [rad] at the hopg (0,0,2) resonance for the beam")
    print("       energy.  A rung is usable for coherent CXR only where dphi << 1.")
    print("rule = the integrator along the substep chain; 'flight' is one step per flight,")
    print("       i.e. exactly what simulate_trajectories emits under that energy_model.")
    print("=" * 132)
    print(" ".join(f"{key:>{w}}" for key, w in _P_WIDTHS))
    for name, comp, E0, thickness in CASES_B:
        segments = simulate_trajectories(
            E0_keV=E0,
            Ne=Ne,
            thickness_ang=thickness,
            composition=comp,
            E_cut_keV=E_CUT_KEV,
            seed=SEED,
            transport_core="lockstep",
        )
        elec_id = _host(segments["elec_id"]).astype(np.int64, copy=False)
        t_ang = _host(segments["t_ang"]).astype(float, copy=False)
        omega = _cxr_resonance_eV(E0, g002) / HBARC_EV_ANG
        ref_rows, ref_parent, _ = _subdivide(segments, comp, LADDER_REFERENCE, rule="midpoint")
        reference = _flight_end_clock(ref_rows, ref_parent, comp, "midpoint")
        first = True
        for rule in ("frozen", "midpoint"):
            for max_loss in LADDER:
                rows, parent, _ = _subdivide(segments, comp, max_loss, rule=rule)
                delta = _flight_end_clock(rows, parent, comp, rule) - reference
                cumulative = np.abs(_accumulate_per_electron(delta, elec_id, t_ang))
                flight = np.abs(delta)
                cols = {
                    "case": name if first else "",
                    "rule": rule if max_loss is LADDER[0] else "",
                    "f": "flight" if max_loss is None else f"{max_loss:.3%}",
                    "dt p50": f"{np.percentile(flight, 50):.2e}",
                    "dt p99": f"{np.percentile(flight, 99):.2e}",
                    "cum dt p99": f"{np.percentile(cumulative, 99):.2e}",
                    "cum dt max": f"{cumulative.max():.2e}",
                    "dphi p99": f"{omega * np.percentile(flight, 99):.2e}",
                    "cum dphi p99": f"{omega * np.percentile(cumulative, 99):.2e}",
                    "cum dphi max": f"{omega * cumulative.max():.2e}",
                }
                print(_row(cols, _P_WIDTHS))
                first = False


def part_c(Ne):
    """Exact physical-flight-incoherent CXR ladder: slice G's target reduction."""
    name, comp, E0, thickness = CASE_C
    g002, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    cxr_grid = _cxr_grid(E0, g002)
    print()
    print("=" * 118)
    print(f"Part C -- physical-flight-incoherent CXR ladder (Ne={Ne}, seed={SEED})")
    print("cxrF = coherent WITHIN each physical flight, incoherent across flights: one")
    print("       coherent kernel call per flight, so a single small case only.  This is")
    print("       the reduction slice G owes the kernels.")
    print("cxrC = the production global-coherent reduction on the SAME rows, for contrast.")
    print(f"Midpoint substeps; reference rung f={LADDER_REFERENCE:.3%}.")
    print("=" * 118)
    print(" ".join(f"{key:>{w}}" for key, w in _C_WIDTHS))
    segments = simulate_trajectories(
        E0_keV=E0,
        Ne=Ne,
        thickness_ang=thickness,
        composition=comp,
        E_cut_keV=E_CUT_KEV,
        seed=SEED,
        transport_core="lockstep",
    )
    ref_rows, ref_parent, _ = _subdivide(segments, comp, LADDER_REFERENCE, rule="midpoint")
    reference = _flight_incoherent_cxr(ref_rows, ref_parent, comp, cxr_grid, Ne)
    global_reference = _radiation_reductions(ref_rows, comp, cxr_grid, only=("cxr_coh",))["cxr_coh"]
    for max_loss in LADDER:
        rows, parent, _ = _subdivide(segments, comp, max_loss, rule="midpoint")
        value = _flight_incoherent_cxr(rows, parent, comp, cxr_grid, Ne)
        globally = _radiation_reductions(rows, comp, cxr_grid, only=("cxr_coh",))["cxr_coh"]
        cols = {
            "case": name if max_loss is LADDER[0] else "",
            "f": "flight" if max_loss is None else f"{max_loss:.3%}",
            "rows": str(rows["L_ang"].size),
            "flights": str(np.unique(parent).size),
            "cxrF L1": f"{_rel_l1(value, reference):.2e}",
            "cxrF max": f"{_rel_max_bin(value, reference):.2e}",
            "cxrC L1": f"{_rel_l1(globally, global_reference):.2e}",
            "cxrC max": f"{_rel_max_bin(globally, global_reference):.2e}",
        }
        print(_row(cols, _C_WIDTHS))


def part_d(Ne):
    """Row-splitting floor versus take-off angle -- an escape-factor result.

    Splitting a flight into substeps at FROZEN energy and clock is an exact
    algebraic identity for the coherent sum, so whatever the spectrum does is
    the kernel's own per-row approximation.  The dominant one is the single
    Beer--Lambert escape factor evaluated at the row midpoint: at a take-off
    angle with small ``n_z`` the escape path is the depth divided by ``n_z``,
    so it varies enormously across one flight and the flight-level row is a
    poor quadrature of it.  This is independent of ``energy_model`` and of the
    slice-E step control; it bounds what any energy-step ladder can resolve in
    the same geometry.
    """
    name, comp, E0, thickness = CASE_C
    g002, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    print()
    print("=" * 130)
    print(f"Part D -- pure row-splitting residual vs take-off geometry (Ne={Ne}, seed={SEED})")
    print(f"{name}; frozen energy AND clock, so the exact coherent answer is unchanged.")
    print("Any nonzero entry is the per-row escape-factor approximation, not an energy step.")
    print("=" * 130)
    segments = simulate_trajectories(
        E0_keV=E0,
        Ne=Ne,
        thickness_ang=thickness,
        composition=comp,
        E_cut_keV=E_CUT_KEV,
        seed=SEED,
        transport_core="lockstep",
    )
    header = ("geometry", 26), ("n_z", 9), ("E_res eV", 10), ("f=1% L1", 10), ("f=0.125% L1", 12)
    print(" ".join(f"{key:>{w}}" for key, w in header))
    for label, n_hat in (
        (f"default {np.rad2deg(CXR_THETA_OBS_RAD):.0f} deg", None),
        ("grazing (1, 0, 0.01)", CXR_GRAZING_N_HAT),
    ):
        kwargs = dict(CXR_KWARGS)
        if n_hat is not None:
            kwargs["n_hat"] = n_hat
        unit = _observation_direction(CXR_THETA_OBS_RAD, n_hat)
        grid = _cxr_grid(E0, g002, n_hat)
        base_rows, _, _ = _subdivide(segments, comp, None)
        with np.errstate(all="ignore"):
            base = mc_spectrum(base_rows, grid, composition=comp, coherent=True, **kwargs)
            split = [
                mc_spectrum(
                    _subdivide(segments, comp, f, refine="geometry")[0],
                    grid,
                    composition=comp,
                    coherent=True,
                    **kwargs,
                )
                for f in (1.0e-2, LADDER_REFERENCE)
            ]
        cols = {
            "geometry": label,
            "n_z": f"{unit[2]:+.3f}",
            "E_res eV": f"{grid.mean():.0f}",
            "f=1% L1": f"{_rel_l1(split[0], base):.2e}",
            "f=0.125% L1": f"{_rel_l1(split[1], base):.2e}",
        }
        print(" ".join(f"{cols[key]:>{w}}" for key, w in header))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small electron counts")
    parser.add_argument("--part", choices=("a", "b", "c", "d", "both"), default="both")
    args = parser.parse_args()
    Ne_a, Ne_b, Ne_c = (200, 25, 6) if args.quick else (NE_A, NE_B, NE_C)
    if args.part in ("a", "both"):
        part_a(Ne_a)
    if args.part in ("b", "both"):
        part_b(Ne_b)
        part_b_phase(Ne_b)
    if args.part == "c":
        part_c(Ne_c)
    if args.part == "d":
        part_d(NE_B if not args.quick else 25)


if __name__ == "__main__":
    main()
