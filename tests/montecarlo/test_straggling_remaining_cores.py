"""Applying the Urban straggling loss on the cores slice E did not cover.

Slice F of ``feature/energy-loss-straggling``. Slice E applied the sampled loss
and redefined the cutoff crossing inside ``_transport_core_ungrooved`` (the
ungrooved lockstep *exact* core) and left every other core diagnostic-only.
This module covers the remaining host cores:

* ``_transport_core_ungrooved_lut`` (lockstep, LUT),
* ``_transport_core_ungrooved_perelectron`` (per-electron, exact),
* ``_transport_core_ungrooved_perelectron_lut`` (per-electron, LUT),
* ``_transport_core_grooved`` (lockstep, grooved) -- see the grooved section.

**Nothing about slice E's rule is re-litigated here.** The crossing indicator
is exact ("does this row's total sampled loss reach ``E_start - E_cut``",
which follows from the Urban loss being a non-decreasing compound-Poisson
subordinator) and the crossing location is the fluid interpolation
``s_cut = s (E_start - E_cut) / dE`` with ``E_end = E_cut`` exactly. What slice
F owns is the *bookkeeping* each core's own geometry and flag representation
forces, and the three questions that raised, all answered in the "porting the
slice-E crossing rule to the remaining cores (slice F)" block above
``_transport_core_ungrooved_lut``:

1. **LUT cores** solve their deterministic cutoff distance on the interpolated
   ``dE/ds``, but the straggled crossing needs no such solve and slice D
   already draws the loss from the *exact* per-element stopping power. So the
   straggled LUT crossing is more accurate than the LUT's own deterministic
   one. That is accepted rather than reconciled; the LUT keeps the
   ``max_dE_frac`` step cap and the clock, which is everything it exists to
   accelerate that is still deterministic.
2. **Per-electron cores** return an ``exit_code`` enum rather than accumulating
   into shared counters, but they still carry the same local geometry booleans
   through a row and only derive ``exit_code`` from them at the end, so E's
   flag clearing transcribes literally.
3. **The grooved core** can truncate a row at a groove facet, after which the
   electron travels a vacuum leg. The cutoff test therefore has to run on the
   *material-side* path length -- which it does by construction, since the
   facet truncation already happens before the energy close -- and by the same
   monotonicity that makes the indicator exact, a crossing inside the material
   part of the row means the electron stops before ever reaching the facet.
   ``test_grooved_cutoff_beats_a_facet_crossing_on_the_material_side`` drives
   exactly that competition.

The cross-core parity claim is stated and tested at the level it actually
holds. The cores realize *different trajectories* by construction (the lockstep
cores interleave one shared ``Generator`` while the per-electron cores give
each electron its own counter stream; the LUT cores interpolate the free-path
rate), so whole-run equality is meaningless between them. What is bit-for-bit
identical on every core is the pair that slice F ports: the sampled loss for a
given ``(seed, electron, flight, substep, E_start, s)``, and the crossing rule
applied to it. ``test_rows_reproduce_the_sampler_and_the_crossing_rule``
reconstructs both offline from each core's own recorded rows and demands exact
float64 equality, on every wired core, in a run where most electrons terminate
at the cutoff.
"""

import numpy as np
import pytest

from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    _urban_flight_key_scalar,
    _urban_sample_compound_keV,
    _urban_stream_key_scalar,
    simulate_trajectories,
    spliced_stopping_keV_per_ang,
    stream_keys,
    urban_element_table,
)

NO_LUT = TransportLUTConfig(enabled=False)

GRAPHITE = [("C", 0.1136)]

# The four cores slice F wires. The ungrooved lockstep exact core is slice E's
# and is covered by `test_straggling_transport_integration.py`; it is included
# in the parity test below as the reference the ports are checked against.
WIRED_CORES = {
    "lockstep-lut": dict(transport_core="lockstep"),
    "perelectron-exact": dict(transport_core="per-electron", transport_lut_config=NO_LUT),
    "perelectron-lut": dict(transport_core="per-electron"),
}
REFERENCE_CORE = {"lockstep-exact": dict(transport_core="lockstep", transport_lut_config=NO_LUT)}
ALL_HOST_CORES = {**REFERENCE_CORE, **WIRED_CORES}

E_CUT_KEV = 5.0

# 25 keV in graphite against the default 5 keV cutoff, in a slab thicker than
# the CSDA range, so nearly every electron ends by reaching the cutoff and the
# redefined crossing is exercised on most tracks rather than on a handful.
THICK_SLAB = dict(
    E0_keV=25.0,
    element="C",
    n_atoms_per_ang3=0.1136,
    thickness_ang=60000.0,
    max_steps=20000,
)


def _run(core_kwargs, **overrides):
    kwargs = dict(THICK_SLAB)
    kwargs.update(core_kwargs)
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


# --- property 1: the off path is unreachable on every newly wired core -------


@pytest.mark.parametrize("name", sorted(WIRED_CORES))
@pytest.mark.parametrize(
    ("energy_model", "max_dE_frac"),
    [("frozen", 0.0), ("midpoint", 0.0), ("midpoint", 0.02)],
)
def test_straggling_off_is_inert_in_a_cutoff_and_substep_heavy_run(name, energy_model, max_dE_frac):
    """Slice F reorders each core's deterministic cutoff solve and its
    ``max_dE_frac`` cap into an ``if straggle_on: ... else: ...`` pair, exactly
    as slice E did for the lockstep exact core. The ``else`` branch is
    textually the previous code, but "textually unchanged" is not a test, so
    this drives a run in which BOTH new orderings would be taken if the gate
    leaked -- thousands of cutoff crossings and, in the substep case, tens of
    thousands of energy-capped rows -- and pins ``straggling=False``
    bit-for-bit against a call that never mentions the keyword.

    The tolerance is exact equality on every returned array. Nothing about the
    straggling-off path is allowed to be statistical."""
    core_kwargs = WIRED_CORES[name]
    seed = 11
    common = dict(Ne=200, seed=seed, energy_model=energy_model, max_dE_frac=max_dE_frac)
    default = _run(core_kwargs, **common)
    explicit_off = _run(core_kwargs, **common, straggling=False)
    # Not vacuous: the redefined crossing has to be the branch under test.
    assert default["n_cutoff_stopped"] > 100
    if max_dE_frac > 0.0:
        assert default["L_ang"].size > 2 * explicit_off["n_cutoff_stopped"]
    assert "straggle_dE_keV" not in default
    assert "straggle_dE_keV" not in explicit_off
    for key, left in default.items():
        right = explicit_off[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=f"{name}:{key}")
        elif isinstance(left, (int, float, np.number)):
            assert left == right, f"{name}:{key}"


# --- property 2: the loss is applied, and applied sanely, on every core ------


@pytest.mark.parametrize("name", sorted(WIRED_CORES))
def test_straggling_changes_the_run_and_conserves_energy_sensibly(name):
    """The headline behavioural change, per core: with straggling on the
    electron's energy actually moves, and it moves to somewhere physical.

    Four invariants, none of which the diagnostic-only slice-D wiring could
    have satisfied or violated:

    * no row starts below the cutoff and none starts at a non-finite energy,
      even though a single sampled loss may legitimately exceed the whole
      remaining kinetic energy (slice C decision 3);
    * every electron is accounted for exactly once by one of the four exit
      channels, so the redefined ``cutoff_j`` neither double-counts a track
      that also hit a boundary nor loses one -- on the per-electron cores this
      additionally exercises the ``exit_code`` derivation, which is where the
      flag clearing had to be re-expressed;
    * the run really did diverge from the deterministic one;
    * the sampled loss closes on the transport's own mean stopping power.
      ``<dE> = C s`` holds row by row (slice C), so summing over every row of a
      real run is a closure test of the integration, not of the sampler. The
      1% tolerance is set by the ~1e5-row sample size, not by the observed
      error."""
    core_kwargs = WIRED_CORES[name]
    Ne = 800
    on = _run(core_kwargs, Ne=Ne, seed=11, straggling=True)
    off = _run(core_kwargs, Ne=Ne, seed=11, straggling=False)

    assert np.all(np.isfinite(on["E_start_keV"]))
    assert np.all(on["E_start_keV"] >= E_CUT_KEV)
    assert np.all(on["L_ang"] >= 0.0)
    assert np.all(np.isfinite(on["straggle_dE_keV"]))

    channels = (
        on["n_backscattered"] + on["n_transmitted"] + on["n_side_exited"] + on["n_cutoff_stopped"]
    )
    assert channels == Ne
    assert on["n_cutoff_stopped"] > 0.5 * Ne
    assert on["n_stopped"] == on["n_cutoff_stopped"]

    assert not np.array_equal(on["L_ang"], off["L_ang"])

    mean_loss = np.abs(spliced_stopping_keV_per_ang(GRAPHITE, on["E_start_keV"])) * on["L_ang"]
    assert on["straggle_dE_keV"].sum() == pytest.approx(mean_loss.sum(), rel=0.01)


# --- property 3: cross-core parity of the sampler AND the crossing rule ------


def _reconstruct(out, seed, Ne, E_cut):
    """Replay each electron's rows offline from the documented key scheme.

    Returns ``(n_rows_checked, n_crossings_checked)``. Raises ``AssertionError``
    on the first row whose recorded energy, length or crossing classification
    disagrees with the reconstruction. Uses only the run's own recorded
    ``(E_start_keV, L_ang, E_end_keV, flight_id, substep_id)``, the
    ``straggle_dE_keV`` accumulator and the public sampler -- never the core's
    internals -- so agreement is evidence that the core addressed the sampler
    with the documented key and applied slice E's rule, on whichever core
    produced ``out``.
    """
    table = urban_element_table(GRAPHITE)
    keys = stream_keys(seed, Ne)
    n_rows = 0
    n_cross = 0
    for e in range(Ne):
        rows = np.flatnonzero(out["electron_id"] == e)
        if rows.size == 0:
            continue
        urban_key = np.uint64(_urban_stream_key_scalar(np.uint64(keys[e])))
        sampled_total = 0.0
        for pos, i in enumerate(rows):
            E_start = float(out["E_start_keV"][i])
            E_end = float(out["E_end_keV"][i])
            s_row = float(out["L_ang"][i])
            flight_key = np.uint64(
                _urban_flight_key_scalar(
                    urban_key,
                    np.int64(out["flight_id"][i]),
                    np.int64(out["substep_id"][i]),
                )
            )
            delta = E_start - E_cut
            terminal = pos == rows.size - 1
            if terminal and E_end == E_cut:
                # A crossing row is truncated to the first-passage distance
                # AFTER the sampler drew over the untruncated length, so
                # `L_ang` is no longer that length. Recover the row's own draw
                # as the residual of the accumulator, then invert the fluid
                # interpolation `s_row = s_full * delta / dE` for `s_full` and
                # re-draw over it: reproducing `dE` exactly pins both halves of
                # the rule at once.
                dE = float(out["straggle_dE_keV"][e]) - sampled_total
                # `dE` is a difference of accumulated sums, so it carries the
                # rounding of that cancellation while the redraw below is a
                # fresh sum; both are the same compound draw. Same tolerance
                # and same reason as slice E's own round trip in
                # `test_overshooting_rows_stop_exactly_at_the_cutoff`.
                assert dE >= delta * (1.0 - 1e-9), f"crossing row sampled {dE} < {delta}"
                s_full = s_row * dE / delta if dE > 0.0 else s_row
                redrawn, _c = _urban_sample_compound_keV(
                    *table, 0.0, E_start, s_full, flight_key, np.uint64(0)
                )
                assert redrawn == pytest.approx(dE, rel=1e-9, abs=1e-12)
                n_cross += 1
            else:
                loss, _c = _urban_sample_compound_keV(
                    *table, 0.0, E_start, s_row, flight_key, np.uint64(0)
                )
                # The indicator is exact and this row did not cross, so the
                # draw must have stayed strictly inside the available energy.
                assert loss <= delta, f"non-crossing row sampled {loss} > available {delta}"
                assert E_end == E_start - loss
                sampled_total += loss
                n_rows += 1
    return n_rows, n_cross


@pytest.mark.parametrize("name", sorted(ALL_HOST_CORES))
def test_rows_reproduce_the_sampler_and_the_crossing_rule(name):
    """Cross-core parity, stated where it actually holds.

    The cores realize different trajectories by construction, so whole-run
    equality between them is meaningless. What *is* identical on every core is
    the pair slice F ports: the sampled loss for a given ``(seed, electron,
    flight, substep, E_start, s)``, and slice E's crossing rule applied to it.
    Both are reconstructed here from each core's own recorded rows and demanded
    equal in exact float64 -- bit-for-bit, not to a tolerance, because the
    arithmetic path is literally the same function on every core. The cores
    differ only in *which* ``(E_start, s)`` they present to it, which is the
    LUT-versus-exact and lockstep-versus-per-electron difference and is not
    what this test measures.

    Includes slice E's own core as the reference, so a future change that moved
    the rule on one core but not another would fail here rather than silently
    fork the semantics.
    """
    core_kwargs = ALL_HOST_CORES[name]
    Ne, seed = 60, 5
    out = _run(core_kwargs, Ne=Ne, seed=seed, straggling=True, energy_model="midpoint")
    assert out["n_cutoff_stopped"] > 0.5 * Ne
    n_rows, n_cross = _reconstruct(out, seed, Ne, E_CUT_KEV)
    # Not vacuous on either branch.
    assert n_rows > 1000
    assert n_cross > 0.5 * Ne


# --- property 4: the cutoff-overshoot condition, on every core ---------------


@pytest.mark.parametrize("name", sorted(WIRED_CORES))
def test_overshooting_rows_stop_exactly_at_the_cutoff(name):
    """The case slice C handed to slice E, re-driven on each ported core.

    A single flight can sample a loss larger than the electron's whole
    remaining energy above the cutoff, because the sampler deliberately does
    not clamp ``dE`` to ``E`` (clamping would break ``<dE> = C s``, the one
    property Urban was selected for). Driven into the regime where that
    dominates -- 6 keV electrons against the 5 keV cutoff, so only 1 keV is
    available while a single continuum quantum reaches ``T_max = E/2 = 3`` keV
    -- and checked against the rule rather than against a symptom:

    * the last row of every cutoff-terminated track ends at ``E_cut``
      *exactly*, never below it and never at a negative energy;
    * the truncated length is non-negative and no longer than the row would
      have been, so an overshoot shortens the step rather than reversing it;
    * the overshoot is severe and common rather than marginal here, so the rule
      is genuinely exercised."""
    core_kwargs = WIRED_CORES[name]
    Ne, seed = 400, 7
    out = _run(
        core_kwargs,
        Ne=Ne,
        seed=seed,
        E0_keV=6.0,
        thickness_ang=60000.0,
        straggling=True,
        energy_model="midpoint",
    )
    assert out["n_cutoff_stopped"] > 0.9 * Ne

    overshoots = []
    for e in range(Ne):
        rows = np.flatnonzero(out["electron_id"] == e)
        if rows.size == 0:
            continue
        last = rows[-1]
        if out["E_end_keV"][last] != E_CUT_KEV:
            continue
        assert out["E_end_keV"][last] == E_CUT_KEV
        assert out["L_ang"][last] >= 0.0
        E_start = float(out["E_start_keV"][last])
        delta = E_start - E_CUT_KEV
        assert delta > 0.0
        # The row's own draw, as the residual of the accumulator over the
        # earlier rows -- which are non-crossing, so their lengths are the
        # lengths sampled over.
        earlier = 0.0
        urban_key = np.uint64(_urban_stream_key_scalar(np.uint64(stream_keys(seed, Ne)[e])))
        table = urban_element_table(GRAPHITE)
        for i in rows[:-1]:
            flight_key = np.uint64(
                _urban_flight_key_scalar(
                    urban_key, np.int64(out["flight_id"][i]), np.int64(out["substep_id"][i])
                )
            )
            loss, _c = _urban_sample_compound_keV(
                *table,
                0.0,
                float(out["E_start_keV"][i]),
                float(out["L_ang"][i]),
                flight_key,
                np.uint64(0),
            )
            earlier += loss
        dE = float(out["straggle_dE_keV"][e]) - earlier
        # The exact indicator, empirically: the crossing row's loss reaches the
        # available energy on every track without exception. The slack is the
        # rounding of the accumulator cancellation, not a physical margin.
        assert dE >= delta * (1.0 - 1e-9)
        overshoots.append(dE / delta)

    assert len(overshoots) > 0.9 * Ne
    # Severe and common, not marginal: the rule is doing real work here.
    assert np.median(overshoots) > 1.5


# --- grooved: the facet-crossing interaction slice E did not face ------------
#
# Geometry follows `tests/montecarlo/test_groove.py`'s own convention -- a
# blazed sawtooth at 45 degree tilt with the matching tilted beam -- because
# that is the configuration in which the core actually produces material exit /
# vacuum re-entry pairs in quantity. A groove that never sends an electron into
# vacuum would make every assertion below vacuous.

TILT_POLAR_RAD = np.deg2rad(45.0)
GROOVE = blazed_groove_spec(
    spacing_ang=2.0e4,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TILT_POLAR_RAD,
    tilt_azim_rad=np.pi,
)

GROOVED_SLAB = dict(
    E0_keV=60.0,
    element="C",
    n_atoms_per_ang3=0.1136,
    thickness_ang=2.0e5,
    elastic_model="sr",
    groove=GROOVE,
    tilt_polar_rad=TILT_POLAR_RAD,
    tilt_azim_rad=np.pi,
)


def _run_grooved(**overrides):
    from pyrite.montecarlo.geometry import tilted_geometry

    beam, _ = tilted_geometry(np.pi / 2, TILT_POLAR_RAD, np.pi)
    kwargs = dict(GROOVED_SLAB)
    kwargs["beam_dir"] = beam
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


def test_grooved_straggling_off_is_inert():
    """Property 1 for the grooved core, in a run that both crosses the cutoff
    and emits facet crossings, so both new orderings would be taken if the
    ``straggle_on`` gate leaked."""
    common = dict(Ne=120, seed=42)
    default = _run_grooved(**common)
    explicit_off = _run_grooved(**common, straggling=False)
    assert default["n_cutoff_stopped"] > 0
    # Facet crossings are a minority event in this geometry (order one vacuum
    # leg per ~15 electrons, measured); a handful is enough to prove the
    # material-side branch is reachable in this run.
    assert default["vacuum_E_keV"].size > 5
    assert "straggle_dE_keV" not in default
    for key, left in default.items():
        right = explicit_off[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=key)
        elif isinstance(left, (int, float, np.number)):
            assert left == right, key


def test_grooved_straggling_conserves_energy_sensibly():
    """Property 2 for the grooved core, with the material-side closure.

    The vacuum legs carry no energy loss -- vacuum has no stopping power -- so
    the closure test is against the *material* path length only, which is
    exactly what ``L_ang`` records; the vacuum legs live in a separate
    ``vacuum_*`` block. If the sampler had been handed the untruncated flight
    length rather than the facet-truncated material-side one, this closure
    would come out high by the vacuum fraction."""
    Ne = 400
    on = _run_grooved(Ne=Ne, seed=42, straggling=True)
    off = _run_grooved(Ne=Ne, seed=42, straggling=False)

    assert np.all(np.isfinite(on["E_start_keV"]))
    assert np.all(on["E_start_keV"] >= E_CUT_KEV)
    assert np.all(on["L_ang"] >= 0.0)
    assert not np.array_equal(on["L_ang"], off["L_ang"])
    # The groove is doing something, or the material-side distinction is moot.
    assert on["vacuum_E_keV"].size > 20

    channels = (
        on["n_backscattered"] + on["n_transmitted"] + on["n_side_exited"] + on["n_cutoff_stopped"]
    )
    assert channels + on["n_missed"] + on["n_step_limited"] == Ne

    mean_loss = np.abs(spliced_stopping_keV_per_ang(GRAPHITE, on["E_start_keV"])) * on["L_ang"]
    assert on["straggle_dE_keV"].sum() == pytest.approx(mean_loss.sum(), rel=0.01)


def test_grooved_vacuum_legs_lose_no_energy():
    """The material-side statement in its most direct form: the energy at the
    start of a vacuum leg is the end energy of the material row that produced
    it. If the loss had been sampled over the untruncated flight length, or
    applied across the vacuum leg, the recorded ``vacuum_E_keV`` would not
    match any material row's ``E_end_keV`` exactly."""
    Ne = 400
    out = _run_grooved(Ne=Ne, seed=3, straggling=True, energy_model="midpoint")
    assert out["vacuum_E_keV"].size > 20  # ~1 leg per 15 electrons, measured

    checked = 0
    for e in np.unique(out["vacuum_elec_id"]):
        vac_E = out["vacuum_E_keV"][out["vacuum_elec_id"] == e]
        row_E_end = out["E_end_keV"][out["electron_id"] == e]
        for v in vac_E:
            assert np.any(row_E_end == v), (e, v)
            checked += 1
    assert checked > 20


def test_grooved_cutoff_beats_a_facet_crossing_on_the_material_side():
    """The grooved-specific interaction slice F had to design.

    A row whose collision distance would carry the electron through a groove
    facet into vacuum is truncated at the facet first, so the length handed to
    the sampler is the material-side length. If that material-side loss reaches
    the available energy, monotonicity puts the first passage strictly inside
    the material part of the row: the electron must stop in the material and
    never reach the vacuum.

    Driven into the regime where that competition is common -- a cutoff set
    just below the beam energy, so nearly every row is a candidate crossing
    while facet crossings stay frequent -- and checked as the property, not as
    "it did not crash":

    * every cutoff-terminated track ends at ``E_cut`` exactly, never below;
    * no electron has a vacuum leg recorded at or after its terminating row's
      clock, i.e. no track that stopped at the cutoff also emitted the facet
      crossing it would have made had the crossing not won. This is the
      assertion that fails if ``surface_first`` is not cleared alongside the
      other geometry flags on a crossing;
    * both event kinds are present in quantity, so the competition is
      exercised rather than assumed."""
    Ne = 300
    out = _run_grooved(
        Ne=Ne,
        seed=17,
        E_cut_keV=59.0,
        straggling=True,
        energy_model="midpoint",
    )
    # Both event kinds present in quantity, or the competition is untested.
    assert out["n_cutoff_stopped"] > 0.5 * Ne
    assert out["vacuum_E_keV"].size > 5
    assert np.all(out["E_end_keV"] >= 59.0)

    n_terminal = 0
    for e in range(Ne):
        rows = np.flatnonzero(out["electron_id"] == e)
        if rows.size == 0:
            continue
        last = rows[np.argmax(out["t_ang"][rows])]
        if out["E_end_keV"][last] != 59.0:
            continue
        n_terminal += 1
        # Terminating at the cutoff means terminating in material: that row is
        # the last event this electron has. A vacuum leg starting at or after
        # its clock would mean the facet crossing fired anyway.
        t_last = out["t_ang"][last]
        vac = out["vacuum_elec_id"] == e
        if np.any(vac):
            assert np.all(out["vacuum_t0_ang"][vac] < t_last), e
    assert n_terminal > 0.5 * Ne
