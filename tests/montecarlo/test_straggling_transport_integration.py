"""Applying the Urban straggling loss inside the ungrooved lockstep core.

Slice E of ``feature/energy-loss-straggling``. Slice C built the sampler and
slice D addressed it per ``(electron, flight, substep)`` without applying it;
this module covers actually subtracting it from the electron's energy in
``_transport_core_ungrooved``, which forces two redefinitions that the
deterministic core got for free.

**Cutoff crossing.** The deterministic core solves ``E_end == E_cut`` for a
truncation distance, which has no meaning once the loss over a step is a draw
rather than a function of distance. The replacement, derived in the
"stochastic energy loss in transport (slice E)" block above
``_transport_core_ungrooved``, splits the question in two:

* the Urban loss is a compound-Poisson *subordinator*, so it is non-decreasing
  in the step length; "this row crosses ``E_cut``" is therefore EXACTLY
  equivalent to "the row's total sampled loss reaches ``E_start - E_cut``".
  The crossing indicator, and hence ``n_cutoff_stopped``, carries no
  approximation at all;
* the crossing *location* is the fluid interpolation at the row's own realized
  rate, ``s_cut = s (E_start - E_cut) / dE``, which collapses term by term
  onto the deterministic ``cutoff_distance = (E_cut - E_j)/dEds`` when the
  loss is deterministic, and sends an overshooting draw to a vanishing step
  rather than to a negative energy.

Overshoot is the common case, not a corner: slice C decision 3 records that
the sampler deliberately does not clamp ``dE`` to ``E`` (clamping would break
``<dE> = C s``, the property Urban was selected for), and
``test_overshooting_rows_stop_exactly_at_the_cutoff`` below measures a median
overshoot of 2x the available energy in the low-``E``/high-cutoff regime.

**Substep invariance.** ``docs/validation/beam-transport/substep-radiation-invariance.md``
states an *algebraic* invariance under ``max_dE_frac`` subdivision. That claim
does not survive a random loss and is replaced by a *distributional* one, in
two parts, each tested here:

1. at frozen energy the invariance is EXACT, not asymptotic -- a compound
   Poisson sum is infinitely divisible, so ``sum_m CP(s_m Sigma) =_d
   CP(s Sigma)`` for any partition. It is distributional, not pathwise: each
   substep addresses its own ``(flight, substep)`` key, so the realized
   numbers differ and only the law is preserved;
2. the only substep dependence is the drift of ``Sigma_i`` with ``E`` inside
   the flight, which shifts the mean by ``-C C' s^2 (N-1)/(2N) + O(s^3)`` --
   and the ``N -> infinity`` limit of that is the exactly integrated mean, so
   substepping under straggling converges in the same direction and at the
   same order as the deterministic frozen model's own left-endpoint quadrature
   error. Straggling adds no substep dependence of its own.

Every test here pins the ungrooved lockstep core with the exact (non-LUT)
stopping power, which is the only core slice E integrates; the remaining cores
stay diagnostic-only until slice F.
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    _urban_flight_key_scalar,
    _urban_sample_compound_keV,
    _urban_stream_key_scalar,
    simulate_trajectories,
    spliced_stopping_keV_per_ang,
    stream_keys,
    urban_element_table,
    urban_loss_moments_keV,
)

NO_LUT = TransportLUTConfig(enabled=False)

GRAPHITE = [("C", 0.1136)]

# The one core slice E integrates. Named explicitly rather than left to
# `transport_core="auto"` so this module cannot silently start testing the
# CUDA core on a machine that has one.
CORE = dict(transport_core="lockstep", transport_lut_config=NO_LUT)

# 25 keV in graphite against the default 5 keV cutoff, in a slab thicker than
# the CSDA range, so nearly every electron ends by reaching the cutoff and the
# redefined crossing is exercised on most tracks rather than on a handful.
THICK_SLAB = dict(
    E0_keV=25.0,
    element="C",
    n_atoms_per_ang3=0.1136,
    thickness_ang=60000.0,
    max_steps=20000,
    **CORE,
)


def _run(**overrides):
    kwargs = dict(THICK_SLAB)
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


# --- property 1: the off path is unreachable ---------------------------------


@pytest.mark.parametrize(
    ("energy_model", "max_dE_frac"),
    [("frozen", 0.0), ("midpoint", 0.0), ("midpoint", 0.02)],
)
def test_straggling_off_is_inert_in_a_cutoff_and_substep_heavy_run(energy_model, max_dE_frac):
    """Slice E reorders the deterministic core's cutoff solve and its
    ``max_dE_frac`` cap into an ``if straggle_on: ... else: ...`` pair. The
    ``else`` branch is textually the previous code, but "textually unchanged"
    is not a test, so this drives a run in which BOTH new orderings would be
    taken if the gate leaked -- thousands of cutoff crossings and, in the
    substep case, tens of thousands of energy-capped rows -- and pins
    ``straggling=False`` bit-for-bit against a call that never mentions the
    keyword.

    The tolerance is exact equality on every returned array. Nothing about
    the straggling-off path is allowed to be statistical."""
    seed = 11
    default = _run(Ne=300, seed=seed, energy_model=energy_model, max_dE_frac=max_dE_frac)
    explicit_off = _run(
        Ne=300,
        seed=seed,
        energy_model=energy_model,
        max_dE_frac=max_dE_frac,
        straggling=False,
    )
    # Not vacuous: the redefined crossing has to be the branch under test.
    assert default["n_cutoff_stopped"] > 100
    if max_dE_frac > 0.0:
        assert default["L_ang"].size > 2 * explicit_off["n_cutoff_stopped"]
    assert "straggle_dE_keV" not in default
    assert "straggle_dE_keV" not in explicit_off
    for key, left in default.items():
        right = explicit_off[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=key)
        elif isinstance(left, (int, float, np.number)):
            assert left == right, key


# --- property 2: the loss is applied, and applied sanely ---------------------


def test_straggling_changes_the_run_and_conserves_energy_sensibly():
    """The headline behavioural change: with straggling on, the electron's
    energy actually moves, and it moves to somewhere physical.

    Four invariants, none of which the diagnostic-only slice-D wiring could
    have satisfied or violated:

    * no row starts below the cutoff and none starts at a non-finite energy,
      even though a single sampled loss may legitimately exceed the whole
      remaining kinetic energy (slice C decision 3);
    * every electron is accounted for exactly once by one of the four exit
      channels, so the redefined ``cutoff_j`` neither double-counts a track
      that also hit a boundary nor loses one;
    * the run really did diverge from the deterministic one;
    * the sampled loss closes on the transport's own mean stopping power.
      ``<dE> = C s`` holds row by row (slice C), so summing over every row of
      a real run is a closure test of the integration, not of the sampler:
      the tolerance is 1% over ~5.6e5 rows, against a measured 0.07%."""
    Ne = 2000
    on = _run(Ne=Ne, seed=11, straggling=True)
    off = _run(Ne=Ne, seed=11, straggling=False)

    E_cut = 5.0
    assert np.all(np.isfinite(on["E_start_keV"]))
    assert np.all(on["E_start_keV"] >= E_cut)
    assert np.all(on["L_ang"] >= 0.0)
    assert np.all(np.isfinite(on["straggle_dE_keV"]))

    channels = (
        on["n_backscattered"] + on["n_transmitted"] + on["n_side_exited"] + on["n_cutoff_stopped"]
    )
    assert channels == Ne
    assert on["n_cutoff_stopped"] > 0.5 * Ne
    assert on["n_stopped"] == on["n_cutoff_stopped"]

    assert not np.array_equal(on["L_ang"], off["L_ang"])
    assert on["n_cutoff_stopped"] != off["n_cutoff_stopped"] or not np.array_equal(
        on["E_start_keV"], off["E_start_keV"]
    )

    mean_loss = np.abs(spliced_stopping_keV_per_ang(GRAPHITE, on["E_start_keV"])) * on["L_ang"]
    assert on["straggle_dE_keV"].sum() == pytest.approx(mean_loss.sum(), rel=0.01)


def test_row_energies_are_the_start_energy_minus_the_sampled_loss():
    """The loss is applied *as drawn*, row by row, not merely reflected in an
    aggregate. Under ``energy_model="midpoint"`` each row carries its own
    ``E_end_keV``, so every non-terminal row can be checked against the loss
    the documented ``(electron, flight, substep)`` key produces offline:
    ``E_end == E_start - dE`` exactly, in float64, with no tolerance.

    Terminal rows are excluded and covered by
    ``test_overshooting_rows_stop_exactly_at_the_cutoff``: a crossing row is
    truncated to the first-passage distance *after* the sampler has drawn over
    the untruncated length, so its recorded ``L_ang`` is no longer the length
    that was sampled over."""
    Ne, seed = 120, 5
    out = _run(Ne=Ne, seed=seed, straggling=True, energy_model="midpoint")
    keys = stream_keys(seed, Ne)
    table = urban_element_table(GRAPHITE)
    checked = 0
    for e in range(Ne):
        rows = np.flatnonzero(out["electron_id"] == e)
        urban_key = np.uint64(_urban_stream_key_scalar(np.uint64(keys[e])))
        for i in rows[:-1]:
            flight_key = np.uint64(
                _urban_flight_key_scalar(
                    urban_key, np.int64(out["flight_id"][i]), np.int64(out["substep_id"][i])
                )
            )
            loss, _counter = _urban_sample_compound_keV(
                *table,
                0.0,
                float(out["E_start_keV"][i]),
                float(out["L_ang"][i]),
                flight_key,
                np.uint64(0),
            )
            assert out["E_end_keV"][i] == float(out["E_start_keV"][i]) - loss
            checked += 1
    assert checked > 1000


def test_overshooting_rows_stop_exactly_at_the_cutoff():
    """The case slice C handed to slice E: a single flight sampling a loss
    larger than the electron's whole remaining energy above the cutoff.

    Driven into the regime where it dominates -- 6 keV electrons against the
    5 keV cutoff, so only 1 keV is available while a single continuum quantum
    reaches ``T_max = E/2 = 3`` keV -- and checked against the crossing rule
    itself rather than against a symptom:

    * the last row of every cutoff-terminated track ends at ``E_cut``
      *exactly*, never below it and never at a negative energy;
    * that row's sampled loss (recovered as the residual of
      ``straggle_dE_keV`` after reconstructing every earlier row offline)
      always reaches the available energy ``E_start - E_cut``. That is the
      crossing condition, and it holding on every single track with no
      exception is the empirical form of "the indicator is exact because the
      loss process is non-decreasing";
    * the overshoot is severe and common rather than marginal, so the rule is
      genuinely exercised: a median of 2x the available energy here;
    * the crossing *location* is the fluid interpolation and nothing else.
      The recorded ``L_ang`` of a crossing row is ``s_full (E_start - E_cut)
      / dE``, so inverting it recovers the untruncated length the sampler
      actually drew over -- and re-running the sampler at that length must
      return the very loss used to compute it. That round trip closes only
      for this rule: travelling the full row instead, or placing the stop at
      a uniformly drawn position, recovers a different length and a different
      loss.

    A rule that clamped the loss, or that stopped the electron at the
    deterministic cutoff distance, would fail the crossing-condition
    assertion; one that let the row run its full length would fail the
    ``E_end == E_cut`` assertion and the round trip."""
    Ne, seed, E_cut = 400, 7, 5.0
    out = _run(
        Ne=Ne,
        seed=seed,
        E0_keV=6.0,
        E_cut_keV=E_cut,
        straggling=True,
        energy_model="midpoint",
    )
    assert np.all(out["E_end_keV"] >= E_cut)
    assert out["E_end_keV"].min() == E_cut

    keys = stream_keys(seed, Ne)
    table = urban_element_table(GRAPHITE)
    overshoot = []
    round_trips = 0
    for e in range(Ne):
        rows = np.flatnonzero(out["electron_id"] == e)
        if out["E_end_keV"][rows[-1]] != E_cut:
            continue  # left through a surface instead
        urban_key = np.uint64(_urban_stream_key_scalar(np.uint64(keys[e])))
        earlier = 0.0
        for i in rows[:-1]:
            flight_key = np.uint64(
                _urban_flight_key_scalar(
                    urban_key, np.int64(out["flight_id"][i]), np.int64(out["substep_id"][i])
                )
            )
            loss, _counter = _urban_sample_compound_keV(
                *table,
                0.0,
                float(out["E_start_keV"][i]),
                float(out["L_ang"][i]),
                flight_key,
                np.uint64(0),
            )
            earlier += loss
        last = rows[-1]
        available = float(out["E_start_keV"][last]) - E_cut
        crossing_loss = out["straggle_dE_keV"][e] - earlier
        overshoot.append(crossing_loss / available)

        # Invert the truncation to recover the length the sampler drew over,
        # then re-draw at that length: it must reproduce `crossing_loss`.
        s_full = float(out["L_ang"][last]) * crossing_loss / available
        flight_key = np.uint64(
            _urban_flight_key_scalar(
                urban_key, np.int64(out["flight_id"][last]), np.int64(out["substep_id"][last])
            )
        )
        redrawn, _counter = _urban_sample_compound_keV(
            *table,
            0.0,
            float(out["E_start_keV"][last]),
            s_full,
            flight_key,
            np.uint64(0),
        )
        # `crossing_loss` is a difference of accumulated sums, so it carries
        # the rounding of that cancellation; the redraw is a fresh sum. Both
        # are the same compound draw, so they agree to float64 rounding of
        # the electron's whole accumulated loss.
        assert redrawn == pytest.approx(crossing_loss, rel=1e-9, abs=1e-12)
        round_trips += 1
    overshoot = np.array(overshoot)
    assert overshoot.size > 0.9 * Ne
    assert round_trips == overshoot.size
    # The crossing condition, on every track without exception.
    assert overshoot.min() >= 1.0
    # ... and it is a real overshoot, not a boundary case.
    assert np.median(overshoot) > 1.5
    assert overshoot.max() > 10.0


# --- property 3: substep invariance ------------------------------------------
#
# Tested on a harness that reproduces the core's substep recursion exactly --
# substep m draws over s/N at the energy the previous substeps left behind,
# from its own `(flight, substep)` key -- rather than through the transport,
# so that the flight length and start energy are held fixed and the only thing
# varying is the subdivision. The transport-level consequence is checked
# separately in `test_max_dE_frac_preserves_the_straggled_loss_in_transport`.

_SUBSTEP_E_KEV = 25.0
_SUBSTEP_S_ANG = 1.0e4  # mean loss 2.25 keV, i.e. dE/E = 0.09 over the flight
_SUBSTEP_N = 32
_SUBSTEP_REPS = 20000
_SUBSTEP_SEED = 3


def _substep_sum(table, s_ang, n_sub, base_key, drift):
    """Total loss over ``s_ang`` split into ``n_sub`` substeps.

    ``drift=False`` freezes every substep at the flight's start energy, which
    is the exactly-invariant case; ``drift=True`` lets ``E`` fall inside the
    flight the way the core does, which is the only source of substep
    dependence."""
    total = 0.0
    energy = _SUBSTEP_E_KEV
    for m in range(n_sub):
        flight_key = np.uint64(
            _urban_flight_key_scalar(np.uint64(base_key), np.int64(0), np.int64(m))
        )
        loss, _counter = _urban_sample_compound_keV(
            *table,
            0.0,
            float(energy if drift else _SUBSTEP_E_KEV),
            float(s_ang / n_sub),
            flight_key,
            np.uint64(0),
        )
        total += loss
        energy -= loss
    return total


def _substep_samples(n_sub, drift):
    table = urban_element_table(GRAPHITE)
    rng = np.random.default_rng(_SUBSTEP_SEED)
    base_keys = rng.integers(0, 2**63, size=_SUBSTEP_REPS, dtype=np.int64)
    return np.array([_substep_sum(table, _SUBSTEP_S_ANG, n_sub, int(k), drift) for k in base_keys])


def test_substep_split_is_distribution_preserving_at_frozen_energy():
    """Part 1 of the re-derivation, and the strongest form the invariance can
    take: at frozen energy, splitting a flight into substeps preserves the
    loss *distribution exactly*, for any partition and any substep count.

    Compound Poisson sums are infinitely divisible --
    ``sum_m Poisson(s_m Sigma) = Poisson(s Sigma)`` with i.i.d. marks -- so
    this is an identity, not a limit. It is distributional and NOT pathwise:
    each substep draws from its own ``(flight, substep)`` key, so the realized
    numbers differ and a bit-for-bit test would be wrong.

    Both moments are therefore compared against the analytic ones and against
    each other. The mean tolerance is 4 standard errors of the Monte Carlo
    estimate (about 3% here, on 20000 repetitions of a heavy-tailed sum);
    the variance tolerance is 15%, set by the fourth moment of a ``1/E^2``
    continuum rather than by the observed error, which is under 3%."""
    unsplit = _substep_samples(1, drift=False)
    split = _substep_samples(_SUBSTEP_N, drift=False)
    analytic_mean, analytic_var = urban_loss_moments_keV(GRAPHITE, _SUBSTEP_E_KEV, _SUBSTEP_S_ANG)

    stderr = np.sqrt(unsplit.var(ddof=1) / unsplit.size + split.var(ddof=1) / split.size)
    assert abs(split.mean() - unsplit.mean()) < 4.0 * stderr
    for sample in (unsplit, split):
        assert sample.mean() == pytest.approx(analytic_mean, rel=0.02)
        assert sample.var(ddof=1) == pytest.approx(analytic_var, rel=0.15)


def test_substep_drift_shift_matches_the_derived_second_order_term():
    """Part 2 of the re-derivation: the ONLY substep dependence is the drift
    of the Poisson rates with ``E`` inside the flight, and it is second order
    in the step.

    Derived: with ``C(E) = |dE/dx|``, equal substeps and ``Y_{m-1}`` the loss
    accumulated before substep ``m``,

        <sum_m X_m> - <X> = -C C' sum_m s_m sigma_{m-1} + O(s^3)
                          = -C C' s^2 (N-1)/(2N) + O(s^3),

    positive because ``C' = dC/dE < 0``: substeps taken at the lower energies
    the earlier substeps left behind lose more. The ``N -> infinity`` limit,
    ``-C C' s^2/2``, is exactly the second-order term of the true integral
    ``int_0^s C(E(s')) ds' = C s - (1/2) C C' s^2 + O(s^3)``, so subdividing
    moves the straggled mean toward the correct value and not away from it --
    the same left-endpoint quadrature error the deterministic frozen model
    already carries, and nothing new introduced by the fluctuation.

    Tested rather than asserted: the measured shift is compared against the
    leading-order prediction, and separately shown to be many standard errors
    away from zero so the agreement is not an agreement between two numbers
    that are both noise. The accepted band is 0.8--1.6 of the prediction: at
    the ``dE/E = 0.09`` used here the measured ratio is ~1.15, the excess
    being the ``O(s^3)`` remainder the expansion drops."""
    unsplit = _substep_samples(1, drift=False)
    drifted = _substep_samples(_SUBSTEP_N, drift=True)
    shift = drifted.mean() - unsplit.mean()
    stderr = np.sqrt(unsplit.var(ddof=1) / unsplit.size + drifted.var(ddof=1) / drifted.size)

    # C and C' by central difference on the model's own analytic mean, which
    # is |dE/dx| by the closure identity -- deliberately not the sampler.
    step = 1.0e-3
    per_ang = lambda energy: urban_loss_moments_keV(GRAPHITE, energy, 1.0)[0]  # noqa: E731
    c_value = per_ang(_SUBSTEP_E_KEV)
    c_prime = (per_ang(_SUBSTEP_E_KEV + step) - per_ang(_SUBSTEP_E_KEV - step)) / (2.0 * step)
    predicted = (
        -c_value * c_prime * _SUBSTEP_S_ANG * _SUBSTEP_S_ANG * (_SUBSTEP_N - 1) / (2.0 * _SUBSTEP_N)
    )

    assert c_prime < 0.0 and predicted > 0.0
    assert shift > 3.0 * stderr, "substep drift shift is indistinguishable from Monte Carlo noise"
    assert 0.8 * predicted < shift < 1.6 * predicted

    # Same statement as a fractional bound a caller can apply without
    # re-deriving: half the flight's own fractional energy loss times the
    # logarithmic slope of the stopping power, to leading order.
    mean_loss = c_value * _SUBSTEP_S_ANG
    bound = 0.5 * abs(c_prime * _SUBSTEP_E_KEV / c_value) * (mean_loss / _SUBSTEP_E_KEV)
    assert shift / unsplit.mean() < 1.3 * bound


def test_max_dE_frac_preserves_the_straggled_loss_in_transport():
    """The same invariance where it is actually consumed: turning
    ``max_dE_frac`` on subdivides every flight in a real run, and the total
    straggled loss per electron must not move.

    This is a distributional comparison of two independent runs, not a paired
    one -- subdividing changes each row's length, so every subsequent draw
    differs and the trajectories separate immediately. Tolerance 3% on the
    mean over 600 electrons and ~1.7e5 rows, against a measured 0.5%, which is
    far tighter than the per-flight bound derived above (flights lose well
    under 1% of ``E`` each, so the per-flight invariance residual is smaller
    than the run-to-run spread) and still loose enough not to fail on the
    Monte Carlo noise of two unpaired runs."""
    common = dict(Ne=600, seed=11, straggling=True, energy_model="midpoint")
    unsplit = _run(**common)
    split = _run(max_dE_frac=0.02, **common)

    assert split["L_ang"].size > unsplit["L_ang"].size
    assert np.all(split["E_end_keV"] >= 5.0)
    assert split["n_cutoff_stopped"] == pytest.approx(unsplit["n_cutoff_stopped"], rel=0.1)
    assert split["straggle_dE_keV"].mean() == pytest.approx(
        unsplit["straggle_dE_keV"].mean(), rel=0.03
    )
