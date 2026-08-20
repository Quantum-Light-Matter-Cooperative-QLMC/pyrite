"""RNG plumbing for the Urban straggling sampler across transport cores.

Slice D of ``feature/energy-loss-straggling``. Slice C implemented and unit
tested the Urban sampler itself as a standalone function
(``tests/montecarlo/test_energy_loss_straggling.py``); nothing called it from
inside a transport core. This module tests that it is correctly *addressed*
-- reachable per ``(electron, flight, substep)`` from all five host transport
cores on a stream domain disjoint from the existing free-path /
scattering-angle draws -- without being applied to ``E_keV``/``E_end_keV``
(that integration is slice E's job; see the "Recommendation for D" paragraph
in the task doc's slice C checklist entry).

Key-derivation scheme under test (``pyrite.montecarlo.transport``):

    urban_key  = _urban_stream_key_scalar(stream_key)                  -- once
                                                                     per electron
    flight_key = _urban_flight_key_scalar(urban_key, flight, substep) -- once
                                                                     per flight

``stream_key`` is ``stream_keys(seed, Ne)[electron]`` on every core: the
lockstep/grooved cores (which otherwise draw from a shared ``Generator``, not
a counter stream) receive it as a new ``stragg_stream_keys`` argument built
once in ``simulate_trajectories``; the per-electron cores reuse their own
existing ``stream_key``/``d_keys`` parameter, which was already exactly that
array. So ``(seed, electron)`` alone determines the straggling key domain
regardless of which core runs it, and ``(flight, substep)`` -- the physical
flight index and the ``max_dE_frac`` substep index within it, both tracked
internally by every core regardless of ``energy_model`` -- addresses one
sampler draw within it.

Three properties, each required by the task doc's slice D checklist entry and
by ``simulate_trajectories``'s ``straggling`` docstring paragraph:

1. ``straggling=False`` (the default) is bit-for-bit identical to a run that
   never passes the keyword at all, on every host core.
2. ``straggling=True`` never perturbs the free-path / scattering-angle draws:
   every returned array except the new ``straggle_dE_keV`` diagnostic is
   identical whether the flag is on or off.
3. ``straggle_dE_keV`` is exactly reproducible offline from a run's own
   recorded per-segment ``(E_start, L_ang)`` and the *position* of each
   segment within its electron's own segment list (the physical-flight index,
   since the default ``max_dE_frac=0`` never subdivides a flight, so there is
   exactly one segment per flight and substep is always 0) via the documented
   key scheme -- bit-for-bit on the host, on every core. Slice C's own
   sampler is a pure function of its scalar inputs, so this also establishes
   the cross-core claim: the same ``(seed, electron, flight, substep)`` given
   the same ``(E, s)`` inputs draws from the same stream and returns the same
   loss on lockstep, per-electron, and (by construction; see
   ``transport_jit_kernel.py``, untestable on this machine without a CUDA
   device) CUDA cores, up to libm ulp for the last.
"""

import numpy as np
import pytest

from pyrite.montecarlo.groove import GrooveSpec
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    _urban_flight_key_scalar,
    _urban_sample_compound_keV,
    _urban_stream_key_scalar,
    simulate_trajectories,
    stream_keys,
    urban_element_table,
)

NO_LUT = TransportLUTConfig(enabled=False)

BASE_KWARGS = dict(
    E0_keV=25.0,
    Ne=40,
    thickness_ang=2000.0,
    element="C",
    n_atoms_per_ang3=0.1136,
    seed=1234,
    max_steps=2000,
)

# The four host cores named in the task doc's slice D checklist entry that
# `simulate_trajectories` can reach without a groove; the grooved core is
# covered separately below since it needs a `GrooveSpec`. Frozen energy model
# and max_dE_frac=0 (both defaults) hold throughout this module: neither is
# slice D's to touch, and holding max_dE_frac at 0 keeps "one segment per
# physical flight, substep always 0" true, which the property-3 offline
# reconstruction below relies on.
CORE_CONFIGS = {
    "lockstep-exact": dict(transport_core="lockstep", transport_lut_config=NO_LUT),
    "lockstep-lut": dict(transport_core="lockstep"),
    "perelectron-exact": dict(transport_core="per-electron", transport_lut_config=NO_LUT),
    "perelectron-lut": dict(transport_core="per-electron"),
}

# Every field `simulate_trajectories` returns regardless of `straggling`,
# i.e. everything except the new diagnostic key. Listed explicitly rather
# than diffed against `result.keys()` so a future field addition fails this
# module's assumption loudly instead of silently narrowing the check.
NON_STRAGGLE_KEYS = (
    "initial_r_ang",
    "initial_v_hat",
    "initial_E_keV",
    "initial_t0_ang",
    "r_mid",
    "v_hat",
    "L_ang",
    "E_keV",
    "E_start_keV",
    "t_ang",
    "t_start_ang",
    "t0_ang",
    "elec_id",
    "electron_id",
    "layer",
    "vacuum_start_ang",
    "vacuum_end_ang",
    "vacuum_E_keV",
    "vacuum_t_ang",
    "vacuum_t0_ang",
    "vacuum_elec_id",
    "n_backscattered",
    "n_transmitted",
    "n_side_exited",
    "n_missed",
    "n_cutoff_stopped",
    "n_step_limited",
    "n_stopped",
    "Ne",
    "thickness_ang",
    "crystal_width_ang",
    "crystal_height_ang",
    "n_layers",
)


def _run(core_kwargs, **overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(core_kwargs)
    kwargs.update(overrides)
    return simulate_trajectories(**kwargs)


def _assert_non_straggle_fields_equal(a, b):
    for key in NON_STRAGGLE_KEYS:
        left, right = a[key], b[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=key)
        else:
            assert left == right, key


@pytest.mark.parametrize("name", sorted(CORE_CONFIGS))
def test_straggling_off_matches_the_unset_default(name):
    """Property 1. ``straggling=False`` is bit-for-bit identical to a call
    that never passes the keyword: the default is a provable no-op, not
    merely an unexercised branch, on every host core."""
    core_kwargs = CORE_CONFIGS[name]
    default = _run(core_kwargs)
    explicit_off = _run(core_kwargs, straggling=False)
    _assert_non_straggle_fields_equal(default, explicit_off)
    assert "straggle_dE_keV" not in default
    assert "straggle_dE_keV" not in explicit_off


def test_grooved_straggling_off_matches_the_unset_default():
    groove = GrooveSpec(spacing_ang=1.0e4, depth_ang=1.0e3, tilt_polar_rad=0.2)
    default = _run({}, groove=groove)
    explicit_off = _run({}, groove=groove, straggling=False)
    _assert_non_straggle_fields_equal(default, explicit_off)
    assert "straggle_dE_keV" not in default


@pytest.mark.parametrize("name", sorted(CORE_CONFIGS))
def test_straggling_on_never_perturbs_existing_draws(name):
    """Property 2. Every pre-existing returned array is identical whether
    straggling is sampled or not; only the new ``straggle_dE_keV`` diagnostic
    differs. The straggling stream (own salted rehash of the electron's
    stream key, own counter starting at 0) never shares state with the
    free-path / scattering-angle draws, so this cannot be an accident of a
    particular seed -- but it is still asserted per seed rather than trusted
    from inspection."""
    core_kwargs = CORE_CONFIGS[name]
    off = _run(core_kwargs, straggling=False)
    on = _run(core_kwargs, straggling=True)
    _assert_non_straggle_fields_equal(off, on)
    assert "straggle_dE_keV" in on
    assert on["straggle_dE_keV"].shape == (BASE_KWARGS["Ne"],)
    assert np.all(np.isfinite(on["straggle_dE_keV"]))
    assert np.all(on["straggle_dE_keV"] >= 0.0)
    # Some flights should actually sample a nonzero loss at these operating
    # conditions (25 keV graphite, order-one inelastic events per flight per
    # slice A) or this test would pass vacuously.
    assert np.any(on["straggle_dE_keV"] > 0.0)


def test_grooved_straggling_on_never_perturbs_existing_draws():
    groove = GrooveSpec(spacing_ang=1.0e4, depth_ang=1.0e3, tilt_polar_rad=0.2)
    off = _run({}, groove=groove, straggling=False)
    on = _run({}, groove=groove, straggling=True)
    _assert_non_straggle_fields_equal(off, on)
    assert "straggle_dE_keV" in on
    assert np.any(on["straggle_dE_keV"] > 0.0)


def _offline_straggle_dE_keV(out, seed, Ne, element, n_atoms_per_ang3):
    """Recompute ``straggle_dE_keV`` from a run's own recorded segments.

    Independent of which core produced ``out``: only the documented key
    scheme, the run's own per-segment ``(E_start, L_ang)``, and each
    segment's position within its electron's own segment list (the flight
    index; substep is always 0 at ``max_dE_frac=0``) are used. Matching
    ``out["straggle_dE_keV"]`` bit-for-bit is therefore a direct test that the
    core addressed the sampler with the key its own docstring claims, not an
    incidental one.
    """
    sk = stream_keys(seed, Ne)
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(
        [(element, n_atoms_per_ang3)]
    )
    elec_id = out["electron_id"]
    E_start = out["E_start_keV"]
    L_ang = out["L_ang"]
    recomputed = np.zeros(Ne)
    for e in range(Ne):
        mask = elec_id == e
        urban_key = np.uint64(_urban_stream_key_scalar(np.uint64(sk[e])))
        E_e = E_start[mask]
        L_e = L_ang[mask]
        total = 0.0
        for flight, (E_j, step_j) in enumerate(zip(E_e, L_e, strict=True)):
            flight_key = np.uint64(
                _urban_flight_key_scalar(urban_key, np.int64(flight), np.int64(0))
            )
            loss, _counter = _urban_sample_compound_keV(
                Z_arr,
                J_arr,
                k_arr,
                coeff_arr,
                E_cross_arr,
                0.0,
                float(E_j),
                float(step_j),
                flight_key,
                np.uint64(0),
            )
            total += loss
        recomputed[e] = total
    return recomputed


@pytest.mark.parametrize("name", sorted(CORE_CONFIGS))
def test_straggle_dE_keV_reproduces_from_seed_electron_flight(name):
    """Property 3. ``straggle_dE_keV`` is exactly reproducible offline from
    ``(seed, electron, flight, substep)`` via the documented key scheme, on
    every host core -- bit-for-bit, since this recomputation runs the exact
    same CPU arithmetic the core did."""
    core_kwargs = CORE_CONFIGS[name]
    out = _run(core_kwargs, straggling=True)
    recomputed = _offline_straggle_dE_keV(
        out,
        BASE_KWARGS["seed"],
        BASE_KWARGS["Ne"],
        BASE_KWARGS["element"],
        BASE_KWARGS["n_atoms_per_ang3"],
    )
    np.testing.assert_array_equal(recomputed, out["straggle_dE_keV"])


def test_grooved_straggle_dE_keV_reproduces_from_seed_electron_flight():
    groove = GrooveSpec(spacing_ang=1.0e4, depth_ang=1.0e3, tilt_polar_rad=0.2)
    out = _run({}, groove=groove, straggling=True)
    recomputed = _offline_straggle_dE_keV(
        out,
        BASE_KWARGS["seed"],
        BASE_KWARGS["Ne"],
        BASE_KWARGS["element"],
        BASE_KWARGS["n_atoms_per_ang3"],
    )
    np.testing.assert_array_equal(recomputed, out["straggle_dE_keV"])


def test_straggling_key_domain_is_seed_and_electron_only():
    """The per-electron straggling key is a pure function of ``(seed,
    electron)`` via ``stream_keys(seed, Ne)`` -- the same array the
    lockstep/grooved cores receive as ``stragg_stream_keys`` and the
    per-electron cores already carried as ``stream_key``/``d_keys`` -- so
    every core addresses the same straggling stream for the same electron
    regardless of its own, otherwise unrelated, RNG design (a shared
    ``Generator`` for lockstep/grooved, a counter stream per electron
    otherwise)."""
    seed, Ne = 999, 17
    sk = stream_keys(seed, Ne)
    for e in (0, 1, Ne - 1):
        first = np.uint64(_urban_stream_key_scalar(np.uint64(sk[e])))
        second = np.uint64(_urban_stream_key_scalar(np.uint64(stream_keys(seed, Ne)[e])))
        assert first == second


def test_straggling_default_is_off():
    """`simulate_trajectories`'s ``straggling`` parameter defaults False."""
    import inspect

    sig = inspect.signature(simulate_trajectories)
    assert sig.parameters["straggling"].default is False
