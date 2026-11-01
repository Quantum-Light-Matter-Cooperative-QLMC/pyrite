"""Electron-block transport and the stream prefix-stability it rests on (#361).

Adaptive electron counts extend a run block by block, so every random input
must be a pure function of ``(seed, electron, draw)``: the block
``[start, stop)`` has to draw exactly its slice of a ``[0, stop)`` run. These
tests pin that for each stream the per-electron/CUDA path consumes, then pin
the consequence end to end -- the block driver's joined output equals one
fixed-N call bit for bit, for several block sizes including non-divisors.

Seeds are arbitrary fixed integers; every comparison is exact (bit-for-bit),
because the claim is equality of addresses, not closeness of distributions.
"""

from dataclasses import asdict

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.block_transport import (
    electron_blocks,
    transport_electron_blocks,
)
from pyrite.montecarlo.transport import _sample_bunch_offsets, simulate_trajectories, stream_keys
from pyrite.montecarlo.transport._jit_radiative import radiative_stream_keys
from pyrite.montecarlo.transport.beam_entry import initial_beam_positions, initial_energies_keV
from pyrite.montecarlo.transport.hard_inelastic import hard_stream_keys
from pyrite.montecarlo.transport.kinematics import counter_normals, counter_uniforms
from pyrite.montecarlo.transverse import (
    TransverseDistribution,
    resolve_transverse_distribution,
    resolved_from_mapping,
    sample_transverse,
)

U64_MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15

_TWISS = asdict(
    resolve_transverse_distribution(
        TransverseDistribution(
            normalized_emittance_x_mm_mrad=1.0, beta_twiss_x_m=0.5, alpha_twiss_x=-0.8
        ),
        energy_keV=30.0,
    )
)
_MICROTRAIN = dict(
    kind="microtrain",
    envelope_rms_fs=200.0,
    microbunch_rms_fs=5.0,
    spacing_fs=40.0,
    timing_jitter_fs=3.0,
    modulation_depth=0.5,
)


def _reference_splitmix64(x):
    """Independent pure-Python SplitMix64 finalizer."""
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return x ^ (x >> 31)


# ---- stream addresses ---------------------------------------------------------


@pytest.mark.parametrize("keys", [stream_keys, hard_stream_keys, radiative_stream_keys])
@pytest.mark.parametrize(("start", "stop"), [(0, 7), (5, 64), (63, 64), (64, 64)])
def test_offset_keys_equal_the_matching_slice(keys, start, stop):
    np.testing.assert_array_equal(keys(91, stop - start, start=start), keys(91, stop)[start:stop])


def test_counter_uniforms_match_an_independent_reference():
    root, start = 0x1234_5678_9ABC_DEF0, 3
    got = counter_uniforms(root, 4, 3, start=start)
    for row, e in enumerate(range(start, start + 4)):
        key = _reference_splitmix64((root + GOLDEN * (e + 1)) & U64_MASK)
        for c in range(3):
            z = _reference_splitmix64((key + GOLDEN * (c + 1)) & U64_MASK)
            assert got[row, c] == ((z >> 12) + 0.5) * 2.0**-52
    assert np.all((got > 0.0) & (got < 1.0))


def test_counter_normals_are_standard_normal():
    """Inverse-CDF normals: moments of N(0, 1) within 5 sigma of their MC error."""
    n = 400_000
    z = counter_normals(0xC0FFEE, n, 1)[:, 0]
    assert abs(z.mean()) < 5.0 / np.sqrt(n)
    assert abs(z.var() - 1.0) < 5.0 * np.sqrt(2.0 / n)
    assert abs(np.mean(z**4) - 3.0) < 5.0 * np.sqrt(96.0 / n)
    # The quantile of the extreme representable uniforms bounds the tails.
    assert np.abs(z).max() < 8.3


# ---- beam and bunch inputs ----------------------------------------------------


@pytest.mark.parametrize("beam", [dict(beam_fwhm_mm=0.1, beam_fwhm_y_mm=0.3), {"twiss": True}])
def test_transverse_entry_blocks_equal_the_slice(beam):
    kw = dict(
        transverse_distribution=_TWISS if "twiss" in beam else None,
        beam_fwhm_mm=beam.get("beam_fwhm_mm"),
        beam_fwhm_y_mm=beam.get("beam_fwhm_y_mm"),
        tilt_polar_rad=0.3,
        tilt_azim_rad=0.1,
        groove=None,
    )
    pos, slopes, _ = initial_beam_positions(5, 50, **kw)
    blk_pos, blk_slopes, _ = initial_beam_positions(5, 20, start=17, **kw)
    np.testing.assert_array_equal(blk_pos, pos[17:37])
    if slopes is not None:
        for whole, part in zip(slopes, blk_slopes, strict=True):
            np.testing.assert_array_equal(part, whole[17:37])


def test_twiss_draw_of_an_electron_ignores_the_electron_count():
    resolved = resolved_from_mapping(_TWISS)
    small = sample_transverse(resolved, 10, 3)
    large = sample_transverse(resolved, 1000, 3)
    for a, b in zip(small, large, strict=True):
        np.testing.assert_array_equal(a, b[:10])


@pytest.mark.parametrize(
    "bunch",
    [
        dict(bunch_length_fs=12.0, long_shape="gaussian"),
        dict(bunch_length_fs=12.0, long_shape="uniform"),
        dict(longitudinal_distribution=_MICROTRAIN),
    ],
)
def test_raw_bunch_draws_are_prefix_stable(bunch):
    """Only the population centroid depends on N; each raw draw does not."""
    kw = dict(bunch_length_fs=None, long_shape="gaussian", long_offsets_fs=None) | bunch
    small = _sample_bunch_offsets(40, seed=8, **kw)
    large = _sample_bunch_offsets(400, seed=8, **kw)
    shift = large[:40] - small
    np.testing.assert_allclose(shift, shift[0], rtol=0.0, atol=1e-9 * np.abs(large).max())
    assert abs(small.mean()) < 1e-9 * np.abs(small).max()


# ---- block driver -------------------------------------------------------------

_BASE = dict(
    E0_keV=30.0,
    thickness_ang=5.0e4,
    element="Si",
    n_atoms_per_ang3=0.04996,
    seed=11,
    transport_core="per-electron",
    crystal_width_mm=2e-3,
    crystal_height_mm=2e-3,
    energy_spread_frac=1e-3,
    straggling=True,
    energy_model="midpoint",
)


def _assert_identical(got, ref):
    assert set(got) == set(ref)
    for key, value in ref.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value, err_msg=key)
        elif isinstance(value, int | float):
            assert got[key] == value, key


@pytest.mark.parametrize(
    ("elastic_model", "beam", "bunch"),
    [
        ("elsepa", dict(beam_fwhm_mm=1e-3), dict(bunch_length_fs=10.0)),
        ("sr", dict(beam_fwhm_mm=1e-3), dict(bunch_length_fs=10.0)),
        (
            "elsepa",
            dict(transverse_distribution=_TWISS),
            dict(longitudinal_distribution=_MICROTRAIN),
        ),
    ],
)
def test_block_transport_equals_one_fixed_n_call(elastic_model, beam, bunch):
    """``"sr"`` takes the per-electron LUT driver, ``"elsepa"`` the exact one."""
    n = 50
    kw = _BASE | beam | {"elastic_model": elastic_model}
    ref = simulate_trajectories(Ne=n, **kw, **bunch)
    assert ref["n_missed"] > 0  # the finite footprint is exercised

    # The population's table range, as the runner's block path supplies it.
    energies = initial_energies_keV(kw["E0_keV"], n, kw["seed"], kw["energy_spread_frac"])
    energy_range = (5.0, float(energies.max()))  # 5 keV: the default cutoff
    for block in (7, 16, 50, 64):

        def simulate_block(start, stop):
            return simulate_trajectories(
                Ne=stop - start, _electron_start=start, _energy_range_keV=energy_range, **kw
            )

        got = transport_electron_blocks(simulate_block, n, block, seed=kw["seed"], bunch=bunch)
        _assert_identical(got, ref)


def test_runner_blocks_equal_fixed_n_spectra():
    """End to end: default shell + coupled radiative modes, spot, spread, bunch,
    brem as a prefix population of the one transport."""
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0, energy_spread_frac=1e-3, bunch_length_fs=5.0),
        tilt_deg=30.0,
        n_electrons=12,
        n_electrons_brem=7,
    )
    case = build_cases(sweep)[0]
    ref_tp = runner._transport_case(case, transport_core="per-electron")
    ref = runner._spectrum_case(case, ref_tp)
    for block in (5, 12):
        tp = runner._transport_case(case, transport_core="per-electron", block_electrons=block)
        _assert_identical(tp["segs"], ref_tp["segs"])
        _assert_identical(runner._spectrum_case(case, tp), ref)


def test_block_driver_rejects_the_lockstep_core():
    case = build_cases(Sweep(material="hopg", n_electrons=4, n_electrons_brem=2))[0]
    with pytest.raises(ValueError, match="lockstep"):
        runner._transport_case(case, transport_core="lockstep", block_electrons=2)
    with pytest.raises(ValueError, match="lockstep"):
        simulate_trajectories(Ne=2, _electron_start=2, **(_BASE | {"transport_core": "lockstep"}))


def test_in_block_bunch_sampling_is_rejected():
    with pytest.raises(ValueError, match="bunch"):
        simulate_trajectories(Ne=2, _electron_start=2, bunch_length_fs=1.0, **_BASE)


def test_electron_blocks_cover_the_range_once():
    assert electron_blocks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert electron_blocks(8, 8) == [(0, 8)]
    assert electron_blocks(0, 3) == []
    with pytest.raises(ValueError):
        electron_blocks(5, 0)
