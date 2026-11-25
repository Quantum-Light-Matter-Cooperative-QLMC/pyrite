"""Prefetched block beam entry is bit-identical to serial sampling (#1454).

The counter-uniform kernel must equal the historical NumPy expression, and a
block run served by :class:`BeamEntryPrefetch` must return exactly the slices
of one population draw for every block split, hit or miss.
"""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.transport.beam_entry import initial_beam_positions
from pyrite.montecarlo.transport.beam_prefetch import BeamEntryPrefetch, sample_beam_positions
from pyrite.montecarlo.transport.kinematics import counter_uniforms, stream_keys
from pyrite.montecarlo.transverse import TransverseDistribution, resolve_transverse_distribution

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _numpy_counter_uniforms(root, n, n_draws, start):
    """The pre-#1454 vectorized expression, kept verbatim as the oracle."""
    keys = stream_keys(root, n, start=start)[:, None]
    counters = np.arange(n_draws, dtype=np.uint64)[None, :]
    x = keys + _GOLDEN * (counters + np.uint64(1))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(12)).astype(np.float64) + 0.5) * (1.0 / 4503599627370496.0)


@pytest.mark.parametrize(
    ("root", "n", "n_draws", "start"),
    [(0, 0, 2, 0), (7, 1000, 1, 0), (2**64 - 1, 513, 4, 12345), (0xDEADBEEF, 64, 2, 2**62)],
)
def test_counter_uniforms_kernel_equals_numpy_expression(root, n, n_draws, start):
    got = counter_uniforms(root, n, n_draws, start=start)
    np.testing.assert_array_equal(got, _numpy_counter_uniforms(root, n, n_draws, start))
    assert got.shape == (n, n_draws)


def _beam(kind):
    kw = dict(
        transverse_distribution=None,
        beam_fwhm_mm=None,
        beam_fwhm_y_mm=None,
        tilt_polar_rad=0.3,
        tilt_azim_rad=0.1,
        groove=None,
        counter_groove_phase=False,
        E0_keV=30.0,
        energy_spread_frac=None,
        bunch_front_tilt=None,
    )
    if kind == "spot":
        kw |= dict(beam_fwhm_mm=0.1, beam_fwhm_y_mm=0.3)
    else:
        kw["transverse_distribution"] = asdict(
            resolve_transverse_distribution(
                TransverseDistribution(
                    normalized_emittance_x_mm_mrad=1.0, beta_twiss_x_m=0.5, alpha_twiss_x=-0.8
                ),
                energy_keV=30.0,
            )
        )
    return kw


def _run_blocks(n, block, kw, core="cuda"):
    with BeamEntryPrefetch(n) as prefetch:
        parts = []
        for start in range(0, n, block):
            p = SimpleNamespace(seed=11, Ne=min(block, n - start), gdf_source=None)
            parts.append(
                prefetch.run(
                    sample_beam_positions, initial_beam_positions, p, core, start=start, **kw
                )  # fmt: skip
            )
    return parts, prefetch


@pytest.mark.parametrize("kind", ["spot", "twiss"])
@pytest.mark.parametrize("block", [1, 7, 32, 50, 100])
def test_prefetched_blocks_equal_population_slices(kind, block):
    kw = _beam(kind)
    n = 100
    pos, slopes, record = initial_beam_positions(11, n, **kw)
    parts, prefetch = _run_blocks(n, block, kw)
    for i, (b_pos, b_slopes, b_record) in enumerate(parts):
        lo, hi = i * block, min(n, (i + 1) * block)
        np.testing.assert_array_equal(b_pos, pos[lo:hi])
        assert b_record == record
        if slopes is not None:
            for whole, part in zip(slopes, b_slopes, strict=True):
                np.testing.assert_array_equal(part, whole[lo:hi])
    assert prefetch.hits == len(parts) - 1
    assert prefetch.misses == 1


def test_changed_arguments_miss_and_sample_serially():
    kw = _beam("spot")
    with BeamEntryPrefetch(40) as prefetch:
        p = SimpleNamespace(seed=11, Ne=20, gdf_source=None)
        prefetch.run(sample_beam_positions, initial_beam_positions, p, "cuda", start=0, **kw)
        changed = dict(kw, tilt_polar_rad=0.4)
        got = prefetch.run(
            sample_beam_positions, initial_beam_positions, p, "cuda", start=20, **changed
        )
    np.testing.assert_array_equal(got[0], initial_beam_positions(11, 20, start=20, **changed)[0])
    assert (prefetch.hits, prefetch.misses) == (0, 2)


@pytest.mark.parametrize("core", ["per-electron", "lockstep"])
def test_host_cores_never_prefetch(core):
    _, prefetch = _run_blocks(60, 20, _beam("spot"), core=core)
    assert (prefetch.hits, prefetch.misses) == (0, 0)


def test_point_beam_is_not_prefetched():
    kw = _beam("spot") | dict(beam_fwhm_mm=None, beam_fwhm_y_mm=None)
    _, prefetch = _run_blocks(60, 20, kw)
    assert (prefetch.hits, prefetch.misses) == (0, 0)


def test_without_prefetcher_samples_directly():
    kw = _beam("spot")
    p = SimpleNamespace(seed=11, Ne=10, gdf_source=None)
    got = sample_beam_positions(initial_beam_positions, p, "cuda", start=5, **kw)
    np.testing.assert_array_equal(got[0], initial_beam_positions(11, 10, start=5, **kw)[0])
