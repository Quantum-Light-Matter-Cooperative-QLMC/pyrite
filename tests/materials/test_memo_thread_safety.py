"""The LRU memos tolerate concurrent hits and evictions (#1423).

``OrderedDict.get`` followed by ``move_to_end`` raises ``KeyError`` when another
thread evicts the key in between. A tiny capacity forces constant eviction.
"""

import sys
import threading

import numpy as np
import pytest

from pyrite.materials import atomic, photon_cross_sections

THREADS = 8
ROUNDS = 600


@pytest.fixture(autouse=True)
def _tight_switch_interval():
    """Preempt every few microseconds so a get/evict interleaving is likely."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


def _hammer(call, grids):
    errors: list[BaseException] = []
    start = threading.Barrier(THREADS, timeout=30)

    def work(seed):
        order = np.random.default_rng(seed).integers(0, len(grids), ROUNDS)
        start.wait()
        try:
            for index in order:
                call(grids[index])
        except BaseException as error:  # noqa: BLE001 - the test reports any failure
            errors.append(error)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return errors


def _grids(count):
    return [np.linspace(2000.0, 20000.0, 8) + 11.0 * k for k in range(count)]


def test_total_attenuation_memo_survives_concurrent_eviction(monkeypatch):
    monkeypatch.setattr(photon_cross_sections, "_TOTAL_MEMO_MAX", 2)
    grids = _grids(6)
    expected = [photon_cross_sections.total_photon_cross_section_ang2("Si", g) for g in grids]
    errors = _hammer(
        lambda g: photon_cross_sections.total_photon_cross_section_ang2("Si", g), grids
    )
    assert errors == []
    for g, want in zip(grids, expected, strict=True):
        np.testing.assert_array_equal(
            photon_cross_sections.total_photon_cross_section_ang2("Si", g), want
        )


def test_henke_memo_survives_concurrent_eviction(monkeypatch):
    monkeypatch.setattr(atomic, "_HENKE_MEMO_MAX", 2)
    grids = _grids(6)
    expected = [atomic.henke_dispersion("Si", g) for g in grids]
    errors = _hammer(lambda g: atomic.henke_dispersion("Si", g), grids)
    assert errors == []
    for g, (fp, fpp) in zip(grids, expected, strict=True):
        got = atomic.henke_dispersion("Si", g)
        np.testing.assert_array_equal(got[0], fp)
        np.testing.assert_array_equal(got[1], fpp)
