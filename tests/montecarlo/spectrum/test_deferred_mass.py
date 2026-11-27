"""Per-electron line mass summed off the main thread (#1481): bitwise the serial sums."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum._deferred_mass import (
    PENDING_KEY,
    _unit_into_total,
    defer_electron_mass,
    settle_electron_mass,
)


def _units(n_electrons=37, n_units=6, seed=1481):
    rng = np.random.default_rng(seed)
    units = []
    for k in range(n_units):
        size = int(rng.integers(1, 5000))
        # Grouped and shuffled ids, few or many electrons, wide dynamic range.
        index = rng.integers(0, n_electrons if k % 2 else 3, size)
        if k % 3 == 0:
            index = np.sort(index)
        mass = rng.uniform(0.0, 1.0, size) * 10.0 ** rng.uniform(-8, 8, size) + 1e-30
        units.append((index.astype(np.int32 if k % 2 else np.int64), mass))
    return units


def _serial(units, n, seed=None):
    """The pre-#1481 sequence: ``(seed or 0.0) + bincount(u_1) + bincount(u_2) ...``."""
    total = 0.0 if seed is None else seed
    for index, mass in units:
        total = total + np.bincount(index, weights=mass, minlength=n)
    return total


@pytest.mark.parametrize("seeded", [False, True])
def test_unit_kernel_equals_serial_bincount_sequence(seeded):
    n = 37
    units = _units(n)
    seed = np.random.default_rng(7).uniform(0.0, 3.0, n) if seeded else None
    total = np.zeros(n) if seed is None else seed.copy()
    scratch, touched = np.zeros(n), np.zeros(n, dtype=np.bool_)
    for index, mass in units:
        assert _unit_into_total(index, mass, total, scratch, touched) == 0
    np.testing.assert_array_equal(total, _serial(units, n, seed))
    assert not scratch.any() and not touched.any()


def test_unit_kernel_flags_out_of_range_ids():
    total, scratch, touched = np.zeros(4), np.zeros(4), np.zeros(4, dtype=np.bool_)
    assert _unit_into_total(np.array([0, 4]), np.ones(2), total, scratch, touched) == -2
    total, scratch, touched = np.zeros(4), np.zeros(4), np.zeros(4, dtype=np.bool_)
    assert _unit_into_total(np.array([-1]), np.ones(1), total, scratch, touched) == -1


def test_host_arrays_are_not_deferred():
    audit = {"n_electrons": 3}
    assert not defer_electron_mass(audit, np.array([0, 1]), np.ones(2))
    assert audit == {"n_electrons": 3}


def test_settle_without_pending_is_a_no_op():
    audit = {"electron_mass": 1.0}
    settle_electron_mass(audit)
    settle_electron_mass(None)
    assert audit == {"electron_mass": 1.0}


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CUDA or ROCm device required")
    except Exception:  # pragma: no cover - depends on the runtime
        pytest.skip("CUDA or ROCm device required")
    return cp


@pytest.mark.parametrize("seeded", [False, True])
def test_device_deferral_equals_ordered_bincount_sequence(seeded):
    """Units queued from a non-default stream; repeated runs are bitwise equal too."""
    from pyrite.montecarlo.spectrum._ordered_bincount import ordered_bincount

    cp = _cupy()
    n = 37
    units = _units(n)
    seed = np.random.default_rng(7).uniform(0.0, 3.0, n) if seeded else None
    results = []
    for _ in range(2):
        audit = {"n_electrons": n}
        if seed is not None:
            audit["electron_mass"] = cp.asarray(seed)
        with cp.cuda.Stream(non_blocking=True):
            for index, mass in units:
                # A device temporary freed right after queueing, as in production.
                assert defer_electron_mass(audit, cp.asarray(index), cp.asarray(mass) * 1.0)
            assert PENDING_KEY in audit and "electron_mass" not in audit
            settle_electron_mass(audit)
        assert PENDING_KEY not in audit
        results.append(cp.asnumpy(audit["electron_mass"]))
    reference = 0.0 if seed is None else cp.asarray(seed)
    for index, mass in units:
        reference = reference + ordered_bincount(cp.asarray(index), cp.asarray(mass), n)
    np.testing.assert_array_equal(results[0], cp.asnumpy(reference))
    np.testing.assert_array_equal(results[1], results[0])


def test_device_deferral_reports_bad_ids_at_settle():
    cp = _cupy()
    audit = {"n_electrons": 3}
    defer_electron_mass(audit, cp.asarray([0, 5]), cp.ones(2))
    with pytest.raises(ValueError, match="outside the 3 line electrons"):
        settle_electron_mass(audit)
    audit = {"n_electrons": 3}
    defer_electron_mass(audit, cp.asarray([0, 5]), cp.ones(2))
    settle_electron_mass(audit, failed=True)  # abandons quietly
    assert "electron_mass" not in audit and PENDING_KEY not in audit
