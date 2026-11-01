"""Whole-history CPU/CUDA accounting, lineage and no-host-sampler contract."""

import numpy as np
import pytest

from pyrite.montecarlo.transport import photon_histories as histories


@pytest.fixture
def cp():
    from pyrite._backend import BackendUnavailableError, select_cupy_backend

    try:
        return select_cupy_backend().xp
    except BackendUnavailableError as error:
        pytest.skip(str(error))


def _batch(n, energy):
    return (
        np.zeros((n, 3)),
        np.tile([0, 0, 1.0], (n, 1)),
        np.full(n, energy),
        np.linspace(0.5, 1.5, n),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
    )


def _run(batch, layers, **options):
    return histories.track_photon_histories(
        *batch,
        layers,
        seed=341_905,
        photon_cutoff_eV=1000,
        electron_cutoff_eV=1000,
        max_generations=128,
        **options,
    )


def _compare(expected, actual):
    for group in ("exits", "charged", "interactions", "ledger"):
        left, right = getattr(expected, group), getattr(actual, group)
        assert left.keys() == right.keys()
        for key in left:
            if left[key].dtype.kind in "biu":
                np.testing.assert_array_equal(right[key], left[key], err_msg=f"{group}.{key}")
            else:
                np.testing.assert_allclose(
                    right[key], left[key], rtol=3e-11, atol=1e-5, err_msg=f"{group}.{key}"
                )
    residual = histories.ledger_residual_eV(actual.ledger)
    assert np.all(np.abs(residual) <= 1e-12 * actual.ledger["incident_eV"])


@pytest.mark.parametrize(
    "element,density,energy", [("C", 0.1, 5e4), ("W", 0.063, 2e4), ("W", 0.063, 6e6)]
)
def test_device_history_matches_cpu_without_host_sampler_fallback(
    cp, monkeypatch, element, density, energy
):
    batch = _batch(64, energy)
    layers = [(0, 2e7, [(element, density)]), (3e7, 4e7, [("Si", 0.05)])]
    aperture = histories.PhotonAperture((0, 0, -1e7), (1, 0, 0), (0, -1, 0), 1e8, 1e8)
    kwargs = dict(aperture=aperture, width_ang=1e8, height_ang=1e8)
    expected = _run(batch, layers, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA histories invoked a CPU flight/final-state sampler")

    for name in (
        "scene_first_interactions",
        "sample_rayleigh",
        "sample_compton",
        "sample_photoelectric",
        "sample_pair",
    ):
        monkeypatch.setattr(histories, name, forbidden)
    actual = _run(batch, layers, backend="cuda", **kwargs)
    _compare(expected, actual)


def test_device_queue_refill_and_batch_controls_preserve_histories(cp, monkeypatch):
    from pyrite.montecarlo.transport import photon_cuda_events as events

    batch = _batch(19, 2e5)
    layers = [(0, 1e8, [("W", 0.063)])]
    normal = _run(batch, layers, backend="cuda")
    monkeypatch.setattr(events, "_INITIAL_DRAW_WIDTH", 2)
    monkeypatch.setattr(events, "_INITIAL_CAPACITY", 1)
    refilled = _run(batch, layers, backend="cuda", cuda_batch_size=3)
    _compare(normal, refilled)
    order = np.random.default_rng(341_906).permutation(19)
    reordered = _run([a[order] for a in batch], layers, backend="cuda", cuda_batch_size=1)
    # History identifiers and keys own order; compare canonically sorted rows.
    for name in ("exits", "charged", "interactions"):
        left, right = getattr(normal, name), getattr(reordered, name)
        for key in left:
            lo = np.lexsort((left["key"], left["history_id"]))
            ro = np.lexsort((right["key"], right["history_id"]))
            if left[key].dtype.kind in "biu":
                np.testing.assert_array_equal(right[key][ro], left[key][lo])
            else:
                np.testing.assert_allclose(right[key][ro], left[key][lo], rtol=3e-11, atol=1e-5)
    for key in normal.ledger:
        np.testing.assert_array_equal(reordered.ledger[key], normal.ledger[key])


def test_cuda_history_backend_is_explicit_even_for_empty_batches(monkeypatch):
    from pyrite import _backend

    def unavailable():
        raise _backend.BackendUnavailableError("test: unavailable")

    monkeypatch.setattr(_backend, "select_cupy_backend", unavailable)
    with pytest.raises(_backend.BackendUnavailableError, match="unavailable"):
        _run(_batch(0, 1e5), [], backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        _run(_batch(0, 1e5), [], backend="auto")
