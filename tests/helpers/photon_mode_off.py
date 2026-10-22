"""Exercise #341 diagnostics without enabling photon MC on production runs."""

from copy import deepcopy

import numpy as np
import pytest


def exercise_photon_reference(seed=341):
    """Run real weighted source/queue/tally code in vacuum; no external tables."""
    from pyrite.montecarlo.runner.photon_emission import _score_exits
    from pyrite.montecarlo.transport.photon_histories import (
        PhotonAperture,
        track_photon_histories,
    )
    from pyrite.montecarlo.transport.photon_sources import sample_continuous_photons

    segments = dict(
        L_ang=np.array([10.0]),
        r_mid=np.zeros((1, 3)),
        v_hat=np.array([[0.0, 0.0, 1.0]]),
        E_keV=np.array([20.0]),
        elec_id=np.array([0]),
    )
    source = sample_continuous_photons(
        segments,
        lambda index, energy, direction: np.full(index.size, 1e-8),
        n_samples=128,
        n_primaries=1,
        seed=seed,
        cutoff_eV=1000,
        ceiling_eV=10_000,
    )
    histories = track_photon_histories(
        *source.track_args,
        [],
        seed=seed,
        photon_cutoff_eV=1000,
        electron_cutoff_eV=1000,
        source_kind=source.source_kind,
        aperture=PhotonAperture((0, 0, 1e6), (1, 0, 0), (0, 1, 0), 1e6, 1e6),
    )
    total, direct, redistributed, _, _ = _score_exits(histories, [1000, 10_000], 1)
    assert total.sum() > 0
    np.testing.assert_array_equal(total, direct)
    np.testing.assert_array_equal(redistributed, [0])


def _production_run(case, monkeypatch):
    """Spy on real transport; fail if mode-off dispatch reaches photon MC."""
    import pyrite.photon_transport as scene
    from pyrite.campaign.profiles import case_content_key
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import photon_emission
    from pyrite.montecarlo.transport import photon_coupling, photon_histories, photon_sources

    transports = []
    real_transport = runner._transport_case

    def capture(*args, **kwargs):
        transport = real_transport(*args, **kwargs)
        transports.append(transport)
        return transport

    def disabled(*args, **kwargs):
        pytest.fail("production mode-off path entered the photon-MC reference")

    with monkeypatch.context() as guard:
        guard.setattr(runner, "_transport_case", capture)
        guard.setattr(runner, "_TIMING", False)
        for module, names in (
            (scene, ["simulate_photon_continuum"]),
            (photon_emission, ["score_bremslib_continuum"]),
            (photon_sources, ["sample_continuous_photons", "sample_bremslib_continuum"]),
            (photon_histories, ["track_photon_histories"]),
            (photon_coupling, ["track_coupled_histories"]),
        ):
            for name in names:
                guard.setattr(module, name, disabled)
        output = runner.run_case(case, transport_core="lockstep")
        identity = case_content_key(case, xsgen_tables={})
    assert len(transports) == 1
    return transports[0], output, identity


def assert_photon_mode_off_bitwise(case, monkeypatch):
    """Return production output only after exact before/after compatibility."""
    from tests.helpers.physics_baseline import transport_signature

    before_transport, before, before_identity = _production_run(case, monkeypatch)
    before_signature = transport_signature(before_transport["segs"])
    before = deepcopy(before)
    exercise_photon_reference(seed=int(case["seed"]))
    transport, after, identity = _production_run(case, monkeypatch)
    assert identity == before_identity
    assert transport_signature(transport["segs"]) == before_signature
    assert after.keys() == before.keys()
    for name, expected in before.items():
        if isinstance(expected, np.ndarray):
            assert after[name].shape == expected.shape, name
            assert after[name].dtype == expected.dtype, name
            assert after[name].tobytes() == expected.tobytes(), name
        else:
            assert after[name] == expected, name
    return transport, after
