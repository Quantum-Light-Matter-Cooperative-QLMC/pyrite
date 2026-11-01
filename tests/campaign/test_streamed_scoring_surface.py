"""Public opt-in for live streamed scoring: ``KernelTuning.stream_block_electrons`` (#865).

Streaming equals the default ``simulate`` up to summation order and never
enters case identity; every other entry point, and every unsupported case,
fails before transport.
"""

import pickle
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite import phases
from pyrite.campaign.profiles import case_content_key
from pyrite.detectors import EnergyBins
from pyrite.execution import LocalExecutor
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.transport_options import transport_tuning

BINS = EnergyBins(cxr=np.linspace(250.0, 320.0, 15), brem=np.arange(1000.0, 30001.0, 1000.0))


def _numerics(block=None, core="per-electron", **kwargs):
    return pr.Numerics(
        cxr_trials=12,
        brem_trials=16,
        energy_bins=BINS,
        execution=cfg.KernelTuning(transport_core=core, stream_block_electrons=block),
        **kwargs,
    )


def _scene(detector=None):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector() if detector is None else detector,
    )


@pytest.mark.parametrize("value", [0, -3, 2.0, True])
def test_invalid_block_is_rejected(value):
    with pytest.raises(ValueError, match="stream_block_electrons must be a positive integer"):
        cfg.KernelTuning(stream_block_electrons=value)


def test_block_stays_out_of_case_identity_and_pickles():
    tuning = cfg.KernelTuning(stream_block_electrons=5)
    assert pickle.loads(pickle.dumps(tuning)) == tuning
    old = cfg.KernelTuning()
    state = dict(old.__dict__)
    del state["stream_block_electrons"]
    restored = cfg.KernelTuning.__new__(cfg.KernelTuning)
    restored.__setstate__(state)
    assert restored == old
    plain = pr.api.build_case(_scene(), _numerics())
    streamed = pr.api.build_case(_scene(), _numerics(5))
    assert case_content_key(plain) == case_content_key(streamed)


@pytest.mark.parametrize("block", [5, 100])
def test_streamed_simulate_matches_the_default(block):
    reference = pr.simulate(_scene(), numerics=_numerics())
    streamed = pr.simulate(_scene(), numerics=_numerics(block))
    assert streamed.case == reference.case
    pairs = {
        "cxr": (streamed.cxr.incoherent, reference.cxr.incoherent),
        "characteristic": (streamed.characteristic, reference.characteristic),
        "brem": (streamed.brem, reference.brem),
    }
    for name, (got, want) in pairs.items():
        np.testing.assert_array_equal(got.photon_energy_eV, want.photon_energy_eV, err_msg=name)
        np.testing.assert_allclose(got.density, want.density, rtol=1e-12, atol=0, err_msg=name)
    assert np.any(reference.brem.density > 0.0)


def _no_transport(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", refuse)


def test_unsupported_case_fails_before_transport(monkeypatch):
    _no_transport(monkeypatch)
    with pytest.raises(ValueError, match="cannot stream"):
        pr.simulate(_scene(), numerics=_numerics(5, temporal_profile=True))


def test_other_entry_points_fail_before_transport(monkeypatch):
    _no_transport(monkeypatch)
    with pytest.raises(ValueError, match="supported only by pyrite.simulate"):
        phases.simulate_transport(replace(_scene(), numerics=_numerics(5)))
    # Planar, directional and photon paths enter transport through this guard.
    with pytest.raises(ValueError, match="supported only by pyrite.simulate"):
        with transport_tuning(_numerics(5)):
            pass
    with pytest.raises(ValueError, match="executor does not support"):
        pr.simulate(_scene(), numerics=_numerics(5, core="auto"), executor=LocalExecutor())
