"""Public opt-in for live streamed scoring: ``KernelTuning.stream_block_electrons`` (#865).

Streaming equals the default ``simulate`` up to summation order and never
enters case identity; every other entry point, and every unsupported case,
fails before transport.
"""

import pickle

import numpy as np
import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite.api import build_case
from pyrite.campaign.profiles import case_content_key
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins
from pyrite.execution import LocalExecutor
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.transport_options import transport_tuning
from pyrite.transport import simulate_trajectories

BINS = EnergyBins(cxr=np.linspace(250.0, 320.0, 15), brem=np.arange(1000.0, 30001.0, 1000.0))


def _numerics(block=None, core="per-electron", bins=BINS, **kwargs):
    return pr.Numerics(
        sampling=FixedSampling(cxr_histories=12, brem_histories=16),
        energy_bins=bins,
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
    plain = build_case(_scene(), _numerics())
    streamed = build_case(_scene(), _numerics(5))
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
        scene = _scene()
        simulate_trajectories(scene.beam, scene.target, numerics=_numerics(5))
    # Planar, directional and photon paths enter transport through this guard.
    with pytest.raises(ValueError, match="supported only by pyrite.simulate"):
        with transport_tuning(_numerics(5)):
            pass
    with pytest.raises(ValueError, match="executor does not support"):
        pr.simulate(_scene(), numerics=_numerics(5, core="auto"), executor=LocalExecutor())


def _memory_numerics(target):
    return pr.Numerics(
        sampling=FixedSampling(cxr_histories=12, brem_histories=16),
        energy_bins=BINS,
        execution=cfg.KernelTuning(transport_core="per-electron", stream_memory_mib=target),
    )


@pytest.mark.parametrize("value", [0, -3, 2.0, True])
def test_invalid_memory_target_is_rejected(value):
    with pytest.raises(ValueError, match="stream_memory_mib must be a positive integer"):
        cfg.KernelTuning(stream_memory_mib=value)


def test_memory_target_excludes_block_electrons():
    with pytest.raises(ValueError, match="exclusive"):
        cfg.KernelTuning(stream_block_electrons=5, stream_memory_mib=5)


def test_memory_target_stays_out_of_case_identity_and_old_pickles_restore():
    plain = build_case(_scene(), _numerics())
    target = build_case(_scene(), _memory_numerics(64))
    assert case_content_key(plain) == case_content_key(target)
    state = dict(cfg.KernelTuning().__dict__)
    del state["stream_memory_mib"]
    restored = cfg.KernelTuning.__new__(cfg.KernelTuning)
    restored.__setstate__(state)
    assert restored == cfg.KernelTuning()


def test_memory_target_refused_outside_simulate():
    with pytest.raises(ValueError, match="streamed scoring"):
        simulate_trajectories(_scene().beam, _scene().target, numerics=_memory_numerics(64))
    with pytest.raises(ValueError, match="streamed scoring"):
        pr.simulate(
            _scene(),
            numerics=pr.Numerics(
                sampling=FixedSampling(cxr_histories=12, brem_histories=16),
                energy_bins=BINS,
                execution=cfg.KernelTuning(stream_memory_mib=64),
            ),
            executor=LocalExecutor(),
        )


def test_memory_target_matches_the_default():
    reference = pr.simulate(_scene(), numerics=_numerics())
    streamed = pr.simulate(_scene(), numerics=_memory_numerics(64))
    assert streamed.case == reference.case
    np.testing.assert_allclose(
        streamed.cxr.incoherent.density, reference.cxr.incoherent.density, rtol=1e-12
    )
    np.testing.assert_allclose(streamed.brem.density, reference.brem.density, rtol=1e-12)


def _spill_numerics(directory, mib, bins=BINS):
    return pr.Numerics(
        sampling=FixedSampling(cxr_histories=12, brem_histories=16),
        energy_bins=bins,
        execution=cfg.KernelTuning(
            transport_core="per-electron",
            stream_block_electrons=5,
            stream_spill_dir=directory,
            stream_spill_mib=mib,
        ),
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_spill_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="stream_spill_mib must be a non-negative integer"):
        cfg.KernelTuning(stream_block_electrons=5, stream_spill_mib=value)


@pytest.mark.parametrize("value", ["", 3])
def test_invalid_spill_directory_is_rejected(value):
    with pytest.raises(ValueError, match="stream_spill_dir must be a nonempty path"):
        cfg.KernelTuning(stream_block_electrons=5, stream_spill_dir=value)


@pytest.mark.parametrize("field", ["stream_spill_dir", "stream_spill_mib"])
def test_spill_controls_require_streaming(field):
    with pytest.raises(ValueError, match="require stream_block_electrons or stream_memory_mib"):
        cfg.KernelTuning(**{field: "/scratch" if field == "stream_spill_dir" else 0})


def test_spill_controls_stay_out_of_identity_and_pickle(tmp_path):
    tuning = cfg.KernelTuning(stream_memory_mib=64, stream_spill_dir=tmp_path, stream_spill_mib=0)
    assert tuning.stream_spill_dir == str(tmp_path)
    assert pickle.loads(pickle.dumps(tuning)) == tuning
    state = dict(cfg.KernelTuning().__dict__)
    del state["stream_spill_dir"], state["stream_spill_mib"]
    restored = cfg.KernelTuning.__new__(cfg.KernelTuning)
    restored.__setstate__(state)
    assert restored == cfg.KernelTuning()
    plain = build_case(_scene(), _numerics())
    spilled = build_case(_scene(), _spill_numerics(tmp_path, 0))
    assert case_content_key(plain) == case_content_key(spilled)


def test_unwritable_spill_directory_fails_before_transport(monkeypatch, tmp_path):
    _no_transport(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ValueError, match="stream_spill_dir"):
        pr.simulate(_scene(), numerics=_spill_numerics(blocker / "spill", 0))


def _spill_parents(monkeypatch):
    from pyrite.montecarlo.runner.line_grid_spill import SpillStore

    parents = []
    init = SpillStore.__init__

    def record(self, parent):
        parents.append(parent)
        init(self, parent)

    monkeypatch.setattr(SpillStore, "__init__", record)
    return parents


def _clear_grid_cache():
    import shutil

    from pyrite.console.outputs import output_dir

    shutil.rmtree(output_dir("cache"), ignore_errors=True)


def test_spill_controls_choose_directory_and_threshold(monkeypatch, tmp_path):
    """The automatic grid spills under the chosen directory, bitwise unchanged."""
    _clear_grid_cache()
    reference = pr.simulate(_scene(), numerics=_numerics(bins=None))
    parents = _spill_parents(monkeypatch)
    # The default run cached its grid: same speed-cache key, no grid pass.
    cached = pr.simulate(_scene(), numerics=_spill_numerics(tmp_path / "s", 0, bins=None))
    assert parents == []
    _clear_grid_cache()
    held = pr.simulate(_scene(), numerics=_spill_numerics(tmp_path / "s", 1024, bins=None))
    assert parents == []
    _clear_grid_cache()
    spilled = pr.simulate(_scene(), numerics=_spill_numerics(tmp_path / "s", 0, bins=None))
    assert parents == [tmp_path / "s"]
    assert list((tmp_path / "s").iterdir()) == []
    want = reference.cxr.incoherent.photon_energy_eV
    for result in (cached, held, spilled):
        assert result.case == reference.case
        np.testing.assert_array_equal(result.cxr.incoherent.photon_energy_eV, want)
