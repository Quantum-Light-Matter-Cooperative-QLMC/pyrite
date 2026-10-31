"""Public CUDA transport schedule controls on KernelTuning (#776).

Lowering and precedence only; schedule exactness on a device is covered by
``tests/montecarlo/test_transport_kernel_tuning_cuda.py``.
"""

import pickle
from dataclasses import replace

import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite import api
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key
from pyrite.execution import LocalExecutor
from pyrite.montecarlo import runner
from pyrite.montecarlo import transport as transport_module
from pyrite.montecarlo.runner.transport_options import _TRANSPORT_TUNING, transport_tuning
from tests.helpers.case_compare import assert_same_case


def _scene(numerics=None):
    scene = pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
    )
    return scene if numerics is None else replace(scene, numerics=numerics)


def _tuned(schedule="continuation", rows=64, core="auto"):
    return cfg.KernelTuning(
        transport_core=core, transport=cfg.TransportKernelTuning(schedule, rows)
    )


def test_defaults_leave_the_process_schedule_alone():
    assert cfg.KernelTuning().transport == cfg.TransportKernelTuning()
    assert pr.Numerics().resolved_execution.transport.schedule is None
    with transport_tuning(pr.Numerics(execution=cfg.KernelTuning())):
        assert _TRANSPORT_TUNING.get() is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schedule": "eager"}, "schedule must be one of"),
        ({"schedule": "auto", "continuation_rows": True}, "positive integer"),
        ({"schedule": "auto", "continuation_rows": 0}, "positive integer"),
        ({"schedule": "auto", "continuation_rows": 2.0}, "positive integer"),
        ({"electron_batch": 0}, "electron_batch must be a positive integer"),
        ({"electron_batch": False}, "electron_batch must be a positive integer"),
        ({"continuation_rows": 64}, "requires schedule"),
        ({"schedule": "full-history", "continuation_rows": 64}, "requires schedule"),
        ({"scratch_mib": 0}, "scratch_mib must be a positive integer"),
        ({"kernel_threads": 96}, "kernel_threads must be one of"),
        ({"kernel_threads": True}, "kernel_threads must be one of"),
    ],
)
def test_invalid_transport_tuning_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cfg.TransportKernelTuning(**kwargs)


def test_kernel_tuning_requires_typed_transport():
    with pytest.raises(TypeError, match="TransportKernelTuning"):
        cfg.KernelTuning(transport={"schedule": "auto"})


def test_schedule_overrides_the_env_default_and_keeps_other_fields(monkeypatch):
    env_default = replace(
        transport_module.DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
        schedule="full-history",
        max_batch=123,
    )
    monkeypatch.setattr(transport_module, "DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG", env_default)
    with transport_tuning(pr.Numerics(execution=_tuned(rows=48))):
        config = _TRANSPORT_TUNING.get()
    assert config.schedule == "continuation"
    assert config.continuation.rows_per_round == 48
    assert config.max_batch == 123
    assert _TRANSPORT_TUNING.get() is None
    with transport_tuning(pr.Numerics(execution=_tuned("auto", None))):
        assert _TRANSPORT_TUNING.get().continuation is None  # driver default quantum
    batch_only = cfg.KernelTuning(transport=cfg.TransportKernelTuning(electron_batch=512))
    with transport_tuning(pr.Numerics(execution=batch_only)):
        config = _TRANSPORT_TUNING.get()
    assert (config.max_batch, config.schedule) == (512, "full-history")  # env schedule kept


def test_scratch_and_launch_threads_lower_onto_the_transport_config():
    tuned = cfg.KernelTuning(
        transport=cfg.TransportKernelTuning(scratch_mib=2048, kernel_threads=32)
    )
    with transport_tuning(pr.Numerics(execution=tuned)):
        config = _TRANSPORT_TUNING.get()
    default = transport_module.DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
    assert (config.scratch_budget_bytes, config.kernel_threads) == (2048 * 1024**2, 32)
    assert (config.max_batch, config.schedule) == (default.max_batch, default.schedule)
    assert default.kernel_threads == 128
    with pytest.raises(ValueError, match="kernel_threads must be one of"):
        replace(default, kernel_threads=48)


def test_scheduling_stays_out_of_case_identity_and_pickles():
    plain = build_case(_scene(pr.Numerics()))
    tuned_numerics = pr.Numerics(execution=_tuned())
    tuned = build_case(_scene(tuned_numerics))
    assert_same_case(plain.to_dict(), tuned.to_dict())
    assert case_content_key(plain) == case_content_key(tuned)
    assert pickle.loads(pickle.dumps(tuned_numerics)) == tuned_numerics


def test_pre_776_pickles_gain_the_default_transport_tuning():
    legacy = object.__new__(cfg.KernelTuning)
    legacy.__dict__.update(transport_core="cuda", cxr_chunk=8, brem_chunk=None)
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored == cfg.KernelTuning(transport_core="cuda", cxr_chunk=8)


class _Stop(Exception):
    pass


def test_transport_case_receives_the_scheduled_config(monkeypatch):
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(runner, "simulate_trajectories", capture)
    case = build_case(_scene(pr.Numerics()))
    with pytest.raises(_Stop):
        runner._transport_case(case, transport_core="per-electron")
    assert "per_electron_config" not in seen
    with transport_tuning(pr.Numerics(execution=_tuned("auto", 32))), pytest.raises(_Stop):
        runner._transport_case(case, transport_core="per-electron")
    assert seen["per_electron_config"].schedule == "auto"
    assert seen["per_electron_config"].continuation.rows_per_round == 32


def test_simulate_applies_kernel_tuning_around_run_case(monkeypatch):
    seen = []

    def fake_run_case(case, transport_core="auto"):
        seen.append((_TRANSPORT_TUNING.get(), transport_core))
        raise _Stop

    monkeypatch.setattr(api, "run_case", fake_run_case)
    scene = _scene(pr.Numerics(execution=_tuned(core="cuda")))
    with pytest.raises(_Stop):
        pr.simulate(scene.beam, scene.target, scene.detector, numerics=scene.numerics)
    ((config, core),) = seen
    assert core == "cuda"
    assert (config.schedule, config.continuation.rows_per_round) == ("continuation", 64)


def test_batch_executors_reject_a_per_call_schedule():
    scene = _scene(pr.Numerics(execution=_tuned()))
    with pytest.raises(ValueError, match="default KernelTuning.transport"):
        pr.simulate(
            scene.beam,
            scene.target,
            scene.detector,
            numerics=scene.numerics,
            executor=LocalExecutor(),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cxr_threads": 48}, "cxr_threads must be one of"),
        ({"cxr_energies_per_block": 5}, "cxr_energies_per_block must be one of"),
        ({"cxr_energies_per_block": True}, "cxr_energies_per_block must be one of"),
    ],
)
def test_invalid_cxr_launch_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cfg.KernelTuning(**kwargs)


def test_cxr_launch_lowers_onto_the_reduction_config():
    from pyrite.montecarlo.runner.transport_options import (
        spectrum_launch,
        spectrum_launch_overrides,
    )

    assert spectrum_launch_overrides() == {}
    tuned = cfg.KernelTuning(cxr_energies_per_block=4)
    with transport_tuning(pr.Numerics(execution=tuned)):
        assert spectrum_launch_overrides() == {"energies_per_block": 4}
        assert _TRANSPORT_TUNING.get() is None  # transport untouched
    with spectrum_launch(cfg.KernelTuning(cxr_threads=256)):
        assert spectrum_launch_overrides() == {"nthreads": 256}
    assert spectrum_launch_overrides() == {}
    old = pickle.loads(pickle.dumps(cfg.KernelTuning()))
    old.__dict__.pop("cxr_threads")
    old.__setstate__(dict(old.__dict__))
    assert old.cxr_threads is None


def test_cxr_launch_stays_out_of_case_identity_and_batch_executors():
    plain, tuned = _scene(), _scene(pr.Numerics(execution=cfg.KernelTuning(cxr_threads=256)))
    assert case_content_key(build_case(plain)) == case_content_key(build_case(tuned))
    with pytest.raises(ValueError, match="default KernelTuning CXR launch"):
        pr.simulate(
            tuned.beam,
            tuned.target,
            tuned.detector,
            numerics=tuned.numerics,
            executor=LocalExecutor(),
        )
