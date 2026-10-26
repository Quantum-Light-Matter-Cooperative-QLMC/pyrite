"""#702: configuration ownership, canonical imports and stable lowering."""

import os
import pickle
import subprocess
import sys
from dataclasses import fields, replace

import pytest

import pyrite as pr
import pyrite.config as cfg
from pyrite._config_policy import (
    KERNEL_TUNING,
    NUMERICAL_APPROXIMATION,
    NUMERICS_FIELD_OWNERS,
    PHYSICAL_MODEL,
    PHYSICS_FIELD_OWNERS,
    SAMPLING_CONVERGENCE,
)
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key


def _scene(numerics):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=1000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0)),
        pr.Detector(),
        physics=cfg.Physics(bremsstrahlung_model="eedl"),
        numerics=numerics,
    )


def test_exhaustive_field_ownership_does_not_drift():
    assert set(PHYSICS_FIELD_OWNERS) == {f.name for f in fields(cfg.Physics)}
    assert set(NUMERICS_FIELD_OWNERS) == {f.name for f in fields(cfg.Numerics)}
    assert set(PHYSICS_FIELD_OWNERS.values()) == {PHYSICAL_MODEL}
    assert set(NUMERICS_FIELD_OWNERS.values()) == {
        KERNEL_TUNING,
        NUMERICAL_APPROXIMATION,
        SAMPLING_CONVERGENCE,
    }


def test_kernel_tuning_alias_and_mirrored_case_identity():
    assert cfg.KernelTuning is cfg.ExecutionConfig is pr.KernelTuning is pr.ExecutionConfig
    kernel = cfg.KernelTuning(transport_core="per-electron", cxr_chunk=7, brem_chunk=3)
    assert type(kernel).__name__ == "ExecutionConfig"
    assert pickle.loads(pickle.dumps(kernel)) == kernel
    old = cfg.Numerics(
        cxr_trials=31,
        brem_trials=17,
        transport_core="per-electron",
        cxr_chunk=7,
        brem_chunk=3,
    )
    new = cfg.Numerics(cxr_trials=31, brem_trials=17, execution=kernel)
    assert new.kernel_tuning is kernel
    assert new.resolved_execution is kernel
    old_case, new_case = build_case(_scene(old)), build_case(_scene(new))
    assert pickle.dumps(old_case.to_dict(), protocol=5) == pickle.dumps(
        new_case.to_dict(), protocol=5
    )
    assert case_content_key(old_case) == case_content_key(new_case)
    updated = new.with_kernel_tuning(replace(kernel, cxr_chunk=8))
    assert updated.kernel_tuning.cxr_chunk == 8
    assert updated.execution is not None


def test_defaults_and_grouped_forms_lower_identically():
    legacy = cfg.Numerics()
    grouped = cfg.Numerics(
        sampling=cfg.FixedSampling(),
        transport=cfg.TransportNumerics(),
        radiation=cfg.RadiationNumerics(),
        execution=cfg.KernelTuning(),
    )
    a, b = build_case(_scene(legacy)), build_case(_scene(grouped))
    assert pickle.dumps(a.to_dict(), protocol=5) == pickle.dumps(b.to_dict(), protocol=5)
    assert case_content_key(a) == case_content_key(b)


def test_mixed_group_conflicts_fail_instead_of_overriding():
    with pytest.raises(ValueError, match="transport_core"):
        cfg.Numerics(transport_core="lockstep", execution=cfg.KernelTuning(transport_core="cuda"))
    with pytest.raises(ValueError, match="max_dE_frac"):
        cfg.Numerics(max_dE_frac=0.1, transport=cfg.TransportNumerics(max_dE_frac=0.2))
    with pytest.raises(ValueError, match="cxr_trials"):
        cfg.Numerics(cxr_trials=45, sampling=cfg.FixedSampling(46, 20))
    with pytest.raises(ValueError, match="legacy precision"):
        cfg.Numerics(precision=cfg.Precision(0.1, 40, 80, 20), sampling=cfg.FixedSampling())


def test_canonical_module_does_not_initialize_schedulers_or_backends():
    source = (
        "import sys; import pyrite.config; "
        "assert 'pyrite.montecarlo.runner' not in sys.modules; "
        "assert 'pyrite.execution.dask' not in sys.modules; "
        "assert 'cupy' not in sys.modules; "
        "assert 'distributed' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", source], check=True, env=os.environ.copy())
