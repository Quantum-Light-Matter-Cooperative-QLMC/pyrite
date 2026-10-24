"""Grouped scientific vs kernel settings on the rebased #681 execution branch."""

from dataclasses import replace
import pickle

import pytest

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key
from pyrite.execution import LocalExecutor, ResourceRequest


def _scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0,
                footprint=pr.Footprint(5.0, 5.0)),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="eedl"),
    )


def _case(numerics):
    return build_case(replace(_scene(), numerics=numerics))


def _same_case(a, b):
    assert pickle.dumps(a.to_dict(), protocol=5) == pickle.dumps(b.to_dict(), protocol=5)
    assert case_content_key(a) == case_content_key(b)


def test_defaults_and_legacy_read_views():
    n = pr.Numerics()
    assert n.transport is n.radiation is n.execution is None
    assert n.resolved_transport == pr.TransportNumerics()
    assert n.resolved_radiation == pr.RadiationNumerics()
    assert n.resolved_execution == pr.ExecutionConfig()
    old = pr.Numerics(31, 17, cxr_chunk=4, brem_chunk=8,
                      energy_model="midpoint", max_dE_frac=0.01,
                      cxr_reflections=pr.CxrReflections(n_families=2))
    assert old.resolved_execution == pr.ExecutionConfig(cxr_chunk=4, brem_chunk=8)
    assert old.resolved_transport.max_dE_frac == 0.01
    assert old.resolved_radiation.cxr_reflections.n_families == 2
    _same_case(_case(n), _case(pr.Numerics(
        transport=pr.TransportNumerics(), radiation=pr.RadiationNumerics(),
        execution=pr.ExecutionConfig(),
    )))


def test_typed_sections_lower_to_identical_case_payload_and_hash():
    old = pr.Numerics(
        cxr_trials=31, brem_trials=17,
        energy_model="midpoint", max_dE_frac=0.01,
        cxr_reflections=pr.CxrReflections(n_families=2, mosaic_nodes=3),
        transport_core="per-electron", cxr_chunk=8, brem_chunk=4,
    )
    typed = pr.Numerics(
        sampling=pr.FixedSampling(31, 17),
        transport=pr.TransportNumerics(max_dE_frac=0.01),
        radiation=pr.RadiationNumerics(
            cxr_reflections=pr.CxrReflections(n_families=2, mosaic_nodes=3)),
        execution=pr.ExecutionConfig(
            transport_core="per-electron", cxr_chunk=8, brem_chunk=4),
    )
    assert typed.energy_model == old.energy_model
    assert typed.max_dE_frac == old.max_dE_frac
    assert typed.cxr_chunk == old.cxr_chunk
    assert typed.brem_chunk == old.brem_chunk
    _same_case(_case(old), _case(typed))
    assert pickle.loads(pickle.dumps(typed)) == typed


def test_adaptive_and_grouped_execution_remain_compatible():
    p = pr.AdaptiveSampling(0.1, 40, 80, 20)
    old = pr.Numerics(precision=pr.Precision.from_dict(p.to_dict()),
                      transport_core="per-electron", cxr_chunk=2)
    new = pr.Numerics(sampling=p, execution=pr.ExecutionConfig(
        transport_core="per-electron", cxr_chunk=2))
    _same_case(_case(old), _case(new))
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        pr.Numerics(sampling=p, execution=pr.ExecutionConfig(transport_core="lockstep"))


def test_conflicting_flat_and_grouped_fields_fail_early():
    with pytest.raises(ValueError, match="energy_model"):
        pr.Numerics(transport=pr.TransportNumerics(energy_model="midpoint"),
                    energy_model="frozen")
    with pytest.raises(ValueError, match="max_dE_frac"):
        pr.Numerics(transport=pr.TransportNumerics(max_dE_frac=0.02), max_dE_frac=0.01)
    with pytest.raises(ValueError, match="cxr_chunk"):
        pr.Numerics(execution=pr.ExecutionConfig(cxr_chunk=5), cxr_chunk=4)
    with pytest.raises(ValueError, match="cxr_reflections"):
        pr.Numerics(radiation=pr.RadiationNumerics(
            cxr_reflections=pr.CxrReflections(n_families=2)),
            cxr_reflections=pr.CxrReflections(n_families=3))
    with pytest.raises(TypeError, match="TransportNumerics"):
        pr.Numerics(transport={"energy_model": "midpoint"})
    with pytest.raises(TypeError, match="ExecutionConfig"):
        pr.Numerics(execution=LocalExecutor())


def test_group_validations():
    with pytest.raises(ValueError, match="energy_model"):
        pr.TransportNumerics(energy_model="oops")
    with pytest.raises(ValueError, match="midpoint"):
        pr.TransportNumerics(energy_model="frozen", max_dE_frac=0.02)
    with pytest.raises(ValueError, match="non-negative"):
        pr.TransportNumerics(max_dE_frac=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        pr.ExecutionConfig(cxr_chunk=True)
    with pytest.raises(TypeError, match="CxrReflections"):
        pr.RadiationNumerics(cxr_reflections=None)


def test_explicit_group_replace_helpers():
    n = pr.Numerics(transport=pr.TransportNumerics(max_dE_frac=0.02),
                    execution=pr.ExecutionConfig(cxr_chunk=3))
    changed = n.with_transport(replace(n.resolved_transport, max_dE_frac=0.03))
    assert changed.max_dE_frac == 0.03
    assert changed.execution == n.execution
    changed = changed.with_execution(replace(changed.resolved_execution, cxr_chunk=6))
    assert changed.cxr_chunk == 6
    changed = changed.with_radiation(pr.RadiationNumerics(
        cxr_reflections=pr.CxrReflections(max_reflections=4)))
    assert changed.cxr_reflections.max_reflections == 4
    assert n.max_dE_frac == 0.02
    assert n.cxr_chunk == 3
    assert changed.describe_numerics()["execution"]["source"] == "typed"
    assert changed.describe_numerics()["transport"]["classification"] == "scientific"
    assert changed.describe_numerics()["execution"]["classification"] == "execution"
    assert replace(n, brem_trials=99).brem_trials == 99


def test_681_executor_and_resource_contract_not_shadowed():
    assert callable(LocalExecutor().execute)
    assert isinstance(ResourceRequest(), ResourceRequest)
    assert "pyrite.execution" not in pr.ExecutionConfig.__module__
