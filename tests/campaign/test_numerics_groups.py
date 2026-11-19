"""Grouped scientific vs kernel settings on the rebased #681 execution branch."""

import pickle
from dataclasses import replace

import pytest

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.campaign.profiles import case_content_key
from pyrite.config import FixedSampling
from pyrite.execution import LocalExecutor, ResourceRequest


def _scene(material="hopg"):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            material,
            thickness_ang=10_000.0,
            tilt_deg=30.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=pr.Physics(bremsstrahlung_model="eedl", radiative_model="uncoupled"),
    )


def _case(numerics):
    return build_case(replace(_scene(), numerics=numerics))


def _same_case(a, b):
    assert pickle.dumps(a.to_dict(), protocol=5) == pickle.dumps(b.to_dict(), protocol=5)
    assert case_content_key(a) == case_content_key(b)


def test_defaults_and_legacy_read_views():
    n = pr.Numerics()
    assert n.transport is n.radiation is n.execution is None
    assert n.resolved_transport == pr.config.TransportNumerics()
    assert n.resolved_radiation == pr.config.RadiationNumerics()
    assert n.resolved_execution == pr.ExecutionConfig()
    old = pr.Numerics(
        sampling=pr.config.FixedSampling(31, 17),
        cxr_chunk=4,
        brem_chunk=8,
        energy_model="midpoint",
        max_dE_frac=0.01,
        cxr_reflections=pr.config.CxrReflections(n_families=2),
    )
    assert old.resolved_execution == pr.ExecutionConfig(cxr_chunk=4, brem_chunk=8)
    assert old.resolved_transport.max_dE_frac == 0.01
    assert old.resolved_radiation.cxr_reflections.n_families == 2
    _same_case(
        _case(n),
        _case(
            pr.Numerics(
                transport=pr.config.TransportNumerics(),
                radiation=pr.config.RadiationNumerics(),
                execution=pr.ExecutionConfig(),
            )
        ),
    )


def test_typed_sections_lower_to_identical_case_payload_and_hash():
    old = pr.Numerics(
        sampling=FixedSampling(cxr_histories=31, brem_histories=17),
        energy_model="midpoint",
        max_dE_frac=0.01,
        cxr_reflections=pr.config.CxrReflections(n_families=2, mosaic_nodes=3),
        transport_core="per-electron",
        cxr_chunk=8,
        brem_chunk=4,
    )
    typed = pr.Numerics(
        sampling=pr.config.FixedSampling(31, 17),
        transport=pr.config.TransportNumerics(max_dE_frac=0.01),
        radiation=pr.config.RadiationNumerics(
            cxr_reflections=pr.config.CxrReflections(n_families=2, mosaic_nodes=3)
        ),
        execution=pr.ExecutionConfig(transport_core="per-electron", cxr_chunk=8, brem_chunk=4),
    )
    assert typed.energy_model == old.energy_model
    assert typed.max_dE_frac == old.max_dE_frac
    assert typed.cxr_chunk == old.cxr_chunk
    assert typed.brem_chunk == old.brem_chunk
    _same_case(_case(old), _case(typed))
    assert pickle.loads(pickle.dumps(typed)) == typed


def test_adaptive_and_grouped_execution_remain_compatible():
    p = pr.config.AdaptiveSampling(0.1, 40, 80, 20)
    old = pr.Numerics(
        sampling=pr.config.AdaptiveSampling.from_dict(p.to_dict()),
        transport_core="per-electron",
        cxr_chunk=2,
    )
    new = pr.Numerics(
        sampling=p, execution=pr.ExecutionConfig(transport_core="per-electron", cxr_chunk=2)
    )
    _same_case(_case(old), _case(new))
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        pr.Numerics(sampling=p, execution=pr.ExecutionConfig(transport_core="lockstep"))


def test_conflicting_flat_and_grouped_fields_fail_early():
    with pytest.raises(ValueError, match="energy_model"):
        pr.Numerics(
            transport=pr.config.TransportNumerics(energy_model="midpoint"), energy_model="frozen"
        )
    with pytest.raises(ValueError, match="max_dE_frac"):
        pr.Numerics(transport=pr.config.TransportNumerics(max_dE_frac=0.02), max_dE_frac=0.01)
    with pytest.raises(ValueError, match="cxr_chunk"):
        pr.Numerics(execution=pr.ExecutionConfig(cxr_chunk=5), cxr_chunk=4)
    with pytest.raises(ValueError, match="cxr_reflections"):
        pr.Numerics(
            radiation=pr.config.RadiationNumerics(
                cxr_reflections=pr.config.CxrReflections(n_families=2)
            ),
            cxr_reflections=pr.config.CxrReflections(n_families=3),
        )
    with pytest.raises(TypeError, match="TransportNumerics"):
        pr.Numerics(transport={"energy_model": "midpoint"})
    with pytest.raises(TypeError, match="ExecutionConfig"):
        pr.Numerics(execution=LocalExecutor())


def test_group_validations():
    with pytest.raises(ValueError, match="energy_model"):
        pr.config.TransportNumerics(energy_model="oops")
    with pytest.raises(ValueError, match="midpoint"):
        pr.config.TransportNumerics(energy_model="frozen", max_dE_frac=0.02)
    with pytest.raises(ValueError, match="non-negative"):
        pr.config.TransportNumerics(max_dE_frac=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        pr.ExecutionConfig(cxr_chunk=True)
    with pytest.raises(TypeError, match="CxrReflections"):
        pr.config.RadiationNumerics(cxr_reflections=None)


def test_explicit_group_replace_helpers():
    n = pr.Numerics(
        transport=pr.config.TransportNumerics(max_dE_frac=0.02),
        execution=pr.ExecutionConfig(cxr_chunk=3),
    )
    changed = n.with_transport(replace(n.resolved_transport, max_dE_frac=0.03))
    assert changed.max_dE_frac == 0.03
    assert changed.execution == n.execution
    changed = changed.with_execution(replace(changed.resolved_execution, cxr_chunk=6))
    assert changed.cxr_chunk == 6
    changed = changed.with_radiation(
        pr.config.RadiationNumerics(cxr_reflections=pr.config.CxrReflections(max_reflections=4))
    )
    assert changed.cxr_reflections.max_reflections == 4
    assert n.max_dE_frac == 0.02
    assert n.cxr_chunk == 3
    assert changed.describe_numerics()["execution"]["source"] == "typed"
    assert changed.describe_numerics()["transport"]["classification"] == "scientific"
    assert changed.describe_numerics()["execution"]["classification"] == "kernel-tuning"
    assert (
        replace(n, sampling=pr.config.FixedSampling(brem_histories=99)).sampling.brem_histories
        == 99
    )


def test_681_executor_and_resource_contract_not_shadowed():
    assert callable(LocalExecutor().execute)
    assert isinstance(ResourceRequest(), ResourceRequest)
    assert "pyrite.execution" not in pr.ExecutionConfig.__module__


def test_line_yield_tolerance_lowers_to_a_selection_record_and_leaves_default_identity():
    from pyrite.montecarlo.runner.reflection_cutoff import REFLECTION_SELECTION_MODEL

    reflections = pr.config.CxrReflections(line_yield_tolerance=1e-3)
    selected = _case(pr.Numerics(cxr_reflections=reflections))
    record = selected.to_dict()["reflection_selection"]
    assert record["model"] == REFLECTION_SELECTION_MODEL and record["tolerance"] == 1e-3
    plain = _case(pr.Numerics())
    assert plain.to_dict()["reflection_selection"]["automatic"] is True
    _same_case(plain, _case(pr.Numerics(cxr_reflections=pr.config.CxrReflections())))
    assert case_content_key(selected) != case_content_key(plain)
    for bad in (0.0, 1.0, -1e-3, True):
        with pytest.raises(ValueError, match="line_yield_tolerance"):
            pr.config.CxrReflections(line_yield_tolerance=bad)
    with pytest.raises(ValueError, match="incoherent lines only"):
        replace(selected, coherent_emission=True)


def test_default_selection_and_explicit_family_precedence():
    scene = _scene("silicon")
    automatic = build_case(scene)
    record = automatic.to_dict()["reflection_selection"]
    assert record["automatic"] is True and record["tolerance"] == 1e-3
    fixed = build_case(scene, pr.Numerics(cxr_reflections=pr.config.CxrReflections(n_families=4)))
    assert "reflection_selection" not in fixed.to_dict()
    assert automatic.hkl_list == fixed.hkl_list
    assert case_content_key(automatic) != case_content_key(fixed)
    # Single-crystal symmetric-cut pins remain authoritative.
    from pyrite.campaign.reflection_selection import resolve_reflection_sweep
    from pyrite.campaign.sweep import Sweep

    _, automatic = resolve_reflection_sweep(Sweep(material="pdte2"), False, False)
    assert not automatic


@pytest.mark.parametrize("material", ["hopg", "hbn"])
def test_fiber_crystals_request_automatic_selection(material):
    case = build_case(_scene(material)).to_dict()
    assert case["reflection_selection"]["automatic"] is True
    assert case["reflection_selection"]["tolerance"] == 1e-3
    assert all(h == k == 0 for h, k, _ in case["hkl_list"])


@pytest.mark.parametrize("mode", ["coherent", "temporal", "roulette", "two-beam", "channeling"])
def test_automatic_selection_falls_back_on_unsupported_routes(mode):
    physics = (
        replace(_scene().physics, emission=("incoherent", "coherent"))
        if mode == "coherent"
        else None
    )
    if mode == "two-beam":
        physics = replace(_scene().physics, transition_radiation="two-beam")
    channeling = (
        pr.quantum.Channeling("planar", plane=(1, 1, 0), admission="accept-inaccurate")
        if mode == "channeling"
        else None
    )
    # Two-beam TR already requires exactly one reflection, independently of
    # the selection policy. Keep that existing admission contract.
    reflections = pr.config.CxrReflections(max_reflections=1 if mode == "two-beam" else None)
    options = {}
    if mode == "two-beam":
        from pyrite.detectors import EnergyBins

        options["sampling"] = FixedSampling(8, 8)
        options["energy_bins"] = EnergyBins(cxr=[500.0, 1000.0, 1500.0, 2000.0])
    if mode == "temporal":
        options["temporal_profile"] = True
    elif mode == "roulette":
        options["line_roulette"] = pr.config.LineRoulette(floor=0.1)
    scene = _scene("silicon")
    case = build_case(
        scene,
        pr.Numerics(cxr_reflections=reflections, **options),
        physics=physics,
        channeling=channeling,
    )
    fixed = build_case(
        scene,
        pr.Numerics(cxr_reflections=replace(reflections, n_families=4), **options),
        physics=physics,
        channeling=channeling,
    )
    assert "reflection_selection" not in case.to_dict()
    _same_case(case, fixed)


@pytest.mark.parametrize("mode", ["blocks", "memory"])
def test_streamed_numerics_keep_automatic_selection(mode):
    """Streamed scoring selects on the electron prefix (#1281); no fallback."""
    execution = pr.ExecutionConfig(
        **{"stream_block_electrons" if mode == "blocks" else "stream_memory_mib": 5}
    )
    case = build_case(_scene("silicon"), pr.Numerics(execution=execution)).to_dict()
    assert case["reflection_selection"]["automatic"] is True


def test_streamed_scorer_applies_selection_without_mutating_the_case():
    from pyrite.montecarlo.runner.streamed_scoring import _streaming_case, streaming_refusal
    from pyrite.montecarlo.runner.streamed_selection import automatic_selection

    case = build_case(_scene("silicon")).to_dict()
    streamed = _streaming_case(case)
    assert "reflection_selection" not in streamed
    assert streamed["hkl_list"] == case["hkl_list"]
    assert streaming_refusal(case) is None and automatic_selection(case)
    assert case["reflection_selection"]["automatic"]
