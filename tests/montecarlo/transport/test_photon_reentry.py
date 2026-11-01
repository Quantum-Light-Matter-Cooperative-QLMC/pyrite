"""Analytic interface continuation and charged escape ownership (#341)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.transport.photon_charged_escape import escaped_charged_rows
from pyrite.montecarlo.transport.photon_coupling import ChargedStageResult
from pyrite.montecarlo.transport.photon_feedback import FeedbackChargedStage
from pyrite.montecarlo.transport.photon_histories import ELECTRON, POSITRON
from pyrite.montecarlo.transport.photon_navigation import PhotonBox
from pyrite.montecarlo.transport.photon_reentry import scene_feedback_stage
from tests.montecarlo.transport.test_photon_coupling import silicon_stage as silicon_stage
from tests.montecarlo.transport.test_photon_feedback import REST, _closed, _rows

SI = (("Si", 0.05),)
EYE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _reference(layers, boxes=(), **kw):
    return scene_feedback_stage(
        layers,
        boxes=boxes,
        photon_cutoff_eV=1e3,
        secondary_threshold_eV=5e3,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        radiative_model="bremslib-soft-hard",
        **kw,
    )


def test_escape_phase_space_keeps_last_track_row_owner_and_species():
    from pyrite.montecarlo.transport.events import EVENT_CUTOFF, EVENT_EXIT_BOTTOM, EVENT_EXIT_TOP

    # Stored out of order; primary positron exits down, its electron daughter up.
    result = dict(
        track_id=np.array([1, 0, 0, 2]),
        flight_id=np.array([0, 1, 0, 0]),
        substep_id=np.zeros(4, int),
        electron_id=np.array([0, 0, 0, 1]),
        event_kind=np.array([EVENT_EXIT_TOP, EVENT_EXIT_BOTTOM, 0, EVENT_CUTOFF]),
        r_mid=np.array([[0, 0, 1], [0, 0, 9], [0, 0, 5], [0, 0, 3]]),
        v_hat=np.array([[0, 0, -1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        L_ang=np.full(4, 2.0),
        E_end_keV=np.array([3.0, 7.0, 8.0, 1.0]),
        secondary_tracks={"generation": np.array([0, 1, 0])},
    )
    rows = _rows([1e4, 1e4], POSITRON)
    out = escaped_charged_rows(result, rows)
    np.testing.assert_array_equal(out["kinetic_eV"], [7000.0, 3000.0])
    np.testing.assert_array_equal(out["species"], [POSITRON, ELECTRON])
    np.testing.assert_array_equal(out["position_ang"][:, 2], [10.0, 0.0])
    np.testing.assert_array_equal(out["parent"], [0, 0])
    assert len(np.unique(out["key"])) == 2
    result["event_kind"][:] = EVENT_CUTOFF
    assert escaped_charged_rows(result, rows)["position_ang"].shape == (0, 3)


@pytest.mark.parametrize("species", [ELECTRON, POSITRON])
def test_real_feedback_retains_all_escaped_kinetic_energy(silicon_stage, species):
    layers, base = silicon_stage
    stage = FeedbackChargedStage(base, 1e3, retain_exits=True)
    result = _closed(stage, _rows([3e5], species, layers[-1][1] - 1.0))
    exits = result.escaped_particles
    budget = exits["kinetic_eV"] + np.where(exits["species"] == POSITRON, REST, 0.0)
    actual = np.bincount(exits["parent"], weights=budget, minlength=1)
    np.testing.assert_allclose(actual, result.escaped_eV, rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(exits["direction"], axis=1), 1.0, atol=1e-12)
    assert exits["kinetic_eV"].size


class _Material:
    """Analytic interface oracle: deposit 10%, optionally reflect at the entry."""

    threshold_eV = 5e3

    def __init__(self, mirror=False):
        self.mirror, self.calls = mirror, []

    def __call__(self, rows, seed):
        self.calls.append((rows, seed))
        n = len(rows["kinetic_eV"])
        zero = np.zeros(n)
        photons = dict(
            position_ang=np.empty((0, 3)),
            direction=np.empty((0, 3)),
            energy_eV=np.empty(0),
            parent=np.empty(0, int),
            ordinal=np.empty(0, int),
        )
        position = rows["position_ang"].copy()
        direction = rows["direction"].copy()
        if self.mirror:
            direction = -direction
        else:
            position[:, 2] = np.where(direction[:, 2] > 0, 10.0, 0.0)
        exits = {
            **rows,
            "position_ang": position,
            "direction": direction,
            "kinetic_eV": rows["kinetic_eV"] * 0.9,
            "key": rows["key"] + np.uint64(100),
            "parent": np.arange(n),
        }
        rest = np.where(rows["species"] == POSITRON, REST, 0.0)
        return ChargedStageResult(
            rows["kinetic_eV"] * 0.9 + rest,
            rows["kinetic_eV"] * 0.1,
            zero,
            zero,
            photons,
            rest,
            exits,
        )


@pytest.mark.parametrize("species", [ELECTRON, POSITRON])
@pytest.mark.parametrize("gap", [0.0, 5.0, 1e6])
def test_target_filter_backscatter_reenters_target_once(gap, species):
    # A proper 180-degree filter rotation exercises both phase-space transforms.
    box = PhotonBox(
        (0.0, 0.0, 15.0 + gap),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        (100.0, 100.0, 5.0),
        SI,
    )
    stage = _reference([(0.0, 10.0, SI)], [box], width_ang=200.0, height_ang=200.0)
    target, mirror = _Material(), _Material(mirror=True)
    stage = replace(
        stage,
        regions=(replace(stage.regions[0], stage=target), replace(stage.regions[1], stage=mirror)),
    )
    result = _closed(stage, _rows([1e4], species, z=1.0))
    assert len(target.calls) == 2 and len(mirror.calls) == 1
    np.testing.assert_array_equal(target.calls[1][0]["direction"], [[0.0, 0.0, -1.0]])
    assert target.calls[0][0]["key"][0] != target.calls[1][0]["key"][0]
    np.testing.assert_allclose(result.deposited_eV, [2710.0], atol=1e-10)
    np.testing.assert_allclose(result.escaped_eV, [7290.0 + (REST if species == POSITRON else 0.0)])
    np.testing.assert_array_equal(result.escaped_particles["position_ang"], [[0.0, 0.0, 0.0]])


def test_vacuum_miss_does_not_annihilate_a_low_energy_positron():
    stage = _reference([(0.0, 10.0, SI)], width_ang=10.0, height_ang=10.0)
    result = _closed(stage, _rows([1.0], POSITRON, z=20.0))
    np.testing.assert_allclose(result.escaped_eV, [REST + 1.0], rtol=2e-16)
    assert result.photons["energy_eV"].size == 0


def test_geometry_ownership_gaps_empty_regions_and_crossing_cap():
    layers = [(0.0, 10.0, SI), (10.0, 20.0, ()), (30.0, 40.0, SI)]
    stage = _reference(layers)
    rows = _rows([1e4, 1e4, 1e4], z=0.0)
    rows["position_ang"][:, 2] = [0.0, 10.0, 30.0]
    np.testing.assert_array_equal(stage.contains(rows), [True, False, True])
    stage.validate_geometry(layers, (), None, None)
    with pytest.raises(ValueError, match="same scene"):
        stage.validate_geometry(layers[:1], (), None, None)
    target = _Material()
    stage = replace(
        stage, regions=tuple(replace(r, stage=target) for r in stage.regions), max_crossings=1
    )
    with pytest.raises(RuntimeError, match="max_crossings"):
        stage(_rows([1e4], z=1.0), 5)


def test_scene_factory_checks_per_region_table_ownership():
    with pytest.raises(ValueError, match="followed by boxes"):
        _reference([(0.0, 10.0, SI), (20.0, 30.0, SI)], inelastic_materials=["silicon"])
    with pytest.raises(ValueError, match="material body"):
        _reference([(0.0, 10.0, ())])
    with pytest.raises(ValueError, match="max_crossings"):
        _reference([(0.0, 10.0, SI)], max_crossings=0)


@pytest.mark.parametrize("species", [ELECTRON, POSITRON])
def test_real_scene_feedback_crosses_target_gap_and_filter(silicon_stage, species):
    _, base = silicon_stage
    transport = {
        k: v
        for k, v in base.transport.items()
        if k not in ("layers", "thickness_ang", "crystal_width_mm", "crystal_height_mm")
    }
    for name in ("stopping_tables", "inelastic_materials"):
        transport[name] = transport[name] * 2
    box = PhotonBox((0.0, 0.0, 3e7), EYE, (1e8, 1e8, 1e7), base.transport["layers"][0][2])
    stage = scene_feedback_stage(
        [(0.0, 1.0, box.composition)], boxes=[box], photon_cutoff_eV=1e3, **transport
    )
    rows = _rows([3e5], species, z=0.5)
    result = _closed(stage, rows)
    # The one-angstrom target cannot cause this much loss: the filter is visited.
    assert result.deposited_eV[0] > 100.0
    repeated = stage(rows, 5)
    for name in result.photons:
        np.testing.assert_array_equal(result.photons[name], repeated.photons[name])
    for name in result.escaped_particles:
        np.testing.assert_array_equal(
            result.escaped_particles[name], repeated.escaped_particles[name]
        )


def test_real_photon_queue_launches_filter_born_electrons(silicon_stage):
    from pyrite.montecarlo.transport.photon_coupling import (
        coupled_ledger_residual_eV,
        track_coupled_histories,
    )
    from tests.montecarlo.transport.test_photon_coupling import _beam

    _, base = silicon_stage
    transport = {
        k: v
        for k, v in base.transport.items()
        if k not in ("layers", "thickness_ang", "crystal_width_mm", "crystal_height_mm")
    }
    for name in ("stopping_tables", "inelastic_materials"):
        transport[name] = transport[name] * 2
    composition = base.transport["layers"][0][2]
    layers = [(0.0, 1.0, composition)]
    box = PhotonBox((0.0, 0.0, 1e9 + 1e4), EYE, (1e10, 1e10, 1e9), composition)
    stage = scene_feedback_stage(layers, boxes=[box], photon_cutoff_eV=1e3, **transport)
    out = track_coupled_histories(
        *_beam(8, 3e5), layers, boxes=[box], seed=341, photon_cutoff_eV=1e3, charged_stage=stage
    )
    filter_rows = out.charged["position_ang"][:, 2] > 1.0
    assert filter_rows.any()
    assert np.all(out.charged["fate"][filter_rows] == "launched")
    assert out.ledger["complete"].all()
    np.testing.assert_allclose(coupled_ledger_residual_eV(out.ledger), 0.0, atol=1e-7)


def test_scene_stage_refuses_conflicting_material_geometry():
    for name, value in (
        ("groove", object()),
        ("thickness_ang", 1.0),
        ("crystal_width_mm", 1.0),
    ):
        with pytest.raises(ValueError, match="owned by the charged scene geometry"):
            _reference([(0.0, 10.0, SI)], **{name: value})
    stage = _reference([(0.0, 10.0, SI)], inelastic_materials="silicon")
    assert stage.regions[0].stage.base.transport["inelastic_materials"] == ["silicon"]


def test_rotated_side_entry_preserves_launch_energy(silicon_stage):
    # Independent verifier's witness: world/local roundoff put x below -10 Å,
    # causing the charged kernel's exact footprint guard to drop the launch.
    _, base = silicon_stage
    transport = {
        k: v
        for k, v in base.transport.items()
        if k not in ("layers", "thickness_ang", "crystal_width_mm", "crystal_height_mm")
    }
    theta = 0.056733668341708544
    c, s = np.cos(theta), np.sin(theta)
    axes = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    center = np.array([0.0, 0.0, 30.0])
    box = PhotonBox(
        tuple(center), tuple(map(tuple, axes)), (10.0, 10.0, 5.0), base.transport["layers"][0][2]
    )
    stage = scene_feedback_stage([], boxes=[box], photon_cutoff_eV=1e3, **transport)
    rows = _rows([3e5])
    rows["position_ang"][:] = center - 30.0 * axes[0]
    rows["direction"][:] = axes[0]
    rows["key"][:] = 1
    result = _closed(stage, rows)
    assert result.deposited_eV[0] > 0.0
    np.testing.assert_allclose(
        result.escaped_particles["position_ang"], [center + 10.0 * axes[0]], atol=1e-12, rtol=0.0
    )


def test_material_stage_cannot_silently_drop_a_launch_or_escaped_state():
    stage = _reference([(0.0, 10.0, SI)])
    for fault, match in (
        ("energy", "visit failed"),
        ("missing", "did not retain"),
        ("escape", "escape phase space failed"),
    ):

        def broken(rows, seed, fault=fault):
            result = _Material()(rows, seed)
            if fault == "energy":
                return replace(result, deposited_eV=np.zeros(1))
            if fault == "missing":
                return replace(result, escaped_particles=None)
            return replace(
                result, escaped_particles={**result.escaped_particles, "kinetic_eV": np.zeros(1)}
            )

        bad = replace(stage, regions=(replace(stage.regions[0], stage=broken),))
        with pytest.raises(RuntimeError, match=match):
            bad(_rows([1e4], z=1.0), 5)
