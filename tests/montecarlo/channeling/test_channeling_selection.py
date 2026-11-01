"""Planar channeling selection, gates, identity and consumer refusals (#506).

Off is the default and must leave case payloads and content keys exactly as
they were; on adds one ``channeling`` record and the ``CHANNELING_MODEL``
marker. Every consumer that cannot run a channeling case refuses it before
any transport or output write (ADR-0017 tier 3).
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign.profiles import case_content_key
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case, channeling_case_record
from pyrite.montecarlo.channeling import CHANNELING_MODEL, DECHANNELING_MARGIN_EV
from pyrite.montecarlo.transport import SegmentEvent, check_segment_event_contract

PLANAR = dict(channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled")


def _scene(physics=None, *, energy_keV=5000.0, **slab):
    slab = {"thickness_ang": 20000.0, "tilt_deg": 3.0, "tilt_azim_deg": 45.0, **slab}
    return pr.Scene(
        pr.Beam(
            energy_keV=energy_keV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
        ),
        pr.geometry.Slab("silicon", footprint=pr.geometry.Footprint(5.0, 5.0), **slab),
        pr.Detector(),
        physics=physics,
    )


def _case(physics=None, numerics=None, **kwargs):
    return api.build_case(_scene(physics, **kwargs), numerics or pr.Numerics())


@pytest.fixture(scope="module")
def planar_case():
    return _case(pr.Physics(**PLANAR))


# -- selection ---------------------------------------------------------------


def test_off_is_the_default_and_adds_nothing():
    default = _case()
    explicit = _case(pr.Physics(channeling_model="off"))
    assert "channeling" not in default
    assert default.to_dict().keys() == explicit.to_dict().keys()
    assert case_content_key(default) == case_content_key(explicit)


def test_planar_adds_one_record_and_the_marker(planar_case):
    off = _case(pr.Physics(radiative_model="uncoupled"))
    record = planar_case["channeling"]
    assert record == channeling_case_record(
        (1, -1, 0),
        temperature_K=293.15,
        margin_eV=DECHANNELING_MARGIN_EV,
        basis_size=16,
        bloch_cells=16,
        dz_ang=250.0,
        zone_max_ang=None,
    )
    assert case_content_key({k: v for k, v in planar_case.items() if k != "channeling"}) == (
        case_content_key(off)
    )
    assert case_content_key(planar_case) != case_content_key(off)
    from pyrite._physics_provenance import resolved_physics_view

    view = resolved_physics_view(planar_case, table_markers={})
    assert ("channeling_model", CHANNELING_MODEL) in view.model_markers
    assert "transport.channeling/planar" in view.descriptor_ids
    assert "emission.channeling/planar" in view.descriptor_ids


def test_numerics_fields_enter_the_record_and_key(planar_case):
    numerics = pr.Numerics(channeling_basis_size=12, channeling_zone_max_ang=1e4)
    case = _case(pr.Physics(**PLANAR), numerics)
    assert case["channeling"]["basis_size"] == 12
    assert case["channeling"]["zone_max_ang"] == 1e4
    assert case_content_key(case) != case_content_key(planar_case)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"channeling_model": "axial"}, "channeling_plane is planar-only"),
        ({"channeling_plane": None}, "requires channeling_plane"),
        ({"channeling_plane": (0, 0, 0)}, "nonzero"),
        ({"channeling_plane": (1, 1)}, "three integer"),
        ({"radiative_model": "auto"}, "radiative_model='uncoupled'"),
        ({"emission": "coherent"}, "incoherent emission only"),
    ],
)
def test_physics_refuses_unsupported_selections(changes, message):
    with pytest.raises(ValueError, match=message):
        pr.Physics(**{**PLANAR, **changes})


def test_physics_refuses_plane_without_model_and_secondaries():
    with pytest.raises(ValueError, match="requires channeling_model='planar'"):
        pr.Physics(channeling_plane=(1, 1, 0))
    with pytest.raises(ValueError, match="secondary transport"):
        pr.Physics(
            **PLANAR,
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
            secondary_threshold_eV=1e4,
        )
    with pytest.raises(ValueError, match="energy_model='midpoint'"):
        pr.Physics(**PLANAR, inelastic_model="continuous").validate_transport("frozen")


@pytest.mark.parametrize(
    "field, value",
    [
        ("channeling_basis_size", 0),
        ("channeling_bloch_cells", True),
        ("channeling_dz_ang", -1.0),
        ("channeling_zone_max_ang", float("inf")),
    ],
)
def test_numerics_validates_channeling_controls(field, value):
    with pytest.raises(ValueError, match=field):
        pr.Numerics(**{field: value})


# -- gates -------------------------------------------------------------------


def test_gates_refuse_high_energy_and_warn_above_30_mev():
    with pytest.raises(ValueError, match="up to 100 MeV"):
        _case(pr.Physics(**PLANAR), energy_keV=150e3)
    with pytest.warns(UserWarning, match="#439"):
        _case(pr.Physics(**PLANAR), energy_keV=40e3)


def test_gates_refuse_crystals_without_thermal_data():
    from pyrite.materials.channeling import ChannelingDataError

    scene = pr.Scene(
        pr.Beam(energy_keV=5000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "hopg", thickness_ang=20000.0, tilt_deg=3.0, footprint=pr.geometry.Footprint(5.0, 5.0)
        ),
        pr.Detector(),
        physics=pr.Physics(**{**PLANAR, "channeling_plane": (0, 0, 1)}),
    )
    with pytest.raises(ChannelingDataError, match="no channeling thermal data"):
        api.build_case(scene, pr.Numerics())


def test_gates_refuse_no_bound_state(monkeypatch):
    from pyrite.materials import channeling as static

    real = static.transverse_states

    def unbound(*args, **kwargs):
        states = real(*args, **kwargs)
        return replace(states, energies_eV=states.energies_eV + 1e6)

    monkeypatch.setattr(static, "transverse_states", unbound)
    with pytest.raises(ValueError, match="no bound transverse state"):
        _case(pr.Physics(**PLANAR))


def test_gates_warn_near_a_low_index_axis():
    with pytest.warns(UserWarning, match="omits axial channeling"):
        _case(pr.Physics(**PLANAR), tilt_deg=0.5)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"groove_spacing_ang": 1e4}, "grooved entrance face"),
        ({"coherent_emission": True}, "coherent emission"),
        ({"mosaic_fwhm_rad": 1e-3}, "mosaic"),
        ({"abs_layers": [(0.0, 1e4, [("Si", 0.05)])]}, "single-layer slab"),
        ({"energy_model": "midpoint", "channeling": {"model": "planar"}}, "record with keys"),
    ],
)
def test_case_refuses_unsupported_geometry(planar_case, changes, message):
    with pytest.raises(ValueError, match=message):
        Case(**{**planar_case.to_dict(), **changes})


# -- consumer refusals -------------------------------------------------------


@pytest.mark.parametrize(
    "options, message",
    [
        ({"trajectory_capture": object()}, "trajectory capture"),
        ({"block_electrons": 4}, "electron-block transport"),
    ],
)
def test_transport_options_refuse_before_work(planar_case, monkeypatch, options, message):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("transport started")

    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", forbidden)
    monkeypatch.setattr(runner, "_case_transport_core", lambda case, core: core)
    with pytest.raises(ValueError, match=message):
        runner._transport_case(planar_case, **options)


def test_adaptive_refuses_before_work(planar_case):
    from pyrite.montecarlo.runner.adaptive import transport_case_adaptive

    with pytest.raises(ValueError, match="adaptive precision"):
        transport_case_adaptive(planar_case, pr.config.Precision(0.1, 40, 60, 20))


# -- event contract ------------------------------------------------------------


def _rows(kinds, *, first_direction=(0.0, 0.0, 1.0)):
    """Two straight rows of one electron, joined end to start."""
    v0 = np.array(first_direction) / np.linalg.norm(first_direction)
    v1 = np.array([0.01, 0.0, 1.0]) / np.linalg.norm([0.01, 0.0, 1.0])
    L = np.array([100.0, 50.0])
    start0 = np.zeros(3)
    end0 = start0 + L[0] * v0
    mid = np.array([start0 + 0.5 * L[0] * v0, end0 + 0.5 * L[1] * v1])
    return {
        "r_mid": mid,
        "v_hat": np.array([v0, v1]),
        "L_ang": L,
        "E_start_keV": np.array([5000.0, 4999.0]),
        "E_end_keV": np.array([4999.0, 4998.0]),
        "t_start_ang": np.array([0.0, 100.0]),
        "t_end_ang": np.array([100.0, 150.0]),
        "electron_id": np.zeros(2, dtype=np.int64),
        "flight_id": np.array([0, 1]),
        "substep_id": np.zeros(2, dtype=np.int64),
        "event_kind": np.array(kinds, dtype=np.int8),
    }


def test_dechannel_row_satisfies_the_event_contract():
    rows = _rows([SegmentEvent.DECHANNEL, SegmentEvent.EXIT_BOTTOM])
    check_segment_event_contract(rows)  # direction may change; the rest is continuous
    rows["E_start_keV"][1] = 4990.0
    with pytest.raises(ValueError, match="energy is discontinuous"):
        check_segment_event_contract(rows)


def test_dechannel_row_must_open_its_electron():
    rows = _rows([SegmentEvent.ELASTIC, SegmentEvent.DECHANNEL])
    with pytest.raises(ValueError, match="dechanneling row is not the electron's first"):
        check_segment_event_contract(rows)
