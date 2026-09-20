"""The physical-event / trajectory-segment contract (issue #85).

Every midpoint row names the event that ended it. Physical events close a
flight; numerical substeps do not, and must not change what a flight radiates
beyond quadrature refinement. The checks run every host core through one
validator so the contract is shared rather than restated per core, and the CUDA
kernels are held to the same validator and to the per-electron core's codes.
"""

import numpy as np
import pytest

from pyrite.montecarlo.geometry import tilted_geometry
from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.spectrum import mc_brem_spectrum, mc_spectrum, subdivide_flights
from pyrite.montecarlo.spectrum.lines import _clip_segments_to_cutoff
from pyrite.montecarlo.transport import (
    SegmentEvent,
    TransportLUTConfig,
    check_segment_event_contract,
    closes_flight,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.events import RESERVED_EVENTS
from tests.helpers import host_backend_only, to_host

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.0632)]
_BASE = dict(
    E0_keV=25.0,
    Ne=80,
    thickness_ang=4000.0,
    composition=CARBON,
    seed=7,
    E_cut_keV=1.0,
    energy_model="midpoint",
)


def _host_rows(result):
    return {
        k: to_host(v) if isinstance(v, np.ndarray) or hasattr(v, "get") else v
        for k, v in result.items()
    }


def _kinds(result):
    return set(np.unique(to_host(result["event_kind"])).tolist())


@pytest.mark.parametrize(
    "core, lut, straggling",
    [
        ("lockstep", False, False),
        ("lockstep", True, False),
        ("lockstep", False, True),
        ("per-electron", False, False),
        ("per-electron", True, False),
        ("per-electron", False, True),
    ],
)
@pytest.mark.parametrize("max_dE_frac", [0.0, 2e-3])
def test_every_host_core_honours_the_event_contract(core, lut, straggling, max_dE_frac):
    result = simulate_trajectories(
        **_BASE,
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
        straggling=straggling,
        max_dE_frac=max_dE_frac,
    )
    check_segment_event_contract(result)
    kinds = _kinds(result)
    assert SegmentEvent.ELASTIC in kinds
    assert kinds & {SegmentEvent.EXIT_TOP, SegmentEvent.EXIT_BOTTOM}
    assert (SegmentEvent.SUBSTEP in kinds) == (max_dE_frac > 0.0)
    assert not kinds & RESERVED_EVENTS


def test_event_codes_agree_with_the_exit_tallies():
    result = simulate_trajectories(**_BASE, transport_core="per-electron")
    kind = result["event_kind"]
    assert (kind == SegmentEvent.EXIT_TOP).sum() == result["n_backscattered"]
    assert (kind == SegmentEvent.EXIT_BOTTOM).sum() == result["n_transmitted"]
    assert (kind == SegmentEvent.CUTOFF).sum() == result["n_cutoff_stopped"]
    assert (kind == SegmentEvent.EXIT_SIDE).sum() == result["n_side_exited"]


def test_side_exits_and_cutoffs_are_terminal_events():
    result = simulate_trajectories(
        **{**_BASE, "E0_keV": 10.0, "thickness_ang": 40000.0},
        transport_core="lockstep",
        crystal_width_mm=1.5e-3,
        crystal_height_mm=1.5e-3,
    )
    check_segment_event_contract(result)
    kind = result["event_kind"]
    assert result["n_side_exited"] > 0 and result["n_cutoff_stopped"] > 0
    assert (kind == SegmentEvent.EXIT_SIDE).sum() == result["n_side_exited"]
    assert (kind == SegmentEvent.CUTOFF).sum() == result["n_cutoff_stopped"]


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
def test_layer_interfaces_close_the_flight_without_turning_the_electron(core):
    layers = [(0.0, 2000.0, CARBON), (2000.0, 4000.0, TUNGSTEN)]
    result = simulate_trajectories(
        **{k: v for k, v in _BASE.items() if k not in ("composition", "thickness_ang")},
        thickness_ang=4000.0,
        layers=layers,
        transport_core=core,
        max_dE_frac=2e-3,
    )
    check_segment_event_contract(result)
    assert SegmentEvent.LAYER_BOUNDARY in _kinds(result)


def test_groove_facets_are_geometry_events_with_a_vacuum_leg():
    tilt = np.deg2rad(45.0)
    beam, _ = tilted_geometry(np.pi / 2, tilt, np.pi)
    spec = blazed_groove_spec(
        spacing_ang=2.0e4, theta_obs_rad=np.pi / 2, tilt_polar_rad=tilt, tilt_azim_rad=np.pi
    )
    result = simulate_trajectories(
        E0_keV=60.0,
        Ne=200,
        thickness_ang=2.0e5,
        composition=CARBON,
        seed=42,
        elastic_model="sr",
        beam_dir=beam,
        tilt_polar_rad=tilt,
        tilt_azim_rad=np.pi,
        groove=spec,
        energy_model="midpoint",
        max_dE_frac=5e-3,
    )
    check_segment_event_contract(result)
    kind = result["event_kind"]
    n_surface = int((kind == SegmentEvent.GROOVE_SURFACE).sum())
    # Every re-entry is a surface event; the rest escaped through a facet.
    assert n_surface >= result["vacuum_elec_id"].size > 0


def test_frozen_rows_keep_their_historical_schema():
    frozen = simulate_trajectories(**{**_BASE, "energy_model": "frozen"}, transport_core="lockstep")
    assert "event_kind" not in frozen
    with pytest.raises(ValueError, match="event-contract fields"):
        check_segment_event_contract(frozen)


def test_frozen_and_midpoint_rows_close_flights_at_the_same_places():
    """Elastic-only trajectories keep their physical meaning: without a step cap
    every midpoint row is a whole flight, exactly as every frozen row is."""
    midpoint = simulate_trajectories(**_BASE, transport_core="per-electron")
    assert closes_flight(midpoint["event_kind"]).all()
    assert np.all(midpoint["substep_id"] == 0)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["event_kind"].__setitem__(0, 42), "unknown event_kind"),
        (lambda r: r["flight_id"].__iadd__(1), "flight 0"),
        (lambda r: r["substep_id"].__setitem__(r["substep_id"] > 0, 0), "substep"),
        (lambda r: r["E_start_keV"].__imul__(0.999), "energy"),
        (lambda r: r["t_start_ang"].__iadd__(1.0), "clock"),
    ],
)
def test_the_validator_rejects_broken_rows(mutate, message):
    rows = simulate_trajectories(**_BASE, transport_core="per-electron", max_dE_frac=2e-3)
    rows = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        check_segment_event_contract(rows)


def test_a_flight_continuing_event_must_keep_the_direction():
    rows = simulate_trajectories(**_BASE, transport_core="per-electron", max_dE_frac=2e-3)
    rows = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    j = int(np.flatnonzero(rows["event_kind"] == SegmentEvent.SUBSTEP)[0])
    rows["v_hat"][j + 1] = rows["v_hat"][j + 1][::-1]
    with pytest.raises(ValueError, match="direction"):
        check_segment_event_contract(rows)


def test_reserved_hard_events_are_physical_boundaries():
    """The contract the hard-event slices (#93/#95) must meet, pinned on
    synthetic rows: a hard radiative event opens a new flight, keeps the
    direction, and may only lower the energy; a delta event is a substep."""
    v = np.array([[0.0, 0.0, 1.0]] * 3)
    rows = {
        "r_mid": np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 15.0], [0.0, 0.0, 25.0]]),
        "v_hat": v,
        "L_ang": np.full(3, 10.0),
        "E_start_keV": np.array([30.0, 29.0, 20.0]),
        "E_end_keV": np.array([29.0, 28.5, 19.5]),
        "t_start_ang": np.array([0.0, 30.0, 60.0]),
        "t_end_ang": np.array([30.0, 60.0, 90.0]),
        "electron_id": np.zeros(3, dtype=np.int64),
        "flight_id": np.array([0, 0, 1]),
        "substep_id": np.array([0, 1, 0]),
        "event_kind": np.array(
            [SegmentEvent.DELTA, SegmentEvent.HARD_RADIATIVE, SegmentEvent.EXIT_BOTTOM],
            dtype=np.int8,
        ),
    }
    check_segment_event_contract(rows)
    assert closes_flight(rows["event_kind"]).tolist() == [False, True, True]

    turned = dict(rows, v_hat=v.copy())
    turned["v_hat"][2] = [0.0, 0.6, 0.8]
    turned["r_mid"] = rows["r_mid"].copy()
    turned["r_mid"][2] = [0.0, 3.0, 24.0]
    with pytest.raises(ValueError, match="direction"):
        check_segment_event_contract(turned)

    gained = dict(rows, E_start_keV=np.array([30.0, 29.0, 29.0]))
    with pytest.raises(ValueError, match="rises"):
        check_segment_event_contract(gained)


def test_subdivision_keeps_the_events_and_the_contract():
    flights = simulate_trajectories(**_BASE, transport_core="lockstep")
    rows, parent = subdivide_flights(flights, composition=CARBON, max_dE_frac=1e-3)
    assert rows["L_ang"].size > flights["L_ang"].size
    # Each parent is re-integrated on its own, so its last piece lands on the
    # next flight's transported start energy only to quadrature accuracy.
    check_segment_event_contract(rows, rtol=1e-4)
    last_piece = np.append(parent[1:] != parent[:-1], True)
    np.testing.assert_array_equal(rows["event_kind"][last_piece], flights["event_kind"])
    assert np.all(rows["event_kind"][~last_piece] == SegmentEvent.SUBSTEP)


def test_a_consumer_cutoff_becomes_the_clipped_rows_end_event():
    flights = simulate_trajectories(**_BASE, transport_core="per-electron", max_dE_frac=2e-3)
    clipped = _host_rows(_clip_segments_to_cutoff(flights, 24.0, CARBON))
    check_segment_event_contract(clipped, rtol=1e-6, atol_ang=1e-2)
    assert SegmentEvent.CUTOFF in _kinds(clipped)


@host_backend_only(
    "flight-grouped incoherent CXR: the segmented complex reduction over "
    "numerical substeps has no device port"
)
def test_numerical_subdivision_alone_does_not_change_the_radiation():
    """Acceptance: splitting flights into substeps changes CXR and bremsstrahlung
    only by quadrature refinement, while the physical events -- and so the
    incoherent partition -- are identical at every rung.

    Stated tolerance, relative grid L1 against the finest rung (f = 1e-4): the
    refined rungs converge monotonically, and at f = 5e-4 CXR is within 1e-3
    and bremsstrahlung within 1e-4 (measured 6.5e-4 and 1.1e-5). The unrefined
    rung's 6% CXR offset is the one-point flight error that
    ``substep-radiation-invariance`` documents, not a decoherence effect.
    """
    kwargs = {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "theta_obs_rad": np.deg2rad(119.0),
        "composition": CARBON,
    }
    cxr_grid = np.arange(880.0, 1070.0, 0.5)
    brem_grid = np.linspace(200.0, 20000.0, 200)
    flights = simulate_trajectories(**{**_BASE, "Ne": 40}, transport_core="lockstep")

    ladder = (1e-3, 5e-4, 1e-4)
    cxr, brem = [], []
    for frac in ladder:
        rows, _ = subdivide_flights(
            flights, composition=CARBON, max_dE_frac=frac, max_substeps=1024
        )
        closing = closes_flight(rows["event_kind"])
        np.testing.assert_array_equal(rows["event_kind"][closing], flights["event_kind"])
        cxr.append(mc_spectrum(rows, cxr_grid, **kwargs))
        brem.append(mc_brem_spectrum(rows, brem_grid, composition=CARBON))

    def l1(values, ref):
        return np.abs(values - ref).sum() / np.abs(ref).sum()

    cxr_err = [l1(c, cxr[-1]) for c in cxr[:-1]]
    brem_err = [l1(b, brem[-1]) for b in brem[:-1]]
    assert cxr_err[1] < cxr_err[0] and brem_err[1] < brem_err[0]
    assert cxr_err[1] <= 1e-3
    assert brem_err[1] <= 1e-4


# ---- CUDA port ---------------------------------------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("lut", [False, True])
def test_cuda_rows_honour_the_event_contract(lut):
    result = simulate_trajectories(
        **{**_BASE, "Ne": 1200},
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=lut),
        max_dE_frac=2e-3,
    )
    check_segment_event_contract(result)
    kind = to_host(result["event_kind"])
    assert (kind == SegmentEvent.EXIT_TOP).sum() == result["n_backscattered"]
    assert (kind == SegmentEvent.EXIT_BOTTOM).sum() == result["n_transmitted"]
    assert (kind == SegmentEvent.CUTOFF).sum() == result["n_cutoff_stopped"]
    assert SegmentEvent.SUBSTEP in set(kind.tolist())


@pytest.mark.hardware
@requires_cuda
def test_cuda_first_row_event_matches_the_cpu_reference():
    """Same addressing and branch order: the first row of each electron ends in
    the same event on both cores (later rows may diverge by few-ulp drift)."""
    common = dict(**{**_BASE, "Ne": 1200}, max_dE_frac=2e-3)
    cpu = simulate_trajectories(**common, transport_core="per-electron")
    gpu = simulate_trajectories(**common, transport_core="cuda")

    def first_events(r):
        eid = to_host(r["electron_id"])
        _, first = np.unique(eid, return_index=True)
        return to_host(r["event_kind"])[first]

    np.testing.assert_array_equal(first_events(cpu), first_events(gpu))
