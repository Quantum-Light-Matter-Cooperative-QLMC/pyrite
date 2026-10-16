"""Planar channeling zone handoff into native transport (cr-zone-handoff-accounting).

Population: every electron has exactly one fate (entry, free, overflow,
bound), so bound + handed-off + exited = 1, with frequencies matching the
ensemble solve. Energy, clock and position are continuous across the
``DECHANNEL`` row (segment contract). A never-channeled electron runs exactly
the amorphous trajectory.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import _ABSENT
from pyrite.montecarlo.channeling.handoff import (
    FATE_BOUND,
    FATE_ENTRY,
    FATE_FREE,
    FATE_OVERFLOW,
    ZoneEnsemble,
    handoff_directions,
    handoff_uniforms,
    sample_handoffs,
)
from pyrite.montecarlo.transport import SegmentEvent, check_segment_event_contract

SMALL = dict(channeling_basis_size=12, channeling_bloch_cells=4, channeling_dz_ang=500.0)
N = 200


def _case(plane=(1, -1, 0), *, n=N, numerics=SMALL, thickness_ang=10000.0):
    physics = pr.Physics(
        channeling_model="planar", channeling_plane=plane, radiative_model="uncoupled"
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=5000.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab(
            "silicon",
            thickness_ang=thickness_ang,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=pr.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=physics,
    )
    return api.build_case(scene, pr.Numerics(cxr_trials=n, brem_trials=n, **numerics))


def _ensemble(**changes):
    base = dict(
        z_ang=np.linspace(0.0, 1e4, 11),
        free_cumulative=np.linspace(0.0, 0.3, 11),
        overflow_cumulative=np.linspace(0.0, 0.1, 11) ** 1.0,
        entry=0.2,
        bound=0.4,
        balance_defect=0.0,
        energy_edges_eV=np.array([300.0, 400.0, 600.0]),
        free_by_energy=np.array([0.75, 0.25]),
        plane_normal=np.array([1.0, 0.0, 0.0]),
        pv_eV=5.4e6,
        psi_edge_rad=0.02,
        psi_critical_rad=0.003,
        barrier_eV=6.6,
        cutoff_eV=306.6,
        stopping_keV_per_ang=3.8e-5,
        path_ang=1e4,
        incidence_rad=0.0,
        divergence_rad=0.0,
    )
    return ZoneEnsemble(**{**base, **changes})


@pytest.fixture(scope="module")
def zone():
    case = _case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small basis: overflow warning
        out = runner._transport_case(case, transport_core="per-electron")
    return case, out["segs"]


# -- sampling ------------------------------------------------------------------


def test_uniforms_are_keyed_per_electron_not_per_batch():
    whole = handoff_uniforms(7, np.arange(10))
    np.testing.assert_array_equal(handoff_uniforms(7, np.arange(3, 7)), whole[3:7])
    assert np.all((whole >= 0.0) & (whole < 1.0))
    assert not np.array_equal(handoff_uniforms(8, np.arange(10)), whole)


def test_directions_set_the_plane_angle_and_keep_the_in_plane_part():
    rng = np.random.default_rng(1)
    normal = np.array([0.6, 0.8, 0.0])
    v = np.column_stack([rng.normal(0, 1e-3, 50), rng.normal(0, 1e-3, 50), np.ones(50)])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    psi = rng.uniform(0.0, 0.02, 50)
    sign = np.where(rng.random(50) < 0.5, -1.0, 1.0)
    out = handoff_directions(v, normal, psi, sign)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-14)
    np.testing.assert_allclose(np.arcsin(out @ normal), sign * psi, atol=1e-14)
    inplane = v - (v @ normal)[:, None] * normal
    out_inplane = out - (out @ normal)[:, None] * normal
    cross = np.cross(inplane, out_inplane)
    np.testing.assert_allclose(cross, 0.0, atol=1e-14)


def test_fates_partition_and_follow_the_ensemble():
    ensemble = _ensemble()
    sample = sample_handoffs(ensemble, 3, np.arange(40000))
    p = ensemble.probabilities
    counts = np.bincount(sample.fate, minlength=4)
    assert counts.sum() == 40000
    sigma = np.sqrt(40000 * p * (1 - p))
    assert np.all(np.abs(counts - 40000 * p) < 4.0 * sigma)
    free = sample.fate == FATE_FREE
    over = sample.fate == FATE_OVERFLOW
    assert np.all((sample.z_ang[free | over] >= 0.0) & (sample.z_ang[free | over] <= 1e4))
    assert np.all(np.isinf(sample.z_ang[sample.fate == FATE_BOUND]))
    assert np.all(sample.z_ang[sample.fate == FATE_ENTRY] == 0.0)
    e = sample.transverse_eV[free]
    assert np.all((e >= 300.0) & (e <= 600.0))
    # The lower bin carries 3/4 of the binned free flux.
    share = np.mean(e < 400.0)
    assert abs(share - 0.75) < 4.0 * np.sqrt(0.75 * 0.25 / e.size)
    np.testing.assert_allclose(sample.psi_rad[free], np.sqrt(2.0 * e / 5.4e6))
    np.testing.assert_array_equal(sample.psi_rad[over], 0.02)
    assert abs(np.mean(sample.sign > 0) - 0.5) < 0.02
    # Uniform cumulative flux in depth gives uniform depths.
    assert abs(np.mean(sample.z_ang[free]) - 5e3) < 4.0 * 1e4 / np.sqrt(12 * free.sum())


def test_no_scattering_limit_dechannels_nothing():
    flat = np.zeros(11)
    ensemble = _ensemble(free_cumulative=flat, overflow_cumulative=flat, entry=0.25, bound=0.75)
    fate = sample_handoffs(ensemble, 5, np.arange(5000)).fate
    assert set(np.unique(fate).tolist()) <= {FATE_ENTRY, FATE_BOUND}


# -- transport -----------------------------------------------------------------


def test_zone_rows_satisfy_the_event_contract_and_open_their_electrons(zone):
    _case_, segs = zone
    check_segment_event_contract(segs)
    tagged = segs["channeling_zone"]
    assert np.all(segs["flight_id"][tagged] == 0)
    kinds = set(np.unique(segs["event_kind"][tagged]).tolist())
    assert kinds <= {SegmentEvent.DECHANNEL, SegmentEvent.EXIT_BOTTOM, SegmentEvent.CUTOFF}
    # Zone loss is the constant S(E0) of the solve.
    S = segs["channeling"]["stopping_keV_per_ang"]
    np.testing.assert_allclose(
        segs["E_start_keV"][tagged] - segs["E_end_keV"][tagged],
        S * segs["L_ang"][tagged],
        rtol=1e-9,
        atol=1e-9,
    )


def test_population_accounting_closes(zone):
    _case_, segs = zone
    info = segs["channeling"]
    fates = info["fates"]
    assert sum(fates.values()) == N == segs["Ne"]
    p = np.array(list(info["probabilities"].values()))
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-12)
    assert abs(info["balance_defect"]) < 2e-2  # O(dz^2) at dz = 500 angstrom
    tagged = segs["channeling_zone"]
    handed = int(np.count_nonzero(segs["event_kind"][tagged] == SegmentEvent.DECHANNEL))
    exited = int(np.count_nonzero(segs["event_kind"][tagged] != SegmentEvent.DECHANNEL))
    assert fates["bound"] == exited
    assert fates["free"] + fates["overflow"] == handed
    assert fates["entry"] + handed + exited == N
    # Electrons still channeled at the face are transmitted, not lost.
    assert (
        segs["n_transmitted"]
        + segs["n_backscattered"]
        + segs["n_side_exited"]
        + segs["n_cutoff_stopped"]
        + segs["n_missed"]
        == N
    )


def test_handoff_rows_continue_with_the_sampled_plane_angle(zone):
    case, segs = zone
    info = segs["channeling"]
    tagged = segs["channeling_zone"]
    dechannel = tagged & (segs["event_kind"] == SegmentEvent.DECHANNEL)
    ids = segs["electron_id"][dechannel]
    nxt = np.flatnonzero(~tagged & (segs["flight_id"] == 1))
    following = {int(e): j for e, j in zip(segs["electron_id"][nxt], nxt, strict=True)}
    from pyrite.materials.channeling import plane_incidence

    normal = plane_incidence(
        "silicon", (1, -1, 0), 5e6, tilt_deg=3.0, tilt_azim_deg=45.0, rotation=np.eye(3)
    ).plane_normal
    angles = np.array([abs(np.arcsin(segs["v_hat"][following[int(e)]] @ normal)) for e in ids])
    # Free handoffs leave above the cutoff angle and inside the basis; overflow
    # at the band edge (H + 1/2) g; the basis reaches at most (H + 1) g.
    pv = 5e6 * (5e6 + 2 * 510998.95) / (5e6 + 510998.95)
    assert np.all(angles >= np.sqrt(2.0 * 300.0 / pv) * (1 - 1e-9))
    assert np.all(angles <= info["psi_edge_rad"] * 13.0 / 12.5 * (1 + 1e-9))
    assert np.any(np.isclose(angles, info["psi_edge_rad"], rtol=1e-9))


def test_never_channeled_electrons_run_the_amorphous_trajectory():
    # (110) planes are near-normal to the beam here: all entry weight is free.
    case = _case((1, 1, 0), n=30, thickness_ang=5000.0)
    on = runner._transport_case(case, transport_core="per-electron")["segs"]
    assert on["channeling"]["fates"]["entry"] == 30
    off = runner._transport_case(replace(case, channeling=_ABSENT), transport_core="per-electron")
    off = off["segs"]
    assert not on["channeling_zone"].any()
    for key in ("r_mid", "v_hat", "L_ang", "E_start_keV", "E_end_keV", "t_end_ang", "event_kind"):
        np.testing.assert_array_equal(on[key], off[key])


def test_run_case_scores_spectra_and_reports_the_zone(zone):
    case, _segs = zone
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = runner.run_case(replace(case, Ne=40, Ne_brem=40), transport_core="per-electron")
    assert np.all(np.isfinite(out["spec"])) and np.all(np.isfinite(out["brem"]))
    assert sum(out["channeling"]["fates"].values()) == 40


def test_zone_cap_shorter_than_the_path_is_refused():
    case = _case(numerics={**SMALL, "channeling_zone_max_ang": 100.0}, n=4)
    with pytest.raises(ValueError, match="forced handoff"):
        runner._transport_case(case, transport_core="per-electron")


def test_overflow_above_ten_percent_warns():
    case = _case(numerics={**SMALL, "channeling_basis_size": 6}, n=4)
    with pytest.warns(UserWarning, match="raise Numerics.channeling_basis_size"):
        runner._transport_case(case, transport_core="per-electron")
