"""#994 milestone 4a: correlated axial fates enter the native launch seam once."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_outgoing_launch import (
    AxialHandoffRecord,
    AxialIncident,
    AxialLaunch,
)
from pyrite.montecarlo.runner.axial_native import axial_native_plan
from pyrite.quantum._admission import AxialAdmissionError


def incident(i):
    return AxialIncident(i, (0.2, 0.3), 3.0, 1.0, 2**63 + i)


def outgoing(i, *, z=4.0):
    return AxialHandoffRecord(
        i, "physical_outgoing", z,
        launch=AxialLaunch(
            incident_id=i, source="outgoing", depth_ang=z,
            r_ang=np.array([i, 0.0, z]), v_hat=np.array([0.0, 0.0, 1.0]),
            E_keV=3500.0, t_ang=3.0 + z, t0_ang=1.0,
            stream_key=2**63 + i, k_perp_inv_ang=(0.0, 0.0),
            lateral_bound_ang=0.0, clock_bound_ang=0.0,
        ),
    )


def test_one_draw_per_primary_and_exact_native_stream_keys():
    records = [AxialHandoffRecord(2, "unresolved", 8.0), outgoing(1),
               AxialHandoffRecord(3, "retained", 10.0), outgoing(0, z=0.0)]
    plan = axial_native_plan(records, [incident(i) for i in (3, 1, 2, 0)], thickness_ang=10.0)
    assert plan.launch_ids.tolist() == [0, 1]
    assert plan.exit_ids.tolist() == [3]
    assert plan.unresolved_ids.tolist() == [2]
    assert plan.launch.stream_keys.tolist() == [2**63, 2**63 + 1]
    np.testing.assert_array_equal(plan.launch.r_ang[:, 2], [0.0, 4.0])
    np.testing.assert_array_equal(plan.launch.t_ang, [3.0, 7.0])
    assert plan.provenance() == {
        "n_fibre_primaries": 4, "n_fibre_outgoing_native": 2,
        "n_fibre_retained_exit": 1, "n_fibre_unresolved": 1,
    }
    assert sum(plan.provenance().values()) - plan.n_primaries == 4


def test_no_outgoing_needs_no_fake_native_primary():
    plan = axial_native_plan(
        [AxialHandoffRecord(0, "retained", 10.0),
         AxialHandoffRecord(1, "unresolved", 7.0)],
        [incident(0), incident(1)], thickness_ang=10.0,
    )
    assert plan.launch is None
    assert plan.launch_ids.size == 0
    assert plan.exit_ids.tolist() == [0]
    assert plan.unresolved_ids.tolist() == [1]


def test_outgoing_exactly_at_back_face_is_not_retransported():
    plan = axial_native_plan([outgoing(0, z=10.0)], [incident(0)], thickness_ang=10.0)
    assert plan.launch is None
    assert plan.exit_ids.tolist() == [0]


@pytest.mark.parametrize("records,people,message", [
    ([outgoing(0), outgoing(0)], [incident(0), incident(1)], "exactly once"),
    ([outgoing(1)], [incident(0)], "exactly once"),
    ([outgoing(0)], [incident(1)], "exactly once"),
    ([replace(outgoing(0), fate="retained")], [incident(0)], "retained"),
    ([replace(outgoing(0), fate="unresolved")], [incident(0)], "cannot launch"),
    ([replace(outgoing(0), depth_ang=20.0)], [incident(0)], "outside"),
    ([replace(outgoing(0), fate="other")], [incident(0)], "unknown"),
    ([replace(outgoing(0), launch=replace(outgoing(0).launch, stream_key=3))],
     [incident(0)], "RNG stream"),
    ([replace(outgoing(0), launch=replace(outgoing(0).launch, E_keV=-1.0))],
     [incident(0)], "invalid phase-space"),
])
def test_invalid_or_tampered_fates_refuse(records, people, message):
    with pytest.raises(AxialAdmissionError, match=message):
        axial_native_plan(records, people, thickness_ang=10.0)
