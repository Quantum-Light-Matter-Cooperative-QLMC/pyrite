"""Mixed elastic transport on the exact CUDA kernel (#894), against the CPU core.

Both cores address draws by (stream, counter), so they share control flow until
libm rounding moves a comparison. These tests check that shared flow, the
schedule and replay contracts of the kernel's per-history state, and the
ensemble agreement of both kernel precisions. They are not the #821 equivalence
study against detailed transport.

Validation: mixed-elastic-split
Validation: mixed-elastic-hinge
Validation: gpu-transport-core
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.transport import PerElectronTransportConfig
from pyrite.montecarlo.transport.continuation import TransportContinuationConfig
from pyrite.montecarlo.transport.events import (
    EVENT_DELTA,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_SOFT_HINGE,
    check_segment_event_contract,
)
from pyrite.montecarlo.transport.mixed_elastic import MixedElasticParameters
from tests.montecarlo.collisions.test_mixed_elastic import (
    _radiative_kwargs,
    _shell_kwargs,
    _sr_table,
    _transport,
)
from tests.montecarlo.transport.test_transport_continuation_cuda import _assert_identical

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

MIXED = MixedElasticParameters()
FULL_HISTORY = PerElectronTransportConfig(seg_capacity=64, probe_electrons=0)
PRECISIONS = ["fp64", "mixed"]


def _cuda(mixed=MIXED, precision="fp64", config=FULL_HISTORY, **kwargs):
    return _transport(
        mixed,
        transport_core="cuda",
        transport_precision=precision,
        per_electron_config=config,
        **kwargs,
    )


def _continued(rows_per_round, active_histories=None):
    continuation = TransportContinuationConfig(
        rows_per_round=rows_per_round, active_histories=active_histories
    )
    return replace(FULL_HISTORY, continuation=continuation)


def _compound():
    return {
        "element": None,
        "n_atoms_per_ang3": None,
        "composition": [("Mo", 0.0186), ("Se", 0.0372)],
        "elastic_tables": [[_sr_table(42), _sr_table(34)]],
    }


def _shell_radiative():
    return {**_shell_kwargs(), **_radiative_kwargs()}


def _row_starts(result):
    return np.flatnonzero(np.diff(result["elec_id"], prepend=-1))


def _final_energies(result):
    ends = np.append(_row_starts(result)[1:], result["elec_id"].size) - 1
    return result["E_end_keV"][ends]


def _first_rows(result, count):
    """Mask of each electron's first ``count`` rows."""
    starts = _row_starts(result)
    index = np.arange(result["elec_id"].size) - np.repeat(
        starts, np.diff(np.append(starts, result["elec_id"].size))
    )
    return index < count


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(dict, id="continuous"),
        pytest.param(_compound, id="compound"),
        pytest.param(_shell_kwargs, id="shell"),
        pytest.param(_radiative_kwargs, id="radiative"),
        pytest.param(_shell_radiative, id="shell-radiative"),
    ],
)
def test_first_rows_follow_the_cpu_core(extra):
    """Same draws, branches and row contents until rounding can matter."""
    kwargs = extra()
    cpu = _transport(MIXED, Ne=200, **kwargs)
    gpu = _cuda(Ne=200, **kwargs)  # FP64: the CPU core's precision

    cpu_rows, gpu_rows = _first_rows(cpu, 4), _first_rows(gpu, 4)
    np.testing.assert_array_equal(cpu["elec_id"][cpu_rows], gpu["elec_id"][gpu_rows])
    np.testing.assert_array_equal(cpu["event_kind"][cpu_rows], gpu["event_kind"][gpu_rows])
    np.testing.assert_allclose(cpu["L_ang"][cpu_rows], gpu["L_ang"][gpu_rows], rtol=1e-9)
    np.testing.assert_allclose(cpu["E_end_keV"][cpu_rows], gpu["E_end_keV"][gpu_rows], rtol=1e-9)
    assert np.any(gpu["event_kind"][gpu_rows] == EVENT_SOFT_HINGE)


def test_zero_c1_is_the_detailed_kernel():
    """The theta_c -> 0 limit: no hinge and no extra draws, so the same rows."""
    detailed = _cuda(None)
    limit = _cuda(MixedElasticParameters(c1=0.0, c2=0.0))
    np.testing.assert_array_equal(limit["event_kind"], detailed["event_kind"])
    np.testing.assert_allclose(limit["L_ang"], detailed["L_ang"], rtol=1e-9)
    np.testing.assert_allclose(limit["E_end_keV"], detailed["E_end_keV"], rtol=1e-9)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(dict, id="continuous"),
        pytest.param(_shell_radiative, id="shell-radiative"),
    ],
)
def test_continuation_replays_the_step_state_exactly(precision, extra):
    """A yield can land between a hinge and its step's end; the rows do not move."""
    kwargs = extra()
    reference = _cuda(precision=precision, Ne=120, **kwargs)
    for config in (_continued(3, 31), _continued(256)):
        _assert_identical(reference, _cuda(precision=precision, config=config, Ne=120, **kwargs))


@pytest.mark.parametrize("precision", PRECISIONS)
def test_capacity_replay_restarts_the_step_state(precision):
    small = replace(FULL_HISTORY, seg_capacity=2)
    _assert_identical(_cuda(precision=precision), _cuda(precision=precision, config=small))


@pytest.mark.slow
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    ("extra", "thickness"),
    [
        pytest.param(dict, 2e4, id="thin"),
        pytest.param(dict, 1e6, id="stopping"),
        pytest.param(_compound, 2e4, id="compound"),
    ],
)
def test_kernel_matches_the_cpu_core_in_aggregate(precision, extra, thickness):
    kwargs = {"Ne": 20_000, "thickness": thickness, **extra()}
    cpu = _transport(MIXED, seed=101, **kwargs)
    gpu = _cuda(precision=precision, seed=202, **kwargs)

    for name in ("n_backscattered", "n_transmitted"):
        a, b = cpu[name], gpu[name]
        assert abs(a - b) < 5.0 * np.sqrt(max(a + b, 1)), name
    for event in (EVENT_SOFT_HINGE, EVENT_DELTA):
        a = np.count_nonzero(cpu["event_kind"] == event)
        b = np.count_nonzero(gpu["event_kind"] == event)
        assert abs(a - b) < 5.0 * np.sqrt(max(a + b, 1)), event
    assert gpu["L_ang"].size / cpu["L_ang"].size == pytest.approx(1.0, abs=0.03)
    # Rows of one history are correlated, so the sample unit is the history.
    for name, (a, b) in {
        "final energy": (_final_energies(cpu), _final_energies(gpu)),
        "rows": (np.bincount(cpu["elec_id"]), np.bincount(gpu["elec_id"])),
    }.items():
        spread = np.hypot(a.std() / np.sqrt(a.size), b.std() / np.sqrt(b.size))
        assert abs(a.mean() - b.mean()) < 5.0 * spread, name


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    ("extra", "Ne"),
    [
        pytest.param(_shell_kwargs, 100, id="inelastic"),
        pytest.param(_radiative_kwargs, 400, id="radiative"),
    ],
)
def test_delta_rows_continue_their_flight(precision, extra, Ne):
    """The kernel keeps a delta's flight open, as the CPU core does."""
    rows = _cuda(precision=precision, seed=23, Ne=Ne, thickness=1e6, **extra())
    kind = rows["event_kind"]
    after = np.flatnonzero(kind[:-1] == EVENT_DELTA)
    after = after[rows["electron_id"][after + 1] == rows["electron_id"][after]]
    assert after.size
    np.testing.assert_array_equal(rows["flight_id"][after + 1], rows["flight_id"][after])
    np.testing.assert_array_equal(rows["substep_id"][after + 1], rows["substep_id"][after] + 1)
    check_segment_event_contract(rows)


@pytest.mark.slow
@pytest.mark.parametrize("precision", PRECISIONS)
def test_thinned_hard_events_match_the_cpu_core(precision):
    kwargs = {"Ne": 2_000, "thickness": 1e6, **_shell_radiative()}
    cpu = _transport(MIXED, seed=303, **kwargs)
    gpu = _cuda(precision=precision, seed=404, **kwargs)
    for event in (EVENT_HARD_INELASTIC, EVENT_HARD_RADIATIVE):
        a = np.count_nonzero(cpu["event_kind"] == event)
        b = np.count_nonzero(gpu["event_kind"] == event)
        assert a > 0
        assert abs(a - b) < 5.0 * np.sqrt(a + b), event


def test_auto_routes_mixed_to_the_kernel():
    auto = _transport(MIXED, Ne=5000, thickness=2e3, transport_core="auto")
    assert np.any(auto["event_kind"] == EVENT_SOFT_HINGE)
    assert isinstance(auto["L_ang"], np.ndarray)
