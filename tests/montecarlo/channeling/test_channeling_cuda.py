"""CPU zone solve with CUDA LaunchState continuation (#521).

Device tests run through the lab scheduler. Integer identities and zone rows
are exact; native first-flight parity allows CPU/CUDA libm roundoff.
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import _ABSENT
from pyrite.montecarlo.transport import check_segment_event_contract
from pyrite.units import ureg

try:
    import cupy

    HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("fp64_transport")]
cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


def _case(*, elastic="elsepa", plane=(1, -1, 0), unit_bearing=False):
    energy_keV = 5.0 * ureg.MeV if unit_bearing else 5000.0
    scene = pr.Scene(
        pr.Beam(energy_keV, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "silicon",
            10000.0,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=pr.Physics(
            channeling_model="planar",
            channeling_plane=plane,
            radiative_model="uncoupled",
            inelastic_model="continuous",
            elastic_model=elastic,
        ),
    )
    numerics = pr.Numerics(
        cxr_trials=32,
        brem_trials=48,
        channeling_basis_size=8,
        channeling_bloch_cells=4,
        channeling_dz_ang=500.0,
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 3000.0, 31), brem=np.array([5000.0, 10000.0])
        ),
    )
    return replace(api.build_case(scene, numerics), seed=17)


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_cuda_selection_keeps_launch_streams_and_downloads_before_zone_join(monkeypatch, requested):
    case = _case()
    expected = runner._transport_case(case, transport_core="per-electron")["segs"]
    original = runner.simulate_trajectories
    calls = []

    def native(*args, **kwargs):
        assert kwargs["transport_core"] == "cuda"
        assert not kwargs["keep_segments_on_device"]
        launch = kwargs["_secondary"].launch
        assert launch.stream_keys.size == args[1]
        calls.append(launch)
        # Exercise the real launch/join path without a local CUDA device.
        return original(*args, **{**kwargs, "transport_core": "per-electron"})

    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(runner, "simulate_trajectories", native)
    actual = runner._transport_case(case, transport_core=requested, keep_segments_on_device=True)[
        "segs"
    ]
    assert calls
    assert actual["channeling"] == expected["channeling"]
    for key in ("electron_id", "flight_id", "event_kind", "L_ang", "r_mid", "v_hat", "E_end_keV"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _first_native(segs):
    rows = np.flatnonzero(~segs["channeling_zone"])
    ids = segs["electron_id"][rows]
    return rows[np.diff(ids, prepend=-1) != 0]


@cuda
@pytest.mark.hardware
@pytest.mark.parametrize("elastic", ["mott", "elsepa"], ids=["lut", "exact"])
def test_cuda_zone_handoff_matches_cpu_and_preserves_seeded_rows(monkeypatch, elastic):
    from pyrite.montecarlo.transport import _jit_launch

    case = _case(elastic=elastic)
    cpu = runner._transport_case(case, transport_core="per-electron")
    calls = []
    factory_name = (
        "make_cuda_transport_core" if elastic == "elsepa" else "make_cuda_transport_lut_core"
    )
    original = getattr(_jit_launch, factory_name)

    def factory(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(_jit_launch, factory_name, factory)
    gpu = runner._transport_case(case, transport_core="cuda", keep_segments_on_device=True)
    assert calls, "the requested CUDA kernel must run"
    c, g = cpu["segs"], gpu["segs"]
    check_segment_event_contract(g)
    assert g["channeling"] == c["channeling"]
    for key in ("initial_r_ang", "initial_v_hat", "initial_E_keV"):
        np.testing.assert_array_equal(g[key], c[key])
    for key in ("electron_id", "flight_id", "event_kind", "L_ang", "E_end_keV", "r_mid", "v_hat"):
        np.testing.assert_array_equal(g[key][g["channeling_zone"]], c[key][c["channeling_zone"]])
    first_c, first_g = _first_native(c), _first_native(g)
    assert first_g.size > 0
    np.testing.assert_array_equal(g["electron_id"][first_g], c["electron_id"][first_c])
    for key in ("r_mid", "v_hat", "L_ang", "E_start_keV", "E_end_keV", "t_end_ang"):
        np.testing.assert_allclose(
            g[key][first_g], c[key][first_c], rtol=2e-11, atol=1e-8, err_msg=key
        )
    repeat = runner._transport_case(case, transport_core="cuda")["segs"]
    for key in ("electron_id", "event_kind", "L_ang", "r_mid", "v_hat", "E_end_keV"):
        np.testing.assert_array_equal(g[key], repeat[key], err_msg=key)
    from pyrite.montecarlo.runner.channeling_emission import case_channeling_spectrum

    np.testing.assert_array_equal(
        case_channeling_spectrum(case, gpu), case_channeling_spectrum(case, cpu)
    )


@cuda
@pytest.mark.hardware
def test_quantity_beam_input_matches_legacy_cpu_transport_on_cuda():
    numeric_case = _case()
    quantity_case = _case(unit_bearing=True)
    cpu = runner._transport_case(numeric_case, transport_core="per-electron")
    gpu = runner._transport_case(quantity_case, transport_core="cuda", keep_segments_on_device=True)
    c, g = cpu["segs"], gpu["segs"]
    check_segment_event_contract(g)
    assert g["channeling"] == c["channeling"]
    for key in ("electron_id", "flight_id", "event_kind", "L_ang", "E_end_keV", "r_mid", "v_hat"):
        np.testing.assert_array_equal(g[key][g["channeling_zone"]], c[key][c["channeling_zone"]])
    first_c, first_g = _first_native(c), _first_native(g)
    assert first_g.size > 0
    np.testing.assert_array_equal(g["electron_id"][first_g], c["electron_id"][first_c])
    for key in ("r_mid", "v_hat", "L_ang", "E_start_keV", "E_end_keV", "t_end_ang"):
        np.testing.assert_allclose(
            g[key][first_g], c[key][first_c], rtol=2e-11, atol=1e-8, err_msg=key
        )


@cuda
@pytest.mark.hardware
def test_never_channeled_cuda_electrons_keep_the_amorphous_streams():
    case = _case(plane=(1, 1, 0))
    on = runner._transport_case(case, transport_core="cuda")["segs"]
    off = runner._transport_case(replace(case, channeling=_ABSENT), transport_core="cuda")["segs"]
    assert on["channeling"]["fates"]["entry"] == max(case["Ne"], case["Ne_brem"])
    assert not on["channeling_zone"].any()
    for key in ("electron_id", "flight_id", "event_kind", "L_ang", "r_mid", "v_hat", "E_end_keV"):
        np.testing.assert_array_equal(on[key], off[key], err_msg=key)
