import inspect
from importlib import resources

import numpy as np
import pytest

from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.spectrum import (
    _clip_segments_to_cutoff,
    mc_brem_spectrum,
)
from pyrite.montecarlo.spectrum.lines import _prepare_spectrum
from pyrite.montecarlo.transport import simulate_trajectories, spliced_stopping_keV_per_ang
from tests.helpers import scaled_rtol, to_host

CARBON = [("C", 0.1136)]


def _carbon_stopping_keV_per_ang(E_keV):
    """Stopping magnitude [keV/Ang], taken from the model the cores evaluate.

    5 keV is already above carbon's Joy--Luo/Berger--Seltzer crossover, so an
    oracle that restated one branch would not describe what transport does.
    """
    return -spliced_stopping_keV_per_ang(CARBON, E_keV)


@pytest.mark.parametrize("transport_core", ["lockstep", "per-electron"])
def test_cutoff_crossing_clips_terminal_flight_exactly(transport_core):
    E0 = 5.01
    cutoff = 5.0
    result = simulate_trajectories(
        E0,
        1,
        1.0e8,
        composition=CARBON,
        E_cut_keV=cutoff,
        elastic_model="sr",
        seed=4,
        max_steps=20,
        transport_core=transport_core,
    )

    terminal_start = result["E_keV"][-1]
    expected_length = (terminal_start - cutoff) / _carbon_stopping_keV_per_ang(terminal_start)
    assert result["L_ang"][-1] == pytest.approx(expected_length, rel=1e-10)
    assert terminal_start - _carbon_stopping_keV_per_ang(terminal_start) * result["L_ang"][-1] == (
        pytest.approx(cutoff, abs=1e-10)
    )
    assert result["n_cutoff_stopped"] == result["n_stopped"] == 1
    assert result["n_step_limited"] == 0
    assert (
        result["n_backscattered"]
        + result["n_transmitted"]
        + result["n_side_exited"]
        + result["n_missed"]
        + result["n_cutoff_stopped"]
        == result["Ne"]
    )


@pytest.mark.parametrize("transport_core", ["lockstep", "per-electron"])
def test_step_limited_histories_raise_with_incomplete_count(transport_core):
    with pytest.raises(
        RuntimeError,
        match=r"n_step_limited=3, Ne=3, max_steps=1",
    ):
        simulate_trajectories(
            30.0,
            3,
            1.0e8,
            composition=CARBON,
            E_cut_keV=5.0,
            elastic_model="sr",
            seed=2,
            max_steps=1,
            transport_core=transport_core,
        )


def test_grooved_step_limited_histories_raise_with_incomplete_count():
    groove = blazed_groove_spec(
        spacing_ang=2.0e4,
        theta_obs_rad=np.pi / 2.0,
        tilt_polar_rad=np.pi / 4.0,
        tilt_azim_rad=np.pi,
    )
    with pytest.raises(RuntimeError, match=r"n_step_limited=2, Ne=2, max_steps=1"):
        simulate_trajectories(
            30.0,
            2,
            1.0e8,
            composition=CARBON,
            E_cut_keV=5.0,
            elastic_model="sr",
            seed=2,
            max_steps=1,
            groove=groove,
            transport_core="lockstep",
        )


@pytest.mark.parametrize("E0", [0.0, -1.0, np.nan, np.inf])
def test_initial_energy_must_be_finite_and_positive(E0):
    with pytest.raises(ValueError, match="E0_keV must be finite and strictly positive"):
        simulate_trajectories(E0, 1, 100.0, composition=CARBON)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, np.nan, np.inf, 30.0, 31.0])
def test_cutoff_must_be_finite_positive_and_below_initial_energy(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        simulate_trajectories(30.0, 1, 100.0, composition=CARBON, E_cut_keV=cutoff)


@pytest.mark.parametrize(
    "beam_dir, message",
    [
        ([0.0, 0.0, 0.0], "nonzero"),
        ([0.0, np.nan, 1.0], "finite three-vector"),
        ([0.0, 1.0], "finite three-vector"),
        ([0.0, 0.0, -1.0], "point into the slab"),
    ],
)
def test_beam_direction_must_be_finite_nonzero_inward(beam_dir, message):
    with pytest.raises(ValueError, match=message):
        simulate_trajectories(30.0, 1, 100.0, composition=CARBON, beam_dir=beam_dir)


def test_population_cutoff_clips_length_and_midpoint_but_not_start_state():
    E0 = 10.0
    cutoff = 9.99
    old_length = 1.0e5
    segments = {
        "r_mid": np.array([[0.0, 0.0, old_length / 2.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([old_length]),
        "E_keV": np.array([E0]),
        "t_ang": np.array([12.0]),
        "t0_ang": np.array([3.0]),
        "elec_id": np.array([0]),
        "layer": np.array([0]),
        "Ne": 1,
    }

    # The clip gathers with a backend index, so its rows come back on whatever
    # device the session selected; compare them on the host.
    raw = _clip_segments_to_cutoff(segments, cutoff, CARBON)
    clipped = {key: to_host(raw[key]) for key in ("L_ang", "r_mid", "E_keV", "t_ang", "t0_ang")}
    expected_length = (E0 - cutoff) / _carbon_stopping_keV_per_ang(E0)
    # ``E - E_cut`` is a deliberate near-cancellation here (10.0 - 9.99), so the
    # clipped length inherits a condition number of E0/(E0 - cutoff) = 1000 on
    # the working precision. fp64 keeps approx's own 1e-6 default; float32 gets
    # 1000 eps ~ 1.2e-4 from that ratio, not from the observed 2.3e-5.
    rel = scaled_rtol(1e-6, eps_multiple=E0 / (E0 - cutoff))
    assert clipped["L_ang"][0] == pytest.approx(expected_length, rel=rel)
    assert clipped["r_mid"][0, 2] == pytest.approx(expected_length / 2.0, rel=rel)
    assert clipped["E_keV"][0] == E0
    assert clipped["t_ang"][0] == 12.0
    assert clipped["t0_ang"][0] == 3.0

    # The line path applies the cutoff in its preparation phase, before any
    # segment array is staged; the brem path still does it inline.
    expected_call = "segments = _clip_segments_to_cutoff(segments, E_cut_keV"
    assert expected_call in inspect.getsource(_prepare_spectrum)
    assert expected_call in inspect.getsource(mc_brem_spectrum)


def test_cuda_source_uses_cpu_reference_cutoff_and_termination_rules():
    from pyrite.montecarlo.transport.cores import make_cpu_transport_core

    cpu_source = inspect.getsource(make_cpu_transport_core)
    # Read the CUDA kernel modules' source as text rather than importing them:
    # they do a module-scope `import cupy`, so an ordinary import would fail
    # outright on a CPU-only machine. The control flow lives in `_jit_kernel.py`
    # and the exit codes it writes in `_jit_device.py`, so both are read.
    transport_package = resources.files("pyrite.montecarlo.transport")
    cuda_source = "\n".join(
        transport_package.joinpath(name).read_text()
        for name in ("_jit_device.py", "_jit_kernel.py")
    )
    for source in (cpu_source, cuda_source):
        assert "cutoff_distance = (E_cut_e - E_j) / dEds" in source
        assert "cutoff_distance < step_j" in source
        assert "cutoff_distance == step_j and not geometry_event" in source
        assert "exit_side_j = False" in source
    assert "I8_CUTOFF_STOPPED = np.int8(0)" in cuda_source
    assert "I8_STEP_LIMITED = np.int8(4)" in cuda_source
    assert "I8_NOT_ENTERED = np.int8(5)" in cuda_source
