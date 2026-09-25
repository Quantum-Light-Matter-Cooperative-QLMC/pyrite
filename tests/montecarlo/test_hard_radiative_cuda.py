"""Coupled BremsLib radiative transport on the exact CUDA kernel.

The device-free half runs both exact CPU cores in
``test_hard_radiative_transport.py``. These tests need a CUDA device and run
through ``pyrite remote``.

Parity scope follows ``test_shell_soft_hard_cuda.py``: the CUDA kernel and the
per-electron CPU core draw the same counter-addressed uniforms, so each
electron's first row agrees to libm rounding; later rows are compared in
aggregate.

Validation: bremslib-radiative-partition, bremslib-radiative-event-spectrum
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    check_segment_event_contract,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.events import EVENT_CUTOFF, EVENT_HARD_RADIATIVE
from tests.helpers.bremslib import synthetic_bremslib_arrays

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

NO_LUT = TransportLUTConfig(enabled=False)
CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)


def _table(scale):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * scale,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * scale,
    )


def _run(*, scale=1e5, Ne=256, seed=42, **kw):
    return simulate_trajectories(
        E0_keV=60.0,
        Ne=Ne,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=seed,
        energy_model="midpoint",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": _table(scale)},
        **kw,
    )


def _first_rows(result):
    return np.flatnonzero(np.diff(result["electron_id"], prepend=-1))


def _photon_rows(result):
    kind = result["event_kind"]
    return (kind == EVENT_HARD_RADIATIVE) | (
        (kind == EVENT_CUTOFF) & (result["hard_radiative_k_eV"] > 0.0)
    )


def test_cuda_radiative_mode_is_deterministic():
    a = _run(**CUDA)
    b = _run(**CUDA)
    for key in ("L_ang", "E_end_keV", "r_mid", "event_kind", "hard_radiative_k_eV"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_cuda_first_row_matches_the_per_electron_cpu_core():
    cpu = _run(**HOST)
    gpu = _run(**CUDA)
    c, g = _first_rows(cpu), _first_rows(gpu)
    np.testing.assert_array_equal(cpu["electron_id"][c], gpu["electron_id"][g])
    np.testing.assert_array_equal(cpu["event_kind"][c], gpu["event_kind"][g])
    np.testing.assert_array_equal(cpu["hard_radiative_Z"][c], gpu["hard_radiative_Z"][g])
    for field in ("L_ang", "E_end_keV", "hard_radiative_k_eV", "r_mid"):
        np.testing.assert_allclose(cpu[field][c], gpu[field][g], rtol=1e-12, err_msg=field)
    # The comparison must reach the photon branch, not only elastic rows.
    assert np.count_nonzero(gpu["event_kind"][g] == EVENT_HARD_RADIATIVE) > 0


def test_cuda_hard_photons_honour_contract_and_debit_energy():
    result = _run(**CUDA)
    check_segment_event_contract(result)
    kind = result["event_kind"]
    photons = result["hard_radiative_k_eV"]
    rows = _photon_rows(result)
    assert np.count_nonzero(kind == EVENT_HARD_RADIATIVE) > 20
    assert np.all(photons[rows] >= 1_000.0)
    assert np.all(photons[~rows] == 0.0)
    assert np.all(result["hard_radiative_Z"][rows] == 6)
    order = np.lexsort((result["substep_id"], result["flight_id"], result["electron_id"]))
    hard = np.flatnonzero((kind == EVENT_HARD_RADIATIVE)[order])
    a, b = order[hard], order[hard + 1]
    assert np.all(result["electron_id"][a] == result["electron_id"][b])
    # The photon debits energy and leaves the electron direction unchanged.
    drop_keV = result["E_end_keV"][a] - result["E_start_keV"][b]
    np.testing.assert_allclose(drop_keV, photons[a] * 1e-3, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result["v_hat"][a], result["v_hat"][b])
    absorbed = (kind == EVENT_CUTOFF) & (photons > 0.0)
    assert np.all(result["E_end_keV"][absorbed] - photons[absorbed] * 1e-3 <= 10.0)


def test_cuda_ensemble_agrees_with_the_cpu_core():
    """Aggregate agreement past the first row, at five binomial/Poisson sigma."""
    Ne = 4000
    common = dict(Ne=Ne, seed=7, scale=3e4)
    cpu = _run(**HOST, **common)
    gpu = _run(**CUDA, **common)
    for name in ("n_backscattered", "n_transmitted", "n_cutoff_stopped"):
        p = 0.5 * (cpu[name] + gpu[name]) / Ne
        sigma = np.sqrt(2.0 * max(p * (1.0 - p), 1.0 / Ne) / Ne)
        assert abs(cpu[name] - gpu[name]) / Ne < 5.0 * sigma, name
    k_cpu = cpu["hard_radiative_k_eV"][_photon_rows(cpu)]
    k_gpu = gpu["hard_radiative_k_eV"][_photon_rows(gpu)]
    assert abs(k_cpu.size - k_gpu.size) < 5.0 * np.sqrt(k_cpu.size + k_gpu.size)
    spread = np.sqrt(k_cpu.var() / k_cpu.size + k_gpu.var() / k_gpu.size)
    assert abs(k_cpu.mean() - k_gpu.mean()) < 5.0 * spread
