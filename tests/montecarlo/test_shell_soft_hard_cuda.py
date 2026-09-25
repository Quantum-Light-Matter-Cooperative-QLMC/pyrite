"""Shell soft/hard inelastic transport on the exact CUDA kernel.

The device-free half (the CUDA LUT guard and runner core selection) lives in
``test_shell_soft_hard_transport.py`` and ``test_shell_soft_hard_config.py``.
These tests need a CUDA device and run through ``pyrite remote``.

Parity scope follows ``test_straggling_cuda.py``: both cores draw the same
counter-addressed uniforms and take the same branches, so each electron's
first row -- at the unperturbed start energy -- agrees to libm rounding.
Later rows can diverge once a last-bit difference moves a sampled loss, so
the rest of the run is compared in aggregate.

Validation: shell-soft-hard-transport
"""

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import SegmentEvent, check_segment_event_contract
from pyrite.montecarlo.transport.shell_transport import hard_event_energy_accounting
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

HARD = int(SegmentEvent.HARD_INELASTIC)
NO_LUT = TransportLUTConfig(enabled=False)
CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _run(key="silicon", *, Ne=64, seed=11, **kw):
    composition = (
        CATALOG.crystal(key).composition
        if key in CATALOG.crystals
        else CATALOG.media[key].composition
    )
    return simulate_trajectories(
        20.0,
        Ne,
        kw.pop("thickness_ang", 2.0e4),
        composition=composition,
        E_cut_keV=10.0,
        seed=seed,
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table(key).arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=[key],
        **kw,
    )


def _first_rows(result):
    return np.flatnonzero(np.diff(result["electron_id"], prepend=-1))


def test_cuda_shell_mode_is_deterministic():
    a = _run(**CUDA, straggling=True, max_dE_frac=0.05)
    b = _run(**CUDA, straggling=True, max_dE_frac=0.05)
    for key in ("L_ang", "E_end_keV", "r_mid", "event_kind", "hard_W_keV", "hard_channel"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("straggling", [False, True])
@pytest.mark.parametrize("key", ["silicon", "mos2"])
def test_cuda_first_row_matches_the_per_electron_cpu_core(key, straggling):
    common = dict(straggling=straggling, max_dE_frac=0.05, Ne=256)
    cpu = _run(key, **HOST, **common)
    gpu = _run(key, **CUDA, **common)
    c, g = _first_rows(cpu), _first_rows(gpu)
    np.testing.assert_array_equal(cpu["electron_id"][c], gpu["electron_id"][g])
    np.testing.assert_array_equal(cpu["event_kind"][c], gpu["event_kind"][g])
    np.testing.assert_array_equal(cpu["hard_channel"][c], gpu["hard_channel"][g])
    for field in ("L_ang", "E_end_keV", "hard_W_keV", "r_mid"):
        np.testing.assert_allclose(cpu[field][c], gpu[field][g], rtol=1e-12, err_msg=field)
    # The comparison must reach the hard branch, not only elastic rows.
    assert np.count_nonzero(gpu["event_kind"][g] == HARD) > 0


def test_cuda_hard_events_honour_contract_and_energy_bookkeeping():
    result = _run(**CUDA, straggling=True, max_dE_frac=0.05)
    check_segment_event_contract(result)
    kind = result["event_kind"]
    hard = kind == HARD
    assert hard.sum() > 50
    order = np.lexsort((result["substep_id"], result["flight_id"], result["electron_id"]))
    rows = np.flatnonzero(hard[order])
    a, b = order[rows], order[rows + 1]
    assert np.all(result["electron_id"][a] == result["electron_id"][b])
    drop = result["E_end_keV"][a] - result["E_start_keV"][b]
    np.testing.assert_allclose(drop, result["hard_W_keV"][a], rtol=1e-12, atol=1e-12)
    absorbed = (kind == int(SegmentEvent.CUTOFF)) & (result["hard_channel"] >= 0)
    assert np.all(result["E_end_keV"][absorbed] - result["hard_W_keV"][absorbed] <= 10.0)
    assert np.all(result["hard_channel"][~hard & ~absorbed] == -1)
    accounting = hard_event_energy_accounting(result, production_threshold_eV=100.0)
    np.testing.assert_allclose(
        accounting["deposit_keV"] + accounting["secondary_keV"] + accounting["binding_keV"],
        accounting["transfer_keV"],
        rtol=1e-12,
    )


def test_cuda_ensemble_agrees_with_the_cpu_core():
    """Aggregate agreement past the first row, at five binomial/Poisson sigma."""
    Ne = 4000
    common = dict(straggling=True, max_dE_frac=0.05, Ne=Ne, seed=7)
    cpu = _run(**HOST, **common)
    gpu = _run(**CUDA, **common)
    for name in ("n_backscattered", "n_cutoff_stopped"):
        p = 0.5 * (cpu[name] + gpu[name]) / Ne
        sigma = np.sqrt(2.0 * max(p * (1.0 - p), 1.0 / Ne) / Ne)
        assert abs(cpu[name] - gpu[name]) / Ne < 5.0 * sigma, name
    n_cpu = np.count_nonzero(cpu["event_kind"] == HARD)
    n_gpu = np.count_nonzero(gpu["event_kind"] == HARD)
    assert abs(n_cpu - n_gpu) < 5.0 * np.sqrt(n_cpu + n_gpu)
    w_cpu = cpu["hard_W_keV"][cpu["event_kind"] == HARD]
    w_gpu = gpu["hard_W_keV"][gpu["event_kind"] == HARD]
    spread = np.sqrt(w_cpu.var() / w_cpu.size + w_gpu.var() / w_gpu.size)
    assert abs(w_cpu.mean() - w_gpu.mean()) < 5.0 * spread
