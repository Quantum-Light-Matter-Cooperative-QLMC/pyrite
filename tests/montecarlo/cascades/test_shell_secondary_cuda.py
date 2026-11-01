"""Secondary transport (#94) on the exact CUDA kernel.

The device-free half lives in ``test_shell_secondary_transport.py``. These
tests need a CUDA device and run through ``pyrite remote``. Parity scope follows
``test_shell_soft_hard_cuda.py``: each electron's first row agrees with the
per-electron CPU core to libm rounding, and whole cascades agree in aggregate.

Validation: shell-secondary-transport
"""

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from pyrite.montecarlo.transport.events import check_segment_event_contract
from pyrite.montecarlo.transport.secondaries import secondary_energy_balance
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

NO_LUT = TransportLUTConfig(enabled=False)
CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _run(key="silicon", *, Ne=64, seed=11, threshold=1000.0, **kw):
    composition = (
        CATALOG.crystal(key).composition
        if key in CATALOG.crystals
        else CATALOG.media[key].composition
    )
    return simulate_trajectories(
        30.0,
        Ne,
        kw.pop("thickness_ang", 3.0e4),
        composition=composition,
        E_cut_keV=2.0,
        seed=seed,
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table(key).arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=[key],
        secondary_threshold_eV=threshold,
        **kw,
    )


def _host(result):
    return {
        k: (v if isinstance(v, dict) or not hasattr(v, "get") else v.get())
        for k, v in result.items()
    }


def _first_rows(result):
    primary = np.flatnonzero(result["generation"] == 0)
    ids = result["electron_id"][primary]
    return primary[np.diff(ids, prepend=-1) != 0]


@pytest.mark.parametrize("key", ["silicon", "mos2"])
def test_cuda_secondary_direction_matches_the_cpu_first_row(key):
    common = dict(straggling=True, max_dE_frac=0.05, Ne=256)
    cpu = _run(key, **HOST, **common)
    gpu = _run(key, **CUDA, **common)
    c, g = _first_rows(cpu), _first_rows(gpu)
    np.testing.assert_array_equal(cpu["hard_channel"][c], gpu["hard_channel"][g])
    hard = cpu["hard_channel"][c] >= 0
    assert np.count_nonzero(hard) > 0
    np.testing.assert_allclose(
        cpu["hard_secondary_v_hat"][c], gpu["hard_secondary_v_hat"][g], rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(gpu["hard_secondary_v_hat"][g][~hard], 0.0)


def test_cuda_cascade_is_deterministic_and_balances_energy():
    a = _run(**CUDA, straggling=True, max_dE_frac=0.05, Ne=128)
    b = _run(**CUDA, straggling=True, max_dE_frac=0.05, Ne=128)
    assert a["secondaries"]["n_generations"] >= 2
    for key in ("L_ang", "E_end_keV", "r_mid", "track_id", "parent_id", "hard_secondary_v_hat"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    check_segment_event_contract(dict(a, electron_id=a["track_id"]))
    terms = secondary_energy_balance(a)
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]


def test_cuda_resident_cascade_matches_downloaded_rows():
    common = dict(straggling=True, max_dE_frac=0.05, Ne=96, **CUDA)
    host = _run(**common)
    resident = _host(_run(keep_segments_on_device=True, **common))
    assert resident["secondaries"] == host["secondaries"]
    for key in ("L_ang", "E_end_keV", "track_id", "generation", "hard_secondary_v_hat"):
        np.testing.assert_array_equal(resident[key], host[key], err_msg=key)


def test_cuda_cascade_ensemble_agrees_with_the_cpu_core():
    """Generation sizes and launch energies at five Poisson/standard-error sigma."""
    common = dict(straggling=True, max_dE_frac=0.05, Ne=3000, seed=7)
    cpu = _run(**HOST, **common)
    gpu = _run(**CUDA, **common)
    n_cpu = cpu["secondaries"]["tracks_per_generation"][1]
    n_gpu = gpu["secondaries"]["tracks_per_generation"][1]
    assert n_cpu > 100 and abs(n_cpu - n_gpu) < 5.0 * np.sqrt(n_cpu + n_gpu)
    e_cpu = cpu["secondary_tracks"]["launch_E_keV"][cpu["secondary_tracks"]["generation"] > 0]
    e_gpu = gpu["secondary_tracks"]["launch_E_keV"][gpu["secondary_tracks"]["generation"] > 0]
    spread = np.sqrt(e_cpu.var() / e_cpu.size + e_gpu.var() / e_gpu.size)
    assert abs(e_cpu.mean() - e_gpu.mean()) < 5.0 * spread
    for g in (cpu, gpu):
        terms = secondary_energy_balance(g)
        assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
