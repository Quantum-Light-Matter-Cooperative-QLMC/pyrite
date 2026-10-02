"""Pair conversion (#275) with the cascade on the exact CUDA kernel.

The device-free half lives in ``test_pair_production.py``. These tests need a
CUDA device and run through ``pyrite remote``. The photon first-interaction
step and the pair sampler run on the host from per-photon streams, so given the
same photon rows they are identical on every core; the electron rows that feed
them agree with the per-electron CPU core only in aggregate past the first row
(``test_hard_radiative_cuda.py``). Pair counts are therefore compared in
aggregate, and the host step is replayed exactly on the CUDA rows.

Validation: photon-pair-first-interaction, pair-production-sampling
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import TransportLUTConfig, pair_production, simulate_trajectories
from pyrite.montecarlo.transport.secondaries import secondary_energy_balance
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table
from tests.helpers.bremslib import synthetic_bremslib_arrays

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
    pytest.mark.filterwarnings("ignore:.*not transported:UserWarning"),
]

NO_LUT = TransportLUTConfig(enabled=False)
CUDA = dict(transport_core="cuda", transport_lut_config=NO_LUT)
HOST = dict(transport_core="per-electron", transport_lut_config=NO_LUT)
THICKNESS_ANG = 1.0e5
SILICON = CATALOG.crystal("silicon").composition


@pytest.fixture(autouse=True)
def _boost_pairs(monkeypatch):
    """Scale EPDL pair cross sections seen by the photon step so a 10 um slab converts."""
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")

    def scaled(element, energies):
        sigma = dict(photon_cross_sections_ang2(element, energies))
        sigma["pair_nuclear"] = sigma["pair_nuclear"] * 1e9
        sigma["pair_electron"] = sigma["pair_electron"] * 1e9
        return sigma

    monkeypatch.setattr(pair_production, "photon_cross_sections_ang2", scaled)


def _cascade(*, Ne=8, seed=11, **core):
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([1e-2, 1e-1, 1.0, 2.0, 5.0]))
    table = prepare_bremslib_table(arrays, atomic_number=14)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 3e3,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 3e3,
    )
    result = simulate_trajectories(
        3000.0,
        Ne,
        THICKNESS_ANG,
        composition=SILICON,
        E_cut_keV=100.0,
        seed=seed,
        energy_model="midpoint",
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        secondary_threshold_eV=100_000.0,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": table},
        pair_production_model="penelope-2024",
        **core,
    )
    return {
        k: (v if isinstance(v, dict) or not hasattr(v, "get") else v.get())
        for k, v in result.items()
    }


def test_cuda_pair_cascade_is_deterministic():
    a, b = _cascade(**CUDA), _cascade(**CUDA)
    assert a["pair_production"]["events"]["row"].size > 0
    for name, value in a["pair_production"]["events"].items():
        np.testing.assert_array_equal(value, b["pair_production"]["events"][name], err_msg=name)
    np.testing.assert_array_equal(a["E_end_keV"], b["E_end_keV"])


def test_cuda_generation_zero_pairs_replay_on_the_host():
    """The recorded pair events of generation 0 equal the host step on the CUDA rows."""
    result = _cascade(**CUDA)
    events = result["pair_production"]["events"]
    n0 = int(np.count_nonzero(result["generation"] == 0))
    assert np.all(result["generation"][:n0] == 0)
    rows = {
        name: np.asarray(result[name])[:n0]
        for name in (
            "electron_id",
            "flight_id",
            "substep_id",
            "r_mid",
            "v_hat",
            "L_ang",
            "t_end_ang",
            "hard_radiative_k_eV",
            "hard_radiative_direction",
        )
    }
    replay, _ = pair_production.convert_hard_photons(
        rows, seed=11, parent_track_offset=0, layers=[(0.0, THICKNESS_ANG, SILICON)]
    )
    first = events["row"] < n0  # global rows; generation 0 comes first
    assert first.any()
    for name, value in replay.items():
        np.testing.assert_array_equal(events[name][first], value, err_msg=name)


def test_cuda_pair_cascade_closes_energy_per_history():
    result = _cascade(**CUDA)
    terms = secondary_energy_balance(result)
    assert terms["pair_rest_mass_keV"] > 0.0
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    per_history = secondary_energy_balance(result, per_history=True)
    np.testing.assert_allclose(per_history["residual_keV"], 0.0, atol=1e-9 * 3000.0)


def test_cuda_pair_counts_agree_with_the_cpu_core():
    """Converted photons and launched pair electrons agree at five Poisson sigma."""
    cpu = _cascade(Ne=48, seed=5, **HOST)["pair_production"]
    gpu = _cascade(Ne=48, seed=5, **CUDA)["pair_production"]
    for count in (
        lambda p: p["photon_counts"]["photons"],
        lambda p: p["photon_counts"]["pair"],
        lambda p: int(np.count_nonzero(p["events"]["electron_launched"])),
    ):
        a, b = count(cpu), count(gpu)
        assert a > 0 and abs(a - b) < 5.0 * np.sqrt(a + b), (a, b)
