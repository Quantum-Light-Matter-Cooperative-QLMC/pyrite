"""Hardware anchors for SBETHE stopping on both CUDA transport paths."""

import numpy as np
import pytest

from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories


@pytest.mark.hardware
@pytest.mark.parametrize("use_lut", [False, True], ids=["exact", "lut"])
def test_cuda_sbethe_constant_stopping_reaches_the_cutoff_at_50_angstrom(use_lut):
    cupy = pytest.importorskip("cupy")
    try:
        has_cuda = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        has_cuda = False
    if not has_cuda:
        pytest.skip("no CUDA device")

    # 100 eV/angstrom = 0.1 keV/angstrom; losing 5 keV takes 50 angstrom.
    table = {
        "stopping_energy_eV": np.array([5.0e3, 10.0e3]),
        "stopping_eV_per_angstrom": np.array([100.0, 100.0]),
    }
    result = simulate_trajectories(
        10.0,
        1,
        1.0e8,
        composition=[("C", 1.0e-30)],
        E_cut_keV=5.0,
        elastic_model="sr",
        seed=4,
        max_steps=20,
        transport_core="cuda",
        transport_lut_config=TransportLUTConfig(enabled=use_lut),
        stopping_tables=[table],
    )

    assert result["L_ang"][-1] == pytest.approx(50.0, rel=1e-10)
    assert result["n_cutoff_stopped"] == 1
