"""Same prepared float64 table on CPU/CUDA; no transport qualification implied."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport.radiative_moments import prepare_radiative_moments
from tests.helpers.bremslib import synthetic_bremslib_arrays

try:
    import cupy as cp

    HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUDA = False

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")]


@pytest.mark.parametrize("cutoff", [1000.0, 10000.0])
def test_same_prepared_layers_and_explicit_coverage(cutoff):
    from pyrite.montecarlo.transport._radiative_moments_device import UploadedRadiativeMoments

    tables = {
        element: prepare_bremslib_table(
            synthetic_bremslib_arrays(theta_step_deg=10), atomic_number=z
        )
        for element, z in [("Si", 14), ("W", 74)]
    }
    for table in tables.values():
        table.incident_energy_keV[[0, -1]] = [10.0, 200.0]
    p = prepare_radiative_moments(
        [[("Si", 0.05)], [("Si", 0.02), ("W", 0.03)]], tables, 10000, 200000, cutoff
    )
    uploaded = UploadedRadiativeMoments.from_host(p)
    assert uploaded.digest == p.digest
    for layer in range(2):
        queries = np.unique(np.r_[p.energy_eV[layer], np.geomspace(10000, 200000, 1001)])
        cpu = np.array([p.evaluate(layer, e) for e in queries])
        gpu = uploaded.evaluate(layer, queries).get()
        assert np.all(gpu >= 0)
        np.testing.assert_allclose(gpu, cpu, rtol=2e-12, atol=1e-24)
        np.testing.assert_array_equal(uploaded.evaluate(layer, queries).get(), gpu)
        bad = [np.nan, np.inf, -1, np.nextafter(10000.0, 0), np.nextafter(200000.0, np.inf)]
        assert np.all(np.isnan(uploaded.evaluate(layer, bad).get()))
