"""CUDA-gated parity: the float32 device flight-grouped path vs the host kernel.

``_grouped._device_grouped`` maps referenced rows onto the coherent route's
segmented intensity kernel (phase slope ``d / hbar c``, formation constants,
flight bounds). It must reproduce the float64 host kernel
``accumulate_grouped_formation`` to float32 accuracy on phases of the size the
per-flight referencing leaves (≲1e3 rad).

Validation: coherent-formation-absorption
"""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.cxr._cpu_reduce import accumulate_grouped_formation
from tests.montecarlo.spectrum.test_grouped_formation_reduce import _case

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")


@pytest.mark.parametrize("n_pol", [1, 2])
def test_device_grouped_matches_host_kernel(n_pol):
    from pyrite.montecarlo.spectrum.cxr import _grouped

    rng = np.random.default_rng(1152)
    lines, d, g_phase, L_esc, coefs, bounds, grids = _case(rng, n_rows=600, n_flights=41)
    # Referenced scale: within-flight extents, not absolute trajectory ages.
    d = d * 2e-2
    g_phase = g_phase * 0.5
    coefs = coefs[:n_pol]
    host = np.zeros(grids[0].size)
    accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, bounds, grids, host, 0.5)
    assert np.max(np.abs(d / HBARC_EV_ANG * grids[0].max())) > 10.0

    dev = cp.zeros(grids[0].size, dtype=cp.float32)
    _grouped._device_grouped(
        tuple(cp.asarray(a) for a in lines),
        cp.asarray(d),
        cp.asarray(g_phase),
        cp.asarray(L_esc),
        [cp.asarray(c) for c in coefs],
        cp.asarray(bounds),
        tuple(cp.asarray(g) for g in grids),
        dev,
        0.5,
    )
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(cp.asnumpy(dev), host, rtol=2e-4, atol=2e-4 * host.max())
