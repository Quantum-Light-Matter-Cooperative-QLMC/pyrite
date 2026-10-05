"""Positron transport (#276) on the exact CUDA kernel.

The device-free half lives in ``test_positron_shell_transport.py`` and
``test_positron_cascade.py``. These tests need a CUDA device and run through
the lab GPU partition. The device Bhabha twins are compared with the Numba
kernels from identical inputs; the cascade, whose positrons start from CUDA
generation-0 rows, is compared in aggregate.

Validation: bhabha-close
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport.hard_inelastic import (
    POSITRON_BRANCH_OFFSET,
    _hard_primary_cosine,
    _hard_secondary_cosine,
    _sample_hard_transfer_eV,
)
from pyrite.montecarlo.transport.secondaries import LAUNCH_POSITRON, secondary_energy_balance

from .test_pair_production_cuda import CUDA, HOST, _boost_pairs  # noqa: F401
from .test_pair_production_cuda import _cascade as _pair_cascade

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
    pytest.mark.filterwarnings("ignore:.*not annihilated:UserWarning"),
]


@pytest.fixture(autouse=True)
def _require_positron_tables(_boost_pairs):  # noqa: F811 - the boosted pair fixture
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    try:
        resolve_catalog_table("silicon", projectile="positron")
    except Exception as error:
        pytest.skip(f"positron tables are not installed: {error}")


def _inputs():
    energies = np.array([6.0e3, 3.0e4, 3.0e4, 1.0e6, 1.0e6, 2.5e6])
    ionization = np.array([0.0, 99.0, 1839.0, 0.0, 149.0, 1839.0])
    resonance = np.array([16.7, 120.0, 2100.0, 16.7, 170.0, 2100.0])
    rows = [
        (e, u, w, b + POSITRON_BRANCH_OFFSET, x)
        for e, u, w in zip(energies, ionization, resonance, strict=True)
        for b in (0, 2)
        for x in (0.0, 0.31, 0.77, 0.999)
    ]
    return [np.array(col) for col in zip(*rows, strict=True)]


def test_device_bhabha_twins_match_the_numba_kernels():
    from pyrite.montecarlo._cupy_jit import jit
    from pyrite.montecarlo.transport import _jit_shell_device as dev

    @jit.rawkernel()
    def probe(E, U, W, branch, u, cutoff, transfer, primary, secondary):
        i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        if i < E.size:
            w = dev._sample_hard_transfer_eV(E[i], U[i], W[i], branch[i], cutoff, u[i])
            transfer[i] = w
            primary[i] = dev._hard_primary_cosine(E[i], U[i], W[i], branch[i], w, u[i])
            secondary[i] = dev._hard_secondary_cosine(E[i], U[i], W[i], branch[i], w, u[i])

    E, U, W, branch, u = _inputs()
    branch = branch.astype(np.int32)
    n = E.size
    out = [cupy.empty(n, dtype=cupy.float64) for _ in range(3)]
    probe(
        (1,),
        (128,),
        (
            cupy.asarray(E),
            cupy.asarray(U),
            cupy.asarray(W),
            cupy.asarray(branch),
            cupy.asarray(u),
            np.float64(50.0),
            *out,
        ),
    )
    transfer, primary, secondary = (o.get() for o in out)
    for i in range(n):
        args = (E[i], U[i], W[i], int(branch[i]))
        w = _sample_hard_transfer_eV(*args, 50.0, u[i])
        assert transfer[i] == pytest.approx(w, rel=1e-12, abs=1e-9)
        assert primary[i] == pytest.approx(_hard_primary_cosine(*args, w, u[i]), abs=1e-12)
        assert secondary[i] == pytest.approx(_hard_secondary_cosine(*args, w, u[i]), abs=1e-12)
    close = branch == 2 + POSITRON_BRANCH_OFFSET
    assert np.any(transfer[close] > 0.5 * (E[close] + U[close]))  # W_max = E is reached


def _positrons(**core):
    return _pair_cascade(positron_transport=True, **core)


def test_cuda_positron_cascade_is_deterministic_and_closes_energy():
    a, b = _positrons(**CUDA), _positrons(**CUDA)
    assert np.any(a["secondary_tracks"]["launch_kind"] == LAUNCH_POSITRON)
    for name in ("E_end_keV", "hard_W_keV", "track_id"):
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    terms = secondary_energy_balance(a)
    assert terms["positron_rest_pending_keV"] > 0.0
    assert abs(terms["residual_keV"]) <= 1e-9 * terms["incident_keV"]
    per_history = secondary_energy_balance(a, per_history=True)
    np.testing.assert_allclose(per_history["residual_keV"], 0.0, atol=1e-9 * 3000.0)


def test_cuda_positron_launches_agree_with_the_cpu_core():
    """Launched positron tracks agree at five Poisson sigma."""
    counts = []
    for core in (HOST, CUDA):
        tracks = _positrons(Ne=48, seed=5, **core)["secondary_tracks"]
        counts.append(int(np.count_nonzero(tracks["launch_kind"] == LAUNCH_POSITRON)))
    a, b = counts
    assert a > 0 and abs(a - b) < 5.0 * np.sqrt(a + b), (a, b)
