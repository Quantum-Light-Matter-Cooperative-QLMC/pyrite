"""Batched CUDA conditional Compton/Rayleigh events with explicit draw pools.

No RNG or queue policy. The production queue can refill a row's draw prefix
when status=1, without truncating the reference rejection budget or borrowing
another photon's stream. All outputs reside on the current CUDA device.
"""

from functools import cache

import numpy as np

from ...materials.crystal import HC_EV_ANG, M_E_EV
from ...materials.photon_differential import (
    coherent_form_factor_table,
    incoherent_scattering_function_table,
)
from ._photon_cuda_sampling import validate_event_inputs
from ._photon_scatter_cuda_kernel import scatter_kernel
from .photon_sampling import MAX_REJECTIONS


@cache
def _device_table(cp, element, channel, device):
    del device  # cache key: each CUDA device owns its immutable table upload
    getter = (
        coherent_form_factor_table
        if channel == "coherent"
        else incoherent_scattering_function_table
    )
    x, factor = getter(element)
    if not np.isfinite(factor).all() or np.any(factor < 0):
        raise ValueError("scattering factors must be finite and non-negative")
    return cp.asarray(x), cp.asarray(factor)


def sample_photon_scattering_cuda(
    element, energy_eV, direction, uniforms, *, channel, require_complete=True
):
    """Conditional events for one element/channel and N host photon rows.

    ``channel`` is ``coherent`` (Rayleigh) or ``incoherent`` (Compton).
    Inputs are energies [eV], unit directions and independent explicit uniform
    sequences, one matrix row per photon. Proposal/rejection laws, clipped
    lin-lin factors, draw ordering and recoil match ``photon_scattering``.
    Rayleigh keeps energy and has no electron (electron directions are NaN).

    Returns device arrays for photon/electron energies [eV], directions,
    ``one_minus_cosine``, ``draws_used`` and integer ``status``. Status 0 is a
    complete event; 1 needs a longer prefix of the same draw stream; 2 exhausts
    the reference MAX_REJECTIONS; 3 has no positive conditional support.
    By default nonzero statuses raise. ``require_complete=False`` permits a
    queue to refill only status-1 rows and retry with their original prefixes.
    Such retries must not change the photon key, discard prefixes or replace
    exhaustion with forced acceptance. No fallback to host sampling occurs.

    Sources, assumptions and limiting cases are those of the unchanged CPU
    conditional samplers: isolated unpolarized atoms, free Compton recoil
    without binding/Doppler corrections, infinite-mass Rayleigh recoil.
    Constant factors recover Thomson/free Klein–Nishina. Binary64 math can
    round differently across backends at acceptance boundaries.
    Validation: photon-compton-scattering, photon-rayleigh-scattering
    """
    if channel not in ("coherent", "incoherent"):
        raise ValueError("channel must be 'coherent' or 'incoherent'")
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a bool")
    energy, ray, draws = validate_event_inputs(energy_eV, direction, uniforms)
    from ..._backend import select_cupy_backend

    cp = select_cupy_backend().xp
    n = energy.size
    packed, used, status = (
        cp.empty((n, 9)),
        cp.empty(n, dtype=cp.int64),
        cp.empty(n, dtype=cp.int32),
    )
    if n:
        x, factor = _device_table(cp, element, channel, cp.cuda.Device().id)
        scatter_kernel(cp)(
            ((n + 127) // 128,),
            (128,),
            (
                cp.asarray(energy),
                cp.asarray(ray),
                cp.asarray(draws),
                x,
                factor,
                np.int64(n),
                np.int32(draws.shape[1]),
                np.int32(x.size),
                np.int32(channel == "coherent"),
                np.int32(MAX_REJECTIONS),
                np.float64(M_E_EV),
                np.float64(HC_EV_ANG),
                np.float64(np.pi),
                packed,
                used,
                status,
            ),
        )
        if require_complete:
            codes = cp.asnumpy(status)
            if np.any(codes == 1):
                raise ValueError("photon uniform sequence exhausted before scattering ended")
            if np.any(codes == 2):
                raise RuntimeError("photon scattering rejection budget exhausted")
            if np.any(codes == 3):
                raise ValueError("scattering factor has no positive conditional support")
    return {
        "photon_energy_eV": packed[:, 0],
        "photon_direction": packed[:, 1:4],
        "electron_energy_eV": packed[:, 4],
        "electron_direction": packed[:, 5:8],
        "one_minus_cosine": packed[:, 8],
        "draws_used": used,
        "status": status,
    }
