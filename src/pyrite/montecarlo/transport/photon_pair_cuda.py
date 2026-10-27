"""Explicit-draw CUDA pair final states for the optional photon queue."""

import numpy as np

from ._photon_cuda_sampling import validate_event_inputs
from ._photon_pair_cuda_kernel import pair_kernel
from .pair_production import (
    _ALPHA,
    _ELECTRON_REST_EV,
    _MAX_REJECTIONS,
    PAIR_THRESHOLD_EV,
    _coulomb_correction,
    _screening_radius,
)


def sample_pairs_cuda(energy_eV, direction, uniforms, *, atomic_number, require_complete=True):
    """PENELOPE-2024 pair products with the CPU reference's exact draw ordering.

    Inputs: host energies (N,) [eV], directions (N,3), uniform prefixes (N,L).
    Both nuclear/electron-field channels use the existing pair approximation;
    the triplet recoil electron is not separately modeled. The established
    high-Z near-threshold flat-density convention is retained. Screening radii
    restrict Z to 1..99, as in the CPU owner. No RNG, rest-energy booking or
    charged propagation occurs here. Queue bookkeeping reserves 2mc² per pair.

    Returns device kinetic energies [eV], unit directions, draw counts and
    statuses: 0 complete, 1 exhausted prefix, 2 reference rejection exhaustion.
    Nonzero status raises unless ``require_complete=False``; a queue can then
    extend the same prefix and retry. This is a numerical backend port, not a
    new pair model or independent full-track validation.
    Source/limits: ``pair_production.sample_pair`` and its cited PENELOPE
    equations, including independent daughter directions and threshold limit.
    Validation: pair-production-sampling
    """
    energy, rays, draws = validate_event_inputs(energy_eV, direction, uniforms)
    if np.any(energy <= PAIR_THRESHOLD_EV):
        raise ValueError("pair production requires a photon energy above 2 m_e c^2")
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a bool")
    if isinstance(atomic_number, bool) or not isinstance(atomic_number, (int, np.integer)):
        raise ValueError("atomic_number must be an integer in 1..99")
    z = int(atomic_number)
    radius, coulomb = _screening_radius(z), _coulomb_correction(z)
    from ..._backend import select_backend

    cp = select_backend("cuda").xp
    n = len(energy)
    packed, used, status = (
        cp.empty((n, 8)),
        cp.empty(n, dtype=cp.int64),
        cp.empty(n, dtype=cp.int32),
    )
    if n:
        pair_kernel(cp)(
            ((n + 127) // 128,),
            (128,),
            (
                cp.asarray(energy),
                cp.asarray(rays),
                cp.asarray(draws),
                np.int64(n),
                np.int32(draws.shape[1]),
                np.int32(_MAX_REJECTIONS),
                np.float64(_ELECTRON_REST_EV),
                np.float64(radius),
                np.float64(coulomb),
                np.float64(_ALPHA * z),
                np.float64(np.pi),
                packed,
                used,
                status,
            ),
        )
        if require_complete:
            codes = cp.asnumpy(status)
            if np.any(codes == 1):
                raise ValueError("pair uniform sequence exhausted before the event ended")
            if np.any(codes == 2):
                raise RuntimeError("pair reduced-energy rejection did not converge")
    return {
        "electron_energy_eV": packed[:, 0],
        "positron_energy_eV": packed[:, 1],
        "electron_direction": packed[:, 2:5],
        "positron_direction": packed[:, 5:8],
        "draws_used": used,
        "status": status,
    }
