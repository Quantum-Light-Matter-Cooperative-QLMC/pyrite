"""Host validation shared by explicit-draw CUDA conditional samplers."""

import numpy as np

from ...materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV


def validate_event_inputs(energy_eV, direction, uniforms):
    energy = np.asarray(energy_eV, dtype=float)
    ray, draws = np.asarray(direction, dtype=float), np.asarray(uniforms, dtype=float)
    if energy.ndim != 1 or ray.shape != (energy.size, 3):
        raise ValueError("energy must have shape (N,) and directions (N, 3)")
    if draws.ndim != 2 or draws.shape[0] != energy.size:
        raise ValueError("uniforms must have shape (N, draws_per_row)")
    if not np.isfinite(energy).all() or np.any((energy < EPDL_E_MIN_EV) | (energy > EPDL_E_MAX_EV)):
        raise ValueError("photon energy is outside the EPDL domain")
    if not np.isfinite(ray).all() or not np.allclose(
        np.linalg.norm(ray, axis=1), 1, rtol=0, atol=1e-12
    ):
        raise ValueError("directions must be finite unit vectors")
    if not np.isfinite(draws).all() or np.any((draws < 0) | (draws >= 1)):
        raise ValueError("photon uniforms must be finite and in [0, 1)")
    return tuple(np.ascontiguousarray(value) for value in (energy, ray, draws))
