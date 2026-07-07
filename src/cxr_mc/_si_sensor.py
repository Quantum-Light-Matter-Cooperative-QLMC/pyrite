"""
_si_sensor.py
=============

Shared silicon-sensor plumbing for the two detector forward models
(`timepix_response.py`, `eaglexo_response.py`): the fixed Si material
constants both derive from, the (grid -> cached-response) keying pattern,
the Poisson acquisition core, and the `.apply()` input-shape guard. Each
response model keeps its own detector physics (charge-sharing MC, QE table,
...) on top of this -- nothing here is model-specific.
"""

import numpy as np

# ---- silicon sensor physics (fixed material constants) -----------------------
W_EHP_EV = 3.65  # mean energy to make one electron-hole pair [eV]
FANO_SI = 0.115  # Fano factor: Var(N) = F * N (sub-Poisson) for Si
SI_DENSITY_G_CM3 = 2.329
SI_A = 28.085  # Si molar mass [g/mol]
# number density n = rho/A * N_A, converted cm^-3 -> Ang^-3 (the unit
# absorption_length_ang wants). 0.602214076 = N_A * 1e-24.
SI_N_PER_ANG3 = SI_DENSITY_G_CM3 / SI_A * 0.602214076


def grid_key(E_grid_eV):
    """Cache-key prefix identifying a uniform energy grid by (size, endpoints).

    Both detectors' `get_response` append their own hardware/MC settings
    after this prefix to build the full cache key."""
    E = np.asarray(E_grid_eV, dtype=float)
    return (E.size, round(float(E[0]), 6), round(float(E[-1]), 6))


def prep_spectrum(spec, E_grid_eV, module_name):
    """Validate + sanitize an input spectrum against a response's energy grid.

    NaN/inf samples (e.g. a bad-geometry case) become zero flux rather than
    poisoning the result; a grid-size mismatch raises with a pointer to that
    module's `get_response`."""
    E = np.asarray(E_grid_eV, dtype=float)
    spec = np.nan_to_num(np.asarray(spec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if spec.shape != E.shape:
        raise ValueError(
            f"spec length {spec.shape} != response grid {E.shape}. "
            f"Build a matching response with {module_name}.get_response(E_grid, ...)."
        )
    return spec


def poisson_core(E_grid_eV, detected_per_s, time_s, rng):
    """Poisson-draw one acquisition from a detected spectral density.

    Expected counts in a bin are rate * bin_width * live_time. Shared core of
    both detectors' `poisson_counts`; `rng` must already be resolved (callers
    own the `None` -> `np.random.default_rng()` default so they can reuse the
    same stream for any follow-on noise they add). Returns (counts, expected)."""
    E = np.asarray(E_grid_eV, dtype=float)
    dE = E[1] - E[0]
    expected = np.clip(np.asarray(detected_per_s, dtype=float) * dE * time_s, 0.0, None)
    return rng.poisson(expected), expected
