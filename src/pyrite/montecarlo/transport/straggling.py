"""Urban energy-loss fluctuation sampler (per-flight straggling)."""

import numpy as np
from numba import int64, njit, uint64

from ...materials._transport_data import TRANSPORT_ELEMENTS
from .kinematics import _SM64_GOLDEN, _SM64_ONE, _splitmix64, _stream_uniform_scalar
from .stopping import _BS_PREFACTOR, _LN2, _MC2_KEV, _element_crossover_keV

# ---- counter-based straggling RNG stream (slice D) ----------------------------
# The Urban sampler (slice C) draws a variable number of uniforms per flight --
# a Poisson channel with a nonzero count consumes exactly one, an empty one
# consumes zero, and each continuum quantum consumes one more (see
# `_urban_poisson_scalar`, `_urban_sample_element_keV`). Sharing the electron's
# own counter (the free-path / scattering-angle stream above) would therefore
# make those draws' addressing depend on whether straggling is on, which is
# exactly the coupling the "off path is bit-for-bit" and "on path never
# perturbs the existing draws" requirements forbid. So straggling gets a
# disjoint key domain instead of a shared counter:
#
#   urban_key  = _urban_stream_key_scalar(stream_key)                  -- once
#                                                                    per electron
#   flight_key = _urban_flight_key_scalar(urban_key, flight, substep) -- once
#                                                                    per flight
#
# and the sampler draws from `(flight_key, counter=0)`. Every `(seed, electron,
# flight, substep)` tuple addresses its own SplitMix64 stream with no shared
# state and no stride to overflow, so the sampler's own draw count can vary
# freely per flight without perturbing anything else -- including itself on a
# later flight. `_URBAN_STREAM_SALT` only has to differ from the other
# constants in this module; XORing it into the electron's stream key before
# re-hashing is what keeps this domain disjoint from `_stream_uniform_scalar`'s.
_URBAN_STREAM_SALT = np.uint64(0xD6E8FEB86659FD93)


@njit(uint64(uint64), cache=True)
def _urban_stream_key_scalar(stream_key):
    """Per-electron straggling key, disjoint from the electron's own stream."""
    return _splitmix64(stream_key ^ _URBAN_STREAM_SALT)


@njit(uint64(uint64, int64, int64), cache=True)
def _urban_flight_key_scalar(urban_key, flight, substep):
    """Per-``(flight, substep)`` straggling key.

    ``(flight, substep)`` pack into one 64-bit index by a 32/32 bit split --
    exact and collision-free for any flight count or substep count below
    2**32, far beyond any reachable ``max_steps`` or ``max_dE_frac`` substep
    count, and simpler than bounding a stride the way a shared counter would
    need. The result is re-hashed through the same SplitMix64 finalizer as
    every other stream in this module.
    """
    combined = (np.uint64(flight) << np.uint64(32)) + np.uint64(substep)
    return _splitmix64(urban_key + _SM64_GOLDEN * combined)


# ---- Urban energy-loss fluctuation model --------------------------------------
# Source: Geant4 Physics Reference Manual, "Energy loss fluctuations", Urban
# model (``G4UniversalFluctuation``), after Bichsel, Rev. Mod. Phys. 60, 663
# (1988). Selected in preference to Landau, Vavilov and Bohr because those three
# each *derive* the mean from their own xi and would therefore contradict the
# Joy--Luo/Berger--Seltzer splice above -- Urban instead takes C = |dE/dx| as its
# normalisation, so whatever mean the transport already computes is reproduced
# exactly. The measured regime forced that choice: per flight the process runs
# 0.003--0.16 close collisions and kappa = 2e-5--2.9e-2, and even integrated over
# the whole CSDA range kappa stays at 0.08--0.16, so neither the Landau nor the
# Gaussian limit is admissible anywhere in the catalog.
#
# The atom is modelled as two excitation levels plus an ionisation continuum:
#
#   E_0 = 10 eV                         ionisation threshold of the continuum
#   E_2 = 10 Z^2 eV,  f_2 = 2/Z         K shell (level ~ binding energy, Z f_2 = 2
#                                       recovers its occupancy)
#   f_1 = 1 - f_2,    f_1 ln E_1 + f_2 ln E_2 = ln I      loosely bound level
#   T_up = T_max = E/2                  Moller ceiling; unrestricted, no delta cut
#   r    = 0.55                         share of the mean carried by the continuum
#
#   Sigma_i = C (f_i/E_i) [ln(2 mc^2 (beta gamma)^2 / E_i) - beta^2]
#                        / [ln(2 mc^2 (beta gamma)^2 / I) - beta^2] (1 - r),  i = 1,2
#   Sigma_3 = C r (T_up - E_0) / (E_0 T_up ln(T_up/E_0))
#
# The loss over a step ``s`` is the compound Poisson sum
#
#   dE = n_1 E_1 + n_2 E_2 + sum_{k=1}^{n_3} E_k,   n_i ~ Poisson(s Sigma_i)
#
# with the continuum quanta drawn from the 1/E^2 spectrum on [E_0, T_up] by its
# exact inverse CDF, E_k = E_0 / (1 - u (T_up - E_0)/T_up), u ~ U[0,1).
#
# Units: C and Sigma_i E_i are keV/Angstrom, E_i and I are keV, s is Angstrom,
# so <n_i> = s Sigma_i is dimensionless and dE is keV. C is |dE/dx| -- the
# stopping powers in this module are negative, and the sampler returns a
# POSITIVE loss, matching the ``E -= |dEds| * s`` convention of the cores.
#
# Mean (the closure identity, and the reason this model was selected):
#   sum_{i=1,2} Sigma_i E_i = C (1-r)/L_I [f_1 L(E_1) + f_2 L(E_2)]
#                           = C (1-r)/L_I [ln(2 mc^2 (beta gamma)^2) - beta^2 - ln I]
#                           = C (1-r)
# using f_1 + f_2 = 1 and the log sum rule, and
#   Sigma_3 <E>_3 = C r,   <E>_3 = E_0 T_up ln(T_up/E_0) / (T_up - E_0),
# so <dE> = C s exactly, for any r, any Z and any I.
#
# Variance: a compound Poisson sum has no cross terms, so
#   Var(dE) = s (Sigma_1 E_1^2 + Sigma_2 E_2^2 + Sigma_3 E_0 T_up),
# using <E^2>_3 = E_0 T_up exactly for the 1/E^2 spectrum on [E_0, T_up].
#
# Limiting case: as s -> 0 every <n_i> -> 0, so P(dE = 0) -> 1 and both moments
# vanish linearly in s. The loss degenerates to the deterministic C s of the
# present cores in mean at every s, and in distribution as s -> 0.
#
# Assumptions and known limits, measured in slice B and accepted rather than
# tuned out:
#   - The PRM's own shape floor ("mean loss at least a few multiples of I") is
#     violated in nearly every catalog cell (dE/I per flight 0.017--2.58). Only
#     the first two moments are trusted here; the shape is not.
#   - Variance against the analytic Moller second moment xi T_max runs 0.73--1.42
#     over the catalog (sigma within -14%/+19%). The PRM's width correction is
#     deliberately NOT applied.
#   - A sampled loss may exceed E_i (E_1 can sit above T_up after the re-solve at
#     low E, and n_3 can exceed 1). Clamping would break the mean, so the sampler
#     does not clamp; the cutoff-crossing bookkeeping owns that.
_URBAN_E0_KEV = 1.0e-2  # ionisation level E_0 = 10 eV
_URBAN_E2_KEV_PER_Z2 = 1.0e-2  # E_2 = 10 Z^2 eV
_URBAN_RATE = 0.55  # r, the model's single tuned parameter
# Above this Poisson mean the inverse-CDF search is replaced by its Gaussian
# limit. Both branches consume a fixed number of uniforms so the stream advances
# deterministically; the regime this model was selected for keeps <n_i> under
# ~2.5 per flight, so the threshold is a guard for oversized steps, not a path
# the transport takes.
_URBAN_POISSON_GAUSS_MIN = 100.0


@njit(cache=True)
def _urban_levels_scalar(Z, J_keV, T_up, two_mc2_bg2, beta_sq):
    """Urban's ``(f_1, E_1, f_2, E_2)`` for one element [keV], with the re-solve.

    The K-shell level ``E_2 = 10 Z^2`` eV is a *parameterisation*, not a measured
    binding energy, and PyRITE runs it far below the energies Geant4 does. It is
    inadmissible when it sits above the Moller ceiling ``T_up = E/2`` (i.e. below
    ``E = 20 Z^2`` eV: C below 0.72 keV, Si below 3.92 keV, S below 5.12 keV, W
    below 109.5 keV) or when its logarithmic factor has gone non-positive. Geant4
    never reaches either boundary and simply clamps a negative count to zero;
    clamping here would DELETE a negative contribution and make the mean
    overshoot -- measured closure 1.0187/1.0098/1.0038/1.0010 for W at 1/2/5/10
    keV. Instead the sum rules are re-solved on the single remaining level,
    ``f_1 = 1``, ``E_1 = I``, which satisfies ``f_1 ln E_1 = ln I`` identically
    and so restores <dE> = C s exactly.

    Validation: energy-loss-straggling
    """
    E_2 = _URBAN_E2_KEV_PER_Z2 * Z * Z
    f_2 = 2.0 / Z if Z > 2.0 else 1.0
    if Z > 2.0 and E_2 < T_up and np.log(two_mc2_bg2 / E_2) - beta_sq > 0.0:
        f_1 = 1.0 - f_2
        E_1 = np.exp((np.log(J_keV) - f_2 * np.log(E_2)) / f_1)
        if np.log(two_mc2_bg2 / E_1) - beta_sq > 0.0:
            return f_1, E_1, f_2, E_2
    return 1.0, J_keV, 0.0, E_2


@njit(cache=True)
def _urban_channels_scalar(Z, J_keV, C_keV_per_ang, E_i):
    """Urban's per-unit-length channel rates for one element.

    Returns ``(valid, Sigma_1, E_1, Sigma_2, E_2, Sigma_3)`` with ``Sigma_i`` in
    Angstrom^-1 and the levels in keV. ``C_keV_per_ang`` is ``|dE/dx|`` for this
    element alone -- the model is applied per element, matching the Bragg-additive
    form of the mean stopping power, which is what keeps the total mean equal to
    the transport's own ``dE/dx``.

    ``valid`` is 0.0 where the parameterisation has no admissible form at all:
    the continuum needs ``T_up > E_0`` (E above 20 eV) and every level needs a
    positive logarithmic factor, which fails once ``2 mc^2 (beta gamma)^2 <= I``
    (below ~0.18 keV for tungsten). Callers fall back to the deterministic loss
    there rather than sample a degenerate distribution.

    Validation: energy-loss-straggling
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    two_mc2_bg2 = 2.0 * _MC2_KEV * tau * (tau + 2.0)
    T_up = 0.5 * E_i

    if T_up <= _URBAN_E0_KEV or C_keV_per_ang <= 0.0:
        return 0.0, 0.0, J_keV, 0.0, 0.0, 0.0

    L_I = np.log(two_mc2_bg2 / J_keV) - beta_sq
    if L_I <= 0.0:
        return 0.0, 0.0, J_keV, 0.0, 0.0, 0.0

    f_1, E_1, f_2, E_2 = _urban_levels_scalar(Z, J_keV, T_up, two_mc2_bg2, beta_sq)
    soft = C_keV_per_ang * (1.0 - _URBAN_RATE) / L_I
    sigma_1 = soft * (f_1 / E_1) * (np.log(two_mc2_bg2 / E_1) - beta_sq)
    sigma_2 = 0.0
    if f_2 > 0.0:
        sigma_2 = soft * (f_2 / E_2) * (np.log(two_mc2_bg2 / E_2) - beta_sq)
    sigma_3 = (
        C_keV_per_ang
        * _URBAN_RATE
        * (T_up - _URBAN_E0_KEV)
        / (_URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV))
    )
    return 1.0, sigma_1, E_1, sigma_2, E_2, sigma_3


@njit(cache=True)
def _urban_moments_element_scalar(Z, J_keV, C_keV_per_ang, E_i, s_ang):
    """Analytic ``(mean, variance)`` of the Urban loss for one element [keV, keV^2].

    The mean is ``C s`` by the closure identity above; it is evaluated from the
    sampled channels rather than shortcut to ``C s`` so that the identity is what
    the tests measure.

    Validation: energy-loss-straggling
    """
    valid, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(
        Z, J_keV, C_keV_per_ang, E_i
    )
    if valid == 0.0:
        return C_keV_per_ang * s_ang, 0.0
    T_up = 0.5 * E_i
    mean_3 = _URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV) / (T_up - _URBAN_E0_KEV)
    mean = s_ang * (sigma_1 * E_1 + sigma_2 * E_2 + sigma_3 * mean_3)
    var = s_ang * (sigma_1 * E_1 * E_1 + sigma_2 * E_2 * E_2 + sigma_3 * _URBAN_E0_KEV * T_up)
    return mean, var


@njit(cache=True)
def _urban_poisson_scalar(lam, key, counter):
    """Poisson variate with mean ``lam``, returning ``(n, counter)``.

    Inverse CDF by the recurrence ``p_{k+1} = p_k lam/(k+1)``, which consumes
    exactly one uniform however large ``n`` comes out -- a counter-addressed
    stream must advance by an amount known to the caller, and Knuth's product
    method would advance it by ``n + 1``. Above ``lam = 100`` the recurrence
    starts from ``exp(-lam) ~ 1e-44`` and costs O(lam) iterations, so it hands
    over to the Gaussian limit (skewness ``lam^-1/2`` <= 0.1 there), which
    consumes two uniforms through Box-Muller.

    Validation: energy-loss-straggling
    """
    if lam <= 0.0:
        return 0, counter
    if lam < _URBAN_POISSON_GAUSS_MIN:
        u = _stream_uniform_scalar(key, counter)
        counter = counter + _SM64_ONE
        p = np.exp(-lam)
        cdf = p
        k = 0
        while u >= cdf and k < 10000:
            k += 1
            p = p * lam / k
            cdf += p
        return k, counter
    u1 = _stream_uniform_scalar(key, counter)
    u2 = _stream_uniform_scalar(key, counter + _SM64_ONE)
    counter = counter + _SM64_ONE + _SM64_ONE
    # u1 is in [0, 1); shift it off the log's singularity.
    z = np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
    n = int(np.floor(lam + np.sqrt(lam) * z + 0.5))
    if n < 0:
        n = 0
    return n, counter


@njit(cache=True)
def _urban_ionisation_keV(u, T_up):
    """Inverse CDF of the ``1/E^2`` continuum on ``[E_0, T_up]`` [keV].

    ``F(E) = (E_0 T_up/(T_up - E_0)) (1/E_0 - 1/E)``, so ``u = 0`` returns ``E_0``
    and ``u -> 1`` returns ``T_up``. The denominator stays in ``(E_0/T_up, 1]``.

    Validation: energy-loss-straggling
    """
    return _URBAN_E0_KEV / (1.0 - u * (T_up - _URBAN_E0_KEV) / T_up)


@njit(cache=True)
def _urban_sample_element_keV(Z, J_keV, C_keV_per_ang, E_i, s_ang, key, counter):
    """Sample the Urban energy loss of one element over ``s_ang``, ``(dE, counter)``.

    ``dE`` is a positive loss in keV. Where the parameterisation has no admissible
    form the deterministic ``C s`` is returned instead, which keeps the mean exact
    and the fallback silent in every moment test.

    Validation: energy-loss-straggling
    """
    valid, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(
        Z, J_keV, C_keV_per_ang, E_i
    )
    if valid == 0.0 or s_ang <= 0.0:
        return C_keV_per_ang * s_ang, counter

    n_1, counter = _urban_poisson_scalar(sigma_1 * s_ang, key, counter)
    n_2, counter = _urban_poisson_scalar(sigma_2 * s_ang, key, counter)
    n_3, counter = _urban_poisson_scalar(sigma_3 * s_ang, key, counter)

    dE = n_1 * E_1 + n_2 * E_2
    T_up = 0.5 * E_i
    for _ in range(n_3):
        u = _stream_uniform_scalar(key, counter)
        counter = counter + _SM64_ONE
        dE += _urban_ionisation_keV(u, T_up)
    return dE, counter


@njit(cache=True)
def _dEds_spliced_element_scalar(J, k, coeff, E_cross, delta, E_i):
    """One element's contribution to the spliced stopping power [keV/Angstrom].

    A deliberate duplicate of the body of :func:`_dEds_spliced_compound_scalar`
    rather than a factoring of it: that function accumulates the Joy--Luo and
    Berger--Seltzer sums separately and combines them once, and re-associating
    the sum would move the last bits of every existing transport result. The two
    agree to float64 rounding, which is what the closure test asserts.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    if E_i < E_cross:
        return -7.85e-4 / E_i * coeff * np.log(1.166 * (E_i + k * J) / J)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    I_rel = J / _MC2_KEV
    return (
        -_BS_PREFACTOR
        / beta_sq
        * coeff
        * (np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta)
    )


@njit(cache=True)
def _urban_sample_compound_keV(
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_i, s_ang, key, counter
):
    """Sample the Urban loss of a compound over ``s_ang`` [keV], ``(dE, counter)``.

    Bragg-additive, one independent Urban draw per element with that element's own
    ``C_i = |dE/dx|_i``, ``Z_i`` and ``I_i``. Summing the per-element means
    reproduces the compound stopping power, so the compound closure is inherited
    from the elemental one rather than asserted separately.

    Validation: energy-loss-straggling
    """
    dE = 0.0
    for i in range(Z_arr.size):
        C_i = -_dEds_spliced_element_scalar(
            J_arr[i], k_arr[i], coeff_arr[i], E_cross_arr[i], delta, E_i
        )
        loss, counter = _urban_sample_element_keV(Z_arr[i], J_arr[i], C_i, E_i, s_ang, key, counter)
        dE += loss
    return dE, counter


def urban_element_table(composition):
    """``(Z, J_keV, k, coeff, E_cross)`` arrays for a normalized composition.

    The host-side table the Urban samplers index, built from the same
    :data:`TRANSPORT_ELEMENTS` entries and the same memoized crossover as
    :func:`spliced_stopping_keV_per_ang`, so the two cannot drift.

    Validation: energy-loss-straggling
    """
    Z_arr = np.empty(len(composition))
    J_arr = np.empty(len(composition))
    k_arr = np.empty(len(composition))
    coeff_arr = np.empty(len(composition))
    E_cross_arr = np.empty(len(composition))
    for i, (element, n_i) in enumerate(composition):
        params = TRANSPORT_ELEMENTS[element]
        Z = float(params["Z"])
        A = float(params["A"])
        J = float(params["J_keV"])
        Z_arr[i] = Z
        J_arr[i] = J
        k_arr[i] = 0.731 + 0.0688 * np.log10(Z)
        coeff_arr[i] = (n_i / 0.602214076) * Z
        E_cross_arr[i] = _element_crossover_keV(element, Z, A, J)
    return Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr


def urban_loss_moments_keV(composition, E_keV, s_ang, delta=0.0):
    """Analytic ``(mean, variance)`` of the Urban loss over ``s_ang`` [keV, keV^2].

    Cold path, for tests, validation instruments and the observable measurement
    that follows; the cores use the scalar helpers. The mean is the closure
    identity ``|dE/dx| s`` evaluated through the model's own channels, not
    shortcut to it.

    Validation: energy-loss-straggling
    """
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
    mean = 0.0
    var = 0.0
    for i in range(Z_arr.size):
        C_i = -_dEds_spliced_element_scalar(
            J_arr[i], k_arr[i], coeff_arr[i], E_cross_arr[i], delta, float(E_keV)
        )
        m_i, v_i = _urban_moments_element_scalar(
            Z_arr[i], J_arr[i], C_i, float(E_keV), float(s_ang)
        )
        mean += m_i
        var += v_i
    return mean, var
