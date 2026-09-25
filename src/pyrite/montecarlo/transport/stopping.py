"""SBETHE table stopping and Joy-Luo / Berger-Seltzer reference models."""

from collections.abc import Mapping

import numpy as np
from numba import njit

from ...materials._transport_data import STERNHEIMER_DENSITY_EFFECT, TRANSPORT_ELEMENTS


def prepare_sbethe_stopping_table(arrays):
    """Validate and log-transform one stored SBETHE collision-stopping table.

    The returned arrays use keV and keV/angstrom and are ready for the scalar
    CPU/device interpolation kernel.  SBETHE supplies strictly positive native
    nodes from 1 keV through 1 GeV; values outside that domain are rejected by
    the host entry point rather than extrapolated.
    """
    energy_keV = np.asarray(arrays["stopping_energy_eV"], dtype=np.float64) * 1.0e-3
    stopping_keV_per_ang = np.asarray(arrays["stopping_eV_per_angstrom"], dtype=np.float64) * 1.0e-3
    if energy_keV.ndim != 1 or stopping_keV_per_ang.shape != energy_keV.shape:
        raise ValueError("SBETHE stopping energy and value arrays must be matching vectors")
    if energy_keV.size < 2 or not np.all(np.diff(energy_keV) > 0.0):
        raise ValueError("SBETHE stopping energy grid must be strictly increasing")
    if not np.all(np.isfinite(energy_keV)) or not np.all(np.isfinite(stopping_keV_per_ang)):
        raise ValueError("SBETHE stopping table contains non-finite values")
    if np.any(energy_keV <= 0.0) or np.any(stopping_keV_per_ang <= 0.0):
        raise ValueError("SBETHE stopping table values must be strictly positive")
    return np.log(energy_keV), np.log(stopping_keV_per_ang)


def prepare_sbethe_stopping_tables(stopping_tables, n_layers):
    """Prepare one SBETHE table per layer; already-prepared ``(logE, logS)`` pass through."""
    if stopping_tables is None:
        return None
    if len(stopping_tables) != n_layers:
        raise ValueError("stopping_tables must contain one SBETHE table per layer")
    prepared_tables = []
    for table in stopping_tables:
        prepared = (
            table
            if not isinstance(table, Mapping) and len(table) == 2 and np.asarray(table[0]).ndim == 1
            else None
        )
        if prepared is None:
            prepared = prepare_sbethe_stopping_table(table)
        prepared_tables.append(prepared)
    return prepared_tables


def pack_sbethe_stopping_tables(tables, n_layers):
    """Pad prepared SBETHE tables for the exact CPU/device transport cores."""
    if tables is None:
        return (
            False,
            np.zeros(n_layers, dtype=np.int32),
            np.zeros((n_layers, 1)),
            np.zeros((n_layers, 1)),
        )
    if len(tables) != n_layers:
        raise ValueError("one SBETHE stopping table is required per material layer")
    counts = np.asarray([len(table[0]) for table in tables], dtype=np.int32)
    width = int(np.max(counts))
    log_energy = np.zeros((n_layers, width), dtype=np.float64)
    log_stopping = np.zeros((n_layers, width), dtype=np.float64)
    for layer, ((energy_row, stopping_row), count) in enumerate(zip(tables, counts, strict=True)):
        log_energy[layer, :count] = energy_row
        log_stopping[layer, :count] = stopping_row
    return True, counts, log_energy, log_stopping


@njit(cache=True)
def _dEds_sbethe_scalar(log_energy_keV, log_stopping_keV_per_ang, E_keV):
    """Log-log interpolate SBETHE collision stopping [keV/angstrom], negative.

    Source: SBETHE (Salvat, April 2024), vendored ``sbethe.f`` ``BETHE`` and
    ``stp.dat`` output. SBETHE evaluates the corrected Bethe expression with
    DHFS shell and Fano density-effect corrections; this function interpolates
    those material-level output nodes rather than re-evaluating the formula.

    Log-log interpolation preserves positivity and exactly reproduces every
    native SBETHE node.  The caller must enforce the table domain; endpoint
    clamps only protect roundoff at an already-validated transport cutoff.
    Assumptions: positive ordered nodes and power-law behavior between adjacent
    nodes. Limiting cases: a native node is returned exactly; a constant table
    remains constant; the signed transport rate is always non-positive.

    Validation: sbethe-corrected-stopping
    """
    log_e = np.log(E_keV)
    if log_e <= log_energy_keV[0]:
        return -np.exp(log_stopping_keV_per_ang[0])
    last = log_energy_keV.size - 1
    if log_e >= log_energy_keV[last]:
        return -np.exp(log_stopping_keV_per_ang[last])
    hi = np.searchsorted(log_energy_keV, log_e)
    lo = hi - 1
    fraction = (log_e - log_energy_keV[lo]) / (log_energy_keV[hi] - log_energy_keV[lo])
    log_stopping = log_stopping_keV_per_ang[lo] + fraction * (
        log_stopping_keV_per_ang[hi] - log_stopping_keV_per_ang[lo]
    )
    return -np.exp(log_stopping)


@njit(cache=True)
def _dEds_sbethe_packed_scalar(log_energy_keV, log_stopping_keV_per_ang, layer, count, E_keV):
    """Evaluate one row of padded SBETHE tables inside an exact transport core."""
    return _dEds_sbethe_scalar(
        log_energy_keV[layer, :count], log_stopping_keV_per_ang[layer, :count], E_keV
    )


def sbethe_stopping_keV_per_ang(log_energy_keV, log_stopping_keV_per_ang, E_keV):
    """Evaluate a prepared SBETHE table without extrapolation."""
    energy = np.asarray(E_keV, dtype=float)
    lower = float(np.nextafter(np.exp(log_energy_keV[0]), -np.inf))
    upper = float(np.nextafter(np.exp(log_energy_keV[-1]), np.inf))
    if np.any(~np.isfinite(energy)) or np.any(energy < lower) or np.any(energy > upper):
        raise ValueError(f"SBETHE stopping energy must be within [{lower:g}, {upper:g}] keV")
    values = np.interp(np.log(energy), log_energy_keV, log_stopping_keV_per_ang)
    return -np.exp(values)


def _dEds_keV_per_ang(Z, A, J_keV, rho_g_cm3, E_keV):
    """Joy-Luo modified Bethe stopping power [keV/Angstrom] (negative)."""
    k = 0.731 + 0.0688 * np.log10(Z)
    # 78500 keV/cm -> 7.85e-4 keV/Angstrom prefactor
    return -7.85e-4 * rho_g_cm3 * Z / (A * E_keV) * np.log(1.166 * (E_keV + k * J_keV) / J_keV)


@njit(cache=True)
def _dEds_compound_scalar(J_arr, k_arr, coeff_arr, E_i):

    total = 0.0
    for i in range(J_arr.size):
        J = J_arr[i]
        k = k_arr[i]
        coeff = coeff_arr[i]

        total += coeff * np.log(1.166 * (E_i + k * J) / J)

    return -7.85e-4 / E_i * total


@njit(cache=True)
def _dEds_compound(J_arr, k_arr, coeff_arr, E_keV):
    out = np.empty_like(E_keV)

    for j in range(E_keV.size):
        E = E_keV[j]
        total = 0.0
        for i in range(J_arr.size):
            J = J_arr[i]
            k = k_arr[i]
            coeff = coeff_arr[i]

            total += coeff * np.log(1.166 * (E + k * J) / J)

        out[j] = -7.85e-4 / E * total
    return out


# ---- Berger-Seltzer / ICRU-37 relativistic collision stopping -----------------
# The relativistic replacement for the Joy-Luo law above:
#
#   -(1/rho) dE/dx = (2 pi r_e^2 m c^2 N_A / beta^2) (Z/A)
#                    [ ln(tau^2 (tau + 2) / (2 (I/mc^2)^2)) + F^-(tau) - delta ]
#   F^-(tau)       = 1 - beta^2 + [tau^2/8 - (2 tau + 1) ln 2] / (tau + 1)^2
#
# with tau = E/mc^2 and 2 pi r_e^2 m c^2 N_A = 0.1535 MeV cm^2/mol. The
# per-element coefficient c_i = n_i Z_i / 0.602214076 is an exact rewrite of
# rho Z/A (eq-stopping-compound-coefficient) and carries over unchanged, so
# Bragg additivity has the same form as Joy-Luo.
#
# 0.1535 MeV cm^2/mol against c_i in mol/cm^3 gives MeV/cm; 1e3 keV/MeV times
# 1e-8 cm/Angstrom is the 1e-5 that makes the prefactor 1.535e-6 keV/Angstrom.
# That pins it against the Joy-Luo path: non-relativistically beta^2 -> 2E/mc^2
# and (tau+2)/2 -> 1, so 1.535e-6/beta^2 * ln(E^2/I^2) -> (7.844e-4/E) ln(E/I)
# against the 7.85e-4 of eq-stopping-joy-luo -- 0.08%, the rounding in the
# conventional constant.
#
# delta is the density-effect correction. It is a bulk property of the medium
# rather than of an element, so it factors out of the Bragg sum and enters as
# one scalar per layer. Callers pass 0.0 until Sternheimer parameters land; over
# 1-300 keV beta*gamma stays below 1.24 and the term is bounded well under the
# Joy-Luo error it replaces.
_BS_PREFACTOR = 1.535e-6  # keV Angstrom^-1 per (mol cm^-3), from 0.1535 MeV cm^2/mol
_MC2_KEV = 510.99895
_LN2 = 0.6931471805599453


def _bs_tau_terms(E_keV):
    """Return ``(tau, beta^2, F^-(tau))`` -- the element-independent factors."""
    tau = E_keV / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    return tau, beta_sq, f_minus


def _dEds_bs_keV_per_ang(Z, A, J_keV, rho_g_cm3, E_keV, delta=0.0):
    """Berger-Seltzer collision stopping power [keV/Angstrom] (negative).

    Elemental form, mirroring :func:`_dEds_keV_per_ang`. Shell corrections are
    omitted, which is why NIST restricts ESTAR collision stopping to >=10 keV;
    the low-energy branch stays with Joy-Luo.
    """
    tau, beta_sq, f_minus = _bs_tau_terms(E_keV)
    I_rel = J_keV / _MC2_KEV
    log_term = np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel))
    return -_BS_PREFACTOR * rho_g_cm3 * Z / (A * beta_sq) * (log_term + f_minus - delta)


@njit(cache=True)
def _dEds_bs_compound_scalar(J_arr, coeff_arr, delta, E_i):
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    total = 0.0
    for i in range(J_arr.size):
        I_rel = J_arr[i] / _MC2_KEV
        total += coeff_arr[i] * (
            np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
        )

    return -_BS_PREFACTOR / beta_sq * total


@njit(cache=True)
def _dEds_bs_packed_scalar(L_Js, L_coeffs, L_deltas, L, n_el, E_i):
    """Berger-Seltzer stopping power read from the padded per-layer tables.

    Same arithmetic as :func:`_dEds_bs_compound_scalar`, but indexing the
    ``(n_layers, max_elements)`` rows the per-electron and CUDA cores use, with
    the per-layer density-effect scalar from ``L_deltas``.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    delta = L_deltas[L]

    total = 0.0
    for i in range(n_el):
        I_rel = L_Js[L, i] / _MC2_KEV
        total += L_coeffs[L, i] * (
            np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
        )
    return -_BS_PREFACTOR / beta_sq * total


@njit(cache=True)
def _dEds_bs_compound(J_arr, coeff_arr, delta, E_keV):
    out = np.empty_like(E_keV)

    for j in range(E_keV.size):
        E = E_keV[j]
        tau = E / _MC2_KEV
        gamma = 1.0 + tau
        beta_sq = 1.0 - 1.0 / (gamma * gamma)
        f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

        total = 0.0
        for i in range(J_arr.size):
            I_rel = J_arr[i] / _MC2_KEV
            total += coeff_arr[i] * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

        out[j] = -_BS_PREFACTOR / beta_sq * total
    return out


# ---- Joy-Luo / Berger-Seltzer splice ------------------------------------------
# Joy-Luo is kept as the low-energy branch: Berger-Seltzer omits shell
# corrections (why NIST restricts ESTAR collision stopping to >=10 keV) and the
# default E_cut_keV is 5 keV, so the low-energy branch is load-bearing.
#
# The crossover is strongly Z-dependent, so there is no good *global* splice
# energy. Measured over the 24 catalog elements the two laws cross at 2.66 keV
# (B) through 10.46 keV (Bi), monotone in I. Forcing one global energy leaves a
# step: the best available choice is 8.0 keV, which still mismatches by 1.84% in
# the worst element. The 2% agreement at 10 keV quoted in stopping-power.md is a
# carbon number and does not generalize.
#
# Because stopping is additive over elements, the splice does not have to be
# global. Each element's contribution is switched at *its own* crossover, so
# every term is continuous by construction and therefore so is the compound, for
# any material, with no per-material tuning and no fitted blend. Each element
# crosses exactly once above the energy where the Berger-Seltzer bracket is
# still positive (that zero sits below 0.71 keV for every catalog element, far
# under any crossover), so the crossover is well defined.
#
# The splice is C0, not C1: the log-slope d ln|dE/ds| / d ln E steps across it by
# 0.0145 for B (2.0% of the local slope) up to 0.0587 for Bi (8.9%), worst at
# high Z where the crossover sits highest. The midpoint solve
# (eq-stopping-cutoff-distance) only needs the value, but a kink inside the
# bracketing interval is the failure mode to check when the cores adopt this.


def _bs_joy_luo_crossover_keV(Z, A, J_keV):
    """Energy [keV] where Joy-Luo and Berger-Seltzer stopping powers are equal.

    Bisection rather than a closed form: the two laws differ in the shape of the
    logarithm, not just a constant. The result depends only on ``Z`` and
    ``J_keV`` (density cancels in the ratio), so it is a per-element constant
    computed once at table-build time.
    """
    Z = float(Z)
    A = float(A)
    J_keV = float(J_keV)

    def difference(E):
        return _dEds_keV_per_ang(Z, A, J_keV, 1.0, E) - _dEds_bs_keV_per_ang(Z, A, J_keV, 1.0, E)

    # Start above the zero of the Berger-Seltzer bracket, below which that law is
    # not physically signed and the difference is meaningless. That zero sits at
    # ~0.86 J for every catalog element, while the crossover is at ~35 J, so 2 J
    # brackets from below with room on both sides.
    lo = 2.0 * J_keV
    hi = 300.0
    if difference(lo) * difference(hi) > 0.0:
        raise ValueError(f"no Joy-Luo/Berger-Seltzer crossover in [{lo}, {hi}] keV for Z={Z}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if difference(lo) * difference(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_i):
    """Stopping power with each element on its own side of its own crossover."""
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = 0.0
    bs_total = 0.0
    for i in range(J_arr.size):
        J = J_arr[i]
        coeff = coeff_arr[i]
        if E_i < E_cross_arr[i]:
            joy_luo_total += coeff * np.log(1.166 * (E_i + k_arr[i] * J) / J)
        else:
            I_rel = J / _MC2_KEV
            bs_total += coeff * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

    return -7.85e-4 / E_i * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


@njit(cache=True)
def _dEds_spliced_compound(J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_keV):
    out = np.empty_like(E_keV)
    for j in range(E_keV.size):
        out[j] = _dEds_spliced_compound_scalar(
            J_arr, k_arr, coeff_arr, E_cross_arr, delta, E_keV[j]
        )
    return out


@njit(cache=True)
def _dEds_spliced_packed_scalar(L_Js, L_ks, L_coeffs, L_E_cross, delta, L, n_el, E_i):
    """Spliced stopping power read from the padded per-layer tables.

    Same arithmetic as :func:`_dEds_spliced_compound_scalar`, but indexing the
    ``(n_layers, max_elements)`` rows the per-electron and CUDA cores use. The
    padding is zero-filled and not a valid element, so ``n_el`` bounds the loop.
    """
    tau = E_i / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = 0.0
    bs_total = 0.0
    for i in range(n_el):
        J = L_Js[L, i]
        coeff = L_coeffs[L, i]
        if E_i < L_E_cross[L, i]:
            joy_luo_total += coeff * np.log(1.166 * (E_i + L_ks[L, i] * J) / J)
        else:
            I_rel = J / _MC2_KEV
            bs_total += coeff * (
                np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus - delta
            )

    return -7.85e-4 / E_i * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


# The crossover is a bisection, far too expensive for a hot loop, and it depends
# only on (Z, J) -- A and rho cancel in the ratio because both laws carry the
# same rho Z/A. Solve it once per element and reuse it for every layer that
# contains that element.
_CROSSOVER_CACHE: dict[str, float] = {}


def _element_crossover_keV(element, Z, A, J_keV):
    """Memoized Joy--Luo/Berger--Seltzer crossover energy [keV] for one element."""
    cached = _CROSSOVER_CACHE.get(element)
    if cached is None:
        cached = _bs_joy_luo_crossover_keV(Z, A, J_keV)
        _CROSSOVER_CACHE[element] = cached
    return cached


def spliced_stopping_keV_per_ang(composition, E_keV):
    """Spliced stopping power [keV/Angstrom] (negative) for a normalized composition.

    The host-side entry point for code outside the transport cores that needs
    the *same* model the cores evaluate: the cost proxy in ``campaign.sweep``
    and the frozen-rule replay in ``spectrum.diagnostics``. Both previously
    carried their own copy of the Joy--Luo constants and drifted the moment the
    model changed; delegating here is what keeps them honest.

    ``composition`` is the ``(element, n_i)`` sequence used everywhere else,
    with ``n_i`` in Angstrom^-3. Plain NumPy, no Numba: these are cold paths and
    should not pay a compile.
    """
    E = np.asarray(E_keV, dtype=float)
    tau = E / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)

    joy_luo_total = np.zeros_like(E)
    bs_total = np.zeros_like(E)
    for element, n_i in composition:
        params = TRANSPORT_ELEMENTS[element]
        Z = float(params["Z"])
        A = float(params["A"])
        J = float(params["J_keV"])
        k = 0.731 + 0.0688 * np.log10(Z)
        coeff = (n_i / 0.602214076) * Z
        I_rel = J / _MC2_KEV
        use_bs = E >= _element_crossover_keV(element, Z, A, J)
        joy_luo_total += np.where(use_bs, 0.0, coeff * np.log(1.166 * (E + k * J) / J))
        bs_total += np.where(
            use_bs,
            coeff * (np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus),
            0.0,
        )

    return -7.85e-4 / E * joy_luo_total - _BS_PREFACTOR / beta_sq * bs_total


def sternheimer_delta(element, E_keV):
    """Sternheimer density-effect correction delta for one element at ``E_keV``.

    Source: Sternheimer, Berger & Seltzer, *Atomic Data and Nuclear Data Tables*
    **30**, 261 (1984), in the PDG's presentation, with coefficients read from
    :data:`~pyrite.materials._transport_data.STERNHEIMER_DENSITY_EFFECT`. With
    ``x = log10(beta gamma)``,

        x < x_0:        delta = delta_0 * 10^(2 (x - x_0))
        x_0 <= x < x_1: delta = 2 ln(10) x - C_bar + a (x_1 - x)^k
        x >= x_1:       delta = 2 ln(10) x - C_bar

    delta_0 is zero for non-conductors, so the first branch vanishes for them.

    NO TRANSPORT PATH CALLS THIS. eq-stopping-bs carries delta as a parameter
    and every call site passes 0.0; this function exists to measure how much
    that omission costs, and is the instrument behind the bound quoted in
    stopping-power.md and pinned by ``test_stopping_density_effect.py``.

    Being per element it is not a compound's delta -- delta is a bulk property
    of the medium and does not Bragg-add. Applying it per element would be
    wrong; bounding a compound by its constituents is what it supports.
    """
    p = STERNHEIMER_DENSITY_EFFECT[element]
    E = np.asarray(E_keV, dtype=float)
    tau = E / _MC2_KEV
    gamma = 1.0 + tau
    x = np.log10(np.sqrt(1.0 - 1.0 / (gamma * gamma)) * gamma)
    asymptote = 2.0 * np.log(10.0) * x - p["C_bar"]
    return np.where(
        x < p["x0"],
        p["delta0"] * 10.0 ** (2.0 * (x - p["x0"])),
        asymptote + np.where(x < p["x1"], p["a"] * np.abs(p["x1"] - x) ** p["k"], 0.0),
    )


# Generation marker for the collision-stopping model every core evaluates. The
# splice is unconditional physics, not a selector, so this is a CONSTANT and
# follows the ``line_kinematics`` precedent: identity payloads and the per-case
# content key hash it, which moves every digest exactly once and orphans the
# records minted under the retired pure Joy--Luo model (rev-and-re-run) rather
# than letting them resume into -- or be served from the CAS for -- a run that
# computes different numbers. Bump it whenever the evaluated model changes:
# adding the density-effect term delta, or moving a crossover, is such a change.
STOPPING_MODEL = "sbethe-corrected-v1"
