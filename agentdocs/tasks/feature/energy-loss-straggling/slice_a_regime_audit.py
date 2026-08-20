"""Slice A of `feature/energy-loss-straggling`: regime audit + sequencing gate.

Two questions, both answered by measurement rather than assertion.

**1. What regime is the per-flight energy loss in?**  Reports, per material and
per energy, the elastic mean free path (the flight length), the Vavilov
parameter ``kappa`` per flight and over the whole CSDA range, and two bracketing
estimates of the inelastic collision count per flight.  ``kappa`` decides which
straggling distribution is admissible: ``kappa >~ 10`` Gaussian, ``0.01 <~ kappa
<~ 10`` Vavilov, ``kappa <~ 0.01`` Landau, and *below* the Landau limit --- a
countable number of collisions per step --- none of the three applies and an
Urban-style model is required.

**2. Does the systematic stopping-power error dominate the straggling bias?**
This is the sequencing gate.  Both phase errors are computed at the same
25 keV / 1 keV-photon / 1 um operating point that
``docs/validation/beam-transport/energy-step-convergence.md`` uses.

Definitions and sources
-----------------------

*Landau parameter* (Rossi; PDG "Passage of particles through matter", eq. for
xi).  For a step ``s`` in a medium of electron density ``n_e``,

.. math::

    \\xi = \\frac{2 \\pi r_e^2 m c^2 n_e s}{\\beta^2}.

Bragg-additive over elements because it depends on the medium only through
``n_e = sum_i n_i Z_i``.  Limiting case: ``s -> 0`` gives ``xi -> 0``, so
``kappa -> 0`` and the loss is a single-collision problem, as it must be.

*Vavilov parameter* ``kappa = xi / T_max``.  For an incident electron the
maximum energy transfer is the Moller value ``T_max = T / 2`` (indistinguishable
particles; the faster outgoing electron is by convention the primary).  Using
the heavy-particle ``T_max = 2 m c^2 beta^2 gamma^2`` here would be wrong and
would overstate ``kappa`` by ``~4/beta^2``.

*Collision-number bracket.*  There is no single "number of inelastic events"
without a lower cutoff, so two estimates bracket it:

- ``N_hard = xi (1/eps_min - 1/eps_max)`` with ``eps_min = I`` --- the count in
  the Rutherford/Moller ``eps^-2`` close-collision tail, which is the count that
  governs whether Landau/Vavilov theory applies at all.
- ``N_total = S s / I`` --- total mean loss divided by the mean excitation
  energy, a crude stand-in for "every excitation, soft ones included".  It is an
  over-estimate to the extent that the mean loss per collision exceeds ``I``.

The truth sits between them.  Both use the transport module's own ``J_keV``
(``I``) so nothing new is introduced.

*Phase sensitivity* (``energy-step-convergence.md``, "The clock this row
converges is CSDA's clock"):

.. math::

    \\delta\\phi = \\frac{\\omega}{c} \\int \\delta(1/\\beta)\\, ds, \\qquad
    \\delta(1/\\beta) = \\frac{\\delta E}{(\\beta\\gamma)^3 m c^2}.

The systematic figure here does **not** use that linearization: it integrates
``dE/ds`` for the retired pure-Joy--Luo model and for the current spliced model
and differences the two exact clocks ``t = int ds / beta(E(s))``.  The
linearized form is printed beside it as a cross-check.

*Jensen straggling bias.*  Straggling perturbs the mean arrival time at second
order because ``1/beta(E)`` is convex:

.. math::

    \\langle 1/\\beta \\rangle - 1/\\beta(\\langle E \\rangle)
      \\simeq \\tfrac{1}{2}\\sigma_E^2 \\frac{d^2 (1/\\beta)}{dE^2},
      \\qquad
    \\frac{d^2 (1/\\beta)}{dE^2} = \\frac{3\\gamma}{(\\beta\\gamma)^5 (mc^2)^2}.

The variance accumulates as ``d sigma_E^2 / ds = xi'(E) T_max(E)`` (the second
moment of the ``eps^-2`` spectrum truncated at ``T_max``; ``xi'`` is ``xi`` per
unit length).  This is the electron analogue of Bohr straggling with the Moller
cutoff in place of the heavy-particle one.  Limiting case: zero variance gives
zero bias, recovering the deterministic CSDA clock exactly.

Because the ``eps^-2`` spectrum is heavy-tailed, the full-cutoff variance is
dominated by rare hard collisions and the second-moment expansion is an
*upper* bound on the bias.  A soft-collision-only variant (cutoff at
``eps_soft_max``) is reported alongside, which is the regime the
``energy-step-convergence`` "~300 eV spread" figure describes.

Run:  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python \\
          agentdocs/tasks/feature/energy-loss-straggling/slice_a_regime_audit.py

Not a ledgered physics claim: slice A produces a measurement and a sequencing
decision, not a new model.  No `Validation:` marker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager

import numpy as np

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.montecarlo import transport

# --- constants -------------------------------------------------------------

R_E_ANG = 2.8179403262e-5  # classical electron radius [Ang]
MC2_KEV = 510.99895  # electron rest energy [keV]
HC_KEV_ANG = 12.39841984  # h c [keV Ang]
XI_COEFF = 2.0 * np.pi * R_E_ANG**2 * MC2_KEV  # keV Ang^2

E_CUT_KEV = 5.0  # the transport cutoff every core uses

# Low-Z through high-Z spread of the 50-material catalog.  hopg/diamond are
# pure carbon at two densities, hbn and 4h_sic are light compounds, silicon is
# the reference semiconductor, mos2/ws2/wse2 climb Z, and ptbi2 is the heaviest
# catalog material (it sits at the weak end of every splice table).
MATERIALS = ("hopg", "diamond", "hbn", "4h_sic", "silicon", "mos2", "ws2", "wse2", "ptbi2")

# The two bare compositions ``energy-step-convergence.md`` and
# ``checks/collision_statistics_refinement.py`` quote their per-flight numbers
# for.  Carried here so the audit reconciles against the doc directly rather
# than against a compound that merely contains the same element.
BARE = {"C(0.1136)": [("C", 0.1136)], "W(0.06305)": [("W", 0.06305)]}
ENERGIES_KEV = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0)

# Monte Carlo cross-check of the analytic flight length.
MC_MATERIALS = ("hopg", "silicon", "wse2")
MC_ENERGIES_KEV = (10.0, 25.0, 100.0)
MC_NE = 400
MC_SEEDS = (7, 101, 2024, 31337)


# --- material plumbing -----------------------------------------------------


def composition(material: str) -> list[tuple[str, float]]:
    """Catalog material -> transport's ``[(element, n_i [Ang^-3]), ...]``."""
    return build_cases(material_sweep(material), n_electrons=8, n_electrons_brem=4)[0][
        "composition"
    ]


def electron_density(comp) -> float:
    """``n_e = sum_i n_i Z_i`` [Ang^-3]."""
    return float(sum(n_i * TRANSPORT_ELEMENTS[el]["Z"] for el, n_i in comp))


def mean_excitation_keV(comp) -> float:
    """Bragg-averaged mean excitation energy: ``ln I = sum n_i Z_i ln I_i / n_e``."""
    n_e = electron_density(comp)
    acc = sum(
        n_i * TRANSPORT_ELEMENTS[el]["Z"] * np.log(TRANSPORT_ELEMENTS[el]["J_keV"])
        for el, n_i in comp
    )
    return float(np.exp(acc / n_e))


def beta_sq(E_keV):
    gamma = 1.0 + np.asarray(E_keV, dtype=float) / MC2_KEV
    return 1.0 - 1.0 / (gamma * gamma)


def elastic_mfp_ang(comp, E_keV) -> float:
    """Flight length: ``1 / sum_i n_i sigma_i`` with transport's own Browning fit.

    Reproduces the cores exactly: they precompute ``mott_numer = 3e-18 Z^1.7
    n_cm3`` (transport.py, the ``L_mott_numer`` build), sum the per-element
    rates in cm^-1, and take ``lam_ang = 1e8 / total_rate``.  ``n_cm3 = n_i
    * 1e24`` for ``n_i`` in Ang^-3.
    """
    E = np.atleast_1d(np.asarray(E_keV, dtype=float))
    rate = np.zeros_like(E)
    for el, n_i in comp:
        Z = float(TRANSPORT_ELEMENTS[el]["Z"])
        rate += n_i * 1.0e24 * transport._sigma_browning_cm2(Z, E)
    lam = 1.0e8 / rate
    return float(lam[0]) if lam.size == 1 else lam


def stopping_keV_per_ang(comp, E_keV) -> float:
    """Magnitude of the current (spliced) stopping power [keV/Ang]."""
    return float(-transport.spliced_stopping_keV_per_ang(comp, np.atleast_1d(E_keV))[0])


def csda_range_ang(comp, E0_keV, e_cut_keV=E_CUT_KEV) -> float:
    """``R = int_{E_cut}^{E0} dE / |dE/ds|``, log-spaced (|dE/ds| ~ 1/E)."""
    if E0_keV <= e_cut_keV:
        return 0.0
    u = np.linspace(np.log(e_cut_keV), np.log(E0_keV), 4001)
    E = np.exp(u)
    return float(np.trapezoid(E / -transport.spliced_stopping_keV_per_ang(comp, E), u))


# --- straggling regime -----------------------------------------------------


def xi_keV(comp, E_keV, s_ang) -> float:
    """Landau ``xi`` [keV] for a step ``s_ang`` at energy ``E_keV``."""
    return XI_COEFF * electron_density(comp) * s_ang / beta_sq(E_keV)


def kappa(comp, E_keV, s_ang) -> float:
    """Vavilov ``kappa = xi / T_max``, Moller ``T_max = E / 2``."""
    return xi_keV(comp, E_keV, s_ang) / (0.5 * E_keV)


def collision_bracket(comp, E_keV, s_ang) -> tuple[float, float]:
    """``(N_hard, N_total)`` inelastic collisions over ``s_ang``. See module doc."""
    I_keV = mean_excitation_keV(comp)
    T_max = 0.5 * E_keV
    if T_max <= I_keV:
        # Below I the eps^-2 close-collision picture has no support at all; the
        # loss is entirely soft excitation and N_hard is not defined.
        n_hard = float("nan")
    else:
        n_hard = xi_keV(comp, E_keV, s_ang) * (1.0 / I_keV - 1.0 / T_max)
    n_total = stopping_keV_per_ang(comp, E_keV) * s_ang / I_keV
    return float(n_hard), float(n_total)


def regime_label(k: float) -> str:
    if k >= 10.0:
        return "Gaussian"
    if k >= 0.01:
        return "Vavilov"
    return "Landau/few"


# --- clocks and phase ------------------------------------------------------


def _integrate_clock(comp, E0_keV, path_ang, n_steps=20000):
    """RK4 the CSDA energy along the path; return ``(E(s), t(s))``, ``t`` in Ang.

    ``t`` is the arrival time in Angstrom with ``c = 1``, matching the
    repository's ``t_ang`` convention: ``t = int ds / beta``.
    """
    s = np.linspace(0.0, path_ang, n_steps + 1)
    h = s[1] - s[0]
    E = np.empty_like(s)
    E[0] = E0_keV

    def dEds(e):
        return float(transport.spliced_stopping_keV_per_ang(comp, np.atleast_1d(e))[0])

    for i in range(n_steps):
        e = E[i]
        k1 = dEds(e)
        k2 = dEds(e + 0.5 * h * k1)
        k3 = dEds(e + 0.5 * h * k2)
        k4 = dEds(e + h * k3)
        E[i + 1] = e + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    inv_beta = 1.0 / np.sqrt(beta_sq(E))
    t = np.concatenate([[0.0], np.cumsum(0.5 * (inv_beta[1:] + inv_beta[:-1]) * h)])
    return s, E, t


def joy_luo_only(on: bool) -> None:
    """Force (or release) the retired pure-Joy--Luo stopping model.

    ``_element_crossover_keV`` is the single seam the host helper consults, so
    pushing it to ``inf`` degenerates the spliced form to Joy--Luo bit-for-bit
    rather than approximately (the mechanism
    ``tests/montecarlo/test_stopping_csda_range.py`` already relies on).
    """
    if on:
        joy_luo_only._saved = transport._element_crossover_keV  # type: ignore[attr-defined]
        transport._element_crossover_keV = lambda *_: np.inf  # type: ignore[assignment]
    else:
        transport._element_crossover_keV = joy_luo_only._saved  # type: ignore[attr-defined]
    transport._CROSSOVER_CACHE = {}


@contextmanager
def retired_model(on: bool):
    """Evaluate a block under the retired pure-Joy--Luo model when ``on``."""
    if not on:
        yield
        return
    joy_luo_only(True)
    try:
        yield
    finally:
        joy_luo_only(False)


def omega_over_c(E_photon_keV: float) -> float:
    """``omega/c = 2 pi / lambda`` [rad/Ang]."""
    return 2.0 * np.pi * E_photon_keV / HC_KEV_ANG


def systematic_phase_rad(comp, E0_keV, path_ang, E_photon_keV):
    """Phase displacement of the retired Joy--Luo clock against the current one.

    Exact: both clocks are integrated, not linearized.  Positive means Joy--Luo
    arrives *early* (it under-stops, so the electron stays faster).
    """
    _, E_bs, t_bs = _integrate_clock(comp, E0_keV, path_ang)
    joy_luo_only(True)
    try:
        _, E_jl, t_jl = _integrate_clock(comp, E0_keV, path_ang)
    finally:
        joy_luo_only(False)
    dphi = omega_over_c(E_photon_keV) * (t_bs[-1] - t_jl[-1])
    return dphi, E_jl[-1], E_bs[-1]


def straggling_phase_spread_rad(comp, E0_keV, path_ang, E_photon_keV, eps_max_keV=None):
    """RMS spread ``sigma_phi`` of the free-space arrival phase [rad].

    The energy deficit ``dE(s)`` is a partial sum, so ``Cov(dE(s), dE(s')) =
    V(min(s, s'))`` for the accumulated variance ``V``.  With
    ``g(s) = 1 / ((beta gamma)^3 mc^2)``,

    .. math::

        \\sigma_t^2 = 2 \\int_0^L g(s) V(s) \\Big[\\int_s^L g(s')\\,ds'\\Big] ds.

    **Caveat, and why this is a diagnostic and not the answer.**  This is the
    spread of the *free-space clock over the whole path*, which is the same
    proxy ``energy-step-convergence.md`` uses for its ~0.3 rad figure.  It is
    NOT the phase that enters the coherent sum: CXR emission reads
    ``(omega n_hat + g) . dr`` per segment, where most of this common-mode
    accumulation cancels.  Treating ``sigma_phi`` here as the Debye--Waller
    exponent would badly overstate the suppression.  Quantifying the real
    exponent is slice H's job, not slice A's.
    """
    s, E, _ = _integrate_clock(comp, E0_keV, path_ang)
    n_e = electron_density(comp)
    xi_rate = XI_COEFF * n_e / beta_sq(E)
    eps_max = 0.5 * E if eps_max_keV is None else np.full_like(E, eps_max_keV)
    h = s[1] - s[0]
    var_rate = xi_rate * eps_max
    V = np.concatenate([[0.0], np.cumsum(0.5 * (var_rate[1:] + var_rate[:-1]) * h)])

    gamma = 1.0 + E / MC2_KEV
    g = 1.0 / ((gamma * gamma - 1.0) ** 1.5 * MC2_KEV)
    tail = np.concatenate([np.cumsum((0.5 * (g[1:] + g[:-1]) * h)[::-1])[::-1], [0.0]])
    sigma_t_sq = 2.0 * float(np.trapezoid(g * V * tail, s))
    return omega_over_c(E_photon_keV) * np.sqrt(max(sigma_t_sq, 0.0))


def straggling_phase_rad(comp, E0_keV, path_ang, E_photon_keV, eps_max_keV=None):
    """Jensen mean-arrival-time bias from energy-loss straggling [rad].

    ``eps_max_keV=None`` uses the full Moller cutoff ``E/2`` (upper bound on the
    variance).  A finite value restricts the sum to soft collisions.
    Returns ``(delta_phi, sigma_E_end_keV)``.
    """
    s, E, _ = _integrate_clock(comp, E0_keV, path_ang)
    n_e = electron_density(comp)
    xi_rate = XI_COEFF * n_e / beta_sq(E)  # keV per Ang
    eps_max = 0.5 * E if eps_max_keV is None else np.full_like(E, eps_max_keV)
    var_rate = xi_rate * eps_max  # d sigma_E^2 / ds  [keV^2/Ang]
    h = s[1] - s[0]
    var = np.concatenate([[0.0], np.cumsum(0.5 * (var_rate[1:] + var_rate[:-1]) * h)])

    gamma = 1.0 + E / MC2_KEV
    bg = np.sqrt(gamma * gamma - 1.0)
    d2_inv_beta = 3.0 * gamma / (bg**5 * MC2_KEV**2)  # d^2(1/beta)/dE^2 [1/keV^2]

    integrand = 0.5 * var * d2_inv_beta
    dt = float(np.trapezoid(integrand, s))
    return omega_over_c(E_photon_keV) * dt, float(np.sqrt(var[-1]))


# --- Monte Carlo cross-check ----------------------------------------------


def measured_flight_lengths(comp, E0_keV, thickness_ang, seeds=MC_SEEDS, Ne=MC_NE):
    """Per-flight path lengths [Ang] from the production transport core."""
    out = []
    for seed in seeds:
        segs = transport.simulate_trajectories(
            E0_keV,
            Ne,
            thickness_ang,
            composition=comp,
            E_cut_keV=E_CUT_KEV,
            seed=seed,
            max_steps=60000,
            elastic_model="mott",
        )
        length = np.asarray(segs["L_ang"], dtype=float)
        # frozen energy model with max_dE_frac=0 -> one row per physical flight
        out.append(length)
    return np.concatenate(out)


# --- reports ---------------------------------------------------------------


def report_regime() -> None:
    print("=" * 100)
    print("PART 1  Per-flight regime audit")
    print("=" * 100)
    print(
        "lam_el  elastic mean free path = flight length [Ang]\n"
        "R_CSDA  continuous-slowing-down range from E0 to E_cut=5 keV [Ang]\n"
        "k_fl    Vavilov kappa over one flight;  k_R  the same over R_CSDA\n"
        "N_hard  Moller eps^-2 collisions per flight;  N_tot  S*lam/I per flight\n"
    )
    header = (
        f"{'material':>9} {'E[keV]':>7} {'lam_el':>9} {'R_CSDA':>11} "
        f"{'lam/R':>8} {'k_fl':>10} {'k_R':>10} {'N_hard':>8} {'N_tot':>8} {'regime(flight)':>15}"
    )
    for material in (*MATERIALS, *BARE):
        comp = BARE[material] if material in BARE else composition(material)
        print()
        print(header)
        for E0 in ENERGIES_KEV:
            lam = elastic_mfp_ang(comp, E0)
            R = csda_range_ang(comp, E0)
            k_fl = kappa(comp, E0, lam)
            k_R = kappa(comp, E0, R) if R > 0 else float("nan")
            n_hard, n_tot = collision_bracket(comp, E0, lam)
            print(
                f"{material:>9} {E0:>7.0f} {lam:>9.2f} {R:>11.1f} "
                f"{lam / R if R > 0 else float('nan'):>8.1e} {k_fl:>10.3e} {k_R:>10.3e} "
                f"{n_hard:>8.3f} {n_tot:>8.2f} {regime_label(k_fl):>15}"
            )


def report_mc() -> None:
    print()
    print("=" * 100)
    print("PART 2  Measured per-flight path-length distribution (production core)")
    print("=" * 100)
    print(
        f"{MC_NE} electrons x {len(MC_SEEDS)} seeds, slab = 2x R_CSDA so most flights\n"
        "close on a collision rather than a boundary.  'analytic' is lam_el at E0;\n"
        "the measured mean is lower because the electron slows and the Browning\n"
        "hazard rises as E falls.\n"
    )
    print(
        f"{'material':>9} {'E[keV]':>7} {'analytic':>9} {'mean':>9} {'median':>9} "
        f"{'p10':>9} {'p90':>9} {'p99':>10} {'N':>9}"
    )
    for material in MC_MATERIALS:
        comp = composition(material)
        for E0 in MC_ENERGIES_KEV:
            R = csda_range_ang(comp, E0)
            lengths = measured_flight_lengths(comp, E0, 2.0 * R)
            print(
                f"{material:>9} {E0:>7.0f} {elastic_mfp_ang(comp, E0):>9.2f} "
                f"{lengths.mean():>9.2f} {np.median(lengths):>9.2f} "
                f"{np.percentile(lengths, 10):>9.2f} {np.percentile(lengths, 90):>9.2f} "
                f"{np.percentile(lengths, 99):>10.2f} {lengths.size:>9d}"
            )


def report_gate() -> None:
    print()
    print("=" * 100)
    print("PART 3  Sequencing gate: systematic stopping bias vs straggling Jensen bias")
    print("=" * 100)
    comp = composition("hopg")
    E_photon = 1.0
    print(
        "Operating point of energy-step-convergence.md: graphite (hopg), 1 um path,\n"
        "1 keV photon.  Systematic = exact clock difference between the retired\n"
        "pure-Joy--Luo model and the current Berger--Seltzer splice.  Straggling =\n"
        "Jensen second-moment bias.  Both are |delta phi| in radians.\n"
    )
    print(
        f"{'E0[keV]':>8} {'L[um]':>7} {'E_JL_end':>9} {'E_BS_end':>9} "
        f"{'dphi_sys':>10} {'sig_full':>9} {'dphi_full':>10} {'sig_soft':>9} {'dphi_soft':>10} "
        f"{'sig_phi*':>10}"
    )
    print("(* free-space clock spread -- a diagnostic, NOT the coherent-sum exponent)")
    for E0, L_um in ((25.0, 1.0), (100.0, 1.0), (25.0, 5.0), (100.0, 10.0)):
        L = L_um * 1.0e4
        dphi_sys, E_jl, E_bs = systematic_phase_rad(comp, E0, L, E_photon)
        dphi_full, sig_full = straggling_phase_rad(comp, E0, L, E_photon)
        # Soft-collision-only variant: cutoff at 100x the mean excitation
        # energy of graphite (~7.8 keV -> 7.8 eV*1000); this is the branch that
        # reproduces the "~300 eV spread" figure in energy-step-convergence.md.
        eps_soft = 30.0 * mean_excitation_keV(comp)
        dphi_soft, sig_soft = straggling_phase_rad(comp, E0, L, E_photon, eps_max_keV=eps_soft)
        spread = straggling_phase_spread_rad(comp, E0, L, E_photon, eps_max_keV=eps_soft)
        print(
            f"{E0:>8.0f} {L_um:>7.1f} {E_jl:>9.3f} {E_bs:>9.3f} "
            f"{abs(dphi_sys):>10.3f} {sig_full:>9.4f} {abs(dphi_full):>10.3f} "
            f"{sig_soft:>9.4f} {abs(dphi_soft):>10.3f} {spread:>10.1f}"
        )

    print()
    print("Cross-check of the exact systematic against the linearized form")
    print("delta_phi = (omega/c) L <delta_E> / (beta gamma)^3 mc^2 :")
    for E0 in (25.0, 100.0):
        L = 1.0e4
        gamma = 1.0 + E0 / MC2_KEV
        bg3_mc2 = (gamma * gamma - 1.0) ** 1.5 * MC2_KEV
        S = stopping_keV_per_ang(comp, E0)
        joy_luo_only(True)
        try:
            S_jl = stopping_keV_per_ang(comp, E0)
        finally:
            joy_luo_only(False)
        dE_mean = 0.5 * (S - S_jl) * L  # path-average of a linearly growing gap
        lin = omega_over_c(E_photon) * L * dE_mean / bg3_mc2
        print(
            f"  E0={E0:>5.0f} keV  (beta gamma)^3 mc^2 = {bg3_mc2:>8.2f} keV  "
            f"S_BS/S_JL = {S / S_jl:>6.4f}  linearized dphi = {abs(lin):>8.3f} rad"
        )

    print()
    print("Residual systematic, for scale: what a REMAINING x% stopping error costs")
    print("at 25 keV / 1 um / 1 keV photon, on the corrected (spliced) clock:")
    E0, L = 25.0, 1.0e4
    gamma = 1.0 + E0 / MC2_KEV
    bg3_mc2 = (gamma * gamma - 1.0) ** 1.5 * MC2_KEV
    S = stopping_keV_per_ang(comp, E0)
    for frac in (0.001, 0.005, 0.01, 0.02, 0.06):
        lin = omega_over_c(E_photon) * L * (0.5 * frac * S * L) / bg3_mc2
        print(f"  {frac * 100:>5.1f}% stopping error -> {lin:>8.3f} rad")


def report_b0() -> None:
    """Slice B0: are the doc's contested figures artifacts of the stopping splice?

    ``docs/validation/beam-transport/energy-step-convergence.md`` states four
    numbers that slice A's measurements did not all reproduce.  The hypothesis
    under test is that they were measured against the retired pure-Joy--Luo
    stopping power and never re-measured after ``2c51754`` / ``dcdb1cd`` /
    ``d15a8ef`` spliced Berger--Seltzer in above each element's crossover.  Every
    figure is therefore recomputed under BOTH models through the
    ``_element_crossover_keV -> inf`` seam, so "explained by the splice" is a
    measured verdict and not an inference from the commit dates.
    """
    print()
    print("=" * 100)
    print("PART 4  B0: contested doc figures, current splice vs retired Joy--Luo")
    print("=" * 100)

    print("\n--- Figures 1 and 2: inelastic events per flight at 25 keV ---")
    print("doc: 'about 1.0 [per flight] in carbon at 25 keV ... about 0.11 in tungsten'")
    print(
        f"\n{'material':>12} {'model':>9} {'lam_el':>8} {'I[keV]':>9} {'S[keV/A]':>10} "
        f"{'N_hard':>9} {'N_tot':>8} {'doc':>7}"
    )
    for material, doc_value in (("C(0.1136)", 1.0), ("W(0.06305)", 0.11)):
        comp = BARE[material]
        lam = elastic_mfp_ang(comp, 25.0)  # elastic only: the splice cannot move it
        I_keV = mean_excitation_keV(comp)
        for label, on in (("splice", False), ("joy_luo", True)):
            with retired_model(on):
                S = stopping_keV_per_ang(comp, 25.0)
                n_hard, n_tot = collision_bracket(comp, 25.0, lam)
            print(
                f"{material:>12} {label:>9} {lam:>8.2f} {I_keV:>9.5f} {S:>10.5f} "
                f"{n_hard:>9.3f} {n_tot:>8.3f} {doc_value:>7.2f}"
            )

    print("\n  Sensitivity: which single input would have to move to reach the doc value?")
    for material, doc_value in (("C(0.1136)", 1.0), ("W(0.06305)", 0.11)):
        comp = BARE[material]
        lam = elastic_mfp_ang(comp, 25.0)
        I_keV = mean_excitation_keV(comp)
        S = stopping_keV_per_ang(comp, 25.0)
        n_tot = S * lam / I_keV
        print(
            f"  {material:>12}  N_tot={n_tot:.3f} vs doc {doc_value:.2f}: "
            f"needs S x{doc_value / n_tot:.2f}, or lam x{doc_value / n_tot:.2f}, "
            f"or I x{n_tot / doc_value:.2f} ({I_keV * n_tot / doc_value * 1e3:.1f} eV)"
        )

    print("\n--- Figure 3: kappa = 0.015, carbon at 25 keV over 1 um ---")
    comp_c = BARE["C(0.1136)"]
    for label in ("splice", "joy_luo"):
        with retired_model(label == "joy_luo"):
            xi = xi_keV(comp_c, 25.0, 1.0e4)
            k = kappa(comp_c, 25.0, 1.0e4)
        print(f"  {label:>9}: xi = {xi:.5f} keV, T_max = 12.5 keV, kappa = {k:.5f}")
    print("  (xi depends on n_e, s and beta only -- no stopping power enters, so the")
    print("   two models are identical by construction, not by coincidence.)")

    print("\n--- Figure 4: the ~0.3 rad Jensen bias, hopg / 25 keV / 1 um / 1 keV photon ---")
    comp = composition("hopg")
    E0, L, E_photon = 25.0, 1.0e4, 1.0
    eps_soft = 30.0 * mean_excitation_keV(comp)
    print(f"{'model':>9} {'cutoff':>12} {'sigma_E[keV]':>13} {'dphi[rad]':>10}")
    for label in ("splice", "joy_luo"):
        with retired_model(label == "joy_luo"):
            for cut_label, cut in (("Moller E/2", None), (f"soft {eps_soft:.2f} keV", eps_soft)):
                dphi, sigma = straggling_phase_rad(comp, E0, L, E_photon, eps_max_keV=cut)
                print(f"{label:>9} {cut_label:>12} {sigma:>13.4f} {abs(dphi):>10.3f}")

    # Invert the bias for the sigma_E the doc's figure implies, and compare it
    # against a Poisson plasmon estimate: N = S L / eps_p events of size eps_p
    # gives sigma_E = sqrt(N) eps_p = sqrt(S L eps_p).  This is the "order 300 eV
    # of loss spread" sentence's own microphysics ("plasmons, most probable loss
    # 25--33 eV in graphite"), so it is the arithmetic the figure most likely used.
    dphi_ref, sigma_ref = straggling_phase_rad(comp, E0, L, E_photon)
    sigma_for_03 = sigma_ref * np.sqrt(0.3 / abs(dphi_ref))
    print(f"\n  sigma_E that would give exactly 0.3 rad: {sigma_for_03:.4f} keV")
    S_c = stopping_keV_per_ang(comp, E0)
    print(f"  mean loss over 1 um in hopg: {S_c * L:.3f} keV (S = {S_c:.5f} keV/Ang)")
    for eps_p_eV in (25.0, 33.0):
        eps_p = eps_p_eV * 1e-3
        sigma_p = np.sqrt(S_c * L * eps_p)
        dphi_p = abs(dphi_ref) * (sigma_p / sigma_ref) ** 2
        print(
            f"  Poisson plasmon, eps_p = {eps_p_eV:.0f} eV: N = {S_c * L / eps_p:.0f}, "
            f"sigma_E = {sigma_p:.4f} keV -> {dphi_p:.3f} rad"
        )


# --- slice B: the Urban parameterisation in the measured regime --------------

URBAN_E0_KEV = 10.0e-3  # ionisation level E_0 = 10 eV (Geant4 PRM)
URBAN_RATE = 0.55  # r, the model's single tuned parameter


def urban_levels(Z: float, I_keV: float):
    """``(f_1, E_1, f_2, E_2)`` from the Urban sum rules [keV].

    Geant4 Physics Reference Manual, "Energy loss fluctuations", Urban model
    (after Bichsel 1988).  ``E_2 = 10 Z^2`` eV approximates the K-shell binding
    energy and ``Z f_2 = 2`` recovers its occupancy; ``f_1 + f_2 = 1`` and
    ``f_1 ln E_1 + f_2 ln E_2 = ln I`` then fix the loosely bound level.
    """
    f_2 = 0.0 if Z <= 1.0 else 2.0 / Z
    f_1 = 1.0 - f_2
    E_2 = 10.0e-3 * Z * Z
    E_1 = float(np.exp((np.log(I_keV) - f_2 * np.log(E_2)) / f_1))
    return f_1, E_1, f_2, E_2


def urban_channels(Z: float, I_keV: float, S_keV_per_ang: float, E_keV: float, s_ang: float):
    """Mean Urban collision counts over a step, and the mean-loss closure check.

    Returns ``(n_1, n_2, n_3, closure, frac_E2)``: the Poisson means
    ``<n_i> = s Sigma_i`` for the two excitation channels and the ionisation
    channel, the ratio of the model's own mean loss to ``S s`` (exactly 1 when
    the sum rules hold -- the limiting-case check), and the fraction of the mean
    loss carried by the ``E_2`` K-shell channel.

    ``C = dE/dx`` is the model's normalisation, so the mean is inherited from
    whatever stopping power the transport supplies rather than re-derived.
    """
    f_1, E_1, f_2, E_2 = urban_levels(Z, I_keV)
    b2 = float(beta_sq(E_keV))
    two_mc2_bg2 = 2.0 * MC2_KEV * (b2 / (1.0 - b2))
    T_up = 0.5 * E_keV  # unrestricted: no delta-ray cut, so T_up = T_max

    def L(level):
        return np.log(two_mc2_bg2 / level) - b2

    L_I = L(I_keV)
    # Geant4 drops a level whose logarithmic factor has gone non-positive; the
    # count is then zero rather than negative.
    sig_1 = max(S_keV_per_ang * (f_1 / E_1) * L(E_1) / L_I * (1.0 - URBAN_RATE), 0.0)
    sig_2 = max(S_keV_per_ang * (f_2 / E_2) * L(E_2) / L_I * (1.0 - URBAN_RATE), 0.0)
    sig_3 = (
        S_keV_per_ang
        * (T_up - URBAN_E0_KEV)
        / (URBAN_E0_KEV * T_up * np.log(T_up / URBAN_E0_KEV))
        * URBAN_RATE
    )
    mean_ion = URBAN_E0_KEV * T_up / (T_up - URBAN_E0_KEV) * np.log(T_up / URBAN_E0_KEV)
    mean_loss = (sig_1 * E_1 + sig_2 * E_2 + sig_3 * mean_ion) * s_ang
    return (
        sig_1 * s_ang,
        sig_2 * s_ang,
        sig_3 * s_ang,
        mean_loss / (S_keV_per_ang * s_ang),
        sig_2 * E_2 / (S_keV_per_ang) if S_keV_per_ang else float("nan"),
    )


def urban_variance_keV2(
    Z: float, I_keV: float, S_keV_per_ang: float, E_keV: float, s_ang: float
) -> float:
    """Urban's variance of the loss over a step [keV^2].

    Urban's loss is a compound Poisson sum, so its variance is
    ``sum_i <n_i> <E^2>_i`` with no cross terms.  The two excitation channels
    deposit a fixed quantum, giving ``<E^2>_i = E_i^2``; the ionisation channel's
    ``1/E^2`` spectrum on ``[E_0, T_up]`` has ``<E^2>_3 = E_0 T_up`` exactly.

    The comparator is the analytic second moment of the same ``eps^-2`` spectrum
    truncated at the Moller ceiling, ``xi T_max`` -- the electron analogue of
    Bohr straggling and the quantity slice A already integrates for its Jensen
    bias.  Agreement here is what makes the selected sampler usable for the
    phase observable; a mismatch is a variance the sampler would have to be
    corrected for, and is slice C's to close.
    """
    f_1, E_1, f_2, E_2 = urban_levels(Z, I_keV)
    n_1, n_2, n_3, _, _ = urban_channels(Z, I_keV, S_keV_per_ang, E_keV, s_ang)
    T_up = 0.5 * E_keV
    var_urban = n_1 * E_1 * E_1 + n_2 * E_2 * E_2 + n_3 * URBAN_E0_KEV * T_up
    return float(var_urban)


def report_urban() -> None:
    """Slice B: does the Urban parameterisation resolve the measured regime?

    Urban is applied per element, matching the Bragg-additive form the mean
    stopping power already uses, so each element carries its own ``Z``, ``I``
    and its own share ``C_i`` of ``dE/dx``.

    Two failure modes are checked by measurement rather than assumed away:

    - ``E_2 = 10 Z^2`` eV exceeds the Moller ceiling ``T_max = E/2`` whenever
      ``E < 20 Z^2`` eV.  The model was built for HEP energies where that never
      happens; here it demands ``E > 1.1 MeV`` for tungsten.  Exciting a level
      above the available energy is unphysical, so the column is flagged.
    - ``ln(2 mc^2 (beta gamma)^2 / E_2) - beta^2`` goes non-positive below
      ``(beta gamma)^2 = E_2 / 2 mc^2``, at which point Geant4 drops the level.

    ``closure`` is the limiting-case check: the sum rules make the model's mean
    loss identically ``S s``, so any departure from 1.000 is an implementation
    error, not physics.
    """
    print()
    print("=" * 100)
    print("PART 5  B: Urban parameterisation against the measured per-flight regime")
    print("=" * 100)
    print(
        "Per element, step = the analytic elastic mean free path of the host material.\n"
        "n1,n2,n3 are Poisson means per flight; dE/I is the mean loss in units of I,\n"
        "which is the quantity the PRM's own 'a few multiples of I_exc' floor bounds.\n"
        "!T flags E_2 > T_max (unphysical level); !L flags the dropped E_2 channel.\n"
    )
    hosts = {
        "C(0.1136)": BARE["C(0.1136)"],
        "W(0.06305)": BARE["W(0.06305)"],
        "silicon": composition("silicon"),
        "ws2": composition("ws2"),
    }
    header = (
        f"{'host':>11} {'el':>3} {'E[keV]':>7} {'I[eV]':>8} {'E_1[eV]':>9} {'E_2[eV]':>10} "
        f"{'f_2':>7} {'n1':>8} {'n2':>9} {'n3':>8} {'dE/I':>7} {'E2 frac':>8} "
        f"{'closure':>8} {'flags':>6}"
    )
    for host, comp in hosts.items():
        print()
        print(header)
        for E0 in ENERGIES_KEV:
            lam = elastic_mfp_ang(comp, E0)
            for element, n_i in comp:
                Z = float(TRANSPORT_ELEMENTS[element]["Z"])
                I_keV = float(TRANSPORT_ELEMENTS[element]["J_keV"])
                S_i = stopping_keV_per_ang([(element, n_i)], E0)
                f_1, E_1, f_2, E_2 = urban_levels(Z, I_keV)
                n_1, n_2, n_3, closure, frac = urban_channels(Z, I_keV, S_i, E0, lam)
                flags = ("!T" if E_2 > 0.5 * E0 else "") + ("!L" if n_2 == 0.0 else "")
                print(
                    f"{host:>11} {element:>3} {E0:>7.0f} {I_keV * 1e3:>8.1f} "
                    f"{E_1 * 1e3:>9.1f} {E_2 * 1e3:>10.1f} {f_2:>7.4f} "
                    f"{n_1:>8.4f} {n_2:>9.2e} {n_3:>8.4f} {S_i * lam / I_keV:>7.3f} "
                    f"{frac:>8.4f} {closure:>8.4f} {flags:>6}"
                )

    print()
    print("Domain boundary: the E_2 = 10 Z^2 eV level is below the Moller ceiling")
    print("T_max = E/2 only for E > 20 Z^2 eV.  Per element:")
    seen = set()
    for comp in hosts.values():
        for element, _ in comp:
            if element in seen:
                continue
            seen.add(element)
            Z = float(TRANSPORT_ELEMENTS[element]["Z"])
            print(
                f"  {element:>3}  Z={Z:>5.1f}  E_2 = {10.0 * Z * Z:>9.1f} eV  "
                f"needs E > {20.0 * Z * Z * 1e-3:>9.2f} keV"
            )

    print()
    print("Variance closure: Urban's compound-Poisson variance against the analytic")
    print("Moller second moment xi T_max, per flight, summed Bragg-additively.")
    print(f"{'host':>11} {'E[keV]':>7} {'sig_urban[keV]':>15} {'sig_moller[keV]':>16} {'ratio':>8}")
    for host, comp in hosts.items():
        for E0 in ENERGIES_KEV:
            lam = elastic_mfp_ang(comp, E0)
            var_u = 0.0
            for element, n_i in comp:
                Z = float(TRANSPORT_ELEMENTS[element]["Z"])
                I_keV = float(TRANSPORT_ELEMENTS[element]["J_keV"])
                S_i = stopping_keV_per_ang([(element, n_i)], E0)
                var_u += urban_variance_keV2(Z, I_keV, S_i, E0, lam)
            var_a = xi_keV(comp, E0, lam) * 0.5 * E0
            print(
                f"{host:>11} {E0:>7.0f} {np.sqrt(var_u):>15.5f} {np.sqrt(var_a):>16.5f} "
                f"{var_u / var_a:>8.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("regime", "mc", "gate", "b0", "urban", "all"),
        default="all",
        help="run one section (the Monte Carlo section is the slow one)",
    )
    args = parser.parse_args()
    if args.part in ("regime", "all"):
        report_regime()
    if args.part in ("mc", "all"):
        report_mc()
    if args.part in ("gate", "all"):
        report_gate()
    if args.part in ("b0", "all"):
        report_b0()
    if args.part in ("urban", "all"):
        report_urban()


if __name__ == "__main__":
    main()
