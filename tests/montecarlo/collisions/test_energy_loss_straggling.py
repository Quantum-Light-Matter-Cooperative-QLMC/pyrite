"""Urban energy-loss fluctuation sampler: units, limits, signs, first two moments.

Slice C of ``feature/energy-loss-straggling``. The model, its source and its
selection are recorded in ``pyrite.montecarlo.transport`` next to the code and in
``agentdocs/tasks/feature/energy-loss-straggling/README.md`` (slice B).

What is pinned here, and why each pin exists:

* **Closure.** ``<dE> = |dE/dx| s`` exactly, per element and per compound. This
  is the property Urban was selected *for* -- Landau, Vavilov and Bohr each
  derive their own mean and would contradict the Joy--Luo/Berger--Seltzer splice.
* **The ``E_2`` re-solve.** ``E_2 = 10 Z^2`` eV is inadmissible below
  ``E = 20 Z^2`` eV; the naive clamp-to-zero Geant4 uses makes the mean
  *overshoot*, and the re-solve on ``f_1 = 1, E_1 = I`` removes the overshoot
  exactly.
* **Variance.** Against the analytic Moller second moment ``xi T_max``. The
  measured deficit is recorded as a range, not tuned away; the PRM's width
  correction is deliberately not applied.
* **The zero-step limit.** Every ``<n_i> -> 0`` as ``s -> 0``, so the sampler
  degenerates to today's deterministic ``|dE/dx| s``.

The *shape* is deliberately not pinned. The PRM's own floor (mean loss at least
a few multiples of ``I``) is violated in nearly every catalog cell, so only the
first two moments are claimed.
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import (
    _BS_PREFACTOR,
    _MC2_KEV,
    _URBAN_E0_KEV,
    _URBAN_RATE,
    TRANSPORT_ELEMENTS,
    _dEds_spliced_compound_scalar,
    _dEds_spliced_element_scalar,
    _urban_channels_scalar,
    _urban_ionisation_keV,
    _urban_levels_scalar,
    _urban_moments_element_scalar,
    _urban_poisson_scalar,
    _urban_sample_compound_keV,
    _urban_sample_element_keV,
    stream_keys,
    urban_element_table,
    urban_loss_moments_keV,
)

ELEMENTS = ("C", "Si", "S", "W")
ENERGIES_KEV = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0)

# Bare-element stand-ins for the catalog's chemistry, at catalog number
# densities: graphite and tungsten are the two ends of slice A's matrix.
COMPOUNDS = {
    "graphite": [("C", 0.1136)],
    "tungsten": [("W", 0.06305)],
    "ws2": [("W", 0.01706), ("S", 0.03412)],
    "silicon": [("Si", 0.04996)],
}


def element_params(element):
    params = TRANSPORT_ELEMENTS[element]
    return float(params["Z"]), float(params["J_keV"])


def beta_sq(E_keV):
    gamma = 1.0 + E_keV / _MC2_KEV
    return 1.0 - 1.0 / (gamma * gamma)


def moller_second_moment_keV2(composition, E_keV, s_ang):
    """``xi T_max``: the analytic second moment of the ``eps^-2`` Moller spectrum.

    ``xi = 2 pi r_e^2 mc^2 N_A (Z/A) rho s / beta^2`` in the same
    ``_BS_PREFACTOR``/``coeff`` bookkeeping the Berger--Seltzer stopping power
    already uses, and ``T_max = E/2``. This is the comparator slice A integrated
    for its Jensen bias, and the one the sampler has to be judged against --
    not the model's own shape.
    """
    _, _, _, coeff_arr, _ = urban_element_table(composition)
    xi = _BS_PREFACTOR * float(coeff_arr.sum()) * s_ang / beta_sq(E_keV)
    return xi * 0.5 * E_keV


def naive_clamp_mean_over_C(Z, J_keV, E_keV):
    """Geant4's clamp: keep the full sum rules, floor a negative count at zero.

    Reference implementation of the failure mode the re-solve exists to remove,
    written out here rather than referenced so the test states what it rejects.
    Returns ``<dE>/(C s)`` -- the closure ratio, which should be 1 and is not.
    """
    b2 = beta_sq(E_keV)
    tau = E_keV / _MC2_KEV
    two_mc2_bg2 = 2.0 * _MC2_KEV * tau * (tau + 2.0)
    T_up = 0.5 * E_keV
    f_2 = 2.0 / Z
    f_1 = 1.0 - f_2
    E_2 = 1.0e-2 * Z * Z
    E_1 = np.exp((np.log(J_keV) - f_2 * np.log(E_2)) / f_1)
    L_I = np.log(two_mc2_bg2 / J_keV) - b2
    soft = (1.0 - _URBAN_RATE) / L_I
    sigma_1 = max(soft * (f_1 / E_1) * (np.log(two_mc2_bg2 / E_1) - b2), 0.0)
    sigma_2 = max(soft * (f_2 / E_2) * (np.log(two_mc2_bg2 / E_2) - b2), 0.0)
    sigma_3 = (
        _URBAN_RATE * (T_up - _URBAN_E0_KEV) / (_URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV))
    )
    mean_3 = _URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV) / (T_up - _URBAN_E0_KEV)
    return sigma_1 * E_1 + sigma_2 * E_2 + sigma_3 * mean_3


def sample_losses(Z, J_keV, C, E_keV, s_ang, n_draws, seed=20260820):
    """``n_draws`` independent element losses, one stream key per draw."""
    out = np.empty(n_draws)
    keys = stream_keys(seed, n_draws)
    for i in range(n_draws):
        out[i], _ = _urban_sample_element_keV(Z, J_keV, C, E_keV, s_ang, keys[i], np.uint64(0))
    return out


# ---- units, limits and signs --------------------------------------------------


def test_sigma_units_are_inverse_length_and_counts_scale_with_step():
    """``<n_i> = s Sigma_i`` is dimensionless and linear in the step."""
    Z, J = element_params("C")
    valid, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(Z, J, 0.01, 25.0)
    assert valid == 1.0
    # Sigma_i [1/Ang] times E_i [keV] must recover C [keV/Ang].
    assert sigma_1 * E_1 + sigma_2 * E_2 > 0.0
    mean_1, var_1 = _urban_moments_element_scalar(Z, J, 0.01, 25.0, 1.0)
    mean_2, var_2 = _urban_moments_element_scalar(Z, J, 0.01, 25.0, 2.0)
    assert mean_2 == pytest.approx(2.0 * mean_1, rel=1e-15)
    assert var_2 == pytest.approx(2.0 * var_1, rel=1e-15)


def test_sampled_loss_is_positive_and_stopping_power_is_negative():
    """Sign convention: ``dE/dx`` is negative, the sampled loss is positive."""
    composition = COMPOUNDS["graphite"]
    dEds = float(_dEds_spliced_compound_scalar(*_arrays(composition), 0.0, 25.0))
    assert dEds < 0.0
    losses = sample_losses(*element_params("C"), -dEds, 25.0, 50.0, 200)
    assert losses.min() >= 0.0
    assert losses.max() > 0.0


def _arrays(composition):
    _, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
    return J_arr, k_arr, coeff_arr, E_cross_arr


def test_ionisation_draw_inverts_its_own_cdf():
    """``E_k = E_0/(1 - u (T_up - E_0)/T_up)`` is the inverse of ``F`` on ``1/E^2``."""
    T_up = 12.5
    E_grid = np.linspace(_URBAN_E0_KEV, T_up, 257)
    norm = _URBAN_E0_KEV * T_up / (T_up - _URBAN_E0_KEV)
    u_grid = norm * (1.0 / _URBAN_E0_KEV - 1.0 / E_grid)
    for E, u in zip(E_grid, u_grid, strict=True):
        assert _urban_ionisation_keV(float(u), T_up) == pytest.approx(float(E), rel=1e-12)
    assert _urban_ionisation_keV(0.0, T_up) == pytest.approx(_URBAN_E0_KEV, rel=1e-15)
    assert _urban_ionisation_keV(1.0 - 1e-15, T_up) <= T_up * (1.0 + 1e-9)


def test_ionisation_draw_moments_match_the_closed_forms():
    """``<E>_3 = E_0 T_up ln(T_up/E_0)/(T_up - E_0)`` and ``<E^2>_3 = E_0 T_up``.

    Evaluated by quadrature over the *energy* variable, where the ``1/E^2``
    density is smooth, rather than over ``u``, where the second moment's
    integrand piles up against ``u = 1``.
    """
    T_up = 12.5
    E = np.linspace(_URBAN_E0_KEV, T_up, 2_000_001)
    density = (_URBAN_E0_KEV * T_up / (T_up - _URBAN_E0_KEV)) / (E * E)
    assert np.trapezoid(density, E) == pytest.approx(1.0, rel=1e-6)
    mean_closed = _URBAN_E0_KEV * T_up * np.log(T_up / _URBAN_E0_KEV) / (T_up - _URBAN_E0_KEV)
    assert np.trapezoid(E * density, E) == pytest.approx(mean_closed, rel=1e-5)
    assert np.trapezoid(E * E * density, E) == pytest.approx(_URBAN_E0_KEV * T_up, rel=1e-5)


# ---- closure: the mean is the transport's own dE/dx ---------------------------


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("E_keV", ENERGIES_KEV)
def test_element_closure_is_exact(element, E_keV):
    """``<dE> = C s`` to float64 rounding, for every element and energy."""
    Z, J = element_params(element)
    C = 0.037  # arbitrary positive |dE/dx|; closure is independent of it
    mean, _ = _urban_moments_element_scalar(Z, J, C, E_keV, 3.0)
    assert mean == pytest.approx(C * 3.0, rel=1e-14)


@pytest.mark.parametrize("name", sorted(COMPOUNDS))
@pytest.mark.parametrize("E_keV", ENERGIES_KEV)
def test_compound_closure_matches_the_spliced_stopping_power(name, E_keV):
    """Per-element Bragg application keeps the compound mean on the splice.

    The comparator is the cores' own ``_dEds_spliced_compound_scalar``, not a
    re-derivation: this is the property that makes the sampler droppable into
    transport without moving any existing mean.
    """
    composition = COMPOUNDS[name]
    s_ang = 7.5
    dEds = float(_dEds_spliced_compound_scalar(*_arrays(composition), 0.0, E_keV))
    mean, _ = urban_loss_moments_keV(composition, E_keV, s_ang)
    assert mean == pytest.approx(-dEds * s_ang, rel=1e-13)


def test_per_element_split_reproduces_the_compound_stopping_power():
    """``_dEds_spliced_element_scalar`` sums back to the compound law."""
    for name, composition in COMPOUNDS.items():
        Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
        for E_keV in ENERGIES_KEV:
            total = sum(
                _dEds_spliced_element_scalar(
                    J_arr[i], k_arr[i], coeff_arr[i], E_cross_arr[i], 0.0, E_keV
                )
                for i in range(Z_arr.size)
            )
            expected = float(
                _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_keV)
            )
            assert total == pytest.approx(expected, rel=1e-14), (name, E_keV)


# ---- the E_2 admissibility re-solve -------------------------------------------


@pytest.mark.parametrize("element", ELEMENTS)
def test_e2_channel_switches_off_exactly_at_20_Z_squared_eV(element):
    """``E_2 = 10 Z^2`` eV clears ``T_max = E/2`` only above ``E = 20 Z^2`` eV.

    C above 0.72 keV, Si above 3.92 keV, S above 5.12 keV, W above 109.5 keV --
    boundaries Geant4 never reaches and PyRITE straddles.
    """
    Z, J = element_params(element)
    E_boundary = 20.0e-3 * Z * Z
    below = _urban_channels_scalar(Z, J, 1.0, E_boundary * 0.999)
    above = _urban_channels_scalar(Z, J, 1.0, E_boundary * 1.001)
    assert below[3] == 0.0, "E_2 channel must be off below the Moller ceiling"
    assert above[3] > 0.0, "E_2 channel must be on above it"
    # Below the boundary the re-solve puts the single level at I itself.
    assert below[2] == pytest.approx(J, rel=1e-15)
    assert above[2] < J


@pytest.mark.parametrize(
    ("E_keV", "clamp_closure"), [(1.0, 1.0187), (2.0, 1.0098), (5.0, 1.0038), (10.0, 1.0010)]
)
def test_resolve_removes_the_naive_clamp_overshoot_for_tungsten(E_keV, clamp_closure):
    """The clamp deletes a negative contribution; the re-solve does not.

    Slice B measured the overshoot at 1.0187/1.0098/1.0038/1.0010 for W at
    1/2/5/10 keV. Reproduced here so the reason the re-solve is load-bearing
    stays visible, then shown removed.
    """
    Z, J = element_params("W")
    assert naive_clamp_mean_over_C(Z, J, E_keV) == pytest.approx(clamp_closure, abs=5e-5)
    mean, _ = _urban_moments_element_scalar(Z, J, 1.0, E_keV, 1.0)
    assert mean == pytest.approx(1.0, rel=1e-14)


def test_resolve_preserves_the_mean_across_the_admissibility_boundary():
    """Closure is 1 on both sides of every boundary, with no step in the mean."""
    Z, J = element_params("W")
    for E_keV in np.linspace(100.0, 120.0, 41):
        mean, _ = _urban_moments_element_scalar(Z, J, 0.05, float(E_keV), 2.0)
        assert mean == pytest.approx(0.05 * 2.0, rel=1e-14)


def test_levels_satisfy_the_log_sum_rule():
    """``f_1 ln E_1 + f_2 ln E_2 = ln I`` -- the constraint closure rests on."""
    for element in ELEMENTS:
        Z, J = element_params(element)
        for E_keV in ENERGIES_KEV:
            b2 = beta_sq(E_keV)
            tau = E_keV / _MC2_KEV
            two = 2.0 * _MC2_KEV * tau * (tau + 2.0)
            f_1, E_1, f_2, E_2 = _urban_levels_scalar(Z, J, 0.5 * E_keV, two, b2)
            assert f_1 + f_2 == pytest.approx(1.0, rel=1e-15)
            rule = f_1 * np.log(E_1) + (f_2 * np.log(E_2) if f_2 > 0.0 else 0.0)
            assert rule == pytest.approx(np.log(J), rel=1e-12)


# ---- variance against the analytic Moller second moment -----------------------


@pytest.mark.parametrize("name", sorted(COMPOUNDS))
@pytest.mark.parametrize("E_keV", ENERGIES_KEV)
def test_variance_tracks_moller_within_the_measured_deficit(name, E_keV):
    """Urban's compound-Poisson variance against ``xi T_max``.

    Measured band over the catalog with the ``E_2`` re-solve in place: 0.70--1.55
    in variance, i.e. sigma within -16%/+24%. Slice B measured 0.73--1.42 with
    the naive clamp; the re-solve widens it because it raises the surviving level
    from the sum-rule ``E_1`` to ``I``, which the mean is invariant to and the
    variance is not. The deficit is RECORDED, not corrected: the PRM's width
    correction is not applied, because applying it would re-open the closure
    property Urban was selected for.
    """
    composition = COMPOUNDS[name]
    s_ang = 4.0
    _, var = urban_loss_moments_keV(composition, E_keV, s_ang)
    ratio = var / moller_second_moment_keV2(composition, E_keV, s_ang)
    assert 0.70 <= ratio <= 1.55, f"{name} {E_keV} keV: variance ratio {ratio:.4f}"


def test_variance_is_the_compound_poisson_sum_with_no_cross_terms():
    """``Var = s (Sigma_1 E_1^2 + Sigma_2 E_2^2 + Sigma_3 E_0 T_up)``."""
    Z, J = element_params("Si")
    C, E_keV, s_ang = 0.02, 25.0, 6.0
    _, sigma_1, E_1, sigma_2, E_2, sigma_3 = _urban_channels_scalar(Z, J, C, E_keV)
    expected = s_ang * (
        sigma_1 * E_1 * E_1 + sigma_2 * E_2 * E_2 + sigma_3 * _URBAN_E0_KEV * 0.5 * E_keV
    )
    _, var = _urban_moments_element_scalar(Z, J, C, E_keV, s_ang)
    assert var == pytest.approx(expected, rel=1e-15)


# ---- the zero-step limit ------------------------------------------------------


def test_zero_step_degenerates_to_the_deterministic_loss():
    """As ``s -> 0`` every ``<n_i> -> 0``: the sampler returns exactly today's loss.

    The limiting case required by the physics contract. At ``s`` small enough
    that all three Poisson means are below 1e-9, every draw is identically zero,
    which is the deterministic ``C s`` to the same order; and the analytic mean
    equals ``C s`` at every ``s``, so the degeneration is in distribution, not
    only in the limit.
    """
    Z, J = element_params("C")
    C, E_keV = 0.0113, 25.0
    s_tiny = 1e-12
    valid, sigma_1, _, sigma_2, _, sigma_3 = _urban_channels_scalar(Z, J, C, E_keV)
    assert valid == 1.0
    assert max(sigma_1, sigma_2, sigma_3) * s_tiny < 1e-9
    losses = sample_losses(Z, J, C, E_keV, s_tiny, 500)
    assert np.all(losses == 0.0)
    mean, var = _urban_moments_element_scalar(Z, J, C, E_keV, s_tiny)
    assert mean == pytest.approx(C * s_tiny, rel=1e-14)
    assert var < mean * 0.5 * E_keV * (1.0 + 1e-12)


def test_zero_step_is_exactly_zero():
    losses = sample_losses(*element_params("W"), 0.05, 25.0, 0.0, 16)
    assert np.all(losses == 0.0)


def test_inadmissible_parameterisation_falls_back_to_the_deterministic_loss():
    """Below ``T_up = E_0`` there is no continuum: return ``C s``, do not sample."""
    Z, J = element_params("W")
    C, s_ang = 0.05, 3.0
    E_keV = 0.015  # T_up = 7.5 eV < E_0 = 10 eV
    assert _urban_channels_scalar(Z, J, C, E_keV)[0] == 0.0
    loss, counter = _urban_sample_element_keV(Z, J, C, E_keV, s_ang, np.uint64(11), np.uint64(3))
    assert loss == C * s_ang
    assert counter == np.uint64(3), "the fallback must not consume the stream"


# ---- the sampler reproduces its own moments -----------------------------------


def test_poisson_variate_matches_its_mean_and_variance():
    """Both single-chunk and decomposed exact-Poisson paths."""
    for lam, n_draws, tol in ((0.7, 40000, 0.03), (3.0, 40000, 0.02), (200.0, 8000, 0.03)):
        counts = np.empty(n_draws)
        keys = stream_keys(4242, n_draws)
        for i in range(n_draws):
            counts[i], _ = _urban_poisson_scalar(lam, keys[i], np.uint64(0))
        assert counts.mean() == pytest.approx(lam, rel=tol)
        assert counts.var() == pytest.approx(lam, rel=4.0 * tol)
        assert counts.min() >= 0


def test_poisson_variate_consumes_a_fixed_number_of_draws():
    """A counter-addressed stream must advance by an amount the caller knows."""
    for lam, expected in ((0.0, 0), (2.5, 1), (64.0, 1), (200.0, 4), (500.0, 8)):
        _, counter = _urban_poisson_scalar(lam, np.uint64(9), np.uint64(17))
        assert int(counter) - 17 == expected


def test_poisson_large_mean_keeps_poisson_skewness_and_split_law():
    """The former rounded-Gaussian branch erased the third cumulant."""
    lam = 200.0
    n_draws = 80000
    keys = stream_keys(8675309, n_draws)
    counts = np.empty(n_draws)
    split_counts = np.empty(n_draws)
    for i in range(n_draws):
        counts[i], _ = _urban_poisson_scalar(lam, keys[i], np.uint64(0))
        left, counter = _urban_poisson_scalar(0.5 * lam, keys[i], np.uint64(0))
        right, _ = _urban_poisson_scalar(0.5 * lam, keys[i], counter)
        split_counts[i] = left + right

    centered = counts - counts.mean()
    skewness = np.mean(centered**3) / np.mean(centered**2) ** 1.5
    assert skewness == pytest.approx(lam**-0.5, abs=0.015)
    assert split_counts.mean() == pytest.approx(counts.mean(), rel=0.003)
    assert split_counts.var() == pytest.approx(counts.var(), rel=0.015)


@pytest.mark.parametrize(
    ("element", "n_i", "E_keV", "s_ang"),
    [("C", 0.1136, 25.0, 5000.0), ("W", 0.06305, 25.0, 2000.0), ("Si", 0.04996, 100.0, 20000.0)],
)
def test_sampled_first_two_moments_match_the_analytic_ones(element, n_i, E_keV, s_ang):
    """Monte Carlo closes on the analytic mean and variance.

    Deterministic, not statistical: the stream is counter-addressed off a fixed
    seed, so these numbers do not move between runs. The steps are CSDA-range
    scale rather than flight scale because the estimators need it -- over one
    flight the mean loss is a fraction of a single continuum quantum
    (``mean/sigma`` ~ 0.05), which is the measured regime, not a defect. The
    variance band is the wider one because the ``1/E^2`` continuum has a finite
    but large fourth moment, so the sample variance converges slowly.
    """
    Z, J = element_params(element)
    _, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table([(element, n_i)])
    C = abs(
        _dEds_spliced_element_scalar(J_arr[0], k_arr[0], coeff_arr[0], E_cross_arr[0], 0.0, E_keV)
    )
    mean, var = _urban_moments_element_scalar(Z, J, C, E_keV, s_ang)
    losses = sample_losses(Z, J, C, E_keV, s_ang, 30000)
    assert losses.mean() == pytest.approx(mean, rel=0.03)
    assert losses.var() == pytest.approx(var, rel=0.20)


def test_compound_sampler_is_bragg_additive_and_reproducible():
    """Same key and counter give the same loss; the counter only moves forward."""
    composition = COMPOUNDS["ws2"]
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
    args = (Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 25.0, 20.0)
    key = stream_keys(31337, 6)[5]
    first, counter_1 = _urban_sample_compound_keV(*args, key, np.uint64(0))
    second, counter_2 = _urban_sample_compound_keV(*args, key, np.uint64(0))
    assert first == second
    assert counter_1 == counter_2
    # One uniform per non-empty Poisson channel plus one per continuum quantum.
    # The E_2 channel is off for both W and S at 25 keV, so two per element is
    # the floor; what matters is that the advance is a function of the inputs.
    assert int(counter_1) >= 2 * len(composition)


def test_compound_sampled_mean_closes_on_a_scaled_stopping_power():
    """A material-table scale changes the mean without changing RNG plumbing."""
    composition = COMPOUNDS["ws2"]
    Z_arr, J_arr, k_arr, coeff_arr, E_cross_arr = urban_element_table(composition)
    E_keV, s_ang, n_draws, stopping_scale = 25.0, 3000.0, 20000, 1.7
    losses = np.empty(n_draws)
    keys = stream_keys(777, n_draws)
    for i in range(n_draws):
        losses[i], _ = _urban_sample_compound_keV(
            Z_arr,
            J_arr,
            k_arr,
            coeff_arr,
            E_cross_arr,
            0.0,
            E_keV,
            s_ang,
            keys[i],
            np.uint64(0),
            stopping_scale,
        )
    expected = (
        -float(_dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_keV))
        * s_ang
        * stopping_scale
    )
    assert losses.mean() == pytest.approx(expected, rel=0.05)
