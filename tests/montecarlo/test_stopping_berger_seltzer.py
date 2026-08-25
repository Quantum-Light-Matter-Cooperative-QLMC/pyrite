"""Berger-Seltzer/ICRU-37 collision stopping: units, limits, and additivity.

Pins the closed form against the two things the repository already knows: the
Joy-Luo prefactor it must reduce to non-relativistically, and the measured
Joy-Luo/ICRU-37 ratios tabulated as ``tbl-stopping-validity-ceiling`` in
``docs/physics/beam-transport/stopping-power.md``.

Validation: relativistic-bethe-stopping
"""

import numpy as np
import pytest

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.montecarlo.transport import (
    _BS_PREFACTOR,
    _LN2,
    _MC2_KEV,
    _bs_joy_luo_crossover_keV,
    _dEds_bs_compound,
    _dEds_bs_compound_scalar,
    _dEds_bs_keV_per_ang,
    _dEds_bs_packed_scalar,
    _dEds_compound_scalar,
    _dEds_keV_per_ang,
    _dEds_spliced_compound,
    _dEds_spliced_compound_scalar,
    _dEds_spliced_packed_scalar,
    spliced_stopping_keV_per_ang,
)

# Carbon at the graphite number density the transport tests use.
_Z, _A, _J_KEV = 6.0, 12.011, 0.078
_N_ANG3 = 0.1136
_RHO = _N_ANG3 * 1e24 * _A / 6.02214076e23
_COEFF = (_N_ANG3 / 0.602214076) * _Z

_SWEEP_KEV = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0])


def test_prefactor_reduces_to_the_joy_luo_constant():
    """1.535e-6/beta^2 -> 7.85e-4/E once beta^2 -> 2E/mc^2."""
    assert _BS_PREFACTOR * _MC2_KEV == pytest.approx(7.85e-4, rel=1e-3)


def test_f_minus_at_zero_energy():
    """F^-(0) = 1 - ln 2, the constant offset the non-relativistic limit keeps."""
    tau = 0.0
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    assert f_minus == pytest.approx(1.0 - np.log(2.0), rel=1e-12)


def test_non_relativistic_limit():
    """Converges to -(P mc^2 / 2E) rho Z/A [2 ln(E/I) + 1 - ln 2] as tau -> 0."""

    def non_relativistic(E_keV):
        bracket = 2.0 * np.log(E_keV / _J_KEV) + 1.0 - _LN2
        return -_BS_PREFACTOR * _MC2_KEV / (2.0 * E_keV) * _RHO * _Z / _A * bracket

    ratios = [
        _dEds_bs_keV_per_ang(_Z, _A, _J_KEV, _RHO, E) / non_relativistic(E)
        for E in (10.0, 1.0, 0.1)
    ]
    # The residual is O(tau) -- the (tau+2)/2 factor and beta^2 against 2E/mc^2 --
    # so it is ~2% at 10 keV and falls with energy toward unity.
    assert ratios[0] == pytest.approx(1.0, abs=3e-2)
    assert ratios[-1] == pytest.approx(1.0, abs=5e-4)
    assert all(abs(b - 1.0) < abs(a - 1.0) for a, b in zip(ratios, ratios[1:], strict=False))


def test_matches_the_documented_validity_ceiling_table():
    """Reproduces tbl-stopping-validity-ceiling (carbon) to its stated precision."""
    expected = {10.0: 0.98, 25.0: 0.94, 50.0: 0.88, 100.0: 0.78, 200.0: 0.63, 300.0: 0.52}
    for E, ratio in expected.items():
        joy_luo = _dEds_keV_per_ang(_Z, _A, _J_KEV, _RHO, E)
        berger_seltzer = _dEds_bs_keV_per_ang(_Z, _A, _J_KEV, _RHO, E)
        assert joy_luo / berger_seltzer == pytest.approx(ratio, abs=0.005)


def test_sign_and_monotonicity_over_the_swept_range():
    """Negative everywhere, and magnitude falls monotonically to 300 keV."""
    values = _dEds_bs_compound(np.array([_J_KEV]), np.array([_COEFF]), 0.0, _SWEEP_KEV.copy())
    assert np.all(values < 0.0)
    assert np.all(np.diff(np.abs(values)) < 0.0)


def test_compound_coefficient_is_an_exact_rewrite_of_rho_z_over_a():
    """c_i = n_i Z_i / 0.602214076 reproduces the elemental rho Z/A form."""
    for E in _SWEEP_KEV:
        elemental = _dEds_bs_keV_per_ang(_Z, _A, _J_KEV, _RHO, E)
        compound = _dEds_bs_compound_scalar(np.array([_J_KEV]), np.array([_COEFF]), 0.0, E)
        assert compound == pytest.approx(elemental, rel=1e-12)


def test_bragg_additivity():
    """A two-element sum equals the sum of the single-element evaluations."""
    J = np.array([_J_KEV, 0.173])  # C, Si
    coeff = np.array([_COEFF, (0.05 / 0.602214076) * 14.0])
    for E in _SWEEP_KEV:
        together = _dEds_bs_compound_scalar(J, coeff, 0.0, E)
        apart = sum(
            _dEds_bs_compound_scalar(J[i : i + 1], coeff[i : i + 1], 0.0, E) for i in (0, 1)
        )
        assert together == pytest.approx(apart, rel=1e-12)


def test_packed_and_vectorized_agree_with_the_scalar_form():
    J = np.array([_J_KEV, 0.173])
    coeff = np.array([_COEFF, (0.05 / 0.602214076) * 14.0])
    L_Js = np.zeros((2, 3))
    L_coeffs = np.zeros((2, 3))
    L_Js[1, :2] = J
    L_coeffs[1, :2] = coeff
    L_deltas = np.array([0.0, 0.0])

    vector = _dEds_bs_compound(J, coeff, 0.0, _SWEEP_KEV.copy())
    for idx, E in enumerate(_SWEEP_KEV):
        scalar = _dEds_bs_compound_scalar(J, coeff, 0.0, E)
        packed = _dEds_bs_packed_scalar(L_Js, L_coeffs, L_deltas, 1, 2, E)
        assert vector[idx] == pytest.approx(scalar, rel=1e-14)
        assert packed == pytest.approx(scalar, rel=1e-14)


def test_density_effect_enters_linearly_and_reduces_the_magnitude():
    """delta is a per-medium scalar multiplying the same Bragg sum."""
    delta = 0.25
    for E in (100.0, 300.0):
        base = _dEds_bs_compound_scalar(np.array([_J_KEV]), np.array([_COEFF]), 0.0, E)
        shifted = _dEds_bs_compound_scalar(np.array([_J_KEV]), np.array([_COEFF]), delta, E)
        assert abs(shifted) < abs(base)

        tau = E / _MC2_KEV
        gamma = 1.0 + tau
        beta_sq = 1.0 - 1.0 / (gamma * gamma)
        expected_shift = _BS_PREFACTOR / beta_sq * _COEFF * delta
        assert shifted - base == pytest.approx(expected_shift, rel=1e-12)


# ---- Joy-Luo / Berger-Seltzer splice ------------------------------------------


def _catalog_compositions():
    """Every catalog material as a list of ``(element, n_i)`` pairs."""
    from pyrite.materials.catalog import load_material_catalog

    cat = load_material_catalog()
    out = {}
    for key in cat.material_keys:
        crystal_key = cat.material(key).crystal_key
        if crystal_key is None:
            continue
        out[key] = list(cat.crystal(crystal_key).composition)
    return out


def test_crossover_is_where_the_two_laws_are_equal():
    """The bisection solves ratio == 1, per element, for all 24 catalog elements."""
    for el, p in TRANSPORT_ELEMENTS.items():
        Z, A, J = float(p["Z"]), float(p["A"]), float(p["J_keV"])
        E_cross = _bs_joy_luo_crossover_keV(Z, A, J)
        assert 1.0 < E_cross < 300.0, f"{el}: crossover {E_cross} keV out of range"
        joy_luo = _dEds_keV_per_ang(Z, A, J, 1.0, E_cross)
        berger_seltzer = _dEds_bs_keV_per_ang(Z, A, J, 1.0, E_cross)
        assert joy_luo == pytest.approx(berger_seltzer, rel=1e-10), el


def test_crossover_spans_a_wide_z_range_so_no_global_splice_is_continuous():
    """Records the measurement that motivates the per-element splice."""
    crossovers = {
        el: _bs_joy_luo_crossover_keV(p["Z"], p["A"], p["J_keV"])
        for el, p in TRANSPORT_ELEMENTS.items()
    }
    assert crossovers["B"] == pytest.approx(2.66, abs=0.05)
    assert crossovers["Bi"] == pytest.approx(10.46, abs=0.05)

    # A single global splice energy cannot be continuous for every element. The
    # best available choice still steps by more than 1.5% in the worst element.
    def worst_mismatch(E_splice):
        worst = 0.0
        for p in TRANSPORT_ELEMENTS.values():
            Z, A, J = float(p["Z"]), float(p["A"]), float(p["J_keV"])
            ratio = _dEds_keV_per_ang(Z, A, J, 1.0, E_splice) / _dEds_bs_keV_per_ang(
                Z, A, J, 1.0, E_splice
            )
            worst = max(worst, abs(ratio - 1.0))
        return worst

    assert worst_mismatch(8.0) == pytest.approx(0.0184, abs=0.002)
    assert min(worst_mismatch(E) for E in np.arange(2.0, 12.0, 0.25)) > 0.015


def _spliced_inputs(composition):
    J, k, coeff, E_cross = [], [], [], []
    for el, n_i in composition:
        p = TRANSPORT_ELEMENTS[el]
        Z, A, J_i = float(p["Z"]), float(p["A"]), float(p["J_keV"])
        J.append(J_i)
        k.append(0.731 + 0.0688 * np.log10(Z))
        coeff.append((n_i / 0.602214076) * Z)
        E_cross.append(_bs_joy_luo_crossover_keV(Z, A, J_i))
    return (np.array(J), np.array(k), np.array(coeff), np.array(E_cross))


def _branch_jump(J, k, coeff, E_cross, index):
    """Relative size of the step the splice takes when element ``index`` switches.

    Evaluated at the crossover itself, so it measures the discontinuity alone and
    not the O(eps) smooth variation either branch has across a finite offset.
    """
    Ec = float(E_cross[index])
    one = slice(index, index + 1)
    joy_luo = _dEds_compound_scalar(J[one], k[one], coeff[one], Ec)
    berger_seltzer = _dEds_bs_compound_scalar(J[one], coeff[one], 0.0, Ec)
    total = _dEds_spliced_compound_scalar(J, k, coeff, E_cross, 0.0, Ec)
    return abs(joy_luo - berger_seltzer) / abs(total)


@pytest.mark.parametrize("element", list(TRANSPORT_ELEMENTS))
def test_splice_is_continuous_for_every_element(element):
    """Value continuity at the crossover, to machine precision, per element."""
    J, k, coeff, E_cross = _spliced_inputs([(element, 0.05)])
    assert _branch_jump(J, k, coeff, E_cross, 0) < 1e-12


def test_splice_is_continuous_for_every_catalog_material():
    """Each element switches at its own crossover, so every compound is continuous."""
    compositions = _catalog_compositions()
    assert len(compositions) > 20
    for key, composition in compositions.items():
        J, k, coeff, E_cross = _spliced_inputs(composition)
        for index in range(len(composition)):
            jump = _branch_jump(J, k, coeff, E_cross, index)
            assert jump < 1e-12, f"{key} at {E_cross[index]} keV: relative step {jump}"


def test_splice_is_continuous_across_a_swept_energy_grid():
    """No step survives in a fine sweep: the largest jump is the smooth gradient."""
    J, k, coeff, E_cross = _spliced_inputs([("C", 0.1136), ("W", 0.0633)])
    grid = np.linspace(1.0, 30.0, 200_001)
    values = _dEds_spliced_compound(J, k, coeff, E_cross, 0.0, grid.copy())
    steps = np.abs(np.diff(values)) / np.abs(values[:-1])
    # A discontinuous splice would show a step orders of magnitude above the
    # neighbouring smooth gradient; require the largest to be within 2x the 99.9th
    # percentile of the rest.
    assert steps.max() < 2.0 * np.percentile(steps, 99.9)


def test_splice_selects_the_expected_branch_away_from_the_crossover():
    """Joy-Luo well below every crossover, Berger-Seltzer well above."""
    composition = [("C", 0.1136)]
    J, k, coeff, E_cross = _spliced_inputs(composition)

    # rel=1e-8, not tighter: the spliced form goes through the number-density
    # coefficient and the elemental form through rho Z/A, so the two differ in
    # the last bits of the density round-trip.
    low = 1.0
    assert low < E_cross.min()
    assert _dEds_spliced_compound_scalar(J, k, coeff, E_cross, 0.0, low) == pytest.approx(
        _dEds_keV_per_ang(_Z, _A, _J_KEV, _RHO, low), rel=1e-8
    )

    high = 100.0
    assert high > E_cross.max()
    assert _dEds_spliced_compound_scalar(J, k, coeff, E_cross, 0.0, high) == pytest.approx(
        _dEds_bs_keV_per_ang(_Z, _A, _J_KEV, _RHO, high), rel=1e-12
    )


def test_splice_derivative_kink_is_bounded():
    """The splice is C0, not C1.

    Measured across the catalog the log-slope steps by 0.0145 (B, 2.0% of the
    local slope) to 0.0587 (Bi, 8.9%) -- worst at high Z, where the crossover
    sits highest. The midpoint solve only needs the value, but this is the
    number to check the cutoff bracket against when the cores adopt the splice.
    """
    h = 1e-5
    for p in TRANSPORT_ELEMENTS.values():
        Z, A, J = float(p["Z"]), float(p["A"]), float(p["J_keV"])
        Ec = _bs_joy_luo_crossover_keV(Z, A, J)

        def log_slope(fn, E, Z=Z, A=A, J=J):
            hi = np.log(abs(fn(Z, A, J, 1.0, E * (1.0 + h))))
            lo = np.log(abs(fn(Z, A, J, 1.0, E * (1.0 - h))))
            return (hi - lo) / (2.0 * h)

        joy_luo_slope = log_slope(_dEds_keV_per_ang, Ec)
        bs_slope = log_slope(_dEds_bs_keV_per_ang, Ec)
        assert abs(bs_slope - joy_luo_slope) < 0.10 * abs(joy_luo_slope)
        assert abs(bs_slope - joy_luo_slope) < 0.06


def test_spliced_vectorized_agrees_with_the_scalar_form():
    J, k, coeff, E_cross = _spliced_inputs([("C", 0.1136), ("Si", 0.05)])
    grid = np.linspace(1.0, 300.0, 401)
    vector = _dEds_spliced_compound(J, k, coeff, E_cross, 0.0, grid.copy())
    for idx in (0, 7, 123, 400):
        expected = _dEds_spliced_compound_scalar(J, k, coeff, E_cross, 0.0, grid[idx])
        assert vector[idx] == pytest.approx(expected, rel=1e-14)


# ---- the four evaluation paths must agree ------------------------------------
# Integration (checklist D) put the spliced model behind four separate
# implementations: the compound-scalar form the lockstep and grooved cores call,
# the padded form the per-electron core calls, the host NumPy helper the cost
# proxy and the diagnostics replay call, and the device-array twin in
# ``spectrum.lines``. They are separate code, so nothing but a test keeps them
# in step -- and the copies that used to live in ``campaign.sweep`` and
# ``spectrum.diagnostics`` are exactly how the last model change drifted.


def test_host_helper_matches_the_compiled_compound_form():
    composition = [("C", 0.1136), ("Si", 0.05)]
    J, k, coeff, E_cross = _spliced_inputs(composition)
    for E in _SWEEP_KEV:
        compiled = _dEds_spliced_compound_scalar(J, k, coeff, E_cross, 0.0, E)
        host = float(spliced_stopping_keV_per_ang(composition, E))
        assert host == pytest.approx(compiled, rel=1e-14)


def test_host_helper_matches_the_padded_form():
    """The per-electron and CUDA cores index padded rows; padding must not leak."""
    composition = [("C", 0.1136), ("Si", 0.05)]
    J, k, coeff, E_cross = _spliced_inputs(composition)
    max_el = 4  # deliberately over-wide: rows 2 and 3 are zero-filled padding
    L_Js = np.zeros((1, max_el))
    L_ks = np.zeros((1, max_el))
    L_coeffs = np.zeros((1, max_el))
    L_E_cross = np.zeros((1, max_el))
    L_Js[0, :2], L_ks[0, :2] = J, k
    L_coeffs[0, :2], L_E_cross[0, :2] = coeff, E_cross

    for E in _SWEEP_KEV:
        packed = _dEds_spliced_packed_scalar(L_Js, L_ks, L_coeffs, L_E_cross, 0.0, 0, 2, E)
        assert packed == pytest.approx(
            float(spliced_stopping_keV_per_ang(composition, E)), rel=1e-14
        )


def test_stopping_mirrors_agree():
    """``spectrum.lines`` carries a device-array twin that cannot delegate.

    Its rows may live on the GPU, so it re-implements the model against ``xp``
    instead of calling the host helper. This is the pin that keeps the two from
    drifting; ``campaign.sweep`` and ``spectrum.diagnostics`` need no such pin
    because they now delegate outright.
    """
    from pyrite._backend import REAL
    from pyrite.montecarlo.spectrum import lines

    composition = [("C", 0.1136), ("Si", 0.05)]
    grid = np.linspace(1.0, 300.0, 97)
    layer_index = np.zeros(grid.size, dtype=np.int64)

    mirrored = lines._spliced_stopping_magnitude_xp(
        np.asarray(grid, dtype=REAL), layer_index, [composition]
    )
    expected = -spliced_stopping_keV_per_ang(composition, grid)
    np.testing.assert_allclose(mirrored, expected, rtol=1e-6)


def test_cost_proxy_delegates_to_the_transport_model():
    """``campaign.sweep``'s CSDA range must track the model it predicts runtime for."""
    from pyrite.campaign.sweep import _dEds_magnitude_keV_per_ang

    composition = [("C", 0.1136)]
    grid = np.linspace(5.0, 300.0, 64)
    np.testing.assert_allclose(
        _dEds_magnitude_keV_per_ang(composition, grid),
        -spliced_stopping_keV_per_ang(composition, grid),
        rtol=0,
        atol=0,
    )


def test_frozen_rule_replay_delegates_to_the_transport_model():
    """``spectrum.diagnostics`` must reapply the rule the cores actually used."""
    from pyrite.montecarlo.spectrum.diagnostics import _stopping_keV_per_ang

    composition = [("C", 0.1136), ("Si", 0.05)]
    grid = np.linspace(5.0, 300.0, 64)
    np.testing.assert_allclose(
        _stopping_keV_per_ang(grid, composition),
        spliced_stopping_keV_per_ang(composition, grid),
        rtol=0,
        atol=0,
    )


# ---- the cutoff solve across the splice --------------------------------------


@pytest.mark.parametrize("transport_core", ["lockstep", "per-electron"])
def test_cutoff_solve_is_well_posed_across_the_crossover(transport_core):
    """The midpoint cutoff solve evaluates dE/ds at (E_start + E_cut)/2.

    That midpoint can land on the slope kink: tungsten crosses at 9.72 keV, so
    an ``E_cut`` of 5 keV puts the evaluation energy right on the crossover for
    every flight starting near 14.4 keV. The splice is C0, so the *value* is
    single-valued there and the solve stays well posed -- but this is the check
    checklist C asked for rather than assumed.
    """
    from pyrite.montecarlo.transport import simulate_trajectories

    tungsten = [("W", 0.0632)]
    result = simulate_trajectories(
        E0_keV=14.5,
        Ne=256,
        thickness_ang=1.0e8,
        composition=tungsten,
        E_cut_keV=5.0,
        elastic_model="sr",
        seed=11,
        energy_model="midpoint",
        transport_core=transport_core,
    )
    assert result["n_cutoff_stopped"] > 0
    # Flights land exactly on the floor, never through it, and every recorded
    # length stays positive and finite across the kink.
    assert np.all(result["E_end_keV"] >= 5.0)
    assert int((result["E_end_keV"] == 5.0).sum()) == result["n_cutoff_stopped"]
    assert np.all(np.isfinite(result["L_ang"]))
    assert np.all(result["L_ang"] > 0.0)


# ---- the CUDA twin ----------------------------------------------------------
# transport/_jit_device.py imports cupy at module scope, so without a GPU it
# cannot even be imported, let alone run -- its own tests skip. The repository's
# existing precedent for this (test_cuda_source_uses_cpu_reference_cutoff_and_
# termination_rules) is to read the source as text; that check covers the cutoff
# and termination control flow but says nothing about the stopping arithmetic.
# These pin the part that changed. They are NOT a substitute for running the
# kernel on a device. The stopping constants and `_dEds_packed` live in
# `_jit_device.py`, beside the other device-side helpers the kernels call.


def _jit_device_source():
    from importlib import resources

    return resources.files("pyrite.montecarlo.transport").joinpath("_jit_device.py").read_text()


def test_cuda_stopping_constants_match_the_cpu_values():
    """A constant edited on one side only is the drift this path is exposed to."""
    import re

    source = _jit_device_source()
    declared = dict(re.findall(r"^(F64_\w+) = np\.float64\(([^)]+)\)", source, re.MULTILINE))
    assert float(declared["F64_MC2_KEV"]) == _MC2_KEV
    assert float(declared["F64_BS_PREFACTOR"]) == _BS_PREFACTOR
    assert float(declared["F64_LN2"]) == _LN2
    assert float(declared["F64_JL_PREFACTOR"]) == 7.85e-4
    assert float(declared["F64_JL_166"]) == 1.166


def test_cuda_stopping_keeps_the_per_element_splice():
    """The device loop must branch per element on its own crossover, not globally."""
    source = _jit_device_source()
    start = source.index("def _dEds_packed(")
    body = source[start : source.index("\n@", start)]

    # per-element branch on that element's own crossover, inside the element loop
    assert "if E_i < L_E_cross[row + i_el]:" in body
    # both branches accumulate separately and are combined with their own prefactor
    assert "joy_luo_total" in body and "bs_total" in body
    assert "-F64_JL_PREFACTOR / E_i * joy_luo_total" in body
    assert "F64_BS_PREFACTOR / beta_sq * bs_total" in body
    # the Berger-Seltzer bracket, with tau^2 (tau + 2) / (2 (I/mc^2)^2)
    assert "tau * tau * (tau + F64_TWO) / (F64_TWO * I_rel * I_rel)" in body
