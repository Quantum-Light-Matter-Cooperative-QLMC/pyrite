"""Automatic case-local line-grid resolution: bandwidth, precedence, identity.

Issue #101. These tests pin the contract that a valid material at a supported
beam energy runs without `pyrite material energy-grid derive` and without an
installed exact-energy row, and that every result-affecting policy value moves
case identity.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._energy_grid_encoding import decode_energy_grid
from pyrite._line_grid_policy import (
    AUTOMATIC_BANDWIDTH_POLICY,
    DEFAULT_RTOL,
    OBSERVABLE_CLASSES,
    LineGridToleranceError,
    coordinate_cache_key,
    kinematic_line_stop_eV,
    line_start_eV,
    resolve_line_grid_policy,
    resolved_coordinates,
)
from pyrite.campaign.config import material_sweep
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.materials.crystal import (
    HBARC_EV_ANG,
    M_E_EV,
    beta_from_Ee,
)

_ENV_NAMES = (
    "PYRITE_ENERGY_GRID_RTOL",
    "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE",
    "PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS",
    "PYRITE_ENERGY_GRID_MAX_SPACING_EV",
    "PYRITE_ENERGY_GRID_ULPS",
    "PYRITE_ENERGY_GRID_MAX_POINTS",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _uncovered_energy_sweep(energy_keV=77.0):
    """A hopg sweep at a beam energy no stored ``line_by_energy`` row covers."""
    sweep = material_sweep("hopg")
    assert energy_keV not in {float(e) for e in sweep.detector.energy_bins.line_by_energy}
    return replace(sweep, beam=replace(sweep.beam, energy_keV=energy_keV))


def _stored_energy_sweep():
    """An explicit stored-row fixture independent of bundled catalog defaults."""
    sweep = material_sweep("hopg")
    stored = np.linspace(10.0, 2600.0, 864)
    return (
        replace(
            sweep,
            beam=replace(sweep.beam, energy_keV=30.0),
            detector=replace(
                sweep.detector,
                energy_bins=replace(
                    sweep.detector.energy_bins,
                    line_by_energy={30.0: stored},
                ),
            ),
        ),
        stored,
    )


# ---- physics: the kinematic bandwidth bound ---------------------------------
def test_leaf_constants_match_the_materials_package():
    """The leaf restates hbar*c and m_e c^2 to stay dependency-free; they must
    not drift from the values the line kernels actually use."""
    from pyrite import _line_grid_policy as policy

    assert policy._HBARC_EV_ANG == HBARC_EV_ANG
    assert policy._ELECTRON_REST_EV == M_E_EV


def test_kinematic_stop_bounds_the_resonance_over_random_directions():
    """Validation: line-grid-kinematic-bandwidth.

    The bound must hold for EVERY emission/observation direction pair, not just
    the nominal geometry: multiple scattering may point an electron anywhere.
    """
    rng = np.random.default_rng(20261101)
    g_mag = 3.745
    for energy_keV in (30.0, 100.0, 300.0):
        stop = kinematic_line_stop_eV([g_mag], energy_keV, round_to_eV=1e-9)
        beta = beta_from_Ee(energy_keV * 1.0e3)
        directions = rng.normal(size=(4000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        g_vec = np.array([0.0, 0.0, g_mag])
        for v_hat, n_hat in zip(directions[::2], directions[1::2], strict=True):
            denominator = 1.0 - beta * float(v_hat @ n_hat)
            if denominator <= 0.0:
                continue
            resonance = HBARC_EV_ANG * beta * float(v_hat @ g_vec) / denominator
            assert resonance <= stop * (1.0 + 1e-12)


def test_kinematic_stop_scales_linearly_with_g_and_vanishes_with_beta():
    single = kinematic_line_stop_eV([1.0], 100.0, round_to_eV=1e-9)
    doubled = kinematic_line_stop_eV([2.0], 100.0, round_to_eV=1e-9)
    assert doubled == pytest.approx(2.0 * single)
    assert kinematic_line_stop_eV([1.0], 1e-6, round_to_eV=1e-12) < single * 1e-3


def test_kinematic_stop_refuses_empty_or_nonphysical_inputs():
    with pytest.raises(ValueError, match="at least one positive"):
        kinematic_line_stop_eV([], 100.0)
    with pytest.raises(ValueError, match="finite and positive"):
        kinematic_line_stop_eV([1.0], -1.0)


def test_line_start_convention_matches_the_catalog():
    assert line_start_eV(60.0) == 10.0
    assert line_start_eV(60.1) == 50.0


# ---- precedence -------------------------------------------------------------
def test_built_in_defaults_are_per_observable():
    policy = resolve_line_grid_policy(start_eV=10.0, stop_eV=1000.0)
    assert dict(policy.rtol) == dict(DEFAULT_RTOL)
    assert len(set(dict(policy.rtol).values())) == len(OBSERVABLE_CLASSES)
    assert dict(policy.sources)["rtol.intrinsic_source"] == "built-in default"


def test_global_environment_rtol_is_a_fallback_not_an_override(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_RTOL", "0.05")
    monkeypatch.setenv("PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE", "0.002")
    policy = resolve_line_grid_policy(start_eV=10.0, stop_eV=1000.0)
    rtol, sources = dict(policy.rtol), dict(policy.sources)
    assert rtol["intrinsic_source"] == 0.002
    assert sources["rtol.intrinsic_source"] == "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE"
    assert rtol["detected_counts"] == 0.05
    assert sources["rtol.detected_counts"] == "PYRITE_ENERGY_GRID_RTOL"


def test_per_call_outranks_environment_which_outranks_stored(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_MAX_SPACING_EV", "2.0")
    policy = resolve_line_grid_policy(
        start_eV=10.0,
        stop_eV=1000.0,
        stored={"max_spacing_eV": 9.0, "rtol": {"detected_counts": 0.02}},
    )
    sources = dict(policy.sources)
    assert policy.max_spacing_eV == 2.0
    assert sources["max_spacing_eV"] == "PYRITE_ENERGY_GRID_MAX_SPACING_EV"
    assert dict(policy.rtol)["detected_counts"] == 0.02
    assert sources["rtol.detected_counts"] == "stored configuration"

    per_call = resolve_line_grid_policy(
        start_eV=10.0,
        stop_eV=1000.0,
        per_call={"max_spacing_eV": 0.5},
        stored={"max_spacing_eV": 9.0},
    )
    assert per_call.max_spacing_eV == 0.5
    assert dict(per_call.sources)["max_spacing_eV"] == "per-call"


def test_unknown_observable_class_is_refused():
    with pytest.raises(ValueError, match="unknown observable classes"):
        resolve_line_grid_policy(start_eV=10.0, stop_eV=1000.0, per_call={"rtol": {"bogus": 0.1}})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PYRITE_ENERGY_GRID_RTOL", "not-a-number"),
        ("PYRITE_ENERGY_GRID_RTOL", "1.5"),
        ("PYRITE_ENERGY_GRID_MAX_SPACING_EV", "0"),
        ("PYRITE_ENERGY_GRID_MAX_POINTS", "1"),
    ],
)
def test_malformed_environment_values_are_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolve_line_grid_policy(start_eV=10.0, stop_eV=1000.0)


# ---- no derivation prerequisite --------------------------------------------
def test_supported_energy_without_a_stored_row_builds_cases():
    case = build_cases(_uncovered_energy_sweep(), n_electrons=2, n_electrons_brem=2)[0]
    payload = case["line_grid_policy"]
    assert payload["bandwidth"]["policy"] == AUTOMATIC_BANDWIDTH_POLICY
    grid = decode_energy_grid(case["E_grid_line"])
    assert grid[0] == payload["bandwidth"]["start_eV"]
    assert grid[-1] == payload["bandwidth"]["stop_eV"]
    # The case-build grid is the COARSEST admissible one; the runner refines it.
    assert float(np.diff(grid).max()) <= payload["resolution"]["max_spacing_eV"] + 1e-9


def test_stored_row_still_wins_over_automatic_resolution():
    sweep, stored = _stored_energy_sweep()
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    assert case.get("line_grid_policy") is None
    np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), stored)


def test_environment_policy_outranks_a_stored_row(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_MAX_SPACING_EV", "1.0")
    sweep, _ = _stored_energy_sweep()
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    payload = case["line_grid_policy"]
    assert payload["resolution"]["max_spacing_eV"] == 1.0
    assert payload["sources"]["max_spacing_eV"] == "PYRITE_ENERGY_GRID_MAX_SPACING_EV"


def test_explicit_fixed_grid_outranks_environment_policy(monkeypatch):
    monkeypatch.setenv("PYRITE_ENERGY_GRID_MAX_SPACING_EV", "1.0")
    sweep = material_sweep("hopg")
    explicit = np.linspace(100.0, 900.0, 9)
    sweep = replace(
        sweep,
        beam=replace(sweep.beam, energy_keV=30.0),
        detector=replace(
            sweep.detector,
            energy_bins=replace(sweep.detector.energy_bins, line=explicit),
        ),
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    assert case.get("line_grid_policy") is None
    np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), explicit)


def test_building_a_case_does_not_mutate_the_catalog_or_profile():
    from pyrite.materials import CATALOG

    before = dict(CATALOG.material("hopg").scan.E_grid_line_by_energy)
    build_cases(_uncovered_energy_sweep(), n_electrons=2, n_electrons_brem=2)
    after = dict(CATALOG.material("hopg").scan.E_grid_line_by_energy)
    assert set(before) == set(after)
    assert all(np.array_equal(before[key], after[key]) for key in before)


# ---- identity ---------------------------------------------------------------
def _automatic_identity(**environment):
    import os

    saved = {name: os.environ.get(name) for name in _ENV_NAMES}
    try:
        for name in _ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(environment)
        case = build_cases(_uncovered_energy_sweep(), n_electrons=2, n_electrons_brem=2)[0]
        return case_content_key(case)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_changing_a_result_affecting_policy_changes_case_identity():
    baseline = _automatic_identity()
    assert baseline == _automatic_identity()
    tighter = _automatic_identity(PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE="0.0001")
    coarser = _automatic_identity(PYRITE_ENERGY_GRID_MAX_SPACING_EV="1.0")
    stricter_ulps = _automatic_identity(PYRITE_ENERGY_GRID_ULPS="32")
    smaller_budget = _automatic_identity(PYRITE_ENERGY_GRID_MAX_POINTS="50000")
    assert len({baseline, tighter, coarser, stricter_ulps, smaller_budget}) == 5


def test_changing_the_configuration_source_alone_changes_identity():
    """The same number from a different precedence layer is recorded, and the
    record is identity-bearing: a precedence regression must not hide behind an
    unchanged digest."""
    from_environment = _automatic_identity(PYRITE_ENERGY_GRID_MAX_SPACING_EV="3.0")
    assert from_environment != _automatic_identity()


def test_changing_resolved_coordinates_changes_case_identity():
    """Resolved coordinates for a stored/explicit grid sit in the case payload,
    so identity follows them directly."""
    sweep = material_sweep("hopg")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=30.0))
    base = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    shifted = replace(
        sweep,
        detector=replace(
            sweep.detector,
            energy_bins=replace(sweep.detector.energy_bins, line=np.linspace(10.0, 2600.0, 865)),
        ),
    )
    other = build_cases(shifted, n_electrons=2, n_electrons_brem=2)[0]
    assert case_content_key(base) != case_content_key(other)


def test_case_payload_without_automatic_resolution_is_unchanged():
    """Absent-by-default: an explicit or stored grid keeps the historical case
    payload, so existing checkpoints stay addressable."""
    sweep, _ = _stored_energy_sweep()
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=2)[0]
    assert "line_grid_policy" not in case.to_dict()


# ---- refusal instead of silent coarsening -----------------------------------
def _payload(*, start_eV=10.0, stop_eV=10_000.0, **resolution):
    base = resolve_line_grid_policy(start_eV=start_eV, stop_eV=stop_eV).payload()
    base["resolution"].update(resolution)
    return base


def test_point_budget_shortfall_reports_the_unmet_tolerance_and_a_correction():
    with pytest.raises(LineGridToleranceError) as caught:
        resolved_coordinates(_payload(max_points=100), 0.01)
    message = str(caught.value)
    assert "rtol=0.001" in message
    assert "'intrinsic_source'" in message
    assert "PYRITE_ENERGY_GRID_MAX_POINTS" in message
    assert "not coarsened automatically" in message


def test_backend_precision_shortfall_reports_the_unmet_tolerance():
    with pytest.raises(LineGridToleranceError) as caught:
        # A narrow, high-energy window: the point budget is satisfied, but one
        # float32 ULP at 10 keV is ~1e-3 eV, so 1e-5 eV spacing is unreachable.
        resolved_coordinates(
            _payload(start_eV=10_000.0, stop_eV=10_001.0, max_spacing_eV=1.0e-5),
            1.0e-5,
            dtype=np.float32,
        )
    message = str(caught.value)
    assert "rtol=0.001" in message
    assert "PYRITE_FP64=1" in message


def test_resolution_never_exceeds_the_declared_maximum_spacing():
    grid, record = resolved_coordinates(_payload(), 12.0)
    assert record["target_spacing_eV"] == 3.0
    assert float(np.diff(grid).max()) <= 3.0 + 1e-9
    assert record["governing_observable"] == "intrinsic_source"


# ---- content-addressed cache ------------------------------------------------
def test_cache_key_depends_on_policy_and_on_resolution_inputs():
    payload = _payload()
    inputs = {"E0_keV": 77.0, "seed": 1}
    key = coordinate_cache_key(payload, inputs)
    assert key == coordinate_cache_key(payload, dict(inputs))
    assert key != coordinate_cache_key(payload, {**inputs, "seed": 2})
    assert key != coordinate_cache_key(_payload(max_points=99), inputs)
