"""Transition radiation as a separate run-case component (#103).

Validation: tr-crossing-superposition, tr-coherent-bunch
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite._spectral_components import cxr_spectrum, incident_spectrum
from pyrite.materials import CATALOG
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.runner.transition_emission import case_layer_eps, case_transition_spectrum
from pyrite.montecarlo.spectrum.transition_radiation import composition_dielectric
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    _take,
    boundary_crossings,
    crossing_transition_radiation,
)

_GRID = (100.0, 2000.0, 20.0)
_HISTORY = {"model": "planar-stack", "coherence": "history"}
_DROP = object()


def _case(**overrides):
    fields = dict(
        name="hopg tr", crystal="hopg", composition=CATALOG.crystal("hopg").composition,
        hkl_list=[(0, 0, 2)], B_ang2=0.8, E0_keV=30.0, thickness_ang=1e4,
        crystal_width_mm=None, crystal_height_mm=None, beam_fwhm_mm=None, surface_hkl=None,
        E_grid=_GRID, E_grid_line=_GRID, E_grid_brem=(0.0, 500.0, 50.0),
        theta_obs_rad=0.3, dtheta_obs_rad=np.deg2rad(2.0), tilt_deg=0.0, tilt_azim_deg=0.0,
        domega_sr=1e-4, beam_uvw=(0, 0, 1), mosaic_fwhm_rad=None, mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1, abs_layers=None, layer_radiators=None, brem_file=None, Ne=24,
        Ne_brem=16, seed=42, spec_chunk=None, brem_chunk=None, energy_model="midpoint",
        elastic_model="elsepa", transition_radiation=_HISTORY,
    )  # fmt: skip
    fields.update(overrides)
    return Case(**{key: value for key, value in fields.items() if value is not _DROP})


@pytest.fixture(scope="module")
def transported():
    case = _case()
    return case, runner._transport_case(case)


def test_spectrum_phase_scores_the_crossings_of_the_continuum_population(transported):
    case, tp = transported
    out = runner._spectrum_case(case, tp)
    energy = np.asarray(out["E_grid"])
    crossings = boundary_crossings(tp["segs"], [(0.0, 1e4, case.composition)])
    mine = _take(crossings, np.flatnonzero(crossings.electron_id < case.Ne_brem))
    assert np.unique(mine.electron_id).size <= case.Ne_brem < np.unique(crossings.electron_id).size
    expected = crossing_transition_radiation(
        mine, energy, tp["n_hat"], [composition_dielectric(case.composition, energy)],
        n_electrons=case.Ne_brem,
    )  # fmt: skip
    np.testing.assert_array_equal(out["spec_transition"], expected)
    assert np.all(expected > 0.0) and expected[0] > expected[-1]
    # Additive with the other line-grid components; deselectable.
    total = cxr_spectrum(out)
    rounding = 1e-12 * np.max(np.abs(total))
    np.testing.assert_allclose(
        total - cxr_spectrum(out, transition=False), expected, rtol=0.0, atol=rounding
    )
    np.testing.assert_allclose(
        incident_spectrum(out) - incident_spectrum(out, transition=False), expected,
        rtol=0.0, atol=1e-12 * np.max(incident_spectrum(out)),
    )  # fmt: skip


def test_directions_stack_and_grazing_is_dark(transported):
    case, tp = transported
    n = np.array([[np.sin(0.3), 0.0, np.cos(0.3)], [0.0, np.sin(2.5), np.cos(2.5)]])
    directional = runner._directional_outputs(case, tp, n)
    stacked = directional["spec_transition_by_direction"]
    assert stacked.shape == (2, directional["E_grid"].size)
    for k, direction in enumerate(n):
        single = case_transition_spectrum(
            case, dict(tp, n_hat=direction, E_grid=directional["E_grid"])
        )
        np.testing.assert_array_equal(stacked[k], single)
    grazing = case_transition_spectrum(case, dict(tp, n_hat=np.array([1.0, 0.0, 0.0])))
    assert np.all(grazing == 0.0)


def test_bunch_coherence_uses_the_physical_population(transported):
    case, tp = transported
    # A bunch of exactly the 16 sampled electrons: the estimate is P / M, nonnegative.
    charge_pc = case.Ne_brem * elementary_charge / 1e-12
    bunch = replace(
        case,
        transition_radiation={"model": "planar-stack", "coherence": "bunch"},
        bunch_charge_pc=charge_pc,
    )
    ctr = case_transition_spectrum(bunch, tp)
    energy = np.asarray(tp["E_grid"])
    crossings = boundary_crossings(tp["segs"], [(0.0, 1e4, case.composition)])
    mine = _take(crossings, np.flatnonzero(crossings.electron_id < case.Ne_brem))
    expected = crossing_transition_radiation(
        mine, energy, tp["n_hat"], [composition_dielectric(case.composition, energy)],
        n_electrons=case.Ne_brem, coherence="bunch", physical_electrons=None,
    )  # fmt: skip
    np.testing.assert_allclose(ctr, expected, rtol=1e-12)
    assert np.all(ctr >= 0.0)


def test_case_record_validation():
    with pytest.raises(ValueError, match="energy_model='midpoint'"):
        _case(energy_model=_DROP)  # the runner default is frozen
    with pytest.raises(ValueError, match="record with keys model, coherence"):
        _case(transition_radiation={"model": "planar-stack"})
    with pytest.raises(ValueError, match="coherence must be"):
        _case(transition_radiation={"model": "planar-stack", "coherence": "crossing"})
    with pytest.raises(ValueError, match="grooved entrance face"):
        _case(groove_spacing_ang=5e3)
    with pytest.raises(ValueError, match="Chantler dielectric data"):
        case_layer_eps(_case(), np.array([1e-3]))


def test_source_only_optical_override_preserves_vacuum_limit_and_refuses_gaps(transported):
    from pyrite.materials.optical import gold_rakic_response

    case, transport = transported
    response = gold_rakic_response()
    # Fixed trajectories exercise only the response/scorer seam, not optical
    # transport. Matched vacuum must remain dark on precisely the same rows.
    optical_transport = dict(transport, E_grid=np.geomspace(0.3, 4.0, 21))
    gold = case_transition_spectrum(case, optical_transport, xp=np, layer_responses=(response,))
    assert np.all(np.isfinite(gold)) and np.max(gold) > 0
    energy, eps = response.segments[0]
    vacuum = replace(response, medium="vacuum fixture", segments=((energy, np.ones_like(eps)),))
    dark = case_transition_spectrum(case, optical_transport, xp=np, layer_responses=(vacuum,))
    np.testing.assert_array_equal(dark, np.zeros(21))
    # Invalid optics must fail even on a geometrically dark direction.
    invalid = dict(transport, E_grid=np.array([0.01]), n_hat=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match=r"transition_radiation/layer\[0\].*Au.*unsupported"):
        case_transition_spectrum(case, invalid, xp=np, layer_responses=(response,))
