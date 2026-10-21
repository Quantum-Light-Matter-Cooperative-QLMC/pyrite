"""Pinned optical response, passivity, support and source-only TR integration."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.optical import OpticalResponse, gold_rakic_response
from pyrite.montecarlo.runner.transition_emission import case_layer_eps
from pyrite.montecarlo.spectrum.transition_radiation import composition_dielectric


def _response(segments=None, **overrides):
    values = dict(
        medium="sample", segments=((np.array([1.0, 2.0]), np.array([1+0j, 3+2j])),) if segments is None else segments,
        source="synthetic passivity fixture", source_sha256="0" * 64,
        license="CC0-1.0", sample_state="synthetic", temperature_K=300.0,
        dataset_id="synthetic", reference="synthetic fixture", source_units="energy eV; epsilon dimensionless",
    )  # fmt: skip
    values.update(overrides)
    return OpticalResponse(**values)


def _rakic_au(energy):
    # Independent oscillator reference: Rakić et al. (1998), LD model;
    # parameters from the source-linked Polyanskiy Au calculation script.
    # This is a reference only, never a production provider or extrapolator.
    plasma = 9.03
    result = 1 - 0.760 * plasma**2 / (energy * (energy + 0.053j))
    for strength, resonance, damping in (
        (0.024, 0.415, 0.241),
        (0.010, 0.830, 0.345),
        (0.071, 2.969, 0.870),
        (0.601, 4.304, 2.494),
        (4.384, 13.32, 2.214),
    ):
        result += strength * plasma**2 / (resonance**2 - energy**2 - 1j * energy * damping)
    return result


def test_gold_pinned_response_matches_rakic_within_tabulation_error():
    response = gold_rakic_response()
    lower, upper = response.bands_eV[0]
    energy = np.geomspace(lower, upper, 2001)
    eps = response.epsilon(energy)
    reference = _rakic_au(energy)
    # ~0.016 spacing in ln E gives O(h^2) interpolation error; 5-significant
    # digit source ordinates and rounded wavelengths add O(1e-4). 2e-3
    # relative complex error covers both, including the interband curvature.
    assert np.max(np.abs(eps - reference) / np.abs(reference)) < 2e-3
    assert np.all(eps.imag > 0)
    assert eps.dtype == np.complex128
    assert response.license == "CC0-1.0"
    assert response.temperature_K is None
    assert response.orientation == "not applicable (isotropic response)"
    assert response.uncertainty == "unspecified by source"
    assert response.dataset_id == "refractiveindex.info/main/Au/nk/Rakic-LD"
    assert "10.1364/AO.37.005271" in response.reference
    assert "wavelength um" in response.source_units
    assert "fit" in response.sample_state
    assert (
        response.source_sha256 == "a91fca57ee99efc77354908fa11b9c4822e8852394042069caaac7610c52e602"
    )
    assert "c5c2f188e848453def5970e347399d653df2ffc2" in response.source
    # Do not claim the paper's broader validity as support for this record.
    assert 0.199 < lower < 0.201 and 4.999 < upper < 5.001
    assert response.digest == gold_rakic_response().digest


def test_gold_tabulation_convergence_against_independent_oscillators():
    response = gold_rakic_response()
    energy, eps = response.segments[0]
    query = np.sqrt(energy[2:-3] * energy[3:-2])
    reference = _rakic_au(query)
    errors = []
    for stride in (4, 2, 1):
        selected = np.unique(np.r_[np.arange(0, energy.size, stride), energy.size - 1])
        sampled = replace(response, segments=((energy[selected], eps[selected]),))
        errors.append(np.max(np.abs(sampled.epsilon(query) - reference) / np.abs(reference)))
    # Piecewise linear convergence until the source's rounded nk values bind.
    assert errors[1] < errors[0] / 2
    assert errors[2] < errors[1] / 2


def test_linear_complex_interpolation_shape_passivity_and_vacuum_limit():
    response = _response()
    query = np.array([[1.0, 1.25], [1.5, 2.0]])
    np.testing.assert_allclose(response.epsilon(query), [[1 + 0j, 1.5 + 0.5j], [2 + 1j, 3 + 2j]])
    assert response.epsilon(1.5).shape == ()
    assert response.epsilon(1.5) == 2 + 1j
    vacuum = _response(((np.array([1.0, 2.0]), np.ones(2, complex)),))
    np.testing.assert_array_equal(vacuum.epsilon(query), np.ones(query.shape))
    # Zero dissipative loss also permits negative real epsilon (evanescence).
    evanescent = _response(((np.array([1.0, 2.0]), np.full(2, -2 + 0j)),))
    np.testing.assert_array_equal(evanescent.epsilon(query), np.full(query.shape, -2 + 0j))


def test_disjoint_support_never_interpolates_through_a_gap():
    response = _response(
        (
            (np.array([1.0, 2.0]), np.array([1 + 0j, 2 + 1j])),
            (np.array([3.0, 4.0]), np.array([3 + 2j, 4 + 3j])),
        )
    )
    np.testing.assert_array_equal(response.epsilon([2.0, 3.0]), [2 + 1j, 3 + 2j])
    with pytest.raises(ValueError, match="sample.*gap"):
        response.epsilon([1.5, 2.5, 3.5])


@pytest.mark.parametrize("energy", [[0], [-1], [np.nan], [np.inf], [], [0.1], [6.0]])
def test_gold_refuses_unsupported_energies(energy):
    with pytest.raises(ValueError, match="Au"):
        gold_rakic_response().epsilon(energy)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"symmetry": "tensor"}, "tensor"),
        ({"convention": "exp(+i omega t)"}, "convention"),
        ({"temperature_K": -1}, "temperature"),
        ({"temperature_K": np.nan}, "temperature"),
        ({"license": ""}, "license"),
        ({"dataset_id": ""}, "dataset identifier"),
        ({"reference": ""}, "reference"),
        ({"source_units": ""}, "source units"),
        ({"orientation": ""}, "orientation"),
        ({"uncertainty": ""}, "uncertainty"),
        ({"source_sha256": "not a hash"}, "SHA-256"),
        ({"segments": ()}, "segment"),
        ({"segments": (([1, 2], [1 - 1j, 2 + 0j]),)}, "passive"),
        ({"segments": (([1, 2], [1, np.inf]),)}, "finite"),
        ({"segments": (([1, 1], [1, 2]),)}, "ordered"),
        ({"segments": (([0, 1], [1, 2]),)}, "positive"),
        ({"segments": (([1, 2], np.ones((2, 3, 3))),)}, "scalar"),
        ({"segments": (([1, 3], [1, 2]), ([2, 4], [1, 2]))}, "disjoint"),
    ],
)
def test_response_refuses_unsupported_input(overrides, message):
    with pytest.raises(ValueError, match=message):
        _response(**overrides)


def test_response_identity_and_immutable_data():
    response = _response()
    for nodes in response.segments[0]:
        with pytest.raises(ValueError):
            nodes.setflags(write=True)
    assert replace(response, temperature_K=301).digest != response.digest
    assert replace(response, sample_state="different film").digest != response.digest
    assert replace(response, source_sha256="1" * 64).digest != response.digest
    assert replace(response, uncertainty="1% source estimate").digest != response.digest
    assert replace(response, dataset_id="another sample").digest != response.digest
    assert (
        replace(response, segments=((np.array([1.0, 2.0]), np.array([1.0, 2.0])),)).digest
        != response.digest
    )


def test_explicit_layers_do_not_fall_back_to_atomic_tables(monkeypatch):
    from pyrite.montecarlo.runner import transition_emission

    def forbidden(*args, **kwargs):
        pytest.fail("explicit optical response must never read atomic tables")

    monkeypatch.setattr(transition_emission, "composition_dielectric", forbidden)
    case = {"thickness_ang": 10.0, "composition": [("Au", 0.059)]}
    response = gold_rakic_response()
    np.testing.assert_array_equal(
        case_layer_eps(case, [1.0, 2.0], layer_responses=(response,))[0],
        response.epsilon([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="unsupported"):
        case_layer_eps(case, [0.01], layer_responses=(response,))
    with pytest.raises(ValueError, match="one optical response per layer"):
        case_layer_eps(case, [1.0], layer_responses=())
    with pytest.raises(ValueError, match="explicit scalar"):
        case_layer_eps(case, [1.0], layer_responses=(None,))


def test_each_layer_uses_its_own_explicit_provider():
    first, second = (
        _response(),
        _response(sample_state="second", segments=((np.array([1.0, 2.0]), np.array([5.0, 7.0])),)),
    )
    case = {"abs_layers": [(0, 1, [("Au", 0.059)]), (1, 2, [("C", 0.1)])]}
    result = case_layer_eps(case, [1.5], layer_responses=(first, second))
    np.testing.assert_array_equal(result, [[2 + 1j], [6 + 0j]])


def test_default_xray_response_is_unchanged_and_optical_needs_explicit_provider():
    case = {"thickness_ang": 10.0, "composition": [("Au", 0.059)]}
    energy = np.array([100.0, 500.0, 2000.0])
    np.testing.assert_array_equal(
        case_layer_eps(case, energy)[0], composition_dielectric(case["composition"], energy)
    )
    with pytest.raises(ValueError, match=r"transition_radiation/layer\[0\].*X-ray floor"):
        case_layer_eps(case, [2.0])
