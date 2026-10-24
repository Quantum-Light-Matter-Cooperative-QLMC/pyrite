"""Portable transverse source state contract (#680); does not run transport."""

from dataclasses import FrozenInstanceError, replace
import json
import pickle

import numpy as np
import pytest

import pyrite as pr
from pyrite import states


def test_public_lazy_state_entrypoint():
    assert pr.TransverseStateSpec is states.TransverseStateSpec
    assert callable(states.laguerre_gauss)  # runtime #508 API remains available


@pytest.mark.parametrize("kind,kwargs,dimension", [
    ("gaussian", dict(waist_ang=80), 2),
    ("gaussian", dict(waist_ang=80, dimension=1), 1),
    ("hermite_gauss", dict(m=1, waist_ang=1), 1),
    ("hermite_gauss", dict(m=1, n=2, waist_ang=1), 2),
    ("laguerre_gauss", dict(ell=-2, p=1, waist_ang=80), 2),
    ("bessel_gauss", dict(ell=1, k_radial_inv_ang=.2, waist_ang=10), 2),
    ("airy", dict(scale_ang=5, truncation=.1), 1),
    ("aperture_probe", dict(kinetic_energy_eV=30_000, semiangle_rad=.001,
                             aberrations={"C10": -4.0, "phi12": .1}), 2),
])
def test_analytic_state_json_and_runtime(kind, kwargs, dimension):
    spec = pr.TransverseStateSpec.recipe(kind, **kwargs)
    assert spec.dimension == dimension
    payload = json.loads(json.dumps(spec.to_dict()))
    assert pr.TransverseStateSpec.from_dict(payload) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert len(spec.digest) == 64
    runtime = spec.to_runtime()
    assert runtime.dimension == dimension


def test_superposition_preserves_phase_and_mixture_is_not_superposition():
    positive = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=1, waist_ang=8)
    negative = pr.TransverseStateSpec.recipe("laguerre_gauss", ell=-1, waist_ang=8)
    coherent = pr.TransverseStateSpec.superpose([(1 + 0j, positive), (0 + 1j, negative)])
    incoherent = pr.TransverseStateSpec.mixture([(3.0, positive), (1.0, negative)])
    serialized = json.loads(json.dumps(coherent.to_dict()))
    assert serialized["terms"][1]["weight"] == [0.0, 1.0]
    assert pr.TransverseStateSpec.from_dict(serialized) == coherent
    assert tuple(w for w, _ in incoherent.terms) == (0.75, 0.25)
    assert incoherent.digest != coherent.digest
    assert incoherent.to_runtime().dimension == 2
    assert pr.TransverseStateSpec.from_dict(incoherent.to_dict()) == incoherent


def test_translation_is_frozen_and_independent_of_caller_mutations():
    center = [2.0, 3.0]
    original = pr.TransverseStateSpec.recipe("gaussian", waist_ang=5, center_ang=center)
    offset = [0.5, -1.0]
    shifted = original.translated(offset)
    center[0] = 99
    offset[0] = 99
    assert dict(original.parameters)["center_ang"] == (2, 3)
    assert dict(shifted.parameters)["offset_ang"] == (.5, -1)
    assert shifted.to_runtime().centre == (2.5, 2.0)
    with pytest.raises(FrozenInstanceError):
        shifted.kind = "gaussian"


def test_sampled_content_checks_and_location_independent_identity(tmp_path):
    array = np.array([[1 + 1j, 2], [1 - 1j, 0]], dtype=np.complex128)
    first = tmp_path / "a.npy"
    second = tmp_path / "b.npy"
    np.save(first, array)
    np.save(second, array)
    a = pr.TransverseStateSpec.from_npy(first, spacing_ang=(.1, .2))
    b = pr.TransverseStateSpec.from_npy(second, spacing_ang=(.1, .2))
    assert a.digest == b.digest
    assert a.to_dict()["parameters"]["path"] != b.to_dict()["parameters"]["path"]
    result = a.to_runtime()
    assert result.dimension == 2
    assert pr.TransverseStateSpec.from_dict(json.loads(json.dumps(a.to_dict()))) == a

    relative = replace(a, parameters=tuple((k, "a.npy" if k == "path" else v)
                                          for k, v in a.parameters))
    with pytest.raises(ValueError, match="base_dir"):
        relative.to_runtime()
    assert relative.to_runtime(base_dir=tmp_path).dimension == 2
    np.save(first, array * 2)
    with pytest.raises(ValueError, match="content differs"):
        a.to_runtime()


def test_configuration_refuses_invalid_inputs(tmp_path):
    recipe = pr.TransverseStateSpec.recipe
    with pytest.raises(ValueError, match="unknown parameters"):
        recipe("gaussian", waist_ang=3, arbitrary="code")
    with pytest.raises(ValueError, match="positive"):
        recipe("gaussian", waist_ang=float("nan"))
    with pytest.raises(ValueError, match="dimension"):
        recipe("gaussian", waist_ang=1, dimension=3)
    with pytest.raises(ValueError, match="aberration"):
        recipe("aperture_probe", kinetic_energy_eV=1000, semiangle_rad=.01,
               aberrations={"evil": 3})
    with pytest.raises(ValueError, match="schema"):
        pr.TransverseStateSpec.from_dict({"schema": "untrusted", "kind": "gaussian"})
    s = recipe("gaussian", waist_ang=1)
    with pytest.raises(ValueError, match="nonnegative"):
        pr.TransverseStateSpec.mixture([(-1, s)])
    with pytest.raises(ValueError, match="finite"):
        pr.TransverseStateSpec.superpose([(complex(float("nan"), 0), s)])
    with pytest.raises(ValueError, match="same dimension"):
        pr.TransverseStateSpec.mixture([(1, s), (1, recipe("gaussian", waist_ang=1, dimension=1))])
    with pytest.raises(ValueError, match="phase"):
        # There is no arbitrary phase plate field in the portable schema.
        recipe("aperture_probe", kinetic_energy_eV=1000, semiangle_rad=.01,
               phase_plate=lambda x, y: x)
    np.save(tmp_path / "bad.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="nonzero"):
        pr.TransverseStateSpec.from_npy(tmp_path / "bad.npy", spacing_ang=.1)


def test_beam_stores_spec_but_refuses_unimplemented_production_injection():
    spec = pr.TransverseStateSpec.recipe("gaussian", waist_ang=10)
    beam = pr.Beam(energy_keV=30, transverse=pr.Spot(),
                   longitudinal=pr.PointBunch(), state=spec)
    assert beam.state == spec
    with pytest.raises(ValueError, match="not yet.*transport"):
        beam.to_spec()
    with pytest.raises(TypeError, match="TransverseStateSpec"):
        pr.Beam(energy_keV=30, transverse=pr.Spot(),
                longitudinal=pr.PointBunch(), state={"kind": "gaussian"})
    # No-state legacy lowering retains its exact field set and behavior.
    legacy = replace(beam, state=None).to_spec()
    assert legacy.energy_keV == 30
