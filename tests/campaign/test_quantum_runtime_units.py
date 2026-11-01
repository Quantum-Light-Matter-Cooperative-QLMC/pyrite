"""Pint normalization at public quantum runtime boundaries (#714)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite import states
from pyrite.planar_source import (
    PlanarStateProjection,
    propagate_planar_source,
    score_planar_source,
)
from pyrite.units import ureg


def test_gaussian_quantity_array_matches_numeric_state():
    numeric = states.gaussian(waist_ang=10.0, center_ang=(0.25, 1.0))
    quantity = states.gaussian(
        waist_ang=1 * ureg.nm,
        center_ang=ureg.Quantity(np.array([25.0, 100.0]), "pm"),
    )
    assert quantity.centre == numeric.centre
    k = np.array([0.0, 0.1])
    np.testing.assert_allclose(quantity.amplitude(k, k), numeric.amplitude(k, k))
    assert all(isinstance(x, float) for x in quantity.centre)


@pytest.mark.parametrize(
    "factory,numeric,quantities",
    [
        (
            states.hermite_gauss,
            dict(m=1, waist_ang=10, center_ang=2),
            dict(m=1, waist_ang=1 * ureg.nm, center_ang=0.2 * ureg.nm),
        ),
        (
            states.laguerre_gauss,
            dict(ell=1, waist_ang=10, center_ang=(1, 2)),
            dict(
                ell=1,
                waist_ang=1 * ureg.nm,
                center_ang=ureg.Quantity(np.array([0.1, 0.2]), "nm"),
            ),
        ),
        (
            states.donut,
            dict(ell=1, waist_ang=10),
            dict(ell=1, waist_ang=1 * ureg.nm),
        ),
        (
            states.bessel_gauss,
            dict(ell=1, waist_ang=10, k_radial_inv_ang=0.1),
            dict(ell=1, waist_ang=1 * ureg.nm, k_radial_inv_ang=1 / ureg.nm),
        ),
        (
            states.airy,
            dict(scale_ang=10, truncation=0.3, center_ang=2),
            dict(scale_ang=1 * ureg.nm, truncation=0.3, center_ang=0.2 * ureg.nm),
        ),
    ],
)
def test_analytic_factories_normalize_identically(factory, numeric, quantities):
    left = factory(**numeric)
    right = factory(**quantities)
    assert right.centre == pytest.approx(left.centre)
    assert right.k_extent == pytest.approx(left.k_extent)
    assert right.k_resolution == pytest.approx(left.k_resolution)


def test_sampled_array_and_translation_are_numeric():
    values = np.ones((8, 8))
    numeric = states.from_array(
        values,
        spacing_ang=(10.0, 20.0),
        origin_ang=(1.0, 2.0),
        kinetic_energy_eV=30_000.0,
    )
    quantity = states.from_array(
        values,
        spacing_ang=ureg.Quantity(np.array([1.0, 2.0]), "nm"),
        origin_ang=(100 * ureg.pm, 0.2 * ureg.nm),
        kinetic_energy_eV=30 * ureg.keV,
    )
    for a, b in zip(quantity.axes, numeric.axes, strict=True):
        np.testing.assert_allclose(a, b)
        assert not hasattr(a, "units")
    assert quantity.kinetic_energy_eV == 30_000.0
    assert not hasattr(quantity.kinetic_energy_eV, "units")
    shift = states.translate(quantity, ureg.Quantity(np.array([0.1, 0.2]), "nm"))
    assert shift.centre == pytest.approx(
        tuple(c + d for c, d in zip(quantity.centre, (1.0, 2.0), strict=True))
    )


def test_aperture_normalizes_angle_and_aberration_coefficients():
    numeric = states.aperture_probe(
        30_000.0,
        semiangle_rad=np.deg2rad(1.0),
        inner_semiangle_rad=0.001,
        aberrations={"C10": -30.0, "phi12": np.pi / 4},
    )
    quantity = states.aperture_probe(
        30 * ureg.keV,
        semiangle_rad=1 * ureg.degree,
        inner_semiangle_rad=1 * ureg.mrad,
        aberrations={"C10": -3 * ureg.nm, "phi12": 45 * ureg.degree},
    )
    assert quantity.momentum == pytest.approx(numeric.momentum)
    assert quantity.k_inner == pytest.approx(numeric.k_inner)
    assert quantity.k_outer == pytest.approx(numeric.k_outer)
    assert quantity.aberrations["C10"] == pytest.approx(-30.0)
    assert quantity.aberrations["phi12"] == pytest.approx(np.pi / 4)
    assert not any(hasattr(v, "units") for v in quantity.aberrations.values())


@pytest.mark.parametrize(
    "factory,kwargs,field",
    [
        (states.gaussian, dict(waist_ang=1 * ureg.second), "waist_ang"),
        (
            states.bessel_gauss,
            dict(ell=1, waist_ang=2, k_radial_inv_ang=1 * ureg.eV),
            "k_radial_inv_ang",
        ),
        (
            states.aperture_probe,
            dict(kinetic_energy_eV=30_000, semiangle_rad=1 * ureg.dimensionless),
            "semiangle_rad",
        ),
        (
            states.aperture_probe,
            dict(
                kinetic_energy_eV=30_000,
                semiangle_rad=0.01,
                aberrations={"phi12": 1 * ureg.dimensionless},
            ),
            "aberrations.phi12",
        ),
    ],
)
def test_runtime_factories_reject_dimension_mismatches(factory, kwargs, field):
    with pytest.raises(ValueError, match=field):
        factory(**kwargs)


def _projection():
    setup = SimpleNamespace(
        region=SimpleNamespace(zone_max_ang=None),
        numerics=SimpleNamespace(step_ang=5.0),
    )
    return PlanarStateProjection(
        "digest",
        "silicon",
        (1, 1, 0),
        30_000.0,
        293.15,
        5.0,
        0.0,
        "averaged",
        setup,
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.zeros(1),
        0.0,
    )


def test_planar_propagation_normalizes_stopping_power_before_solver(monkeypatch):
    import pyrite.planar_source as ps

    observed = {}
    solver = SimpleNamespace(
        propagate=lambda depth, step, **kwargs: observed.update(depth=depth, step=step, **kwargs)
    )
    monkeypatch.setattr(ps, "_solver_for", lambda _: solver)
    projection = _projection()
    propagate_planar_source(
        projection,
        depth_ang=4 * ureg.nm,
        stopping_eV_per_ang=10 * ureg.eV / ureg.nm,
    )
    assert observed["depth"] == 40.0
    assert observed["stopping_eV_per_ang"] == pytest.approx(1.0)
    assert observed["step"] == 5.0
    assert not any(hasattr(v, "units") for v in observed.values())
    with pytest.raises(ValueError, match="propagate_planar_source.depth_ang"):
        propagate_planar_source(projection, depth_ang=1 * ureg.keV)


def test_planar_scoring_normalizes_quantity_grids_and_angles(monkeypatch):
    import pyrite.planar_source as ps
    from pyrite.montecarlo.channeling import emission

    monkeypatch.setattr(ps, "_solver_for", lambda _: object())
    monkeypatch.setattr(emission, "channeling_spectrum", lambda *a, **kw: (a, kw))
    args, kwargs = score_planar_source(
        _projection(),
        depth_ang=4 * ureg.nm,
        photon_energy_eV=ureg.Quantity(np.array([5.0, 6.0]), "keV"),
        theta_rad=ureg.Quantity(np.array([0.0, 1.0]), "mrad"),
        phi_rad=90 * ureg.degree,
        sample_ang=1 * ureg.nm,
        divergence_y_rad=1 * ureg.mrad,
        energy_spread_eV=0.1 * ureg.keV,
        aperture_rad=2 * ureg.mrad,
        layers=[(0 * ureg.nm, 4 * ureg.nm, "silicon")],
        path_limits_ang=ureg.Quantity(np.array([1, 2]), "nm"),
        surface_normal=ureg.Quantity(np.array([0, 0, 1]), "dimensionless"),
    )
    assert args[1] == 40.0
    np.testing.assert_allclose(args[2], [5000.0, 6000.0])
    np.testing.assert_allclose(args[3], [0.0, 0.001])
    assert args[4] == pytest.approx(np.pi / 2)
    assert kwargs["sample_ang"] == 10.0
    assert kwargs["divergence_y_rad"] == pytest.approx(0.001)
    assert kwargs["energy_spread_eV"] == 100.0
    assert kwargs["aperture_rad"] == pytest.approx(0.002)
    np.testing.assert_array_equal(kwargs["path_limits_ang"], [10.0, 20.0])
    assert kwargs["layers"] == ((0.0, 40.0, "silicon"),)
    for value in args[1:]:
        assert not hasattr(value, "units")
    for value in kwargs.values():
        if not isinstance(value, tuple):
            assert not hasattr(value, "units")

    with pytest.raises(ValueError, match="score_planar_source.theta_rad"):
        score_planar_source(
            _projection(),
            depth_ang=40,
            photon_energy_eV=[5000, 6000],
            theta_rad=1 * ureg.dimensionless,
        )
    with pytest.raises(ValueError, match="score_planar_source.photon_energy_eV"):
        score_planar_source(
            _projection(),
            depth_ang=40,
            photon_energy_eV=ureg.Quantity(np.array([1, 2]), "nm"),
            theta_rad=0,
        )


def test_runtime_units_preserve_portable_spec_identity_and_json():
    import json

    from pyrite.quantum import TransverseStateSpec

    bare = TransverseStateSpec.recipe("gaussian", waist_ang=10.0, center_ang=(0.25, 0.0))
    unitful = TransverseStateSpec.recipe(
        "gaussian", waist_ang=1 * ureg.nm, center_ang=(25 * ureg.pm, 0.0)
    )
    assert bare == unitful
    assert bare.digest == unitful.digest
    assert bare.to_dict() == unitful.to_dict()
    assert isinstance(json.loads(unitful.canonical_json()), dict)
    assert unitful.to_runtime().centre == pytest.approx(bare.to_runtime().centre)
