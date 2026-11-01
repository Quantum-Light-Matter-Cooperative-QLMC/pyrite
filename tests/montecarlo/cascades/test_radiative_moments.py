"""Numerical moment-table anchors, independent host partitions and edge cases."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
from pyrite.montecarlo.transport.radiative_moments import (
    MomentPolicy,
    interpolate_moments,
    prepare_radiative_moments,
)
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture(scope="module")
def tables():
    tables = {
        element: prepare_bremslib_table(
            synthetic_bremslib_arrays(theta_step_deg=10),
            atomic_number=z,
            key=f"synthetic/{element}",
            digest="0" * 64,
        )
        for element, z in [("Si", 14), ("W", 74)]
    }
    # The general SDCS staging grid goes through exp(log(E)); fix synthetic
    # endpoint spellings so nextafter coverage tests have exact 10/200 keV bounds.
    for table in tables.values():
        table.incident_energy_keV[[0, -1]] = [10.0, 200.0]
    return tables


@pytest.mark.parametrize("dielectric", [False, True])
@pytest.mark.parametrize("cutoff", [1000.0, 10000.0])
@pytest.mark.slow
def test_held_out_components_and_closure(tables, dielectric, cutoff):
    from pyrite.materials import MediumSpec
    from pyrite.materials.attenuation import plasma_energy_eV

    compositions = [[("Si", 0.05)], [("W", 0.03), ("Si", 0.02)]]
    prepared = prepare_radiative_moments(
        compositions, tables, 10000, 200000, cutoff, dielectric=dielectric
    )
    assert max(prepared.max_error_fraction) <= 1
    # A second held-out set uses irrational log positions, incident knots,
    # cutoff adjacency and table limits; host partition is a separate evaluator.
    energy = np.unique(
        np.r_[
            10000,
            np.nextafter(10000.0, np.inf),
            20000,
            50000,
            100000,
            200000,
            10000 * 20 ** np.mod(np.arange(1, 103) * np.sqrt(2), 1),
        ]
    )
    policy = MomentPolicy()
    for layer, comp in enumerate(compositions):
        plasma = plasma_energy_eV(MediumSpec("anchor", tuple(comp))) if dielectric else 0.0
        for e in energy:
            expected = np.zeros(3)
            total = 0.0
            for element, density in comp:
                part = build_radiative_partition(
                    tables[element], e, cutoff, plasma_energy_eV=plasma
                )
                full = build_radiative_partition(tables[element], e, e, plasma_energy_eV=plasma)
                expected += (
                    density
                    * 1e16
                    * np.array(
                        [
                            part.soft_stopping_cs_eV_cm2,
                            part.hard_rate_cs_cm2,
                            part.hard_stopping_cs_eV_cm2,
                        ]
                    )
                )
                total += density * 1e16 * full.soft_stopping_cs_eV_cm2
            actual = prepared.evaluate(layer, e)
            assert np.all(actual >= 0)
            assert np.all(
                np.abs(actual - expected) <= np.array(policy.atol) + policy.rtol * expected
            )
            assert abs(actual[0] + actual[2] - total) <= (policy.atol[0] + policy.rtol * total)
        if cutoff == 10000:
            np.testing.assert_array_equal(prepared.evaluate(layer, cutoff)[1:], 0.0)


def test_coverage_and_inputs_are_explicit(tables):
    p = prepare_radiative_moments([[("Si", 0.05)]], tables, 10000, 200000, 1000)
    for e in [0, -1, np.nan, np.inf, np.nextafter(10000.0, 0), np.nextafter(200000.0, np.inf)]:
        with pytest.raises(ValueError, match="coverage"):
            p.evaluate(0, e)
        with pytest.raises(ValueError, match="coverage"):
            interpolate_moments(p.energy_eV[0], p.log_energy[0], p.moments[0], e)
    with pytest.raises(ValueError, match="layer"):
        p.evaluate(-1, 20000)
    with pytest.raises(ValueError, match="outside the BremsLib"):
        prepare_radiative_moments([[("Si", 0.05)]], tables, 9000, 200000, 1000)
    with pytest.raises(ValueError, match="outside the BremsLib"):
        prepare_radiative_moments([[("Si", 0.05)]], tables, np.nextafter(10000.0, 0), 200000, 1000)
    with pytest.raises(ValueError, match="outside the BremsLib"):
        prepare_radiative_moments(
            [[("Si", 0.05)]], tables, 10000, np.nextafter(200000.0, np.inf), 1000
        )
    with pytest.raises(ValueError, match="max_nodes"):
        prepare_radiative_moments(
            [[("Si", 0.05)]], tables, 10000, 200000, 1000, policy=MomentPolicy(max_nodes=2)
        )
    with pytest.raises(ValueError, match="densities"):
        prepare_radiative_moments([[("Si", -0.05)]], tables, 10000, 200000, 1000)
    with pytest.raises(ValueError, match="cutoff"):
        prepare_radiative_moments([[("Si", 0.05)]], tables, 10000, 200000, 10001)
    with pytest.raises(ValueError):
        p.moments[0][0, 0] = 0


def test_identity_covers_layer_plasma_projectile_policy_and_actual_data(tables):
    def build(comp=None, **kwargs):
        return prepare_radiative_moments(
            comp or [[("Si", 0.05)]],
            kwargs.pop("tables", tables),
            10000,
            kwargs.pop("max_energy", 200000),
            kwargs.pop("cutoff", 1000),
            **kwargs,
        )

    base = build()
    assert base.digest == build().digest
    variants = [
        build(dielectric=False),
        build([[("Si", 0.04)]]),
        build(projectile="positron"),
        build(policy=MomentPolicy(rtol=1e-4)),
        build(cutoff=2000),
        build(max_energy=100000),
        build(
            tables={
                **tables,
                "Si": replace(tables["Si"], scaled_sdcs_mb=tables["Si"].scaled_sdcs_mb * 2),
            }
        ),
    ]
    assert len({base.digest, *(v.digest for v in variants)}) == len(variants) + 1


@pytest.mark.parametrize("scale", [0.0, 1e-15])
def test_zero_and_tiny_rates(tables, scale):
    t = replace(tables["Si"], scaled_sdcs_mb=tables["Si"].scaled_sdcs_mb * scale)
    p = prepare_radiative_moments([[("Si", 0.05)]], {"Si": t}, 10000, 200000, 10000)
    for energy in [10000, 10000.0001, 15372, 200000]:
        assert np.all(p.evaluate(0, energy) >= 0)
    if scale == 0:
        np.testing.assert_array_equal(p.moments[0], 0)


@pytest.mark.parametrize("dielectric", [False, True])
def test_constant_sdcs_against_analytic_moments(tables, dielectric):
    """Independent constant-chi primitives, including SI plasma calculation."""
    from scipy.constants import c, e, electron_mass, epsilon_0, hbar

    table = replace(tables["Si"], scaled_sdcs_mb=np.full_like(tables["Si"].scaled_sdcs_mb, 10.0))
    prepared = prepare_radiative_moments(
        [[("Si", 0.05)]], {"Si": table}, 10000, 200000, 1000, dielectric=dielectric
    )
    plasma = hbar * np.sqrt(14 * 0.05e30 * e**2 / (epsilon_0 * electron_mass)) / e
    scale = 0.05e16 * 1e-27 * 14**2 * 10
    policy = MomentPolicy()
    for energy in [10000.0, 13721.0, 37373.0, 113171.0, 200000.0]:
        a = (1 + energy * e / (electron_mass * c**2)) * plasma if dielectric else 0
        if a:
            soft = 1000 - a * np.arctan(1000 / a)
            hard = 0.5 * np.log((energy**2 + a**2) / (1000**2 + a**2))
            first = energy - a * np.arctan(energy / a) - soft
        else:
            soft, hard, first = 1000, np.log(energy / 1000), energy - 1000
        expected = scale * np.array([soft, hard, first])
        actual = prepared.evaluate(0, energy)
        assert np.all(np.abs(actual - expected) <= np.array(policy.atol) + policy.rtol * expected)
