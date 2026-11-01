"""Runner/scorer dielectric coupling contract, issue #439.

Checks are fast and use synthetic tables; no vendor library or remote box.
"""

import numpy as np
import pytest

from pyrite.materials import MediumSpec
from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.montecarlo.spectrum import brem_events
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport.hard_radiative import pack_radiative_layer_tables
from tests.helpers.bremslib import synthetic_bremslib_arrays


def test_multilayer_packing_uses_full_composition_not_element_plasma():
    si = prepare_bremslib_table(
        synthetic_bremslib_arrays(t1_MeV=np.array([10.0, 30.0, 50.0, 100.0])),
        atomic_number=14,
        key="synthetic/Si",
        digest="0" * 64,
    )
    o = prepare_bremslib_table(
        synthetic_bremslib_arrays(t1_MeV=np.array([10.0, 30.0, 50.0, 100.0])),
        atomic_number=8,
        key="synthetic/O",
        digest="1" * 64,
    )
    compositions = [[("Si", 0.05)], [("Si", 0.025), ("O", 0.05)]]
    packed = pack_radiative_layer_tables(
        compositions, {"Si": si, "O": o}, 10e6, 100e6, dielectric=True
    )
    plasma = packed[-1]
    assert plasma.shape == (2,)
    for j, comp in enumerate(compositions):
        assert plasma[j] == pytest.approx(plasma_energy_eV(MediumSpec(f"layer{j}", tuple(comp))))
    assert not np.isclose(plasma[0], plasma[1])
    off = pack_radiative_layer_tables(compositions, {"Si": si, "O": o}, 10e6, 100e6)
    assert np.all(off[-1] == 0)


def test_packed_multilayer_rates_match_host_partition():
    from pyrite.montecarlo.transport._jit_radiative import radiative_layer_moments_scalar
    from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition

    t = prepare_bremslib_table(
        synthetic_bremslib_arrays(t1_MeV=np.array([10.0, 30.0, 50.0, 100.0])),
        atomic_number=14,
        key="synthetic/Si",
        digest="2" * 64,
    )
    compositions = [[("Si", 0.05)], [("Si", 0.03)]]
    packed = pack_radiative_layer_tables(compositions, {"Si": t}, 10e6, 100e6, dielectric=True)
    for layer in range(2):
        rates = np.zeros(1)
        soft, rate = radiative_layer_moments_scalar(
            *packed[:8], layer, 50e6, 1000.0, rates, packed[-1]
        )
        expected = build_radiative_partition(t, 50e6, 1000.0, plasma_energy_eV=packed[-1][layer])
        factor = compositions[layer][0][1] * 1e24 * 1e-8
        assert soft == pytest.approx(factor * expected.soft_stopping_cs_eV_cm2, rel=3e-6)
        assert rate == pytest.approx(factor * expected.hard_rate_cs_cm2, rel=3e-6)
        assert rates[0] == pytest.approx(rate)


def test_coupled_scorer_honors_transport_plasma_and_rejects_mismatch(monkeypatch):
    comp = [("Si", 0.05)]
    plasma = plasma_energy_eV(MediumSpec("layer", tuple(comp)))
    metadata = {
        "model": "bremslib-soft-hard",
        "cutoff_eV": 1000.0,
        "bremslib_tables": (),
        "dielectric_model": "ter-mikaelian-v1",
        "plasma_energy_eV_by_layer": (plasma,),
    }
    captured = []

    def fake_mc_brem(segments, grid, **kwargs):
        captured.append(kwargs["dielectric_plasma_eV"])
        return np.ones_like(grid)

    monkeypatch.setattr(brem_events, "mc_brem_spectrum", fake_mc_brem)
    grid = np.array([100.0, 1000.0])
    args = dict(cutoff_eV=1000.0, bremslib_tables={}, composition=comp)
    output = brem_events.mc_coupled_brem_spectrum({"radiative": metadata}, grid, **args)
    np.testing.assert_allclose(output, 1)
    assert captured[0] == pytest.approx(plasma)

    with pytest.raises(ValueError, match="scoring material does not match"):
        brem_events.mc_coupled_brem_spectrum(
            {"radiative": metadata}, grid, **{**args, "composition": [("Si", 0.07)]}
        )
    assert len(captured) == 1

    old = {
        "radiative": {
            key: value
            for key, value in metadata.items()
            if key not in {"dielectric_model", "plasma_energy_eV_by_layer"}
        }
    }
    brem_events.mc_coupled_brem_spectrum(old, grid, **args)
    assert captured[-1] == 0.0


def test_unknown_dielectric_model_refused():
    with pytest.raises(ValueError, match="unknown dielectric"):
        brem_events._scoring_plasma_eV(
            {"radiative": {"dielectric_model": "other"}},
            {"composition": [("Si", 0.05)]},
        )


@pytest.mark.parametrize("beam_keV", [40.0, 3.0e3, 1.0e5])
def test_coupled_core_args_suppress_at_every_beam_energy(beam_keV):
    """Always-on coupling: no step in the radiative model at 30 MeV."""
    from pyrite.montecarlo.transport.hard_radiative import radiative_core_args

    si = prepare_bremslib_table(
        synthetic_bremslib_arrays(t1_MeV=np.array([5.0e-3, 1.0, 30.0, 100.0])),
        atomic_number=14,
        key="synthetic/Si",
        digest="3" * 64,
    )
    composition = [("Si", 0.05)]
    args = radiative_core_args(
        [composition], {"Si": si}, 1000.0, np.array([5.0]), np.array([beam_keV]), 7, 4
    )
    expected = plasma_energy_eV(MediumSpec("layer0", tuple(composition)))
    np.testing.assert_allclose(args[-1], [expected])
