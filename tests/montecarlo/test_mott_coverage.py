"""Mott transport rejects energies outside configured SRD 64 tables (#318)."""

import numpy as np
import pytest

from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
from tests.helpers.mott_synthetic import A0_SQ_CM2, synthetic_transport_cm2


@pytest.fixture
def mott_tables(tmp_path, monkeypatch):
    # Synthetic physics, with the real SRD 64 export's 50 eV--300 keV bounds.
    for element, z in [("Si", 14), ("W", 74)]:
        energy = np.array([50.0, 1e3, 1e4, 3e5])
        sigma = synthetic_transport_cm2(z, energy) / A0_SQ_CM2
        path = tmp_path / f"DisplayCalcTCSTableFor{element}.csv"
        path.write_text(
            "SYNTHETIC TEST FIXTURE - NOT NIST SRD 64 DATA\n"
            + "\n".join(
                f"{i}, {e:.17g}, {s:.17e}"
                for i, (e, s) in enumerate(zip(energy, sigma, strict=True), 1)
            )
            + "\n"
        )
    monkeypatch.setenv("PYRITE_MOTT_TABLES_DIR", str(tmp_path))
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    return tmp_path


def _run(energy=300.0, **kwargs):
    return simulate_trajectories(
        energy,
        16,
        1.0,
        element="Si",
        n_atoms_per_ang3=0.05,
        elastic_model="mott",
        seed=41,
        max_steps=8,
        **kwargs,
    )


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
@pytest.mark.parametrize("lut", [False, True])
def test_mott_table_endpoints_are_inclusive(mott_tables, core, lut):
    result = _run(
        E_cut_keV=0.05,
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
    )
    np.testing.assert_array_equal(result["initial_E_keV"], np.full(16, 300.0))


@pytest.mark.parametrize("core", ["auto", "lockstep", "per-electron", "cuda"])
@pytest.mark.parametrize("lut", [False, True])
@pytest.mark.parametrize(
    "energy, options",
    [
        (np.nextafter(300.0, np.inf), {}),
        (300.0, {"energy_spread_frac": 0.01}),
        (300.0, {"E_cut_keV": np.nextafter(0.05, 0.0)}),
        (300.0, {"E_cut_by_electrons": np.array([0.01] + [1.0] * 15)}),
    ],
)
def test_mott_outside_table_fails_before_any_core(mott_tables, core, lut, energy, options):
    # CUDA rejection is host-side: no device or kernel invocation is needed.
    with pytest.raises(ValueError, match=r'Si.*\[0.05, 300\] keV.*elastic_model="elsepa"'):
        _run(
            energy,
            transport_core=core,
            transport_lut_config=TransportLUTConfig(enabled=lut),
            **options,
        )


def test_mott_checks_every_layer_element(mott_tables):
    path = mott_tables / "DisplayCalcTCSTableForW.csv"
    path.write_text(path.read_text().replace("300000,", "200000,"))
    with pytest.raises(ValueError, match=r"W.*\[0.05, 200\] keV"):
        _run(layers=[(0.0, 0.5, [("Si", 0.05)]), (0.5, 1.0, [("W", 0.063)])])


def test_mott_checks_drawn_range_without_extending_it(mott_tables):
    result = _run(290.0, energy_spread_frac=0.001, transport_core="lockstep")
    assert np.max(result["initial_E_keV"]) < 300.0


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
@pytest.mark.parametrize("lut", [False, True])
def test_mott_guard_preserves_in_range_results_bit_for_bit(mott_tables, monkeypatch, core, lut):
    from pyrite.montecarlo.transport import api

    options = dict(
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=lut),
        energy_spread_frac=0.001,
    )
    checked = _run(290.0, **options)
    monkeypatch.setattr(api, "check_mott_coverage", lambda *_: None)
    unchecked = _run(290.0, **options)
    assert checked.keys() == unchecked.keys()
    for key in checked:
        np.testing.assert_array_equal(checked[key], unchecked[key], err_msg=key)
