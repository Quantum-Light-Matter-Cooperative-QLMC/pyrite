"""Issue #795: bitwise equivalence to the retained scalar completion sampler.

The scalar sampler in hard_radiative.py is the executable pre-batch reference.
These tables have Si/W atomic numbers but synthetic, intentionally distinct
angular shapes; the optional released-table test covers physical tables.
Validation: bremslib-radiative-partition.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import hard_radiative as rad
from tests.helpers.bremslib import synthetic_bremslib_arrays


def _tables():
    base = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    tungsten = replace(
        base,
        atomic_number=74,
        scaled_ddcs_mb_sr=base.scaled_ddcs_mb_sr * (1.0 + base.theta_rad[None, None, :]),
    )
    return {"Si": base, "W": tungsten}


def _case(elements, seed=795):
    n = len(elements)
    indices = np.arange(n)
    incident = np.array([10.0, 20.0, 60.0, 200.0])[indices % 4]
    energy = 1e3 * incident * np.array([1e-7, 0.05, 0.5, 1.0])[indices % 4]
    energy[indices % 7 == 0] = np.nextafter(1e3 * incident[indices % 7 == 0], 0.0)
    directions = np.tile([0.6, 0.0, 0.8], (n, 1))
    directions[indices % 3 == 0] = [0.0, 0.0, 1.0]
    directions[indices % 3 == 1] = [1.0, 0.0, 0.0]
    # Gaps in the row ordering must not consume either RNG draw.
    return {
        "hard_radiative_k_eV": np.r_[0.0, energy, 0.0],
        "hard_radiative_Z": np.r_[0, elements, 0],
        "E_end_keV": np.r_[10.0, incident, 10.0],
        "v_hat": np.vstack(([0.0, 0.0, 1.0], directions, [0.0, 0.0, 1.0])),
    }


def _scalar_reference(segments, tables, seed):
    by_z = {t.atomic_number: t for t in tables.values()}
    directions = np.zeros((len(segments["hard_radiative_k_eV"]), 3))
    momenta = np.zeros_like(directions)
    rng = np.random.default_rng(seed ^ 0xD6023FEA38B57C29)
    for i in np.flatnonzero(segments["hard_radiative_k_eV"] > 0.0):
        photon = rad.hard_radiative_photon_at_energy(
            by_z[int(segments["hard_radiative_Z"][i])],
            float(segments["E_end_keV"][i]) * 1e3,
            float(segments["hard_radiative_k_eV"][i]),
            segments["v_hat"][i],
            float(rng.random()),
            float(rng.random()),
        )
        directions[i] = photon.direction
        momenta[i] = photon.target_momentum_eV_c
    return directions, momenta


@pytest.mark.parametrize("elements", [[], [14], [74], [14] * 275, [74] * 275, [14, 74] * 150])
def test_batch_completion_is_bitwise_scalar_reference(elements, monkeypatch):
    tables = _tables()
    segments = _case(elements)
    expected = _scalar_reference(segments, tables, 795)
    original_stage = rad.stage_bremslib_table
    staged = []

    def track(table):
        staged.append(table.atomic_number)
        return original_stage(table)

    monkeypatch.setattr(rad, "stage_bremslib_table", track)
    for _ in range(2):
        staged.clear()
        rad.complete_hard_radiative_events(segments, tables, 795)
        assert staged == list(dict.fromkeys(elements))
        np.testing.assert_array_equal(segments["hard_radiative_direction"], expected[0])
        np.testing.assert_array_equal(segments["hard_radiative_target_momentum_eV_c"], expected[1])


def test_batch_completion_caps_interpolation_rows(monkeypatch):
    segments = _case([14, 74] * 137)
    tables = _tables()
    real = rad._complete_hard_radiative_chunk
    lengths = []

    def probe(staged, incident, energy, incoming, uniforms):
        lengths.append((len(energy), len(energy) * staged.table.theta_rad.size))
        return real(staged, incident, energy, incoming, uniforms)

    monkeypatch.setattr(rad, "_complete_hard_radiative_chunk", probe)
    rad.complete_hard_radiative_events(segments, tables, 795)
    assert len(lengths) > 2
    assert all(n <= rad._COMPLETION_MAX_PHOTONS for n, _ in lengths)
    assert all(rows <= rad._COMPLETION_MAX_ANGLE_ROWS for _, rows in lengths)


@pytest.mark.parametrize(
    "bad, error", [("energy", "photon energy"), ("direction", "electron_direction")]
)
def test_batch_completion_keeps_scalar_validation_messages(bad, error):
    segments = _case([14, 14])
    if bad == "energy":
        segments["hard_radiative_k_eV"][1] = segments["E_end_keV"][1] * 1e3 + 1.0
    else:
        segments["v_hat"][1] = [0.0, 0.0, 0.5]
    with pytest.raises(ValueError, match=error):
        rad.complete_hard_radiative_events(segments, _tables(), 795)


@pytest.mark.parametrize("element", ["Si", "W"])
def test_released_si_w_tables_match_scalar_completion(element):
    """Optional physical-table check on the lab environment with BremsLib installed."""
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    try:
        table = load_bremsstrahlung_tables([element])[element]
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are unavailable: {exc}")
    n = 67
    k = np.geomspace(1_000.0, 299_999.0, n)
    k[0] = 300_000.0
    k[1] = np.nextafter(300_000.0, 0.0)
    segments = {
        "hard_radiative_k_eV": np.r_[0.0, k],
        "hard_radiative_Z": np.r_[0, np.full(n, table.atomic_number)],
        "E_end_keV": np.full(n + 1, 300.0),
        "v_hat": np.tile([0.6, 0.0, 0.8], (n + 1, 1)),
    }
    reference = _scalar_reference(segments, {element: table}, 79599)
    rad.complete_hard_radiative_events(segments, {element: table}, 79599)
    np.testing.assert_array_equal(segments["hard_radiative_direction"], reference[0])
    np.testing.assert_array_equal(segments["hard_radiative_target_momentum_eV_c"], reference[1])
