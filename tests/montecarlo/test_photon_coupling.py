"""Photon histories coupled to photon-born electron transport (#341).

Validation: photon-charged-coupling
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport.photon_coupling import (
    ChargedStageResult,
    coupled_ledger_residual_eV,
    track_coupled_histories,
)
from pyrite.montecarlo.transport.photon_histories import (
    ELECTRON,
    POSITRON,
    daughter_stream_key,
)
from pyrite.montecarlo.transport.photon_navigation import PhotonBox

CU = (("Cu", 0.0849),)
EYE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class HalfRadiator:
    """Each launched electron deposits half its energy and radiates one forward photon.

    The second ordinal photon (below the 1 keV photon cutoff) checks local
    deposition of soft returned photons.
    """

    threshold_eV = 5e3

    def __init__(self):
        self.calls = []

    def __call__(self, rows, seed):
        self.calls.append((dict(rows), seed))
        kinetic = rows["kinetic_eV"]
        n = kinetic.size
        soft = np.full(n, 500.0)
        hard = kinetic / 2 - soft
        photons = {
            "position_ang": np.repeat(rows["position_ang"], 2, axis=0),
            "direction": np.tile([0.0, 0.0, 1.0], (2 * n, 1)),
            "energy_eV": np.column_stack([hard, soft]).ravel(),
            "parent": np.repeat(np.arange(n), 2),
            "ordinal": np.tile([0, 1], n),
        }
        zero = np.zeros(n)
        return ChargedStageResult(zero, kinetic / 2, zero, kinetic / 2, photons)


def _beam(n, energy, origin=(0.0, 0.0, -1.0)):
    return (
        np.tile(origin, (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.full(n, float(energy)),
        np.full(n, 2.0),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
    )


def _run(stage, layers, n=200, energy=6e4, **kw):
    options = {"seed": 7, "photon_cutoff_eV": 1e3, "charged_stage": stage} | kw
    return track_coupled_histories(*_beam(n, energy), layers, **options)


def _assert_closed(out):
    residual = coupled_ledger_residual_eV(out.ledger)
    assert np.all(np.abs(residual) <= 1e-12 * out.ledger["incident_eV"])


def test_stage_receives_only_launchable_electrons_and_photons_return():
    stage = HalfRadiator()
    out = _run(stage, [(0.0, 1e7, CU)])
    _assert_closed(out)
    assert out.rounds >= 1 and out.ledger["complete"].all()
    rows, seed = stage.calls[0]
    assert np.all(rows["species"] == ELECTRON)
    assert np.all(rows["kinetic_eV"] > stage.threshold_eV)
    first = out.charged["round"] == 0
    launched = first & (out.charged["fate"] == "launched")
    assert launched.sum() == rows["kinetic_eV"].size
    # Returned photons restart the queue in the next round with lineage keys.
    second = out.interactions["round"] == 1
    returned_keys = {daughter_stream_key(k, 0) for k in rows["key"]}
    exits = out.exits["round"] == 1
    descended = set(out.exits["key"][exits]) | set(out.interactions["key"][second])
    assert descended & returned_keys
    assert np.all(out.exits["weight"] == 2.0)
    soft = 500.0 * 2.0 * sum(r["kinetic_eV"].size for r, _ in stage.calls)
    assert out.ledger["radiated_local_eV"].sum() == pytest.approx(soft)


def test_positrons_stay_pending_and_box_electrons_deposit_locally():
    box = PhotonBox((0.0, 0.0, 3e7), EYE, (1e8, 1e8, 2.5e7), CU)
    stage = HalfRadiator()
    out = _run(stage, [(0.0, 1e6, CU)], energy=4e6, n=300, boxes=[box])
    _assert_closed(out)
    fate = out.charged["fate"]
    positron = out.charged["species"] == POSITRON
    assert positron.any() and np.all(fate[positron] == "positron")
    pending = (out.charged["weight"] * out.charged["kinetic_eV"])[positron].sum()
    assert out.ledger["positron_pending_eV"].sum() == pytest.approx(pending)
    in_box = out.charged["position_ang"][:, 2] >= 5e6
    assert in_box.any() and np.all(fate[in_box & ~positron] == "local")
    for rows, _ in stage.calls:
        assert np.all(rows["position_ang"][:, 2] < 1e6)


def test_round_cap_marks_launches_unresolved():
    out = _run(HalfRadiator(), [(0.0, 1e7, CU)], max_rounds=0)
    _assert_closed(out)
    assert out.rounds == 0
    unresolved = out.charged["fate"] == "unresolved"
    assert unresolved.any()
    assert not out.ledger["complete"].all()
    assert out.ledger["charged_unresolved_eV"].sum() > 0


def test_coupling_is_reproducible_and_stage_seeds_differ_by_round():
    first, second = HalfRadiator(), HalfRadiator()
    a = _run(first, [(0.0, 1e7, CU)])
    b = _run(second, [(0.0, 1e7, CU)])
    for name in a.ledger:
        np.testing.assert_array_equal(a.ledger[name], b.ledger[name])
    seeds = [seed for _, seed in first.calls]
    assert seeds == [seed for _, seed in second.calls]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"electron_cutoff_eV": 1e3}, "threshold_eV"), ({"max_rounds": -1}, "max_rounds")],
)
def test_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _run(HalfRadiator(), [(0.0, 1e7, CU)], **kwargs)


@pytest.fixture(scope="module")
def silicon_stage():
    from pyrite.materials import CATALOG
    from pyrite.montecarlo import shell_configuration
    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from pyrite.montecarlo.transport.photon_coupling import cascade_charged_stage
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    composition = CATALOG.crystal("silicon").composition
    layers = [(0.0, 5e7, composition)]
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([1e-2, 1e-1, 1.0, 2.0, 5.0]))
    stage = cascade_charged_stage(
        layers,
        E_cut_keV=50.0,
        max_steps=500_000,
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        secondary_threshold_eV=50_000.0,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": prepare_bremslib_table(arrays, atomic_number=14)},
        transport_core="per-electron",
    )
    return layers, stage


@pytest.mark.slow
def test_cascade_stage_closes_per_row_and_returns_its_radiation(silicon_stage):
    layers, stage = silicon_stage
    n = 8
    angle = np.linspace(0.0, 1.2, n)
    rows = {
        "position_ang": np.column_stack([np.zeros(n), np.zeros(n), np.linspace(1e7, 4e7, n)]),
        "direction": np.column_stack([np.sin(angle), np.zeros(n), np.cos(angle)]),
        "kinetic_eV": np.linspace(3e5, 1.5e6, n),
        "key": np.arange(11, 11 + n, dtype=np.uint64),
    }
    result = stage(rows, 5)
    total = result.escaped_eV + result.deposited_eV + result.binding_eV + result.radiated_eV
    np.testing.assert_allclose(total, rows["kinetic_eV"], rtol=1e-12)
    photons = result.photons
    assert photons["energy_eV"].size and np.all(photons["energy_eV"] > 0)
    radiated = np.bincount(photons["parent"], weights=photons["energy_eV"], minlength=n)
    np.testing.assert_allclose(radiated, result.radiated_eV, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(photons["direction"], axis=1), 1.0, atol=1e-12)
    for parent in range(n):
        ordinals = photons["ordinal"][photons["parent"] == parent]
        np.testing.assert_array_equal(ordinals, np.arange(ordinals.size))
    again = stage(rows, 5)
    np.testing.assert_array_equal(again.photons["energy_eV"], photons["energy_eV"])


def test_coupled_silicon_histories_close(silicon_stage):
    layers, stage = silicon_stage
    out = track_coupled_histories(
        *_beam(60, 2e6), layers, seed=3, photon_cutoff_eV=1e3, charged_stage=stage
    )
    _assert_closed(out)
    assert out.rounds >= 1
    assert (out.charged["fate"] == "launched").any()
    assert out.ledger["charged_deposited_eV"].sum() > 0
