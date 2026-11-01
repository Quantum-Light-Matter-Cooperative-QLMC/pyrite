"""Host photon history queue, charged handoff and ordinary aperture hits (#341)."""

import numpy as np
import pytest

from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV
from pyrite.montecarlo.transport.photon_histories import (
    POSITRON,
    RELAXATION_CHANNEL,
    PhotonAperture,
    daughter_stream_key,
    detected_spectrum,
    ledger_residual_eV,
    photon_stream_key,
    track_photon_histories,
)
from pyrite.montecarlo.transport.photon_navigation import PHOTON_CHANNEL_ORDER

CU = 0.0849  # atoms/Å³
FAR = PhotonAperture((0, 0, 1e9), (1, 0, 0), (0, 1, 0), 1e12, 1e12)


def _beam(n, energy_eV, origin=(0, 0, -1.0), direction=(0, 0, 1.0)):
    return (
        np.tile(origin, (n, 1)).astype(float),
        np.tile(direction, (n, 1)).astype(float),
        np.full(n, float(energy_eV)),
        np.ones(n),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
    )


def _track(batch, layers, **kwargs):
    options = {"seed": 341, "photon_cutoff_eV": 1e3, "electron_cutoff_eV": 1e3} | kwargs
    return track_photon_histories(*batch, layers, **options)


def _assert_closed(histories):
    ledger = histories.ledger
    residual = ledger_residual_eV(ledger)
    assert np.all(np.abs(residual) <= 1e-12 * ledger["incident_eV"])


def test_vacuum_hits_aperture_at_analytic_point():
    batch = list(_beam(3, 5e4))
    batch[1] = np.array([[0, 0, 1.0], [0.6, 0, 0.8], [0, 0, -1.0]])
    aperture = PhotonAperture((0, 0, 10.0), (1, 0, 0), (0, 1, 0), 10.0, 1.0)
    histories = _track(batch, [], aperture=aperture)
    exits = histories.exits
    assert exits["detected"].tolist() == [True, True, False]
    np.testing.assert_allclose(exits["local_ang"][:2], [[0, 0], [8.25, 0]], atol=1e-12)
    assert np.isnan(exits["local_ang"][2]).all()
    assert histories.interactions["channel"].size == 0
    assert histories.ledger["detected_eV"].sum() == 1e5
    assert histories.ledger["escaped_eV"].sum() == 5e4
    _assert_closed(histories)


def test_aperture_is_absorbing_boundary_before_material():
    aperture = PhotonAperture((0, 0, 5.0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    histories = _track(_beam(50, 2e4), [(10.0, 1e9, [("Cu", CU)])], aperture=aperture)
    assert histories.exits["detected"].all()
    assert histories.interactions["channel"].size == 0


def test_aperture_rejects_non_orthonormal_axes():
    with pytest.raises(ValueError, match="orthonormal"):
        PhotonAperture((0, 0, 1), (1, 0, 0), (1, 0, 0), 1, 1)
    with pytest.raises(ValueError, match="positive"):
        PhotonAperture((0, 0, 1), (1, 0, 0), (0, 1, 0), 0, 1)


def test_uncollided_transmission_matches_beer_lambert():
    n, energy, thickness = 4000, 3e4, 2e4
    histories = _track(_beam(n, energy), [(0.0, thickness, [("Cu", CU)])], aperture=FAR)
    exits = histories.exits
    direct = np.sum(exits["detected"] & (exits["generation"] == 0))
    mu = CU * sum(v[0] for v in photon_cross_sections_ang2("Cu", [energy]).values())
    expected = np.exp(-mu * thickness)
    assert abs(direct / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n)
    _assert_closed(histories)


def test_first_collision_channels_follow_partial_rates():
    n, energy = 3000, 1e5
    histories = _track(_beam(n, energy), [(0.0, 1e12, [("Cu", CU)])])
    first = histories.interactions["generation"] == 0
    channels = histories.interactions["channel"][first]
    assert channels.size == n
    sigma = photon_cross_sections_ang2("Cu", [energy])
    total = sum(v[0] for v in sigma.values())
    for index, name in enumerate(PHOTON_CHANNEL_ORDER):
        p = sigma[name][0] / total
        assert abs(np.mean(channels == index) - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-12


def test_backscatter_reaches_upstream_aperture_only_after_interaction():
    # Source points away from the detector; only scattered/fluorescent photons arrive.
    upstream = PhotonAperture((0, 0, -1e6), (0, 1, 0), (1, 0, 0), 1e9, 1e9)
    histories = _track(_beam(1500, 3e4), [(0.0, 1e9, [("Cu", CU)])], aperture=upstream)
    exits = histories.exits
    hits = exits["detected"].astype(bool)
    assert hits.sum() > 0
    assert np.all(exits["generation"][hits] >= 1)
    assert np.all(exits["direction"][hits, 2] < 0)
    _assert_closed(histories)


def test_secondary_fluorescence_emits_eadl_cu_k_lines():
    k_shell = load_eadl_relaxation("Cu").subshells[0]
    k_lines = k_shell.radiative_energy_eV[k_shell.radiative_probability > 0.05]
    histories = _track(_beam(1500, 2e4), [(0.0, 1e9, [("Cu", CU)])], aperture=FAR)
    exits = histories.exits
    fluorescence = exits["energy_eV"][exits["generation"] >= 1]
    # EADL Kα1/Kα2/Kβ1 transition energies, unshifted by transport.
    assert set(k_lines) <= set(fluorescence)
    auger = histories.charged["channel"] == RELAXATION_CHANNEL
    assert np.all(histories.charged["kinetic_eV"][auger] > 1e3)
    _assert_closed(histories)


@pytest.mark.parametrize(
    "element,density,energy",
    [("C", 0.113, 1.5e4), ("Si", 0.05, 5e5), ("Cu", CU, 9.5e3), ("W", 0.063, 2e6)],
)
def test_per_history_energy_closes(element, density, energy):
    histories = _track(_beam(200, energy), [(0.0, 1e8, [(element, density)])], aperture=FAR)
    _assert_closed(histories)
    assert histories.ledger["complete"].all()


def test_pair_rows_and_rest_energy():
    histories = _track(_beam(300, 2e7), [(0.0, 1e9, [("W", 0.063)])], electron_cutoff_eV=1e5)
    channels = histories.interactions["channel"]
    pairs = np.isin(
        channels, [PHOTON_CHANNEL_ORDER.index(n) for n in ("pair_nuclear", "pair_electron")]
    )
    positrons = histories.charged["species"] == POSITRON
    assert pairs.sum() > 0
    assert positrons.sum() == pairs.sum()
    assert histories.ledger["positron_rest_eV"].sum() == pytest.approx(
        pairs.sum() * PAIR_THRESHOLD_EV
    )
    _assert_closed(histories)


def test_generation_cap_reports_unresolved_energy():
    histories = _track(_beam(100, 1e5), [(0.0, 1e12, [("C", 0.113)])], max_generations=1)
    ledger = histories.ledger
    assert np.all(histories.interactions["generation"] == 0)
    assert not ledger["complete"].all()
    assert ledger["unresolved_eV"].sum() > 0
    assert ledger["deposited_eV"].sum() < ledger["incident_eV"].sum()
    _assert_closed(histories)


def test_histories_do_not_depend_on_order_or_chunking():
    layers = [(0.0, 5e6, [("Cu", CU)]), (6e6, 2e7, [("Si", 0.05)])]
    batch = _beam(120, 6e4)
    whole = _track(batch, layers, aperture=FAR)
    order = np.random.default_rng(3).permutation(120)
    parts = [_track([a[i] for a in batch], layers, aperture=FAR) for i in np.array_split(order, 3)]

    def by_key(tables):
        rows = {}
        for table in tables:
            for i, key in enumerate(table["key"]):
                rows[int(key)] = (table["energy_eV"][i], tuple(table["direction"][i]))
        return rows

    assert by_key([whole.exits]) == by_key([p.exits for p in parts])
    for name in ("charged_kinetic_eV", "detected_eV", "deposited_eV"):
        merged = np.zeros(120)
        for part in parts:
            merged[part.ledger["history_id"]] = part.ledger[name]
        np.testing.assert_array_equal(whole.ledger[name], merged)


def test_weight_scales_every_ledger_term():
    batch = _beam(40, 4e4)
    layers = [(0.0, 1e7, [("Cu", CU)])]
    one = _track(batch, layers)
    doubled = _track((*batch[:3], 2 * batch[3], *batch[4:]), layers)
    for name in ("incident_eV", "escaped_eV", "deposited_eV", "charged_kinetic_eV"):
        np.testing.assert_array_equal(doubled.ledger[name], 2 * one.ledger[name])


def test_stream_keys_follow_lineage():
    root = photon_stream_key(7, 3, 0)
    assert root == photon_stream_key(7, np.array([3]), np.array([0]))[0]
    assert root != photon_stream_key(7, 3, 1) != photon_stream_key(8, 3, 0)
    assert daughter_stream_key(root, 0) != daughter_stream_key(root, 1) != root


def test_detected_spectrum_normalization():
    histories = _track(_beam(500, 2e4), [(0.0, 1e5, [("Cu", CU)])], aperture=FAR)
    edges = np.linspace(1e3, 2.1e4, 21)
    spectrum = detected_spectrum(histories, edges, 500)
    hits = histories.exits["detected"].astype(bool)
    total = histories.exits["weight"][hits].sum()
    assert np.sum(spectrum * np.diff(edges)) * 500 == pytest.approx(total)
    with pytest.raises(ValueError, match="increasing"):
        detected_spectrum(histories, [1.0, 1.0], 1)


@pytest.mark.parametrize("large_energy", [5e3, 2e4, 6e4])
@pytest.mark.parametrize("reverse", [False, True])
def test_detected_spectrum_keeps_small_hit_in_an_independent_bin(large_energy, reverse):
    batch = list(_beam(2, large_energy))
    batch[2][1] = 4e4
    batch[3][0] = 1e20
    if reverse:
        batch = [array[::-1] for array in batch]
    histories = _track(batch, [], aperture=FAR)
    assert histories.exits["detected"].all()
    expected = [2.5e15 if large_energy == 2e4 else 0.0, 2.5e-5]
    np.testing.assert_array_equal(detected_spectrum(histories, [1e4, 3e4, 5e4], 2), expected)


def test_detected_spectrum_preserves_bin_edges_and_discards_outside_hits():
    batch = list(_beam(5, 2e4))
    batch[2] = np.array([1e4, 3e4, 5e4, 8e3, 6e4])
    batch[3] = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    histories = _track(batch, [], aperture=FAR)
    np.testing.assert_array_equal(
        detected_spectrum(histories, [1e4, 3e4, 5e4], 2), np.array([1.0, 6.0]) / 4e4
    )


@pytest.mark.parametrize("column", [4, 5])
@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_source_ids_refuse_unsigned_values_outside_int64(column, value):
    batch = list(_beam(1, 2e4))
    batch[column] = np.array([value], dtype=np.uint64)
    with pytest.raises(ValueError, match="fit in int64"):
        _track(batch, [], aperture=FAR)


def test_source_ids_preserve_the_largest_supported_label():
    batch = list(_beam(1, 2e4))
    batch[4] = batch[5] = np.array([2**63 - 1], dtype=np.uint64)
    histories = _track(batch, [], aperture=FAR)
    np.testing.assert_array_equal(histories.exits["history_id"], [2**63 - 1])
    np.testing.assert_array_equal(histories.ledger["history_id"], [2**63 - 1])


@pytest.mark.parametrize(
    "change,match",
    [
        ({"energy": 5e2}, "exceed the photon cutoff"),
        ({"weight": 0.0}, "positive"),
        ({"direction": (0, 0, 2.0)}, "unit"),
        ({"duplicate": True}, "unique"),
    ],
)
def test_invalid_batches_raise(change, match):
    origin, direction, energy, weight, history, event = _beam(2, 1e4)
    if "energy" in change:
        energy[:] = change["energy"]
    if "weight" in change:
        weight[:] = change["weight"]
    if "direction" in change:
        direction[:] = change["direction"]
    if "duplicate" in change:
        history[:] = 0
    with pytest.raises(ValueError, match=match):
        _track((origin, direction, energy, weight, history, event), [])


def test_empty_batch_is_valid():
    histories = _track(_beam(0, 1e4), [(0.0, 1.0, [("Cu", CU)])])
    assert histories.exits["energy_eV"].shape == (0,)
    assert histories.exits["direction"].shape == (0, 3)
    assert histories.ledger["history_id"].size == 0
