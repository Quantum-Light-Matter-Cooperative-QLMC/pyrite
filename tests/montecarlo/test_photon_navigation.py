"""First-flight geometry and channel competition; no production photon tracker.

Validation: photon-channel-first-interaction
"""

import numpy as np
import pytest

from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.montecarlo.transport import photon_navigation as nav
from pyrite.montecarlo.transport.pair_production import (
    photon_first_interactions as legacy_first_interactions,
)


@pytest.fixture
def rates(monkeypatch):
    """Synthetic rates with exact powers-of-two totals, independent of EPDL."""

    def cross_sections(element, energy):
        values = {"C": [1, 0, 2, 1, 0], "Si": [0, 1, 0, 2, 1]}[element]
        return {
            c: np.full_like(energy, v, dtype=float)
            for c, v in zip(nav.PHOTON_CHANNEL_ORDER, values, strict=True)
        }

    monkeypatch.setattr(nav, "photon_cross_sections_ang2", cross_sections)


def _sample(origins, directions, depth, layers, *, draw=0.0, **kwargs):
    origins, directions = np.asarray(origins, float), np.asarray(directions, float)
    n = len(origins)
    uniforms = np.column_stack(
        [np.broadcast_to(-np.expm1(-np.asarray(depth)), n), np.broadcast_to(draw, n)]
    )
    return nav.photon_first_interactions(
        origins, directions, np.full(n, 1e5), uniforms, layers, **kwargs
    )


def test_all_channel_element_intervals_and_zero_rates(rates):
    layers = [(0, 10, [("C", 1.0), ("Si", 1.0)])]
    # Eight equiprobable units: C=(PE, IC, IC, PN), Si=(CO, PN, PN, PT).
    hit = _sample(
        np.zeros((8, 3)), np.tile([0, 0, 1], (8, 1)), 0.5, layers, draw=(np.arange(8) + 0.5) / 8
    )
    np.testing.assert_array_equal(hit["element"], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(hit["channel"], [0, 2, 2, 3, 1, 3, 3, 4])
    np.testing.assert_allclose(hit["distance_ang"], 0.5 / 8)
    # Inverse-CDF ties must skip zero-rate entries, including a zero final channel.
    for draw, channel in [(0, 0), (0.25, 2), (0.75, 3), (np.nextafter(1.0, 0), 3)]:
        h = _sample([[0, 0, 0]], [[0, 0, 1]], 0.5, [(0, 1, [("C", 1)])], draw=draw)
        assert h["channel"][0] == channel


def test_piecewise_depth_through_vacuum_and_backwards(rates):
    layers = [(0, 1, [("C", 0.25)]), (3, 5, [("Si", 0.5)])]
    hit = _sample(
        [[0, 0, -2], [0, 0, 6], [0, 0, -2]], [[0, 0, 1], [0, 0, -1], [0, 0, 1]], [2, 4.5, 6], layers
    )
    np.testing.assert_array_equal(hit["layer"], [1, 0, -1])
    np.testing.assert_allclose(hit["distance_ang"][:2], [5.5, 5.5])
    assert np.isinf(hit["distance_ang"][2])
    assert hit["channel"][2] == hit["element"][2] == -1


def test_vacuum_and_lateral_infinite_slab(rates):
    for layers in ([], [(0, 1, [])], [(0, 1, [("C", 0)])]):
        hit = _sample([[0, 0, 0.5]], [[1, 0, 0]], 0.5, layers)
        assert np.isinf(hit["distance_ang"][0])
        assert hit["layer"][0] == -1
    hit = _sample([[0, 0, 0.5]], [[1, 0, 0]], 2, [(0, 1, [("C", 0.25)])])
    assert hit["distance_ang"][0] == pytest.approx(2)


def test_boundaries_have_one_owner_without_displacement(rates):
    layers = [(0, 1, [("C", 0.25)]), (1, 2, [("Si", 0.5)])]
    origins = [[0, 0, 1]] * 3 + [[0, 0, 2], [0, 0, 0]]
    directions = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0, 1], [0, 0, -1]]
    hit = _sample(origins, directions, 0, layers)
    np.testing.assert_array_equal(hit["layer"], [1, 0, 1, -1, -1])
    np.testing.assert_array_equal(hit["distance_ang"][:3], 0)
    # Exactly one optical depth through layer 0: continue at layer 1 entry.
    hit = _sample([[0, 0, 0]], [[0, 0, 1]], 1, layers)
    assert hit["layer"][0] == 1
    assert hit["distance_ang"][0] == 1


def test_prism_entry_misses_tangencies_and_sides(rates):
    layers = [(0, 2, [("C", 0.25)])]
    origins = [[-3, 0, 1], [3, 0, 1], [0, 0, -2], [-3, 1, 1], [0, 0, 1]]
    directions = [[1, 0, 0], [1, 0, 0], [0, 0, -1], [1, 0, 0], [1, 0, 0]]
    start, end = nav.ray_layer_intervals(origins, directions, layers, width_ang=2, height_ang=2)
    np.testing.assert_array_equal(start[:, 0], [2, 0, 0, 0, 0])
    np.testing.assert_array_equal(end[:, 0], [4, 0, 0, 0, 1])
    hit = _sample(origins, directions, 0.5, layers, width_ang=2, height_ang=2)
    np.testing.assert_allclose(hit["distance_ang"][[0, 4]], [2.5, 0.5])
    np.testing.assert_array_equal(hit["layer"], [0, -1, -1, -1, 0])


def test_oblique_three_axis_intersection():
    direction = np.array([[1, 2, 2]]) / 3
    start, end = nav.ray_layer_intervals(
        [[-2, -3, -2]], direction, [(0, 4, [])], width_ang=4, height_ang=4
    )
    # z entry at s=3; y exit at s=7.5 (before z exit=9, x exit=12).
    np.testing.assert_allclose(start, [[3]])
    np.testing.assert_allclose(end, [[7.5]])


def test_epdl_channel_frequencies_and_escape():
    n, energy = 12_000, 6e6
    composition = [("C", 0.08), ("Pb", 0.02)]
    rates = np.array(
        [
            density * np.array(list(photon_cross_sections_ang2(el, energy).values()))
            for el, density in composition
        ]
    )
    mu = rates.sum()
    rng = np.random.default_rng(341)
    hit = nav.photon_first_interactions(
        np.zeros((n, 3)),
        np.tile([0, 0, 1], (n, 1)),
        np.full(n, energy),
        rng.random((n, 2)),
        [(0, 1 / mu, composition)],
    )
    escaped = hit["layer"] == -1
    assert abs(escaped.mean() - np.exp(-1)) < 5 * np.sqrt(np.exp(-1) * (1 - np.exp(-1)) / n)
    # Unconditional probabilities test every channel/element, including rare triplet events.
    expected = n * (1 - np.exp(-1)) * rates / mu
    for element in range(2):
        for channel in range(5):
            observed = np.count_nonzero((hit["element"] == element) & (hit["channel"] == channel))
            assert (
                abs(observed - expected[element, channel])
                < 5 * np.sqrt(expected[element, channel]) + 1
            )


def test_batch_reordering_chunking_and_legacy_depths():
    rng = np.random.default_rng(3410)
    n = 51
    origins = rng.uniform([-1, -1, 1], [1, 1, 99], (n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    energies = np.geomspace(1e3, 1e7, n)
    uniforms = rng.random((n, 2))
    layers = [(0, 50, [("C", 0.1)]), (50, 100, [("Pb", 0.03)])]
    kwargs = {"width_ang": 200, "height_ang": 200}
    whole = nav.photon_first_interactions(origins, directions, energies, uniforms, layers, **kwargs)
    permutation = rng.permutation(n)
    reordered = nav.photon_first_interactions(
        origins[permutation],
        directions[permutation],
        energies[permutation],
        uniforms[permutation],
        layers,
        **kwargs,
    )
    chunks = [
        nav.photon_first_interactions(
            origins[s], directions[s], energies[s], uniforms[s], layers, **kwargs
        )
        for s in [slice(0, 7), slice(7, n)]
    ]
    for key in whole:
        np.testing.assert_array_equal(reordered[key], whole[key][permutation])
        np.testing.assert_array_equal(np.concatenate([h[key] for h in chunks]), whole[key])
    old = legacy_first_interactions(origins, directions, energies, uniforms, layers, **kwargs)
    np.testing.assert_array_equal(whole["layer"], old["layer"])
    np.testing.assert_allclose(whole["distance_ang"], old["distance_ang"], rtol=1e-14)


def test_empty_batch():
    hit = nav.photon_first_interactions(
        np.empty((0, 3)), np.empty((0, 3)), [], np.empty((0, 2)), [(0, 1, [("C", 1)])]
    )
    assert all(value.shape == (0,) for value in hit.values())


def _plane_crossing_reference(origin, direction, uniforms, layers, footprint):
    """Scalar oracle: split at every plane, classify midpoints, spend depth.

    Uses synthetic per-atom rates from the fixture, no navigation helpers or
    EPDL getter. Midpoint classification is independent of ray/box clipping.
    """
    planes = [(2, face) for top, bottom, _ in layers for face in (top, bottom)]
    if footprint is not None:
        planes += [(axis, face) for axis in (0, 1) for face in (-footprint / 2, footprint / 2)]
    crossings = {0.0}
    for axis, face in planes:
        if direction[axis] != 0:
            distance = (face - origin[axis]) / direction[axis]
            if distance > 0:
                crossings.add(float(distance))
    edges = [*sorted(crossings), np.inf]
    depth = -np.log1p(-uniforms[0])
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        midpoint = start + 1 if np.isinf(end) else (start + end) / 2
        point = origin + midpoint * direction
        if footprint is not None and not all(
            -footprint / 2 <= q < footprint / 2 for q in point[:2]
        ):
            continue
        for layer, (top, bottom, composition) in enumerate(layers):
            if not top <= point[2] < bottom:
                continue
            weights = [
                (atom, channel, density * rate)
                for atom, (element, density) in enumerate(composition)
                for channel, rate in enumerate(
                    {"C": [1, 0, 2, 1, 0], "Si": [0, 1, 0, 2, 1]}[element]
                )
            ]
            total = sum(weight for _, _, weight in weights)
            if total == 0:
                continue
            if depth >= total * (end - start):
                depth -= total * (end - start)
                continue
            threshold = uniforms[1] * total
            for atom, channel, weight in weights:
                if threshold < weight:
                    return start + depth / total, layer, channel, atom
                threshold -= weight
    return np.inf, -1, -1, -1


@pytest.mark.parametrize("footprint", [None, 4.0])
def test_mixed_material_flights_match_plane_crossing_reference(rates, footprint):
    rng = np.random.default_rng(34101)
    n = 300
    origins = rng.uniform(-6, 6, (n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Force shared/outer faces, parallel directions and inward/outward entry.
    origins[:6] = [[0, 0, z] for z in (0, 1, 3, 5, 5, 0)]
    directions[:6] = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, -1]]
    uniforms = rng.random((n, 2))
    uniforms[:6, 0] = 0
    layers = [(0, 1, [("C", 0.25)]), (2, 3, []), (3, 5, [("Si", 0.5), ("C", 0.125)])]
    hit = nav.photon_first_interactions(
        origins,
        directions,
        np.full(n, 1e5),
        uniforms,
        layers,
        width_ang=footprint,
        height_ang=footprint,
    )
    reference = np.array(
        [
            _plane_crossing_reference(origin, direction, draws, layers, footprint)
            for origin, direction, draws in zip(origins, directions, uniforms, strict=True)
        ]
    )
    # Independent float64 plane crossings/depth accumulation: a few ulps.
    np.testing.assert_allclose(hit["distance_ang"], reference[:, 0], rtol=1e-13, atol=1e-14)
    for column, key in enumerate(("layer", "channel", "element"), start=1):
        np.testing.assert_array_equal(hit[key], reference[:, column])


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, 1e308])
def test_invalid_rates_are_refused(monkeypatch, value):
    monkeypatch.setattr(
        nav,
        "photon_cross_sections_ang2",
        lambda element, energy: {c: np.full_like(energy, value) for c in nav.PHOTON_CHANNEL_ORDER},
    )
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        _sample([[0, 0, 0]], [[0, 0, 1]], 0.5, [(0, 1, [("C", 1)])])


@pytest.mark.parametrize(
    "change,match",
    [
        ({"origins": [[0, 0]]}, "shape"),
        ({"directions": [[0, 0, 2]]}, "unit"),
        ({"origins": [[0, np.nan, 0]]}, "finite"),
        ({"energies_eV": [0]}, "domain"),
        ({"energies_eV": [np.inf]}, "domain"),
        ({"uniforms": [[0, 1]]}, "uniforms"),
        ({"uniforms": [[-1, 0]]}, "uniforms"),
        ({"uniforms": [[np.nan, 0]]}, "uniforms"),
        ({"uniforms": [0, 0]}, "shape"),
        ({"layers": [(0, 2, []), (1, 3, [])]}, "ordered"),
        ({"layers": [(0, 0, [])]}, "ordered"),
        ({"layers": [(0, 1, [("C", -1)])]}, "densities"),
        ({"width_ang": 1}, "both"),
    ],
)
def test_invalid_inputs(change, match):
    args = dict(
        origins=[[0, 0, 0]],
        directions=[[0, 0, 1]],
        energies_eV=[1e5],
        uniforms=[[0, 0]],
        layers=[(0, 1, [("C", 1)])],
    )
    args.update(change)
    with pytest.raises(ValueError, match=match):
        nav.photon_first_interactions(**args)
