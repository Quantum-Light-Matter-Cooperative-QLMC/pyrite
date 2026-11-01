"""CUDA first-flight contract and parity; device cases require real hardware."""

import numpy as np
import pytest

from pyrite.montecarlo.transport import photon_navigation as nav
from pyrite.montecarlo.transport.photon_cuda import scene_first_interactions_cuda


@pytest.fixture
def cp():
    from pyrite._backend import BackendUnavailableError, select_cupy_backend

    try:
        return select_cupy_backend().xp
    except BackendUnavailableError as error:
        pytest.skip(str(error))


@pytest.fixture
def rates(monkeypatch):
    def cross_sections(element, energy):
        values = {"C": [1, 0, 2, 1, 0], "Si": [0, 1, 0, 2, 1]}[element]
        return {
            c: np.full_like(energy, v, dtype=float)
            for c, v in zip(nav.PHOTON_CHANNEL_ORDER, values, strict=True)
        }

    monkeypatch.setattr(nav, "photon_cross_sections_ang2", cross_sections)


def _compare(cp, origins, directions, energies, uniforms, layers, boxes=(), **kwargs):
    expected = nav.scene_first_interactions(
        origins, directions, energies, uniforms, layers, boxes, **kwargs
    )
    actual = scene_first_interactions_cuda(
        origins, directions, energies, uniforms, layers, boxes, **kwargs
    )
    array = scene_first_interactions_cuda(
        origins, directions, energies, uniforms, layers, boxes, implementation="array", **kwargs
    )
    for key, value in actual.items():
        assert isinstance(value, cp.ndarray)
        assert value.dtype == expected[key].dtype
        host = cp.asnumpy(value)
        if key == "distance_ang":
            # Binary64 clipping/projection/reduction paths can round differently.
            np.testing.assert_allclose(host, expected[key], rtol=2e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(host, expected[key])
        if key == "distance_ang":
            np.testing.assert_allclose(host, cp.asnumpy(array[key]), rtol=2e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(host, cp.asnumpy(array[key]))
    return actual


def test_explicit_cuda_never_falls_back(monkeypatch):
    from pyrite import _backend

    def unavailable():
        raise _backend.BackendUnavailableError("test: CUDA unavailable")

    monkeypatch.setattr(_backend, "select_cupy_backend", unavailable)
    with pytest.raises(_backend.BackendUnavailableError, match="CUDA unavailable"):
        scene_first_interactions_cuda(np.empty((0, 3)), np.empty((0, 3)), [], np.empty((0, 2)), [])


def test_invalid_inputs_fail_before_device_selection(monkeypatch):
    from pyrite import _backend

    def unexpected(*args, **kwargs):
        pytest.fail("invalid inputs must fail before CUDA selection")

    monkeypatch.setattr(_backend, "select_cupy_backend", unexpected)
    with pytest.raises(ValueError, match="unit vectors"):
        scene_first_interactions_cuda([[0, 0, 0]], [[0, 0, 2]], [1e5], [[0, 0]], [])
    with pytest.raises(ValueError, match="uniforms"):
        scene_first_interactions_cuda([[0, 0, 0]], [[0, 0, 1]], [1e5], [[0, 1]], [])
    with pytest.raises(ValueError, match="ordered"):
        scene_first_interactions_cuda([[0, 0, 0]], [[0, 0, 1]], [1e5], [[0, 0]], [(1, 0, [])])

    for options in ({"implementation": "auto"}, {"threads": True}, {"threads": 0}):
        with pytest.raises(ValueError, match="implementation|threads"):
            scene_first_interactions_cuda([[0, 0, 0]], [[0, 0, 1]], [1e5], [[0, 0]], [], **options)


@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512])
def test_fused_launch_sizes_and_noncontiguous_inputs(cp, rates, threads):
    origins = np.zeros((19, 6))[:, ::2]
    directions = np.tile([0, 4, 0, 4, 1, 4], (19, 1))[:, ::2]
    energies = np.full(38, 1e5)[::2]
    uniforms = np.random.default_rng(341_81).random((19, 4))[:, ::2]
    layers = [(0, 1, [("C", 1)]), (1, 3, [("Si", 1)]), (4, 5, [("C", 1)])]
    expected = nav.scene_first_interactions(origins, directions, energies, uniforms, layers)
    actual = scene_first_interactions_cuda(
        origins, directions, energies, uniforms, layers, threads=threads
    )
    for key in expected:
        np.testing.assert_array_equal(cp.asnumpy(actual[key]), expected[key])


def test_cuda_box_only_contacts_thin_layers_and_grazing_rays(cp, rates):
    axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for scale in (1e-9, 1.0, 1e9):
        boxes = [
            nav.PhotonBox(
                (0, 0, 1.5 * scale), axes, (scale, scale, 0.5 * scale), (("Si", 1 / scale),)
            ),
            nav.PhotonBox(
                (0, 0, 0.5 * scale), axes, (scale, scale, 0.5 * scale), (("C", 1 / scale),)
            ),
        ]
        nav.validate_photon_regions([], boxes)
        origins = scale * np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 2],
                [1, 0, 0],
                [-1, 0, 0.5],
                [0, 0, 0.5],
                [0, 0, 0.5],
            ]
        )
        directions = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, -1],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1e-14],
                [1, 0, -1e-14],
            ]
        )
        n = len(origins)
        uniforms = np.tile([0.25, np.nextafter(1.0, 0)], (n, 1))
        uniforms[:6, 0] = 0
        _compare(cp, origins, directions, np.full(n, 1e5), uniforms, [], boxes)


@pytest.mark.parametrize("footprint", [None, 4.0])
def test_cuda_boundaries_channels_gaps_and_reverse_flights(cp, rates, footprint):
    layers = [(0, 1, [("C", 0.25)]), (1, 2, [("Si", 0.5)]), (3, 5, [("C", 0.25)])]
    origins = np.array([[0, 0, 1]] * 3 + [[0, 0, 2], [0, 0, 0], [0, 0, -1], [0, 0, 6]])
    directions = np.array(
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, -1]]
    )
    uniforms = np.zeros((7, 2))
    _compare(
        cp,
        origins,
        directions,
        np.full(7, 1e5),
        uniforms,
        layers,
        width_ang=footprint,
        height_ang=footprint,
    )
    # Exact interface depth and all nonzero atom/channel intervals, including
    # zero-probability CDF slots and the largest permitted channel draw.
    draws = np.r_[(np.arange(8) + 0.5) / 8, 0.0, 0.25, 0.75, np.nextafter(1.0, 0)]
    uniforms = np.column_stack([np.full(draws.size, -np.expm1(-1)), draws])
    _compare(
        cp,
        np.zeros((draws.size, 3)),
        np.tile([0, 0, 1], (draws.size, 1)),
        np.full(draws.size, 1e5),
        uniforms,
        [(0, 1, [("C", 0.25)]), (1, 2, [("C", 1), ("Si", 1)])],
    )


@pytest.mark.parametrize("layers", [[], [(0, 1, [])], [(0, 1, [("C", 0)])]])
def test_cuda_vacuum_and_empty_batches(cp, rates, layers):
    _compare(cp, np.empty((0, 3)), np.empty((0, 3)), [], np.empty((0, 2)), layers)
    hit = _compare(cp, [[0, 0, 0.5]], [[1, 0, 0]], [1e5], [[0.5, 0]], layers)
    assert cp.isinf(hit["distance_ang"]).all()
    assert all(bool((hit[key] == -1).all()) for key in ("layer", "channel", "element"))


def test_cuda_epdl_oriented_boxes_and_batch_invariance(cp):
    rng = np.random.default_rng(341_700)
    n = 513
    origins = rng.uniform(-2e7, 2e7, (n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    energies, uniforms = np.geomspace(1e3, 1e7, n), rng.random((n, 2))
    layers = [(0, 1e7, [("C", 0.08), ("Pb", 0.02)]), (2e7, 3e7, [("Si", 0.05)])]
    a = np.pi / 7
    box = nav.PhotonBox(
        (4e7, 0, 0),
        ((np.cos(a), 0, np.sin(a)), (0, 1, 0), (-np.sin(a), 0, np.cos(a))),
        (5e6, 2e7, 1e7),
        (("Pb", 0.03),),
    )
    # Aim some rows at the remote box so this checks box collisions, not just misses.
    origins[:50] = [4e7, 0, -2e7]
    directions[:50] = [0, 0, 1]
    kwargs = {"width_ang": 4e7, "height_ang": 4e7}
    nav.validate_photon_regions(layers, [box], **kwargs)
    whole = _compare(cp, origins, directions, energies, uniforms, layers, [box], **kwargs)
    assert bool((whole["layer"] == 2).any())
    permutation = rng.permutation(n)
    reordered = scene_first_interactions_cuda(
        origins[permutation],
        directions[permutation],
        energies[permutation],
        uniforms[permutation],
        layers,
        [box],
        **kwargs,
    )
    chunks = [
        scene_first_interactions_cuda(
            origins[s], directions[s], energies[s], uniforms[s], layers, [box], **kwargs
        )
        for s in (slice(0, 7), slice(7, n))
    ]
    for key in whole:
        np.testing.assert_array_equal(
            cp.asnumpy(reordered[key]), cp.asnumpy(whole[key][permutation])
        )
        np.testing.assert_array_equal(
            cp.asnumpy(cp.concatenate([h[key] for h in chunks])), cp.asnumpy(whole[key])
        )
