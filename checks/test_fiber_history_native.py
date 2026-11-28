"""Geometry and normalization prerequisites for the native roll research seam."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fiber_history import (
    SECTOR_RAD,
    polarization_rows,
    primary_roll_rows,
    roll_vectors,
    validate_history_roll_request,
)


def test_roll_preserves_basal_and_magnitude():
    phi = np.linspace(0, SECTOR_RAD, 97, endpoint=False)
    g = np.array([2.0, -1.0, 3.0])
    c, s = np.cos(phi), np.sin(phi)
    actual = roll_vectors(g, c, s, xp=np)
    np.testing.assert_allclose(actual[:, 2], 3.0)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(g))
    basal = roll_vectors([0.0, 0.0, 3.0], c, s, xp=np)
    np.testing.assert_array_equal(basal, np.broadcast_to([0.0, 0.0, 3.0], basal.shape))


def test_roll_then_mosaic_not_reverse():
    g = np.array([1.0, 2.0, 3.0])
    phi, tilt = 0.4, 0.13
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    C, S = np.cos(tilt), np.sin(tilt)
    M = np.array([[1, 0, 0], [0, C, -S], [0, S, C]])
    got = roll_vectors(g, np.array([c]), np.array([s]), xp=np, mosaic=M)[0]
    np.testing.assert_allclose(got, M @ R @ g, rtol=1e-14, atol=1e-14)
    assert not np.allclose(got, R @ M @ g)


def test_polarization_matches_scalar():
    n = np.array([0.6, 0.0, 0.8])
    g = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], n])
    es, ep = polarization_rows(n, g, xp=np)
    for i, row in enumerate(g):
        normal = np.cross(n, row)
        if np.linalg.norm(normal) < 1e-12:
            normal = np.cross(n, [0.0, 1.0, 0.0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(es[i], normal, atol=1e-14)
        np.testing.assert_allclose(ep[i], np.cross(normal, n), atol=1e-14)
    np.testing.assert_allclose(es @ n, 0.0, atol=1e-14)
    np.testing.assert_allclose(ep @ n, 0.0, atol=1e-14)


def test_sparse_primary_assignment_is_partition_invariant():
    rolls = np.linspace(0, SECTOR_RAD, 100, endpoint=False)
    ids = np.array([91, 2, 91, 7, 2, 4, 91])
    for rows in (ids, ids[::2], ids[::-1], ids[:3], ids[3:]):
        angles, inverse = primary_roll_rows(rows, rolls, xp=np)
        np.testing.assert_array_equal(angles[inverse], rolls[rows])


@pytest.mark.parametrize("ids", [np.array([-1]), np.array([4]), np.array([1.5])])
def test_invalid_primary_ids(ids):
    with pytest.raises(ValueError, match="primary IDs"):
        primary_roll_rows(ids, np.zeros(4), xp=np)


def request(**overrides):
    options = dict(
        _fiber_history_rolls=np.zeros(4),
        crystal="hopg",
        coherent=False,
        line_quadrature="bin-mean",
        truncation_audit=None,
        temporal=None,
        coefficient_capture=None,
        layers=None,
        groove=None,
        sinc_cutoff=None,
        mosaic_sampling="quadrature",
        phase_retention=None,
        recip_miscut_rad=None,
        beam_uvw=None,
        surface_hkl=(0, 0, 1),
        hkl_list=[(1, 0, 2)],
        segments={"Ne": 4},
        electron_limit=None,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


def test_supported_request():
    validate_history_roll_request(request())


@pytest.mark.parametrize(
    "override",
    [
        {"coherent": True},
        {"truncation_audit": {}},
        {"mosaic_sampling": "history"},
        {"line_quadrature": "node"},
        {"recip_miscut_rad": (0.1, 0.0)},
        {"surface_hkl": (1, 0, 0)},
        {"hkl_list": [(0, 0, 2)]},
        {"segments": {"Ne": 4, "cone_split": {}}},
        {"_fiber_history_rolls": np.zeros(3)},
        {"_fiber_history_rolls": np.full(4, SECTOR_RAD)},
    ],
)
def test_unsupported_requests_fail_closed(override):
    with pytest.raises(
        ValueError,
        match=r"history fiber rolls|score basal reflections|expected one finite sector roll",
    ):
        validate_history_roll_request(request(**override))
