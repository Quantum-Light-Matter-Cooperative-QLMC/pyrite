"""Pinned transparent-medium inputs; no production or absorption validation."""

import json
from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HC_EV_ANG
from pyrite.materials.optical import OpticalResponse, fused_silica_franta_response


def test_silica_source_values_and_provenance():
    response = fused_silica_franta_response()
    assert response.medium == "fused silica"
    assert response.dataset_id == "refractiveindex.info/main/SiO2/nk/Franta"
    assert (
        response.source_sha256 == "89e3065c48040e9347fb8707371d5528ab45ff4e05be5fd6644658b588ce458e"
    )
    assert "c5c2f188e848453def5970e347399d653df2ffc2" in response.source
    assert "10.1117/12.2227580" in response.reference
    assert "fitted" in response.sample_state
    assert "not a measured absorption bound" in response.sample_state
    assert response.temperature_K is None
    assert response.license == "CC0-1.0"
    # Independently transcribed rows from the pinned CC0 Franta record:
    # wavelength [um], n, k. The broad table includes absorbing material.
    wavelength = np.array([0.024797, 0.500495])
    n = np.array([0.93894898518, 1.46244814336])
    k = np.array([0.066160890781, 0.0])
    expected = n * n - k * k + 2j * n * k
    np.testing.assert_allclose(
        response.epsilon(HC_EV_ANG / (1e4 * wavelength)), expected, rtol=2e-15
    )
    np.testing.assert_allclose(response.bands_eV, [[HC_EV_ANG / (125.141e4), HC_EV_ANG / 247.97]])


def test_silica_portable_response_preserves_data_and_full_band(monkeypatch):
    response = fused_silica_franta_response()
    record = json.loads(json.dumps(response.to_record(), allow_nan=False))

    def forbidden(*args, **kwargs):
        pytest.fail("portable optical records must not reload source data")

    monkeypatch.setattr("pathlib.Path.read_bytes", forbidden)
    restored = OpticalResponse.from_record(record)
    assert restored.digest == response.digest
    for band in (response.bands_eV[0], (1.5, 3.0)):
        assert restored.require_band(*band) is None
    np.testing.assert_array_equal(
        restored.epsilon([1.5, 2.0, 3.0]), response.epsilon([1.5, 2.0, 3.0])
    )


def test_full_band_cannot_bridge_gap_even_with_supported_endpoints():
    response = replace(
        fused_silica_franta_response(),
        segments=(
            (np.array([1.0, 2.0]), np.ones(2, complex)),
            (np.array([3.0, 4.0]), np.ones(2, complex)),
        ),
    )
    np.testing.assert_array_equal(response.epsilon([1.0, 4.0]), [1.0, 1.0])
    with pytest.raises(ValueError, match="fused silica.*gap"):
        response.require_band(1.0, 4.0)
    response.require_band(1.0, 2.0)
    response.require_band(3.0, 4.0)


@pytest.mark.parametrize("band", [(0, 1), (-1, 1), (1, 1), (2, 1), (np.nan, 2), (1, np.inf)])
def test_full_band_requires_finite_positive_ordered_endpoints(band):
    with pytest.raises(ValueError, match="finite eV"):
        fused_silica_franta_response().require_band(*band)


@pytest.mark.parametrize("band", [(0.001, 2), (2, 100)])
def test_silica_full_band_refuses_extrapolation(band):
    with pytest.raises(ValueError, match="unsupported"):
        fused_silica_franta_response().require_band(*band)


def test_silica_provider_rejects_corrupt_packaged_source(monkeypatch):
    monkeypatch.setattr("pathlib.Path.read_bytes", lambda *args: b"changed optical data")
    with pytest.raises(ValueError, match="SHA-256"):
        fused_silica_franta_response()
