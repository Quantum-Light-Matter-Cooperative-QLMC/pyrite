"""Self-contained optical records preserve support and reject stale identities."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.optical import OpticalResponse, gold_rakic_response


def test_json_round_trip_preserves_response_identity_without_source_access(monkeypatch):
    response = gold_rakic_response()
    record = json.loads(json.dumps(response.to_record(), allow_nan=False))

    def forbidden(*args, **kwargs):
        pytest.fail("restoring a self-contained record must not read source files")

    monkeypatch.setattr("pathlib.Path.read_bytes", forbidden)
    restored = OpticalResponse.from_record(record)
    assert restored.digest == response.digest
    assert restored.bands_eV == response.bands_eV
    energy = np.geomspace(*response.bands_eV[0], 31)
    np.testing.assert_array_equal(restored.epsilon(energy), response.epsilon(energy))
    for (nodes, eps), (original_nodes, original_eps) in zip(
        restored.segments, response.segments, strict=True
    ):
        np.testing.assert_array_equal(nodes, original_nodes)
        np.testing.assert_array_equal(eps, original_eps)
        with pytest.raises(ValueError):
            eps.setflags(write=True)


@pytest.mark.parametrize("temperature", [None, 300, 300.0])
def test_record_preserves_disjoint_bands_and_signed_zero(temperature):
    eps = np.empty(2, complex)
    eps.real, eps.imag = [-0.0, 1.0], [-0.0, 0.0]
    response = replace(
        gold_rakic_response(),
        temperature_K=temperature,
        segments=((np.array([1.0, 2.0]), eps), (np.array([3.0, 4.0]), np.ones(2, complex))),
    )
    restored = OpticalResponse.from_record(json.loads(json.dumps(response.to_record())))
    assert restored.digest == response.digest
    assert restored.bands_eV == ((1.0, 2.0), (3.0, 4.0))
    with pytest.raises(ValueError, match="gap"):
        restored.epsilon(2.5)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "schema", "future", "schema"),
        (None, "interpolation", "linear-index", "interpolation"),
        (None, "digest", "0" * 64, "digest"),
        (None, "segments", [], "nonempty"),
        ("metadata", "source", "changed source", "digest"),
        ("metadata", "temperature_K", True, "temperature"),
        ("metadata", "temperature_K", float("nan"), "temperature"),
        ("metadata", "temperature_K", 10**1000, "temperature"),
        ("metadata", "medium", ["Au"], "string"),
        ("metadata", "symmetry", "tensor", "anisotropic"),
        ("metadata", "convention", "exp(+i omega t)", "convention"),
        ("segment", "energy_eV", [1.0, 2.0], "match"),
        ("segment", "epsilon_real", [1.0, "2.0"], "JSON numbers"),
        ("segment", "epsilon_real", [1.0, True], "JSON numbers"),
        ("segment", "epsilon_real", [[1.0], [2.0]], "JSON numbers"),
        ("segment", "epsilon_real", [1.0, float("inf")], "finite"),
        ("segment", "epsilon_real", [1.0, 10**1000], "finite"),
    ],
)
def test_records_fail_closed(section, key, value, message):
    record = gold_rakic_response().to_record()
    target = (
        record
        if section is None
        else (record["segments"][0] if section == "segment" else record[section])
    )
    target[key] = value
    with pytest.raises(ValueError, match=message):
        OpticalResponse.from_record(record)


@pytest.mark.parametrize("section", [None, "metadata", "segment"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_records_reject_unknown_or_missing_fields(section, mutation):
    record = deepcopy(gold_rakic_response().to_record())
    target = (
        record
        if section is None
        else (record["segments"][0] if section == "segment" else record[section])
    )
    if mutation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unknown"] = None
    with pytest.raises(ValueError, match="fields"):
        OpticalResponse.from_record(record)


def test_record_is_detached_from_immutable_response():
    response = gold_rakic_response()
    before = response.digest
    record = response.to_record()
    record["metadata"]["source"] = "edited"
    record["segments"][0]["epsilon_real"][0] = 0.0
    assert response.digest == before
    with pytest.raises(ValueError, match="digest"):
        OpticalResponse.from_record(record)


@pytest.mark.parametrize("column", ["energy_eV", "epsilon_real", "epsilon_imag"])
def test_stale_digest_rejects_changed_numeric_content(column):
    record = gold_rakic_response().to_record()
    record["segments"][0][column][0] += 1e-6
    with pytest.raises(ValueError, match="digest"):
        OpticalResponse.from_record(record)


@pytest.mark.parametrize("record", [None, [], {"schema": "pyrite.optical-response-v1"}])
def test_record_requires_complete_mapping(record):
    with pytest.raises(ValueError, match="fields"):
        OpticalResponse.from_record(record)
