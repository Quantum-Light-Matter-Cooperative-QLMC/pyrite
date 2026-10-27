"""Issue #703: response numerical policy and legacy scientific identity."""

from __future__ import annotations

import pickle
from dataclasses import asdict, fields, replace

import pytest

from pyrite.config import Timepix3ResponseNumerics
from pyrite.detectors import Timepix3, Timepix3Response
from pyrite.instrument.acquisition import AcquisitionBatch as LegacyBatch
from pyrite.instrument.observation import response_payload
from pyrite.results import AcquisitionBatch, PixelCountResult


def test_grouped_policy_preserves_legacy_response_identity_and_pickle() -> None:
    numerics = Timepix3ResponseNumerics(dE_mc=200, dE_out=100, n_mc=128, seed=9)
    grouped = Timepix3Response(thickness_um=300, bias_v=100, numerics=numerics)
    legacy = Timepix3(thickness_um=300, bias_v=100, dE_mc=200, dE_out=100, n_mc=128, seed=9)
    assert grouped == legacy
    assert asdict(grouped) == asdict(legacy)
    assert response_payload(grouped) == response_payload(legacy)
    assert grouped.response_numerics == numerics
    assert "numerics" not in {field.name for field in fields(grouped)}
    assert pickle.loads(pickle.dumps(grouped)) == legacy
    assert replace(grouped, bias_v=90).bias_v == 90
    assert PixelCountResult is AcquisitionBatch is LegacyBatch


def test_policy_validation_and_conflicting_legacy_kwargs() -> None:
    with pytest.raises(ValueError, match="dE_mc"):
        Timepix3ResponseNumerics(dE_mc=0)
    with pytest.raises(TypeError, match="n_mc"):
        Timepix3ResponseNumerics(n_mc=True)
    with pytest.raises(ValueError, match="seed"):
        Timepix3ResponseNumerics(seed=-1)
    with pytest.raises(TypeError, match="Timepix3.numerics"):
        Timepix3(numerics=object())
    with pytest.raises(ValueError, match="conflicts"):
        Timepix3(n_mc=256, numerics=Timepix3ResponseNumerics(n_mc=128))


def test_catalog_flat_and_nested_response_rows_lower_identically() -> None:
    from pyrite.campaign.observation import _response_from_config
    from pyrite.materials._beam_detector_parse import _parse_observation_response
    from pyrite.materials._catalog_decode import _Errors

    flat = {"kind": "timepix3", "thickness_um": 300, "n_mc": 128, "seed": 8}
    nested = {"kind": "timepix3", "thickness_um": 300, "numerics": {"n_mc": 128, "seed": 8}}
    first, second = _Errors(), _Errors()
    one = _parse_observation_response(flat, "profile.response", first)
    two = _parse_observation_response(nested, "profile.response", second)
    assert not first.items and not second.items
    assert _response_from_config(one) == _response_from_config(two)
    assert response_payload(_response_from_config(one)) == response_payload(
        _response_from_config(two)
    )

    invalid = _Errors()
    _parse_observation_response(
        {"kind": "timepix3", "n_mc": 128, "numerics": {"n_mc": 128}},
        "profile.response",
        invalid,
    )
    assert any("conflicts" in item for item in invalid.items)


def test_policy_respects_pint_units() -> None:
    import pint

    ureg = pint.UnitRegistry()
    config = Timepix3ResponseNumerics(dE_mc=0.2 * ureg.keV, dE_out=100 * ureg.eV)
    assert config.dE_mc == pytest.approx(200.0)
    assert config.dE_out == pytest.approx(100.0)
    sensor = Timepix3(thickness_um=0.3 * ureg.mm, bias_v=100 * ureg.V)
    assert response_payload(sensor) == response_payload(Timepix3(thickness_um=300.0, bias_v=100.0))
    with pytest.raises(ValueError, match="energy"):
        Timepix3ResponseNumerics(dE_mc=1 * ureg.mm)
