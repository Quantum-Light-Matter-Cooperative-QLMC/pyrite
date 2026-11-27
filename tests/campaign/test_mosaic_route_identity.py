"""The automatic mosaic route (#1207) moves only the digests whose output changes."""

import dataclasses

from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.mosaic_route import MOSAIC_ROUTE_DEFAULT_MODEL
from pyrite.campaign.profiles import dataset_identity


def _digest(settings=None, **overrides):
    sweep = material_sweep("hopg", **overrides)
    identity = dataset_identity(
        "hopg", settings or default_settings(), sweep, bremsstrahlung_model="eedl"
    )
    return identity["parameter_sha256"], identity["resolved_parameters"]["sweep"]


def test_perfect_crystal_auto_route_keeps_the_historical_analytic_digest():
    auto, payload = _digest()
    analytic, _ = _digest(mosaic_route="analytic", mosaic_nodes=5)
    assert auto == analytic
    assert payload["mosaic_route"] == "analytic" and payload["mosaic_nodes"] == 5
    assert "mosaic_route_model" not in payload


def test_incoherent_mosaic_auto_route_hashes_the_sampled_route_and_marker():
    auto, payload = _digest(mosaic=True)
    analytic, _ = _digest(mosaic=True, mosaic_route="analytic", mosaic_nodes=5)
    explicit, _ = _digest(mosaic=True, mosaic_route="mc-sampled", mosaic_nodes=256)
    assert len({auto, analytic, explicit}) == 3
    assert payload["mosaic_route"] == "mc-sampled" and payload["mosaic_nodes"] == 256
    assert payload["mosaic_route_model"] == MOSAIC_ROUTE_DEFAULT_MODEL


def test_coherent_mosaic_auto_route_keeps_the_analytic_digest():
    settings = dataclasses.replace(default_settings(), emission="coherent")
    assert settings.coherent_emission
    auto, _ = _digest(settings, mosaic=True)
    analytic, _ = _digest(settings, mosaic=True, mosaic_route="analytic", mosaic_nodes=5)
    assert auto == analytic
