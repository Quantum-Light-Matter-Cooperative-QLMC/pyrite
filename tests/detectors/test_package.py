"""Package boundary for detector forward models."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "pyrite.detectors",
        "pyrite.detectors._si_sensor",
        "pyrite.detectors.timepix_response",
        "pyrite.detectors.eaglexo_response",
        "pyrite.detectors.grating",
    ],
)
def test_detector_forward_models_live_in_detectors_package(module_name: str) -> None:
    """Every direct detector model has a stable module beneath ``detectors``."""
    assert importlib.import_module(module_name)
