"""B1: numeric-only detector-response grids refuse Pint quantities (#755)."""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.detectors import eaglexo_response, timepix_response
from pyrite.units import ureg

GRID_EV = np.linspace(1_000.0, 10_000.0, 32)


@pytest.mark.parametrize(
    "call",
    [
        lambda q: timepix_response.TimepixResponse(q),
        lambda q: timepix_response.get_response(q),
        lambda q: timepix_response.absorption_efficiency(q),
        lambda q: timepix_response.energy_fwhm_eV(q),
        lambda q: timepix_response.poisson_counts(q, np.ones(q.shape), 1.0),
        lambda q: eaglexo_response.EagleResponse(q),
        lambda q: eaglexo_response.get_response(q),
        lambda q: eaglexo_response.qe(q),
        lambda q: eaglexo_response.qe_absorption_model(q),
        lambda q: eaglexo_response.energy_fwhm_eV(q),
        lambda q: eaglexo_response.poisson_counts(q, np.ones(q.shape), 1.0),
    ],
)
def test_energy_grids_refuse_quantities_instead_of_stripping(call) -> None:
    # A keV quantity was previously read as eV magnitudes with only a warning.
    with pytest.raises(TypeError, match=r"E_.*eV is numeric-only.*'eV'"):
        call(GRID_EV / 1_000.0 * ureg.keV)


@pytest.mark.parametrize("name", ["dE_mc", "dE_out", "thickness_um", "bias_v"])
def test_timepix_response_settings_refuse_quantities(name) -> None:
    value = {"dE_mc": 0.05 * ureg.keV, "dE_out": 25 * ureg.eV}.get(name, 1 * ureg.dimensionless)
    with pytest.raises(TypeError, match=f"{name} is numeric-only"):
        timepix_response.get_response(GRID_EV, **{name: value})


def test_response_detection_efficiency_refuses_quantity_energies() -> None:
    response = eaglexo_response.get_response(GRID_EV)
    with pytest.raises(TypeError, match="E_eV is numeric-only"):
        response.detection_efficiency(GRID_EV * ureg.eV)
    np.testing.assert_array_equal(response.detection_efficiency(GRID_EV), response.qe)
