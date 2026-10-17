"""Uniaxial layered thermal rows: source values, window and projection limits."""

import numpy as np
import pytest

from pyrite.materials.channeling import (
    ChannelingDataError,
    UniaxialThermalSite,
    load_thermal_sites,
    planar_profile,
    thermal_b_factor,
)


@pytest.mark.parametrize(
    ("element", "u_basal", "u_c"),
    # Schoenfeld, Huang & Moss (1983) Table 3, 2H-MoS2 at room temperature.
    [("Mo", 0.0048, 0.0077), ("S", 0.0038, 0.0049)],
)
def test_mos2_uniaxial_rows_match_source(element, u_basal, u_c):
    site = load_thermal_sites()["mos2", element]
    assert isinstance(site, UniaxialThermalSite)
    assert (site.u_basal_ang2, site.u_c_ang2) == (u_basal, u_c)
    assert (site.temperature_K, site.tolerance_K) == (295.0, 10.0)
    assert site.doi == "10.1107/S0108768183002645"
    # Paper's B column: Mo 0.38/0.61, S 0.30/0.39 angstrom^2.
    assert site.b_factor_along(1.0, 295.0) == pytest.approx(8 * np.pi**2 * u_c)
    assert site.b_factor_along(0.0, 295.0) == pytest.approx(8 * np.pi**2 * u_basal)


def test_uniaxial_row_is_not_extrapolated_or_isotropic():
    site = load_thermal_sites()["mos2", "Mo"]
    with pytest.raises(ChannelingDataError, match="holds only"):
        site.b_factor_along(1.0, 77.0)
    with pytest.raises(ChannelingDataError, match="no isotropic B"):
        thermal_b_factor("mos2", "Mo", 295.0)
    with pytest.raises(ChannelingDataError, match="holds only"):
        planar_profile("mos2", (0, 0, 1), 310.0)


def test_uniaxial_projection_limits():
    # Isotropic limit: projection is direction-independent.
    iso = UniaxialThermalSite("x", "X", 1.0, 300.0, 0.0, 0.01, 0.01, "", "")
    for c2 in (0.0, 0.3, 1.0):
        assert iso.b_factor_along(c2, 300.0) == pytest.approx(8 * np.pi**2 * 0.01)
    # Off-axis normal interpolates n.U.n = u_b sin^2 + u_c cos^2.
    site = load_thermal_sites()["mos2", "S"]
    assert site.b_factor_along(0.25, 295.0) == pytest.approx(
        8 * np.pi**2 * (0.75 * 0.0038 + 0.25 * 0.0049)
    )
