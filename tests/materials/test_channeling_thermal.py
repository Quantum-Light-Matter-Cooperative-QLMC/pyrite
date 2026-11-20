"""Channeling thermal rows: uniaxial layered source values, window and projection limits; GaAs rows."""

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


def test_graphite_uniaxial_row_matches_source_and_flags_assumed_temperature():
    site = load_thermal_sites()["hopg", "C"]
    assert isinstance(site, UniaxialThermalSite)
    # Trucano & Chen (1975) Table 2: U11 = 0.0031 (fixed from Ludsteck 1972),
    # U33 = 0.016(2) (C1) and 0.017(2) (C2); the row is the equal-multiplicity mean.
    assert site.u_basal_ang2 == 0.0031
    assert site.u_c_ang2 == pytest.approx((0.016 + 0.017) / 2)
    assert site.doi == "10.1038/258136a0"
    # The source states no temperature: 300 K is an assumption, and the row says so.
    assert (site.temperature_K, site.tolerance_K) == (300.0, 10.0)
    assert "assumed" in site.source and "not stated" in site.source
    # Basal projection reproduces the catalog's B33 = 8 pi^2 * 0.0165 = 1.30 A^2.
    assert site.b_factor_along(1.0, 300.0) == pytest.approx(1.30, abs=0.005)
    with pytest.raises(ChannelingDataError, match="holds only"):
        site.b_factor_along(1.0, 77.0)
    with pytest.raises(ChannelingDataError, match="holds only"):
        planar_profile("hopg", (0, 0, 1), 77.0)


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


def test_gaas_rows_reproduce_the_measured_room_temperature_displacements():
    # Schowalter et al. (2009) Table 6 (LDA) rows; Stahn et al. (1998) measured
    # <u^2> at 287 K: Ga 0.00844, As 0.00716 angstrom^2 (quoted in Schowalter
    # Sec. 4). Linear interpolation between the 200/400 K rows stays within 2%.
    for element, u2 in (("Ga", 0.00844), ("As", 0.00716)):
        b = thermal_b_factor("gaas", element, 287.0)
        assert b == pytest.approx(8.0 * np.pi**2 * u2, rel=0.02), element
    assert thermal_b_factor("gaas", "Ga", 300.0) > thermal_b_factor("gaas", "As", 300.0)
