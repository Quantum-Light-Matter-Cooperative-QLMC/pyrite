"""Internal bridge from source-backed uniaxial ADPs to scalar line couplings.

A common site covariance permits one effective B per reciprocal vector. This
is not a general atom-dependent structure factor, nor a change to the catalog
or the production reflection selector.
"""

from __future__ import annotations

import numpy as np


def common_uniaxial_b(crystal: str, hkl, temperature_K: float) -> float:
    """Project a common uniaxial covariance onto a nonzero reflection [A^2].

    Uses the existing channeling thermal records, not the catalog scalar.
    Every element must have a uniaxial row with identical covariance at the
    requested temperature; otherwise a scalar B cannot represent the sites.
    For the supported hexagonal crystals the reciprocal (001) is parallel to
    the direct c axis. Missing data and unsupported temperatures are errors.
    """
    from .channeling.thermal import ChannelingDataError, UniaxialThermalSite, thermal_site
    from .crystal import CRYSTALS, reciprocal_g_vector

    info = CRYSTALS[crystal]
    lattice = info["lattice"]
    g, norm = reciprocal_g_vector(hkl, lattice)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("a finite nonzero reciprocal vector is required")
    axis, axis_norm = reciprocal_g_vector((0, 0, 1), lattice)
    cos2_c = float((g @ axis / (norm * axis_norm)) ** 2)
    if not np.isfinite(temperature_K):
        raise ValueError("temperature_K must be finite")
    sites = [thermal_site(crystal, el) for el in sorted({el for el, _ in info["basis"]})]
    if not sites or not all(isinstance(site, UniaxialThermalSite) for site in sites):
        raise ChannelingDataError(f"{crystal}: common uniaxial thermal data required")
    # Compare the full covariance, not just its projection for this reflection.
    covariances = [
        site.covariance(axis, temperature_K)
        for site in sites
        if isinstance(site, UniaxialThermalSite)
    ]
    if any(not np.array_equal(cov, covariances[0]) for cov in covariances[1:]):
        raise ChannelingDataError(
            f"{crystal}: site-dependent thermal covariance requires site weights"
        )
    site = sites[0]
    assert isinstance(site, UniaxialThermalSite)
    return site.b_factor_along(cos2_c, temperature_K)
