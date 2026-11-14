"""Native diffracted bremsstrahlung (DBR) and Kossel lines of a case (#1166, #1176).

Every perfect single-slab crystal case adds DBR to its incoherent and coherent
line spectra (``spec``/``spec_coherent``), channeling on or off. Cases outside
the claim -- layer stacks, mosaic crystals, grooved faces, non-crystalline
targets -- add nothing. The same cases add the signed Kossel term (#1176) to
``spec_characteristic``.

Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
"""

from __future__ import annotations

import numpy as np


def _in_claim(case, segments) -> bool:
    """Perfect single-slab crystal with reflections and segment directions."""
    return not (
        not case.get("hkl_list")
        or case.get("abs_layers") is not None
        or case.get("layer_radiators") is not None
        or case.get("groove_spacing_ang") is not None
        or case.get("mosaic_fwhm_rad")
        or case.get("mosaic_mc_fwhm_rad")
        or segments.get("v_hat") is None
    )


def case_diffracted_brem(case, segments, E_grid, n_hat, Ne) -> np.ndarray:
    """DBR density of ``case`` on the line grid toward ``n_hat`` [photons/eV/sr/electron].

    Uses the case's continuum model (``bremsstrahlung_model``, its BremsLib
    tables) and ``E_cut_brem_keV`` on the brem electron population, and the
    case's ``line_quadrature``.

    Validation: diffracted-bremsstrahlung
    """
    from .. import runner
    from ..spectrum.diffracted_brem import bragg_reflections, diffracted_brem_spectrum

    grid = np.asarray(E_grid, dtype=float)
    out = np.zeros(grid.size)
    if not _in_claim(case, segments):
        return out
    reflections = bragg_reflections(case, n_hat, grid)
    if not reflections:
        return out
    return diffracted_brem_spectrum(
        segments,
        grid,
        reflections,
        n_hat,
        case["composition"],
        float(case["thickness_ang"]),
        model=case.get("bremsstrahlung_model", "eedl"),
        tables=runner._case_bremslib_tables(case),
        Ne=int(segments["Ne"] if Ne is None else Ne),
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        quadrature=case.get("line_quadrature", "node"),
    )


def case_diffracted_characteristic(case, segments, E_grid, n_hat, Ne) -> np.ndarray:
    """Kossel term of ``case``'s characteristic spectrum toward ``n_hat`` (#1176).

    Signed correction to ``spec_characteristic`` [photons/eV/sr/electron]: the
    two-beam diffracted gain and direct-beam depletion of each line near a
    Laue Bragg energy. Uses the case's ``shell_ionization_model``,
    ``E_cut_brem_keV`` and ``line_quadrature``; zero outside the claim.

    Validation: diffracted-characteristic-radiation
    """
    from ..spectrum.diffracted_brem import bragg_reflections
    from ..spectrum.diffracted_characteristic import diffracted_characteristic_spectrum

    grid = np.asarray(E_grid, dtype=float)
    if not _in_claim(case, segments):
        return np.zeros(grid.size)
    reflections = bragg_reflections(case, n_hat, grid)
    if not reflections:
        return np.zeros(grid.size)
    return diffracted_characteristic_spectrum(
        segments,
        grid,
        reflections,
        n_hat,
        case["composition"],
        float(case["thickness_ang"]),
        Ne=int(segments["Ne"] if Ne is None else Ne),
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        ionization_model=case.get("shell_ionization_model") or "eedl",
        quadrature=case.get("line_quadrature", "node"),
    )
