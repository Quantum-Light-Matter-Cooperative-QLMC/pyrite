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


def _dbr_options(case) -> dict:
    """The case's continuum model, tables and brem cutoff for DBR sources."""
    from .. import runner

    return dict(
        model=case.get("bremsstrahlung_model", "eedl"),
        tables=runner._case_bremslib_tables(case),
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
    )


def _kossel_reflections(case, n_hat, grid) -> list:
    """``(g, |chi_g|, chi_0, {element: phi_Z})`` of each Laue reflection near ``grid``."""
    from ..spectrum.diffracted_brem import bragg_reflections
    from ..spectrum.diffracted_characteristic import emitter_phases

    return [
        (g, chi, chi_0, emitter_phases(case["crystal"], hkl, chi_mode, case.get("B_ang2", 0.0)))
        for g, chi, chi_0, (hkl, chi_mode) in bragg_reflections(case, n_hat, grid, detail=True)
    ]


def _yield_options(case) -> dict:
    return dict(
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        ionization_model=case.get("shell_ionization_model") or "eedl",
    )


def case_diffracted_brem(case, segments, E_grid, n_hat, Ne) -> np.ndarray:
    """DBR density of ``case`` on the line grid toward ``n_hat`` [photons/eV/sr/electron].

    Uses the case's continuum model (``bremsstrahlung_model``, its BremsLib
    tables) and ``E_cut_brem_keV`` on the brem electron population, and the
    case's ``line_quadrature``.

    Validation: diffracted-bremsstrahlung
    """
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
        Ne=int(segments["Ne"] if Ne is None else Ne),
        quadrature=case.get("line_quadrature", "node"),
        **_dbr_options(case),
    )


def case_diffracted_characteristic(case, segments, E_grid, n_hat, Ne) -> np.ndarray:
    """Kossel term of ``case``'s characteristic spectrum toward ``n_hat`` (#1176).

    Signed correction to ``spec_characteristic`` [photons/eV/sr/electron]: the
    two-beam diffracted gain, direct-beam depletion and standing-wave term of
    each line near a Laue Bragg energy, with each element's emitter phase on
    the crystal basis. Uses the case's ``shell_ionization_model``,
    ``E_cut_brem_keV`` and ``line_quadrature``; zero outside the claim.

    Validation: diffracted-characteristic-radiation
    """
    from ..spectrum.diffracted_characteristic import diffracted_characteristic_spectrum

    grid = np.asarray(E_grid, dtype=float)
    if not _in_claim(case, segments):
        return np.zeros(grid.size)
    reflections = _kossel_reflections(case, n_hat, grid)
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
        quadrature=case.get("line_quadrature", "node"),
        **_yield_options(case),
    )


class DiffractedDepthAccumulator:
    """DBR and Kossel source depth weights summed over whole-electron blocks.

    Both terms are linear in their 64-bin depth weights, so blocks add the
    weights (:meth:`add`) and the two-beam mode is solved once
    (:meth:`spectra`) rather than per block. Equals the joined
    :func:`case_diffracted_brem` / :func:`case_diffracted_characteristic`
    up to summation order.

    Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
    """

    def __init__(self, case, E_grid, n_hat):
        self.case, self.n_hat = case, n_hat
        self.grid = np.asarray(E_grid, dtype=float)
        in_claim = _in_claim(case, {"v_hat": True})  # segment directions checked per block
        self.reflections = _kossel_reflections(case, n_hat, self.grid) if in_claim else []
        self.dbr: list | None = None
        self.lines: dict = {}
        self.edges = None

    def add(self, segments, Ne) -> None:
        """Add one block's depth weights (``Ne`` electrons normalize every block)."""
        from ..spectrum.diffracted_brem import dbr_depth_weights
        from ..spectrum.diffracted_characteristic import line_depth_yields

        case = self.case
        if not self.reflections or not _in_claim(case, segments):
            return
        composition, thickness = case["composition"], float(case["thickness_ang"])
        dbr = dbr_depth_weights(
            segments,
            [r[:3] for r in self.reflections],
            self.n_hat,
            composition,
            thickness,
            Ne=int(Ne),
            **_dbr_options(case),
        )
        self.dbr = dbr if self.dbr is None else [a + b for a, b in zip(self.dbr, dbr, strict=True)]
        self.edges, lines = line_depth_yields(
            segments, composition, thickness, Ne=int(Ne), **_yield_options(case)
        )
        for energy, fwhm, weight, element in lines:
            key = (element, energy)
            if key in self.lines:
                self.lines[key][2] = self.lines[key][2] + weight
            else:
                self.lines[key] = [energy, fwhm, weight, element]

    def spectra(self) -> tuple[np.ndarray, np.ndarray]:
        """``(DBR, Kossel)`` on the line grid from the summed depth weights."""
        from ..spectrum.diffracted_brem import dbr_from_depth_weights
        from ..spectrum.diffracted_characteristic import kossel_from_yields

        zero = np.zeros(self.grid.size)
        if self.dbr is None:
            return zero, zero.copy()
        case = self.case
        quadrature = case.get("line_quadrature", "node")
        dbr = dbr_from_depth_weights(
            self.dbr,
            self.grid,
            [r[:3] for r in self.reflections],
            self.n_hat,
            float(case["thickness_ang"]),
            quadrature=quadrature,
        )
        lines = [tuple(line) for line in self.lines.values()]
        kossel = kossel_from_yields(
            self.edges, lines, self.grid, self.reflections, self.n_hat, quadrature=quadrature
        )
        return dbr, kossel
