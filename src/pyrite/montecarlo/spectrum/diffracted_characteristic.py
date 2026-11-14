"""Kossel lines: two-beam Laue diffraction of characteristic radiation (#1176).

A characteristic line emitted isotropically inside a perfect crystal reaches a
detector along ``n`` through the photon mode that a plane wave from ``n``
excites (reciprocity, :mod:`~pyrite.montecarlo.spectrum.diffracted_brem`).
Near the Bragg energy ``E_B`` of a reflection ``g`` the mode has a transmitted
amplitude ``T_s(u, E)`` and a diffracted amplitude ``D_s(u, E)`` at the
remaining depth ``u = L - z``. A line of unattenuated density
``Y L(E)`` [photons/sr, ``L`` the unit-area natural Lorentzian] emitted at
depth ``z`` reaches ``n`` as::

    Y L(E) (1/2) sum_s [|T_s(L - z, E)|^2 + |D_s(L - z, E)|^2]

because the source is isotropic (``j(Q_hat) = j(n)``) and incoherent (no
``T``--``D`` cross term). Native characteristic scoring already books the
``chi_g -> 0`` limit, plain attenuation ``|T_0|^2 = exp(-k Im chi_0 u / n_s)``.
This module returns the Kossel term only::

    S(E) = Y L(E) sum_k w_k (1/2) sum_s [|T_s|^2 + |D_s|^2 - |T_0|^2](u_k, E)

which vanishes off Bragg and, without absorption, equals
``Y L(E) (1/2) sum_s (1 - Q_s/(k n_s)) |D_s|^2`` by flux conservation
``n_s |T|^2 + (Q_s/k) |D|^2 = n_s``: symmetric Laue redistributes no flux.

Assumptions: those of ``diffracted-bremsstrahlung`` (two-beam Laue, perfect
single slab, ``chi_g chi_-g -> |chi_g|^2``, so no anomalous (Borrmann)
absorption: Kossel contrast is reliable for ``mu L / n_s <~ 1``), plus: unpolarized
isotropic lines; the two-beam absorption ``Im chi_0`` is photoabsorption while
native escape uses the total ``mu`` (the subtraction uses the same
``chi_0``, so the term still vanishes off Bragg); a line enters a reflection
only within 100 FWHM of that reflection's Bragg window. The same direct-beam
depletion of diffracted bremsstrahlung is not scored: at the #1166 mirror
rays it is 6e-6 to 1e-5 of the DBR gain.

Validation: diffracted-characteristic-radiation
"""

from __future__ import annotations

import numpy as np

from ..._backend import REAL, _to_cpu, xp
from .diffracted_brem import (
    _DEPTH_BINS,
    _HBARC_EV_ANG,
    _depth_histogram,
    bragg_window,
    deposit,
    laue_mode_intensities,
)

#: Lines farther than this many FWHM outside a Bragg window are skipped.
_LINE_REACH_FWHM = 100.0


def line_depth_yields(
    segments, composition, thickness_ang, *, Ne, E_cut_keV=None, ionization_model="eedl"
):
    """``(edges, lines)``: unattenuated characteristic yield per depth bin and line.

    ``lines`` is ``[(E_line_eV, fwhm_eV, weight), ...]`` with ``weight[k]``
    the line's photons/sr/incident electron emitted in depth bin ``k``,
    before escape. Rows, cutoffs, cross sections, cascade yields and channeling
    flux weights are those of
    :func:`~pyrite.montecarlo.spectrum.characteristic.mc_characteristic_spectrum`
    for a single slab.

    Validation: diffracted-characteristic-radiation
    """
    from ...materials.attenuation import _normalize_composition
    from . import _characteristic_fused
    from .characteristic import (
        _cascade_line_yields,
        _characteristic_transport_cutoff_keV,
        _device_shell_tables,
        _interpolate_shell_cross_sections,
        _shell_flux_weight,
        load_characteristic_cross_sections,
    )
    from .cxr import _clip_segments_to_cutoff

    comp = _normalize_composition(None, None, composition)
    tables = {
        el: load_characteristic_cross_sections(el, ionization_model=ionization_model)
        for el, _ in comp
    }
    cutoff = max(table.recommended_cutoff_eV for table in tables.values())
    segments = _clip_segments_to_cutoff(
        segments, _characteristic_transport_cutoff_keV(E_cut_keV), comp
    )
    rows = xp.flatnonzero(xp.asarray(segments["elec_id"]) < int(Ne))
    L = xp.asarray(segments["L_ang"], dtype=REAL)[rows]
    field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    T = xp.asarray(segments[field], dtype=REAL)[rows]
    z = _to_cpu(xp.asarray(segments["r_mid"], dtype=float)[rows, 2])
    half = 0.5 * np.abs(_to_cpu(L) * _to_cpu(xp.asarray(segments["v_hat"], dtype=float)[rows, 2]))
    lines = []
    edges = np.linspace(0.0, float(thickness_ang), _DEPTH_BINS + 1)
    for el, number_density in comp:
        table = tables[el]
        _transfer, response = _cascade_line_yields(table, cutoff)
        keep = (table.line_energy_eV > cutoff) & np.any(response > 0.0, axis=0)
        if not np.any(keep) or int(rows.size) == 0:
            continue
        shells = _device_shell_tables(table)
        sigma = _interpolate_shell_cross_sections(
            table, T, shells, _characteristic_fused.pack_shell_tables(shells)
        )
        weight = _shell_flux_weight(segments, table, rows)
        if weight is not None:
            sigma = sigma * weight
        per_shell = _to_cpu(sigma * (number_density * 1.0e24 * L * 1.0e-8)[:, None])
        edges, by_shell = _depth_histogram(
            z - half, z + half, per_shell, thickness_ang, _DEPTH_BINS
        )
        by_line = by_shell @ response[:, keep] / (4.0 * np.pi * int(Ne))
        for energy, fwhm, column in zip(
            table.line_energy_eV[keep], table.cxr_fwhm_eV[keep], by_line.T, strict=True
        ):
            lines.append((float(energy), float(fwhm), column))
    return edges, lines


def kossel_term(depth_weight, edges, n_hat, g, s_hat, E_window, chi_g_abs, chi_0):
    """``sum_k w_k (1/2) sum_s [|T_s|^2 + |D_s|^2 - |T_0|^2](L - z_k, E)`` on ``E_window``.

    Validation: diffracted-characteristic-radiation
    """
    thickness = float(edges[-1])
    u = thickness - 0.5 * (edges[:-1] + edges[1:])
    T2, D2 = laue_mode_intensities(E_window, n_hat, g, s_hat, chi_g_abs, chi_0, u)
    k = np.asarray(E_window, dtype=float) / _HBARC_EV_ANG
    T0 = np.exp(-k[None, :] * np.imag(chi_0) * u[:, None] / float(n_hat @ s_hat))
    return depth_weight @ (T2 + D2 - T0)


def _lorentzian(E, centre, fwhm):
    """Unit-area Lorentzian density [1/eV]."""
    return (0.5 * fwhm / np.pi) / ((E - centre) ** 2 + 0.25 * fwhm**2)


def diffracted_characteristic_spectrum(
    segments, energy_grid, reflections, n_hat, composition, thickness_ang, *,
    Ne, E_cut_keV=None, ionization_model="eedl", quadrature="node", s_hat=(0.0, 0.0, 1.0),
):  # fmt: skip
    """Kossel term of the characteristic spectrum toward ``n_hat`` [photons/eV/sr/electron].

    Signed: the diffracted gain and the direct-beam depletion of each line
    near each reflection's Bragg energy, to be added to the native
    characteristic spectrum. ``reflections`` as in
    :func:`~pyrite.montecarlo.spectrum.diffracted_brem.diffracted_brem_spectrum`.

    Validation: diffracted-characteristic-radiation
    """
    grid = np.asarray(energy_grid, dtype=float)
    out = np.zeros(grid.size)
    if not reflections:
        return out
    n = np.asarray(n_hat, dtype=float)
    n = n / np.linalg.norm(n)
    s = np.asarray(s_hat, dtype=float)
    edges, lines = line_depth_yields(
        segments, composition, thickness_ang,
        Ne=Ne, E_cut_keV=E_cut_keV, ionization_model=ionization_model,
    )  # fmt: skip
    for g, chi_g_abs, chi_0 in reflections:
        E_B = -_HBARC_EV_ANG * float(g @ g) / (2.0 * float(n @ g))
        near = [
            line
            for line in lines
            if abs(line[0] - E_B) <= 1500.0 + _LINE_REACH_FWHM * line[1] and np.any(line[2] > 0)
        ]
        if not near:
            continue
        window = bragg_window(
            float(edges[-1]), n, g, s, E_B, chi_g_abs, chi_0,
            max_step=0.125 * min(line[1] for line in near),
        )  # fmt: skip
        reach = [
            line
            for line in near
            if window[0] - _LINE_REACH_FWHM * line[1]
            <= line[0]
            <= window[-1] + _LINE_REACH_FWHM * line[1]
        ]
        if not reach:
            continue
        modal = kossel_term(np.eye(len(edges) - 1), edges, n, g, s, window, chi_g_abs, chi_0)
        S = np.zeros(window.size)
        for energy, fwhm, depth_weight in reach:
            S += _lorentzian(window, energy, fwhm) * (depth_weight @ modal)
        out += deposit(window, S, grid, quadrature)
    return out
