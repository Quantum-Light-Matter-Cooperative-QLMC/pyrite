"""Kossel lines: two-beam Laue diffraction of characteristic radiation (#1176).

A characteristic line emitted isotropically inside a perfect crystal reaches a
detector along ``n`` through the photon mode that a plane wave from ``n``
excites (reciprocity, :mod:`~pyrite.montecarlo.spectrum.diffracted_brem`).
Near the Bragg energy ``E_B`` of a reflection ``g`` the mode has a transmitted
amplitude ``T_s(u, E)`` and a diffracted amplitude ``D_s(u, E)`` at the
remaining depth ``u = L - z``. A dipole at lattice site ``r_j`` emitting
isotropically couples to the mode field ``T e_0 + D e_h exp(-i g.r_j)`` (the
incoming wave runs along ``-n`` and diffracts with ``-g``), so a line of
unattenuated density ``Y L(E)`` [photons/sr, ``L`` the unit-area natural
Lorentzian] emitted at depth ``z`` reaches ``n`` as::

    Y L(E) (1/2) sum_s [|T_s|^2 + |D_s|^2 + 2 C_s Re(conj(T_s) D_s exp(-i g.r_j))]

Emitters are incoherent with each other, but each one's ``T``--``D`` cross
term survives: it is the X-ray standing-wave term, first order in ``chi_g``.
With ``D`` computed for a real coupling ``|chi_g|``, the site average gives
the emitter phase ``phi_Z = (chi_mode / |chi_g|) <exp(-i g.r_j)>_{j in Z}
exp(-W)`` per element ``Z`` (``chi_mode = conj(chi(-hkl))``, ``exp(-W)`` the
Debye--Waller coherent fraction; ``phi = 0`` off the lattice). Native
characteristic scoring already books the ``chi_g -> 0`` limit, plain
attenuation ``|T_0|^2 = exp(-k Im chi_0 u / n_s)``. This module returns the
Kossel term only::

    S(E) = Y L(E) sum_k w_k [(1/2) sum_s (|T_s|^2 + |D_s|^2) - |T_0|^2 + Re(phi_Z X)](u_k, E)

with ``X = (1/2) sum_s 2 C_s conj(T_s) D_s``. It vanishes off Bragg. Without
absorption the intensity part equals ``(1/2) sum_s (1 - Q_s/(k n_s)) |D_s|^2``
by flux conservation ``n_s |T|^2 + (Q_s/k) |D|^2 = n_s``; the cross term is
dispersive, and kinematically its band is ``-pi Im(phi)`` times the coupling.

Assumptions: those of ``diffracted-bremsstrahlung`` (two-beam Laue, perfect
single slab, ``chi_g chi_-g -> |chi_g|^2``, so no anomalous (Borrmann)
absorption: Kossel contrast is reliable for ``eps mu L / gamma << 1``, ``eps =
|chi''_g| / chi''_0`` with ``chi''_g`` the Fourier coefficient of ``Im chi(r)``;
not enforced), plus: unpolarized isotropic lines from atoms on the
crystal's own sites, isotropic Debye--Waller factor from the case ``B``; the
two-beam absorption ``Im chi_0`` is photoabsorption while
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

    ``lines`` is ``[(E_line_eV, fwhm_eV, weight, element), ...]`` with ``weight[k]``
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
            lines.append((float(energy), float(fwhm), column, el))
    return edges, lines


def kossel_term(depth_weight, edges, n_hat, g, s_hat, E_window, chi_g_abs, chi_0, phase=0.0):
    """Depth-weighted Kossel bracket on ``E_window`` for emitter phase ``phase``.

    ``sum_k w_k [(1/2) sum_s (|T_s|^2 + |D_s|^2) - |T_0|^2 + Re(phase X)](L - z_k, E)``.

    Validation: diffracted-characteristic-radiation
    """
    intensity, cross = _kossel_modes(edges, n_hat, g, s_hat, E_window, chi_g_abs, chi_0)
    return depth_weight @ (intensity + np.real(phase * cross))


def _kossel_modes(edges, n_hat, g, s_hat, E_window, chi_g_abs, chi_0):
    """Per depth bin: ``|T|^2 + |D|^2 - |T_0|^2`` and the complex cross term ``X``."""
    thickness = float(edges[-1])
    u = thickness - 0.5 * (edges[:-1] + edges[1:])
    T2, D2, X = laue_mode_intensities(E_window, n_hat, g, s_hat, chi_g_abs, chi_0, u, cross=True)
    k = np.asarray(E_window, dtype=float) / _HBARC_EV_ANG
    T0 = np.exp(-k[None, :] * np.imag(chi_0) * u[:, None] / float(n_hat @ s_hat))
    return T2 + D2 - T0, X


def emitter_phases(crystal, hkl, chi_mode, B_ang2=0.0):
    """``{element: phi_Z}``: the standing-wave phase of each emitting element at ``hkl``.

    ``phi_Z = (chi_mode / |chi_mode|) <exp(-2 pi i hkl.R_j)>_{j in Z} exp(-W)``,
    ``R_j`` the fractional basis sites of ``Z`` and ``exp(-W)`` the amplitude
    Debye--Waller factor of ``B_ang2``.

    Validation: diffracted-characteristic-radiation
    """
    from ...materials.crystal import CRYSTALS, debye_waller, reciprocal_g_vector

    info = CRYSTALS[crystal]
    if abs(chi_mode) == 0.0:
        return {}
    unit = chi_mode / abs(chi_mode)
    fraction = float(debye_waller(reciprocal_g_vector(hkl, info["lattice"])[1], B_ang2))
    sites: dict[str, list[complex]] = {}
    for element, R in info["basis"]:
        phase = np.exp(-2j * np.pi * float(np.dot(np.asarray(hkl, dtype=float), R)))
        sites.setdefault(str(element), []).append(complex(phase))
    return {el: complex(unit * np.mean(p) * fraction) for el, p in sites.items()}


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
    :func:`~pyrite.montecarlo.spectrum.diffracted_brem.diffracted_brem_spectrum`,
    optionally with a fourth item ``{element: phi_Z}`` (:func:`emitter_phases`);
    elements without a phase emit off the lattice (no standing-wave term).

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
    for g, chi_g_abs, chi_0, *rest in reflections:
        phases = rest[0] if rest else {}
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
        intensity, cross = _kossel_modes(edges, n, g, s, window, chi_g_abs, chi_0)
        S = np.zeros(window.size)
        for energy, fwhm, depth_weight, element in reach:
            modal = intensity + np.real(phases.get(element, 0.0) * cross)
            S += _lorentzian(window, energy, fwhm) * (depth_weight @ modal)
        out += deposit(window, S, grid, quadrature)
    return out
