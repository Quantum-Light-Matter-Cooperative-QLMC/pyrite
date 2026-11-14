"""Diffracted bremsstrahlung (DBR): forward incoherent brem Bragg-diffracted into ``n`` (#1166).

A crystal reflection ``g`` diffracts photons travelling along
``Q = k n + g`` into the detector direction ``n`` near the Bragg energy
``E_B = -hbar c g^2 / (2 n.g)`` (``n.g < 0``). Incoherent bremsstrahlung
emitted along ``Q_hat`` (forward, inside the brem cone when ``Q_hat`` is near
the beam) therefore reaches a Bragg-angle detector as a narrow line at
``E_B``. Native PXR/CBS scoring omits it, even without channeling.

**Reciprocity.** The photon mode of ``n`` is the time-reversed two-beam Laue
mode: it enters through the exit face with unit transmitted and zero
diffracted amplitude and obeys, along the inward depth ``u`` from the exit
face, ``dP/du = i N P`` with::

    N = [[k chi_0 / (2 n_s), a], [b, q + k^2 chi_0 / (2 Q_s)]],  q = -det / (2 Q_s),
    a b = k^4 |chi_g|^2 C_s^2 / (4 k n_s Q_s),  |b|^2 = k^4 |chi_g|^2 C_s^2 / (4 Q_s^2)

(``det = g^2 + 2 k n.g``, subscript ``s`` the inward face-normal component,
``C_s`` the polarization factor, ``Im chi_0 > 0`` absorbing). Its diffracted
field amplitude at depth ``u`` is
``D_s(u) = b [exp(i lambda_1 u) - exp(i lambda_2 u)] / (lambda_1 - lambda_2)``,
``lambda_1,2`` the eigenvalues of ``N``. An incoherent source radiates into a
mode in proportion to the squared mode amplitude, so a segment emitting the
unattenuated brem density ``j(Q_hat, E)`` [photons/eV/sr] at depth ``z``
contributes ``j(Q_hat, E) (1/2) sum_s |D_s(L - z, E)|^2`` per steradian of
``n`` (``|Q| = k`` to first order in ``chi``). ``Im chi_0`` carries the
absorption of both beams, ``Re chi_0`` the refraction shift and ``ab`` the
Pendelloesung depletion. Kinematically (``ab, chi_0 -> 0``)
``int dE |D|^2 = |b|^2 2 pi u hbar c Q_s / |n.g|``, linear in the remaining
depth.

**Estimator.** The brem density per row is the track-length estimator of
:func:`~pyrite.montecarlo.spectrum.brem.mc_brem_spectrum` toward ``Q_hat`` at
``E_B``, without its escape factor (the mode carries it). Rows are
distributed uniformly over their depth range onto a depth grid; the line
shape is the depth-weighted ``|D|^2`` on a fine energy grid around ``E_B``,
then averaged onto the output bins or sampled at the nodes.

Assumptions: ``chi_g chi_-g`` replaced by ``|chi_g|^2`` (no anomalous,
Borrmann, absorption: negligible for thin crystals, conservative for thick
ones); two-beam Laue geometry (``n.s > 0``, ``Q.s > 0``), both angles
above ``3 sqrt(|chi_0|)``; perfect single-slab crystal (no mosaic, no layer
stacks, laterally infinite); first-order two-beam equations with ``chi_g``,
``chi_0`` flat across the line; unpolarized brem; brem density and direction
frozen at ``E_B`` and ``Q_hat(E_B)`` across the line (it is eV wide); the
extinction dip of the forward brem toward ``n`` itself (at most
``j(n) int |D|^2``) is not subtracted from the continuum. Routes that score
lines through ``_cxr_for_segments`` alone -- the adaptive ``"cxr"`` stopping
measure, the energy-grid bandwidth and convergence probes and line roulette --
do not see the DBR line.

Validation: diffracted-bremsstrahlung
"""

from __future__ import annotations

import numpy as np

from ..._backend import _to_cpu, xp

#: Native two-beam Laue model marker: DBR in the line spectra (#1166) and the
#: Kossel term of the characteristic spectrum (v2, #1176).
DIFFRACTED_BREM_MODEL = "two-beam-laue-dbr-kossel-v2"

_HBARC_EV_ANG = 1973.269804
_DEPTH_BINS = 64
_WIDTHS = 200.0
_MAX_NODES = 40000


def laue_mode_intensities(energy_eV, n_hat, g, s_hat, chi_g_abs, chi_0, depth_ang):
    """Polarization-mean ``(|T_s(u, E)|^2, |D_s(u, E)|^2)`` of the two-beam Laue mode.

    Both arrays are ``(n_depth, n_E)``. ``T`` is the transmitted amplitude
    (``P(0) = (1, 0)``), ``T(u) = exp(i lambda_2 u) [(lambda_2 - N_22)
    (exp(i x) - 1)/(lambda_1 - lambda_2) + exp(i x)]`` with
    ``x = (lambda_1 - lambda_2) u``; it reduces to ``exp(i k chi_0 u / 2 n_s)``
    for ``chi_g -> 0`` and obeys ``n_s |T|^2 + (Q_s/k) |D|^2 = n_s`` without
    absorption (#1176).

    Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
    """
    E = np.atleast_1d(np.asarray(energy_eV, dtype=float))
    u = np.atleast_1d(np.asarray(depth_ang, dtype=float))
    n = np.asarray(n_hat, float)
    g = np.asarray(g, float)
    k = E / _HBARC_EV_ANG
    Q = k[:, None] * n[None, :] + g[None, :]
    Qs = Q @ s_hat
    ns = float(n @ s_hat)
    q = -(g @ g + 2.0 * k * (n @ g)) / (2.0 * Qs)
    kap0 = k * chi_0 / (2.0 * ns)
    kaph = k**2 * chi_0 / (2.0 * Qs)
    es = np.cross(n, g)
    es /= np.linalg.norm(es)
    Qhat = Q / np.linalg.norm(Q, axis=1)[:, None]
    T2 = np.zeros((u.size, E.size))
    D2 = np.zeros((u.size, E.size))
    for e in (es, np.cross(es, n)):
        C = np.linalg.norm(e[None, :] - Qhat * (Qhat @ e)[:, None], axis=1)
        b2 = k**4 * chi_g_abs**2 * C**2 / (4.0 * Qs**2)
        ab = k**4 * chi_g_abs**2 * C**2 / (4.0 * k * ns * Qs)
        half = 0.5 * (q + kaph - kap0)
        root = np.sqrt(half * half + ab)
        # The expression is symmetric in lambda_1 <-> lambda_2; Im(root) >= 0 keeps
        # exp(i (lambda_1 - lambda_2) u) bounded in thick absorbing crystals.
        root = np.where(root.imag < 0.0, -root, root)
        lam2 = 0.5 * (kap0 + q + kaph) - root
        split = 2.0 * root  # lambda_1 - lambda_2
        x = split[None, :] * u[:, None]
        ratio = np.where(
            np.abs(x) > 1e-6,
            np.expm1(1j * x) / np.where(split == 0, 1.0, split)[None, :],
            1j * u[:, None] * (1.0 + 0.5j * x),
        )
        phase = np.exp(1j * lam2[None, :] * u[:, None])
        D2 += 0.5 * b2[None, :] * np.abs(phase * ratio) ** 2
        transmitted = phase * ((lam2 - q - kaph)[None, :] * ratio + np.exp(1j * x))
        T2 += 0.5 * np.abs(transmitted) ** 2
    return T2, D2


def laue_diffracted_intensity(energy_eV, n_hat, g, s_hat, chi_g_abs, chi_0, depth_ang):
    """Polarization-mean ``|D_s(u, E)|^2`` of the two-beam Laue mode, ``(n_depth, n_E)``.

    Validation: diffracted-bremsstrahlung
    """
    return laue_mode_intensities(energy_eV, n_hat, g, s_hat, chi_g_abs, chi_0, depth_ang)[1]


def _segment_density(segments, energy_eV, direction, composition, *, model, tables, Ne, E_cut_keV):
    """``(z0, z1, j)``: each scored row's depth range and unattenuated brem density.

    ``j`` [photons/eV/sr/incident electron] is the track-length density of
    :func:`~pyrite.montecarlo.spectrum.brem.mc_brem_spectrum` toward
    ``direction`` at the single photon energy ``energy_eV``, before escape.

    Validation: diffracted-bremsstrahlung
    """
    from ..transport import TRANSPORT_ELEMENTS
    from .brem import (
        _brem_dsigma_dk,
        _bremslib_cells,
        _evaluate_prepared_eedl,
    )
    from .brem_bremslib import stage_bremslib_elements
    from .brem_fallbacks import _prepare_mc_eedl_contexts
    from .channeling_flux import flux_scaled_length
    from .cxr import _clip_segments_to_cutoff

    segments = _clip_segments_to_cutoff(segments, E_cut_keV, composition)
    rows = xp.flatnonzero(xp.asarray(segments["elec_id"]) < Ne)
    L = xp.asarray(segments["L_ang"], dtype=float)[rows]
    field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    T = xp.asarray(segments[field], dtype=float)[rows]
    v = xp.asarray(segments["v_hat"], dtype=float)[rows]
    z = xp.asarray(segments["r_mid"], dtype=float)[rows, 2]
    half = 0.5 * xp.abs(L * v[:, 2])
    path_cm = flux_scaled_length(segments, L, rows) * 1e-8
    E = xp.asarray([float(energy_eV)])
    density = xp.zeros(int(rows.size))
    staged, covers = {}, {}
    if model == "bremslib":
        staged, covers = stage_bremslib_elements(composition, tables, T)
        cos_theta = v @ xp.asarray(direction, dtype=float)
    contexts = _prepare_mc_eedl_contexts(composition, T, E, tables if staged else None)
    for element, n_i in composition:
        context = contexts.get(element)
        directional = None
        if element in staged:
            state, dsig = _bremslib_cells(staged[element], T, cos_theta, E, 0.0, 1.0)
            if covers[element]:
                density += n_i * 1e24 * path_cm * dsig[:, 0]
                continue
            directional = (state, dsig)
        if context is not None:
            dsig = _evaluate_prepared_eedl(context.prepared, context.state, E)
            if not context.all_available:
                fallback = _brem_dsigma_dk(TRANSPORT_ELEMENTS[element]["Z"], T, E)
                dsig = xp.where(context.state.available[:, None], dsig, fallback)
        else:
            dsig = _brem_dsigma_dk(TRANSPORT_ELEMENTS[element]["Z"], T, E)
        if directional is not None:
            dsig = xp.where(directional[0].available[:, None], directional[1], dsig)
        density += n_i * 1e24 * path_cm * dsig[:, 0]
    density = density / (4.0 * np.pi) / Ne
    return (_to_cpu(z - half), _to_cpu(z + half), _to_cpu(density))


def _depth_histogram(z0, z1, weight, thickness, bins):
    """Row weights spread uniformly over ``[z0, z1]`` onto ``bins`` equal depth bins.

    ``weight`` is ``(n_row,)`` or ``(n_row, m)``; the result is ``(bins,)`` or
    ``(bins, m)``.

    Validation: diffracted-bremsstrahlung
    """
    edges = np.linspace(0.0, thickness, bins + 1)
    lo = np.clip(z0, 0.0, thickness)
    hi = np.clip(z1, 0.0, thickness)
    weight = np.asarray(weight, dtype=float)
    columns = weight.reshape(weight.shape[0], -1)
    out = np.zeros((bins, columns.shape[1]))
    for start in range(0, lo.size, 100_000):
        sl = slice(start, start + 100_000)
        a, c = lo[sl, None], hi[sl, None]
        overlap = np.clip(np.minimum(c, edges[None, 1:]) - np.maximum(a, edges[None, :-1]), 0, None)
        width = c - a
        point = (width[:, 0] <= 0.0)[:, None] & (a >= edges[None, :-1]) & (a < edges[None, 1:])
        share = np.where(width > 0.0, overlap / np.where(width > 0.0, width, 1.0), point)
        out += share.T @ columns[sl]
    return edges, (out[:, 0] if weight.ndim == 1 else out)


def bragg_window(thickness, n_hat, g, s_hat, E_B, chi_g_abs, chi_0, max_step=np.inf):
    """Fine energy nodes around the refraction-shifted Bragg energy of one reflection.

    The grid resolves ``2 pi / L`` in ``q`` (and ``max_step``) over ``+-200``
    kinematic widths plus 20 Pendelloesung widths, within +-1.5 keV and
    ``_MAX_NODES`` nodes.

    Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
    """
    k = E_B / _HBARC_EV_ANG
    Qs = float((k * n_hat + g) @ s_hat)
    dE_dq = _HBARC_EV_ANG * Qs / abs(float(n_hat @ g))
    coupling = k**2 * chi_g_abs / np.sqrt(4.0 * k * float(n_hat @ s_hat) * Qs)
    width_q = 2.0 * np.pi / thickness
    half_q = _WIDTHS * width_q + 20.0 * 2.0 * coupling
    span = min(half_q * dE_dq, 1500.0)
    step = min(width_q / 4.0 * dE_dq, 2.0 * span / 2000.0, max_step)
    step = max(step, 2.0 * span / _MAX_NODES)
    # Resonance q = Re(kappa_0 - kappa_h): the refraction shift of the line.
    shift = float(np.real(chi_0)) * (k / (2.0 * float(n_hat @ s_hat)) - k**2 / (2.0 * Qs)) * dE_dq
    return E_B + shift + np.arange(-span, span + step, step)


def reflection_line(depth_weight, edges, energy_grid, n_hat, g, s_hat, E_B, chi_g_abs, chi_0):
    """``(E_fine, S)`` of one reflection: the depth-weighted DBR line [photons/eV/sr].

    ``depth_weight[k]`` is the brem density emitted in depth bin ``k``; the
    line ``S(E) = sum_k depth_weight_k |D(L - z_k, E)|^2`` is sampled on
    :func:`bragg_window`.

    Validation: diffracted-bremsstrahlung
    """
    thickness = float(edges[-1])
    u = thickness - 0.5 * (edges[:-1] + edges[1:])
    E = bragg_window(thickness, n_hat, g, s_hat, E_B, chi_g_abs, chi_0)
    S = depth_weight @ laue_diffracted_intensity(E, n_hat, g, s_hat, chi_g_abs, chi_0, u)
    return E, S


def deposit(E_fine, S, grid, quadrature):
    """Bin-mean average (band-conserving) or node samples of ``S`` on ``grid``.

    Validation: diffracted-bremsstrahlung
    """
    grid = np.asarray(grid, dtype=float)
    if quadrature != "bin-mean":
        return np.interp(grid, E_fine, S, left=0.0, right=0.0)
    edges = np.concatenate(
        ([grid[0] - 0.5 * (grid[1] - grid[0])], 0.5 * (grid[1:] + grid[:-1]),
         [grid[-1] + 0.5 * (grid[-1] - grid[-2])])
    )  # fmt: skip
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(E_fine))))
    at = np.interp(edges, E_fine, cumulative)
    return np.diff(at) / np.diff(edges)


def bragg_reflections(case, n_hat, energy_grid, margin_eV=1500.0):
    """``[(g_sample, |chi_g|, chi_0), ...]`` of ``case`` with a Laue Bragg line in the grid.

    A reflection qualifies when ``n.g < 0`` puts ``E_B`` within ``margin_eV``
    of the grid and both ``n.s`` and ``Q_hat.s`` exceed ``3 sqrt(|chi_0|)``
    (Laue, outside total external reflection). ``chi_0`` is returned with
    ``Im > 0`` absorbing.

    Validation: diffracted-bremsstrahlung
    """
    from ...materials.crystal import (
        CRYSTALS,
        _orientation_R,
        chi_0,
        emission_coupling_tables,
        reciprocal_g_vector,
    )

    lattice = CRYSTALS[case["crystal"]]["lattice"]
    rotation = _orientation_R(
        lattice,
        case.get("beam_uvw"),
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case.get("surface_hkl"),
    )
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    n = np.asarray(n_hat, dtype=float)
    n = n / np.linalg.norm(n)
    s_hat = np.array([0.0, 0.0, 1.0])
    grid = np.asarray(energy_grid, dtype=float)
    out = []
    for hkl in (tuple(int(v) for v in h) for h in case["hkl_list"]):
        g = rotation @ np.asarray(reciprocal_g_vector(hkl, lattice)[0], dtype=float)
        if n @ g >= 0.0:
            continue
        E_B = -_HBARC_EV_ANG * float(g @ g) / (2.0 * float(n @ g))
        if not grid[0] - margin_eV <= E_B <= grid[-1] + margin_eV:
            continue
        Q = E_B / _HBARC_EV_ANG * n + g
        c0 = complex(chi_0(case["crystal"], E_B))
        floor = 3.0 * np.sqrt(abs(c0))
        if n @ s_hat <= floor or Q @ s_hat <= floor * np.linalg.norm(Q):
            continue
        cr, ci, _, _ = emission_coupling_tables(
            case["crystal"], [hkl], np.array([E_B]), case.get("B_ang2", 0.0), use_henke=True
        )
        chi = abs(complex(float(np.ravel(cr)[0]), float(np.ravel(ci)[0])))
        out.append((g, chi, complex(c0.real, abs(c0.imag))))
    return out


def diffracted_brem_spectrum(
    segments, energy_grid, reflections, n_hat, composition, thickness_ang, *,
    model, tables, Ne, E_cut_keV, quadrature="node", s_hat=(0.0, 0.0, 1.0),
):  # fmt: skip
    """DBR density on ``energy_grid`` toward ``n_hat`` [photons/eV/sr/incident electron].

    ``reflections`` is ``[(g_sample, chi_g_abs, chi_0), ...]`` with ``chi_0``
    in the absorbing ``Im > 0`` convention, each evaluated at its Bragg energy
    (see :func:`bragg_reflections`).

    Validation: diffracted-bremsstrahlung
    """
    grid = np.asarray(energy_grid, dtype=float)
    out = np.zeros(grid.size)
    n = np.asarray(n_hat, dtype=float)
    n = n / np.linalg.norm(n)
    s = np.asarray(s_hat, dtype=float)
    for g, chi_g_abs, chi_0 in reflections:
        E_B = -_HBARC_EV_ANG * float(g @ g) / (2.0 * float(n @ g))
        direction = E_B / _HBARC_EV_ANG * n + g
        direction /= np.linalg.norm(direction)
        z0, z1, j = _segment_density(
            segments, E_B, direction, composition,
            model=model, tables=tables, Ne=Ne, E_cut_keV=E_cut_keV,
        )  # fmt: skip
        if not np.any(j > 0.0):
            continue
        edges, weight = _depth_histogram(z0, z1, j, float(thickness_ang), _DEPTH_BINS)
        E, S = reflection_line(weight, edges, grid, n, g, s, E_B, chi_g_abs, chi_0)
        out += deposit(E, S, grid, quadrature)
    return out
