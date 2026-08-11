"""Bremsstrahlung spectrum and external-background loading."""

import numpy as np

from ...materials.attenuation import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _normalize_composition,
)
from ...materials.crystal import ALPHA_FS
from .._backend import REAL, _to_cpu, xp
from ..groove import escape_distance_ang
from ..transport import TRANSPORT_ELEMENTS
from .lines import (
    _clip_segments_to_cutoff,
    _escape_length,
    _observation_direction,
    _segment_escape_distance,
    _validate_groove_escape_direction,
)

_USE_JIT_BREM_REDUCTION = True

R_E_CM2 = 7.9407877e-26  # classical electron radius squared [cm^2]
_BREM_MC2_KEV = 510.99895  # electron rest energy [keV]


def _brem_incident_state(T_keV):
    """Return incident momentum ``p/(m_e c)`` and beta for segment energies.

    These depend only on the emitting segment energy, not on photon energy or
    composition element, so the CUDA reduction path computes them once per
    spectrum instead of once per ``(segment, energy-block, element)``.
    """
    T = xp.asarray(T_keV, dtype=REAL)
    p_i = xp.sqrt(T * (T + 2.0 * _BREM_MC2_KEV)) / _BREM_MC2_KEV
    beta_i = p_i / (1.0 + T / _BREM_MC2_KEV)
    return p_i, beta_i


def _brem_incident_prefactor_core(L_ang, p_i, beta_i, Z, density_cm3):
    """Energy-independent part of ``n L dsigma/dk`` for one element.

    The remaining CUDA-cell work depends on the final-state momentum/beta and
    therefore still depends on photon energy. Hoisting this factor removes the
    incident-side Elwert exponential and ``1/p_i**2`` work from every energy
    block while preserving the Bethe-Heitler + Elwert expression algebraically.
    """
    zi = 2.0 * xp.pi * Z * ALPHA_FS
    den_i = 1.0 - xp.exp(-zi / beta_i)
    return (
        density_cm3
        * L_ang
        * 1.0e-8
        * (16.0 / 3.0)
        * ALPHA_FS
        * R_E_CM2
        * Z
        * Z
        * beta_i
        * den_i
        / (p_i * p_i)
    )


if hasattr(xp, "fuse"):
    _brem_incident_prefactor_core = xp.fuse()(_brem_incident_prefactor_core)


def _brem_incident_prefactor(L_ang, p_i, beta_i, Z, density_cm3):
    Z = REAL(Z)
    density_cm3 = REAL(density_cm3)
    return _brem_incident_prefactor_core(L_ang, p_i, beta_i, Z, density_cm3)


def _brem_dsigma_dk_core(T_i, k, Z):
    """Pure-elementwise Bethe-Heitler + Elwert core, one fused GPU kernel.

    ``T_i`` (kinetic energy [keV]) and ``k`` (photon energy [keV]) are the
    already-reshaped, mutually broadcasting operands (segments x grid); ``Z`` is
    the scalar atomic number. Split out of ``_brem_dsigma_dk`` so the whole chain
    (sqrt/divide/log/exp/where/maximum -- ~15 CuPy elementwise kernels, each
    allocating a full [seg, E] temporary and dominating ``cxr.brem`` at 300 keV)
    JIT-fuses to a SINGLE kernel under CuPy (see ``xp.fuse`` wrap below). On NumPy
    it runs eager with the identical ops, so it is bit-for-bit the old inline
    expression and the brem-spectrum golden is unchanged."""
    mc2 = _BREM_MC2_KEV
    T_f = T_i - k
    ok = (T_f > 1e-6) & (k > 0.0)  # k>0: no photon (and no 1/k blowup) at k=0
    T_f = xp.where(ok, T_f, 1e-6)

    p_i = xp.sqrt(T_i * (T_i + 2.0 * mc2)) / mc2
    p_f = xp.sqrt(T_f * (T_f + 2.0 * mc2)) / mc2
    beta_i = p_i / (1.0 + T_i / mc2)
    beta_f = p_f / (1.0 + T_f / mc2)

    born_log = xp.log((p_i + p_f) / xp.maximum(p_i - p_f, 1e-30))
    elwert = (
        beta_i
        / beta_f
        * (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_i))
        / (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_f))
    )

    dsig = (
        16.0
        / 3.0
        * ALPHA_FS
        * R_E_CM2
        * Z**2
        / xp.maximum(k * 1e3, 1e-30)
        / p_i**2
        * born_log
        * elwert
    )  # per eV
    return xp.where(ok, dsig, 0.0)


if hasattr(xp, "fuse"):  # CuPy exposes fuse(); NumPy/dpnp do not -> eager fallback
    _brem_dsigma_dk_core = xp.fuse()(_brem_dsigma_dk_core)


def _brem_dsigma_dk(Z, T_keV, k_eV):
    """
    Bremsstrahlung cross section differential in photon energy,
    dsigma/dk [cm^2/eV]: nonrelativistic Bethe-Heitler in Born approximation
    with the Elwert Coulomb correction (cf. Koch & Motz, Rev. Mod. Phys. 31,
    920 (1959)), evaluated with relativistic electron momenta:

        dsigma/dk = (16/3) alpha r_e^2 Z^2 (1/k) (1/p_i^2)
                    ln[(p_i+p_f)/(p_i-p_f)] * f_Elwert,
        f_Elwert  = (beta_i/beta_f) (1-exp(-2 pi Z alpha/beta_i))
                                  / (1-exp(-2 pi Z alpha/beta_f)),

    with p in units of m_e c. Broadcasts T_keV (segments) against k_eV
    (spectral grid); zero where k >= T. Adequate for Z <~ 30 and
    T <~ 100 keV; swap in Seltzer-Berger tables for better accuracy. The
    elementwise math lives in the fused ``_brem_dsigma_dk_core``.

    Validation: brem-spectrum
    """
    T_i = xp.asarray(T_keV, dtype=REAL)[:, None]
    k = xp.asarray(k_eV, dtype=REAL)[None, :] / 1e3  # keV
    # Z must arrive dtype-tagged, not as a bare Python int: cupy.fuse types an
    # untyped scalar operand by value (min_scalar_type), so the
    # 16/3*alpha*r_e^2 prefactor -- a scalar*scalar product with Z**2 -- gets
    # inferred as float16 and flushes ~3e-27 to zero, silently zeroing the
    # whole cross section on every fused (non-raw-kernel) GPU path.
    Z = REAL(Z)
    return _brem_dsigma_dk_core(T_i, k, Z)


def mc_brem_spectrum(
    segments,
    E_grid_eV,
    element=None,
    n_atoms_per_ang3=None,
    theta_obs_rad=np.deg2rad(119.0),
    n_hat=None,
    chunk=20000,
    composition=None,
    layers=None,
    groove=None,
    electron_limit=None,
    E_cut_keV=None,
):
    """
    Incoherent bremsstrahlung background d2N/dE dOmega
    [photons / eV / sr / electron] from the same Monte Carlo segments as
    mc_spectrum: each segment radiates n * dsigma/dk * L_seg photons/eV at
    its (start) kinetic energy, attenuated by the Beer-Lambert escape factor
    from the segment midpoint along the observation direction.

    Approximations: emission taken isotropic (1/4pi) -- the standard
    assumption at weakly relativistic energies once electron directions are
    scattering-randomized (and the one Zhai et al. adopt for their
    estimates); the tiny coherent fraction of the continuum (which is what
    forms the CBS lines) is not subtracted.

    NOTE: run the transport with a LOW E_cut_keV (~1 keV) for backgrounds --
    electrons below the CXR cutoff still radiate in the soft X-ray window.

    composition: [(element, n_per_Ang3), ...] for compounds; the emission is
    additive over elements (each weighted by its own Z^2 cross section), and
    the self-absorption uses the summed attenuation.

    Finite transverse dimensions stored on ``segments`` attenuate each photon
    to the first of the rectangular prism's six faces along the fixed far-field
    observation direction. With both dimensions omitted, the original z-only
    slab escape branches are retained unchanged.

    groove: optional GrooveSpec replacing the flat/prism escape length with the
    exact periodic working-facet distance from ``escape_distance_ang``. In the
    supported blazed geometry, ``n_hat = (cos(tp), 0, -sin(tp))`` crosses
    working facets outward and is parallel to relief facets, so the first
    crossing is the complete material path and cannot be followed by re-entry.
    This changes only the existing Beer--Lambert factor; emission cross sections
    and kinematics remain unchanged. Transport supplies material segments only,
    so vacuum legs do not radiate.

    Assumptions: straight photon rays, y-invariant and laterally periodic
    grooves, single-slab absorption, and the exact working-facet-normal
    observation direction. Layers and other directions raise rather than
    silently using flat attenuation. Source: exact periodic ray-plane
    intersections; see
    ``docs/validation/geometry/blazed-groove-geometry.md``.
    Limiting case: ``groove=None`` retains the original flat/prism path
    bit-for-bit; vanishing groove depth tends to the flat entrance-face path.

    Validation: brem-spectrum, finite-transverse-crystal, blazed-groove-geometry
    """
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    segments = _clip_segments_to_cutoff(segments, E_cut_keV, comp, layers)
    thickness = segments["thickness_ang"]
    if electron_limit is None:
        Ne = segments["Ne"]
    else:
        Ne = electron_limit

    n_hat = _observation_direction(theta_obs_rad, n_hat)
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        _validate_groove_escape_direction(n_hat, groove)

    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    mu = _mu_total_inv_ang(comp, E_grid)  # (NE,) [1/Ang], single-slab fallback
    # The Henke absorption tables span ~20 eV - 30 keV; outside that the wide
    # brem grid gets NaN (above 30 keV) or inf (at E=0), and a single bad bin
    # makes brem_wide -- and its integrated count rate -- NaN. Hard X-rays escape
    # essentially unattenuated, so treat an unavailable mu as zero (transparent).
    mu = xp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    # layered (film-on-substrate) absorber: precompute each layer's mu(E_grid);
    # the per-segment z-path dz folds in inside the chunk loop. None -> single slab.
    if layers is not None:
        inv_nz = 1.0 / max(abs(float(n_hat[2])), 1e-12)
        layer_mu = [
            xp.nan_to_num(_mu_total_inv_ang(c, E_grid), nan=0.0, posinf=0.0, neginf=0.0)
            for (_, _, c) in layers
        ]

    seg_elec_id = xp.asarray(segments["elec_id"])
    brem_electron = seg_elec_id < Ne

    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)[brem_electron]
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[brem_electron]
    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)[brem_electron]

    z_mid = seg_r[:, 2]
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    if groove is not None:
        L_esc = xp.asarray(
            escape_distance_ang(seg_r[:, 0], z_mid, groove),
            dtype=REAL,
        )
    elif finite_footprint:
        L_esc = _segment_escape_distance(segments, n_hat, xp=xp)[brem_electron]
    else:
        L_esc = _escape_length(z_mid, thickness, n_hat[2])

    spec = xp.zeros(E_grid.size, dtype=REAL)

    # GPU float32 fast path: evaluate Bethe-Heitler/Elwert, absorption, and the
    # segment reduction in one raw kernel. Instead of a dense T_abs[M, NE]
    # matrix, pass only each segment's path length through each absorber layer.
    _use_jit_brem_reduction = (
        _USE_JIT_BREM_REDUCTION
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_jit_brem_reduction:
        from .brem_jit_kernel import (
            DEFAULT_BREM_KERNEL_CONFIG,
            run_brem_reduction_kernel,
        )

        if layers is None:
            path_by_layer = L_esc[:, None]
            mu_by_layer = mu[None, :]
        else:
            path_cols = []
            if finite_footprint:
                for z_top, z_bot, _ in layers:
                    path_cols.append(
                        _layer_path_length(z_mid, n_hat[2], L_esc, float(z_top), float(z_bot))
                    )
            else:
                for z_top, z_bot, _ in layers:
                    dz = _layer_dz(z_mid, n_hat[2], float(z_top), float(z_bot))
                    path_cols.append(dz * inv_nz)
            path_by_layer = xp.stack(path_cols, axis=1)
            mu_by_layer = xp.stack(layer_mu, axis=0)

        path_by_layer = xp.ascontiguousarray(path_by_layer, dtype=REAL)
        mu_by_layer = xp.ascontiguousarray(mu_by_layer, dtype=REAL)
        path_flat = path_by_layer.reshape(-1)
        mu_flat = mu_by_layer.reshape(-1)
        T_jit = xp.ascontiguousarray(seg_E, dtype=REAL)
        L_jit = xp.ascontiguousarray(seg_L, dtype=REAL)
        E_jit = xp.ascontiguousarray(E_grid, dtype=REAL)
        p_i_jit, beta_i_jit = _brem_incident_state(T_jit)
        p_i_jit = xp.ascontiguousarray(p_i_jit, dtype=REAL)
        beta_i_jit = xp.ascontiguousarray(beta_i_jit, dtype=REAL)
        n_abs_layers = int(path_by_layer.shape[1])

        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            incident_prefactor = xp.ascontiguousarray(
                _brem_incident_prefactor(
                    L_jit,
                    p_i_jit,
                    beta_i_jit,
                    Z_i,
                    n_i * 1e24,
                ),
                dtype=REAL,
            )
            run_brem_reduction_kernel(
                T_jit,
                L_jit,
                path_flat,
                mu_flat,
                E_jit,
                Z=Z_i,
                density_cm3=n_i * 1e24,
                n_layers=n_abs_layers,
                p_i=p_i_jit,
                incident_prefactor=incident_prefactor,
                out=spec,
                config=DEFAULT_BREM_KERNEL_CONFIG,
            )
        return _to_cpu(spec / (4.0 * xp.pi) / Ne)

    M = seg_E.size
    for j0 in range(0, M, chunk):
        sl = slice(j0, min(j0 + chunk, M))
        if layers is None:
            T_abs = xp.exp(-L_esc[sl][:, None] * mu[None, :])
        elif finite_footprint:
            tau = 0.0
            for (z_top, z_bot, _), mu_i in zip(layers, layer_mu, strict=False):
                path = _layer_path_length(
                    z_mid[sl], n_hat[2], L_esc[sl], float(z_top), float(z_bot)
                )
                tau = tau + path[:, None] * mu_i[None, :]
            T_abs = xp.exp(-tau)
        else:
            tau = 0.0
            for (z_top, z_bot, _), mu_i in zip(layers, layer_mu, strict=False):
                dz = _layer_dz(z_mid[sl], n_hat[2], float(z_top), float(z_bot))
                tau = tau + (dz * inv_nz)[:, None] * mu_i[None, :]
            T_abs = xp.exp(-tau)
        path_cm = seg_L[sl] * 1e-8
        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            dsig = _brem_dsigma_dk(Z_i, seg_E[sl], E_grid)
            spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)
    return _to_cpu(spec / (4.0 * xp.pi) / Ne)


def load_external_brem(path, E_grid_eV):
    """
    Interpolate an EXTERNAL bremsstrahlung background onto the spectral grid
    -- e.g. a NIST DTSA-II simulation, which is what Zhai et al. use both
    for their simulated backgrounds (refs 96-100) and, with a PIXE-style
    numerical fit, for their experimental subtraction (SI S3).

    File format: two columns (energy [eV], intensity), whitespace- or
    comma-separated; '#' comment lines and non-numeric headers are skipped.
    The intensity must already be in DETECTED units matching your plots
    (e.g. Phs/eV/s/nA: from a DTSA-II counts export, divide counts/channel
    by channel width [eV] x live time [s] x beam current [nA]). It is
    treated as an as-detected spectrum: window efficiency and detector
    resolution are NOT re-applied. Energies outside the file's range
    interpolate to zero.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].startswith(("#", "//")):
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # header / text line
    if not rows:
        raise ValueError(f"no numeric (E, intensity) rows found in {path}")
    arr = np.array(sorted(rows))
    return np.interp(np.asarray(E_grid_eV, dtype=float), arr[:, 0], arr[:, 1], left=0.0, right=0.0)
