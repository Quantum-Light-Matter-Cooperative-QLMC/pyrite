"""Request validation and one-shot setup for the CXR line spectrum.

:class:`SpectrumRequest` is the frozen argument record ``mc_spectrum`` builds;
:func:`_prepare_spectrum` turns it into the ``_SpectrumSetup`` every
accumulation route consumes -- staged segment arrays, tabulation grids,
observation geometry, and the route-independent scalars.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ....materials.attenuation import _normalize_composition
from ....materials.crystal import CRYSTALS, HBARC_EV_ANG, refractive_index
from ...geometry import _mosaic_quadrature, _orientation_R
from ...transport import C_ANG_PER_FS, beta_from_keV
from ._kernels import (
    _clip_segments_to_cutoff,
    _elemental_log_mu_table,
    _line_tabulation_grid,
    _matvec3,
    _observation_direction,
    _validate_groove_escape_direction,
)


@dataclass(frozen=True, eq=False)
class SpectrumRequest:
    """The 26 inputs of :func:`mc_spectrum`, bound into one value.

    Grouping them is what lets the spectrum phases below be module-level
    functions instead of closures over ``mc_spectrum``'s locals: a phase takes
    the request (and the setup derived from it) rather than reaching into an
    enclosing scope. :func:`mc_spectrum` keeps its historical keyword signature
    and builds the request itself, so no caller sees this type.

    Frozen because a phase must not be able to edit the inputs a later phase
    reads. ``eq=False`` keeps the inherited identity ``__hash__``: several
    fields (``segments``, ``hkl_list``, the energy grid) are unhashable
    containers, and the batched path's ``_table_cache`` key is an explicit
    tuple rather than a hash of this object.
    """

    segments: Any
    E_grid_eV: Any
    crystal: Any
    hkl_list: Any
    theta_obs_rad: Any = np.deg2rad(119.0)
    B_ang2: Any = None
    use_henke: Any = True
    absorber_element: Any = "C"
    chunk: Any = 40000
    n_hat: Any = None
    composition: Any = None
    beam_uvw: Any = None
    azimuth_rad: Any = 0.0
    recip_miscut_rad: Any = None
    sinc_cutoff: Any = None
    components: Any = False
    layers: Any = None
    mosaic_fwhm_rad: Any = None
    mosaic_nodes: Any = 1
    surface_hkl: Any = None
    groove: Any = None
    coherent: Any = False
    electron_limit: Any = None
    E_cut_keV: Any = None
    _table_cache: Any = None
    longitudinal_rms_fs: Any = None


@dataclass
class _SpectrumSetup:
    """Everything :func:`_prepare_spectrum` derives from a
    :class:`SpectrumRequest`: resolved geometry, staged device arrays, the
    energy tabulations, the flight-grouping decision, the coherent and
    inter-electron decoherence precompute, and the three accumulation buffers.

    This is the state ``mc_spectrum``'s inner closures used to capture. Passing
    it explicitly is what makes the accumulation phases module-level functions a
    test can drive directly.

    Mutable, and deliberately so: ``spec`` / ``spec_pxr`` / ``spec_cbs`` are
    accumulated into in place by whichever route runs, and ``L_esc_all`` is
    filled in by the per-hkl route's stacking prologue.
    """

    request: SpectrumRequest
    info: Any
    abs_comp: Any
    segments: Any
    R_orient: Any
    thickness: Any
    Ne: Any
    n_hat: Any
    n_hat_d: Any
    E_grid: Any
    spec: Any
    spec_pxr: Any
    spec_cbs: Any
    seg_r: Any
    seg_elec_id: Any
    line_electron: Any
    v_all: Any
    v_dot_n_all: Any
    denom_all: Any
    gamma_all: Any
    t_L_all: Any
    gid_all: Any
    grouped: Any
    E_tab: Any
    E_tab_g: Any
    log_mu_tab_g: Any
    n_re_tab_g: Any
    mosaic_quad: Any
    finite_footprint: Any
    cdtype: Any
    omega_grid: Any
    delta_omega_grid: Any
    d_all: Any
    seg_r_geom: Any
    d_all_geom: Any
    decoherence_active: Any
    finite_footprint_now: Any
    finite_footprint_F: Any
    decoherence_A_pop: Any
    xy0_pop: Any
    L_esc_all: Any = None


def _prepare_spectrum(request):
    """Phase 1: validate the request and build every quantity that depends on
    neither the reflection nor the crystallite orientation.

    Covers argument compatibility, crystal lookup and orientation, cutoff
    clipping, device staging of the segment arrays, the physical-flight
    grouping decision, the edge-resolved chi/U/mu/n tabulation grid, the
    segment-only kinematics both accumulation routes share, the coherent
    propagation-phase precompute, and the inter-electron decoherence
    population terms.

    Returns the :class:`_SpectrumSetup` the accumulation phases consume.
    """
    segments = request.segments
    E_grid_eV = request.E_grid_eV
    crystal = request.crystal
    theta_obs_rad = request.theta_obs_rad
    B_ang2 = request.B_ang2
    use_henke = request.use_henke
    absorber_element = request.absorber_element
    n_hat = request.n_hat
    composition = request.composition
    beam_uvw = request.beam_uvw
    azimuth_rad = request.azimuth_rad
    recip_miscut_rad = request.recip_miscut_rad
    components = request.components
    layers = request.layers
    mosaic_fwhm_rad = request.mosaic_fwhm_rad
    mosaic_nodes = request.mosaic_nodes
    surface_hkl = request.surface_hkl
    groove = request.groove
    coherent = request.coherent
    electron_limit = request.electron_limit
    E_cut_keV = request.E_cut_keV
    longitudinal_rms_fs = request.longitudinal_rms_fs

    if coherent and components:
        raise ValueError(
            "coherent=True is incompatible with components=True: the PXR/CBS "
            "split is ambiguous under coherence (the A_PXR*A_CBS cross term "
            "survives). Request the coherent total, or components incoherently."
        )
    if B_ang2 is None:
        raise ValueError(
            "mc_spectrum: B_ang2 (Debye-Waller B-factor [Ang^2]) is required; "
            "pass the material's value (no silent default)."
        )
    if coherent and layers is not None:
        raise NotImplementedError(
            "the in-medium dispersion does not cover the coherent path through "
            "a LAYERED absorber: the dispersive propagation phase needs a "
            "per-layer delta accumulated along the escape path -- the real "
            "partner of _stack_tau's per-layer mu -- which is not modelled. "
            "Single-slab absorbers (with or without a groove or a finite "
            "footprint) are supported."
        )
    # Precompute that only some routes bind. As a closure these names simply
    # went unbound on the paths that never read them; as dataclass fields they
    # need an explicit absent value. Nothing below reads one where the original
    # would have raised UnboundLocalError.
    cdtype = omega_grid = delta_omega_grid = d_all = None
    seg_r_geom = d_all_geom = xy0_pop = None
    finite_footprint_now = finite_footprint_F = decoherence_A_pop = None

    info = CRYSTALS[crystal]
    n_atoms = len(info["basis"]) / info["V_cell"]
    abs_comp = _normalize_composition(absorber_element, n_atoms, composition)
    segments = _clip_segments_to_cutoff(segments, E_cut_keV, abs_comp, layers)

    # crystal orientation: rotation applied to all reciprocal vectors
    R_orient = _orientation_R(
        info["lattice"],
        beam_uvw,
        azimuth_rad,
        recip_miscut_rad,
        surface_hkl=surface_hkl,
    )
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
    spec = xp.zeros(E_grid.size, dtype=REAL)
    spec_pxr = xp.zeros(E_grid.size, dtype=REAL)
    spec_cbs = xp.zeros(E_grid.size, dtype=REAL)

    # Every per-row emission coefficient is evaluated at ONE energy along the
    # row. Under ``energy_model="midpoint"`` transport supplies the propagator's
    # own representative energy, so the row becomes a midpoint evaluation of its
    # emission integral rather than a left-endpoint one; the resulting t_L =
    # L/beta(E_repr) is then exactly the transported flight duration
    # ``t_end - t_start``, and ``t_ang + t_L/2`` exactly its midpoint age.
    # Frozen rows carry no representative energy and stay bit-for-bit.
    # Validation: substep-radiation-invariance
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    seg_E = xp.asarray(segments[E_field], dtype=REAL)
    seg_v = xp.asarray(segments["v_hat"], dtype=REAL)
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)
    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)
    seg_elec_id = xp.asarray(segments["elec_id"])
    line_electron = seg_elec_id < Ne
    beta_all = beta_from_keV(seg_E)  # speed/c per segment
    v_all = beta_all[:, None] * seg_v  # velocity vectors (c=1)

    # Physical-flight grouping for the DEFAULT (incoherent) reduction. Numerical
    # substeps of one flight are integration detail: summing |A_j Q_j|^2 over
    # them treats each as an independent emitter and divides the line peak by
    # the substep count (each row carries (t_L/N)^2 where the flight carries
    # t_L^2), so a tighter energy tolerance would silently destroy the line.
    # Rows of one flight therefore add COHERENTLY and only whole flights add
    # incoherently. Groups are keyed by ``(electron_id, flight_id)`` VALUE, not
    # by adjacency: the lockstep core emits step-major, so one flight's substeps
    # are separated by every other electron's rows for that step.
    # Validation: substep-radiation-invariance
    gid_all = None
    grouped = False
    if not coherent and segments.get("flight_id") is not None and seg_E.size:
        flight_key = _to_cpu(xp.asarray(segments["flight_id"]))
        electron_key = _to_cpu(xp.asarray(segments["elec_id"]))
        order = np.lexsort((flight_key, electron_key))
        new_group = np.empty(flight_key.size, dtype=bool)
        new_group[0] = True
        new_group[1:] = (flight_key[order][1:] != flight_key[order][:-1]) | (
            electron_key[order][1:] != electron_key[order][:-1]
        )
        # All-singleton groups are the one-row-per-flight case, whose grouped
        # reduction is algebraically the incoherent one; keep the proven path.
        grouped = not bool(new_group.all())
        if grouped:
            gid_all = np.empty(flight_key.size, dtype=np.int64)
            gid_all[order] = np.cumsum(new_group) - 1
    if grouped and getattr(xp, "__name__", "") != "numpy":
        raise ValueError(
            "flight-grouped incoherent CXR is host-only: the segmented complex "
            "reduction over numerical substeps has no device port yet. Run the "
            "spectrum on the NumPy backend, or transport without max_dE_frac."
        )
    if grouped and layers is not None:
        # Same unmodelled per-layer delta as the coherent path: the grouped
        # reduction carries the dispersive propagation phase too.
        raise NotImplementedError(
            "the in-medium dispersion does not cover numerical substeps through "
            "a LAYERED absorber: the intra-flight coherent sum needs the "
            "per-layer dispersive propagation phase, which is not modelled."
        )
    if grouped and components:
        raise ValueError(
            "components=True is incompatible with numerical substeps: the "
            "PXR/CBS split is ambiguous once a flight's substep fields add "
            "coherently (the A_PXR*A_CBS cross term survives). Request the "
            "total, or transport without max_dE_frac."
        )

    # chi_g / U_g are smooth in energy AWAY from absorption edges, so evaluate
    # them on a tabulation grid and interpolate at the per-segment resonance
    # energies ON THE GPU (step 3) -- a few ms over ~10^3 grid points instead of
    # ~0.25 s/case over ~10^5 segments, and E_res stays on the device (no
    # per-reflection GPU->CPU->GPU round-trip; that structure-factor CPU cost was
    # what capped GPU utilisation on a fast card). The grid is a 1 eV mesh UNION
    # the basis and absorber elements' native Henke energies, which densely
    # sample the edges -- a plain uniform mesh mis-resolves the edge jumps (tens
    # of % at e.g. the C K-edge). Window matches the keep mask below.
    _pad = 0.2 * (float(E_grid_eV[-1]) - float(E_grid_eV[0]))
    _lo, _hi = float(E_grid_eV[0]) - _pad, float(E_grid_eV[-1]) + _pad
    _lo = max(_lo, 1.0)  # keep tabulation energies positive: chi_g/U_g need lambda = HC_EV_ANG / E
    E_tab = _line_tabulation_grid(info, abs_comp, _lo, _hi)
    E_tab_g = xp.asarray(E_tab, dtype=REAL)
    # Each elemental coefficient is stored as log(mu_i) [log(1/Ang)] and
    # interpolated linearly against log(E) before summing. This reproduces the
    # pinned xraydb non-f1 Chantler rule; interpolating the compound total in
    # either linear or log space does not. Explicit absorber elements contribute
    # their native nodes to E_tab even when absent from the crystal basis.
    # Applied only to single-slab/finite-footprint routes. Layered and grooved
    # escape retain exact per-point mu. No per-segment host/device transfer is
    # restored. Independently rederived with no physics divergence; human
    # sign-off remains required.
    # Validation: line-absorption-tabulation
    log_mu_tab_g = xp.asarray(_elemental_log_mu_table(abs_comp, E_tab), dtype=REAL)

    # Real part of the crystal's bulk refractive index n(E) = sqrt(1 + chi_0(E)),
    # tabulated on the SAME edge-resolved grid as chi/U/mu (delta = 1 - Re n has
    # its own edge structure, from the f1 cusp). The in-medium dispersion is
    # unconditional, so this table is always built.
    n_re_tab_g = xp.asarray(
        np.asarray(refractive_index(crystal, E_tab, use_henke).real), dtype=REAL
    )

    n_hat_d = xp.asarray(n_hat, dtype=REAL)  # detector dir is g-independent: hoist

    # Segment-only kinematics shared by every reflection/orientation and by both
    # the batched and compatibility paths. These used to be recomputed inside
    # ``_accumulate`` for every g row.
    v_dot_n_all = _matvec3(v_all, n_hat_d)
    denom_all = 1.0 - v_dot_n_all
    gamma_all = 1.0 / xp.sqrt(1.0 - beta_all * beta_all)
    t_L_all = seg_L / beta_all

    # coherent (phased) sum precompute: the per-segment retardation scalar
    # d_j = t_abs,j - n_hat.r_j [Ang, c=1] (emission-time phase minus far-field
    # retardation) and photon wavenumber k_gamma(E) = E / hbar c [1/Ang].
    # ``t_ang`` remains segment-start age; add half the constant-velocity flight
    # time so it describes the same midpoint as ``r_mid``. This one d_all feeds
    # both coherent reduction routes. Validation: coherent-segment-midpoint-time.
    # Each reflection adds its spatial susceptibility phase -g.r_j inside
    # _accumulate. All-zero t0_ang leaves the physical trajectory phase.
    # Inert unless a complex per-row field is actually reduced, i.e. under
    # coherent=True or the flight-grouped incoherent path.
    if coherent or grouped:
        cdtype = xp.result_type(REAL, 1j)
        seg_t0 = xp.asarray(segments.get("t0_ang", np.zeros(seg_E.size)), dtype=REAL)
        seg_t = xp.asarray(segments.get("t_ang", np.zeros(seg_E.size)), dtype=REAL)
        seg_t_mid = seg_t + 0.5 * t_L_all
        d_all = (seg_t_mid + seg_t0) - _matvec3(seg_r, n_hat_d)
        omega_grid = E_grid / HBARC_EV_ANG

        # In-medium propagation phase: each segment's field picks up
        # -delta(E) omega(E) L_esc,j over the SAME in-crystal escape path the
        # Beer-Lambert factor runs over -- the real partner of that absorption.
        # Deliberately NOT k(E) n_hat.r_j, which would charge the medium for the
        # whole flight to the detector. Tabulated on the OUTPUT grid: it is a
        # propagation phase read across the spectrum, not a coupling frozen at
        # the line energy. Derivation:
        # docs/physics/radiation-physics/photon-escape-and-dispersion.md
        # Validation: xray-in-medium-propagation-phase
        delta_omega_grid = (
            xp.asarray(
                1.0 - np.asarray(refractive_index(crystal, E_grid_eV, use_henke).real),
                dtype=REAL,
            )
            * omega_grid
        )

    # Empirical inter-electron decoherence for the coherent path. Squaring one
    # realization of the sampled per-electron offsets is speckle, not shot
    # noise: it does not shrink with electron count. Blend instead, per (row,
    # energy), Total = (1-F)*Grouped + F*Flat, where Flat is today's reduction
    # fed the electron's OFFSET-FREE position/time (hence the
    # d_all_geom/seg_r_geom swap below, a strict no-op when no offset is
    # configured), Grouped = sum_e |S_e|^2 groups the same segments by electron,
    # and F is the empirical characteristic function of the offset population
    # transport already drew. Derivation and the closed form it converges to:
    # docs/physics/radiation-physics/coherent-emission.md
    #
    # F must be applied PER ROW (reflection x mosaic orientation), before
    # summing across rows: q_perp depends on g, so a single scalar F(E) on the
    # row-summed spectrum would be wrong wherever two rows share an energy bin.
    #
    # Validation: coherent-inter-electron-decoherence
    decoherence_active = False
    if coherent:
        seg_r_geom = seg_r
        d_all_geom = seg_t_mid - _matvec3(seg_r, n_hat_d)
        t0_pop = xp.asarray(segments.get("initial_t0_ang", np.zeros(0)), dtype=REAL)[:Ne]
        xy0_pop = xp.asarray(
            np.asarray(segments.get("initial_r_ang", np.zeros((0, 3))))[:, :2], dtype=REAL
        )[:Ne]
        decoherence_active = bool(
            (t0_pop.size and xp.any(t0_pop != 0.0)) or (xy0_pop.size and xp.any(xy0_pop != 0.0))
        )
        if longitudinal_rms_fs is not None:
            longitudinal_rms_fs = float(longitudinal_rms_fs)
            if not np.isfinite(longitudinal_rms_fs) or longitudinal_rms_fs <= 0.0:
                raise ValueError("longitudinal_rms_fs must be finite and positive")
        if decoherence_active:
            finite_footprint_now = (
                segments.get("crystal_width_ang") is not None
                and segments.get("crystal_height_ang") is not None
            )
            if finite_footprint_now:
                if longitudinal_rms_fs is None:
                    raise ValueError(
                        "coherent emission with a finite crystal footprint and "
                        "nonzero bunch offsets requires longitudinal_rms_fs; "
                        "only the fully longitudinally decohered limit is "
                        "supported"
                    )
                sigma_z_ang = longitudinal_rms_fs * C_ANG_PER_FS
                # Decide support in host float64 so a borderline duration does
                # not run on float32 CUDA while the CPU rejects it.
                omega_host = np.asarray(_to_cpu(omega_grid), dtype=float)
                finite_footprint_F_host = np.exp(-((omega_host * sigma_z_ang) ** 2))
                finite_footprint_F = xp.asarray(finite_footprint_F_host, dtype=REAL)
                # Average only the independent longitudinal arrival time. Keep
                # each electron at its sampled transverse position in BOTH the
                # flat and grouped terms: its phase, hit/miss history, and
                # finite-prism attenuation are coupled and must stay together.
                # F_z then blends those terms exactly, conditional on this
                # transverse/transport realization. The CUDA-JIT grouped
                # reductions consume these same arrays.
                d_all_geom = seg_t_mid - _matvec3(seg_r, n_hat_d)
            else:
                seg_r_geom = seg_r.copy()
                seg_elec_id_clamped = xp.clip(seg_elec_id, 0, max(Ne - 1, 0))
                seg_r_geom[:, :2] = seg_r_geom[:, :2] - xy0_pop[seg_elec_id_clamped]
                d_all_geom = seg_t_mid - _matvec3(seg_r_geom, n_hat_d)
                # A_e = t0_e - n_hat_perp . dr_perp,e does not depend on g (the
                # reciprocal vector varies per row; n_hat is fixed for the whole
                # call), so hoist it once here; B_e(row) = g_perp . dr_perp,e is
                # cheap and stays inside the per-row helper below.
                decoherence_A_pop = t0_pop - xy0_pop @ n_hat_d[:2]

    # mosaic crystallite-orientation quadrature: None -> perfect crystal (default;
    # today's single-orientation result bit-for-bit). Otherwise a list of
    # (rotation, weight) tilting g across the Gaussian mosaic cone, summed
    # incoherently below (docs/physics/materials/crystal-mosaicity.md route 2).
    mosaic_quad = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes)

    # Finite transverse dimensions make the escape distance a nearest-face
    # query rather than the plain slab path. It is g-independent either way,
    # so both routes hoist it out of their per-reflection work.
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )

    return _SpectrumSetup(
        request=request,
        info=info,
        abs_comp=abs_comp,
        segments=segments,
        R_orient=R_orient,
        thickness=thickness,
        Ne=Ne,
        n_hat=n_hat,
        n_hat_d=n_hat_d,
        E_grid=E_grid,
        spec=spec,
        spec_pxr=spec_pxr,
        spec_cbs=spec_cbs,
        seg_r=seg_r,
        seg_elec_id=seg_elec_id,
        line_electron=line_electron,
        v_all=v_all,
        v_dot_n_all=v_dot_n_all,
        denom_all=denom_all,
        gamma_all=gamma_all,
        t_L_all=t_L_all,
        gid_all=gid_all,
        grouped=grouped,
        E_tab=E_tab,
        E_tab_g=E_tab_g,
        log_mu_tab_g=log_mu_tab_g,
        n_re_tab_g=n_re_tab_g,
        mosaic_quad=mosaic_quad,
        finite_footprint=finite_footprint,
        cdtype=cdtype,
        omega_grid=omega_grid,
        delta_omega_grid=delta_omega_grid,
        d_all=d_all,
        seg_r_geom=seg_r_geom,
        d_all_geom=d_all_geom,
        decoherence_active=decoherence_active,
        finite_footprint_now=finite_footprint_now,
        finite_footprint_F=finite_footprint_F,
        decoherence_A_pop=decoherence_A_pop,
        xy0_pop=xy0_pop,
    )
