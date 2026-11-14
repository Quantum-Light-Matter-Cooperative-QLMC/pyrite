"""Request validation and one-shot setup for the CXR CXR spectrum.

:class:`SpectrumRequest` is the frozen argument record ``mc_cxr_spectrum`` builds;
:func:`_prepare_spectrum` turns it into the ``_SpectrumSetup`` every
accumulation route consumes -- staged segment arrays, tabulation grids,
observation geometry, and the route-independent scalars.

Validation: line-tabulation-nan-ceiling
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...._backend import REAL, BackendResourceError, _to_cpu, xp
from ....materials.attenuation import _normalize_composition
from ....materials.crystal import CRYSTALS, HBARC_EV_ANG, refractive_index
from ...geometry import _mosaic_quadrature, _orientation_R
from ...transport import C_ANG_PER_FS, beta_from_keV
from ..coherent_population import pair_scale
from ..coherent_transverse import require_footprint_support, transverse_spot
from ..segment_escape import segment_escape_gradient, segment_escape_pieces
from ._bin_quadrature import BIN_MEAN_QUADRATURE, bin_axis, validate_cxr_quadrature
from ._formation import expand_escape_pieces
from ._kernels import (
    _clip_segments_to_cutoff,
    _cxr_setup_ceiling,
    _cxr_tabulation_grid,
    _elemental_log_mu_table,
    _matvec3,
    _observation_direction,
    _vacuum_doppler,
    _validate_groove_escape_direction,
)
from ._line_tables import line_table_bytes
from ._node_window import DEFAULT_NODE_SUPPORT_CUTOFF, NodeTail, uniform_step
from ._temporal import segment_arrival_times

# Generation marker for the line route's photon-escape model, hashed into the
# dataset and case-content identities like ``characteristic_model``. A
# constant, not a selector: the incoherent route scores the segment-mean
# escape since issue #181, and the phased-field routes the exact per-piece
# formation integral under absorption since its second slice; each move
# orphans the previous generation's line spectra once. v4 (#513): the CBS
# potential coupling U_g is the static Z - f0(g), no longer carrying the
# anomalous f' of edge-prone elements. v5 (#439): gamma and the Doppler
# factor 1 - v.n_hat are formed cancellation-free and the in-medium root reads
# a float64-formed decrement table, which moves float32 device lines at MeV.
# v6 (#1099): every line route places, widens and phases lines on the exact
# recoil root (line-recoil-in-medium-root), a percent-level shift at 10-100 keV.
# v7 (#1134): PXR/CBS amplitudes are the exact second-order elements (quantum
# CB denominators and transition currents, cbs-amplitude), percent-level
# sub-100 keV line intensities.
# v8 (#1191): incoherent node scoring defaults to the support-windowed
# reduction with its analytic far tail (node-support-window-tail), a ~1e-5
# relative yield shift on the frozen #1142/#1143 scenes.
CXR_ESCAPE_MODEL = "segment-mean-v8-node-support-tail"


#: Size of the most recent line setups' tabulation, for performance telemetry.
#: The runner clears it before a case and reads it after, when timing is on.
SETUP_STATS: dict[str, float] = {}


@dataclass(frozen=True, eq=False)
class SpectrumRequest:
    """The 28 inputs of :func:`mc_cxr_spectrum`, bound into one value.

    Grouping them is what lets the spectrum phases below be module-level
    functions instead of closures over ``mc_cxr_spectrum``'s locals: a phase takes
    the request (and the setup derived from it) rather than reaching into an
    enclosing scope. :func:`mc_cxr_spectrum` keeps its historical keyword signature
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
    node_support_cutoff: Any = "auto"
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
    line_quadrature: Any = "node"
    truncation_audit: Any = None
    temporal: Any = None
    coefficient_capture: Any = None
    physical_electrons: Any = None
    coherent_flat_omission_limit: Any = 0.0
    phase_retention: Any = None


@dataclass
class _SpectrumSetup:
    """Everything :func:`_prepare_spectrum` derives from a
    :class:`SpectrumRequest`: resolved geometry, staged device arrays, the
    energy tabulations, the flight-grouping decision, the coherent and
    inter-electron decoherence precompute, and the three accumulation buffers.

    This is the state ``mc_cxr_spectrum``'s inner closures used to capture. Passing
    it explicitly is what makes the accumulation phases module-level functions a
    test can drive directly.

    Mutable, and deliberately so: ``spec`` / ``spec_pxr`` / ``spec_cbs`` are
    accumulated into in place by whichever route runs.
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
    cxr_electron: Any
    v_all: Any
    denom_all: Any
    gamma_all: Any
    t_L_all: Any
    gid_all: Any
    grouped: Any
    E_tab: Any
    E_tab_g: Any
    log_mu_tab_g: Any
    delta_tab_g: Any
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
    analytic_offsets: Any
    finite_footprint_F: Any
    transverse: Any
    decoherence_A_pop: Any
    xy0_pop: Any
    # Incoherent routes only: padded linear escape pieces per segment
    # (``segment_escape.segment_escape_pieces``), for the segment-mean
    # transmission. Validation: segment-escape-average
    escape_pieces: Any = None
    # Phased-field routes only (coherent, flight-grouped): escape distance at
    # the start and end of every piece row, ordered along travel, for the
    # complex formation integral. Validation: coherent-formation-absorption
    escape_ends: Any = None
    escape_gradient: Any = None
    piece_fraction: Any = None
    # Phased-field routes only: the electron's own midpoint age [Angstrom, c=1]
    # (no bunch ``t0``), over which the exact-recoil phase accumulates.
    # Validation: line-recoil-in-medium-root
    seg_t_mid: Any = None
    # Flight-grouped route only: float64 ``(d, r_mid)`` per row (see setup).
    grouped_phase64: Any = None
    # Bin-mean quadrature only: FP64 bin edges and inverse widths on the
    # accumulation routes' device (``_bin_quadrature.bin_axis``).
    bin_edges: Any = None
    bin_inv_width: Any = None
    # Support-windowed node reduction only: the per-call far-tail accumulator
    # (``_node_window.NodeTail``), applied before finalization.
    # Validation: node-support-window-tail
    node_tail: Any = None
    # The same accumulator when the per-row route windows too, else None: the
    # automatic device policy keeps per-row lines on the dense reducer.
    row_tail: Any = None
    # Per-hkl route under an experimental bin reduction (#1192) only: queued
    # lines per target spectrum, reduced together once all rows are scored.
    bin_line_queues: Any = None
    # Opt-in temporal profile only: this call's float64 buffer and the
    # realized / offset-free retarded arrival times per row.
    # Validation: temporal-intensity-profile
    temporal_buf: Any = None
    temporal_tau: Any = None
    temporal_tau_geo: Any = None
    # Read-only coefficient observers receive the current row's midpoint
    # susceptibility phase, aligned with idx/coefs (not all segment rows).
    capture_phase_rad: Any = None
    # Matching reducer-geometry ``d`` with the exact-recoil shift.
    capture_centre_ang: Any = None
    # Production intensity weight for the captured (reflection, orientation).
    capture_mosaic_weight: float = 1.0
    # Nonincreasing transverse bound sup F_perp for the row, or None.
    capture_transverse_envelope: Any = None


def _admit_line_tables(n_hkl, n_tab):
    """Refuse a reflection list whose chi/U tables exceed the device budget.

    The tables scale with ``n_hkl * n_tab`` and are independent of the
    segment chunk, so chunk admission cannot bound them (#1167). Failing
    here names the cause instead of an allocator OOM that chunk-halving
    retries cannot fix.
    """
    from ...runner import _RESOURCE_POLICY  # runner imports this module

    budget = _RESOURCE_POLICY.device_budget_bytes
    if not _RESOURCE_POLICY.gpu or budget is None:
        return
    need = line_table_bytes(n_hkl, n_tab, np.dtype(REAL).itemsize)
    if need > budget:
        raise BackendResourceError(
            f"CXR line tables for {n_hkl} reflections x {n_tab} tabulation points "
            f"need {need / 2**30:.2f} GiB, over the device budget "
            f"{budget / 2**30:.2f} GiB; score the incoherent spectrum over "
            f"reflection subsets and sum them, or narrow the energy grid"
        )


def _flight_groups(segments):
    """Physical-flight group ids of the rows: ``(grouped, gid_all)``.

    Groups are keyed by ``(owner, flight_id)`` value, the owner being the
    track under secondary transport (#94) and the electron otherwise.
    ``grouped`` is False when every group is a single row; ``gid_all`` then
    stays None. On a device backend the flag is decided first by sorting one
    packed key, instead of downloading both key columns and lexsorting them on
    the host (~0.5 s per call at 2e7 rows); only grouped rows then pay for the
    device-resident ids. Validation: substep-radiation-invariance
    """
    track = segments.get("track_id")
    owner = segments["elec_id"] if track is None else track
    if getattr(xp, "__name__", "") != "numpy":
        flight = xp.asarray(segments["flight_id"])
        owner = xp.asarray(owner)
        if not _device_rows_share_flight(flight, owner):
            return False, None
        return True, _device_flight_ids(flight, owner)
    flight_key = _to_cpu(xp.asarray(segments["flight_id"]))
    electron_key = _to_cpu(xp.asarray(owner))
    order = np.lexsort((flight_key, electron_key))
    new_group = np.empty(flight_key.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (flight_key[order][1:] != flight_key[order][:-1]) | (
        electron_key[order][1:] != electron_key[order][:-1]
    )
    # All-singleton groups are the one-row-per-flight case, whose grouped
    # reduction is algebraically the incoherent one; keep the proven path.
    if new_group.all():
        return False, None
    gid_all = np.empty(flight_key.size, dtype=np.int64)
    gid_all[order] = np.cumsum(new_group) - 1
    return True, gid_all


def _device_rows_share_flight(flight, owner):
    """Whether two rows share one ``(owner, flight)`` key, decided on the device.

    Packs both keys into one int64 offset from their minima, sorts it, and
    compares neighbours: two host scalar readbacks instead of two full-column
    downloads. Falls back to a stable two-key sort when the packed key would
    not fit in int64.
    """
    flight = flight.astype(np.int64, copy=False)
    owner = owner.astype(np.int64, copy=False)
    f_lo, f_hi, o_lo, o_hi = (
        int(v) for v in _to_cpu(xp.stack([flight.min(), flight.max(), owner.min(), owner.max()]))
    )
    span = f_hi - f_lo + 1
    if (o_hi - o_lo + 1) * span < 2**63:
        key = xp.sort((owner - o_lo) * span + (flight - f_lo))
        return bool(_to_cpu(xp.any(key[1:] == key[:-1])))
    order = xp.lexsort(xp.stack([flight, owner]))
    flight, owner = flight[order], owner[order]
    return bool(_to_cpu(xp.any((flight[1:] == flight[:-1]) & (owner[1:] == owner[:-1]))))


def _device_flight_ids(flight, owner):
    """Dense ``(owner, flight)`` group ids, computed on the device.

    Same ids as the host branch of :func:`_flight_groups`: groups numbered in
    ``(owner, flight)`` lexicographic order. Validation: substep-radiation-invariance
    """
    order = xp.lexsort(xp.stack([flight.astype(np.int64), owner.astype(np.int64)]))
    f, o = flight[order], owner[order]
    new_group = xp.ones(f.size, dtype=bool)
    new_group[1:] = (f[1:] != f[:-1]) | (o[1:] != o[:-1])
    gid = xp.empty(f.size, dtype=np.int64)
    gid[order] = xp.cumsum(new_group) - 1
    return gid


def _grouped_phase64(segments, t_L_all, n_hat, grouped):
    """Float64 ``(d, r_mid)`` per row for the flight-grouped reduction, else None.

    ``d = t_mid + t0 - n_hat.r_mid``, the ``d_all`` of the phased routes,
    kept in float64 so ``_grouped`` can reference every row to its flight's
    first row before any float32 cast: absolute ages and ``g.r`` reach ~1e4
    rad, the intra-flight differences only the flight's own extent. Reads the
    pieces' float64 midpoints/ages (``expand_escape_pieces``), not their REAL
    casts: positions near 1e7 Angstrom (footprints, bunch offsets) would
    otherwise carry ~1 Angstrom per-row quantization into ``n_hat.r``.
    Validation: coherent-formation-absorption
    """
    if not grouped:
        return None
    n_rows = int(xp.asarray(segments["L_ang"]).shape[0])
    zeros = np.zeros(n_rows)
    r64 = xp.asarray(segments.get("r_mid_f64", segments["r_mid"]), dtype=np.float64)
    t64 = xp.asarray(segments.get("t_ang_f64", segments.get("t_ang", zeros)), dtype=np.float64)
    t64 = t64 + xp.asarray(segments.get("t0_ang", zeros), dtype=np.float64)
    t64 = t64 + 0.5 * t_L_all.astype(np.float64)
    return t64 - r64 @ xp.asarray(n_hat, dtype=np.float64), r64


def _refuse_grooved_offset_factorization(request, xy0_pop, analytic_offsets, xp):
    """Refuse coherent reductions that assume a groove-independent ``|S_e|``.

    A groove's escape path depends on the lateral emission point, so a
    transverse offset moves ``|S_e|`` as well as its phase. That breaks the
    phase-only factorization of the empirical blend and the translation
    invariance of the analytic average. Pairing complete sampled fields needs
    neither, so that route stays available.
    Validation: coherent-inter-electron-decoherence
    """
    if (
        request.groove is not None
        and bool(xp.any(xy0_pop != 0.0))
        and (request.physical_electrons is None or analytic_offsets)
    ):
        raise ValueError(
            "coherent emission from a grooved target with transverse beam offsets "
            "is only supported by the physical-bunch pair estimator on an infinite "
            "slab: pass physical_electrons, without a finite footprint or imposed "
            "bunch front, or use transverse=PointSpot()"
        )


def _refuse_incompatible_request(request):
    """Refuse option combinations that are invalid before any setup runs."""
    if request.coherent and request.components:
        raise ValueError(
            "coherent=True is incompatible with components=True: the PXR/CBS "
            "split is ambiguous under coherence (the A_PXR*A_CBS cross term "
            "survives). Request the coherent total, or components incoherently."
        )
    if request.B_ang2 is None:
        raise ValueError(
            "mc_spectrum: B_ang2 (Debye-Waller B-factor [Ang^2]) is required; "
            "pass the material's value (no silent default)."
        )
    if request.coherent and request.segments.get("track_id") is not None:
        raise NotImplementedError(
            "coherent spectra are not supported with secondary transport: the "
            "coherent sum runs within one electron_id, which would make a whole "
            "shower one emitter (see shell-soft-hard-transport.md)"
        )
    if request.coherent and request.layers is not None:
        raise NotImplementedError(
            "the in-medium dispersion does not cover the coherent path through "
            "a LAYERED absorber: the dispersive propagation phase needs a "
            "per-layer delta accumulated along the escape path -- the real "
            "partner of _stack_tau's per-layer mu -- which is not modelled. "
            "Single-slab absorbers (with or without a groove or a finite "
            "footprint) are supported."
        )
    _refuse_node_support_cutoff(request)


def _refuse_node_support_cutoff(request):
    """Validate ``node_support_cutoff`` and refuse the routes it does not cover."""
    x_c = request.node_support_cutoff
    if x_c is None or x_c == "auto":
        return
    if not np.isfinite(x_c) or x_c <= 0:
        raise ValueError("node_support_cutoff must be finite and positive")
    if request.coherent:
        raise ValueError(
            "node_support_cutoff windows the incoherent node reduction only; the "
            "coherent route's windowing is sinc_cutoff"
        )
    if request.sinc_cutoff is not None:
        raise ValueError(
            "node_support_cutoff (windowed, tail kept) and sinc_cutoff (tail dropped) are exclusive"
        )
    if request.line_quadrature != "node":
        raise ValueError("node_support_cutoff applies to line_quadrature='node' only")


def _node_tail(request, E_grid_host):
    """The far-tail accumulator of the windowed node reduction, or None.

    NumPy/float32 CuPy backends and uniform grids only; elsewhere the full-node
    reduction runs, which is the reference the window approximates. Under
    ``"auto"`` on CuPy only the flight-grouped route windows (``windows_rows``).
    Validation: node-support-window-tail
    """
    x_c = request.node_support_cutoff
    automatic = x_c == "auto"
    if x_c is None or (
        automatic
        and (
            request.coherent or request.sinc_cutoff is not None or request.line_quadrature != "node"
        )
    ):
        return None
    if automatic:
        x_c = DEFAULT_NODE_SUPPORT_CUTOFF
    if uniform_step(E_grid_host) is None:
        return None
    n_targets = 3 if request.components else 1
    if getattr(xp, "__name__", "") == "cupy":
        if np.dtype(REAL) != np.float32:
            return None
        from ._node_window_cuda import DeviceNodeTail

        tail = DeviceNodeTail(E_grid_host, n_targets, x_c)
    else:
        tail = NodeTail(E_grid_host, n_targets, x_c)
    tail.resolved_only = automatic
    # Automatic device policy: the dense float32 per-row reducer outruns the
    # support scatter on every measured uncapped scene (#1191 qualification),
    # so only the flight-grouped route windows there. Full-node per-row
    # scoring is the reference, so this is exact for those lines.
    tail.windows_rows = not (automatic and getattr(xp, "__name__", "") == "cupy")
    return tail


def _row_tail(node_tail):
    """The accumulator the per-row route windows into, or None (dense reducer)."""
    return node_tail if node_tail is not None and node_tail.windows_rows else None


def _refuse_grouped_reduction(request, grouped):
    """Refuse routes the flight-grouped incoherent reduction does not cover."""
    if not grouped:
        return
    backend = getattr(xp, "__name__", "")
    if backend != "numpy" and not (backend == "cupy" and np.dtype(REAL) == np.float32):
        raise ValueError(
            "flight-grouped incoherent CXR runs on the NumPy backend or on the "
            "float32 CuPy segmented kernel only; this device configuration "
            f"({backend}, {np.dtype(REAL).name}) has no port. Run the spectrum on "
            "the NumPy backend, unset PYRITE_FP64, or transport without max_dE_frac."
        )
    if request.layers is not None:
        # Same unmodelled per-layer delta as the coherent path: the grouped
        # reduction carries the dispersive propagation phase too.
        raise NotImplementedError(
            "the in-medium dispersion does not cover numerical substeps through "
            "a LAYERED absorber: the intra-flight coherent sum needs the "
            "per-layer dispersive propagation phase, which is not modelled."
        )
    if request.components:
        raise ValueError(
            "components=True is incompatible with numerical substeps: the "
            "PXR/CBS split is ambiguous once a flight's substep fields add "
            "coherently (the A_PXR*A_CBS cross term survives). Request the "
            "total, or transport without max_dE_frac."
        )


def _bin_mean_axis(request, grouped, E_grid_eV):
    """Bin edges and inverse widths for ``line_quadrature='bin-mean'``, else ``(None, None)``.

    Closed-form bin integration covers the incoherent, per-row intensity sum
    only. Each refusal names a reduction whose square is taken after a sum of
    amplitudes, or a truncation the bin masses would silently disagree with.

    Validation: sinc-bin-integration
    """
    if validate_cxr_quadrature(request.line_quadrature) != BIN_MEAN_QUADRATURE:
        return None, None
    if request.coherent:
        raise ValueError(
            "line_quadrature='bin-mean' is incoherent-only: the coherent route "
            "squares a sum of amplitudes, which has no per-line bin mass (#117)"
        )
    if grouped:
        raise ValueError(
            "line_quadrature='bin-mean' is incompatible with numerical substeps: "
            "flight-grouped rows add amplitudes before squaring. Transport "
            "without max_dE_frac, or use line_quadrature='node'."
        )
    if request.sinc_cutoff is not None:
        raise ValueError(
            "line_quadrature='bin-mean' integrates each line's whole profile; "
            "sinc_cutoff truncation is not supported with it"
        )
    return bin_axis(E_grid_eV)


@dataclass(frozen=True)
class _CoherentOffsets:
    """Offset-free geometry and population terms of the coherent decoherence blend.

    Absent (all ``None``/``False``) off the coherent route; ``seg_r_geom`` and
    ``d_all_geom`` are the sampled geometry itself when no offset is configured.
    """

    seg_r_geom: Any = None
    d_all_geom: Any = None
    xy0_pop: Any = None
    decoherence_active: bool = False
    finite_footprint_now: Any = None
    analytic_offsets: bool = False
    finite_footprint_F: Any = None
    transverse: Any = None
    decoherence_A_pop: Any = None


_NO_COHERENT_OFFSETS = _CoherentOffsets()


def _electron_rows(seg_elec_id, Ne):
    """Each segment's electron index into the ``Ne`` population arrays."""
    return xp.clip(seg_elec_id, 0, max(Ne - 1, 0))


def _offset_free_geometry(seg_r, seg_elec_id, xy0_pop, Ne, seg_t_mid, n_hat_d):
    """Segment midpoints and retardation with each electron's face offset removed."""
    seg_r_geom = seg_r.copy()
    seg_r_geom[:, :2] = seg_r_geom[:, :2] - xy0_pop[_electron_rows(seg_elec_id, Ne)]
    return seg_r_geom, seg_t_mid - _matvec3(seg_r_geom, n_hat_d)


def _longitudinal_rms_fs(value, segments):
    """The configured Gaussian bunch length [fs] as a float, or ``None``."""
    if value is None:
        return None
    value = float(value)
    if (
        not np.isfinite(value)
        or value < 0.0
        or (value == 0.0 and not (segments.get("beam_entry") or {}).get("bunch_front_gradient"))
    ):
        raise ValueError(
            "longitudinal_rms_fs must be finite and positive "
            "(zero is supported with an imposed bunch front)"
        )
    return value


def _footprint_transverse_spot(
    segments, xy0_pop, finite_footprint_F_host, finite_footprint_now, seg_r_geom, n_hat
):
    """The recorded Gaussian spot a surviving transverse cross term averages over.

    ``None`` when no transverse offset is sampled or F_z vanishes everywhere.
    """
    if not (
        xy0_pop.size
        and bool(xp.any(xy0_pop != 0.0))
        and bool(np.any(finite_footprint_F_host > 0.0))
    ):
        return None
    transverse = transverse_spot(segments.get("beam_entry"))
    if transverse is None:
        raise ValueError(
            "coherent emission with a finite crystal footprint and "
            "transverse offsets requires a recorded Gaussian beam "
            "spot (transport beam_entry) unless F_z vanishes"
        )
    if int(segments.get("n_missed", 0) or 0):
        raise ValueError(
            "finite-footprint coherent transverse average needs every "
            "incident electron to hit the footprint"
        )
    if finite_footprint_now:
        require_footprint_support(
            transverse,
            seg_r_geom,
            n_hat,
            segments["thickness_ang"],
            segments["crystal_width_ang"],
            segments["crystal_height_ang"],
            xp=xp,
        )
    return transverse


def _coherent_offsets(
    request, segments, seg_r, seg_elec_id, seg_t_mid, omega_grid, n_hat, n_hat_d, Ne
):
    """Empirical inter-electron decoherence terms for the coherent path.

    Squaring one realization of the sampled per-electron offsets is speckle, not
    shot noise: it does not shrink with electron count. Blend instead, per (row,
    energy), Total = (1-F)*Grouped + F*Flat, where Flat is today's reduction fed
    the electron's OFFSET-FREE position/time (hence the d_all_geom/seg_r_geom
    swap, a strict no-op when no offset is configured), Grouped = sum_e |S_e|^2
    groups the same segments by electron, and F is the empirical characteristic
    function of the offset population transport already drew. Derivation and
    the closed form it converges to:
    docs/physics/radiation-physics/coherent-emission.md

    F must be applied PER ROW (reflection x mosaic orientation), before summing
    across rows: q_perp depends on g, so a single scalar F(E) on the row-summed
    spectrum would be wrong wherever two rows share an energy bin.

    Validation: coherent-inter-electron-decoherence
    """
    longitudinal_rms_fs = _longitudinal_rms_fs(request.longitudinal_rms_fs, segments)
    seg_r_geom = seg_r
    d_all_geom = seg_t_mid - _matvec3(seg_r, n_hat_d)
    t0_pop = xp.asarray(segments.get("initial_t0_ang", np.zeros(0)), dtype=REAL)[:Ne]
    xy0_pop = xp.asarray(
        np.asarray(segments.get("initial_r_ang", np.zeros((0, 3))))[:, :2], dtype=REAL
    )[:Ne]
    decoherence_active = bool(
        (t0_pop.size and xp.any(t0_pop != 0.0)) or (xy0_pop.size and xp.any(xy0_pop != 0.0))
    )
    if not decoherence_active:
        return _CoherentOffsets(seg_r_geom, d_all_geom, xy0_pop)
    finite_footprint_now = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    analytic_offsets = bool(
        finite_footprint_now or (segments.get("beam_entry") or {}).get("bunch_front_gradient")
    )
    _refuse_grooved_offset_factorization(request, xy0_pop, analytic_offsets, xp)
    if not analytic_offsets:
        seg_r_geom, d_all_geom = _offset_free_geometry(
            seg_r, seg_elec_id, xy0_pop, Ne, seg_t_mid, n_hat_d
        )
        # A_e = t0_e - n_hat_perp . dr_perp,e does not depend on g (the
        # reciprocal vector varies per row; n_hat is fixed for the whole
        # call), so hoist it once here; B_e(row) = g_perp . dr_perp,e is
        # cheap and stays inside the per-row helper.
        return _CoherentOffsets(
            seg_r_geom,
            d_all_geom,
            xy0_pop,
            decoherence_active,
            finite_footprint_now,
            decoherence_A_pop=t0_pop - xy0_pop @ n_hat_d[:2],
        )
    if longitudinal_rms_fs is None:
        raise ValueError(
            "analytic coherent averaging of sampled "
            "bunch offsets needs a Gaussian bunch length to average them "
            "analytically: set Beam(longitudinal=LongitudinalDistribution('gaussian', ...)), "
            "or use Slab(width=None, height=None) for a laterally infinite crystal"
        )
    sigma_z_ang = longitudinal_rms_fs * C_ANG_PER_FS
    # Decide support in host float64 so a borderline duration does not run on
    # float32 CUDA while the CPU rejects it.
    omega_host = np.asarray(_to_cpu(omega_grid), dtype=float)
    finite_footprint_F_host = np.exp(-((omega_host * sigma_z_ang) ** 2))
    finite_footprint_F = xp.asarray(finite_footprint_F_host, dtype=REAL)
    # Offset-free fields: remove each electron's face point and full arrival
    # time (bunch offset plus face-arrival delay). Amplitudes and escape keep
    # the sampled positions. Any surviving cross term then needs translation
    # invariance: a recorded Gaussian spot well inside the footprint, averaged
    # analytically per row as F_z * F_perp. F_z = 0 everywhere leaves only
    # phase-free self terms, so no spot is needed.
    # Validation: transverse-bunch-form-factor
    seg_r_geom, d_all_geom = _offset_free_geometry(
        seg_r, seg_elec_id, xy0_pop, Ne, seg_t_mid, n_hat_d
    )
    transverse = _footprint_transverse_spot(
        segments, xy0_pop, finite_footprint_F_host, finite_footprint_now, seg_r_geom, n_hat
    )
    return _CoherentOffsets(
        seg_r_geom,
        d_all_geom,
        xy0_pop,
        decoherence_active,
        finite_footprint_now,
        analytic_offsets,
        finite_footprint_F,
        transverse,
    )


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

    Validation: coherent-formation-absorption, line-kinematics-precision
    Validation: sinc-bin-integration
    Validation: transverse-bunch-form-factor
    """
    segments = request.segments
    E_grid_eV = request.E_grid_eV
    crystal = request.crystal
    theta_obs_rad = request.theta_obs_rad
    use_henke = request.use_henke
    absorber_element = request.absorber_element
    n_hat = request.n_hat
    composition = request.composition
    beam_uvw = request.beam_uvw
    azimuth_rad = request.azimuth_rad
    recip_miscut_rad = request.recip_miscut_rad
    layers = request.layers
    mosaic_fwhm_rad = request.mosaic_fwhm_rad
    mosaic_nodes = request.mosaic_nodes
    surface_hkl = request.surface_hkl
    groove = request.groove
    coherent = request.coherent
    electron_limit = request.electron_limit
    E_cut_keV = request.E_cut_keV

    _refuse_incompatible_request(request)
    # Precompute that only some routes bind. As a closure these names simply
    # went unbound on the paths that never read them; as dataclass fields they
    # need an explicit absent value. Nothing below reads one where the original
    # would have raised UnboundLocalError.
    cdtype = omega_grid = delta_omega_grid = d_all = seg_t_mid = None
    grouped_phase64 = None

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

    if coherent and request.physical_electrons is not None:
        pair_scale(request.physical_electrons, Ne)

    n_hat = _observation_direction(theta_obs_rad, n_hat)
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        _validate_groove_escape_direction(n_hat, groove)
    # Fail closed on an output grid the working precision cannot resolve. Every
    # route treats the grid as strictly ascending: window bounds and line-table
    # interpolation both search its coordinates. A grid whose step is finer
    # than REAL's spacing at these energies
    # collapses adjacent nodes on the cast -- a 2.5e-5 eV step at 1600 eV
    # loses 80% of its nodes to float32's 1.2e-4 eV ulp, making the coordinate
    # axis non-injective and therefore the spectrum
    # ambiguous. Checked on the host input, before the upload, so it
    # costs no device synchronization.
    E_grid_host = np.asarray(E_grid_eV, dtype=REAL)
    if E_grid_host.size > 1:
        collapsed = int(np.count_nonzero(np.diff(E_grid_host) <= 0))
        if collapsed:
            raise ValueError(
                f"E_grid_eV must be strictly increasing in the backend's working "
                f"precision (REAL={np.dtype(REAL).name}), but {collapsed} of "
                f"{E_grid_host.size - 1} adjacent nodes are not separated after the "
                f"cast. Coarsen the grid to at least the precision's spacing at "
                f"these energies, or run the spectrum on the NumPy backend "
                f"(PYRITE_MC_BACKEND=cpu) or with PYRITE_FP64=1."
            )
    E_grid = xp.asarray(E_grid_host)
    spec = xp.zeros(E_grid.size, dtype=REAL)
    spec_pxr = xp.zeros(E_grid.size, dtype=REAL)
    spec_cbs = xp.zeros(E_grid.size, dtype=REAL)

    n_rows = int(xp.asarray(segments["L_ang"]).shape[0])

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
    if not coherent and segments.get("flight_id") is not None and n_rows:
        grouped, gid_all = _flight_groups(segments)
    _refuse_grouped_reduction(request, grouped)

    # The phased-field reductions (coherent, flight-grouped) evaluate each
    # segment's exact complex formation integral under absorption, which needs
    # the escape path affine along a row. Split every segment at its linear
    # escape pieces into collinear rows -- each with its own length, midpoint,
    # and start age -- and carry the escape distance at both ends. The pieces
    # of a straight flight sum exactly to its field, and grouping keys are
    # gathered with them so a flight's pieces still add coherently. Incoherent
    # pieces also need their own face root, but retain the parent duration and
    # fraction-weighted transmission; layered attenuation is evaluated below.
    # Validation: coherent-formation-absorption
    escape_ends = None
    if n_rows:
        segments, piece_owner, esc_start, esc_end = expand_escape_pieces(
            segments, n_hat, layers=layers, groove=groove, keep_f64=grouped, xp=xp
        )
        if coherent or grouped:
            escape_ends = (esc_start, esc_end)
        if gid_all is not None:
            gid_all = gid_all[piece_owner]

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
    cxr_electron = seg_elec_id < Ne
    beta_all = beta_from_keV(seg_E)  # speed/c per segment
    v_all = beta_all[:, None] * seg_v  # velocity vectors (c=1)

    bin_edges, bin_inv_width = _bin_mean_axis(request, grouped, E_grid_eV)
    node_tail = _node_tail(request, E_grid_host)

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
    # Root iteration reads NaN refraction past its support when use_henke is
    # enabled. Keep the finite prefix and NaN boundary to preserve all delta
    # reads. Static coupling/attenuation suffixes can differ; final-root
    # susceptibility remains NaN there, preserving existing score NaN masks.
    # Without anomalous refraction retain the original all-table bound.
    # Validation: line-tabulation-nan-ceiling
    _ceiling = _cxr_setup_ceiling(info, abs_comp, use_henke)
    if _ceiling < _hi:
        _hi = max(_ceiling, _lo + 1.0)
    E_tab = _cxr_tabulation_grid(info, abs_comp, _lo, _hi)
    E_tab_g = xp.asarray(E_tab, dtype=REAL)
    # chi/U re+im per reflection, log(mu) per absorber, Re n.
    _table_rows = 4 * len(request.hkl_list) + len(abs_comp) + 1
    SETUP_STATS["line_tab_points"] = max(SETUP_STATS.get("line_tab_points", 0), int(E_tab.size))
    SETUP_STATS["line_table_mib"] = max(
        SETUP_STATS.get("line_table_mib", 0.0),
        _table_rows * E_tab.size * np.dtype(REAL).itemsize / 2**20,
    )
    _admit_line_tables(len(request.hkl_list), E_tab.size)
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

    # Decrement delta = 1 - Re n of the crystal's bulk refractive index
    # n(E) = sqrt(1 + chi_0(E)), tabulated on the SAME edge-resolved grid as
    # chi/U/mu (delta has its own edge structure, from the f1 cusp). Formed in
    # float64 before the cast: Re n itself rounds delta ~1e-6 to ~6% in float32.
    # The in-medium dispersion is unconditional, so this table is always built.
    # Validation: line-kinematics-precision
    delta_tab_g = xp.asarray(
        1.0 - np.asarray(refractive_index(crystal, E_tab, use_henke).real, dtype=np.float64),
        dtype=REAL,
    )

    n_hat_d = xp.asarray(n_hat, dtype=REAL)  # detector dir is g-independent: hoist

    # Segment-only kinematics shared by every reflection/orientation and by both
    # the batched and compatibility paths. These used to be recomputed inside
    # ``_accumulate`` for every g row.
    # Validation: line-kinematics-precision
    gamma_all, denom_all = _vacuum_doppler(seg_E, beta_all, seg_v, n_hat_d)
    t_L_all = seg_L / beta_all
    piece_fraction = xp.ones_like(seg_L)
    if not (coherent or grouped):
        t_L_all = xp.asarray(segments.get("line_parent_L_ang", seg_L), dtype=REAL) / beta_all
        piece_fraction = xp.asarray(segments.get("line_piece_fraction", piece_fraction), dtype=REAL)
    # On each affine piece L(r), grad L = -e/(e.n_hat). Tangential
    # continuity then gives k_eff/omega = n_hat + delta grad L.
    # Validation: xray-in-medium-resonance
    escape_gradient = segment_escape_gradient(segments, n_hat, groove=groove, xp=xp)

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
        grouped_phase64 = _grouped_phase64(segments, t_L_all, n_hat, grouped)

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

    # Inter-electron decoherence terms; see _coherent_offsets.
    flat_omission_limit = float(request.coherent_flat_omission_limit)
    if request.phase_retention is not None:
        from ..phase_retention import PhaseRetentionPolicy

        if not isinstance(request.phase_retention, PhaseRetentionPolicy):
            raise ValueError("phase_retention must be a PhaseRetentionPolicy or None")
    if not np.isfinite(flat_omission_limit) or flat_omission_limit < 0.0:
        raise ValueError("coherent_flat_omission_limit must be finite and nonnegative")
    offsets = (
        _coherent_offsets(
            request, segments, seg_r, seg_elec_id, seg_t_mid, omega_grid, n_hat, n_hat_d, Ne
        )
        if coherent
        else _NO_COHERENT_OFFSETS
    )

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
    # Incoherent emission scores the segment mean of exp(-tau), not its
    # midpoint value (issue #181, as #176 did for characteristic lines). The
    # escape pieces are g-independent, so both routes share one pass; mu
    # varies per (segment, reflection) and is applied by the route. The
    # coherent route and the flight-grouped reduction -- a coherent sum per
    # flight -- instead carry ``escape_ends`` on their piece rows (above).
    escape_pieces = (
        None
        if coherent or grouped
        else segment_escape_pieces(segments, n_hat, layers=layers, groove=groove, xp=xp)
    )

    seg_r_geom, d_all_geom = offsets.seg_r_geom, offsets.d_all_geom
    if coherent and request.physical_electrons is not None and not offsets.analytic_offsets:
        # Pair complete electron fields, including their sampled offsets. This
        # remains unbiased when phase offsets and trajectory fields correlate.
        # Validation: coherent-physical-bunch-population
        seg_r_geom = seg_r
        d_all_geom = d_all

    temporal_buf = temporal_tau = temporal_tau_geo = None
    if request.temporal is not None:
        temporal_buf = request.temporal.buffer()
        temporal_tau, _ = segment_arrival_times(segments, n_hat, xp=xp)
        temporal_tau_geo = temporal_tau
        if offsets.decoherence_active and (
            request.physical_electrons is None or offsets.analytic_offsets
        ):
            # Same offset-free arrival as ``d_all_geom``, in float64.
            t0_seg = xp.asarray(
                segments.get("t0_ang", np.zeros(temporal_tau.size)),
                dtype=np.float64,
            )
            temporal_tau_geo = temporal_tau - t0_seg
            xy0 = xp.asarray(offsets.xy0_pop, dtype=np.float64)[_electron_rows(seg_elec_id, Ne)]
            n64 = xp.asarray(n_hat_d, dtype=np.float64)
            temporal_tau_geo = temporal_tau_geo + xy0 @ n64[:2]

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
        cxr_electron=cxr_electron,
        v_all=v_all,
        denom_all=denom_all,
        gamma_all=gamma_all,
        t_L_all=t_L_all,
        gid_all=gid_all,
        grouped=grouped,
        E_tab=E_tab,
        E_tab_g=E_tab_g,
        log_mu_tab_g=log_mu_tab_g,
        delta_tab_g=delta_tab_g,
        mosaic_quad=mosaic_quad,
        finite_footprint=finite_footprint,
        cdtype=cdtype,
        omega_grid=omega_grid,
        delta_omega_grid=delta_omega_grid,
        d_all=d_all,
        seg_r_geom=seg_r_geom,
        d_all_geom=d_all_geom,
        decoherence_active=offsets.decoherence_active,
        finite_footprint_now=offsets.finite_footprint_now,
        analytic_offsets=offsets.analytic_offsets,
        finite_footprint_F=offsets.finite_footprint_F,
        transverse=offsets.transverse,
        decoherence_A_pop=offsets.decoherence_A_pop,
        xy0_pop=offsets.xy0_pop,
        bin_edges=bin_edges,
        node_tail=node_tail,
        row_tail=_row_tail(node_tail),
        bin_inv_width=bin_inv_width,
        escape_pieces=escape_pieces,
        escape_ends=escape_ends,
        escape_gradient=escape_gradient,
        piece_fraction=piece_fraction,
        seg_t_mid=seg_t_mid,
        grouped_phase64=grouped_phase64,
        temporal_buf=temporal_buf,
        temporal_tau=temporal_tau,
        temporal_tau_geo=temporal_tau_geo,
    )
