"""Batch sizing, layer-table packing, and the two per-electron transport
run loops (LUT and exact-stopping variants)."""

import logging
from dataclasses import dataclass
from functools import cache

import numpy as np

from ..._env import env_value
from .cores import (
    EXIT_BACKSCATTERED,
    EXIT_CUTOFF_STOPPED,
    EXIT_SIDE,
    EXIT_STEP_LIMITED,
    EXIT_TRANSMITTED,
)
from .kinematics import _beta_array, stream_keys
from .scattering import _elsepa_rate_scalar, _sigma_browning_cm2
from .stopping import _dEds_spliced_compound

logger = logging.getLogger(__name__)


def pack_layer_tables(
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
):
    """Pad the per-layer element lists into ``(n_layers, max_elements)`` rows.

    The lockstep core indexes a Python list of ragged arrays, which neither the
    per-electron core nor CUDA can do. Padding is zero-filled and never read:
    ``L_nel`` bounds every element loop.
    """
    n_layers = len(L_Zs)
    max_el = max(arr.size for arr in L_Zs)
    Js = np.zeros((n_layers, max_el), dtype=np.float64)
    Zs = np.zeros((n_layers, max_el), dtype=np.float64)
    ks = np.zeros((n_layers, max_el), dtype=np.float64)
    coeffs = np.zeros((n_layers, max_el), dtype=np.float64)
    E_cross = np.zeros((n_layers, max_el), dtype=np.float64)
    ncm3 = np.zeros((n_layers, max_el), dtype=np.float64)
    sr_rate_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_denom1s = np.zeros((n_layers, max_el), dtype=np.float64)
    mott_denom2s = np.zeros((n_layers, max_el), dtype=np.float64)
    sr_joy_numers = np.zeros((n_layers, max_el), dtype=np.float64)
    nel = np.zeros(n_layers, dtype=np.int32)
    for L in range(n_layers):
        n = L_Zs[L].size
        nel[L] = n
        Js[L, :n] = L_Js[L]
        Zs[L, :n] = L_Zs[L]
        ks[L, :n] = L_ks[L]
        sr_rate_numers[L, :n] = L_sr_rate_numer[L]
        mott_numers[L, :n] = L_mott_numer[L]
        mott_denom1s[L, :n] = L_mott_denom1[L]
        mott_denom2s[L, :n] = L_mott_denom2[L]
        sr_joy_numers[L, :n] = L_sr_joy_numer[L]
        coeffs[L, :n] = L_coeffs[L]
        E_cross[L, :n] = L_E_cross[L]
        ncm3[L, :n] = L_ncm3[L]

    packed_tables = (
        Js,
        Zs,
        ks,
        coeffs,
        E_cross,
        ncm3,
        sr_rate_numers,
        mott_numers,
        mott_denom1s,
        mott_denom2s,
        sr_joy_numers,
        nel,
    )

    return packed_tables


def _percentile_summary(values):
    """Compact, JSON-safe summary of one non-negative per-flight diagnostic."""
    if values.size == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    p50, p90, p99 = np.percentile(values, (50.0, 90.0, 99.0))
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(np.max(values)),
    }


def _flight_diagnostic_summary(
    E_start_keV,
    L_ang,
    elec_id,
    layer,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    L_ncm3,
    elastic_model,
    elsepa_group=None,
):
    """Estimate frozen-state transport error per physical flight, then reduce it.

    The current segment is one physical flight. Diagnostics recompute its
    left-endpoint stopping and elastic hazard at the stored start energy and at
    the predicted end energy. They do not feed values back into propagation and
    retain only fixed-size percentile summaries.
    """

    def _host(array):
        get = getattr(array, "get", None)
        return np.asarray(get() if get is not None else array)

    E_start = _host(E_start_keV).astype(float, copy=False)
    length = _host(L_ang).astype(float, copy=False)
    electron = _host(elec_id).astype(np.int64, copy=False)
    layer_index = _host(layer).astype(np.int64, copy=False)
    n_flights = E_start.size

    stopping = np.zeros(n_flights, dtype=float)
    for L, (J_arr, k_arr, coeff_arr, E_cross_arr) in enumerate(
        zip(L_Js, L_ks, L_coeffs, L_E_cross, strict=True)
    ):
        mask = layer_index == L
        E = E_start[mask]
        if E.size == 0:
            continue
        stopping[mask] = _dEds_spliced_compound(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E.copy())

    E_end = E_start + stopping * length
    hazard_start = np.zeros(n_flights, dtype=float)
    hazard_end = np.zeros(n_flights, dtype=float)
    for L, (Z_arr, ncm3_arr) in enumerate(zip(L_Zs, L_ncm3, strict=True)):
        mask = layer_index == L
        start = E_start[mask]
        end = E_end[mask]
        if start.size == 0:
            continue
        start_total = np.zeros(start.size, dtype=float)
        end_total = np.zeros(end.size, dtype=float)
        for i_el, (Z, ncm3) in enumerate(zip(Z_arr, ncm3_arr, strict=True)):
            if elastic_model == "elsepa":
                assert elsepa_group is not None
                _, el_start, el_len, el_logE, el_log_rate = elsepa_group[:5]
                for values, total in ((start, start_total), (end, end_total)):
                    for j, E in enumerate(values):
                        total[j] += _elsepa_rate_scalar(
                            E, el_logE, el_log_rate, el_start[L, i_el], el_len[L, i_el]
                        )
            elif elastic_model == "mott":
                start_total += _sigma_browning_cm2(Z, start) * ncm3
                end_total += _sigma_browning_cm2(Z, end) * ncm3
            else:
                alpha_start = 3.4e-3 * Z**0.67 / start
                alpha_end = 3.4e-3 * Z**0.67 / end
                start_rel = (start + 511.0) / (start + 1024.0)
                end_rel = (end + 511.0) / (end + 1024.0)
                start_total += (
                    ncm3
                    * 5.21e-21
                    * Z**2
                    / start**2
                    * 4.0
                    * np.pi
                    / (alpha_start * (1.0 + alpha_start))
                    * start_rel**2
                )
                end_total += (
                    ncm3
                    * 5.21e-21
                    * Z**2
                    / end**2
                    * 4.0
                    * np.pi
                    / (alpha_end * (1.0 + alpha_end))
                    * end_rel**2
                )
        hazard_start[mask] = start_total
        hazard_end[mask] = end_total

    midpoint = 0.5 * (E_start + E_end)
    left_clock = length / _beta_array(E_start)
    midpoint_clock = length / _beta_array(midpoint)
    fractional_loss = np.maximum(0.0, (E_start - E_end) / E_start)
    relative_hazard_change = np.abs(hazard_end - hazard_start) / hazard_start
    relative_clock_error = np.abs(midpoint_clock - left_clock) / midpoint_clock
    cutoff = np.asarray(E_cut_by_electrons, dtype=float)[electron]
    cutoff_overshoot = np.maximum(0.0, cutoff - E_end)

    return {
        "n_flights": int(n_flights),
        "fractional_energy_loss": _percentile_summary(fractional_loss),
        "relative_hazard_change": _percentile_summary(relative_hazard_change),
        "relative_clock_error_estimate": _percentile_summary(relative_clock_error),
        "cutoff_overshoot_keV": _percentile_summary(cutoff_overshoot),
    }


@dataclass(frozen=True)
class PerElectronTransportConfig:
    """Host-side batching policy for the per-electron cores.

    No field changes results: ``seg_capacity`` only sets how many segment slots
    each electron is given before an overflow forces a replay, and
    ``scratch_budget_bytes`` only sets how many electrons share one launch.
    Both exist because the CUDA path materializes a dense ``(batch, capacity)``
    scratch grid, and that grid must fit in VRAM alongside the case's tables.

    ``seg_capacity`` is only the *initial* guess, and the driver stops trusting it
    as soon as it has a measurement. The cores report the true segment count even
    for electrons that overflowed, so every batch -- including one that overflowed
    and has to be replayed -- says exactly what the material needs, and ``cap`` is
    reset to that running maximum times ``capacity_headroom``. Capacity trades
    against batch size out of a fixed byte budget, so an over-provisioned ``cap``
    costs launches just as an under-provisioned one costs replays.

    ``probe_electrons`` shortens the first batch, which is otherwise most of the
    run and is the one batch sized by the guess. Its segments are kept -- output
    slots are addressed by electron index, so a short first batch is just a short
    batch -- and it bounds what a wrong ``seg_capacity`` can cost to a probe rather
    than a full batch. Set it to 0 to let the first batch run full width, which is
    what a GPU run wants: there a batch costs about one electron lifetime whatever
    its width, so a discarded batch and a probe cost the same launch.
    """

    seg_capacity: int = 512
    scratch_budget_bytes: int = 512 * 1024 * 1024
    capacity_headroom: float = 1.25
    max_batch: int = 65536
    probe_electrons: int = 1024


DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG = PerElectronTransportConfig()

TRANSPORT_CORES = ("auto", "lockstep", "per-electron", "cuda")

# Electron count above which "auto" transports on the device. Measured crossover
# is Ne ~ 500-1000 for both a light (hopg) and a heavy (MoSe2) material on an
# RTX 5080: below it the launch and staging overhead outweighs the kernel, above
# it the CUDA core pulls away monotonically (4.1x at Ne=4000, 4.9-15.9x at
# Ne=16000). 1000 is the conservative end of that band, so no run that would
# have been faster on the CPU core is moved off it. See
# `docs/repo-design/compute/gpu-transport-rawkernel.md`.
CUDA_TRANSPORT_MIN_ELECTRONS = 1000


@cache
def _cuda_transport_available():
    """Whether this process can transport on the device.

    Requires the resolved spectrum backend to BE the CUDA device (a ROCm or SYCL
    backend has no `cupyx.jit` transport kernel, and a CPU backend has nowhere to
    run one) and the kernel module to import. Cached: the backend is fixed at
    import and a failed import will not start succeeding.
    """

    from ..._backend import BACKEND

    if BACKEND.name != "cuda":
        return False
    try:
        from ._jit_launch import make_cuda_transport_core  # noqa: F401
    except Exception as error:  # pragma: no cover - needs a broken CuPy install
        logger.warning("CUDA transport kernel unavailable, using the CPU core: %s", error)
        return False
    return True


def resolve_transport_core(requested, Ne, groove=None):
    """Resolve ``transport_core`` for a run of ``Ne`` electrons.

    ``"auto"`` is the default and the only value that resolves: it takes the CUDA
    core when this process has a CUDA device, the run is ungrooved, and
    ``Ne > CUDA_TRANSPORT_MIN_ELECTRONS``; otherwise the lockstep CPU core. Every
    explicit value is returned unchanged so a caller that names a core still gets
    that core or an error, never a silent substitution.

    ``PYRITE_MC_TRANSPORT_CORE`` overrides the *requested* value for the whole
    process, which is how a run pins the historical CPU core (``=lockstep``)
    without touching call sites -- reproducing a pre-existing result, or
    bisecting a device/host difference.

    Validation: gpu-transport-core
    """

    pinned = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if pinned:
        if pinned not in TRANSPORT_CORES:
            raise ValueError(
                f"PYRITE_MC_TRANSPORT_CORE must be one of {', '.join(TRANSPORT_CORES)}; got {pinned!r}"
            )
        requested = pinned
    if requested not in TRANSPORT_CORES:
        raise ValueError(f"transport_core must be one of {', '.join(TRANSPORT_CORES)}")
    if requested != "auto":
        return requested
    if groove is not None or int(Ne) <= CUDA_TRANSPORT_MIN_ELECTRONS:
        return "lockstep"
    return "cuda" if _cuda_transport_available() else "lockstep"


# float64 mid(3) + dir(3) + len + E + t0 = 9, int64 id, int16 layer.
_SEG_SCRATCH_BYTES = 9 * 8 + 8 + 2


def _batch_size(cap, config):
    per_electron = cap * _SEG_SCRATCH_BYTES
    n = int(config.scratch_budget_bytes // max(per_electron, 1))
    return max(1, min(n, int(config.max_batch)))


def _capacity_for(seen_max, config):
    """Slots per electron given the largest segment count seen so far.

    Never below ``seen_max`` whatever the headroom, so a replay is always given a
    capacity that fits and the retry loop cannot fail to make progress.
    """
    return max(1, seen_max, int(np.ceil(seen_max * config.capacity_headroom)))


def _batch_electrons(e, cap, Ne, config):
    """How many electrons the batch starting at ``e`` covers.

    Non-increasing in ``cap``, which a capacity replay relies on: the retry must
    never need more electrons than the snapshot it restores.
    """
    n = min(_batch_size(cap, config), Ne - e)
    if e == 0 and config.probe_electrons > 0:
        n = min(n, int(config.probe_electrons))
    return n


def _drive_per_electron_batches(
    core,
    core_args,
    xp,
    Ne,
    max_segments,
    energy_model_code,
    d_keys,
    d_pos,
    d_dirs,
    d_E,
    d_clock,
    d_stragg,
    straggling_args,
    out_bufs,
    to_host,
    config,
    keep_on_device,
    inelastic_args=None,
    radiative_args=None,
    secondaries=False,
):
    """Run capacity-replayed batches for either exact or LUT transport.

    ``inelastic_args`` (shell soft/hard mode only) is appended to every core
    call, and each batch then also carries the two hard-event row columns.
    ``radiative_args`` (coupled radiative mode, exact cores only) follows it,
    with ``()`` standing in for an absent shell mode, and adds the two
    hard-photon row columns after any hard-inelastic ones. ``secondaries``
    gives the shell mode's secondary-direction column rows (#94); off, that
    column is allocated empty and the core never writes it.
    """
    from ..runner import _nsys_pop, _nsys_push

    midpoint = energy_model_code == 1
    inelastic = inelastic_args is not None
    radiative = radiative_args is not None
    extra_args = (inelastic_args,) if inelastic else ()
    if radiative:
        extra_args = (inelastic_args if inelastic else (), radiative_args)
    batches = []
    cap = max(1, int(config.seg_capacity))
    seen_max = 0
    nseg = 0
    n_back = n_trans = n_side = n_cutoff = n_step_limited = 0
    e = 0
    while e < Ne:
        m = _batch_electrons(e, cap, Ne, config)
        sl = slice(e, e + m)
        # The core mutates these arrays in place. Counter-addressed streams make
        # restoring this snapshot sufficient for an exact capacity replay.
        snap = (d_pos[sl].copy(), d_dirs[sl].copy(), d_E[sl].copy(), d_clock[sl].copy())

        while True:
            _nsys_push("cxr.transport.scratch")
            scratch = _alloc_scratch(xp, m, cap, midpoint)
            if inelastic:
                scratch += _alloc_hard_scratch(xp, m, cap, secondaries)
            if radiative:
                scratch += _alloc_hard_scratch(xp, m, cap)
            seg_count = xp.zeros(m, dtype=xp.int64)
            exit_code = xp.zeros(m, dtype=xp.int8)
            _nsys_pop()
            _nsys_push("cxr.transport.launch")
            core(
                (e, m, cap, d_keys),
                *core_args,
                scratch,
                (seg_count, exit_code),
                straggling_args,
                *extra_args,
            )
            _nsys_pop()

            # CUDA launches asynchronously; this scalar read is the device sync.
            _nsys_push("cxr.transport.capsync")
            needed = int(seg_count.max())
            _nsys_pop()
            seen_max = max(seen_max, needed)
            if needed <= cap:
                break
            cap = _capacity_for(seen_max, config)
            d_pos[sl], d_dirs[sl], d_E[sl], d_clock[sl] = snap
            m = _batch_electrons(e, cap, Ne, config)
            sl = slice(e, e + m)
            snap = tuple(a[:m] for a in snap)

        total = int(seg_count.sum())
        if nseg + total > max_segments:
            raise RuntimeError("segment buffer exhausted")
        _nsys_push("cxr.transport.compact")
        keep = xp.arange(cap)[None, :] < seg_count[:, None]
        s_dir, s_mid, s_len, s_E, s_t0, s_id, s_lay = scratch[:7]
        slots = (
            s_dir.reshape(m, cap, 3),
            s_mid.reshape(m, cap, 3),
            s_len.reshape(m, cap),
            s_E.reshape(m, cap),
            s_t0.reshape(m, cap),
            s_id.reshape(m, cap),
            s_lay.reshape(m, cap),
        )
        if midpoint:
            # An empty column (secondary directions when off) stays empty.
            slots += tuple(
                a.reshape((m, cap) + a.shape[1:]) if a.shape[0] else None for a in scratch[7:]
            )
        if keep_on_device:
            batches.append(
                tuple(a[keep] if a is not None else b for a, b in zip(slots, scratch, strict=False))
            )
        else:
            dst = slice(nseg, nseg + total)
            for buf, a in zip(out_bufs, slots, strict=True):
                if a is not None:
                    buf[dst] = to_host(a[keep])
        nseg += total
        _nsys_pop()

        _nsys_push("cxr.transport.exitcodes")
        n_back += int((exit_code == EXIT_BACKSCATTERED).sum())
        n_trans += int((exit_code == EXIT_TRANSMITTED).sum())
        n_side += int((exit_code == EXIT_SIDE).sum())
        n_cutoff += int((exit_code == EXIT_CUTOFF_STOPPED).sum())
        n_step_limited += int((exit_code == EXIT_STEP_LIMITED).sum())
        _nsys_pop()
        e += m
        if seen_max > 0:
            cap = _capacity_for(seen_max, config)

    joined = None
    if keep_on_device:
        _nsys_push("cxr.transport.join")
        empty = _alloc_scratch(xp, 0, 1, midpoint)
        if inelastic:
            empty += _alloc_hard_scratch(xp, 0, 1, secondaries)
        if radiative:
            empty += _alloc_hard_scratch(xp, 0, 1)
        joined = tuple(
            xp.concatenate([b[i] for b in batches]) if batches else empty[i]
            for i in range(len(out_bufs))
        )
        _nsys_pop()

    return nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, joined, to_host(d_stragg)


def _run_per_electron_transport_lut(
    core,
    xp,
    Ne,
    seed,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    alive,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_event,
    stragg_layer_tables,
    straggle_on,
    stragg_dE,
    config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_on_device=False,
    inelastic=None,
    keys=None,
):
    """Drive the CPU/CUDA LUT per-electron core with capacity replay.

    ``stragg_layer_tables`` is the ``(L_Js, L_Zs, L_ks, L_coeffs, L_E_cross)``
    padded per-element tables (slice D straggling; see
    :func:`_transport_core_ungrooved_perelectron_lut`), ``straggle_on`` gates
    it, and ``stragg_dE`` is the Ne-sized per-electron accumulator the core
    writes into -- downloaded once at the end like the compacted segments,
    since it lives outside the capacity-replay scratch/compaction path.
    """
    on_device = xp is not np
    to_dev = xp.asarray if on_device else (lambda a: a)
    to_host = xp.asnumpy if on_device else (lambda a: a)

    from ..runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.upload")
    d_keys = to_dev(stream_keys(seed, Ne) if keys is None else keys)
    d_alive = to_dev(alive)
    d_clock = to_dev(clock)
    d_pos = to_dev(pos)
    d_dirs = to_dev(dirs)
    d_E = to_dev(E_keV)
    d_E_cut = to_dev(E_cut_by_electrons)
    d_bounds = to_dev(internal_bounds)
    d_nel = to_dev(L_nel)
    d_top = to_dev(L_top)
    d_bot = to_dev(L_bot)
    d_total_rate = to_dev(lut.total_rate)
    d_dEds = to_dev(lut.dEds)
    d_inv_beta = to_dev(lut.inv_beta)
    d_cdf = to_dev(lut.cdf)
    d_alpha = to_dev(lut.alpha)
    d_stragg_layers = tuple(to_dev(a) for a in stragg_layer_tables)
    d_stragg = to_dev(stragg_dE)
    _nsys_pop()

    # Grouped kernel argument tuples (issue #66), built once and passed through;
    # the core never constructs them. ``scratch`` IS the segments tuple.
    control = (max_steps, max_segments, elastic_model_code, energy_model_code, max_dE_frac)
    geometry = (
        n_layers,
        d_bounds,
        z_total,
        finite_footprint,
        width_ang,
        height_ang,
        d_nel,
        d_top,
        d_bot,
    )
    lut_args = (
        lut.log_E_min,
        lut.inv_dlogE,
        lut.n_energy,
        d_total_rate,
        d_dEds,
        d_inv_beta,
        d_cdf,
        d_alpha,
    )
    state = (d_alive, d_clock, d_pos, d_dirs, d_E, d_E_cut)
    straggling_args = (straggle_on, d_stragg)

    out_bufs = (seg_dir, seg_mid, seg_len, seg_E, seg_t0, seg_id, seg_lay)
    if energy_model_code == 1:
        out_bufs += (seg_E_end, seg_t_end, seg_flight, seg_substep, seg_event)
    inelastic_args = None
    if inelastic is not None:
        # ``inelastic`` is ``(core_args, seg_hard_W, seg_hard_channel,
        # seg_hard_secondary_dir, secondaries_on)``.
        inelastic_args = tuple(to_dev(a) if isinstance(a, np.ndarray) else a for a in inelastic[0])
        out_bufs += inelastic[1:4]
    core_args = (control, geometry, lut_args, d_stragg_layers, state)
    return _drive_per_electron_batches(
        core,
        core_args,
        xp,
        Ne,
        max_segments,
        energy_model_code,
        d_keys,
        d_pos,
        d_dirs,
        d_E,
        d_clock,
        d_stragg,
        straggling_args,
        out_bufs,
        to_host,
        config,
        keep_on_device,
        inelastic_args,
        secondaries=None if inelastic is None else bool(inelastic[4]),
    )


def _run_per_electron_transport(
    core,
    xp,
    Ne,
    seed,
    max_steps,
    max_segments,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    alive,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    layer_tables,
    L_top,
    L_bot,
    mott,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_event,
    straggle_on,
    stragg_dE,
    config=DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
    keep_on_device=False,
    inelastic=None,
    radiative=None,
    keys=None,
):
    """Drive ``core`` over electron batches and compact the result.

    ``straggle_on``/``stragg_dE`` are slice D's straggling gate and Ne-sized
    per-electron accumulator; see the LUT driver's docstring
    (:func:`_run_per_electron_transport_lut`) for why it is downloaded
    separately from the compacted segments.

    ``core`` is either :func:`_transport_core_ungrooved_perelectron` or the CUDA
    kernel launcher; ``xp`` is the matching array module. The two share this
    driver so batching, capacity growth, and compaction cannot drift apart.

    Output is electron-major and step-minor, which is a pure function of the
    electron index rather than of execution order, so a run is reproducible
    across batch sizes and (on CUDA) launch geometries. That addressing is also
    what lets the first batch be a short capacity probe: its segments land in the
    same slots they would have anyway, so measuring costs only the electrons it
    transports.

    Per-electron state and the material tables are moved to the device once and
    stay there; only the compacted segments of each batch cross the bus.

    ``keep_on_device`` removes even that crossing. The compacted batches are
    held in ``xp``'s own memory and joined with one ``concatenate``, which is
    returned as an eighth value for the caller to hand out in place of its
    ``seg_*`` buffers (those are then not allocated at all). Joining is what
    sizes the output, because the segment total is not known until the last
    batch has run; it costs one device-to-device pass over the payload -- against
    the pageable host copy it replaces, that is roughly two orders of magnitude
    of bandwidth -- and holds both copies of the payload while it runs.

    Returns ``(nseg, n_back, n_trans, n_side, n_cutoff, n_step_limited, joined,
    stragg_dE)``, where ``joined`` is ``None`` unless ``keep_on_device`` and
    ``stragg_dE`` is all-zero unless ``straggle_on``.
    """
    on_device = xp is not np
    to_dev = xp.asarray if on_device else (lambda a: a)
    to_host = xp.asnumpy if on_device else (lambda a: a)

    # NVTX sub-ranges splitting this driver into upload / per-batch launch /
    # capacity sync / compaction / join, so a capture attributes the transport
    # phase instead of leaving it in the unlabelled host remainder. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ..runner import _nsys_pop, _nsys_push

    _nsys_push("cxr.transport.upload")
    d_keys = to_dev(stream_keys(seed, Ne) if keys is None else keys)
    d_alive = to_dev(alive)
    d_clock = to_dev(clock)
    d_pos = to_dev(pos)
    d_dirs = to_dev(dirs)
    d_E = to_dev(E_keV)
    d_E_cut = to_dev(E_cut_by_electrons)
    d_bounds = to_dev(internal_bounds)
    d_top = to_dev(L_top)
    d_bot = to_dev(L_bot)
    d_layers = tuple(to_dev(a) if isinstance(a, np.ndarray) else a for a in layer_tables)
    d_mott = tuple(to_dev(a) for a in mott)
    d_stragg = to_dev(stragg_dE)
    _nsys_pop()

    # Grouped kernel argument tuples (issue #66), built once and passed through;
    # the core never constructs them. ``scratch`` IS the segments tuple.
    control = (max_steps, max_segments, elastic_model_code, energy_model_code, max_dE_frac)
    geometry = (
        n_layers,
        d_bounds,
        z_total,
        finite_footprint,
        width_ang,
        height_ang,
        d_layers[11],
        d_top,
        d_bot,
    )
    d_materials = d_layers[:11] + d_layers[12:]
    state = (d_alive, d_clock, d_pos, d_dirs, d_E, d_E_cut)
    straggling_args = (straggle_on, d_stragg)

    out_bufs = (seg_dir, seg_mid, seg_len, seg_E, seg_t0, seg_id, seg_lay)
    if energy_model_code == 1:
        out_bufs += (seg_E_end, seg_t_end, seg_flight, seg_substep, seg_event)
    inelastic_args = None
    if inelastic is not None:
        # ``inelastic`` is ``(core_args, seg_hard_W, seg_hard_channel,
        # seg_hard_secondary_dir, secondaries_on)``.
        inelastic_args = tuple(to_dev(a) if isinstance(a, np.ndarray) else a for a in inelastic[0])
        out_bufs += inelastic[1:4]
    radiative_args = None
    if radiative is not None:
        # ``radiative`` is ``(core_args, seg_rad_k_eV, seg_rad_Z)``.
        radiative_args = tuple(to_dev(a) if isinstance(a, np.ndarray) else a for a in radiative[0])
        out_bufs += radiative[1:]
    core_args = (control, geometry, d_materials, d_mott, state)
    return _drive_per_electron_batches(
        core,
        core_args,
        xp,
        Ne,
        max_segments,
        energy_model_code,
        d_keys,
        d_pos,
        d_dirs,
        d_E,
        d_clock,
        d_stragg,
        straggling_args,
        out_bufs,
        to_host,
        config,
        keep_on_device,
        inelastic_args,
        radiative_args,
        secondaries=None if inelastic is None else bool(inelastic[4]),
    )


def _alloc_hard_scratch(xp, m, cap, secondaries=None):
    """Hard-event row columns (float64 value, int16 code) for one batch.

    The shell soft/hard and coupled radiative modes each append one pair. Both
    modes require the midpoint schema, so these are full-width slot buffers
    after :func:`_alloc_scratch`'s twelve. The shell mode (``secondaries`` not
    None) adds the ``(n, 3)`` secondary-direction column, with zero rows unless
    secondaries are transported.
    """
    n = m * cap
    out = (xp.empty(n, dtype=xp.float64), xp.empty(n, dtype=xp.int16))
    if secondaries is not None:
        out += (xp.empty((n if secondaries else 0, 3), dtype=xp.float64),)
    return out


def _alloc_scratch(xp, m, cap, midpoint=False):
    """Slot buffers for one batch, in the segment field order.

    The five flight end-state/identity/event buffers exist only under the
    midpoint rule; frozen runs allocate them at length zero so the core
    signature stays fixed while the frozen row schema stays exactly seven fields
    wide.
    """
    n = m * cap
    n_end = n if midpoint else 0
    return (
        xp.empty((n, 3), dtype=xp.float64),
        xp.empty((n, 3), dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.float64),
        xp.empty(n, dtype=xp.int64),
        xp.empty(n, dtype=xp.int16),
        xp.empty(n_end, dtype=xp.float64),
        xp.empty(n_end, dtype=xp.float64),
        xp.empty(n_end, dtype=xp.int64),
        xp.empty(n_end, dtype=xp.int64),
        xp.empty(n_end, dtype=xp.int8),
    )
