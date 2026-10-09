"""
montecarlo.runner

Parallel case driver: the per-case transport + spectrum + brem worker
(run_case and its CPU/GPU phase split) and run_cases, which pipelines the
CPU transport across a worker pool behind the single-CUDA-context GPU phase.
The phase functions stay module-level so they pickle into Windows spawn
workers.

Validation: blazed-groove-geometry
Validation: surface-hkl-orientation
"""

import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

from ..._backend import BACKEND
from ..._energy_grid_encoding import decode_energy_grid as decode_energy_grid
from ..._env import env_value, set_canonical_env
from .. import spectrum as _spectrum_mod
from ..case import Case
from ..geometry import tilted_geometry as tilted_geometry
from ..groove import blazed_groove_spec as blazed_groove_spec
from ..spectrum import (
    _segments_in_layer,
    _segments_on_device,
    mc_spectrum,
)
from ..spectrum import (
    mc_brem_spectrum as mc_brem_spectrum,
)
from ..spectrum import (
    mc_characteristic_spectrum as mc_characteristic_spectrum,
)
from ..spectrum.lines import _setup as _line_setup
from ..spectrum.lines._temporal import case_temporal_profiles, temporal_outputs
from ..trajectories import TrajectoryCapture
from ..transport import (
    TransportLUTConfig,
    simulate_trajectories,
)

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/repo-design/compute/compute-performance-optimization.md). With PYRITE_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = env_value("PYRITE_MC_TIMING", "") not in ("", "0")
_WORKER_INHERITED_CORE: str | None = None
# Per-batch transport progress sink (#272): ``callable(electrons_done, Ne)``
# set by ``run_cases`` around a case it transports in THIS process and read by
# ``_transport_case``. A context variable keeps ``run_case``/``_transport_case``
# call shapes unchanged and never reaches pool workers, whose transports report
# only per case.
_TRANSPORT_PROGRESS: ContextVar[Any] = ContextVar("pyrite_transport_progress", default=None)


from .artifacts import _case_geometry
from .case_tables import (
    _case_bremslib_table_records as _case_bremslib_table_records,
)
from .case_tables import (
    _case_bremslib_tables as _case_bremslib_tables,
)
from .case_tables import (
    _case_elastic_kwargs,
    _case_inelastic_kwargs,
    _case_radiative_kwargs,
    _case_stopping_tables,
)
from .case_tables import (
    _case_elastic_table_records as _case_elastic_table_records,
)
from .case_tables import (
    _case_stopping_table_records as _case_stopping_table_records,
)
from .case_tables import (
    cached_case_table_markers as cached_case_table_markers,
)
from .case_tables import (
    case_table_markers as case_table_markers,
)
from .chunking import (
    _EEDL_BREM_DENSE_INTERMEDIATES,
    _RESOURCE_POLICY,
    _adaptive_chunk,
    _admit_chunk,
)
from .chunking import (
    _env_chunk as _env_chunk,
)
from .chunking import (
    _real_itemsize as _real_itemsize,
)
from .directions import directional_outputs, validated_directions
from .host_cpus import _cgroup_cpu_quota as _cgroup_cpu_quota
from .host_cpus import _usable_cpus
from .line_grid import (
    check_line_truncation,
    line_truncation_audit,
    resolve_line_grid,
)
from .line_grid import longitudinal_rms_fs as _longitudinal_rms_fs

_RESOURCE_POLICY.n_cpus = _usable_cpus()


def _nsys_range(message):
    """Return an NVTX range when the remote Nsight profiler is enabled."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return nullcontext()
    from cupyx.profiler import time_range

    return time_range(message)


def _nsys_push(message):
    """Open an NVTX range (paired with :func:`_nsys_pop`) when Nsight is on.

    A push/pop pair instead of :func:`_nsys_range` for bracketing a straight-
    line block deep in a hot loop without indenting it -- the block must have a
    single exit so the pop always runs. No-op off the profiled GPU path."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePush(message)


def _nsys_pop():
    """Close the range opened by the matching :func:`_nsys_push`."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePop()


def _process_pool_kwargs():
    """Use exec-based workers under Nsight; forkserver can deadlock its injection."""
    if not _RESOURCE_POLICY.nsys:
        return {}
    import multiprocessing

    return {"mp_context": multiprocessing.get_context("spawn")}


from .block_transport import transport_case_blocks
from .electron_blocks import (
    MAX_ELECTRON_BLOCKS,
    device_headroom_bytes,
    electron_block_count,
    iter_electron_blocks,
    restore_audit,
    snapshot_audit,
)
from .oom import (
    _ensure_pool_limit as _ensure_pool_limit,
)
from .oom import (
    _maybe_free_pool,
    _SpectrumPhaseOOM,
)
from .oom import _should_free as _should_free
from .step_budget import (
    TRANSPORT_MAX_STEPS as TRANSPORT_MAX_STEPS,
)
from .step_budget import (
    TRANSPORT_MAX_STEPS_CEILING as TRANSPORT_MAX_STEPS_CEILING,
)
from .step_budget import retry_step_budget
from .timing import _TimingAgg as _TimingAgg


def _is_gpu_oom(error):
    return isinstance(error, _RESOURCE_POLICY.gpu_oom) or BACKEND.is_oom_error(error)


def run_case(
    case: Case | Mapping[str, Any],
    record_timing: bool = False,
    keep_segments_on_device: bool = False,
    transport_core: str = "auto",
    trajectory_capture: TrajectoryCapture | None = None,
    observation_directions=None,
) -> dict[str, Any]:
    """Run transport, line emission, and bremsstrahlung for one case.

    :class:`~pyrite.montecarlo.case.Case` owns the input schema and units;
    arbitrary mappings remain accepted for one compatibility window. Electron
    energies and cutoffs are in keV, photon grids in eV, lengths in angstrom
    unless suffixed ``_mm``, angles follow their suffix, and solid angle is sr.

    ``transport_core="auto"`` lets the electron count and backend select the
    core. ``keep_segments_on_device`` avoids a host round trip only when CUDA
    transport and spectrum execute in one process; an allocation failure falls
    back to downloaded segments without changing the result.

    Parameters
    ----------
    case
        Validated typed case; a mapping remains accepted for compatibility.
    record_timing
        Include internal transport and spectrum timing metrics.
    keep_segments_on_device
        Keep CUDA transport segments on-device for same-process spectra.
    transport_core
        ``"auto"``, ``"lockstep"``, ``"per-electron"``, or ``"cuda"``.
    trajectory_capture
        Opt-in writer of this case's transport result; ``None`` (default)
        writes nothing. Capture never changes transport draws or spectra.
    observation_directions
        Optional ``(N, 3)`` sample-frame unit vectors of a physical detector.
        When given, the output also carries ``"directional"``: spectra at every
        direction from the same transport (see :func:`run_case_directions`).
        The scalar arrays are unchanged by it.

    Returns
    -------
    dict
        Line and bremsstrahlung arrays and grids, transport fractions, segment
        count, resolved crystal, incident energy, and optional timing metrics.
    """
    _adaptive.validate_directions(case, observation_directions)
    transport = _transport_case(
        case,
        record_timing,
        transport_core=transport_core,
        keep_segments_on_device=keep_segments_on_device,
        trajectory_capture=trajectory_capture,
    )
    out = _spectrum_case(case, transport, record_timing)
    if observation_directions is not None:
        out["directional"] = _directional_outputs(case, transport, observation_directions)
    return out


def _directional_outputs(case, transport, n_hats, spectrum=None) -> dict[str, Any]:
    """Evaluate observation directions on one transport; see :mod:`.directions`."""
    return directional_outputs(
        case, transport, n_hats, _spectrum_case if spectrum is None else spectrum
    )


def run_case_directions(
    case: Case | Mapping[str, Any],
    n_hats,
    *,
    transport_core: str = "auto",
) -> dict[str, Any]:
    """Evaluate spectra at multiple sample-frame directions after one transport.

    Returned ``spec_by_direction`` and ``brem_wide_by_direction`` have leading
    direction dimension. ``spec_coherent_by_direction`` is present when the
    case requests coherent emission. Every direction consumes the same
    transported electron segments; this function never places downstream
    photon geometry in the electron navigator.
    """
    _adaptive.validate_directions(case, n_hats)
    directions = validated_directions(n_hats)
    transport = _transport_case(
        case,
        transport_core=transport_core,
        keep_segments_on_device=True,
    )
    return _directional_outputs(case, transport, directions)


def _beam_kwargs(case):
    """Beam phase-space kwargs a case dict forwards to ``simulate_trajectories``:
    the transverse spot (elliptical ``beam_fwhm_mm`` / ``beam_fwhm_y_mm``) and
    the legacy longitudinal bunch (``bunch_length_fs`` / ``long_shape`` /
    ``long_offsets_fs``) or resolved ``longitudinal_distribution`` policy, the
    resolved ``transverse_distribution`` (Twiss) policy that replaces the spot,
    and ``energy_spread_frac``.
    Absent keys default to the point-bunch isotropic beam, bit-for-bit with the
    pre-BeamSpec case dict."""
    return dict(
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        beam_fwhm_y_mm=case.get("beam_fwhm_y_mm"),
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
        transverse_distribution=case.get("transverse_distribution"),
        energy_spread_frac=case.get("energy_spread_frac"),
        **({"gdf_source": case["gdf_source"]} if "gdf_source" in case else {}),
    )


def _transport_case(
    case,
    record_timing=False,
    transport_core="auto",
    keep_segments_on_device=False,
    trajectory_capture=None,
    block_electrons=None,
    block_monitor=None,
):
    """Transport phase of run_case: the line + brem trajectories. Returns the
    segments + geometry + grids the spectrum phase consumes.

    Runs on the CPU (pure numpy/numba) unless the case resolves to the CUDA core
    -- see :func:`_case_transport_core`. run_cases farms the CPU core out to a
    worker pool so the transport of upcoming cases overlaps the GPU work on the
    current one; it pins ``transport_core="lockstep"`` when it does, because a
    pool of worker processes must not each open a CUDA context on the one device
    this process is already driving.

    keep_segments_on_device: see :func:`run_case`. Requested only where transport
    and spectrum share a process.

    trajectory_capture: optional ``TrajectoryCapture``; writes the result here,
    in whichever process transported it, before the spectrum phase sees it.

    block_electrons: internal (#361). Transport ``[0, max(Ne, Ne_brem))`` in
    electron blocks of this size on the per-electron/CUDA core; the joined
    segments equal the single-call run bit for bit (see
    :mod:`.block_transport`). ``None`` keeps the single call. The lockstep
    core is rejected.

    block_monitor: internal (#361), with ``block_electrons`` and
    ``Ne == Ne_brem``. Its hooks see every block and may stop transport early
    (:mod:`.adaptive`); the case then continues as the fixed-N case at the
    realized count -- cutoffs, line grid and spectrum inputs alike.

    Validation: grazing-beam-projection
    """
    if case.get("adaptive_precision") is not None:
        return _adaptive.transport_requested_precision(
            case,
            record_timing,
            transport_core=transport_core,
            keep_segments_on_device=keep_segments_on_device,
            trajectory_capture=trajectory_capture,
            block_electrons=block_electrons,
            block_monitor=block_monitor,
        )
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    E_grid, E_brem, beam, n_hat, groove = _case_geometry(case)
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    # film-on-substrate stack drives multilayer transport too (substrate
    # backscatter / substrate brem); None -> single-material slab (unchanged).
    layers = case.get("abs_layers")
    beam_kw = _beam_kwargs(case)

    Ne = case["Ne"]
    Ne_brem = case["Ne_brem"]
    Ne_transport = max(Ne, Ne_brem)

    electron_ids = np.arange(Ne_transport)

    line_mask = electron_ids < Ne
    brem_mask = electron_ids < Ne_brem

    combined_transport_mask = line_mask & brem_mask
    line_only_transport_mask = line_mask & ~brem_mask
    brem_only_transport_mask = brem_mask & ~line_mask

    E_cut_lines = case.get("E_cut_lines_keV", 5.0)
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)

    E_cut_by_electrons = np.empty(Ne_transport, dtype=np.float64)

    E_cut_by_electrons[line_only_transport_mask] = E_cut_lines
    E_cut_by_electrons[brem_only_transport_mask] = E_cut_brem
    E_cut_by_electrons[combined_transport_mask] = min(E_cut_lines, E_cut_brem)

    core = _case_transport_core(case, transport_core)
    # Coupled radiative rows complete their photons on the host.
    resident = keep_segments_on_device and core == "cuda" and "radiative_model" not in case
    straggling = bool(case.get("straggling", False))
    # The CUDA LUT kernel has no shell soft/hard mode, so production shell runs
    # select the exact CUDA kernel rather than failing. Direct
    # simulate_trajectories calls retain the fail-closed CUDA-LUT guard as a
    # lower-level contract.
    exact_only = case.get("inelastic_model") is not None
    transport_lut_config = (
        TransportLUTConfig(enabled=False) if core == "cuda" and exact_only else None
    )
    stopping_tables = _case_stopping_tables(case)

    def _transport(keep):
        return retry_step_budget(_simulate, keep)

    # Passed only when a caller asked, so progress-free calls keep their shape.
    transport_progress = _TRANSPORT_PROGRESS.get()
    progress_kw = {} if transport_progress is None else {"transport_progress": transport_progress}

    if block_electrons is not None and core == "lockstep":
        raise ValueError(
            "electron blocks need the per-electron or CUDA transport core; "
            f"{transport_core!r} resolved to 'lockstep'"
        )
    if block_electrons is not None and "gdf_source" in case:
        raise ValueError("electron blocks do not support GDF beams")
    if block_monitor is not None and (block_electrons is None or Ne != Ne_brem):
        raise ValueError("a block monitor needs electron blocks and Ne == Ne_brem")

    def _simulate(keep, max_steps, start=0, stop=Ne_transport, block_kw=None):
        # A block overrides progress and withholds the bunch (see below).
        call_kw = {**beam_kw, **progress_kw, **({} if block_kw is None else block_kw)}
        return simulate_trajectories(
            case["E0_keV"],
            stop - start,
            case["thickness_ang"],
            E_cut_by_electrons=E_cut_by_electrons[start:stop],
            composition=case["composition"],
            seed=case["seed"],
            beam_dir=beam,
            layers=layers,
            crystal_width_mm=case.get("crystal_width_mm"),
            crystal_height_mm=case.get("crystal_height_mm"),
            tilt_polar_rad=tilt_polar_rad,
            tilt_azim_rad=tilt_azim_rad,
            groove=groove,
            transport_core=core,
            keep_segments_on_device=keep,
            max_steps=max_steps,
            energy_model=case.get("energy_model", "frozen"),
            max_dE_frac=case.get("max_dE_frac", 0.0),
            straggling=straggling,
            stopping_tables=stopping_tables,
            **_case_inelastic_kwargs(case),
            **_case_elastic_kwargs(case),
            **_case_radiative_kwargs(case),
            **(
                {"transport_lut_config": transport_lut_config}
                if transport_lut_config is not None
                else {}
            ),
            **call_kw,
        )

    transport = _transport
    if block_electrons is not None:
        transport = partial(
            transport_case_blocks,
            simulate=_simulate,
            case=case,
            E_cut_by_electrons=E_cut_by_electrons,
            block_electrons=block_electrons,
            beam_kw=beam_kw,
            progress=transport_progress,
            monitor=block_monitor,
        )
    if resident:
        try:
            segs_all = transport(True)
        except _RESOURCE_POLICY.gpu_oom:
            # Residency holds the whole payload plus the join's second copy. The
            # streams are counter-addressed, so replaying the same seed with the
            # segments downloaded reproduces this run exactly -- the retry costs
            # the bus, not the result.
            BACKEND.release_memory()
            segs_all = transport(False)
    else:
        segs_all = transport(False)
    if block_monitor is not None and int(segs_all["Ne"]) != Ne_transport:
        # Stopped early: from here on this is the fixed-N case at the realized
        # count, whose cutoffs are the leading slice (Ne == Ne_brem).
        Ne = Ne_brem = Ne_transport = int(segs_all["Ne"])
        case = {**case, "Ne": Ne, "Ne_brem": Ne_brem}
        E_cut_by_electrons = E_cut_by_electrons[:Ne_transport]

    # Line resolution needs the transport distribution, so it is chosen after
    # the case's own trajectories exist and before the spectrum phase. No second
    # Monte Carlo job is started for either path; see runner/line_grid.py.
    E_grid, diagnostic_grid_result = resolve_line_grid(
        case, segs_all, n_hat, Ne, E_grid, layers, groove
    )

    spectrum_inputs: dict[str, Any] = dict(
        E_grid=E_grid,
        E_brem=E_brem,
        n_hat=n_hat,
        Ne_lines=Ne,
        Ne_brem=Ne_brem,
        groove=groove,
        diagnostic_grid=diagnostic_grid_result,
    )
    if trajectory_capture is not None:
        # Read-only snapshot of exactly what the spectrum phase consumes, so a
        # later spectrum_from_artifact replays it without re-transporting.
        trajectory_capture.write(
            case,
            segs_all,
            settings={
                "transport_core": core,
                "segments_on_device": not isinstance(segs_all["L_ang"], np.ndarray),
                "Ne_transport": Ne_transport,
                "E_cut_by_electrons": E_cut_by_electrons,
                "beam_dir": beam,
                "n_hat": n_hat,
                "stopping_tables": stopping_tables is not None,
            },
            spectrum_inputs=spectrum_inputs,
        )

    tp: dict[str, Any] = dict(spectrum_inputs, segs=segs_all)
    if timed:
        tp["_t_transport"] = perf_counter() - t0
        tp["_t_worker_return"] = perf_counter()

    return tp


from .emission import _brem_for_case as _brem_for_case
from .emission import _brem_wide_from_segments as _brem_wide_from_segments
from .emission import _characteristic_from_segments as _characteristic_from_segments


def _lines_for_segments(
    segs,
    E_grid,
    case,
    n_hat,
    abs_layers,
    groove,
    *,
    coherent=None,
    Ne=None,
    table_cache=None,
    truncation_audit=None,
    temporal=None,
    coefficient_capture=None,
):
    """:func:`_lines_for_segments_once` in electron-aligned device blocks (#192).

    On a GPU, an incoherent sum whose segments exceed the pool headroom runs in
    disjoint electron blocks whose spectra, audit sums and collected lines add
    (``iter_electron_blocks``). A block OOM restores the audit and doubles the block
    count up to ``MAX_ELECTRON_BLOCKS``, then re-raises to the spectrum phase's
    chunk-halving retry. A case that fits runs as one unchanged call; coherent
    sums, which may couple electrons, are never split.
    """
    once = partial(
        _lines_for_segments_once,
        E_grid=E_grid,
        case=case,
        n_hat=n_hat,
        abs_layers=abs_layers,
        groove=groove,
        coherent=coherent,
        Ne=Ne,
        table_cache=table_cache,
        truncation_audit=truncation_audit,
        temporal=temporal,
        coefficient_capture=coefficient_capture,
    )
    wants_coherent = case.get("coherent_emission", False) if coherent is None else coherent
    # An in-place temporal profile (#292) cannot be split and retried.
    if bool(wants_coherent) or temporal is not None or not _RESOURCE_POLICY.gpu:
        return once(segs)
    n_segments = int(segs["L_ang"].shape[0])
    n_blocks = max(
        electron_block_count(n_segments, device_headroom_bytes()),
        int(case.get("_min_line_electron_blocks", 1)),
    )
    while True:
        saved = snapshot_audit(truncation_audit)
        if case.get("_profile_line_grid_stages"):
            print(
                f"line-grid profile: electron_blocks={n_blocks} segments={n_segments}",
                file=sys.stderr,
                flush=True,
            )
        try:
            if n_blocks == 1:
                return once(segs)
            spec = None
            for block in iter_electron_blocks(segs, n_blocks):
                part = once(block)
                spec = part if spec is None else spec + part
            return spec
        except Exception as error:
            if not _is_gpu_oom(error) or n_blocks >= MAX_ELECTRON_BLOCKS:
                raise
            restore_audit(truncation_audit, saved)
            BACKEND.release_memory()
            n_blocks = min(MAX_ELECTRON_BLOCKS, 2 * n_blocks)


def _lines_for_segments_once(
    segs,
    E_grid,
    case,
    n_hat,
    abs_layers,
    groove,
    *,
    coherent=None,
    Ne=None,
    table_cache=None,
    truncation_audit=None,
    temporal=None,
    coefficient_capture=None,
):
    """Coherent line spectrum on ``E_grid`` from already-transported line
    segments ``segs``. Single slab (``layer_radiators`` absent) radiates from
    all segments via the case's scalar crystal keys; a multilayer stack sums
    each CRYSTALLINE layer's lines incoherently, every line self-absorbing
    through the whole stack. Pure move of _spectrum_case's line block, shared
    with :func:`_lines_for_case` so a line-only reline reproduces the SAME
    spectrum as a live sweep.

    ``coherent`` overrides the coherence of the interference kernel: ``None``
    (default) derives it from ``case["coherent_emission"]`` (back-compat);
    ``False`` forces the incoherent line sum, ``True`` the coherent one. The
    dual-spectra runner passes both flags in turn over the SAME ``segs`` so one
    transport yields both the incoherent ``spec`` and the ``spec_coherent``.

    Line kinematics always run on the crystal's bulk in-medium dispersion
    ``k = Re n(omega) omega``; there is no vacuum switch.

    ``truncation_audit`` is forwarded to every ``mc_spectrum`` call (#192), and
    so is the opt-in ``temporal`` profile, which every layer adds into (#292)."""
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(E_grid.size),
        E_grid.size,
    )
    if coherent is None:
        coherent = bool(case.get("coherent_emission", False))
    else:
        coherent = bool(coherent)
    # Divergence-only case key (#116); mc_spectrum refuses it on coherent calls.
    line_quadrature = case.get("line_quadrature", "node")
    longitudinal_rms_fs = _longitudinal_rms_fs(case)
    if radiators is None:
        return mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case.get("beam_uvw"),
            surface_hkl=case.get("surface_hkl"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            recip_miscut_rad=case.get("recip_miscut_rad"),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            groove=groove,
            coherent=coherent,
            longitudinal_rms_fs=longitudinal_rms_fs,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            _table_cache=table_cache,
            line_quadrature=line_quadrature,
            truncation_audit=truncation_audit,
            temporal=temporal,
            coefficient_capture=coefficient_capture,
            **mosaic_kw,
        )
    if coefficient_capture is not None:
        raise ValueError("coefficient_capture supports single-radiator cases only")
    assert case.get("groove_spacing_ang") is None
    spec = np.zeros(E_grid.shape, dtype=float)
    for L, rad in enumerate(radiators):
        if rad is None:
            continue
        sL = _segments_in_layer(segs, L)
        if sL["L_ang"].size == 0:
            continue
        spec = spec + mc_spectrum(
            sL,
            E_grid,
            crystal=rad["crystal"],
            hkl_list=rad["hkl_list"],
            n_hat=n_hat,
            B_ang2=rad["B_ang2"],
            composition=abs_layers[L][2],
            beam_uvw=rad.get("beam_uvw"),
            surface_hkl=rad.get("surface_hkl"),
            azimuth_rad=rad.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
            recip_miscut_rad=rad.get("recip_miscut_rad", case.get("recip_miscut_rad")),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            coherent=coherent,
            longitudinal_rms_fs=longitudinal_rms_fs,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            _table_cache=table_cache,
            line_quadrature=line_quadrature,
            truncation_audit=truncation_audit,
            temporal=temporal,
            **mosaic_kw,
        )
    return spec


def _transport_lines_for_case(case, E_grid=None):
    """Re-run the exact live transport population used by all line emitters.

    Characteristic radiation uses the lower-cutoff bremsstrahlung population,
    while PXR/CBS uses the line population. Delegating to
    :func:`_transport_case` preserves both electron counts, per-electron
    cutoffs, the seed, geometry, and transport-core selection so ``reline``
    cannot regenerate a different characteristic yield from a live run.
    """
    transport_case = case
    if "E_grid" not in case and "E_grid_line" not in case:
        if E_grid is None:
            raise ValueError("E_grid is required when the case has no encoded line grid")
        transport_case = {**case, "E_grid": E_grid}
    transport = _transport_case(transport_case)
    return (
        transport["segs"],
        transport["E_brem"],
        transport["n_hat"],
        case.get("abs_layers"),
        transport.get("groove"),
    )


def _lines_for_case(case, E_grid, *, coherent=None):
    """Regenerate a case's line spectrum on ``E_grid`` from scratch (re-transport
    + per-layer line kernel via :func:`_lines_for_segments`). Returns one
    PXR/CBS ``spec``, without characteristic emission. ``coherent`` overrides
    the kernel coherence (``None`` = derive from ``case["coherent_emission"]``). The line half of run_case's transport +
    spectrum phases factored out so :func:`pyrite.runs.run.repair_line_spec`
    (``pyrite reline``) reuses the EXACT live-sweep line path rather than
    re-deriving it by hand."""
    segs, _E_brem, n_hat, abs_layers, groove = _transport_lines_for_case(case, E_grid)
    segs = _segments_on_device(segs)
    return _lines_for_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=coherent,
        Ne=case["Ne"],
    )


def _line_pair_for_case(case, E_grid, *, want_coherent, return_characteristic=False):
    """Reline mirror of the runner's one-transport / dual-kernel invariant: one
    re-transport of ``case`` yields the incoherent ``spec`` and (when
    ``want_coherent``) a ``spec_coherent`` from the SAME segments, so a
    ``pyrite reline`` that moves a ``coherent``/``both`` checkpoint onto a new grid
    keeps both arrays on that grid instead of leaving ``spec_coherent`` stale.
    Returns ``(spec, spec_coherent_or_None)``, both PXR/CBS only. With
    ``return_characteristic=True``, append the separate characteristic
    component as a third item."""
    segs, _E_brem, n_hat, abs_layers, groove = _transport_lines_for_case(case, E_grid)
    # Both kernels read the same segments; stage one device copy as the live
    # sweep does rather than uploading the pair separately.
    segs = _segments_on_device(segs)
    Ne_lines = case["Ne"]
    table_cache = {}
    spec = _lines_for_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=False,
        Ne=Ne_lines,
        table_cache=table_cache,
    )
    spec_coherent = (
        _lines_for_segments(
            segs,
            E_grid,
            case,
            n_hat,
            abs_layers,
            groove,
            coherent=True,
            Ne=Ne_lines,
            table_cache=table_cache,
        )
        if want_coherent
        else None
    )
    if not return_characteristic:
        return spec, spec_coherent
    characteristic = _characteristic_from_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove=groove,
        Ne=case["Ne_brem"],
    )
    return spec, spec_coherent, characteristic


def _stage_counters(tp):
    """Work sizes of one case's spectrum phase, for performance telemetry.

    Sizes sit next to the phase times so a slow case shows *what* grew: the
    resolved line/brem axes, the transported segments, the line tabulation mesh
    and its tables, and host memory at the end of the phase.
    """
    counters = {
        "_line_axis_nodes": int(tp["E_grid"].size),
        "_brem_axis_nodes": int(tp["E_brem"].size),
    }
    segments = tp.get("segs")
    if isinstance(segments, Mapping) and "L_ang" in segments:
        counters["_segments"] = int(segments["L_ang"].size)
    for key, value in _line_setup.SETUP_STATS.items():
        counters[f"_{key}"] = value
    counters.update(_host_rss_mib())
    return counters


def _host_rss_mib():
    """Current and peak resident set size of this process, in MiB (Linux)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        current = pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except ImportError, OSError, ValueError, IndexError:
        return {}
    return {"_host_rss_mib": round(current, 1), "_host_rss_peak_mib": round(peak, 1)}


def _effective_spec_chunk(case, tp):
    """Resolve one case's line-spectrum chunk without changing the case."""
    return _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size),
        tp["E_grid"].size,
    )


def _effective_brem_chunk(case, tp):
    """Resolve one case's bremsstrahlung chunk without changing the case."""
    return _admit_chunk(
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES),
        tp["E_brem"].size,
        intermediates=_EEDL_BREM_DENSE_INTERMEDIATES,
    )


def _halve_case_spec_chunk(case, tp):
    """Halve this case's effective line chunk in place; preserve brem tuning."""
    spec_cur = (
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size)
    )
    case["spec_chunk"] = max(1, spec_cur // 2)


def _halve_case_brem_chunk(case, tp):
    """Halve this case's effective brem chunk in place; preserve line tuning."""
    brem_cur = (
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES)
    )
    case["brem_chunk"] = max(1, brem_cur // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)


def _spectrum_case_retry(
    case,
    tp,
    max_retries=_RESOURCE_POLICY.gpu_oom_retries,
    record_timing=False,
    spec_chunk_cap=None,
):
    """Run the GPU phase, retrying line OOMs with progressively halved chunks.

    ``spec_chunk_cap`` carries a successful fallback forward within one
    :func:`run_cases` call. It only lowers the resolved line chunk. The original
    case and its brem chunk stay pristine for checkpoint/recompute compatibility.
    OOMs outside the coherent-line phase propagate without teaching a false cap.
    """
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    work = case
    initial_spec_chunk = _effective_spec_chunk(work, tp)
    initial_brem_chunk = _effective_brem_chunk(work, tp)
    if spec_chunk_cap is not None and initial_spec_chunk > spec_chunk_cap:
        work = dict(work)
        work["spec_chunk"] = spec_chunk_cap
        initial_spec_chunk = spec_chunk_cap
    line_retries = 0
    brem_retries = 0
    generic_retries = 0
    for attempt in range(max_retries + 1):
        try:
            out = _spectrum_case(work, tp, record_timing)
            if timed:
                out["_t_spectrum"] = perf_counter() - t0
            out["_gpu_oom_retries"] = attempt
            out["_line_gpu_oom_retries"] = line_retries
            out["_brem_gpu_oom_retries"] = brem_retries
            out["_generic_gpu_oom_retries"] = generic_retries
            out["_attempted_spec_chunk"] = initial_spec_chunk
            out["_effective_spec_chunk"] = _effective_spec_chunk(work, tp)
            out["_attempted_brem_chunk"] = initial_brem_chunk
            out["_effective_brem_chunk"] = _effective_brem_chunk(work, tp)
            return out
        except _SpectrumPhaseOOM as tagged:
            if attempt == max_retries:
                raise tagged.error from tagged
            BACKEND.release_memory()
            work = dict(work)
            if tagged.phase == "line":
                line_retries += 1
                _halve_case_spec_chunk(work, tp)
            else:
                brem_retries += 1
                _halve_case_brem_chunk(work, tp)
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            if attempt == max_retries:
                raise
            BACKEND.release_memory()
            work = dict(work)
            generic_retries += 1
            _halve_case_chunks(work, tp)


def _spectrum_case(case, tp, record_timing=False):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    case = _adaptive.realized_case(case, tp)
    timed = _TIMING or record_timing
    if timed:
        _line_setup.SETUP_STATS.clear()
    with _nsys_range(f"cxr.spectrum_case:{case.get('name', 'case')}"):
        out = _spectrum_case_impl(case, tp, record_timing)
    if timed:
        out.update(_stage_counters(tp))
    _adaptive.attach_sampling(case, tp, out)
    return out


def _spectrum_case_impl(case, tp, record_timing=False):
    """Implement :func:`_spectrum_case` inside its optional Nsight range.

    Validation: temporal-intensity-profile
    """
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs = tp["segs"]
    # Every kernel below runs over the SAME segments, and each would otherwise
    # upload its own slice of them: a case pushes ~116 B/segment across the bus
    # where the union of what it reads is ~48. Stage one device copy for the
    # whole case and they all read it in place. `segs` itself stays host-side --
    # the run summary at the bottom counts segments and backscatter on it.
    segs_dev = _segments_on_device(segs)
    Ne_lines = tp["Ne_lines"]
    Ne_brem = tp["Ne_brem"]
    # optional film-on-substrate stack (None -> single slab, unchanged)
    abs_layers = case.get("abs_layers")

    # LINES: each CRYSTALLINE layer radiates its own PXR/CBS lines, summed
    # INCOHERENTLY (separate crystals -> no cross-layer coherence); every line
    # self-absorbs through the WHOLE stack (layers=abs_layers). `layer_radiators`
    # is a per-layer list aligned with the stack -- a dict of crystal params for a
    # crystalline layer (film or crystalline substrate), None for an amorphous one
    # (no coherent lines). layer_radiators absent -> single slab: the film radiates
    # from ALL its segments via the case's scalar crystal keys (bit-for-bit the
    # pre-multilayer path). See docs/physics/materials/multilayer-materials.md (per-layer radiation).
    # DUAL-SPECTRA: `spec` is ALWAYS the incoherent line sum (kept for every
    # emission mode so no `record["spec"]` consumer KeyErrors); when the profile
    # emission includes coherent (`case["coherent_emission"]` true for
    # emission="coherent"|"both") a SECOND line sum on the SAME `segs` yields
    # `spec_coherent`. One transport, two kernels -- both share `case["spec_chunk"]`
    # so the OOM retry (_SpectrumPhaseOOM -> _halve_case_spec_chunk) covers the 2x
    # complex coherent grid too.
    want_coherent = bool(case.get("coherent_emission", False))
    spec_coherent = None
    line_table_cache = {}
    # Opt-in temporal profile (#292): one accumulator per emission policy.
    temporal, temporal_coherent = case_temporal_profiles(case, tp, want_coherent)
    truncation_audit = line_truncation_audit(case, E_grid, n_electrons=Ne_lines)
    with _nsys_range("cxr.lines"):
        try:
            spec = _lines_for_segments(
                segs_dev,
                E_grid,
                case,
                n_hat,
                abs_layers,
                tp.get("groove"),
                coherent=False,
                Ne=Ne_lines,
                table_cache=line_table_cache,
                truncation_audit=truncation_audit,
                temporal=temporal,
            )

            if want_coherent:
                spec_coherent = _lines_for_segments(
                    segs_dev,
                    E_grid,
                    case,
                    n_hat,
                    abs_layers,
                    tp.get("groove"),
                    coherent=True,
                    Ne=Ne_lines,
                    table_cache=line_table_cache,
                    temporal=temporal_coherent,
                )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("line", error) from error
    # Refuse a truncating measured bandwidth before the other components run.
    truncation_record = (
        None if truncation_audit is None else check_line_truncation(case, truncation_audit)
    )

    # CHARACTERISTIC: EEDL shell-ionization track-length estimator on the
    # lower-cutoff bremsstrahlung electron population. Atomic relaxation is
    # incoherent, so one component serves both the PXR/CBS spectrum and its
    # coherent companion. It stays a separate array; consumers combine
    # components through pyrite._spectral_components (issue #123).
    with _nsys_range("cxr.characteristic"):
        try:
            spec_characteristic = _characteristic_from_segments(
                segs_dev,
                E_grid,
                case,
                n_hat,
                abs_layers,
                groove=tp.get("groove"),
                Ne=Ne_brem,
            )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("brem", error) from error

    # BREM: EVERY layer radiates with its OWN composition (each Z^2 cross
    # section); each layer's brem self-absorbs through the whole stack, summed
    # over layers (a single layer is exactly the old single-material brem).
    # Factored into _brem_wide_from_segments so run.repair_brem_wide reuses this
    # SAME path (via _brem_for_case) and can't drift back to single-slab brem.
    with _nsys_range("cxr.brem"):
        try:
            brem_wide = _brem_wide_from_segments(
                segs_dev,
                E_brem,
                case,
                n_hat,
                abs_layers,
                groove=tp.get("groove"),
                Ne=Ne_brem,
            )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("brem", error) from error
    with _nsys_range("cxr.interpolate"):
        brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Return this case's GPU scratch on the A2 cadence so the CuPy memory pool
    # can't accumulate (and fragment) across a long sweep until it fills the card.
    if _RESOURCE_POLICY.gpu:
        _maybe_free_pool()
        if timed:
            stats = BACKEND.allocator_stats()
            out_pool = {
                "_allocator_used_mib": stats["used_mib"],
                "_allocator_reserved_mib": stats["reserved_mib"],
                "_allocator_peak_mib": stats["peak_mib"],
                # Additive legacy aliases keep cxr.performance.v1 readers valid.
                "_cupy_pool_used_mib": stats["used_mib"],
                "_cupy_pool_reserved_mib": stats["reserved_mib"],
                "_cupy_pool_peak_mib": stats["peak_mib"],
                "_backend": BACKEND.name,
                "_backend_vendor": BACKEND.vendor,
                "_backend_device": BACKEND.device.name,
            }
        else:
            out_pool = {}
    else:
        out_pool = {}
    out = dict(
        E_grid=E_grid,
        spec=spec,
        spec_characteristic=spec_characteristic,
        brem=brem,
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        # Finite-crystal footprint-hit fraction: launched electrons whose
        # projected entry landed ON the transverse footprint / all launched.
        # 1.0 for the laterally infinite slab (n_missed==0 -> nothing can miss);
        # < 1 only when a finite crystal_width/height clips the beam spot. Drives
        # the analysis_app "electron hit fraction" heatmap (bright=all hit).
        hit_frac=1.0 - segs["n_missed"] / segs["Ne"],
        n_segments=int(segs["L_ang"].size),
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )
    if tp.get("diagnostic_grid") is not None:
        # Same record either way; two names because the consumers differ.
        # ``energy_grid.derive`` reads the diagnostic key; run/checkpoint
        # provenance reads the resolved key for automatic case-local grids.
        out["line_grid_diagnostic"] = tp["diagnostic_grid"]
        if case.get("line_grid_policy") is not None:
            out["line_grid_resolved"] = tp["diagnostic_grid"]
            if truncation_record is not None:
                out["line_grid_resolved"] = {
                    **tp["diagnostic_grid"],
                    "truncation_audit": truncation_record,
                }
    if spec_coherent is not None:
        out["spec_coherent"] = spec_coherent
    out.update(temporal_outputs(temporal, temporal_coherent))
    if timed:
        # Ride the phase deltas back to the driver on the result dict; run_cases'
        # _TimingAgg.collect strips both keys before the result is stored. Carry
        # _t_transport through so the CPU-pool path (where transport time only
        # exists inside this worker) can report the split too.
        out["_t_spectrum"] = perf_counter() - t0
        if "_t_transport" in tp:
            out["_t_transport"] = tp["_t_transport"]
        out.update(out_pool)
    return out


def _worker_init(force_cpu=False):
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.

    Also keeps THIS process's transport off the device. Both pools run here, and
    a worker that resolved `transport_core="auto"` onto the device would open
    exactly the per-worker CUDA context the single-context design exists to
    avoid -- and, in the transport pool, would then have to pickle its segments
    back down anyway. The pin is an environment variable because it has to reach
    every call site in the worker, including the ones that never see a run_cases
    argument; it is process-local (spawn/fork copies the environment) and never
    touches the driver's own resolution.

    Only "cuda" and "auto" are redirected. An inherited PYRITE_MC_TRANSPORT_CORE
    naming a CPU core is a deliberate choice that a worker can honor, and
    overwriting it made the pin a no-op for every pooled run: a sweep pinned to
    "per-electron" silently transported on lockstep instead, which a 2026-08-08
    remote-host A/B caught only because the two arms came out bit-identical.

    force_cpu: when True (the engine="cpu" full-case pool), rebind THIS
    worker process's copy of runner._RESOURCE_POLICY.gpu and spectrum.xp/REAL to their CPU
    equivalents, so _spectrum_case (via mc_spectrum/mc_brem_spectrum) takes
    the NumPy path even when cupy is importable and a real GPU is present on
    the box. A no-op fork/spawn-local mutation: it never touches the driver
    process's globals. Harmless when _RESOURCE_POLICY.gpu is already False.
    """
    global _WORKER_INHERITED_CORE
    _WORKER_INHERITED_CORE = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if _WORKER_INHERITED_CORE in ("", "auto", "cuda"):
        set_canonical_env("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    if force_cpu:
        _RESOURCE_POLICY.gpu = False
        _spectrum_mod.xp = np
        _spectrum_mod.REAL = np.float64
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # typed signatures matter: the untyped pseudo-handle (-1) gets
        # truncated on 64-bit and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        try:
            if hasattr(os, "nice"):
                os.nice(10)  # type: ignore[reportAttributeAccessIssue]  # POSIX fallback
        except Exception:
            pass


@contextmanager
def _cpu_spectrum_backend():
    """Temporarily execute spectrum helpers with NumPy in the driver."""

    previous = (_RESOURCE_POLICY.gpu, _spectrum_mod.xp, _spectrum_mod.REAL)
    _RESOURCE_POLICY.gpu = False
    _spectrum_mod.xp = np
    _spectrum_mod.REAL = np.float64
    try:
        yield
    finally:
        _RESOURCE_POLICY.gpu, _spectrum_mod.xp, _spectrum_mod.REAL = previous


from . import adaptive as _adaptive
from .adaptive import case_transport_core as _case_transport_core
from .artifacts import STREAM_MAX_SEGMENTS as STREAM_MAX_SEGMENTS
from .artifacts import spectrum_from_artifact as spectrum_from_artifact
from .artifacts import stream_spectrum_from_artifact as stream_spectrum_from_artifact
from .pool import _admit_cpu_fallback as _admit_cpu_fallback
from .pool import _available_mem_mb as _available_mem_mb
from .pool import _case_progress_label as _case_progress_label
from .pool import _cpu_pool_workers as _cpu_pool_workers
from .pool import _gpu_pipeline_prefetch as _gpu_pipeline_prefetch
from .pool import _gpu_pipeline_workers as _gpu_pipeline_workers
from .pool import _mem_worker_cap as _mem_worker_cap
from .pool import _pipeline_slot_cap as _pipeline_slot_cap
from .scheduling import _cuda_transport_run as _cuda_transport_run
from .scheduling import case_runtime_plan as case_runtime_plan
from .scheduling import run_cases as run_cases
from .scheduling import runtime_plan as runtime_plan

for _exported in (
    case_runtime_plan,
    _cuda_transport_run,
    runtime_plan,
    run_cases,
    spectrum_from_artifact,
    stream_spectrum_from_artifact,
):
    _exported.__module__ = __name__

del _exported
