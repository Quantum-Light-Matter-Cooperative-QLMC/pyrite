"""One host transport pass behind :func:`.api.simulate_trajectories`.

Stages run in the historical order -- validate controls, build layer tables,
sample the entrance state, prepare lookup/stopping inputs, allocate row
buffers, dispatch one core, assemble the result -- so arrays, dtypes, row
order, RNG streams and error precedence match the former single function.
Accelerator modules stay lazily imported.

Validation: sbethe-corrected-stopping
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._numerics import DEFAULT_RADIATIVE_CUTOFF_EV
from ..geometry import validate_transverse_dimensions
from .arguments import TransportInputs, resolve_cutoffs_by_electron, resolve_layers_and_elastic
from .batching import (
    _flight_diagnostic_summary,
    _run_per_electron_transport,
    _run_per_electron_transport_lut,
    pack_layer_tables,
    resolve_transport_core,
)
from .beam_entry import (
    initial_beam_positions,
    initial_energies_keV,
    normalized_beam_dir,
    primary_directions,
    resolve_bunch_times,
    table_energy_range,
)
from .cores import (
    _transport_core_grooved,
    _transport_core_ungrooved_lut,
    _transport_core_ungrooved_lut_inelastic,
    _transport_core_ungrooved_perelectron_lut,
    _transport_core_ungrooved_perelectron_lut_inelastic,
    exact_ungrooved_core,
)
from .events import TransportStepLimitError
from .hard_inelastic import hard_keys_from_stream_keys, validate_inelastic_args
from .hard_radiative import validate_radiative_args
from .kinematics import _sample_bunch_offsets, stream_keys
from .layer_tables import build_layer_tables
from .lut import build_transport_energy_lut
from .scattering import check_elsepa_coverage, check_mott_coverage
from .stopping import (
    pack_sbethe_stopping_tables,
    prepare_sbethe_stopping_tables,
)


def _nsys_push(name):
    # NVTX ranges split the host transport phase in a GPU capture; no-op off
    # the profiled path. Lazy import: runner imports this module.
    from ..runner import _nsys_push as push

    push(name)


def _nsys_pop():
    from ..runner import _nsys_pop as pop

    pop()


@dataclass(frozen=True)
class _Controls:
    """Validated model switches and normalized geometry/cutoff inputs."""

    width_ang: Any
    height_ang: Any
    layers: Any
    elastic_tables: Any
    shell_mode: bool
    radiative_mode: bool
    radiative_cutoff_eV: Any
    transport_core: str
    max_dE_frac: float
    E_cut: np.ndarray


@dataclass(frozen=True)
class _Stack:
    """Host per-layer tables in :func:`.layer_tables.build_layer_tables` order."""

    z_total: float
    n_layers: int
    stopping_tables: Any
    L_Zs: Any
    L_Js: Any
    L_ncm3: Any
    L_ks: Any
    L_coeffs: Any
    L_E_cross: Any
    mott_tables: Any
    L_sr_rate_numer: Any
    L_mott_numer: Any
    L_mott_denom1: Any
    L_mott_denom2: Any
    L_sr_joy_numer: Any
    mott_group: Any
    L_top: np.ndarray
    L_bot: np.ndarray
    internal_bounds: np.ndarray

    @property
    def layer_arrays(self):
        return (
            self.L_Js,
            self.L_Zs,
            self.L_ks,
            self.L_coeffs,
            self.L_E_cross,
            self.L_ncm3,
            self.L_sr_rate_numer,
            self.L_mott_numer,
            self.L_mott_denom1,
            self.L_mott_denom2,
            self.L_sr_joy_numer,
        )


@dataclass(frozen=True)
class _Entry:
    """Sampled entrance state; ``pos``/``dirs``/``E_keV``/``alive``/``clock`` mutate."""

    pos: np.ndarray
    dirs: np.ndarray
    E_keV: np.ndarray
    alive: np.ndarray
    clock: np.ndarray
    n_missed: int
    t0_electron: np.ndarray
    beam_entry: Any
    initial_r_ang: np.ndarray
    initial_v_hat: np.ndarray
    initial_E_keV: np.ndarray


@dataclass(frozen=True)
class _Pass:
    """Everything one core dispatch reads, prepared on the host."""

    inputs: TransportInputs
    controls: _Controls
    stack: _Stack
    entry: _Entry
    rng: np.random.Generator
    transport_keys: np.ndarray
    shell_tables: Any
    stopping_tables: Any
    radiative_args: Any
    transport_lut: Any
    sbethe_group: tuple
    buffers: dict
    sec_on: bool
    inelastic_args: Any
    straggle_on: bool
    stragg_dE: np.ndarray
    stragg_stream_keys: np.ndarray
    stragg_layer_tables: tuple


@dataclass(frozen=True)
class _CoreOutcome:
    nseg: int
    n_back: int
    n_trans: int
    n_side: int
    n_cutoff: int
    n_step_limited: int
    dev_segs: Any
    stragg_dE: Any
    seg_xp: Any
    vacuum: tuple


_NO_VACUUM = (
    np.empty((0, 3), dtype=float),
    np.empty((0, 3), dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=np.int64),
)
# Segment buffer names in the lockstep ``segments`` tuple order.
_ROW_FIELDS = (
    "seg_dir",
    "seg_mid",
    "seg_len",
    "seg_E",
    "seg_t0",
    "seg_id",
    "seg_lay",
    "seg_E_end",
    "seg_t_end",
    "seg_flight",
    "seg_substep",
    "seg_event",
)


def run_transport_pass(p: TransportInputs):
    """Transport one population (primaries, a block, or one cascade generation)."""
    electron_start = int(p._electron_start)
    # ``_secondary`` is the cascade's per-generation pass (secondaries.py).
    launch = None if p._secondary is None else p._secondary.launch
    block_keys = stream_keys(p.seed, p.Ne, start=electron_start)
    # Per-electron/CUDA keys: a block or cascade generation supplies its own.
    transport_keys = launch.stream_keys if launch is not None else block_keys
    c = _validated_controls(p, launch, electron_start)

    _nsys_push("pyrite.transport.tables")
    stack = _layer_stack(p, c)
    _nsys_pop()

    _nsys_push("pyrite.transport.sample")
    rng = np.random.default_rng(p.seed)
    entry, table_lo, table_hi, shell_tables, stopping, radiative_args = _sample_entry(
        p, c, stack, launch, electron_start
    )
    _nsys_pop()

    transport_lut = _transport_lut(p, c, stack, table_lo, table_hi, stopping)
    sbethe_group = pack_sbethe_stopping_tables(stopping, stack.n_layers)

    _nsys_push("pyrite.transport.alloc")
    # Secondary launch directions (#94): rows only while a cascade runs.
    sec_on = c.shell_mode and p._secondary is not None
    buffers = _row_buffers(p, c, sec_on)
    inelastic_args = None
    if shell_tables is not None:
        inelastic_args = shell_tables.core_args(hard_keys_from_stream_keys(transport_keys))
    _nsys_pop()

    # Straggling (slice D): False skips the Urban sampler on every core,
    # BIT-FOR-BIT. Lockstep/grooved cores key it by ``stragg_stream_keys``;
    # per-electron cores reuse their own stream keys. ``stragg_layer_tables``
    # gives the per-electron LUT cores (CPU and CUDA) the per-element split
    # the LUT lacks.
    straggle_on = bool(p.straggling)
    if straggle_on:
        stragg_layer_tables = pack_layer_tables(*stack.layer_arrays)[:5] + sbethe_group
    else:
        stragg_layer_tables = (np.zeros((1, 1), dtype=np.float64),) * 5 + sbethe_group
    prepared = _Pass(
        p,
        c,
        stack,
        entry,
        rng,
        transport_keys,
        shell_tables,
        stopping,
        radiative_args,
        transport_lut,
        sbethe_group,
        buffers,
        sec_on,
        inelastic_args,
        straggle_on,
        np.zeros(p.Ne) if straggle_on else np.zeros(0),
        block_keys if straggle_on else np.zeros(1, dtype=np.uint64),
        stragg_layer_tables,
    )

    _nsys_push("pyrite.transport.core")
    outcome = _dispatch_core(prepared)
    _nsys_pop()

    if outcome.n_step_limited:
        raise TransportStepLimitError(outcome.n_step_limited, p.Ne, p.max_steps)
    return _assemble_result(prepared, outcome)


def _validated_controls(p, launch, electron_start):
    """Validate scalar controls and model switches in the historical order."""
    if not np.isfinite(p.E0_keV) or p.E0_keV <= 0.0:
        raise ValueError("E0_keV must be finite and strictly positive")

    width_mm, height_mm = validate_transverse_dimensions(
        p.crystal_width_mm, p.crystal_height_mm, unit="mm"
    )
    layers, elastic_tables = resolve_layers_and_elastic(
        p.layers,
        thickness_ang=p.thickness_ang,
        element=p.element,
        n_atoms_per_ang3=p.n_atoms_per_ang3,
        composition=p.composition,
        elastic_model=p.elastic_model,
        elastic_tables=p.elastic_tables,
        energy_model=p.energy_model,
    )
    shell_mode = validate_inelastic_args(
        p.inelastic_model,
        p.inelastic_cutoff_eV,
        p.inelastic_materials,
        energy_model=p.energy_model,
        groove=p.groove,
        stopping_tables=p.stopping_tables,
    )
    radiative_mode = validate_radiative_args(
        p.radiative_model,
        p.radiative_cutoff_eV,
        p.bremslib_tables,
        energy_model=p.energy_model,
        groove=p.groove,
        keep_segments_on_device=p.keep_segments_on_device,
    )
    radiative_cutoff_eV = p.radiative_cutoff_eV
    if radiative_mode and radiative_cutoff_eV is None:
        radiative_cutoff_eV = DEFAULT_RADIATIVE_CUTOFF_EV

    transport_core = _checked_core(p, launch, electron_start)
    max_dE_frac = float(p.max_dE_frac)
    if max_dE_frac < 0.0:
        raise ValueError("max_dE_frac must be non-negative")
    # Substepping a frozen flight is exactly the mis-phased configuration the
    # slice-E convergence study rejected: it multiplies rows without improving
    # the clock, so the two options are not independently selectable.
    if max_dE_frac > 0.0 and p.energy_model != "midpoint":
        raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")

    return _Controls(
        None if width_mm is None else width_mm * 1.0e7,
        None if height_mm is None else height_mm * 1.0e7,
        layers,
        elastic_tables,
        shell_mode,
        radiative_mode,
        radiative_cutoff_eV,
        transport_core,
        max_dE_frac,
        resolve_cutoffs_by_electron(p.E_cut_by_electrons, p.E_cut_keV, p.Ne),
    )


def _checked_core(p, launch, electron_start):
    """Resolved transport core, checked against groove, blocks and device rows."""
    transport_core = p.transport_core
    if launch is None:  # a launched generation names its per-electron core
        transport_core = resolve_transport_core(transport_core, p.Ne, p.groove)
    if transport_core != "lockstep" and p.groove is not None:
        raise ValueError("grooved transport is only implemented for the lockstep core")
    if electron_start and transport_core == "lockstep":
        raise ValueError("electron blocks need counter-addressed streams, not the lockstep core")
    if p.keep_segments_on_device and transport_core != "cuda":
        raise ValueError(
            "keep_segments_on_device requires transport_core='cuda'; "
            f"{p.transport_core!r} resolved to {transport_core!r}"
        )
    return transport_core


def _layer_stack(p, c):
    layers = c.layers
    n_layers = len(layers)
    stopping_tables = prepare_sbethe_stopping_tables(p.stopping_tables, n_layers)
    tables = build_layer_tables(
        layers, p.elastic_model, c.elastic_tables, p.atomic_electron_deflection,
        p.inelastic_cutoff_eV,
    )  # fmt: skip
    L_top = np.asarray([float(a) for (a, _, _) in layers], dtype=float)
    L_bot = np.asarray([float(b) for (_, b, _) in layers], dtype=float)
    return _Stack(
        float(layers[-1][1]), n_layers, stopping_tables, *tables, L_top, L_bot, L_bot[:-1].copy()
    )


def _sample_entry(p, c, stack, launch, electron_start):
    """Entrance state, table range, and the energy-dependent tables it selects."""
    from ..bunch_front import validate_front_source

    bunch_front_tilt = validate_front_source(
        p.bunch_front_tilt,
        transverse=p.transverse_distribution,
        energy_spread=p.energy_spread_frac,
        source="gpt_gdf" if p.gdf_source is not None else "analytic",
        longitudinal=p.longitudinal_distribution,
        offsets=p.long_offsets_fs,
        shape=p.long_shape,
        groove=p.groove,
    )
    pos, transverse_slopes, beam_entry = initial_beam_positions(
        p.seed,
        p.Ne,
        transverse_distribution=p.transverse_distribution,
        beam_fwhm_mm=p.beam_fwhm_mm,
        beam_fwhm_y_mm=p.beam_fwhm_y_mm,
        tilt_polar_rad=p.tilt_polar_rad,
        tilt_azim_rad=p.tilt_azim_rad,
        groove=p.groove,
        start=electron_start,
        E0_keV=p.E0_keV,
        energy_spread_frac=p.energy_spread_frac,
        bunch_front_tilt=bunch_front_tilt,
    )
    beam_dir = normalized_beam_dir(p.beam_dir)
    dirs = primary_directions(beam_dir, transverse_slopes, p.Ne)
    E_keV = initial_energies_keV(p.E0_keV, p.Ne, p.seed, p.energy_spread_frac, start=electron_start)
    gdf_t0 = None
    if p.gdf_source is not None:
        pos, dirs, E_keV, gdf_t0 = _gdf_entry(p)
    elif launch is not None:
        pos, dirs, E_keV = launch.r_ang.copy(), launch.v_hat.copy(), launch.E_keV.copy()
    E_cut = c.E_cut
    if not np.all(E_cut < E_keV):
        raise ValueError("each electron cutoff energy must be below its initial energy")
    table_lo, table_hi = table_energy_range(E_cut, E_keV, p._energy_range_keV)
    _check_table_coverage(p, c, stack, E_keV)
    shell_tables, stopping = _shell_tables(p, c, stack, table_lo, table_hi)
    radiative_args = None
    if c.radiative_mode:
        from .hard_radiative import radiative_core_args

        radiative_args = radiative_core_args(
            [layer[2] for layer in c.layers],
            p.bremslib_tables,
            c.radiative_cutoff_eV,
            np.array([table_lo]),
            np.array([table_hi]),
            p.seed,
            p.Ne,
            electron_start,
        )
        if launch is not None:
            radiative_args = (launch.radiative_keys,) + radiative_args[1:]
    if c.width_ang is not None:
        assert c.height_ang is not None
        alive = (np.abs(pos[:, 0]) <= c.width_ang / 2.0) & (np.abs(pos[:, 1]) <= c.height_ang / 2.0)
        n_missed = int((~alive).sum())
    else:
        alive = np.ones(p.Ne, dtype=bool)
        n_missed = 0
    # per-electron clock: cumulative flight "time" sum(L/beta) [Ang, c=1], the
    # same unit as the radiation interaction time t_L. Recorded at each segment's
    # START so a segment carries (depth, energy, age) -- consumed by the
    # penetration / electron-lifetime plots (-> fs via c = 2997.92 Ang/fs).
    clock = np.zeros(p.Ne)
    # per-electron bunch arrival [Ang, c=1], SEPARATE from the relative-age clock
    # (t_abs = t_ang + t0_ang); None/None -> all-zero point bunch, bit-for-bit.
    t0_electron = _sample_bunch_offsets(
        p.Ne,
        p.bunch_length_fs,
        p.long_shape,
        p.long_offsets_fs,
        p.seed,
        p.longitudinal_distribution,
    )
    t0_electron, clock, beam_entry = resolve_bunch_times(
        t0_electron, clock, beam_entry, pos, E_keV, gdf_t0=gdf_t0, launch=launch
    )
    # Snapshot before transport mutates ``pos``, ``dirs``, and ``E``. These
    # arrays describe incident phase space, including particles that miss a
    # finite footprint.
    entry = _Entry(
        pos, dirs, E_keV, alive, clock, n_missed, t0_electron, beam_entry,
        pos.copy(), dirs.copy(), E_keV.copy(),
    )  # fmt: skip
    return entry, table_lo, table_hi, shell_tables, stopping, radiative_args


def _gdf_entry(p):
    """GPT/GDF records replacing the analytic entrance state."""
    from ..gdf import load_gdf_beam

    if (
        p.groove is not None
        or p.beam_fwhm_mm
        or p.beam_fwhm_y_mm
        or p.transverse_distribution is not None
        or p.energy_spread_frac
        or p.bunch_length_fs is not None
        or p.long_offsets_fs is not None
        or p.longitudinal_distribution is not None
    ):
        raise ValueError("gpt_gdf requires a flat entrance and no analytic phase-space settings")
    source = p.gdf_source
    gdf = load_gdf_beam(
        source["path"],
        source["time_s"],
        source["tolerance_s"],
        source["normalization"],
        screen_position_m=source.get("screen_position_m"),
        screen_tolerance_m=source.get("screen_tolerance_m", 1e-9),
    )
    if gdf.sha256 != source["sha256"]:
        raise ValueError("GDF file changed after case construction; rebuild the run")
    return gdf.sample(
        p.Ne,
        p.seed,
        source["z_origin_m"],
        p.tilt_polar_rad,
        p.tilt_azim_rad,
        energy_keV=float(p.E0_keV) if source.get("shape_only", False) else None,
    )


def _check_table_coverage(p, c, stack, E_keV):
    """Reject transport energies outside the SBETHE/ELSEPA/Mott tables."""
    E_min = float(np.min(c.E_cut))
    E_max = float(np.max(E_keV))
    for prepared in stack.stopping_tables or ():
        lower = float(np.nextafter(np.exp(prepared[0][0]), -np.inf))
        upper = float(np.nextafter(np.exp(prepared[0][-1]), np.inf))
        if E_min < lower or E_max > upper:
            raise ValueError(
                f"transport energy range must be within SBETHE table [{lower:g}, {upper:g}] keV"
            )
    check_elsepa_coverage(c.elastic_tables, E_min, E_max)
    if p.elastic_model == "mott":
        check_mott_coverage(c.layers, E_min, E_max)


def _shell_tables(p, c, stack, table_lo, table_hi):
    """Shell soft/hard tables and the continuous stopping the cores apply."""
    if not c.shell_mode:
        return None, stack.stopping_tables
    # Lazy: the host shell model pulls in catalog and EEDL data.
    from .shell_transport import build_shell_inelastic_tables

    # Checked by validate_inelastic_args, which also requires stopping_tables.
    assert p.inelastic_materials is not None and p.inelastic_cutoff_eV is not None
    assert stack.stopping_tables is not None
    _nsys_push("pyrite.transport.inelastic")
    E_range_keV = (table_lo, table_hi)
    if p._secondary is not None:  # one table set for every generation
        lo, hi = p._secondary.table_range_keV
        E_range_keV = (min(lo, E_range_keV[0]), max(hi, E_range_keV[1]))
    shell_tables = build_shell_inelastic_tables(
        p.inelastic_materials, float(p.inelastic_cutoff_eV), stack.stopping_tables, *E_range_keV,
        projectile=getattr(p._secondary, "projectile", "electron"),
    )  # fmt: skip
    _nsys_pop()
    # The cores' continuous stopping is the soft share from here on.
    return shell_tables, list(shell_tables.soft_stopping_tables)


def _transport_lut(p, c, stack, table_lo, table_hi, stopping):
    # The LUT stores a screened-Rutherford screening parameter per energy,
    # which has no ELSEPA counterpart, so ELSEPA runs the exact cores; it
    # does not run the coupled radiative mode either.
    if (
        p.groove is not None
        or not p.transport_lut_config.enabled
        or c.radiative_mode
        or p.elastic_model == "elsepa"
    ):
        return None
    _nsys_push("pyrite.transport.lut")
    transport_lut = build_transport_energy_lut(
        table_lo,
        table_hi,
        _ELASTIC_CODES[p.elastic_model],
        stack.L_Js,
        stack.L_Zs,
        stack.L_ks,
        stack.L_coeffs,
        stack.L_E_cross,
        stack.L_sr_rate_numer,
        stack.L_mott_numer,
        stack.L_mott_denom1,
        stack.L_mott_denom2,
        stack.L_sr_joy_numer,
        stack.mott_tables,
        config=p.transport_lut_config,
        stopping_tables=stopping,
    )
    _nsys_pop()
    return transport_lut


_ELASTIC_CODES = {"sr": 0, "mott": 1, "elsepa": 2}


def _row_buffers(p, c, sec_on):
    """Preallocate fixed-capacity output buffers.

    ``max_steps`` is already a conservative safety bound in ordinary runs;
    groove re-entry events can add material segments without consuming it, so
    the compiled core raises a clear buffer error if a pathological case
    exceeds this capacity. A device-resident run fills none of them -- its
    batches are joined on the device instead -- so it reserves no rows.
    """
    n_rows = 0 if p.keep_segments_on_device else p.Ne * p.max_steps
    # Flight end state exists only under the controlled propagator; the frozen
    # rule keeps the historical seven-field row exactly.
    n_end_rows = n_rows if p.energy_model == "midpoint" else 0
    n_hard_rows = n_end_rows if c.shell_mode else 0
    n_rad_rows = n_end_rows if c.radiative_mode else 0
    return {
        "seg_mid": np.empty((n_rows, 3), dtype=float),
        "seg_dir": np.empty((n_rows, 3), dtype=float),
        "seg_len": np.empty(n_rows, dtype=float),
        "seg_E": np.empty(n_rows, dtype=float),
        "seg_t0": np.empty(n_rows, dtype=float),
        "seg_id": np.empty(n_rows, dtype=np.int64),
        "seg_lay": np.empty(n_rows, dtype=np.int16),
        "seg_E_end": np.empty(n_end_rows, dtype=float),
        "seg_t_end": np.empty(n_end_rows, dtype=float),
        "seg_flight": np.empty(n_end_rows, dtype=np.int64),
        "seg_substep": np.empty(n_end_rows, dtype=np.int64),
        "seg_event": np.empty(n_end_rows, dtype=np.int8),
        "seg_hard_W": np.empty(n_hard_rows, dtype=float),
        "seg_hard_ch": np.empty(n_hard_rows, dtype=np.int16),
        "seg_hard_dir": np.empty((n_hard_rows if sec_on else 0, 3), dtype=float),
        "seg_rad_k": np.empty(n_rad_rows, dtype=float),
        "seg_rad_Z": np.empty(n_rad_rows, dtype=np.int16),
    }


def _dispatch_core(t):
    groove = t.inputs.groove
    lockstep = t.controls.transport_core == "lockstep"
    if groove is None and t.transport_lut is not None and not lockstep:
        return _run_per_electron(t, lut=True)
    if groove is None and t.transport_lut is not None:
        return _run_lockstep_lut(t)
    if groove is None and not lockstep:
        # Per-electron streams and run-to-completion ordering. Not bit-for-bit
        # with the lockstep core -- see `_transport_core_ungrooved_perelectron`.
        return _run_per_electron(t, lut=False)
    if groove is None:
        return _run_lockstep_exact(t)
    return _run_grooved(t)


def _per_electron_core(t, lut):
    """Per-electron core and its array module; CUDA imports stay lazy."""
    shell_mode = t.controls.shell_mode
    cuda = t.controls.transport_core == "cuda"
    if lut and cuda:
        if shell_mode:
            raise NotImplementedError(
                "inelastic_model='shell-soft-hard' is not implemented on the CUDA "
                "LUT core; disable the LUT to reach the exact CUDA core"
            )
        from ._jit_launch import make_cuda_transport_lut_core

        return make_cuda_transport_lut_core()
    if lut:
        if shell_mode:
            return _transport_core_ungrooved_perelectron_lut_inelastic, np
        return _transport_core_ungrooved_perelectron_lut, np
    if cuda:
        from ._jit_launch import make_cuda_transport_core

        return make_cuda_transport_core()
    core = exact_ungrooved_core(
        per_electron=True, inelastic=shell_mode, radiative=t.controls.radiative_mode
    )
    return core, np


def _run_per_electron(t, *, lut):
    """Batched per-electron CPU/CUDA transport, LUT or exact."""
    p, c, s, e, b = t.inputs, t.controls, t.stack, t.entry, t.buffers
    core, core_xp = _per_electron_core(t, lut)
    common = (
        core,
        core_xp,
        p.Ne,
        p.seed,
        p.max_steps,
        p.Ne * p.max_steps,
        s.n_layers,
        s.internal_bounds,
        _ELASTIC_CODES[p.elastic_model],
        _energy_model_code(p),
        c.max_dE_frac,
        s.z_total,
        c.width_ang is not None,
        0.0 if c.width_ang is None else float(c.width_ang),
        0.0 if c.height_ang is None else float(c.height_ang),
        e.alive,
        e.clock,
        e.pos,
        e.dirs,
        c.E_cut,
    )
    rows = tuple(b[name] for name in _ROW_FIELDS)
    options = {
        "config": p.per_electron_config,
        "keep_on_device": p.keep_segments_on_device,
        "inelastic": (
            (t.inelastic_args, b["seg_hard_W"], b["seg_hard_ch"], b["seg_hard_dir"], t.sec_on)
            if c.shell_mode
            else None
        ),
        "keys": t.transport_keys,
        "on_batch": p.transport_progress,
    }
    if lut:
        lut_tables = (t.transport_lut.n_el, s.L_top, s.L_bot, t.transport_lut, e.E_keV)
        straggling = (t.stragg_layer_tables, t.straggle_on, t.stragg_dE)
        counts = _run_per_electron_transport_lut(
            *common, *lut_tables, *rows, *straggling, **options
        )
    else:
        packed_per_layer_tables = pack_layer_tables(*s.layer_arrays)
        packed_per_layer_tables += t.sbethe_group
        tables = (packed_per_layer_tables, s.L_top, s.L_bot, s.mott_group, e.E_keV)
        radiative = (t.radiative_args, b["seg_rad_k"], b["seg_rad_Z"])
        counts = _run_per_electron_transport(
            *common, *tables, *rows, t.straggle_on, t.stragg_dE, **options,
            radiative=radiative if c.radiative_mode else None,
        )  # fmt: skip
    seg_xp = core_xp if p.keep_segments_on_device else np
    return _CoreOutcome(*counts, seg_xp, _NO_VACUUM)


def _energy_model_code(p):
    return 1 if p.energy_model == "midpoint" else 0


def _lockstep_args(t):
    """Grouped kernel argument tuples (issue #66) shared by the lockstep cores."""
    p, c, e = t.inputs, t.controls, t.entry
    control = (
        p.max_steps,
        p.Ne * p.max_steps,
        _ELASTIC_CODES[p.elastic_model],
        _energy_model_code(p),
        c.max_dE_frac,
    )
    state = (e.alive, e.clock, e.pos, e.dirs, e.E_keV, c.E_cut)
    segments = tuple(t.buffers[name] for name in _ROW_FIELDS)
    if c.shell_mode:
        segments += (t.buffers["seg_hard_W"], t.buffers["seg_hard_ch"], t.buffers["seg_hard_dir"])
    if c.radiative_mode:
        segments += (t.buffers["seg_rad_k"], t.buffers["seg_rad_Z"])
    straggling = (t.straggle_on, t.stragg_stream_keys, t.stragg_dE)
    return control, state, segments, straggling


def _geometry(t, n_el):
    """Lockstep geometry tuple; ``n_el`` is the LUT or ragged element count."""
    c, s = t.controls, t.stack
    return (
        s.n_layers,
        s.internal_bounds,
        s.z_total,
        c.width_ang is not None,
        0.0 if c.width_ang is None else float(c.width_ang),
        0.0 if c.height_ang is None else float(c.height_ang),
        n_el,
        s.L_top,
        s.L_bot,
    )


def _ragged_n_el(s):
    # The exact lockstep cores index ragged rows and never read L_nel; it rides
    # in the geometry tuple only so every core shares one layout.
    return np.fromiter((zs.size for zs in s.L_Zs), dtype=np.int32, count=s.n_layers)


def _lockstep_outcome(t, counts, vacuum=_NO_VACUUM):
    return _CoreOutcome(*counts, None, t.stragg_dE, np, vacuum)


def _run_lockstep_lut(t):
    s, lut = t.stack, t.transport_lut
    control, state, segments, straggling = _lockstep_args(t)
    shell_mode = t.controls.shell_mode
    core = _transport_core_ungrooved_lut_inelastic if shell_mode else _transport_core_ungrooved_lut
    counts = core(
        t.inputs.Ne,
        t.rng,
        control,
        _geometry(t, lut.n_el),
        (
            lut.log_E_min,
            lut.inv_dlogE,
            lut.n_energy,
            lut.total_rate,
            lut.dEds,
            lut.inv_beta,
            lut.cdf,
            lut.alpha,
        ),
        (s.L_Js, s.L_Zs, s.L_ks, s.L_coeffs, s.L_E_cross) + t.sbethe_group,
        state,
        segments,
        straggling,
        *((t.inelastic_args,) if shell_mode else ()),
    )
    return _lockstep_outcome(t, counts)


def _run_lockstep_exact(t):
    c = t.controls
    control, state, segments, straggling = _lockstep_args(t)
    exact_core = exact_ungrooved_core(
        per_electron=False, inelastic=c.shell_mode, radiative=c.radiative_mode
    )
    extra_args = (t.inelastic_args,) if c.shell_mode else ()
    if c.radiative_mode:
        extra_args = (t.inelastic_args if c.shell_mode else (), t.radiative_args)
    counts = exact_core(
        t.inputs.Ne,
        t.rng,
        control,
        _geometry(t, _ragged_n_el(t.stack)),
        t.stack.layer_arrays + t.sbethe_group,
        t.stack.mott_group,
        state,
        segments,
        straggling,
        *extra_args,
    )
    return _lockstep_outcome(t, counts)


def _run_grooved(t):
    p, groove = t.inputs, t.inputs.groove
    max_vac = p.Ne * p.max_steps
    vac_start_buf = np.empty((max_vac, 3), dtype=float)
    vac_end_buf = np.empty((max_vac, 3), dtype=float)
    vac_E_buf = np.empty(max_vac, dtype=float)
    vac_t0_buf = np.empty(max_vac, dtype=float)
    vac_id_buf = np.empty(max_vac, dtype=np.int64)

    tp = float(groove.tilt_polar_rad)
    control, state, segments, straggling = _lockstep_args(t)
    nseg, nvac, *counts = _transport_core_grooved(
        p.Ne,
        t.rng,
        control,
        _geometry(t, _ragged_n_el(t.stack)),
        (
            float(groove.spacing_ang),
            float(groove.depth_ang),
            float(np.sin(tp)),
            float(np.cos(tp)),
            max_vac,
            vac_start_buf,
            vac_end_buf,
            vac_E_buf,
            vac_t0_buf,
            vac_id_buf,
        ),
        t.stack.layer_arrays + t.sbethe_group,
        t.stack.mott_group,
        state,
        segments,
        straggling,
    )
    vacuum = tuple(
        buf[:nvac] for buf in (vac_start_buf, vac_end_buf, vac_E_buf, vac_t0_buf, vac_id_buf)
    )
    return _lockstep_outcome(t, (nseg, *counts), vacuum)


def _segment_rows(t, o):
    """Per-segment arrays sized to ``nseg``, host or device-joined."""
    b, nseg = t.buffers, o.nseg
    if o.dev_segs is None:
        return {name: b[name][:nseg] for name in b}
    # Already sized to `nseg` by the join, in the scratch's field order.
    rows = {name: b[name][:nseg] for name in b}
    rows.update(zip(_ROW_FIELDS[:7], o.dev_segs[:7], strict=True))
    if t.inputs.energy_model == "midpoint":
        rows.update(zip(_ROW_FIELDS[7:12], o.dev_segs[7:12], strict=True))
    if t.controls.shell_mode:
        hard = ("seg_hard_W", "seg_hard_ch", "seg_hard_dir")
        rows.update(zip(hard, o.dev_segs[12:15], strict=True))
    return rows


def _assemble_result(t, o):
    p, c, s, e = t.inputs, t.controls, t.stack, t.entry
    _nsys_push("pyrite.transport.output")
    rows = _segment_rows(t, o)
    E_seg, elec_id = rows["seg_E"], rows["seg_id"]
    vacuum_start_ang, vacuum_end_ang, vacuum_E_keV, vacuum_t_ang, vacuum_elec_id = o.vacuum
    # Per-segment, so it follows the segments: gathering on the device costs one
    # Ne-sized upload of `t0_electron` and saves an nseg-sized download.
    seg_xp, t0_electron = o.seg_xp, e.t0_electron
    t0_by_electron = t0_electron if seg_xp is np else seg_xp.asarray(t0_electron)
    t0_ang = t0_by_electron[elec_id] if elec_id.size else seg_xp.empty(0, dtype=float)
    vacuum_t0_ang = t0_electron[vacuum_elec_id] if vacuum_elec_id.size else np.empty(0, dtype=float)
    _nsys_pop()

    result = {
        # Initial sampled phase space is diagnostic-only.  Keep per-electron
        # arrays (including missed entries), separate from per-segment arrays,
        # so beam metrics describe the incident bunch rather than its transport.
        "initial_r_ang": e.initial_r_ang,
        "initial_v_hat": e.initial_v_hat,
        "initial_E_keV": e.initial_E_keV,
        "initial_t0_ang": t0_electron.copy(),
        "r_mid": rows["seg_mid"],
        "v_hat": rows["seg_dir"],
        "L_ang": rows["seg_len"],
        # `E_keV`/`t_ang` are unscheduled compatibility aliases of the canonical
        # flight-start fields and never become midpoint/representative values.
        "E_keV": E_seg,
        "E_start_keV": E_seg,
        "t_ang": rows["seg_t0"],  # segment-start age sum(L/beta) [Ang, c=1]
        "t_start_ang": rows["seg_t0"],
        "t0_ang": t0_ang,  # per-electron longitudinal bunch offset [Ang, c=1]
        # `elec_id` is an unscheduled compatibility alias of `electron_id`.
        "elec_id": elec_id,  # emitting electron index in [0, Ne)
        "electron_id": elec_id,
        "layer": rows["seg_lay"],  # emitting layer index in [0, n_layers)
        "vacuum_start_ang": vacuum_start_ang,
        "vacuum_end_ang": vacuum_end_ang,
        "vacuum_E_keV": vacuum_E_keV,
        "vacuum_t_ang": vacuum_t_ang,
        "vacuum_t0_ang": vacuum_t0_ang,
        "vacuum_elec_id": vacuum_elec_id,
        "n_backscattered": o.n_back,
        "n_transmitted": o.n_trans,
        "n_side_exited": o.n_side,
        "n_missed": e.n_missed,
        "n_cutoff_stopped": int(o.n_cutoff),
        "n_step_limited": 0,
        "n_stopped": int(o.n_cutoff),
        "Ne": p.Ne,
        "thickness_ang": s.z_total,
        "crystal_width_ang": c.width_ang,
        "crystal_height_ang": c.height_ang,
        "n_layers": s.n_layers,
        "beam_entry": e.beam_entry,  # plain-data analytic spot, or None
    }
    _add_optional_fields(t, o, rows, result)
    return result


def _add_optional_fields(t, o, rows, result):
    """Mode-dependent result fields; absent modes add nothing (bit-for-bit)."""
    p, c, s = t.inputs, t.controls, t.stack
    if t.stopping_tables is not None:
        # Prepared (log-transformed) per-layer SBETHE tables, so post-transport
        # consumers that re-solve the cutoff (the spectrum cutoff clip) use the
        # same stopping model the transport ran with instead of the reference
        # splice. Metadata, not a per-row array: absent from _SEG_ARRAYS.
        result["stopping_tables"] = tuple(t.stopping_tables)
    if t.straggle_on:
        # The summed per-electron Urban-SAMPLED loss. Every host core and the
        # exact CUDA core apply it; the CUDA LUT combination fails closed.
        result["straggle_dE_keV"] = o.stragg_dE
    if p.energy_model == "midpoint":
        E_end = rows["seg_E_end"]
        result["E_end_keV"] = E_end
        result["t_end_ang"] = rows["seg_t_end"]
        # Representative energy used by the clock and radiation quadrature.
        # The explicit RK2 stopping update itself uses (E_start + E_pred)/2.
        result["E_repr_keV"] = 0.5 * (rows["seg_E"] + E_end)
        # `(electron_id, flight_id)` is the stable physical key; `substep_id`
        # indexes numerical rows inside one flight and is integration detail.
        result["flight_id"] = rows["seg_flight"]
        result["substep_id"] = rows["seg_substep"]
        # What ended each row; `transport.events` holds the codes and the
        # contract (`check_segment_event_contract`).
        result["event_kind"] = rows["seg_event"]
    if t.shell_tables is not None:
        result.update(t.shell_tables.result_fields(rows["seg_hard_W"], rows["seg_hard_ch"]))
        if t.sec_on:
            result["hard_secondary_v_hat"] = rows["seg_hard_dir"]
    if c.radiative_mode:
        from .hard_radiative import add_radiative_result_fields

        add_radiative_result_fields(
            result,
            rows["seg_rad_k"],
            rows["seg_rad_Z"],
            p.bremslib_tables,
            c.radiative_cutoff_eV,
            p.seed if p._secondary is None else p._secondary.photon_seed,
            t.radiative_args[-1],
        )
    if p.collect_diagnostics:
        result["transport_diagnostics"] = _flight_diagnostic_summary(
            rows["seg_E"],
            rows["seg_len"],
            rows["seg_id"],
            rows["seg_lay"],
            c.E_cut,
            s.L_Js,
            s.L_Zs,
            s.L_ks,
            s.L_coeffs,
            s.L_E_cross,
            s.L_ncm3,
            p.elastic_model,
            s.mott_group[5:],
        )
