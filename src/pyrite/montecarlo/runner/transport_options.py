"""Transport option forwarding and opt-in capability checks."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from ..case import refuse_axial_channeling, refuse_channeling, refuse_transition_radiation

# Public KernelTuning.transport override (#776): a ``PerElectronTransportConfig``
# set by in-process callers around ``run_case`` and read by ``_transport_case``.
# Like the progress sink, it keeps case payloads and call shapes unchanged, so
# these controls stay out of case identity; it never reaches pool workers.
_TRANSPORT_TUNING: ContextVar[Any] = ContextVar("pyrite_transport_tuning", default=None)


def tuned_transport_kwargs():
    """``simulate_trajectories`` kwargs of the active override, if any."""
    config = _TRANSPORT_TUNING.get()
    return {} if config is None else {"per_electron_config": config}


def transport_kernel_tuning(numerics):
    """``numerics``' ``TransportKernelTuning``, or None for a duck-typed stand-in."""
    execution = getattr(numerics, "resolved_execution", None)
    return None if execution is None else execution.transport


def per_electron_config(tuning):
    """The process default ``PerElectronTransportConfig`` with ``tuning`` applied."""
    from ..transport import DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
    from ..transport.continuation import TransportContinuationConfig

    config = DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
    if tuning.electron_batch is not None:
        config = replace(config, max_batch=tuning.electron_batch)
    if tuning.schedule is not None:
        rows = tuning.continuation_rows
        config = replace(
            config,
            schedule=tuning.schedule,
            continuation=None if rows is None else TransportContinuationConfig(rows_per_round=rows),
        )
    return config


@contextmanager
def transport_tuning(numerics):
    """Apply ``KernelTuning.transport`` to in-process runs.

    ``numerics`` is a ``Numerics``; internal stand-ins carrying only
    ``transport_core``, and unset fields, keep the process defaults
    (``PYRITE_MC_TRANSPORT_SCHEDULE``, ``PYRITE_MC_ELECTRON_BATCH_MAX``).
    """
    tuning = transport_kernel_tuning(numerics)
    if tuning is None or tuning == type(tuning)():
        yield
        return
    token = _TRANSPORT_TUNING.set(per_electron_config(tuning))
    try:
        yield
    finally:
        _TRANSPORT_TUNING.reset(token)


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
        bunch_front_tilt=case.get("bunch_front_tilt"),
        **({"gdf_source": case["gdf_source"]} if "gdf_source" in case else {}),
    )


def _check_opt_in_transport(
    case, trajectory_capture, block_electrons, *, transport_core="auto", block_monitor=None
):
    """Refuse transport options an opt-in component lacks, before any work (ADR-0017 tier 3)."""
    if case.get("photon_transport") is not None:
        from .photon_aperture import prepare_photon_case

        if block_electrons is not None or block_monitor is not None:
            raise ValueError("photon transport does not support block/adaptive transport")
        prepare_photon_case(case, transport_core)

    refuse_axial_channeling(case)
    if trajectory_capture is not None:
        refuse_channeling(case, "trajectory capture")
    if block_electrons is not None:
        refuse_channeling(case, "electron-block transport")
        refuse_transition_radiation(case, "electron-block transport")


@contextmanager
def run_progress(progress):
    """Route in-process transport batches and spectrum stages to ``progress``.

    ``progress`` receives :class:`pyrite.phases.SimulationProgress` events;
    ``None`` installs nothing, so progress-free runs keep their call shapes.
    Pool and executor workers never see the sink.
    """
    if progress is None:
        yield
        return
    from ..._simulation_progress import SimulationProgress
    from . import _SPECTRUM_PROGRESS, _TRANSPORT_PROGRESS

    def transport(done, total):
        progress(SimulationProgress("transport", "electrons", int(done), int(total)))

    def spectrum(stage, done, total):
        progress(SimulationProgress("spectrum", stage, int(done), int(total)))

    tokens = _TRANSPORT_PROGRESS.set(transport), _SPECTRUM_PROGRESS.set(spectrum)
    try:
        yield
    finally:
        _SPECTRUM_PROGRESS.reset(tokens[1])
        _TRANSPORT_PROGRESS.reset(tokens[0])
