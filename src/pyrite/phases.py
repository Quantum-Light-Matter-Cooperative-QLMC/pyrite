"""Run a scalar simulation's transport and spectrum phases separately (#776).

:func:`pyrite.simulate` transports electrons and then scores their spectra in
one call. :func:`simulate_transport` stops after transport and keeps the
segments in memory (on the device for CUDA transport); :func:`score_spectra`
scores them, as often as needed, without transporting again. Scoring the same
transport twice with different spectral chunks repeats only the spectrum work,
which is what spectral kernel tuning needs. Persistent replay from disk remains
``pyrite run --trajectories`` plus ``pyrite trajectories score``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ._numerics_groups import KernelTuning
from ._simulation_progress import SimulationProgress
from .api import _scalar_result, _source_case
from .campaign.model import Numerics, Scene
from .campaign.resolution import ResolvedSimulation, resolve_simulation
from .detectors import PlanarDetector
from .montecarlo import Case
from .results.model import Result

__all__ = ["SimulationProgress", "TransportedScene", "score_spectra", "simulate_transport"]

Progress = Callable[[SimulationProgress], None]


@dataclass(frozen=True, eq=False)
class TransportedScene:
    """One scalar scene after electron transport, before spectra.

    Returned by :func:`simulate_transport`; pass it to :func:`score_spectra`.
    It holds every transported segment, so its memory is that of the
    transport. CUDA segments stay resident on the device until the object is
    released, or until a spectrum-phase device OOM moves them, unchanged, to
    host memory.
    """

    scene: Scene
    numerics: Numerics
    case: Case
    _request: ResolvedSimulation = field(repr=False)
    _xsgen_tables: dict[str, str] = field(repr=False)
    _transport: dict[str, Any] = field(repr=False)

    @property
    def segment_count(self) -> int:
        """Number of transported electron segments the spectra score."""
        return int(self._transport["segs"]["L_ang"].shape[0])


def simulate_transport(
    scene: Scene, *, numerics: Numerics | None = None, progress: Progress | None = None
) -> TransportedScene:
    """Transport ``scene``'s electrons without scoring any spectrum.

    Applies the scene's (or ``numerics``') sampling, seed and
    :class:`~pyrite.config.KernelTuning` exactly as :func:`pyrite.simulate`
    does, so ``score_spectra(simulate_transport(scene))`` equals
    ``simulate(scene)`` on the same backend. ``progress`` receives a
    :class:`SimulationProgress` after each transport batch.

    Raises
    ------
    ValueError
        For a pixelated (planar) detector or a separate observation; those
        paths score directions inside their own transport loop.
    """
    from .montecarlo.runner import _transport_case
    from .montecarlo.runner.transport_options import run_progress, transport_tuning

    if not isinstance(scene, Scene):
        raise TypeError("scene must be a pyrite.Scene")
    request = resolve_simulation(scene, numerics=numerics, require_detector=True)
    if isinstance(request.scene.detector, PlanarDetector) or request.observation is not None:
        raise ValueError("simulate_transport supports scalar detectors without an observation")
    case, xsgen_tables = _source_case(request)
    with transport_tuning(request.numerics), run_progress(progress):
        transport = _transport_case(
            case, transport_core=request.numerics.transport_core, keep_segments_on_device=True
        )
    return TransportedScene(request.scene, request.numerics, case, request, xsgen_tables, transport)


def score_spectra(
    transported: TransportedScene,
    *,
    kernel_tuning: KernelTuning | None = None,
    progress: Progress | None = None,
) -> Result:
    """Score the spectra of an existing transport; never transports.

    ``kernel_tuning`` may change only the spectral execution controls
    (``cxr_chunk``, ``brem_chunk``, ``cxr_threads``, ``cxr_energies_per_block``); its transport fields must equal the ones
    that produced ``transported``. Chunks change memory use and speed, not the
    physics: ``Result.case`` and its identity digest are those of the transport.
    Repeated calls reuse the same segments. ``progress`` receives one
    :class:`SimulationProgress` per completed spectrum stage.
    """
    from .montecarlo.runner import _spectrum_with_recovery
    from .montecarlo.runner.transport_options import run_progress, spectrum_launch

    if not isinstance(transported, TransportedScene):
        raise TypeError("transported must come from simulate_transport")
    fixed = transported.numerics.kernel_tuning
    tuning = fixed if kernel_tuning is None else kernel_tuning
    if not isinstance(tuning, KernelTuning):
        raise TypeError("kernel_tuning must be a KernelTuning")
    if (tuning.transport_core, tuning.transport) != (fixed.transport_core, fixed.transport):
        raise ValueError(
            "score_spectra may change only the spectral chunks and CXR launch; "
            "transport settings were fixed by simulate_transport"
        )
    case = replace(transported.case, spec_chunk=tuning.cxr_chunk, brem_chunk=tuning.brem_chunk)
    # The spectrum phase may replace entries (CPU fallback); keep ours intact,
    # except that a device-OOM move of the rows to host memory (#905) also
    # applies here, or this object would keep the device rows alive.
    with run_progress(progress), spectrum_launch(tuning):
        output = _spectrum_with_recovery(
            case, dict(transported._transport), segs_owner=transported._transport
        )
    if transported.case.get("channeling") is not None:
        output["channeling"] = transported._transport["segs"]["channeling"]
    numerics = transported.numerics.with_kernel_tuning(tuning)
    return _scalar_result(
        transported._request, transported.case, transported._xsgen_tables, output, numerics
    )
