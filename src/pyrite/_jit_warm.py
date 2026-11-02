"""Opt-in JIT cache warming with fixed representative one-electron cases.

``pyrite cache warm`` runs each case of a profile through :func:`pyrite.simulate`
with one CXR and one continuum electron. Compiled specializations depend on
argument types and compile-time flags, not on history counts, so a warmed case
compiles exactly what a full-size case with the same physics flags needs, and
the native Numba and CuPy caches persist it. Results are discarded; nothing is
written outside those caches. Never run on import or install.

CPU transport cores specialize on the (inelastic, radiative) flags. The default
HOPG case resolves to shell soft/hard collisions with coupled radiative loss; a
material without shell data falls back to continuous stopping (with coupled
radiative loss when its BremsLib tables are installed), which is a different
core. ``standard`` therefore pins each flag combination explicitly rather than
naming materials.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarmCase:
    """One representative case: a name, what it covers, and its physics options."""

    name: str
    covers: str
    physics: dict[str, Any]


_SHELL = {"inelastic_model": "shell-soft-hard", "inelastic_cutoff_eV": 50.0}

PROFILES: dict[str, tuple[WarmCase, ...]] = {
    "minimal": (WarmCase("default", "default physics (README case)", {}),),
    "standard": (
        WarmCase("default", "default physics (README case)", {}),
        WarmCase(
            "continuous",
            "continuous stopping, coupled radiative (materials without shell data)",
            {"inelastic_model": "continuous"},
        ),
        WarmCase(
            "uncoupled", "shell soft/hard, uncoupled continuum", {"radiative_model": "uncoupled"}
        ),
        WarmCase(
            "continuous-uncoupled",
            "continuous stopping, uncoupled continuum",
            {"inelastic_model": "continuous", "radiative_model": "uncoupled"},
        ),
        WarmCase(
            "secondaries",
            "shell soft/hard with secondary transport above 1 keV",
            {**_SHELL, "secondary_threshold_eV": 1000.0},
        ),
    ),
}


def _scene():
    import pyrite as pr

    beam = pr.Beam(
        energy_keV=30.0,
        transverse=pr.beams.TransverseSpot(),
        longitudinal=pr.beams.PointBunch(),
    )
    target = pr.geometry.Slab(
        "hopg",
        thickness_ang=10_000.0,
        tilt_deg=30.0,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    return pr, beam, target


def _run_case(case: WarmCase, cupy: bool) -> dict:
    from contextlib import nullcontext

    from ._jit_cache import CupyCompileRecorder, NumbaCompileRecorder

    pr, beam, target = _scene()
    record: dict[str, Any] = {"case": case.name, "covers": case.covers}
    start = time.perf_counter()
    cupy_recorder = CupyCompileRecorder() if cupy else None
    try:
        with (
            NumbaCompileRecorder() as numba_recorder,
            cupy_recorder or nullcontext(),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            pr.simulate(
                beam,
                target,
                pr.Detector(),
                physics=pr.Physics(**case.physics),
                numerics=pr.Numerics(cxr_trials=1, brem_trials=1),
            )
    except Exception as exc:  # noqa: BLE001 - one unavailable case must not stop the rest
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        record["status"] = "ok"
        record["warnings"] = sorted({str(w.message) for w in caught})
        record["numba_compiles"] = len(numba_recorder.records)
        record["numba_compile_s"] = numba_recorder.total_s
        if cupy_recorder is not None:
            record["cupy"] = cupy_recorder.summary()
    record["seconds"] = time.perf_counter() - start
    return record


def warm(profile: str, *, progress: Callable[[WarmCase], None] | None = None) -> dict:
    """Compile and cache the kernels of every case in ``profile``.

    Uses the backend ``PYRITE_MC_BACKEND`` selects and the cache directories
    already in the environment. Returns per-case status, wall time and compile
    counts; a failed case (for example missing physics tables) is reported and
    the remaining cases still run.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown warm profile {profile!r}; expected one of {sorted(PROFILES)}")
    from ._backend import CUPY_BACKENDS, select_backend

    backend = select_backend().name
    cupy = backend in CUPY_BACKENDS
    cases = []
    for case in PROFILES[profile]:
        if progress is not None:
            progress(case)
        cases.append(_run_case(case, cupy))
    return {"profile": profile, "backend": backend, "cases": cases}
