"""Measurement worker process of :func:`pyrite.optimization.tune_kernels` (#776).

Run as ``python -m pyrite.optimization._kernel_worker PAYLOAD``. The payload
pickle holds the scene, warm-up and repeat counts. The worker then answers one
JSON request per stdin line with JSON lines on its original stdout (anything
the simulation prints goes to stderr):

``{"phase": "transport"|"spectrum"|"total", "tuning": {...}, "label": str}``
    ``warmups`` untimed then ``repeats`` synchronized runs of the phase under
    that ``KernelTuning``. Streams ``{"event": "progress", ...}`` and ends with
    ``{"event": "result", "samples_s": [...], "check": {...}}`` or
    ``{"event": "error", "message": str}``.

``transport`` times :func:`pyrite.transport.simulate_trajectories` only and checks a
digest of every transported array. ``spectrum`` transports once per worker
with the baseline tuning, then times :func:`pyrite.score_spectra` on
that kept transport. ``total`` times :func:`pyrite.simulate`. Spectra are
saved as ``<workdir>/<label>.npz`` for the coordinator to compare.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

FAMILIES = ("cxr.incoherent", "cxr.coherent", "characteristic", "channeling", "transition", "brem")
_PROGRESS_INTERVAL_S = 0.25
_CHECKSUM_CHUNK_WORDS = 1 << 24  # bounds checksum scratch on the device


def tuning_from_dict(data: dict[str, Any]) -> Any:
    from .._numerics_groups import KernelTuning, TransportKernelTuning

    return KernelTuning(
        transport_core=data["transport_core"],
        cxr_chunk=data["cxr_chunk"],
        brem_chunk=data["brem_chunk"],
        transport=TransportKernelTuning(**data["transport"]),
        cxr_threads=data.get("cxr_threads"),
        cxr_energies_per_block=data.get("cxr_energies_per_block"),
    )


def environment() -> dict[str, Any]:
    """Device, precision and library versions the timings belong to."""
    from .. import __version__
    from .._backend import BACKEND

    env: dict[str, Any] = {
        "backend": BACKEND.name,
        "device": getattr(BACKEND.device, "name", str(BACKEND.device)),
        "precision": "fp64" if os.environ.get("PYRITE_FP64", "0") not in ("", "0") else "fp32",
        "pyrite_version": __version__,
    }
    try:
        import cupy

        env |= {
            "cupy": cupy.__version__,
            "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cupy.cuda.runtime.driverGetVersion(),
        }
    except Exception:  # noqa: BLE001 - provenance only; the CUDA check is separate
        pass
    return env


def _named_arrays(value: Any, name: str = "") -> Iterator[tuple[str, Any]]:
    """Every host or device array nested in ``value``, in a stable order, with its path."""
    from .._backend import BACKEND

    if isinstance(value, np.ndarray) or BACKEND.is_array(value):
        yield name, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _named_arrays(value[key], f"{name}/{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _named_arrays(item, f"{name}/{index}")


def _chunk_sums(xp: Any, chunk: Any, start: int) -> tuple[int, int]:
    """Two wrapping sums of position-keyed words; each term is a bijection of its word."""
    constant = partial(xp.asarray, dtype=xp.uint64)
    positions = xp.arange(start, start + chunk.size, dtype=xp.uint64)
    keyed = chunk.astype(xp.uint64) + positions * constant(0x9E3779B97F4A7C15)
    first = (keyed ^ (keyed >> constant(30))) * constant(0xBF58476D1CE4E5B9)
    second = (first ^ (first >> constant(31))) * constant(0x94D049BB133111EB)
    return int(xp.sum(first, dtype=xp.uint64)), int(xp.sum(second, dtype=xp.uint64))


def _array_checksum(xp: Any, array: Any) -> tuple[int, int]:
    """Two wrapping 64-bit sums over an array's bits, computed with ``xp``.

    Any change to one word changes both sums, and reordered words change them
    with high probability. Host arrays upload one bounded chunk at a time, and
    the value does not depend on where the array lives. This detects accidental
    differences between candidates; it is not cryptographic.
    """
    source = np if isinstance(array, np.ndarray) else xp
    words = source.ravel(array).view(f"u{min(array.dtype.itemsize, 8)}")
    totals = [0, 0]
    for start in range(0, int(words.size), _CHECKSUM_CHUNK_WORDS):
        chunk = xp.asarray(words[start : start + _CHECKSUM_CHUNK_WORDS])
        for index, value in enumerate(_chunk_sums(xp, chunk, start)):
            totals[index] = (totals[index] + value) % (1 << 64)
    return totals[0], totals[1]


def transport_digest(transported: Any) -> str:
    """Identity of every transported array and the resolved line grid.

    Checksums run on the device, so a candidate need not download its segments
    to be compared. Digests are only compared within one
    ``tune_kernels`` call, so their value carries no compatibility promise.
    """
    from .._backend import BACKEND

    tp = transported._transport
    digest = hashlib.sha256()
    for name, value in (*_named_arrays(tp["segs"], "segs"), *_named_arrays(tp["E_grid"], "E_grid")):
        checksum = _array_checksum(BACKEND.xp, value)
        digest.update(f"{name}:{value.dtype}:{value.shape}:{checksum}".encode())
    return digest.hexdigest()


def save_spectra(result: Any, path: Path) -> None:
    arrays = {}
    for name in FAMILIES:
        density = result
        for part in name.split("."):
            density = getattr(density, part)
        if density is not None:
            arrays[f"{name}:grid"] = np.asarray(density.photon_energy_eV)
            arrays[f"{name}:density"] = np.asarray(density.density)
    np.savez(path, allow_pickle=False, **arrays)


class _Session:
    def __init__(self, payload: dict[str, Any], send) -> None:
        import pyrite

        from .._backend import BACKEND

        if BACKEND.name != "cuda":
            raise RuntimeError(f"CUDA required, resolved backend={BACKEND.name!r}")
        self.pyrite, self.backend, self.send = pyrite, BACKEND, send
        self.scene = payload["scene"]
        self.workdir = Path(payload["workdir"])
        self.warmups, self.repeats = payload["warmups"], payload["repeats"]
        self.base = tuning_from_dict(payload["base"])
        self._transport = None  # kept for spectrum candidates
        self._warmed: set[str] = set()

    def _forward(self, label: str, run: int, runs: int):
        last = [0.0]

        def progress(event):
            now = perf_counter()
            if event.done < event.total and now - last[0] < _PROGRESS_INTERVAL_S:
                return
            last[0] = now
            self.send(
                event="progress", label=label, run=run, runs=runs, phase=event.phase,
                stage=event.stage, done=event.done, total=event.total,
            )  # fmt: skip

        return progress

    def _kept_transport(self):
        from ..campaign.trajectories import scene_trajectories

        if self._transport is None:
            numerics = self.scene.numerics.with_kernel_tuning(self.base)
            self._transport = scene_trajectories(self.scene, numerics=numerics)
        return self._transport

    def measure(self, phase: str, tuning: Any, label: str, warmups: int, repeats: int):
        from ..campaign.trajectories import scene_trajectories
        from ..scoring import score_spectra

        numerics = self.scene.numerics.with_kernel_tuning(tuning)
        if phase == "spectrum":
            transported = self._kept_transport()
            spectral = transported.numerics.kernel_tuning
            spectral = type(spectral)(
                transport_core=spectral.transport_core,
                transport=spectral.transport,
                cxr_chunk=tuning.cxr_chunk,
                brem_chunk=tuning.brem_chunk,
                cxr_threads=tuning.cxr_threads,
                cxr_energies_per_block=tuning.cxr_energies_per_block,
            )

            def run(progress):
                return score_spectra(
                    transported,
                    numerics=transported.numerics.with_kernel_tuning(spectral),
                    progress=progress,
                )
        elif phase == "transport":

            def run(progress):
                return scene_trajectories(self.scene, numerics=numerics, progress=progress)
        else:

            def run(progress):
                return self.pyrite.simulate(self.scene, numerics=numerics, progress=progress)

        if phase not in self._warmed:
            # A phase's first measurement (its baseline) still ran ~15% slow
            # after one warm-up on an RTX 5080; give it one more.
            self._warmed.add(phase)
            warmups += 1
        self.backend.release_memory()  # each candidate starts from an empty pool
        samples, value = [], None
        runs = warmups + repeats
        for index in range(runs):
            # Free the previous run's arrays first: holding them would make
            # every later run allocate fresh device memory and time slower.
            value = None
            self.backend.synchronize()
            started = perf_counter()
            value = run(self._forward(label, index + 1, runs))
            self.backend.synchronize()
            if index >= warmups:
                samples.append(perf_counter() - started)
        assert value is not None  # at least one timed run
        if phase == "transport":
            check: dict[str, Any] = {"digest": transport_digest(value)}
        else:
            path = self.workdir / f"{label}.npz"
            save_spectra(value, path)
            check = {"spectra": str(path)}
            recovery = value.provenance.get("spectrum_recovery")
            if recovery:
                check["recovery"] = recovery
        del value
        return samples, check


def main(payload_path: str) -> None:
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # library prints must not corrupt the protocol stream
    sys.stdout = sys.stderr

    def send(**message):
        protocol.write(json.dumps(message) + "\n")

    session = _Session(pickle.loads(Path(payload_path).read_bytes()), send)
    send(event="ready", environment=environment())
    for line in sys.stdin:
        request = json.loads(line)
        try:
            samples, check = session.measure(
                request["phase"],
                tuning_from_dict(request["tuning"]),
                request["label"],
                request.get("warmups", session.warmups),
                request.get("repeats", session.repeats),
            )
        except Exception as error:  # noqa: BLE001 - reported to the coordinator as a rejection
            session.backend.release_memory()
            send(event="error", message=f"{type(error).__name__}: {error}"[-2500:])
            continue
        send(event="result", samples_s=samples, check=check)


if __name__ == "__main__":
    main(sys.argv[1])
