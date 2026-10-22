"""Real CUDA parity of optical-band TR/CTR complex fields and pair reductions (#615).

Stage a clean committed tree with ``pyrite remote sync`` and run in a bounded
SLURM GPU allocation on that host:
``uv run --extra nvidia pyrite-dev validation-records -- checks/optical_transition_cuda.py``.
Missing CUDA fails; there is no skip or CPU fallback. Prints JSON evidence.

Compares, on a multilayer gold/vacuum stack with Johnson–Christy and Rakić
optics over their common optical band, the NumPy chunked reference with the
CUDA fused kernel and the CuPy array route: per-history power and the complex
bunch field before squaring, then the optical scorer's self, pair and
per-electron yields for sampled (importance-weighted) and Gaussian laws.
Tolerances are peak-scaled absolute floors (near-zero cancellations) plus a
relative bound; they are set from float64 solve rounding, not copied from the
X-ray parity checks.
Validation: optical-planar-tr, optical-ctr-population (measurement only).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from pyrite.materials.crystal import M_E_EV
from pyrite.materials.optical import gold_johnson_christy_response, gold_rakic_response
from pyrite.montecarlo.spectrum import transition_radiation_crossings as crossings_module
from pyrite.montecarlo.spectrum.optical_transition import (
    OpticalBunchSource,
    optical_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    BoundaryCrossings,
    history_field_sums,
)

RTOL, ATOL_OVER_PEAK = 1e-9, 1e-11


def _histories(m: int, rng) -> BoundaryCrossings:
    """``m`` straight histories through three layers, random offsets and tilts."""
    planes = np.array([0.0, 300.0, 900.0, 1500.0])  # Au | vacuum gap | Au film [Å]
    k = planes.size
    beta = np.sqrt(1.0 - 1.0 / (1.0 + 30e3 / M_E_EV) ** 2)
    tilt = rng.normal(0.0, 0.05, (m, 2))
    direction = np.column_stack([tilt, np.ones(m)])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    t0 = rng.normal(0.0, 400.0, m)
    entry = np.column_stack([rng.normal(0.0, 600.0, (m, 2)), np.zeros(m)])
    path = planes[None, :] / direction[:, 2:3]  # (m, k)
    position = entry[:, None, :] + path[..., None] * direction[:, None, :]
    position[..., 2] = planes[None, :]
    time = t0[:, None] + path / beta
    rows = m * k
    return BoundaryCrossings(
        electron_id=np.repeat(np.arange(m), k), track_id=np.repeat(np.arange(m), k),
        charge=-np.ones(rows, np.int8), position_ang=position.reshape(-1, 3),
        direction=np.repeat(direction, k, axis=0),
        kinetic_energy_keV=np.full(rows, 30.0), time_ang=time.reshape(-1),
        interface=np.tile(np.arange(k), m), normal=np.tile([0.0, 0.0, 1.0], (rows, 1)),
        layer_before=np.tile(np.r_[-1, 0, 1, 2], m), layer_after=np.tile(np.r_[0, 1, 2, -1], m),
        z_interfaces_ang=planes, n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _compare(name, gpu, cpu, out):
    gpu, cpu = np.asarray(gpu), np.asarray(cpu)
    peak = float(np.max(np.abs(cpu)))
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL_OVER_PEAK * peak, err_msg=name)
    out[name] = float(np.max(np.abs(gpu - cpu)) / peak)


def main() -> int:
    """Run the parity comparisons; return the process exit code."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    rng = np.random.default_rng(615)
    m = 257
    crossings = _histories(m, rng)
    jc, rakic = gold_johnson_christy_response(), gold_rakic_response()
    energy = np.geomspace(0.7, 4.9, 97)
    eps_host = [jc.epsilon(energy), np.ones(energy.size, complex), rakic.epsilon(energy)]
    theta = np.linspace(0.05, 1.45, 9)
    out: dict[str, float] = {}
    for label, sign in (("forward", 1.0), ("backward", -1.0)):
        n = np.stack([np.sin(theta), 0.3 * np.sin(theta), sign * np.cos(theta)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        e2, n2 = energy[:, None], n[None]
        eps2 = [e[:, None] for e in eps_host]
        cpu_power, cpu_field = history_field_sums(crossings, e2, n2, eps2, coherence="bunch")
        routes = {"fused": True, "array": False}
        for route, fused in routes.items():
            crossings_module._USE_JIT_TRANSITION = fused
            power, field = history_field_sums(crossings, e2, n2, eps2, coherence="bunch", xp=cp)
            cp.cuda.get_current_stream().synchronize()
            _compare(f"{label}/{route}/power", cp.asnumpy(power), cpu_power, out)
            _compare(f"{label}/{route}/field_re", cp.asnumpy(field).real, cpu_field.real, out)
            _compare(f"{label}/{route}/field_im", cp.asnumpy(field).imag, cpu_field.imag, out)
        crossings_module._USE_JIT_TRANSITION = True
        weights = rng.uniform(0.2, 3.0, m)
        sources = {
            "sampled-weighted": OpticalBunchSource.sampled(1.0, "parity", weights=weights),
            "gaussian": OpticalBunchSource.gaussian(1.0, "parity", rms_t_fs=0.2, rms_x_um=0.05),
        }
        for law, source in sources.items():
            scored = crossings if law != "gaussian" else _offset_free(crossings)
            cpu = optical_transition_radiation(scored, e2, n2, eps2, source=source, n_histories=m)
            gpu = optical_transition_radiation(
                scored, e2, n2, eps2, source=source, n_histories=m, xp=cp
            )
            for part in ("self_yield", "pair_yield", "yield_per_electron"):
                _compare(f"{label}/{law}/{part}", getattr(gpu, part), getattr(cpu, part), out)
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device = properties["name"]
    print(
        json.dumps(
            {
                "validation_ids": ["optical-planar-tr", "optical-ctr-population"],
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "device": device.decode() if isinstance(device, bytes) else device,
                "compute_capability": cp.cuda.Device().compute_capability,
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "histories": m,
                "crossings": len(crossings),
                "energies": int(energy.size),
                "band_eV": [float(energy[0]), float(energy[-1])],
                "directions_per_face": int(theta.size),
                "response_digests": {"jc": jc.digest, "rakic": rakic.digest},
                "rtol": RTOL,
                "atol_over_peak": ATOL_OVER_PEAK,
                "peak_normalized_max_abs_difference": out,
            },
            sort_keys=True,
        )
    )
    return 0


def _offset_free(crossings: BoundaryCrossings) -> BoundaryCrossings:
    """Remove each history's entrance time and transverse offset (Gaussian law)."""
    first = np.r_[True, crossings.electron_id[1:] != crossings.electron_id[:-1]]
    index = np.cumsum(first) - 1
    t0 = crossings.time_ang[first][index]
    r0 = crossings.position_ang[first][index]
    position = crossings.position_ang.copy()
    position[:, :2] -= r0[:, :2]
    return replace(crossings, time_ang=crossings.time_ang - t0, position_ang=position)


if __name__ == "__main__":
    sys.exit(main())
