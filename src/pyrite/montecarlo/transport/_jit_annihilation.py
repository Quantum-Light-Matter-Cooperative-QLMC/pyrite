"""Device-resident two-photon annihilation primitives.

PENELOPE-2024 section 3.4, Eqs. 3.184--3.197, free target electrons at rest.
The rejection bound is the true maximum gamma**2 + 2*gamma - 1. At rest
the two photons are antiparallel and have energy m_e c**2. Inputs and all
outputs stay on the device; invalid events or exhausted rejection budgets
have NaN outputs and a false ``valid`` flag, without a host synchronization.

Only imported by callers requesting CUDA emission. Per-event keys feed the
existing counter-based SplitMix64 stream: two draws per rejection attempt,
then the azimuth, then four first-interaction draws. At rest uses two axis
draws followed by four first-interaction draws. Event order and batch size
cannot change a stream. The legacy host emitter retains its Philox streams.

Validation: heitler-annihilation, positron-annihilation-at-rest
"""

import cupy as cp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import F64_ONE, F64_PI, F64_TWO, U64_ONE, U64_ZERO, _stream_uniform
from ._jit_shell_device import _store_rotated
from .annihilation import _MAX_REJECTIONS, _PI_RE2_ANG2, ELECTRON_REST_KEV

F64_MC2 = np.float64(ELECTRON_REST_KEV)
F64_PI_RE2 = np.float64(_PI_RE2_ANG2)
F64_NAN = np.float64(np.nan)
F64_INF = np.float64(np.inf)
I8_REST = np.int8(0)
I8_FLIGHT = np.int8(1)
U64_TWO = np.uint64(2)


@jit.rawkernel(device=True)
def _heitler_cross_section_ang2(kinetic_keV):
    """PENELOPE-2024 Eq. 3.189 per free electron [Angstrom^2].

    Positive kinetic energy in keV; sigma*beta tends to pi*r_e**2 at rest.
    Validation: heitler-annihilation
    """
    g = F64_ONE + kinetic_keV / F64_MC2
    s2 = g * g - F64_ONE
    if not (kinetic_keV > np.float64(0.0) and s2 > np.float64(0.0) and s2 < F64_INF):
        return F64_NAN
    s = cp.sqrt(s2)
    return (
        F64_PI_RE2
        / ((g + F64_ONE) * s2)
        * ((g * g + np.float64(4.0) * g + F64_ONE) * cp.log(g + s) - (np.float64(3.0) + g) * s)
    )


@jit.rawkernel()
def _cross_sections(kinetic, out):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < kinetic.size:
        out[i] = _heitler_cross_section_ang2(kinetic[i])


@jit.rawkernel()
def _sample(kinds, kinetic, directions, keys, max_rejections, energies, photons, uniforms, valid):
    """Eqs. 3.184--3.197, with isotropic two-photon emission at rest.

    Free target electron at rest; conserves four-momentum in flight and
    tends to two m_e*c**2 photons as kinetic energy vanishes.
    Validation: heitler-annihilation, positron-annihilation-at-rest
    """
    i = np.int64(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i >= kinetic.size:
        return
    key = keys[i]
    counter = U64_ZERO
    if kinds[i] == I8_REST:
        cosine = F64_TWO * _stream_uniform(key, counter) - F64_ONE
        phi = F64_TWO * F64_PI * _stream_uniform(key, counter + U64_ONE)
        counter += U64_TWO
        sine = cp.sqrt(max(np.float64(0.0), F64_ONE - cosine * cosine))
        photons[i * 6] = sine * cp.cos(phi)
        photons[i * 6 + 1] = sine * cp.sin(phi)
        photons[i * 6 + 2] = cosine
        for j in range(3):
            photons[i * 6 + 3 + j] = -photons[i * 6 + j]
        energies[i * 2] = F64_MC2
        energies[i * 2 + 1] = F64_MC2
    elif kinds[i] == I8_FLIGHT:
        energy = kinetic[i]
        if not (energy > np.float64(0.0) and energy < F64_INF):
            return
        dx = directions[i * 3]
        dy = directions[i * 3 + 1]
        dz = directions[i * 3 + 2]
        norm = cp.sqrt(dx * dx + dy * dy + dz * dz)
        if not (norm > np.float64(0.0) and norm < F64_INF):
            return
        dx /= norm
        dy /= norm
        dz /= norm
        g = F64_ONE + energy / F64_MC2
        s2 = g * g - F64_ONE
        if not (s2 > np.float64(0.0) and s2 < F64_INF):
            return
        s = cp.sqrt(s2)
        zmin = F64_ONE / (g + F64_ONE + s)
        ratio = (F64_ONE - zmin) / zmin
        a = g * g + np.float64(4.0) * g + F64_ONE
        bound = g * g + F64_TWO * g - F64_ONE
        zeta = np.float64(-1.0)
        for _attempt in range(max_rejections):
            v = zmin * ratio ** _stream_uniform(key, counter)
            u = _stream_uniform(key, counter + U64_ONE)
            counter += U64_TWO
            if u * bound <= a - (g + F64_ONE) ** 2 * v - F64_ONE / v:
                zeta = min(v, F64_ONE - v)
                break
        if zeta < np.float64(0.0):
            return
        phi = F64_TWO * F64_PI * _stream_uniform(key, counter)
        counter += U64_ONE
        total = energy + F64_TWO * F64_MC2
        energies[i * 2] = zeta * total
        energies[i * 2 + 1] = total - energies[i * 2]
        c0 = min(F64_ONE, max(-F64_ONE, (g + F64_ONE - F64_ONE / zeta) / s))
        c1 = min(F64_ONE, max(-F64_ONE, (g + F64_ONE - F64_ONE / (F64_ONE - zeta)) / s))
        _store_rotated(photons, i * 6, dx, dy, dz, c0, phi)
        _store_rotated(photons, i * 6 + 3, dx, dy, dz, c1, phi + F64_PI)
    else:
        return
    for j in range(4):
        uniforms[i * 4 + j] = _stream_uniform(key, counter)
        counter += U64_ONE
    valid[i] = True


def heitler_cross_section_device(kinetic_keV):
    """Eq. 3.189 per electron [Angstrom^2], device-in/device-out.

    ``kinetic_keV`` is a CuPy array of positive finite energies [keV]. The
    output has the same shape and float64 precision, with no host transfer.
    Nonpositive/nonfinite energies, gamma rounded to 1, or overflow of
    gamma squared produce NaN. Accuracy follows the host expression.
    Low-energy limit: sigma*beta -> pi*r_e**2. Free target electrons at rest.
    Validation: heitler-annihilation
    """
    if not isinstance(kinetic_keV, cp.ndarray):
        raise TypeError("kinetic_keV must be a CuPy array")
    kinetic = cp.ascontiguousarray(kinetic_keV, dtype=cp.float64)
    out = cp.empty(kinetic_keV.shape, dtype=cp.float64)
    if kinetic.size:
        _cross_sections(((kinetic.size + 127) // 128,), (128,), (kinetic.ravel(), out.ravel()))
    return out


def sample_annihilation_device(
    kinds, kinetic_keV, directions, keys, *, max_rejections=_MAX_REJECTIONS
):
    """Batched two-photon sampler, device-in/device-out with per-event streams.

    PENELOPE-2024 Eqs. 3.184--3.197, free electron at rest, no binding or
    Doppler broadening. ``kinds`` (N,) is 0 at rest or 1 in flight;
    ``kinetic_keV`` (N,) is positive and finite in flight, ignored at rest;
    ``directions`` (N,3) is finite and nonzero in flight, ignored at rest;
    ``keys`` (N,) contains integer uint64 stream keys. All inputs must be
    CuPy arrays; dtype conversion and contiguous copies stay on the device.

    Returns CuPy arrays ``k_keV`` (N,2), ``direction`` (N,2,3),
    ``uniforms`` (N,2,2) for photon first interactions, and ``valid`` (N,).
    Invalid events or rejection exhaustion have NaN payloads and false
    flags; checking flags is the caller's responsibility at its host boundary.
    Energies whose gamma rounds to 1 or whose gamma squared overflows are
    invalid; the float64 formula has the same conditioning as the host.
    Empty batches launch no kernel. At rest energies are exactly m_e*c**2
    and directions are exactly antiparallel; in flight four-momentum closes.
    Validation: heitler-annihilation, positron-annihilation-at-rest
    """
    if not all(isinstance(value, cp.ndarray) for value in (kinds, kinetic_keV, directions, keys)):
        raise TypeError("all annihilation sampler inputs must be CuPy arrays")
    n = kinetic_keV.size
    if (
        kinetic_keV.shape != (n,)
        or kinds.shape != (n,)
        or keys.shape != (n,)
        or directions.shape != (n, 3)
    ):
        raise ValueError("expected kinds, kinetic_keV, keys (N,) and directions (N, 3)")
    if kinds.dtype.kind not in "iu" or keys.dtype.kind not in "iu":
        raise TypeError("kinds and keys must have integer dtypes")
    if not isinstance(max_rejections, int) or not 0 <= max_rejections <= np.iinfo(np.int32).max:
        raise ValueError("max_rejections must be a nonnegative int32-sized integer")
    # Keep kind values wide enough that e.g. 256 cannot silently become at rest.
    kinds = cp.ascontiguousarray(kinds, dtype=cp.int64)
    kinetic = cp.ascontiguousarray(kinetic_keV, dtype=cp.float64)
    directions = cp.ascontiguousarray(directions, dtype=cp.float64)
    keys = cp.ascontiguousarray(keys, dtype=cp.uint64)
    out = {
        "k_keV": cp.full((n, 2), cp.nan, dtype=cp.float64),
        "direction": cp.full((n, 2, 3), cp.nan, dtype=cp.float64),
        "uniforms": cp.full((n, 2, 2), cp.nan, dtype=cp.float64),
        "valid": cp.zeros(n, dtype=cp.bool_),
    }
    if n:
        _sample(
            ((n + 127) // 128,),
            (128,),
            (
                kinds,
                kinetic,
                directions.ravel(),
                keys,
                np.int32(max_rejections),
                out["k_keV"].ravel(),
                out["direction"].ravel(),
                out["uniforms"].ravel(),
                out["valid"],
            ),
        )
    return out
