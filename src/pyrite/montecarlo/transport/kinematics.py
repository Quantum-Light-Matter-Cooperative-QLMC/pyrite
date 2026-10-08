"""RNG stream primitives and relativistic kinematics for electron transport."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numba import float64, int64, njit, uint64
from scipy.special import ndtri

# Speed of light in transport-clock units: the electron clock sum(L/beta) is in
# Angstrom (c=1), so a longitudinal bunch length in fs converts via
# c = 2997.924580 Ang/fs. Mirrors sweep.C_ANG_PER_FS / plots.trajectories
# .C_ANG_PER_FS (same physical constant; defined locally to keep transport free
# of a plots/sweep import cycle).
C_ANG_PER_FS = 2997.924580


def _sample_bunch_offsets(
    Ne,
    bunch_length_fs,
    long_shape,
    long_offsets_fs,
    seed,
    longitudinal_distribution: Mapping[str, Any] | None = None,
):
    """Per-electron longitudinal arrival offset ``Delta t`` [Angstrom, c=1],
    centered on the bunch centroid (decision 3).

    Source/convention: a longitudinal bunch is a distribution of electron
    arrival TIMES; ``Delta t_e`` [fs] converts to the transport clock's Angstrom
    units via ``c = 2997.924580 Ang/fs`` (:data:`C_ANG_PER_FS`). ``long_shape``
    selects the sampling law about a zero centroid:

    * ``"gaussian"`` -- ``Delta t ~ Normal(0, sigma)`` with RMS ``sigma =
      bunch_length_fs`` (the RMS convention, NOT FWHM).
    * ``"uniform"`` -- a flat-top of the SAME RMS: half-width ``sqrt(3)*sigma``.

    ``long_offsets_fs`` supplies explicit per-particle offsets (measured /
    arbitrary / microbunched profiles) and OVERRIDES ``long_shape`` /
    ``bunch_length_fs``. The draw uses an independent child namespace
    (``SeedSequence(seed).spawn(4)[3]`` -- the next index after the spot's
    ``spawn(2)[1]`` and the groove phase's ``spawn(3)[2]``), so enabling the
    bunch NEVER perturbs the main free-path / scattering draws. Inside it each
    electron's draws are counter-addressed by ``(seed, e, c)``
    (:func:`counter_normals`; ``c`` = 0 Gaussian/centre, 1 microbunch,
    2 jitter, 3 envelope, 4 mixture uniform), so electron ``e``'s raw offset
    never depends on the electron count. Only the centroid subtraction does:
    it is a property of the realized population.

    Limiting case: ``bunch_length_fs=None`` and ``long_offsets_fs=None`` ->
    all-zero (the legacy point bunch, bit-for-bit); ``bunch_length_fs -> 0``
    recovers it continuously.

    Validation: longitudinal-bunch-sampling
    """
    if longitudinal_distribution is not None:
        if bunch_length_fs is not None or long_offsets_fs is not None or long_shape != "gaussian":
            raise ValueError("longitudinal_distribution is incompatible with legacy bunch fields")
        kind = longitudinal_distribution.get("kind")
        root = child_stream_root(seed, 4, 3)
        if kind in ("gaussian", "compressed"):
            sigma_fs = longitudinal_distribution.get("rms_duration_fs")
            if sigma_fs is None:
                raise ValueError(f"{kind} resolution requires rms_duration_fs")
            dt = float(sigma_fs) * counter_normals(root, Ne, 1)[:, 0]
        elif kind == "microtrain":
            envelope_fs = float(longitudinal_distribution["envelope_rms_fs"])
            microbunch_fs = float(longitudinal_distribution["microbunch_rms_fs"])
            spacing_fs = float(longitudinal_distribution["spacing_fs"])
            jitter_fs = float(longitudinal_distribution["timing_jitter_fs"])
            depth = float(longitudinal_distribution["modulation_depth"])
            if not (
                np.isfinite(envelope_fs)
                and np.isfinite(microbunch_fs)
                and np.isfinite(spacing_fs)
                and envelope_fs > 0.0
                and microbunch_fs >= 0.0
                and spacing_fs > 0.0
                and jitter_fs >= 0.0
                and 0.0 <= depth <= 1.0
            ):
                raise ValueError("invalid resolved microtrain parameters")
            center_variance = envelope_fs**2 - microbunch_fs**2 - jitter_fs**2
            if center_variance <= 0.0:
                raise ValueError(
                    "microtrain envelope RMS must exceed combined microbunch width and jitter"
                )
            center_sigma_fs = np.sqrt(center_variance)
            z = counter_normals(root, Ne, 4)
            centers = np.rint((center_sigma_fs / spacing_fs) * z[:, 0])
            train = centers * spacing_fs + microbunch_fs * z[:, 1] + jitter_fs * z[:, 2]
            unmodulated = envelope_fs * z[:, 3]
            if depth == 1.0:
                dt = train
            elif depth == 0.0:
                dt = unmodulated
            else:
                mask = counter_uniforms(root, Ne, 5)[:, 4] < depth
                dt = np.where(mask, train, unmodulated)
        else:
            raise ValueError(f"unknown resolved longitudinal kind {kind!r}")
        dt = dt * C_ANG_PER_FS
    elif long_offsets_fs is not None:
        dt = np.asarray(long_offsets_fs, dtype=float)
        if dt.ndim != 1:
            raise ValueError("long_offsets_fs must be one-dimensional")
        if dt.size != Ne:
            raise ValueError(
                f"long_offsets_fs has {dt.size} entries but Ne={Ne}; supply one "
                "explicit longitudinal offset per electron"
            )
        if not np.all(np.isfinite(dt)):
            raise ValueError("long_offsets_fs must contain only finite values")
        dt = dt * C_ANG_PER_FS
    elif bunch_length_fs is not None:
        sigma_fs = float(bunch_length_fs)
        if not np.isfinite(sigma_fs) or sigma_fs < 0.0:
            raise ValueError("bunch_length_fs must be finite and non-negative")
        sigma_ang = sigma_fs * C_ANG_PER_FS
        root = child_stream_root(seed, 4, 3)
        if long_shape == "gaussian":
            dt = sigma_ang * counter_normals(root, Ne, 1)[:, 0]
        elif long_shape == "uniform":
            half_width = np.sqrt(3.0) * sigma_ang  # flat-top of the same RMS
            dt = half_width * (2.0 * counter_uniforms(root, Ne, 1)[:, 0] - 1.0)
        else:
            raise ValueError(
                f"long_shape must be 'gaussian' or 'uniform' (or supply "
                f"long_offsets_fs), got {long_shape!r}"
            )
    else:
        return np.zeros(Ne)
    return dt - dt.mean()  # center on the bunch centroid (t=0 == centroid)


def _beta_array(E_keV):
    g = 1.0 + E_keV / 510.99895
    return (1.0 - 1.0 / (g * g)) ** 0.5


beta_from_keV = _beta_array


@njit(cache=True)
def beta_from_keV_scalar(E_i):
    g = 1.0 + E_i / 510.99895
    g_inv_square = 1.0 / (g * g)
    return (1.0 - g_inv_square) ** 0.5


# ---- counter-based per-electron RNG -------------------------------------------
# The lockstep core draws every random number from one shared `Generator`, so its
# stream order is "step-major, electron-minor" and cannot be reproduced by
# threads that run electrons to completion independently. The per-electron core
# below and its CUDA port instead address randomness by (seed, electron, draw
# index) through a SplitMix64 counter hash. Nothing is carried between draws, so
# a stream is replayable: re-running a batch after a capacity overflow, changing
# the batch size, or changing the CUDA launch geometry all give identical
# numbers. That is what lets the GPU kernel be bit-for-bit against a CPU
# reference rather than merely statistically similar.
#
# SplitMix64's finalizer is a bijection on 64 bits and passes BigCrush as a
# counter-mode generator (Steele, Lea & Flood, OOPSLA 2014). Streams are keyed
# through the same finalizer so adjacent electron ids do not produce correlated
# sequences.

_SM64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SM64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SM64_MIX2 = np.uint64(0x94D049BB133111EB)
_SM64_S27 = np.uint64(27)
_SM64_S30 = np.uint64(30)
_SM64_S31 = np.uint64(31)
_SM64_S11 = np.uint64(11)
_SM64_ZERO = np.uint64(0)
_SM64_ONE = np.uint64(1)
# 2**-53: the same 53-bit mantissa scaling numpy's `random()` uses, so draws land
# in [0, 1) with uniform spacing.
_U53_SCALE = 1.0 / 9007199254740992.0


# Signatures are explicit because numba otherwise unifies these expressions to
# int64, which turns every `>>` into an arithmetic shift and silently biases the
# generator (draws lose their top bit).
@njit(uint64(uint64), cache=True)
def _splitmix64(x):
    """SplitMix64 finalizer, used here as a counter-based hash."""
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


@njit(uint64(int64, int64), cache=True)
def _stream_key_scalar(seed, elec_id):
    """Per-electron stream key. Independent of batch size and launch geometry."""
    return _splitmix64(np.uint64(seed) + _SM64_GOLDEN * (np.uint64(elec_id) + _SM64_ONE))


@njit(float64(uint64, uint64), cache=True)
def _stream_uniform_scalar(key, counter):
    """Draw ``counter`` of the stream identified by ``key``, in [0, 1)."""
    z = _splitmix64(key + _SM64_GOLDEN * (counter + _SM64_ONE))
    return np.float64(z >> _SM64_S11) * _U53_SCALE


def stream_keys(seed, Ne, *, start=0):
    """Per-electron stream keys for electrons ``[start, start + Ne)``.

    The key of electron ``e`` is a pure function of ``(seed, e)``, so the keys
    for ``[start, stop)`` equal ``stream_keys(seed, stop)[start:]`` exactly: a
    block of electrons addresses the same streams as the matching slice of one
    larger run. Keys are built on the host so the CUDA kernel needs no 64-bit
    integer casts in device code, and so both cores provably address the same
    streams.
    """
    start = int(start)
    if start < 0 or Ne < 0:
        raise ValueError("stream_keys needs a non-negative start and count")
    e = np.arange(start, start + Ne, dtype=np.uint64)
    x = np.uint64(seed) + _SM64_GOLDEN * (e + _SM64_ONE)
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


# 2**-52: counter uniforms keep 52 bits so ``(m + 0.5) * 2**-52`` is exact and
# lies strictly inside (0, 1), which the normal quantile below requires.
_U52_SCALE = 1.0 / 4503599627370496.0
_SM64_S12 = np.uint64(12)


def child_stream_root(seed, n_children, index):
    """Root key of the ``SeedSequence(seed).spawn(n_children)[index]`` namespace.

    Host-side beam and bunch inputs keep their historical child-stream
    assignment (see ``docs/computation/random-streams.md``); the child's first
    64-bit state word roots a counter-addressed per-electron stream inside it,
    so the namespaces stay disjoint from transport and from each other.
    """
    child = np.random.SeedSequence(seed).spawn(n_children)[index]
    return int(child.generate_state(1, np.uint64)[0])


def counter_uniforms(root, n_electrons, n_draws, *, start=0):
    """Open-interval uniforms ``u[e - start, c]`` for electrons ``[start, start+n)``.

    Draw ``c`` of electron ``e`` is
    ``((splitmix64(k_e + PHI*(c+1)) >> 12) + 1/2) * 2**-52`` with
    ``k_e = stream_keys(root, ...)[e]`` -- the transport counter construction
    with a 52-bit mantissa so the half-offset is exact. It depends only on
    ``(root, e, c)``: never on the electron count, the block, or other draws.
    """
    keys = stream_keys(root, n_electrons, start=start)[:, None]
    counters = np.arange(n_draws, dtype=np.uint64)[None, :]
    x = keys + _SM64_GOLDEN * (counters + _SM64_ONE)
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    x = x ^ (x >> _SM64_S31)
    return ((x >> _SM64_S12).astype(np.float64) + 0.5) * _U52_SCALE


def counter_normals(root, n_electrons, n_draws, *, start=0):
    """Standard normals by inverse CDF of :func:`counter_uniforms`.

    One uniform per normal keeps the address one-to-one: draw ``c`` of
    electron ``e`` is ``Phi^-1(u(e, c))``. The tails are truncated at
    ``|z| <= 8.2095``, the quantile of the smallest representable ``u``.

    Validation: beam-phase-space-injection, longitudinal-bunch-sampling
    """
    return ndtri(counter_uniforms(root, n_electrons, n_draws, start=start))
