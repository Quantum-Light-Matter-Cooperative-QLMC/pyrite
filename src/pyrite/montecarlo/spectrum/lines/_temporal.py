"""Opt-in temporal intensity profile ``I(t) = |E(t)|^2`` of the CXR lines.

The line kernel builds each segment's spectral field

    E_j(E) = c_j F_j(E) exp{i[omega(E) d_j - g.r_j - delta(E) omega(E) L_esc,j]},

``c_j`` the amplitude frozen at the segment's resonance, ``F_j`` its finite-time
(formation) factor and ``d_j = t_abs,j - n_hat.r_j`` its retarded arrival time
(Ang, ``c = 1``). With the unitary transform

    E(t) = (2 pi hbar c)^{-1/2} int E(E) exp(-i E t / hbar c) dE,

``int |E(t)|^2 dt = int |E(E)|^2 dE`` (Parseval), so the profile carries the same
photons per sr per incident electron as the spectrum, now per unit arrival time.
Polarizations, reflections and mosaic orientations stay incoherent, exactly as
in the spectrum. The output is upstream of any detector response.

Two routes, matching the spectrum's two emission policies:

``incoherent``
    The random-phase observable is ``sum_j |E_j(t)|^2``. The inverse transform
    of ``t_L sinc(a (E - E_r))`` with a frozen amplitude is a rectangular pulse
    of duration ``2 hbar c a`` (``= D t_L``, the Doppler-compressed flight time)
    centred on ``d_j``, so each line adds a box of its whole spectral mass
    ``w pi / a``. Numerical substeps of one flight are contiguous in arrival
    time, so their boxes tile the flight's box and squaring per row equals
    squaring per flight. Absorption along the row enters as its mean
    transmission (flat box); dispersive group delay ``~ delta L_esc`` is
    neglected.

``coherent``
    Each (reflection, orientation, polarization) row's field is rebuilt on a
    uniform energy grid spanning the line axis, and one FFT gives ``E(t)`` on
    the matching time grid, band-limited to that axis. With bunch offsets the
    ensemble average is taken in the time domain: per-electron self terms with
    each electron's realized arrival offset, plus the cross-electron term
    filtered by the offsets' characteristic function ``chi(omega)``,

        I = sum_e |S_e(t)|^2 + |IFT[chi sum_e S_e^geo]|^2 - sum_e |IFT[chi S_e^geo]|^2,

    whose Parseval image is the spectrum's ``(1-F) Grouped + F Flat`` blend,
    ``F = |chi|^2``.

The time grid is shared by every call of one case (directions, layers, both
policies) and is periodic with period ``2 hbar c pi / dE``, chosen at twice the
span of arrival times so band-limited ringing does not wrap onto the signal.

Validation: temporal-intensity-profile
"""

from dataclasses import dataclass, field

import numpy as np

from ...._backend import _to_cpu, xp
from ....materials.crystal import HBARC_EV_ANG
from ...transport import beta_from_keV
from ...transport.kinematics import C_ANG_PER_FS

#: Upper bound on the FFT length (time samples). One coherent row materializes
#: ``block x n`` complex fields, so the bound keeps opt-in runs from silently
#: allocating unbounded memory on a very wide line axis or a very long bunch.
TEMPORAL_MAX_SAMPLES = 1 << 20
#: Complex elements per materialized coherent field block (~64 MiB at fp64).
_FIELD_BLOCK_ELEMENTS = 1 << 22
#: Period over arrival-time span: zero padding against wrap-around ringing.
_PERIOD_PADDING = 2.0


@dataclass(eq=False)
class TemporalProfile:
    """Shared time grid and accumulator for one case's ``I(t)``.

    ``t_ang[k] = t_start_ang + k dt_ang`` for ``k < n``; the conjugate energy
    grid is ``E_start_eV + m dE_eV``. ``intensity`` accumulates photons per sr
    per incident electron per Ang (``c = 1``); :meth:`result` reports per fs.
    """

    t_start_ang: float
    dt_ang: float
    E_start_eV: float
    dE_eV: float
    n: int
    intensity: object = field(repr=False, default=None)

    def __post_init__(self):
        if self.intensity is None:
            self.intensity = xp.zeros(self.n, dtype=np.float64)

    @property
    def period_ang(self):
        return self.n * self.dt_ang

    def energy_grid(self):
        return self.E_start_eV + self.dE_eV * xp.arange(self.n, dtype=np.float64)

    def time_edges_ang(self):
        """Bin edges centred on the samples, for box integration."""
        return self.t_start_ang + self.dt_ang * (np.arange(self.n + 1) - 0.5)

    def empty_like(self):
        """A fresh accumulator on the same grid."""
        return TemporalProfile(
            t_start_ang=self.t_start_ang,
            dt_ang=self.dt_ang,
            E_start_eV=self.E_start_eV,
            dE_eV=self.dE_eV,
            n=self.n,
        )

    def buffer(self):
        return xp.zeros(self.n, dtype=np.float64)

    def commit(self, buf, scale):
        """Add one spectrum call's buffer, normalized per incident electron."""
        self.intensity += buf * float(scale)

    def result(self):
        """``{"t_fs": (n,), "intensity": (n,)}``, intensity in photons/sr/electron/fs."""
        t = (self.t_start_ang + self.dt_ang * np.arange(self.n)) / C_ANG_PER_FS
        return {
            "t_fs": t,
            "intensity": np.asarray(_to_cpu(self.intensity), dtype=float) * C_ANG_PER_FS,
        }


def segment_arrival_times(segments, n_hat, *, xp=xp):
    """Float64 ``(d, t_L)``: retarded midpoint arrival ``t_abs - n_hat.r`` and flight time.

    Matches the coherent route's ``d_all`` (start age plus half the
    constant-velocity flight, plus the bunch offset ``t0``) in float64.
    """
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    E = xp.asarray(segments[E_field], dtype=np.float64)
    n_rows = int(E.shape[0])
    beta = beta_from_keV(E)
    t_L = xp.asarray(segments["L_ang"], dtype=np.float64) / beta
    zeros = np.zeros(n_rows)
    t_start = xp.asarray(
        segments.get("t_ang") if segments.get("t_ang") is not None else zeros, dtype=np.float64
    )
    t0 = xp.asarray(
        segments.get("t0_ang") if segments.get("t0_ang") is not None else zeros, dtype=np.float64
    )
    r = xp.asarray(segments["r_mid"], dtype=np.float64)
    n = xp.asarray(np.asarray(n_hat, dtype=np.float64))
    d = t_start + 0.5 * t_L + t0 - (r[:, 0] * n[0] + r[:, 1] * n[1] + r[:, 2] * n[2])
    return d, t_L


def temporal_profile_for(segments, E_grid_eV, n_hats, *, max_samples=TEMPORAL_MAX_SAMPLES):
    """Build the shared :class:`TemporalProfile` for one case.

    The arrival window covers every segment's pulse ``d +- t_L`` (``D <= 2``
    bounds the pulse half-duration by ``t_L``) over all observation
    directions ``n_hats``; the energy grid spans the line axis.

    Raises
    ------
    ValueError
        If the grid would exceed ``max_samples`` time samples.
    """
    E_grid = np.asarray(E_grid_eV, dtype=float)
    span_E = float(E_grid[-1] - E_grid[0]) if E_grid.size > 1 else 0.0
    if span_E <= 0.0:
        raise ValueError("temporal profile needs a line axis with nonzero energy span")
    lo, hi = np.inf, -np.inf
    for n_hat in np.atleast_2d(np.asarray(n_hats, dtype=float)):
        n_hat = n_hat / np.linalg.norm(n_hat)
        if int(np.asarray(segments["L_ang"]).shape[0]) == 0:
            continue
        d, t_L = segment_arrival_times(segments, n_hat, xp=np)
        lo = min(lo, float(np.min(d - t_L)))
        hi = max(hi, float(np.max(d + t_L)))
    if not np.isfinite(lo):
        lo, hi = 0.0, 1.0
    span_t = max(hi - lo, 1.0)
    # Period P = 2 pi hbar c / dE >= padding * span; n samples cover the axis.
    dE = 2.0 * np.pi * HBARC_EV_ANG / (_PERIOD_PADDING * span_t)
    n = int(np.ceil(span_E / dE)) + 1
    if n > max_samples:
        raise ValueError(
            f"temporal profile needs {n} time samples (line axis {span_E:.4g} eV x "
            f"arrival span {span_t / C_ANG_PER_FS:.4g} fs), above the "
            f"{max_samples} cap; narrow the line axis or shorten the bunch"
        )
    dt = 2.0 * np.pi * HBARC_EV_ANG / (n * dE)
    # Centre the signal in the period so ringing on both sides stays unwrapped.
    t_start = 0.5 * (lo + hi) - 0.5 * n * dt
    return TemporalProfile(
        t_start_ang=float(t_start),
        dt_ang=float(dt),
        E_start_eV=float(E_grid[0]),
        dE_eV=float(dE),
        n=n,
    )


def case_temporal_profiles(case, tp, want_coherent):
    """Runner accumulators ``(incoherent, coherent_or_None)`` for one case.

    ``(None, None)`` unless the case opts in. A directional run passes its
    joint ``tp["temporal_grid"]`` so every direction shares one time grid.
    """
    if not case.get("temporal_profile"):
        return None, None
    grid = tp.get("temporal_grid") or temporal_profile_for(tp["segs"], tp["E_grid"], [tp["n_hat"]])
    return grid.empty_like(), (grid.empty_like() if want_coherent else None)


def temporal_outputs(temporal, temporal_coherent):
    """Runner output keys for finished accumulators (empty when off)."""
    if temporal is None:
        return {}
    profile = temporal.result()
    out = {"temporal_t_fs": profile["t_fs"], "temporal_intensity": profile["intensity"]}
    if temporal_coherent is not None:
        out["temporal_intensity_coherent"] = temporal_coherent.result()["intensity"]
    return out


def add_boxes(profile, buf, tau, half_width, mass):
    """Add rectangular pulses (centre ``tau``, half-width, mass) into ``buf``.

    Each pulse's mass is split exactly across the bins it overlaps; mass
    outside the periodic window is dropped. All inputs are ``(n_line,)``
    arrays on the backend; ``buf`` holds mass per Ang after division by
    ``dt``.
    """
    if tau.size == 0:
        return
    n = profile.n
    dt = profile.dt_ang
    e0 = profile.t_start_ang - 0.5 * dt
    tau = tau.astype(np.float64)
    hw = xp.maximum(half_width.astype(np.float64), 0.0)
    m = mass.astype(np.float64)
    a = xp.clip((tau - hw - e0) / dt, 0.0, float(n))
    b = xp.clip((tau + hw - e0) / dt, 0.0, float(n))
    length = b - a
    finite = xp.isfinite(m) & (m > 0.0)
    # Zero-width pulses deposit their whole mass in the bin holding tau.
    centre = (tau - e0) / dt
    point = finite & ~(hw > 0.0) & (centre >= 0.0) & (centre < float(n))
    live = (finite & (hw > 0.0) & (length > 0.0)) | point
    h = xp.where(live, m / xp.where(hw > 0.0, 2.0 * hw / dt, 1.0), 0.0)  # mass per bin unit
    ia = xp.minimum(xp.floor(a).astype(np.int64), n - 1)
    ib = xp.minimum(xp.floor(b).astype(np.int64), n - 1)
    same = live & ~point & (ia == ib)
    split = live & ~point & (ia != ib)
    acc = xp.zeros(n, dtype=np.float64)
    acc += xp.bincount(ia, weights=xp.where(point, m, 0.0), minlength=n)
    acc += xp.bincount(ia, weights=xp.where(same, h * length, 0.0), minlength=n)
    acc += xp.bincount(ia, weights=xp.where(split, h * (ia + 1 - a), 0.0), minlength=n)
    acc += xp.bincount(ib, weights=xp.where(split, h * (b - ib), 0.0), minlength=n)
    # Interior bins ia+1 .. ib-1 each take h (one full bin): a difference array.
    diff = xp.zeros(n + 1, dtype=np.float64)
    diff += xp.bincount(ia + 1, weights=xp.where(split, h, 0.0), minlength=n + 1)
    diff -= xp.bincount(ib, weights=xp.where(split, h, 0.0), minlength=n + 1)
    acc += xp.cumsum(diff)[:n]
    buf += acc / dt


def formation_mean_transmission(apb, bma, q):
    """Mean ``exp(-tau)`` along a linear escape piece from formation coefficients.

    ``(a^2 - b^2)/(tau_end - tau_start) = -apb bma / (4 q)``, which tends to
    ``(apb/2)^2`` as ``q -> 0``. Validation: temporal-intensity-profile
    """
    q = q.astype(np.float64)
    apb = apb.astype(np.float64)
    bma = bma.astype(np.float64)
    zero = q == 0.0
    return xp.where(zero, 0.25 * apb * apb, -apb * bma / (4.0 * xp.where(zero, 1.0, q)))


def _field_ifft_power(profile, fields):
    """``sum over leading rows of |E(t)|^2`` for fields on the profile's energy grid.

    ``fields`` is ``(rows, n)`` complex, phased relative to ``t_start``; the
    numpy FFT kernel ``exp(-2 pi i m k / n)`` is ``exp(-i omega_m k dt)`` up to
    a per-sample global phase. Returns ``(n,)`` mass per Ang.
    """
    spectrum = xp.fft.fft(fields, axis=-1)
    power = spectrum.real**2 + spectrum.imag**2
    scale = profile.dE_eV**2 / (2.0 * np.pi * HBARC_EV_ANG)
    return power.sum(axis=0) * scale


@dataclass(eq=False)
class CoherentRow:
    """One coherent row's per-segment inputs, already restricted to live lines.

    ``coefs`` per polarization; ``d``/``g_phase`` realized; ``d_geo``/
    ``g_phase_geo`` offset-free (equal to the realized ones without offsets);
    ``lines`` the ``_FormationLines`` tuple; ``L_esc`` the midpoint escape.
    """

    coefs: list
    d: object
    g_phase: object
    d_geo: object
    g_phase_geo: object
    lines: tuple
    L_esc: object
    electron: object


def _row_fields(profile, row, sel, omega, delta_omega, *, geometric):
    """Per-segment phased formation fields ``(len(sel), n)`` for each polarization."""
    from ._formation import formation_profile

    E_vac, a_vac, half_dL, apb, bma, q = (xp.asarray(a, dtype=np.float64)[sel] for a in row.lines)
    E_t = profile.energy_grid()
    F = formation_profile(
        E_t, delta_omega, E_vac, a_vac, half_dL, apb, bma, q, sinc_cutoff=None, xp=xp
    )
    d = (row.d_geo if geometric else row.d)[sel].astype(np.float64) - profile.t_start_ang
    gp = (row.g_phase_geo if geometric else row.g_phase)[sel].astype(np.float64)
    L = row.L_esc[sel].astype(np.float64)
    arg = d[:, None] * omega[None, :] - gp[:, None] - L[:, None] * delta_omega[None, :]
    SP = F * xp.exp(1j * arg)
    return [xp.asarray(c, dtype=np.complex128)[sel][:, None] * SP for c in row.coefs]


def _block_rows(n):
    return max(1, _FIELD_BLOCK_ELEMENTS // max(n, 1))


def add_coherent_row(profile, buf, row, wm, *, delta_omega, chi=None):
    """Add one coherent row's ``I(t)`` (times mosaic weight ``wm``) into ``buf``.

    ``chi`` is ``None`` without bunch offsets (one fully coherent sum), else
    the offsets' complex characteristic function on the profile's energy grid.
    """
    n = profile.n
    omega = profile.energy_grid() / HBARC_EV_ANG
    n_seg = int(row.d.shape[0])
    if n_seg == 0:
        return
    block = _block_rows(n)
    if chi is None:
        totals = [xp.zeros(n, dtype=np.complex128) for _ in row.coefs]
        for j0 in range(0, n_seg, block):
            sel = xp.arange(j0, min(j0 + block, n_seg))
            for tot, f in zip(
                totals,
                _row_fields(profile, row, sel, omega, delta_omega, geometric=False),
                strict=True,
            ):
                tot += f.sum(axis=0)
        buf += _field_ifft_power(profile, xp.stack(totals)) * wm
        return

    # Bunch offsets: per-electron fields, then the time-domain blend.
    gid = np.asarray(_to_cpu(row.electron))
    perm = np.argsort(gid, kind="stable")
    gid = gid[perm]
    starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
    bounds = np.append(starts, gid.size)
    perm_xp = xp.asarray(perm)
    flat_geo = [xp.zeros(n, dtype=np.complex128) for _ in row.coefs]
    self_term = xp.zeros(n, dtype=np.float64)
    cross_self = xp.zeros(n, dtype=np.float64)
    k0 = 0
    n_groups = bounds.size - 1
    while k0 < n_groups:
        # Whole electrons per block, at least one.
        k1 = k0 + 1
        while k1 < n_groups and bounds[k1 + 1] - bounds[k0] <= block:
            k1 += 1
        rows = perm_xp[int(bounds[k0]) : int(bounds[k1])]
        offsets = bounds[k0:k1] - bounds[k0]
        geo = _row_fields(profile, row, rows, omega, delta_omega, geometric=True)
        # The realized field of each electron differs from its geometric one
        # by one constant offset phase: read it from the electron's first row.
        first = rows[xp.asarray(offsets)]
        dA = (row.d[first].astype(np.float64) - row.d_geo[first].astype(np.float64))[:, None]
        dB = (row.g_phase[first].astype(np.float64) - row.g_phase_geo[first].astype(np.float64))[
            :, None
        ]
        shift = xp.exp(1j * (dA * omega[None, :] - dB))
        for f, flat in zip(geo, flat_geo, strict=True):
            per_e = xp.add.reduceat(f, xp.asarray(offsets), axis=0)
            flat += per_e.sum(axis=0)
            self_term += _field_ifft_power(profile, per_e * shift)
            cross_self += _field_ifft_power(profile, per_e * chi[None, :])
        k0 = k1
    cross = _field_ifft_power(profile, xp.stack(flat_geo) * chi[None, :])
    buf += (self_term + cross - cross_self) * wm


def coherent_offset_chi(st, profile, g_vec_d):
    """Complex offset characteristic function on the profile's energy grid.

    Mirrors ``_row_decoherence_factor``: the empirical
    ``mean_e exp[i(omega A_e - B_e)]`` for an infinite slab, the analytic
    Gaussian longitudinal ``exp[-(omega sigma_z)^2 / 2]`` for a finite
    footprint. Returns ``None`` without offsets.
    """
    if not st.decoherence_active:
        return None
    omega = profile.energy_grid() / HBARC_EV_ANG
    if st.finite_footprint_now:
        sigma_z = float(st.request.longitudinal_rms_fs) * C_ANG_PER_FS
        return xp.exp(-0.5 * (omega * sigma_z) ** 2).astype(np.complex128)
    A = xp.asarray(st.decoherence_A_pop, dtype=np.float64)
    B = xp.asarray(st.xy0_pop, dtype=np.float64) @ xp.asarray(g_vec_d, dtype=np.float64)[:2]
    chi = xp.zeros(profile.n, dtype=np.complex128)
    block = _block_rows(profile.n)
    for j0 in range(0, A.size, block):
        sl = slice(j0, min(j0 + block, A.size))
        chi += xp.exp(1j * (A[sl][:, None] * omega[None, :] - B[sl][:, None])).sum(axis=0)
    return chi / max(A.size, 1)


def delta_omega_on_profile(st, profile):
    """``delta(E) omega(E)`` interpolated from the line axis to the profile grid."""
    E_t = profile.energy_grid()
    return xp.interp(
        E_t, xp.asarray(st.E_grid, dtype=np.float64), st.delta_omega_grid.astype(np.float64)
    )


__all__ = [
    "TEMPORAL_MAX_SAMPLES",
    "CoherentRow",
    "TemporalProfile",
    "add_boxes",
    "add_coherent_row",
    "case_temporal_profiles",
    "coherent_offset_chi",
    "delta_omega_on_profile",
    "formation_mean_transmission",
    "segment_arrival_times",
    "temporal_outputs",
    "temporal_profile_for",
]
