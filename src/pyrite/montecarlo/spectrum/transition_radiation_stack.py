"""O(N) solve of the planar transition-radiation stack by polarization (#962).

Same boundary problem as the dense system of
:mod:`~pyrite.montecarlo.spectrum.transition_radiation`, for stacks with many
layers. In a flat isotropic stack ``k_perp`` is conserved and TE and TM
decouple. With ``k_hat = k_perp / |k_perp|`` (``x`` at normal incidence) and
``s_hat = z x k_hat``, the free waves of medium ``j`` have tangential fields
``E = a + b``, ``H = Y_j (a - b)``, up-going ``a`` and down-going ``b``:

    TE:  E = E . s_hat,  H = H . k_hat,  Y = -lambda / omega,
    TM:  E = E . k_hat,  H = H . s_hat,  Y = eps omega / lambda.

Interface ``i`` joins media ``i`` and ``i + 1``. A jump ``(s_E, s_H)``, field
above minus field below, is the source. Let ``rho_i`` be the ratio of up- to
down-going waves at the top of medium ``i`` from everything below it, and
``A_i`` the up-going amplitude there from the sources below (both zero in the
substrate). Eliminating the down-going wave of medium ``i`` from the two
continuity equations gives ``a' = Gamma_i b' + u_i`` above interface ``i``, with

    D_i     = (1 + rho_i) Y_{i+1} + (1 - rho_i) Y_i,
    Gamma_i = [(1 + rho_i) Y_{i+1} - (1 - rho_i) Y_i] / D_i,
    u_i     = tau_i A_i + alpha_i s_E,i + beta_i s_H,i,
    tau_i   = 2 Y_i / D_i,
    alpha_i = (1 - rho_i) Y_i / D_i,
    beta_i  = (1 + rho_i) / D_i,

and ``rho_{i+1} = p^2 Gamma_i``, ``A_{i+1} = p u_i`` for the layer phase
``p = exp(i lambda_{i+1} d_{i+1})``, ``|p| <= 1``. The vacuum above has no
down-going wave, so its amplitude is ``u`` of the last interface. Thick absorbing
or evanescent layers cannot overflow, because their phases are bounded; ``rho``,
``tau`` and ``1 / D`` are not bounded when ``Y`` is not real.
Unrolling the recursion gives the response to a jump at interface ``i``,

    a_top = sum_i C_i (alpha_i s_E,i + beta_i s_H,i),
    C_last = 1,  C_i = C_{i+1} tau_{i+1} p_{i+1},

which is the transposed solve of ``interface_response``, one interface at a
time. Both forms share two singularities with the dense system: ``lambda = 0`` in
a layer, and the Cherenkov pole of an own field in a lossless medium. The
recursion adds a removable pole at ``D_i = 0``, a bound mode of the lower
half-stack where the dense matrix is regular. It also rounds ``1 + rho`` where
``rho -> -1``. That rounding reaches the top through ``C_i / D_i``, so it is
estimated relative to the pair's terms: ``u sum_i |C_i / D_i| / max_i |C_i beta_i|``
for the response, with ``u`` the unit roundoff, and the same rounding carried up
beside the amplitude and divided by the running sum of term sizes. Non-finite
results and estimates above ``_MAX_CANCELLATION_ERROR`` raise
:class:`numpy.linalg.LinAlgError`, for the whole call.

The estimate covers only that rounding. Conditioning near lossless resonances and
phase rounding ``u sum |lambda d|`` are not estimated and can be larger, up to full
scale at a resonance, where the dense solve is also unreliable. Cancellation
between terms can leave an amplitude less accurate relative to itself than the
estimate. Deep responses can lose pointwise accuracy in lossless stacks; absorbing
stacks keep it.

Validation: optical-stack-fast-solver
"""

from __future__ import annotations

import numpy as np

from .transition_radiation import _outgoing_root, _particular_field

__all__ = ["stack_amplitude", "stack_response", "straight_charge_amplitude"]

_USE_JIT_STACK = True
#: Largest accepted rounding of ``1 + rho`` at the top, relative to a pair's largest term (#962).
_MAX_CANCELLATION_ERROR = 1e-8


def _fused_kernel(xp) -> bool:
    """CUDA CuPy (not ROCm/HIP) with the fused kernels enabled."""
    if not _USE_JIT_STACK or getattr(xp, "__name__", "") != "cupy":
        return False
    return not getattr(xp.cuda.runtime, "is_hip", False)


def _modes(omega, k_perp, media, xp):
    """``k_hat`` and ``s_hat`` ``(B, 2)``, ``lambda`` per medium, and ``Y`` with (TE, TM) last."""
    k2 = xp.sum(k_perp**2, axis=-1)
    k = xp.hypot(k_perp[:, 0], k_perp[:, 1])
    safe = xp.where(k > 0.0, k, 1.0)
    k_hat = xp.stack([xp.where(k > 0.0, k_perp[:, 0] / safe, 1.0), k_perp[:, 1] / safe], axis=-1)
    s_hat = xp.stack([-k_hat[:, 1], k_hat[:, 0]], axis=-1)
    lam = _outgoing_root(media * omega**2 - k2, xp)
    admittance = xp.stack([-lam / omega, media * omega / lam], axis=-1)
    return k_hat, s_hat, lam, admittance


def _require_resolved(array, reach, xp):
    """Refuse a non-finite result, or a pair whose ``1 + rho`` rounding reaches the top."""
    if not bool(xp.all(xp.isfinite(array))):
        raise np.linalg.LinAlgError(
            "stack solve: singular pair (recursion pole D = 0, lambda = 0 or a lossless "
            "own-field pole); detune the energy or direction"
        )
    if bool(xp.any(reach * np.finfo(float).eps / 2 > _MAX_CANCELLATION_ERROR)):
        raise np.linalg.LinAlgError(
            f"stack solve: estimated cancellation error exceeds {_MAX_CANCELLATION_ERROR:g}; "
            "1 + rho is not resolved along this stack"
        )


def _coefficients(lam, admittance, z_if, xp):
    """Yield ``(alpha, beta, tau, p, 1 / |D|)`` per interface, bottom up, each ``(B, 2)``.

    ``p`` is the phase of the medium above the interface, 1 for the vacuum.
    """
    n_if = lam.shape[0] - 1
    thickness = xp.asarray(np.diff(np.asarray(z_if)))
    rho = xp.zeros_like(admittance[0])
    for i in range(n_if):
        below, above = admittance[i], admittance[i + 1]
        plus, minus = 1.0 + rho, 1.0 - rho
        denominator = plus * above + minus * below
        alpha = minus * below / denominator
        beta = plus / denominator
        tau = 2.0 * below / denominator
        phase = xp.ones_like(rho)
        if i < n_if - 1:
            phase = xp.exp(1j * lam[i + 1] * thickness[i])[:, None]
            rho = phase**2 * (plus * above - minus * below) / denominator
        yield alpha, beta, tau, phase, 1.0 / xp.abs(denominator)


def stack_response(omega, k_perp, media, z_if, *, xp=np):
    """Top-wave response ``(B, n_if, 4, 2)`` and vacuum ``lambda`` of a stack.

    ``media`` is ``(n_media, B)``, substrate first and vacuum last, and ``z_if``
    the interface heights in 1/eV. The response has the layout of
    :class:`~pyrite.montecarlo.spectrum.transition_radiation.InterfaceResponse`.

    Validation: optical-stack-fast-solver
    """
    if _fused_kernel(xp):
        from .transition_radiation_stack_kernel import device_stack_response

        response, lam_top, reach = device_stack_response(omega, k_perp, media, z_if)
        _require_resolved(response, reach, xp)
        return response, lam_top
    k_hat, s_hat, lam, admittance = _modes(omega, k_perp, media, xp)
    terms = list(_coefficients(lam, admittance, z_if, xp))
    ss = s_hat[:, None, :] * s_hat[:, :, None]
    kk = k_hat[:, None, :] * k_hat[:, :, None]
    source_k_out_s = k_hat[:, :, None] * s_hat[:, None, :]
    source_s_out_k = s_hat[:, :, None] * k_hat[:, None, :]
    response = xp.zeros((omega.size, len(terms), 4, 2), dtype=complex)
    weight = xp.ones_like(admittance[0])
    reach = xp.zeros_like(weight.real)
    peak = xp.zeros_like(weight.real)
    for i in range(len(terms) - 1, -1, -1):
        alpha, beta, tau, _, inverse_d = terms[i]
        reach = reach + xp.abs(weight) * inverse_d
        peak = xp.maximum(peak, xp.abs(weight * beta))
        g_e, g_h = (weight * alpha)[:, None, None, :], (weight * beta)[:, None, None, :]
        response[:, i, :2] = g_e[..., 0] * ss + g_e[..., 1] * kk
        response[:, i, 2:] = g_h[..., 0] * source_k_out_s + g_h[..., 1] * source_s_out_k
        if i > 0:
            weight = weight * tau * terms[i - 1][3]
    _require_resolved(response, xp.max(reach / peak, axis=-1), xp)
    return response, lam[-1]


def stack_amplitude(omega, k_perp, media, z_if, sources, *, xp=np):
    """Top-wave tangential ``E`` ``(B, 2)`` and vacuum ``lambda`` for jumps ``sources``.

    ``sources`` iterates over the interfaces bottom up, giving ``(B, 4)`` arrays
    ``(E_x, E_y, H_x, H_y)`` of the jump, field above minus field below.

    Validation: optical-stack-fast-solver
    """
    k_hat, s_hat, lam, admittance = _modes(omega, k_perp, media, xp)
    up = xp.zeros_like(admittance[0])
    reach = xp.zeros_like(up.real)
    size = xp.zeros_like(up.real)
    terms = _coefficients(lam, admittance, z_if, xp)
    for (alpha, beta, tau, phase, inverse_d), source in zip(terms, sources, strict=True):
        e_t, h_t = source[:, :2], source[:, 2:]
        s_e = xp.stack([xp.sum(s_hat * e_t, axis=-1), xp.sum(k_hat * e_t, axis=-1)], axis=-1)
        s_h = xp.stack([xp.sum(k_hat * h_t, axis=-1), xp.sum(s_hat * h_t, axis=-1)], axis=-1)
        up = phase * (tau * up + alpha * s_e + beta * s_h)
        decay = xp.abs(phase)
        reach = decay * (xp.abs(tau) * reach + inverse_d * xp.abs(s_h))
        size = decay * (xp.abs(tau) * size + xp.abs(alpha * s_e) + xp.abs(beta * s_h))
    e_t = up[:, 0, None] * s_hat + up[:, 1, None] * k_hat
    ratio = xp.where(size > 0.0, reach / xp.where(size > 0.0, size, 1.0), 0.0)
    _require_resolved(e_t, xp.max(ratio, axis=-1), xp)
    return e_t, lam[-1]


def _straight_charge_sources(omega, k_perp, v, q, media, z_if, xp):
    below = _particular_field(media[0], omega, k_perp, q, v, xp)
    for i, z0 in enumerate(z_if):
        above = _particular_field(media[i + 1], omega, k_perp, q, v, xp)
        jump = xp.concatenate(
            [below[0][:, :2] - above[0][:, :2], below[1][:, :2] - above[1][:, :2]], axis=-1
        )
        yield jump * xp.exp(1j * q * float(z0))[:, None]
        below = above


def straight_charge_amplitude(omega, k_perp, v, media, z_if, *, xp=np):
    """Top-wave tangential ``E`` ``(B, 2)`` and vacuum ``lambda`` of a straight charge.

    Validation: optical-stack-fast-solver
    """
    if _fused_kernel(xp):
        from .transition_radiation_stack_kernel import device_stack_amplitude

        e_t, lam_top, reach = device_stack_amplitude(omega, k_perp, v, media, z_if)
        _require_resolved(e_t, reach, xp)
        return e_t, lam_top
    q = (omega - xp.sum(k_perp * v[:, :2], axis=-1)) / v[:, 2]
    sources = _straight_charge_sources(omega, k_perp, v, q, media, z_if, xp)
    return stack_amplitude(omega, k_perp, media, z_if, sources, xp=xp)
