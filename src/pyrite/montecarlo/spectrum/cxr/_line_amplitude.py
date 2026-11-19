"""Amplitude-only twin of the fused line-amplitude kernel (#862).

:func:`_cxr_amp_sq_total_core` is :func:`._kernels._cxr_amp_sq_core` without
the PXR/CBS component outputs: the same expressions in the same order, so its
``|A|^2`` equals that kernel's first output bit for bit (checked in
``tests/montecarlo/spectrum/test_cxr_fused_cuda_parity.py``). It adds no
physics; the ledgered claim stays with ``_cxr_amp_sq_core``.
"""

from ...._backend import xp
from ._recoil import _quantum_line_factors


def _cxr_amp_sq_total_core(
    chi_re,
    chi_im,
    u_re,
    u_im,
    g2,
    g_dot_e,
    om,
    v_dot_e,
    vdg,
    k_dot_g,
    k_dot_v,
    gamma,
    detuning,
    k_dot_e,
    h,
):
    """``|A|^2`` of one polarization, without the PXR/CBS components."""
    pxr_num, f_cbs = _quantum_line_factors(
        h, gamma, vdg, k_dot_v, k_dot_g, g2, om * om, v_dot_e, g_dot_e, k_dot_e
    )
    f_pxr = pxr_num / detuning
    re = chi_re * f_pxr + u_re * f_cbs
    im = chi_im * f_pxr + u_im * f_cbs
    return re * re + im * im


if hasattr(xp, "fuse"):
    _cxr_amp_sq_total_core = xp.fuse()(_cxr_amp_sq_total_core)
