"""Fused segment-block prologue of the batched CXR line route (#862).

Each helper is one launch on the CuPy float32 path and bit-for-bit the CuPy
chain it replaces (:mod:`._fused`); off that path it runs the chain.
"""

from . import _fused
from ._recoil import (
    _line_recoil_terms,
    _recoil_inverse_energy,
    _recoil_line_allowed,
    _recoil_slope_factor,
)


def block_recoil_pair(st, bt, sb, vx, vy, vz, gamma):
    """``(r_num, r_den, inv_E, denom_vac, v_dot_g)`` ahead of the in-medium root.

    One fused launch on the CuPy float32 path (bit-for-bit the chain below,
    #862); otherwise the chain: the exact-recoil terms of
    :func:`_line_recoil_terms`, the shifted vacuum denominator and ``v.g - r_num``.
    """
    inv_E = _recoil_inverse_energy(gamma)
    denom_row = st.denom_all[sb][:, None]
    fused = _fused.recoil_pair_terms(
        vx, vy, vz, bt.gx, bt.gy, bt.gz, bt.g2, bt.n_dot_g, denom_row, inv_E
    )
    if fused is not None:
        r_num, r_den, denom_vac, v_dot_g = fused
        return r_num, r_den, inv_E, denom_vac, v_dot_g
    r_num, r_den, inv_E = _line_recoil_terms(gamma, bt.g2, bt.n_dot_g)
    v_dot_g = (vx * bt.gx + vy * bt.gy + vz * bt.gz) - r_num
    return r_num, r_den, inv_E, denom_row + r_den, v_dot_g


def block_n_dot_g(bt, grad, delta):
    """``n.g`` shifted by the refractive term ``delta (grad L . g)`` over the block."""
    g0, g1, g2 = grad[:, 0, None], grad[:, 1, None], grad[:, 2, None]
    fused = _fused.block_n_dot_g(g0, g1, g2, bt.gx, bt.gy, bt.gz, delta, bt.n_dot_g)
    if fused is not None:
        return fused
    grad_dot_g = g0 * bt.gx + g1 * bt.gy + g2 * bt.gz
    return bt.n_dot_g + delta * grad_dot_g


def block_slope_keep(denom, omega_res, inv_E, gamma, keep):
    """``(slope, keep)``: the sinc slope ``denom kappa`` and ``keep`` with the
    positive-mass-shell guard of :func:`_recoil_line_allowed` applied."""
    fused = _fused.recoil_slope_keep(denom, omega_res, inv_E, gamma, keep)
    if fused is not None:
        return fused
    slope = denom * _recoil_slope_factor(omega_res, inv_E)
    return slope, keep & _recoil_line_allowed(denom, omega_res, inv_E, gamma)
