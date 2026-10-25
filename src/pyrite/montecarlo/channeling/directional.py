"""Direction-resolved electronic channels for uniaxial crystals.

:class:`DirectionalChannels` replaces the radial scalar response of
:class:`~.operators.ElectronicChannels` by one that depends on the angle
``alpha`` between the momentum transfer ``q = (q_x, q_y, q_z)`` and the optic
axis ``c``. Only the valence band is direction-resolved: each valence
oscillator ``k`` keeps its energy ``W_k`` and takes the strength
``f_k(cos^2 alpha)`` of :class:`~..transport.directional_loss.DirectionalValence`
(uniaxial ``eps(q_hat, w) = eps_perp sin^2 alpha + eps_par cos^2 alpha``).

Frame: ``x`` plane normal, ``y`` in-plane and normal to the beam, ``z`` the
beam, with ``q_z = q_min,k = W_k/(hbar v)`` of each oscillator (the paraxial
longitudinal transfer). ``c_axis`` names the frame axis that carries ``c``.
Shells below the valence band and the close (binary) collisions stay
isotropic, the latter with the sphere-averaged valence strengths. The optical
limit is assumed at every ``q``: ``f_k`` is not momentum dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from ..transport.directional_loss import DirectionalValence
from ..transport.shell_oscillators import MaterialShellOscillators, ShellOscillator
from .operators import ElectronicChannels, electronic_channels

_PHI_NODES, _PHI_WEIGHTS = np.polynomial.legendre.leggauss(16)
_PHI = 0.25 * np.pi * (_PHI_NODES + 1.0)
_PHI_W = 0.25 * np.pi * _PHI_WEIGHTS / (0.5 * np.pi)  # mean over [0, pi/2]


@dataclass(frozen=True)
class DirectionalChannels(ElectronicChannels):
    """:class:`ElectronicChannels` whose first ``K`` oscillators are the directional valence band.

    ``distant`` (radial, used by the full-loss diagonal) is the azimuthal mean
    of :meth:`distant_pq` over the transverse plane; :meth:`weight_pq` is the
    full direction-resolved ``dsigma/d^2q_perp``.

    Validation: cr-valence-directional-loss
    """

    valence: DirectionalValence
    c_axis: str

    def _cos2(self, p, q_y, q2):
        if self.c_axis == "x":
            numerator = p**2
        elif self.c_axis == "y":
            numerator = q_y**2
        else:
            numerator = self.distant_qmin_inv_ang**2
        return numerator / q2

    def distant_pq(self, p, q_y):
        """Distant longitudinal ``dsigma/d^2q_perp`` per electron [angstrom^4] at ``(q_x, q_y)``.

        Validation: cr-valence-directional-loss
        """
        px, qy = np.broadcast_arrays(np.asarray(p, dtype=float), np.asarray(q_y, dtype=float))
        shape = px.shape
        px, qy = px.ravel(), qy.ravel()
        result = np.empty(px.size)
        # Momentum points are independent; retain the full shell sum at each
        # point while bounding the point x oscillator interpolation scratch.
        block_size = max(1, 65536 // self.distant_qmin_inv_ang.size)
        for start in range(0, result.size, block_size):
            block = slice(start, start + block_size)
            result[block] = self._distant_pq_block(px[block], qy[block])
        return result.reshape(shape)[()]

    def _distant_pq_block(self, p, q_y):
        """Direction-resolved distant weight on one bounded momentum block."""
        p = np.asarray(p, dtype=float)[..., None]
        q_y = np.asarray(q_y, dtype=float)[..., None]
        q2 = p**2 + q_y**2 + self.distant_qmin_inv_ang**2
        count = self.valence.energies_eV.size
        ratio = np.ones_like(q2)
        ratio[..., :count] = self.valence.ratio(self._cos2(p, q_y, q2)[..., :count])
        recoil = np.sqrt(HBARC_EV_ANG**2 * q2 + M_E_EV**2) - M_E_EV
        value = self.distant_ang2 * ratio / (np.pi * q2 * (1.0 + recoil / M_E_EV))
        return np.where(q2 <= self.distant_qmax_inv_ang**2, value, 0.0).sum(axis=-1)

    def distant(self, q_perp):
        """Azimuthal mean of :meth:`distant_pq` over ``|q_perp|`` (quarter-plane symmetric)."""
        q = np.asarray(q_perp, dtype=float)[..., None]
        return (self.distant_pq(q * np.cos(_PHI), q * np.sin(_PHI)) * _PHI_W).sum(axis=-1)

    def weight_pq(self, p, q_y):
        """Total direction-resolved ``dsigma/d^2q_perp`` per electron [angstrom^4]."""
        return self.distant_pq(p, q_y) + self.close(np.hypot(p, q_y))


def directional_channels(
    oscillators: MaterialShellOscillators,
    valence: DirectionalValence,
    kinetic_energy_eV: float,
    c_axis: str = "x",
) -> DirectionalChannels:
    """Channels with the conduction band of ``oscillators`` replaced by ``valence``.

    The replaced oscillator must carry exactly ``valence.electrons``, and the
    bound shells keep their strengths and resonances. Build ``oscillators``
    with :func:`~..transport.directional_loss.catalog_directional_oscillators`
    to keep the mean excitation energy closed.

    Validation: cr-valence-directional-loss
    """
    if c_axis not in ("x", "y", "z"):
        raise ValueError("c_axis must be 'x', 'y' or 'z'")
    cb = [o for o in oscillators.oscillators if o.atomic_number == 0]
    rest = tuple(o for o in oscillators.oscillators if o.atomic_number != 0)
    if len(cb) != 1 or abs(cb[0].strength - valence.electrons) > 1e-9 * valence.electrons:
        raise ValueError("oscillators need one conduction band carrying the valence electrons")
    shells = tuple(
        ShellOscillator(0, f"cb{k}", float(f), 0.0, float(w))
        for k, (f, w) in enumerate(zip(valence.mean_strengths, valence.energies_eV, strict=True))
    )
    base = electronic_channels(replace(oscillators, oscillators=shells + rest), kinetic_energy_eV)
    return DirectionalChannels(
        **{f.name: getattr(base, f.name) for f in fields(ElectronicChannels)},
        valence=valence,
        c_axis=c_axis,
    )


def plane_c_axis(normal, optic_axis=(0.0, 0.0, 1.0)) -> str | None:
    """Frame axis carrying ``c`` for a plane with this ``normal``, or ``None`` if beam-dependent.

    ``c`` along the plane normal (basal plane) is ``x`` for every beam. Any other
    orientation needs the beam azimuth (``y``: beam normal to ``c``, ``z``: beam
    along ``c``), which a plane alone does not fix.

    Validation: cr-valence-directional-loss
    """
    n = np.asarray(normal, dtype=float)
    c = np.asarray(optic_axis, dtype=float)
    cosine = abs(n @ c) / (np.linalg.norm(n) * np.linalg.norm(c))
    return "x" if cosine > 1.0 - 1e-9 else None
