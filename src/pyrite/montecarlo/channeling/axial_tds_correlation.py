r"""Correlated thermal displacements within an axial string (#1045). Research CPU reference.

Opt-in refinement of the independent-atom continuum TDS jumps
(``cr-axial-continuum-tds-jumps``). Atoms ``i, j`` of one string have jointly
Gaussian transverse displacements with
$\langle u_{i\alpha}u_{j\beta}\rangle=a^2c_{i-j}\delta_{\alpha\beta}$. Within
the sudden (q_z -> 0) approximation the per-length dissipator gains, for each
partner at correlation ``c``, the pair kernel
$\chi_t\chi_{t'}(e^{c a^2\,t\cdot t'}-1)$ next to the self kernel
($c=1$). For $0\le c\le1$ that kernel is exactly the independent-string
kernel of a string with phase $\varphi(q)e^{-(1-c)a^2q^2/2}$ and displacement
variance $c\,a^2$ (common mode $v\sim N(0,ca^2)$):

$$
\mathbb E_v\big[w^{(c)}_t(v)\,w^{(c)}_{t'}(v)^*\big]
=\chi_t\chi_{t'}\big(e^{ca^2t\cdot t'}-1\big),\qquad
w^{(c)}_t(v)=e^{-(1-c)a^2t^2/2}\big(e^{-it\cdot v}-e^{-ca^2t^2/2}\big),
$$

so each correlated shell is one extra :class:`TdsString` (linear density
``partners * lambda``) and the existing sampler, loss and jump apply
unchanged. ``c = 0`` adds nothing (bitwise the uncorrelated model); ``c = 1``
with one partner doubles the rate (Andersen et al. 1982, Sec. 6).

Validation: cr-axial-correlated-tds
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .axial_tds_continuum import TdsString

#: Si transverse displacement correlations ``c = 1 - <du_perp^2>/<du_perp^2>_uncorr``
#: at 300 K, adiabatic bond-charge model, digitized from Nielsen & Weber,
#: J. Phys. C 13, 2449 (1980), Figs. 3(b) and 4(b) (+-0.005). Keys are the
#: separation in units of a/4 (squared, to stay exact).
NIELSEN_WEBER_SI_300K: dict[int, float] = {3: 0.240, 16: 0.071, 64: 0.028}
#: Power-law envelope ``c = k r^-p`` (r in a/4) fitted to the 300 K transverse
#: points of N&W Fig. 4(b) with r >= 2.8 (nine atoms, 10% rms): used only for
#: string partners N&W did not compute.
NIELSEN_WEBER_SI_ENVELOPE: tuple[float, float] = (0.381, 1.186)


@dataclass(frozen=True)
class PairCorrelation:
    """One correlation shell of a string: ``partners`` atoms per atom at ``coefficient``.

    ``coefficient`` is the transverse displacement correlation
    ``<u_0 . u_j> / <u_0 . u_0>`` in ``[0, 1]``; ``partners`` counts both
    sides (2 for ``+-j`` on a uniform string).
    Validation: cr-axial-correlated-tds
    """

    coefficient: float
    partners: float


def correlated_strings(
    string: TdsString, shells: Sequence[PairCorrelation] = ()
) -> tuple[TdsString, ...]:
    """``string`` plus one pair-kernel string per correlated shell.

    Zero-coefficient shells add nothing, so no shells (or all zero) returns
    ``(string,)`` itself: bitwise the uncorrelated model.
    Validation: cr-axial-correlated-tds
    """
    out = [string]
    a2 = float(string.displacement_ang2)
    for shell in shells:
        c, m = float(shell.coefficient), float(shell.partners)
        if not (np.isfinite(c) and 0.0 <= c <= 1.0):
            raise ValueError("pair correlation must lie in [0, 1]")
        if not (np.isfinite(m) and m >= 0.0):
            raise ValueError("partner count must be finite and nonnegative")
        if c == 0.0 or m == 0.0:
            continue
        damping = 0.5 * (1.0 - c) * a2

        def phase(q, base=string.phase, damping=damping):
            return base(q) * np.exp(-damping * np.asarray(q) ** 2)

        out.append(
            replace(
                string,
                displacement_ang2=c * a2,
                linear_density_inv_ang=m * string.linear_density_inv_ang,
                phase=phase,
            )
        )
    return tuple(out)


def _string_partners(axis) -> list[tuple[int, int]]:
    """``(r^2 in (a/4)^2, partners)`` along a Si string, nearest first (diamond)."""
    uvw = tuple(sorted(abs(int(v)) for v in axis))
    if uvw == (0, 0, 1):  # atoms every a: +-4 in a/4
        offsets = [4 * n for n in range(-12, 13) if n]
    elif uvw == (1, 1, 1):  # period a sqrt(3): A at 0, B at +sqrt(3) (a/4)
        offsets = [4 * n for n in range(-12, 13) if n] + [1 + 4 * n for n in range(-12, 12)]
        offsets = [o * o * 3 for o in offsets]
        return _grouped(offsets)
    else:
        raise ValueError("Si string correlations exist for <100> and <111> only")
    return _grouped([o * o for o in offsets])


def _grouped(squares) -> list[tuple[int, int]]:
    values, counts = np.unique(np.asarray(squares, dtype=np.int64), return_counts=True)
    return [(int(v), int(k)) for v, k in zip(values, counts, strict=True)]


def silicon_string_correlations(axis, partners: int = 10) -> tuple[PairCorrelation, ...]:
    """Room-temperature Si shells of the ``partners`` nearest string atoms.

    Coefficients are the Nielsen & Weber (1980) 300 K bond-charge-model values
    where computed, else their fitted envelope; a degenerate shell cut by
    ``partners`` keeps a fractional count. ``partners = 6`` is the range of
    Andersen et al. (1982) Table 2's ``C`` (1.238 for <100>, 1.420 for <111>
    here, against their 1.25 and 1.40); ``~1/(alpha Z) = 10`` is the
    coherently scattering range.
    Validation: cr-axial-correlated-tds
    """
    if isinstance(partners, bool) or int(partners) != partners or partners < 0:
        raise ValueError("partners must be a nonnegative integer")
    k, p = NIELSEN_WEBER_SI_ENVELOPE
    shells, left = [], int(partners)
    for r2, count in _string_partners(axis):
        if left == 0:
            break
        c = NIELSEN_WEBER_SI_300K.get(r2, k * float(r2) ** (-0.5 * p))
        take = min(count, left)
        shells.append(PairCorrelation(float(c), float(take)))
        left -= take
    return tuple(shells)
