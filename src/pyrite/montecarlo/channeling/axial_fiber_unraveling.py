"""Bloch-fiber trajectory unraveling with lattice and kick jumps (#623).

Research CPU reference. A trajectory carries a classical transverse carrier
(Bloch momentum) ``kappa`` and retained coefficients in the spectral subspace
P(kappa) of an order window R around it. Lattice-transfer jumps (TDS) keep
``kappa``; kick jumps (electronic) multiply by ``sqrt(rho_q)`` and shift the
carrier by a vector of magnitude ``q`` at uniform azimuth. All jump operators
commute with lattice translations up to that uniform kick, so the density
matrix stays block diagonal in the carrier and a classical carrier label is
exact. Every jump output is split into retained P(kappa'), represented
outgoing B - P(kappa') on an outgoing order box B containing R (measured with
the Weyl instrument), and unresolved (outside B after folding); the last two
terminate. Outputs are never wrapped
into the window. Material operators, matching, return flux, stopping and
admission remain caller obligations under #623/#624.

Validation: cr-axial-fiber-unraveling
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import brentq

from .axial_phase_space import phase_space_stream, sample_phase_space
from .axial_tds_continuum import StringChannel, TdsString, order_extents, string_jump

_FIBER_DOMAIN = 0x6230002
_MAX_CONDITION = 1e8
_COMPLEMENT_RTOL = 1e-9  # relative roundoff allowed in lattice_loss - sum L^dagger R L


@dataclass(frozen=True)
class FiberEvent:
    """Terminal record of one incident; see :func:`unravel_fiber_outgoing`.

    ``carrier_inv_ang`` is the folded Bloch carrier of the final state; the
    Weyl momentum index is an FFT-layout order index relative to it, with
    reciprocal spacing ``reciprocal_inv_ang`` of the provider.
    ``lattice_jumps`` counts every TDS jump (lattice columns or continuum
    strings); ``kick_jumps`` counts electronic kicks.
    """

    incident_id: int
    kind: str
    depth_ang: float
    weight: float
    lattice_jumps: int
    kick_jumps: int
    carrier_inv_ang: tuple[float, float]
    position_index: tuple[int, int] | None = None
    momentum_index: tuple[int, int] | None = None


@dataclass(frozen=True)
class FiberChannels:
    r"""Window-order operators shared by every carrier.

    ``order_max`` = M gives a centred (2M+1)^2 window in C order; a pair
    (M_x, M_y) gives the rectangular window [-M_x, M_x] x [-M_y, M_y] with
    (2M_x+1)(2M_y+1) orders (x outer). Spectra returned by
    ``spectrum(kappa)`` use that order. Choose per-axis extents by reciprocal
    reach (:func:`matched_order_extents`): on a cell with unequal ``|g|`` a
    square window reaches farther in momentum along the larger-``|g|`` axis.
    ``lattice_columns`` (J, 2D_x+1, 2D_y+1) are Kraus columns over transfer
    orders [-D_x, D_x] x [-D_y, D_y] [Å^-1/2];
    ``lattice_loss`` (K, K) is the full lattice loss effect on window inputs,
    including transfers beyond D [Å^-1], so that
    ``lattice_loss - sum_j L_j^dagger R L_j >= 0``. ``kick_columns``
    (I, 2H+1, 2H+1) are harmonics of ``sqrt(c_i rho_i)`` [Å^-1/2] with kick
    magnitudes ``kick_q_inv_ang`` (I,). ``kick_azimuths`` = None draws a
    uniform azimuth; a sequence selects uniformly among those angles (for
    commensurate oracle tests). ``cut(energies)`` returns the retained cut [eV].
    Kick columns may be rectangular (2H_x+1, 2H_y+1).
    ``outgoing_order_max`` = B >= M (default M; integer or per-axis pair,
    B_i >= M_i) sets the centred outgoing box [-B_x, B_x] x [-B_y, B_y] on
    which jump outputs are represented and measured; the retained window
    stays [-M_x, M_x] x [-M_y, M_y]. Lattice columns must cover
    ``D_i >= B_i + M_i`` on each axis: a box
    order then receives every transfer from every window order, so the
    represented outputs are exact and ``lattice_loss - sum_j L_j^dagger B L_j``
    is the PSD sum over outputs outside the box (sufficient for any
    covariance, per axis; necessary for output exactness on every axis and,
    on axes with ``M_i >= 1``, for remainder positivity). Columns must factor the covariance
    truncated at their full width: zero-padding a narrower factorization
    defeats the check and is caught only by the runtime complement refusal.
    Kick columns define their own model and need no such condition.
    ``tds_strings`` adds independent-string (continuum) TDS jumps
    (:mod:`.axial_tds_continuum`): carrier-moving, no transfer truncation
    and no D condition. Use them instead of, or alongside, lattice columns.
    The cell is rectangular: ``reciprocal_inv_ang`` are the axis-aligned
    primitive vectors and carriers fold into the half-open zone
    ``[-g/2, g/2)`` per axis.
    """

    order_max: int | tuple[int, int]
    reciprocal_inv_ang: tuple[float, float]
    spectrum: Callable[[tuple[float, float]], tuple[np.ndarray, np.ndarray]]
    cut: Callable[[np.ndarray], float]
    lattice_columns: np.ndarray
    lattice_loss: np.ndarray
    kick_columns: np.ndarray
    kick_q_inv_ang: np.ndarray
    kick_azimuths: Sequence[float] | None = None
    outgoing_order_max: int | tuple[int, int] | None = None
    tds_strings: Sequence[TdsString] = ()


def fiber_stream(seed: int, incident_id: int) -> np.random.Generator:
    """Version-2 jump-draw stream (group-then-channel draws) for one incident identity (order independent).

    Validation: cr-axial-fiber-unraveling
    """
    for name, value in (("seed", seed), ("incident_id", incident_id)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    key = np.random.SeedSequence((int(seed), _FIBER_DOMAIN, 2, int(incident_id)))
    return np.random.Generator(np.random.Philox(key))


def matched_order_extents(
    reciprocal_inv_ang: tuple[float, float], reach_inv_ang: float
) -> tuple[int, int]:
    """Smallest per-axis extents ``(M_x, M_y)`` with ``|g_i| M_i >= reach`` [Å^-1].

    Matching the reciprocal reach per axis keeps the plane-wave momentum cut
    (approximately) isotropic on a rectangular cell; ``reach = |g_x| M`` on a
    square cell returns ``(M, M)``. A relative ``1e-9`` slack absorbs roundoff.

    Validation: cr-axial-fiber-unraveling
    """
    g = np.abs(np.asarray(reciprocal_inv_ang, dtype=float))
    if g.shape != (2,) or not np.all(np.isfinite(g) & (g > 0)):
        raise ValueError("reciprocal_inv_ang must be two finite nonzero spacings")
    if not np.isfinite(reach_inv_ang) or reach_inv_ang < 0:
        raise ValueError("reach_inv_ang must be finite and nonnegative")
    m = np.ceil(reach_inv_ang / g * (1.0 - 1e-9))
    return int(m[0]), int(m[1])


def _autocorrelation_loss(columns: np.ndarray, order_max: tuple[int, int]) -> np.ndarray:
    """``sum_i S_i^dagger S_i`` on window inputs, all outputs: ``C_i(p - p')``.

    ``C(d) = sum_t conj(s[t]) s[t + d]`` is the inverse FFT of ``|S^|^2``. Lags
    are nonzero for ``|d_i| <= 2H_i`` and read for ``|d_i| <= 2M_i``; the
    per-axis FFT size ``max(2H_i + 2M_i + 1, 4H_i + 1)`` keeps both free of
    aliasing.

    Validation: cr-axial-fiber-unraveling
    """
    halves = (columns.shape[1] // 2, columns.shape[2] // 2)
    size = tuple(max(2 * h + 2 * m + 1, 4 * h + 1) for h, m in zip(halves, order_max, strict=True))
    spectrum = np.fft.fft2(columns, s=size)
    corr = np.fft.ifft2(np.sum(np.abs(spectrum) ** 2, axis=0))
    nx, ny = (np.arange(-m, m + 1) for m in order_max)
    px, py = (a.ravel() for a in np.meshgrid(nx, ny, indexing="ij"))
    return corr[(px[:, None] - px[None, :]) % size[0], (py[:, None] - py[None, :]) % size[1]]


def _crop(full_half: int, box: int) -> tuple[np.ndarray, np.ndarray]:
    """Index columns copying centred orders ``[-min, min]`` from a full array to the box."""
    half = min(full_half, box)
    orders = np.arange(-half, half + 1)
    return (orders + full_half)[:, None], (orders + box)[:, None]


def _extents(channels: FiberChannels) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validated per-axis window ``(M_x, M_y)`` and outgoing box ``(B_x, B_y)``.

    Validation: cr-axial-fiber-unraveling
    """
    m = order_extents(channels.order_max, "order_max")
    box = channels.outgoing_order_max
    if box is None:
        return m, m
    try:
        b = order_extents(box, "outgoing_order_max")
    except ValueError:
        b = (-1, -1)
    if b[0] < m[0] or b[1] < m[1]:
        raise ValueError("outgoing_order_max must be integer extents >= order_max per axis")
    return m, b


def _check_channels(channels: FiberChannels, k: int) -> None:
    """Refuse malformed channels; lattice exactness needs ``D_i >= B_i + M_i`` per axis.

    Validation: cr-axial-fiber-unraveling
    """
    columns, q = np.asarray(channels.kick_columns), np.asarray(channels.kick_q_inv_ang)
    for name, array in (("lattice_columns", channels.lattice_columns), ("kick_columns", columns)):
        shape = np.shape(array)
        if len(shape) != 3 or shape[1] % 2 == 0 or shape[2] % 2 == 0:
            raise ValueError(f"{name} must be (n, 2D_x+1, 2D_y+1) centred odd columns")
    if columns.ndim != 3 or q.shape != (len(columns),):
        raise ValueError("kick_q_inv_ang must give one magnitude per kick column")
    if np.asarray(channels.lattice_loss).shape != (k, k):
        raise ValueError("lattice_loss must act on the order window")
    if channels.kick_azimuths is not None and len(channels.kick_azimuths) == 0:
        raise ValueError("kick_azimuths must be None or non-empty")
    m, b = _extents(channels)
    d = np.array(np.shape(channels.lattice_columns)[1:]) // 2
    if np.any(d < np.add(b, m)):
        raise ValueError(
            "lattice columns must span transfers D >= outgoing_order_max + order_max per axis"
        )


def _fold(moved: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Integer shift putting ``moved`` in ``[-g/2, g/2)``; boundary ties are rounded away.

    Validation: cr-axial-fiber-unraveling
    """
    return np.floor(np.round(moved / g, 12) + 0.5).astype(int)


@dataclass
class _Fiber:
    vectors: np.ndarray  # (K, n) retained columns
    eigen: tuple[np.ndarray, np.ndarray, np.ndarray]
    hbar_v: float

    def evolve(self, c: np.ndarray, depth: float) -> np.ndarray:
        lam, right, inverse = self.eigen
        return right @ (np.exp(-1j * depth / self.hbar_v * lam) * (inverse @ c))


class _Engine:
    def __init__(self, channels: FiberChannels, hbar_v: float, cache_size: int = 16):
        self.ch, self.hbar_v = channels, float(hbar_v)
        m, b = _extents(channels)
        self.m, self.box = np.array(m), np.array(b)
        self.shape = tuple(int(x) for x in 2 * self.m + 1)
        self.k = self.shape[0] * self.shape[1]
        self.g = np.asarray(channels.reciprocal_inv_ang, dtype=float)
        _check_channels(channels, self.k)
        self.loss = channels.lattice_loss + _autocorrelation_loss(channels.kick_columns, m)
        d = np.array(channels.lattice_columns.shape[1:]) // 2
        self.lattice_size = tuple(int(x) for x in np.array(self.shape) + 2 * d)
        self.lattice_fft = np.fft.fft2(channels.lattice_columns, s=self.lattice_size)
        self.kick_half = np.array(channels.kick_columns.shape[1:]) // 2
        self.kick_size = tuple(int(x) for x in np.array(self.shape) + 2 * self.kick_half)
        self.box_shape = tuple(int(x) for x in 2 * self.box + 1)
        inner = [np.arange(bi - mi, bi + mi + 1) for bi, mi in zip(b, m, strict=True)]
        self.window_in_box = np.ravel_multi_index(
            np.meshgrid(*inner, indexing="ij"), self.box_shape
        ).ravel()
        self.lattice_crop = tuple(
            _crop(int(mi + di), bi) for mi, di, bi in zip(m, d, b, strict=True)
        )
        self.kick_fft = np.fft.fft2(channels.kick_columns, s=self.kick_size)
        self.kick_power = np.abs(self.kick_fft) ** 2
        nx, ny = (np.arange(-mi, mi + 1) for mi in m)
        orders = np.stack(np.meshgrid(nx, ny, indexing="ij"), -1).reshape(-1, 2)
        self.strings = [StringChannel(t, self.g, orders) for t in channels.tds_strings]
        for channel in self.strings:
            self.loss = self.loss + channel.loss
        self.cache: OrderedDict[tuple[float, float], _Fiber] = OrderedDict()
        self.cache_size = cache_size

    def fiber(self, kappa: tuple[float, float]) -> _Fiber:
        key = (round(float(kappa[0]), 12), round(float(kappa[1]), 12))
        if key not in self.cache:
            energies, vectors = self.ch.spectrum(key)
            keep = energies < self.ch.cut(energies)
            v = np.asarray(vectors, dtype=complex)[:, keep]
            h_eff = np.diag(energies[keep]) - 0.5j * self.hbar_v * (v.conj().T @ self.loss @ v)
            lam, right = np.linalg.eig(h_eff)
            if np.linalg.cond(right) > _MAX_CONDITION:
                raise FloatingPointError("no-jump generator eigenvectors are ill conditioned")
            self.cache[key] = _Fiber(v, (lam, right, np.linalg.inv(right)), self.hbar_v)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def lattice_outputs(self, chi: np.ndarray) -> np.ndarray:
        """``B L_j chi`` on the outgoing box for every Kraus column, shape (J, Kb).

        Full linear-convolution orders [-(M_i+D_i), M_i+D_i] sit at index
        ``p_i + M_i + D_i`` per axis.
        Validation: cr-axial-fiber-unraveling
        """
        padded = np.fft.fft2(chi.reshape(self.shape), s=self.lattice_size)
        full = np.fft.ifft2(self.lattice_fft * padded)
        (source_x, target_x), (source_y, target_y) = self.lattice_crop
        out = np.zeros((len(full), *self.box_shape), dtype=complex)
        out[:, target_x, target_y.T] = full[:, source_x, source_y.T]
        return out.reshape(len(full), -1)

    def kick_rates(self, chi: np.ndarray) -> np.ndarray:
        """``||S_i chi||^2`` over all outputs, by Parseval of the linear convolution.

        Validation: cr-axial-fiber-unraveling
        """
        padded = np.fft.fft2(chi.reshape(self.shape), s=self.kick_size)
        norm = self.kick_size[0] * self.kick_size[1]
        return np.einsum("ixy,xy->i", self.kick_power, np.abs(padded) ** 2) / norm

    def kick_output(self, i: int, chi: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Outgoing-box part of ``S_i chi`` after relabelling orders by ``shift``.

        Box order p' reads unrelabelled order p' - shift at full index
        ``p' - shift + M + H`` (per axis).
        Validation: cr-axial-fiber-unraveling
        """
        padded = np.fft.fft2(chi.reshape(self.shape), s=self.kick_size)
        full = np.fft.ifft2(self.kick_fft[i] * padded)  # output order p at index p + M + H
        m, h, b = self.m, self.kick_half, self.box
        size = np.array(self.kick_size)
        out = np.zeros(self.box_shape, dtype=complex)
        for axis_index in np.ndindex(out.shape):
            source = np.array(axis_index) - b - shift + m + h
            if np.all(source >= 0) and np.all(source < size):
                out[axis_index] = full[source[0], source[1]]
        return out.ravel()


def _measure(engine, phi_orders, window, seed, incident):
    shape = engine.box_shape
    layout = np.zeros(shape, dtype=complex)
    nx, ny = (np.arange(-b, b + 1) for b in engine.box)
    layout[np.ix_(nx % shape[0], ny % shape[1])] = phi_orders.reshape(shape)
    phi = np.fft.ifft2(layout, norm="ortho")
    return sample_phase_space(phi, window, phase_space_stream(seed, incident))


def _split(fiber: _Fiber, out: np.ndarray, outside: float, rng, window_in_box: np.ndarray):
    """Choose retained / outgoing / unresolved for one box output.

    ``window_in_box`` maps window orders into the box; P(kappa) lives on the window.
    Validation: cr-axial-fiber-unraveling
    """
    c = fiber.vectors.conj().T @ out[window_in_box]
    f = out.copy()
    f[window_in_box] -= fiber.vectors @ c
    weights = np.array([np.vdot(c, c).real, np.vdot(f, f).real, max(outside, 0.0)])
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("selected jump channel has no finite output")
    choice = int(rng.choice(3, p=weights / total))
    return choice, c, f


def _lattice_channel(engine, chi, lattice_total, rng):
    """Draw a Kraus column given a lattice jump; ``None`` is the out-of-box remainder.

    Conditional weights ``||B L_j chi||^2 / <chi|Gamma_lat|chi>`` and the
    remainder; with the group draw this is the one-stage channel law.
    Validation: cr-axial-fiber-unraveling
    """
    outs = engine.lattice_outputs(chi)
    represented = np.sum(np.abs(outs) ** 2, axis=1)
    unresolved = lattice_total - float(represented.sum())
    if unresolved < -_COMPLEMENT_RTOL * max(lattice_total, float(represented.sum())):
        raise ValueError("lattice_loss is smaller than its represented Kraus outputs")
    weights = np.append(represented, max(unresolved, 0.0))
    j = int(rng.choice(weights.size, p=weights / weights.sum()))
    return None if j == len(represented) else outs[j]


def _jump(engine, fiber, chi, kappa, rng):
    """Draw one jump from ``chi``: group, channel, fold and split.

    Groups: lattice columns ``<chi|Gamma_lat|chi>``, each continuum string
    ``<chi|Gamma_s|chi>``, each kick node ``||S_i chi||^2``. Returns
    ``(kind, kappa, fiber, choice, c_new, f)``; ``choice`` None is unresolved.
    Validation: cr-axial-fiber-unraveling
    """
    lattice_total = float(np.vdot(chi, engine.ch.lattice_loss @ chi).real)
    strings = [channel.rate(chi) for channel in engine.strings]
    kick = engine.kick_rates(chi)
    rates = np.maximum(np.concatenate([[lattice_total], strings, kick]), 0.0)
    total = float(rates.sum())
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("jump at a state with no finite jump rate")
    group = int(rng.choice(rates.size, p=rates / total))
    if group == 0:
        out = _lattice_channel(engine, chi, lattice_total, rng)
        if out is None:
            return "tds", kappa, fiber, None, None, None
        return ("tds", kappa, fiber, *_split(fiber, out, 0.0, rng, engine.window_in_box))
    if group <= len(strings):
        result = string_jump(
            engine.strings[group - 1],
            chi,
            kappa,
            tuple(engine.m),
            tuple(engine.box),
            rng,
            lambda moved: _fold(moved, engine.g),
        )
        if result is None:
            return "tds", kappa, fiber, None, None, None
        kappa, out = result
        fiber = engine.fiber(kappa)
        return ("tds", kappa, fiber, *_split(fiber, out, 0.0, rng, engine.window_in_box))
    i = group - 1 - len(strings)
    q = float(engine.ch.kick_q_inv_ang[i])
    azimuths = engine.ch.kick_azimuths
    theta = (
        2 * np.pi * rng.random()
        if azimuths is None
        else float(azimuths[rng.integers(len(azimuths))])
    )
    moved = np.array(kappa) + q * np.array([np.cos(theta), np.sin(theta)])
    shift = _fold(moved, engine.g)
    kappa = tuple(float(x) for x in moved - shift * engine.g)
    fiber = engine.fiber(kappa)
    out = engine.kick_output(i, chi, shift).ravel()
    outside = float(kick[i]) - float(np.vdot(out, out).real)
    return ("kick", kappa, fiber, *_split(fiber, out, outside, rng, engine.window_in_box))


def _waiting(fiber, c, threshold, start, stop, step, tolerance):
    depth, current = start, c
    while depth < stop:
        span = min(step, stop - depth)
        trial = fiber.evolve(current, span)
        if float(np.vdot(trial, trial).real) <= threshold:
            # SciPy's root-finder wrapper keeps the callable in a reference
            # cycle; hold the fibre in a list emptied afterwards so that cyclic
            # garbage does not pin MB-sized eigenbases between GC passes.
            held = [fiber, current]

            def excess(s, held=held):
                state = held[0].evolve(held[1], s)
                return float(np.vdot(state, state).real) - threshold

            offset = brentq(excess, 0.0, span, xtol=tolerance)
            held.clear()
            return depth + offset, fiber.evolve(current, offset)
        depth, current = depth + span, trial
    return None, current


def _trajectory(engine, entry, kappa, window, max_depth, seed, incident, step, tolerance):
    """One terminal-event history; law in :func:`unravel_fiber_outgoing`.

    Validation: cr-axial-fiber-unraveling
    """
    rng = fiber_stream(seed, incident)
    fiber = engine.fiber(kappa)
    c = fiber.vectors.conj().T @ entry
    weight = float(np.vdot(c, c).real)
    if weight <= 0:
        raise ValueError("entry state has no retained weight")
    c = c / np.sqrt(weight)
    depth, counts = 0.0, {"tds": 0, "kick": 0}

    def event(kind, at, **extra):
        return FiberEvent(incident, kind, at, weight, counts["tds"], counts["kick"], kappa, **extra)

    def outgoing(f, at):
        o = _measure(engine, f, window, seed, incident)
        return event(
            "outgoing", at, position_index=o.position_index, momentum_index=o.momentum_index
        )

    while True:
        depth, state = _waiting(fiber, c, rng.random(), depth, max_depth, step, tolerance)
        if depth is None:
            return event("retained", max_depth)
        chi = fiber.vectors @ (state / np.linalg.norm(state))
        kind, kappa, fiber, choice, c_new, f = _jump(engine, fiber, chi, kappa, rng)
        counts[kind] += 1
        if choice is None:
            return event("unresolved", depth)
        if choice == 0:
            c = c_new / np.linalg.norm(c_new)
        elif choice == 1:
            return outgoing(f, depth)
        else:
            return event("unresolved", depth)


def unravel_fiber_outgoing(
    channels: FiberChannels,
    entry_orders: np.ndarray,
    *,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    hbar_v_eV_ang: float,
    window,
    max_depth_ang: float,
    incident_ids,
    seed: int = 0,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
) -> list[FiberEvent]:
    r"""Unravel the carrier-fibred absorbing model into terminal events.

    Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (4), (8). In fibre
    $\kappa$ the retained state evolves under
    $P(\kappa)[H(\kappa)-i\hbar v(\Gamma_\mathrm{lat}+\Gamma_\mathrm{kick})/2]P(\kappa)$
    with $\Gamma_\mathrm{kick}=\sum_iS_i^\dagger S_i$. At a jump (depth from the
    norm law, root found) a channel is drawn with weights
    $\lVert BL_j\chi\rVert^2$, $\langle\chi|\Gamma_\mathrm{lat}|\chi\rangle-
    \sum_j\lVert BL_j\chi\rVert^2$ (unresolved) and $\lVert S_i\chi\rVert^2$,
    with $B$ the outgoing order box. A kick then draws its azimuth, folds
    $\kappa+q\hat e$ into the first zone and relabels orders; every output is
    split into retained $P(\kappa')$ (continue), represented outgoing
    $B-P(\kappa')$ (Weyl-measured, terminate) and outside the box
    (unresolved, terminate). A kick's rate does not depend on the
    azimuth because $e^{i\mathbf q\cdot\mathbf r}$ is unitary and commutes
    with $S_i$, so drawing the azimuth after the node is exact.

    ``entry_orders`` (K,) is the entrance amplitude in the window order basis
    at ``carrier_inv_ang``; its projection on P gives the incident ``weight``.
    ``window`` is the unit-norm Weyl window on the (2B_x+1, 2B_y+1) outgoing
    box grid; outgoing indices refer to that grid.
    Units: eV, Å, jumps Å^-1/2. Zero channels leave every incident retained.
    Each incident uses :func:`fiber_stream` for jumps and
    :func:`.phase_space_stream` for its one measurement. No event is a
    physical escape or native launch.

    Validation: cr-axial-fiber-unraveling
    """
    for name, value in (
        ("hbar_v_eV_ang", hbar_v_eV_ang),
        ("max_depth_ang", max_depth_ang),
        ("search_step_ang", search_step_ang),
        ("depth_tolerance_ang", depth_tolerance_ang),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    engine = _Engine(channels, hbar_v_eV_ang)
    entry = np.asarray(entry_orders, dtype=complex)
    if entry.shape != (engine.k,) or not np.all(np.isfinite(entry)):
        raise ValueError("entry_orders must be finite on the order window")
    g = np.asarray(window, dtype=complex)
    if g.shape != engine.box_shape or not np.isclose(np.vdot(g, g).real, 1.0):
        raise ValueError("window must be a unit-norm array on the outgoing box shape")
    kappa = (float(carrier_inv_ang[0]), float(carrier_inv_ang[1]))
    return [
        _trajectory(
            engine,
            entry,
            kappa,
            g,
            float(max_depth_ang),
            seed,
            int(i),
            search_step_ang,
            depth_tolerance_ang,
        )
        for i in incident_ids
    ]
