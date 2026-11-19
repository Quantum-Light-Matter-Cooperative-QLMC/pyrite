"""Device route of automatic line-axis sizing for resident populations (#1275).

When the transport segments and the collected line population stay on the
accelerator, the sinc-spacing samples, their weighted quantile and the
measured ``stop`` search run there instead of copying tens of millions of rows
to the host (:mod:`.diagnostics`, :mod:`.cxr_seeds`). Only scalars cross back.

Each per-element value is formed with the host route's operations, operand
dtypes and order, one IEEE operation per CuPy elementwise kernel (no FMA
contraction across them), so samples, sorted order and tail products equal
the host's. The documented differences are reductions and one dot product:

- the beam-direction dot product is ``v_x n_x + v_y n_y + v_z n_z`` summed
  left to right, where the host's ``direction @ n_hat`` leaves order and FMA
  use to BLAS;
- the cumulative weight, the aliased weight share and the stop search's tail
  and mass sums are device tree reductions in float64, where the host sums
  sequentially (``cumsum``), pairwise per block, and -- for the total line mass
  of a float32 population -- in float32.

The selected quantile is a sample width and the stop a ``round_to_eV`` lattice
point, so these roundings move ``E_grid`` only when a reduction lands within
its rounding of the threshold; the recorded fractions differ in their last
bits. Grouped flights (rows out of strict ``(owner, flight)`` order) keep the
host route.

Validation: line-grid-resonance-bandwidth
"""

import numpy as np

from ..._backend import REAL, _to_cpu, array_namespace, exact_rawkernel_options, is_device_array, xp
from ...materials.crystal import HBARC_EV_ANG

#: Share of free device memory a resident population may take; beyond it the
#: collector moves the population to the host and the host route sizes it.
DEVICE_POPULATION_SHARE = 1.0 / 3.0


def device_resident(*arrays) -> bool:
    """Whether every array lives on the selected accelerator."""
    return bool(arrays) and all(is_device_array(array) for array in arrays)


def _module(array):
    """The array's module: CuPy on the device; NumPy runs the same route in tests."""
    return array_namespace(array)


def device_sinc_samples(segments, n_hat, *, electron_limit=None):
    """Device :func:`.diagnostics.sinc_feature_samples`, or ``None`` for the host route.

    Returns float64 ``(width, weight)`` device arrays of the valid lines. Rows
    sharing an ``(owner, flight_id)`` key are flight-grouped on the host
    route; any row out of strict key order returns ``None`` so that route,
    and its exact reduction, sizes the case.

    Validation: line-grid-resonance-bandwidth
    """
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    cp = _module(segments["L_ang"])
    energy = segments[energy_field].astype(cp.float64, copy=False)
    length = segments["L_ang"].astype(cp.float64, copy=False)
    direction = segments["v_hat"].astype(cp.float64, copy=False)
    keys = None
    if segments.get("flight_id") is not None:
        track = segments.get("track_id")
        keys = (segments["elec_id"] if track is None else track, segments["flight_id"])
    if electron_limit is not None:
        line = segments["elec_id"] < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]
        if keys is not None:
            keys = (keys[0][line], keys[1][line])
    if keys is not None and not _strictly_ordered(*keys):
        return None
    del keys
    # transport.kinematics._beta_array; NumPy's ``** 0.5`` is ``sqrt``.
    gamma = 1.0 + energy / 510.99895
    beta = cp.sqrt(1.0 - 1.0 / (gamma * gamma))
    del gamma
    t_L = length / beta
    n = [float(component) for component in np.asarray(n_hat, dtype=float)]
    dot = direction[:, 0] * n[0]
    dot = dot + direction[:, 1] * n[1]
    dot = dot + direction[:, 2] * n[2]
    denominator = 1.0 - beta * dot
    del dot, beta
    width = 2.0 * np.pi * HBARC_EV_ANG / (denominator * t_L)
    weight = t_L * t_L
    valid = (
        cp.isfinite(width)
        & cp.isfinite(weight)
        & (width > 0.0)
        & (weight > 0.0)
        & (denominator > 0.0)
    )
    keep = cp.flatnonzero(valid)
    return width[keep], weight[keep]


def _strictly_ordered(owner, flight) -> bool:
    later = owner[1:] > owner[:-1]
    later |= (owner[1:] == owner[:-1]) & (flight[1:] > flight[:-1])
    return bool(later.all())


def device_sinc_spacing(width, weight, *, aliased_weight_limit):
    """Device :func:`.diagnostics.sinc_spacing_from_samples` of float64 samples.

    The same stable width order (CuPy's sort is stable), the first index whose
    cumulative weight reaches ``aliased_weight_limit`` of the total, and the
    weight share strictly below the selected width. The cumulative and share
    sums are device scans/reductions (module docstring).

    Validation: line-grid-resonance-bandwidth
    """
    cp = _module(width)
    if width.size == 0:
        raise ValueError("cannot derive sinc spacing: diagnostic transport has no valid segments")
    order = cp.argsort(width)
    width = width[order]
    weight = weight[order]
    del order
    cumulative = cp.cumsum(weight)
    total = float(cumulative[-1])
    target = cp.asarray([float(aliased_weight_limit) * total])
    index = int(cp.searchsorted(cumulative, target, side="left")[0])
    del cumulative
    step = float(width[min(index, width.size - 1)])
    below = int(cp.searchsorted(width, cp.asarray([step]), side="left")[0])
    aliased = float(weight[:below].sum()) / total
    return step, aliased, int(width.size)


_TAIL_PREAMBLE = """
template <typename T>
__device__ double pyrite_line_mass(T energy, T weight, T width) {
    T mass = weight * width;
    return (isfinite(mass) && mass > (T)0) ? (double)mass : 0.0;
}
template <typename T>
__device__ double pyrite_tail_product(T energy, T weight, T width, T stop) {
    T mass = weight * width;
    if (!(isfinite(mass) && mass > (T)0)) return 0.0;
    double distance = (double)(T)(stop - energy);
    double bound = 1.0;
    if (distance > 0.0) {
        bound = (double)width / (PI_SQUARED * distance);
        if (bound > 1.0) bound = 1.0;
    }
    return (double)mass * bound;
}
"""

_KERNELS: dict = {}


def _host_mass_sum(energy, weight, width):
    mass = weight * width
    return np.where(np.isfinite(mass) & (mass > 0.0), mass, 0.0).astype(np.float64).sum()


def _host_tail_sum(energy, weight, width, stop):
    from ._population_tail import tail_products

    out = np.empty(energy.size)
    return out[: tail_products(energy, weight, width, stop, out)].sum()


def _kernels(cp):
    """``(mass, tail)`` float64 sum kernels, compiled without FMA contraction.

    On NumPy arrays (tests of this route without a device) the same per-line
    values summed by NumPy.
    """
    if cp is np:
        return _host_mass_sum, _host_tail_sum
    cached = _KERNELS.get(id(cp))
    if cached is None:
        preamble = f"#define PI_SQUARED {np.pi**2!r}\n" + _TAIL_PREAMBLE
        options = exact_rawkernel_options(cp)
        mass = cp.ReductionKernel(
            "T energy, T weight, T width",
            "float64 total",
            "pyrite_line_mass(energy, weight, width)",
            "a + b",
            "total = a",
            "0",
            "pyrite_line_mass_sum",
            reduce_type="double",
            preamble=preamble,
            options=options,
        )
        tail = cp.ReductionKernel(
            "T energy, T weight, T width, T stop",
            "float64 total",
            "pyrite_tail_product(energy, weight, width, stop)",
            "a + b",
            "total = a",
            "0",
            "pyrite_tail_sum",
            reduce_type="double",
            preamble=preamble,
            options=options,
        )
        cached = _KERNELS[id(cp)] = (mass, tail)
    return cached


class DeviceTail:
    """Tail-bound sums of resident populations for the measured ``stop`` search.

    Per line the products of :func:`._population_tail.tail_products` (``mass``
    and ``stop - energy`` in the arrays' dtype, ``stop`` cast to it, bound and
    product in float64), summed per population array by one fused device
    reduction and added on the host in population order.

    Validation: line-grid-resonance-bandwidth
    """

    def __init__(self, populations):
        self.arrays = [(p.energy_eV, p.weight, p.width_eV) for p in populations if p.energy_eV.size]
        cp = _module(self.arrays[0][0]) if self.arrays else None
        self._cp = cp
        self.total = 0.0
        self.n_lines = 0
        self.min_energy = np.inf
        self.max_energy = -np.inf
        if cp is None:
            return
        mass_sum, _ = _kernels(cp)
        for energy, weight, width in self.arrays:
            mass = weight * width
            keep = cp.isfinite(mass) & (mass > 0.0)
            count = int(cp.count_nonzero(keep))
            if not count:
                continue
            self.total += float(mass_sum(energy, weight, width))
            self.n_lines += count
            self.min_energy = min(self.min_energy, float(cp.where(keep, energy, cp.inf).min()))
            self.max_energy = max(self.max_energy, float(cp.where(keep, energy, -cp.inf).max()))

    def numerator(self, stop: float) -> float:
        """``sum(mass * sincsq_upper_tail_bound(width, stop - energy))`` over kept lines."""
        _, tail_sum = _kernels(self._cp)
        numerator = 0.0
        for energy, weight, width in self.arrays:
            numerator += float(tail_sum(energy, weight, width, energy.dtype.type(stop)))
        return numerator


def resident_population_budget(array) -> int | None:
    """Device bytes a population collected beside ``array`` may hold, or ``None``.

    ``None`` (host collection) unless ``array`` is device-resident on CUDA:
    :data:`DEVICE_POPULATION_SHARE` of the free device memory, counting the
    pool's cache within its limit. A population over it is sized on the host.
    """
    if not device_resident(array):
        return None
    from ..transport.continuation import _free_device_bytes

    free = _free_device_bytes(_module(array))
    return None if free is None else int(free * DEVICE_POPULATION_SHARE)


def collect_lines(audit, lines):
    """Append one chunk's ``(resonance, width, coefficient)`` to ``audit["collect"]``.

    In ``REAL`` dtype, on the host; or, while ``audit["device_bytes"]`` (a
    remaining byte budget) covers it, as device arrays, so the measured sizing
    reads them where they are. A chunk over the budget moves every collected
    chunk to the host and ends device collection, so a population is wholly on
    one side. The values are the same either way.

    Validation: line-grid-resonance-bandwidth
    """
    collected = audit["collect"]
    budget = audit.get("device_bytes")
    if budget is not None and is_device_array(lines[0]):
        chunk = tuple(xp.asarray(array, dtype=REAL) for array in lines)
        size = sum(int(array.nbytes) for array in chunk)
        if size <= budget:
            audit["device_bytes"] = budget - size
            collected.append(chunk)
            return
        audit["device_bytes"] = None
        collected[:] = [tuple(np.asarray(_to_cpu(a)) for a in old) for old in collected]
    collected.append(tuple(np.asarray(_to_cpu(array), dtype=REAL) for array in lines))
