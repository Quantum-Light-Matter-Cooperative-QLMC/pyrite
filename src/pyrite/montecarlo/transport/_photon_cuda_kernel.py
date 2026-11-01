"""Fused binary64 first-flight geometry and sampling; no RNG or new physics.

The two kernels replace the array reference's launch/compaction sequence.
Intervals are independent ray/region pairs. One sampling thread per ray walks
positive-rate intervals in stable (entry, region) order, then its atom/channel
CDF. The scan uses O(regions²) comparisons and O(rays*regions) interval storage;
it targets the bounded analytic scenes of #341, not large mesh scenes.
Validation: photon-channel-first-interaction
Validation: photon-scene-navigation
"""

from functools import cache

import numpy as np

SOURCE = r"""
#define PHOTON_INF __longlong_as_double(0x7ff0000000000000LL)

__device__ void clip_axis(double o, double d, double lo, double hi,
                          double &start, double &end) {
    if (d == 0.0) {
        if (!(o >= lo && o < hi)) end = -PHOTON_INF;
    } else {
        const double a = (lo - o) / d, b = (hi - o) / d;
        start = fmax(start, fmin(a, b));
        end = fmin(end, fmax(a, b));
    }
}

extern "C" __global__ void photon_intervals(
    const double *origins, const double *directions, const double *layers,
    const double *boxes, long long n, int k, int r, double hx, double hy,
    double *starts, double *ends) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * r) return;
    const long long ray = tid / r;
    const int region = tid % r;
    double start = 0.0, end = PHOTON_INF;
    if (region < k) {
        clip_axis(origins[3*ray+2], directions[3*ray+2],
                  layers[2*region], layers[2*region+1], start, end);
        if (isfinite(hx)) {
            clip_axis(origins[3*ray], directions[3*ray], -hx, hx, start, end);
            clip_axis(origins[3*ray+1], directions[3*ray+1], -hy, hy, start, end);
        }
    } else {
        // Box rows: center(3), orthonormal local axes(9), half extents(3).
        const double *box = boxes + (region-k)*15;
        for (int axis = 0; axis < 3; ++axis) {
            double o = 0.0, d = 0.0;
            for (int j = 0; j < 3; ++j) {
                o += (origins[3*ray+j] - box[j]) * box[3+3*axis+j];
                d += directions[3*ray+j] * box[3+3*axis+j];
            }
            clip_axis(o, d, -box[12+axis], box[12+axis], start, end);
        }
    }
    starts[tid] = end > start ? start : 0.0;
    ends[tid] = end > start ? end : 0.0;
}

extern "C" __global__ void photon_sample(
    const double *starts, const double *ends, const double *mu,
    const double *rates, const int *materials, const int *offsets,
    const double *tau, const double *draws, long long n, int r, int g, int cols,
    double *distance, long long *layer, long long *channel, long long *element) {
    const long long ray = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= n) return;
    distance[ray] = PHOTON_INF;
    layer[ray] = channel[ray] = element[ray] = -1;
    double previous_start = -PHOTON_INF, cumulative = 0.0;
    int previous_region = -1;
    for (int visit = 0; visit < r; ++visit) {
        double first_start = PHOTON_INF;
        int region = -1;
        for (int j = 0; j < r; ++j) {
            const double lo = starts[ray*r+j], hi = ends[ray*r+j];
            if (!(hi > lo && mu[ray*g+materials[j]] > 0.0)) continue;
            if (lo < previous_start || (lo == previous_start && j <= previous_region)) continue;
            if (lo < first_start) { first_start = lo; region = j; }
        }
        if (region < 0) return;
        const int material = materials[region];
        const double rate = mu[ray*g+material];
        const double before = cumulative;
        cumulative += rate * (ends[ray*r+region] - first_start);
        if (cumulative > tau[ray]) {
            layer[ray] = region;
            distance[ray] = first_start + (tau[ray] - before) / rate;
            const int begin = offsets[material], end = offsets[material+1];
            double total = 0.0;
            for (int j = begin; j < end; ++j) total += rates[ray*cols+j];
            const double threshold = fmin(draws[ray] * total, nextafter(total, 0.0));
            double cdf = 0.0;
            for (int j = begin; j < end; ++j) {
                cdf += rates[ray*cols+j];
                if (cdf > threshold) {
                    element[ray] = (j-begin) / 5;
                    channel[ray] = (j-begin) % 5;
                    return;
                }
            }
            return;
        }
        previous_start = first_start;
        previous_region = region;
    }
}
"""


@cache
def _kernels(cp):
    from ..._backend import exact_rawkernel_options

    module = cp.RawModule(code=SOURCE, options=exact_rawkernel_options(cp))
    return module.get_function("photon_intervals"), module.get_function("photon_sample")


def _material_rates(compositions, energies):
    from .photon_navigation import _coefficients

    unique, indices, rates = {}, [], []
    for composition in compositions:
        key = tuple((element, float(density)) for element, density in composition)
        if key not in unique:
            unique[key] = len(rates)
            rates.append(_coefficients(composition, energies))
        indices.append(unique[key])
    mu = np.stack([rate.sum(axis=1) for rate in rates], axis=1)
    if not np.isfinite(mu).all():
        raise ValueError("total photon rates must be finite")
    offsets = np.cumsum([0, *[rate.shape[1] for rate in rates]], dtype=np.int32)
    return np.concatenate(rates, axis=1), mu, np.asarray(indices, dtype=np.int32), offsets


def fused_first_interactions(
    cp, origins, directions, energies, uniforms, layers, boxes, width_ang, height_ang, *, threads
):
    """Validated host inputs to two fused kernels, with device-resident outputs."""
    n, r = len(energies), len(layers) + len(boxes)
    result = {"distance_ang": cp.empty(n)}
    result.update({key: cp.empty(n, dtype=cp.int64) for key in ("layer", "channel", "element")})
    if not n or not r:
        result["distance_ang"].fill(np.inf)
        for key in ("layer", "channel", "element"):
            result[key].fill(-1)
        return result
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    rates, mu, materials, offsets = _material_rates(compositions, energies)
    bounds = np.asarray([(top, bottom) for top, bottom, _ in layers], dtype=float)
    box_data = np.asarray(
        [[*box.center_ang, *np.asarray(box.axes).ravel(), *box.half_extent_ang] for box in boxes],
        dtype=float,
    )
    starts, ends = cp.empty((n, r)), cp.empty((n, r))
    intervals, sample = _kernels(cp)
    intervals(
        ((n * r + threads - 1) // threads,),
        (threads,),
        (
            cp.asarray(np.ascontiguousarray(origins)),
            cp.asarray(np.ascontiguousarray(directions)),
            cp.asarray(bounds),
            cp.asarray(box_data),
            np.int64(n),
            np.int32(len(layers)),
            np.int32(r),
            np.float64(np.inf if width_ang is None else width_ang / 2),
            np.float64(np.inf if height_ang is None else height_ang / 2),
            starts,
            ends,
        ),
    )
    sample(
        ((n + threads - 1) // threads,),
        (threads,),
        (
            starts,
            ends,
            cp.asarray(mu),
            cp.asarray(rates),
            cp.asarray(materials),
            cp.asarray(offsets),
            cp.asarray(-np.log1p(-uniforms[:, 0])),
            cp.asarray(np.ascontiguousarray(uniforms[:, 1])),
            np.int64(n),
            np.int32(r),
            np.int32(mu.shape[1]),
            np.int32(rates.shape[1]),
            result["distance_ang"],
            result["layer"],
            result["channel"],
            result["element"],
        ),
    )
    return result
