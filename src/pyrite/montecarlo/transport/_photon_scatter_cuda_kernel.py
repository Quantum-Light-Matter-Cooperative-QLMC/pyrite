"""Binary64 CUDA twins of the explicit-draw CPU photon scatter samplers.

Same clipped lin-lin EPDL factors, proposal/rejection laws, draw order and
least-aligned-axis rotation as photon_scattering/photon_sampling. No RNG.
Validation: photon-compton-scattering, photon-rayleigh-scattering
"""

from functools import cache

COMMON_SOURCE = r"""
__device__ double photon_draw(const double *u, int width, int &at, int &status) {
    if (at >= width) { status = 1; return 0.0; }
    return u[at++];
}

__device__ int photon_upper_bound(const double *x, int size, double query) {
    int lo = 0, hi = size;
    while (lo < hi) {
        const int mid = lo + (hi-lo)/2;
        if (x[mid] <= query) lo = mid+1; else hi = mid;
    }
    return lo;
}

__device__ double photon_interp(const double *x, const double *y, int size, double q) {
    if (q < x[0]) return y[0];
    const int hi = photon_upper_bound(x, size, q);
    if (hi >= size) return y[size-1];
    const int lo = hi-1;
    return y[lo] + (q-x[lo])*((y[hi]-y[lo])/(x[hi]-x[lo]));
}

__device__ void photon_cross(const double *a, const double *b, double *out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ double photon_norm(const double *v) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

__device__ void photon_rotate(const double *ray, double delta, double phi,
                              double *out, double *transverse) {
    int axis = 0;
    for (int j = 1; j < 3; ++j) if (fabs(ray[j]) < fabs(ray[axis])) axis = j;
    double reference[3] = {0, 0, 0}, tangent[3], other[3];
    reference[axis] = 1.0;
    photon_cross(ray, reference, tangent);
    const double length = photon_norm(tangent);
    for (int j = 0; j < 3; ++j) tangent[j] /= length;
    photon_cross(ray, tangent, other);
    const double c = cos(phi), s = sin(phi), polar = sqrt(delta*(2.0-delta));
    for (int j = 0; j < 3; ++j) {
        transverse[j] = c*tangent[j] + s*other[j];
        out[j] = ray[j]*(1.0-delta) + transverse[j]*polar;
    }
}
"""

SOURCE = (
    COMMON_SOURCE
    + r"""
extern "C" __global__ void photon_scatter(
    const double *energy, const double *rays, const double *uniforms,
    const double *x, const double *factor, long long n, int width, int size,
    int coherent, int max_rejections, double rest, double hc, double pi,
    double *out, long long *draws_used, int *statuses) {
    const long long row = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= n) return;
    double *result = out + 9*row;
    for (int j = 0; j < 9; ++j) result[j] = __longlong_as_double(0x7ff8000000000000LL);
    const double e = energy[row], limit = e/hc, alpha = e/rest;
    const double *ray = rays + 3*row, *u = uniforms + row*width;
    const int stop = photon_upper_bound(x, size, limit);
    const bool partial = x[stop-1] < limit;
    const int intervals = stop-1 + (partial ? 1 : 0);
    const double endpoint = photon_interp(x, factor, size, limit);
    double total = 0.0;
    if (coherent) {
        for (int j = 0; j < intervals; ++j) {
            const double lo = x[j], hi = j+1 < stop ? x[j+1] : limit;
            const double high_factor = j+1 < stop ? factor[j+1] : endpoint;
            const double ceiling = fmax(factor[j], high_factor);
            total += (hi-lo)*(hi+lo)*(ceiling*ceiling);
        }
    } else {
        for (int j = 0; j < stop; ++j) total = fmax(total, factor[j]);
        total = fmax(total, endpoint);
    }
    int at = 0, status = total > 0.0 && isfinite(total) ? 2 : 3;
    if (!(total > 0.0) || !isfinite(total)) { draws_used[row] = 0; statuses[row] = status; return; }
    for (int trial = 0; trial < max_rejections; ++trial) {
        double delta, epsilon, acceptance;
        if (coherent) {
            const double threshold = fmin(photon_draw(u,width,at,status)*total, nextafter(total,0.0));
            double cumulative = 0.0, lo = 0.0, hi = 0.0, ceiling = 0.0;
            for (int j = 0; j < intervals; ++j) {
                lo = x[j]; hi = j+1 < stop ? x[j+1] : limit;
                const double high_factor = j+1 < stop ? factor[j+1] : endpoint;
                const double maximum = fmax(factor[j],high_factor);
                ceiling = maximum*maximum;
                cumulative += (hi-lo)*(hi+lo)*ceiling;
                if (cumulative > threshold) break;
            }
            const double square_width = (hi-lo)*(hi+lo);
            const double value = sqrt(lo*lo + photon_draw(u,width,at,status)*square_width);
            delta = fmin(2.0, fmax(0.0, 2.0*(value/limit)*(value/limit)));
            const double form = photon_interp(x,factor,size,value);
            acceptance = (form*form/ceiling)*(1.0+(1.0-delta)*(1.0-delta))/2.0;
            epsilon = 1.0;
        } else {
            const double logarithm = log1p(2.0*alpha), epsilon0 = 1.0/(1.0+2.0*alpha);
            const double spread = -expm1(-2.0*logarithm);
            const double branch = photon_draw(u,width,at,status);
            const double v = photon_draw(u,width,at,status);
            if (branch < logarithm/(logarithm+spread/2.0)) {
                const double exponent = logarithm*v;
                epsilon = exp(-exponent); delta = expm1(exponent)/alpha;
            } else {
                epsilon = sqrt(epsilon0*epsilon0 + spread*v);
                delta = spread*(1.0-v)/((1.0+epsilon)*epsilon*alpha);
            }
            delta = fmin(2.0,fmax(0.0,delta));
            acceptance = 1.0-epsilon*delta*(2.0-delta)/(1.0+epsilon*epsilon);
            acceptance *= photon_interp(x,factor,size,limit*sqrt(delta/2.0))/total;
        }
        const double accept = photon_draw(u,width,at,status);
        if (status == 1) break;
        if (accept < acceptance) {
            const double phi = 2.0*pi*photon_draw(u,width,at,status);
            if (status == 1) break;
            double outgoing[3], transverse[3], electron[3];
            photon_rotate(ray,delta,phi,outgoing,transverse);
            const double denominator = 1.0+alpha*delta;
            result[0] = coherent ? e : e/denominator;
            result[4] = coherent ? 0.0 : e*(alpha*delta/denominator);
            result[8] = delta;
            for (int j = 0; j < 3; ++j) {
                result[1+j] = outgoing[j];
                electron[j] = (result[4]+result[0]*delta)*ray[j]
                              - result[0]*sqrt(delta*(2.0-delta))*transverse[j];
            }
            if (!coherent) {
                const double norm = photon_norm(electron);
                for (int j = 0; j < 3; ++j) result[5+j] = norm > 0.0 ? electron[j]/norm : ray[j];
            }
            status = 0; break;
        }
    }
    draws_used[row] = at; statuses[row] = status;
}
"""
)


@cache
def scatter_kernel(cp):
    return cp.RawKernel(SOURCE, "photon_scatter", options=("--std=c++17", "--fmad=false"))
