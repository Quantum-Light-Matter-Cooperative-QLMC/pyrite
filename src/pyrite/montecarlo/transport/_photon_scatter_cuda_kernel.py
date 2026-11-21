"""Binary64 CUDA twins of the explicit-draw CPU photon scatter samplers.

Same clipped lin-lin EPDL factors, proposal/rejection laws, draw order,
least-aligned-axis rotation and impulse-approximation Compton energy
(``photon_scattering._doppler_energy``: PENELOPE-2024 §2.3, Eqs. 2.38–2.56,
term-for-term in the same evaluation order) as photon_scattering/photon_sampling.
No RNG.
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

#: Workspace bound on occupied EADL subshells per element (host-checked).
MAX_COMPTON_SHELLS = 64

COMPTON_SOURCE = r"""
#define MAX_COMPTON_SHELLS 64
#define COMPTON_INF __longlong_as_double(0x7ff0000000000000LL)
#define COMPTON_NAN __longlong_as_double(0x7ff8000000000000LL)

__device__ double compton_cdf(double j0, double p) {
    const double d1 = sqrt(0.5), d2 = sqrt(2.0);
    if (p < 0.0) { const double a = d1 - d2*j0*p; return 0.5*exp(0.5 - a*a); }
    const double a = d1 + d2*j0*p;
    return 1.0 - 0.5*exp(0.5 - a*a);
}

__device__ double compton_inverse(double j0, double level) {
    const double d1 = sqrt(0.5), d2 = sqrt(2.0);
    if (level < 0.5) {
        if (level <= 0.0) return -COMPTON_INF;
        return (d1 - sqrt(0.5 - log(2.0*level)))/(d2*j0);
    }
    const double tail = 1.0 - level;
    if (tail <= 0.0) return COMPTON_INF;
    return (sqrt(0.5 - log(2.0*tail)) - d1)/(d2*j0);
}

__device__ double compton_pz(double tau, double tau_c, double delta) {
    const double rest = 1.0 - tau;
    return (tau - tau_c)/(tau_c*sqrt(rest*rest + 2.0*tau*delta));
}

__device__ double compton_ratio(double p, double tau_c, double delta, double tau_max) {
    const double t = p*p;
    const double a = 1.0 - t*tau_c*tau_c;
    const double b = tau_c*(1.0 - t*tau_c*(1.0 - delta));
    const double c = tau_c*tau_c*(1.0 - t);
    const double q = b + copysign(sqrt(fmax(b*b - a*c, 0.0)), b);
    const double roots[2] = {q != 0.0 ? c/q : COMPTON_NAN, a != 0.0 ? q/a : COMPTON_NAN};
    double best = fmin(tau_c, tau_max), miss = COMPTON_INF;
    for (int k = 0; k < 2; ++k) {
        if (!isfinite(roots[k])) continue;
        const double root = fmin(fmax(roots[k], 0.0), tau_max);
        const double error = fabs(compton_pz(root, tau_c, delta) - p);
        if (error < miss) { best = root; miss = error; }
    }
    return best;
}

// Shell proposal support (photon_scattering._shell_support): returns the
// cumulative weight total; zero means no final state at this angle.
__device__ double compton_support(
    double e, double alpha, double delta, double fine, const double *shell_f,
    const double *shell_u, const double *shell_j0, int n_shells,
    double *cumulative, double *limits) {
    const double tau_c = 1.0/(1.0 + alpha*delta), lowest = -1.0/fine;
    double running = 0.0;
    for (int i = 0; i < n_shells; ++i) {
        const double fraction = shell_u[i]/e;
        const double scale = tau_c*sqrt(fraction*fraction + 2.0*(1.0 - fraction)*delta);
        const double highest = scale > 0.0 ? (alpha*delta*tau_c - fraction)/scale/fine : lowest;
        limits[i] = highest;
        if (fraction < 1.0 && highest > lowest)
            running += shell_f[i]*(compton_cdf(shell_j0[i], highest) - compton_cdf(shell_j0[i], lowest));
        cumulative[i] = running;
    }
    return running;
}

// photon_scattering._doppler_energy: true with photon energy and shell index;
// otherwise status is 1 (draws exhausted) or 2 (rejection budget).
__device__ bool compton_doppler(
    const double *u, int width, int &at, int &status, double e, double alpha,
    double delta, double fine, const double *shell_u, const double *shell_j0,
    int n_shells, const double *cumulative, const double *limits, double running,
    int max_rejections, double &photon, int &shell) {
    const double tau_c = 1.0/(1.0 + alpha*delta), lowest = -1.0/fine;
    const double transfer = sqrt((alpha*delta*tau_c)*(alpha*delta*tau_c) + 2.0*tau_c*delta);
    const double slope = transfer + tau_c*delta*(1.0 - alpha*tau_c)/transfer;
    const double ceiling = 1.0 + 0.2*fabs(slope);
    for (int trial = 0; trial < max_rejections; ++trial) {
        const double threshold = fmin(photon_draw(u,width,at,status)*running, nextafter(running,0.0));
        shell = 0;
        while (shell < n_shells-1 && cumulative[shell] <= threshold) ++shell;
        const double j0 = shell_j0[shell], highest = limits[shell];
        const double low = compton_cdf(j0, lowest), high = compton_cdf(j0, highest);
        double momentum = compton_inverse(j0, low + photon_draw(u,width,at,status)*(high - low));
        momentum = fmin(fmax(momentum, lowest), highest)*fine;
        const double factor = fmax(1.0 + slope*fmin(fmax(momentum, -0.2), 0.2), 0.0);
        const double accept = photon_draw(u,width,at,status);
        if (status == 1) return false;
        if (accept*ceiling < factor) {
            const double tau_max = 1.0 - shell_u[shell]/e;
            photon = e*compton_ratio(momentum, tau_c, delta, tau_max);
            return true;
        }
    }
    status = 2;
    return false;
}
"""

SOURCE = (
    COMMON_SOURCE
    + COMPTON_SOURCE
    + r"""
extern "C" __global__ void photon_scatter(
    const double *energy, const double *rays, const double *uniforms,
    const double *x, const double *factor, const double *shell_f,
    const double *shell_u, const double *shell_j0, const double *shell_designator,
    long long n, int width, int size, int n_shells,
    int coherent, int max_rejections, double rest, double hc, double pi, double fine,
    double *out, long long *draws_used, int *statuses) {
    const long long row = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= n) return;
    double *result = out + 11*row;
    for (int j = 0; j < 11; ++j) result[j] = __longlong_as_double(0x7ff8000000000000LL);
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
    double cumulative[MAX_COMPTON_SHELLS], limits[MAX_COMPTON_SHELLS];
    const bool supported = total > 0.0 && isfinite(total) && (coherent
        || compton_support(e, alpha, 2.0, fine, shell_f, shell_u, shell_j0, n_shells,
                           cumulative, limits) > 0.0);
    int at = 0, status = supported ? 2 : 3;
    if (!supported) { draws_used[row] = 0; statuses[row] = status; return; }
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
            double running = 0.0;
            if (!coherent) {
                running = compton_support(e, alpha, delta, fine, shell_f, shell_u, shell_j0,
                                          n_shells, cumulative, limits);
                if (!(running > 0.0)) continue;
            }
            const double phi = 2.0*pi*photon_draw(u,width,at,status);
            if (status == 1) break;
            double outgoing[3], transverse[3], electron[3];
            photon_rotate(ray,delta,phi,outgoing,transverse);
            for (int j = 0; j < 3; ++j) result[1+j] = outgoing[j];
            result[8] = delta;
            if (coherent) {
                result[0] = e; result[4] = 0.0; result[9] = 0.0; result[10] = 0.0;
                status = 0; break;
            }
            double photon; int shell;
            if (!compton_doppler(u, width, at, status, e, alpha, delta, fine, shell_u, shell_j0,
                                 n_shells, cumulative, limits, running, max_rejections,
                                 photon, shell)) {
                for (int j = 0; j < 11; ++j) result[j] = __longlong_as_double(0x7ff8000000000000LL);
                break;
            }
            result[0] = photon;
            result[9] = shell_u[shell];
            result[10] = shell_designator[shell];
            result[4] = fmax(e - photon - shell_u[shell], 0.0);
            for (int j = 0; j < 3; ++j)
                electron[j] = (e - photon*(1.0-delta))*ray[j]
                              - photon*sqrt(delta*(2.0-delta))*transverse[j];
            const double norm = photon_norm(electron);
            for (int j = 0; j < 3; ++j) result[5+j] = norm > 0.0 ? electron[j]/norm : ray[j];
            status = 0; break;
        }
    }
    draws_used[row] = at; statuses[row] = status;
}
"""
)


@cache
def scatter_kernel(cp):
    from ..._backend import exact_rawkernel_options

    return cp.RawKernel(SOURCE, "photon_scatter", options=exact_rawkernel_options(cp))
