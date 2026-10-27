"""CUDA twin of the PENELOPE pair reference, including its threshold convention.

Validation: pair-production-sampling
"""

from functools import cache

from ._photon_scatter_cuda_kernel import COMMON_SOURCE

SOURCE = (
    COMMON_SOURCE
    + r"""
__device__ void photon_pair_phi(double eps, double kappa, double radius,
                                double coulomb, double az, double &p1, double &p2) {
    const double b = radius/(2.0*kappa*eps*(1.0-eps));
    const double angle = atan(1.0/b), logarithm = log1p(b*b);
    const double tail = 4.0-4.0*b*angle-3.0*log1p(1.0/(b*b));
    const double g1 = 7.0/3.0-2.0*logarithm-6.0*b*angle-b*b*tail;
    const double g2 = 11.0/6.0-2.0*logarithm-3.0*b*angle+0.5*b*b*tail;
    const double x = 2.0/kappa;
    const double f0 = (-1.774-12.10*az+11.18*az*az)*sqrt(x)
        +(8.523+73.26*az-44.41*az*az)*x
        -(13.52+121.1*az-96.41*az*az)*pow(x,1.5)
        +(8.946+62.05*az-63.41*az*az)*x*x;
    const double g0 = 4.0*log(radius)-4.0*coulomb+f0;
    p1 = fmax(g1+g0,0.0); p2 = fmax(g2+g0,0.0);
}

__device__ void photon_pair_direction(const double *incident, double kinetic,
                                      double u, double phi, double rest, double *out) {
    const double beta = sqrt(kinetic*(kinetic+2.0*rest))/(kinetic+rest);
    const double x = 2.0*u-1.0, cosine = (x+beta)/(x*beta+1.0);
    const double length = photon_norm(incident);
    double ray[3], reference[3] = {0,0,0}, tangent[3], other[3];
    for (int j = 0; j < 3; ++j) ray[j] = incident[j]/length;
    reference[fabs(ray[0]) < 0.9 ? 0 : 1] = 1.0;
    photon_cross(ray,reference,tangent);
    const double norm = photon_norm(tangent);
    for (int j = 0; j < 3; ++j) tangent[j] /= norm;
    photon_cross(ray,tangent,other);
    const double sine = sqrt(fmax(0.0,1.0-cosine*cosine));
    for (int j = 0; j < 3; ++j)
        out[j] = cosine*ray[j] + sine*(cos(phi)*tangent[j]+sin(phi)*other[j]);
    const double final_norm = photon_norm(out);
    for (int j = 0; j < 3; ++j) out[j] /= final_norm;
}

extern "C" __global__ void photon_pair(
    const double *energy, const double *rays, const double *uniforms,
    long long n, int width, int max_rejections, double rest, double radius,
    double coulomb, double az, double pi, double *out, long long *used, int *statuses) {
    const long long row = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= n) return;
    double *result = out+8*row;
    for (int j = 0; j < 8; ++j) result[j] = __longlong_as_double(0x7ff8000000000000LL);
    const double e = energy[row], kappa = e/rest, lo = 1.0/kappa, half_width = 0.5-lo;
    const double *u = uniforms+row*width;
    int at = 0, status = 2;
    double ph1,ph2,eps = 0.5;
    photon_pair_phi(0.5,kappa,radius,coulomb,az,ph1,ph2);
    const double w1 = 2.0/3.0*half_width*half_width*ph1, w2 = ph2;
    bool accepted = false;
    if (w1+w2 <= 0.0) {
        eps = lo+2.0*half_width*photon_draw(u,width,at,status);
        accepted = status != 1;
    } else {
        const double probability = w1/(w1+w2);
        for (int trial = 0; trial < max_rejections; ++trial) {
            const bool branch = photon_draw(u,width,at,status) < probability;
            const double xi = 2.0*photon_draw(u,width,at,status)-1.0;
            eps = 0.5+half_width*(branch ? cbrt(xi) : xi);
            double p1,p2;
            photon_pair_phi(eps,kappa,radius,coulomb,az,p1,p2);
            const double accept = branch ? p1/ph1 : p2/ph2;
            const double draw = photon_draw(u,width,at,status);
            if (status == 1) break;
            if (draw <= accept) { accepted = true; break; }
        }
    }
    if (accepted) {
        result[0] = fmax(eps*e-rest,0.0);
        result[1] = e-2.0*rest-result[0];
        const double um = photon_draw(u,width,at,status);
        const double pm = 2.0*pi*photon_draw(u,width,at,status);
        const double up = photon_draw(u,width,at,status);
        const double pp = 2.0*pi*photon_draw(u,width,at,status);
        if (status != 1) {
            photon_pair_direction(rays+3*row,result[0],um,pm,rest,result+2);
            photon_pair_direction(rays+3*row,result[1],up,pp,rest,result+5);
            status = 0;
        }
    }
    used[row] = at; statuses[row] = status;
}
"""
)


@cache
def pair_kernel(cp):
    return cp.RawKernel(SOURCE, "photon_pair", options=("--std=c++17", "--fmad=false"))
