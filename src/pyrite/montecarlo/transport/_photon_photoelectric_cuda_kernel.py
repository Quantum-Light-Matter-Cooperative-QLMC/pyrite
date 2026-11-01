"""CUDA subshell, Sauter photoelectron and depth-first EADL event sampling.

Explicit draw prefixes and output capacity have distinct retry statuses. Neither
budget can silently drop an emission or force an accepted angular proposal.
Validation: photon-photoelectric-sampling, eadl-sampled-relaxation
"""

from functools import cache

from ._photon_scatter_cuda_kernel import COMMON_SOURCE

SOURCE = (
    COMMON_SOURCE
    + r"""
extern "C" __global__ void photon_photoelectric(
    const double *energy, const double *rays, const double *uniforms, const double *rates,
    const int *primary_shells, const int *designators, const double *binding,
    const int *offsets, const double *probability, const int *first, const int *second,
    long long n, int width, int ns, int capacity, int max_rejections,
    double cutoff, double rest, double pi, double *electron, int *primary,
    int *transitions, int *terminal, double *directions, int *counts,
    long long *used, int *statuses) {
    const long long row = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= n) return;
    const double nan = __longlong_as_double(0x7ff8000000000000LL);
    int at = 0, nt = 0, nh = 0, flag = 2;
    primary[row] = -1;
    for (int j = 0; j < 4; ++j) electron[4*row+j] = nan;
    for (int j = 0; j < capacity; ++j) {
        transitions[row*capacity+j] = -1;
        for (int k = 0; k < 3; ++k) directions[(row*capacity+j)*3+k] = nan;
    }
    for (int j = 0; j <= capacity; ++j) terminal[row*(capacity+1)+j] = -1;
    #define PE_DONE(code) { used[row]=at; statuses[row]=(code); counts[2*row]=nt; counts[2*row+1]=nh; return; }
    const double *u = uniforms+row*width;
    double total = 0.0;
    for (int j = 0; j < ns; ++j) total += rates[row*ns+j];
    if (!(total > 0.0) || !isfinite(total)) PE_DONE(3);
    const double threshold = fmin(photon_draw(u,width,at,flag)*total,nextafter(total,0.0));
    if (flag == 1) PE_DONE(1);
    double cumulative = 0.0;
    int shell = -1;
    for (int j = 0; j < ns; ++j) {
        cumulative += rates[row*ns+j];
        if (cumulative > threshold) { shell=primary_shells[j]; break; }
    }
    if (shell < 0) PE_DONE(3);
    primary[row] = designators[shell];
    const double kinetic = energy[row]-binding[shell];
    electron[4*row] = kinetic;
    if (kinetic > 0.0) {
        const double reduced = kinetic/rest, gamma = 1.0+reduced;
        const double beta = sqrt(reduced*(reduced+2.0))/gamma;
        const double a = 1.0/(beta*(gamma*gamma)*(1.0+beta));
        const double c = beta*gamma*reduced*(reduced-1.0)/2.0;
        const double maximum = 2.0/(a+2.0);
        double delta = 0.0;
        bool accepted = false;
        for (int trial = 0; trial < max_rejections; ++trial) {
            const double w = maximum*sqrt(photon_draw(u,width,at,flag));
            delta = fmin(2.0,fmax(0.0,a*w/(1.0-w)));
            const double acceptance = (2.0-delta)*(1.0-w/(1.0+c*a))/2.0;
            const double accept = photon_draw(u,width,at,flag);
            if (flag == 1) PE_DONE(1);
            if (accept < acceptance) { accepted=true; break; }
        }
        if (!accepted) PE_DONE(2);
        const double phi = 2.0*pi*photon_draw(u,width,at,flag);
        if (flag == 1) PE_DONE(1);
        double transverse[3];
        photon_rotate(rays+3*row,delta,phi,electron+4*row+1,transverse);
    } else {
        for (int j = 0; j < 3; ++j) electron[4*row+1+j] = rays[3*row+j];
    }
    // Strictly later daughter shells bound the depth-first stack by shell count.
    // The host validates <=64 shells; this is workspace, not a transition cap.
    int stack[64], pending = 1;
    stack[0] = shell;
    while (pending) {
        shell = stack[--pending];
        const int begin = offsets[shell], end = offsets[shell+1];
        if (binding[shell] <= cutoff || begin == end) {
            if (nh >= capacity+1) PE_DONE(4);
            terminal[row*(capacity+1)+nh++] = shell;
            continue;
        }
        if (nt >= capacity) PE_DONE(4);
        const double choice = fmin(photon_draw(u,width,at,flag)*probability[end-1],
                                   nextafter(probability[end-1],0.0));
        if (flag == 1) PE_DONE(1);
        int lo = begin, hi = end;
        while (lo < hi) {
            const int mid = lo+(hi-lo)/2;
            if (probability[mid] <= choice) lo=mid+1; else hi=mid;
        }
        transitions[row*capacity+nt++] = lo;
        if (second[lo] >= 0) stack[pending++] = second[lo];
        stack[pending++] = first[lo];
    }
    // The CPU queue assigns isotropic directions only after the full cascade.
    for (int j = 0; j < nt; ++j) {
        const double cosine = 1.0-2.0*photon_draw(u,width,at,flag);
        const double phi = 2.0*pi*photon_draw(u,width,at,flag);
        if (flag == 1) PE_DONE(1);
        const double sine = sqrt(fmax(0.0,1.0-cosine*cosine));
        double *out = directions+(row*capacity+j)*3;
        out[0]=sine*cos(phi); out[1]=sine*sin(phi); out[2]=cosine;
    }
    PE_DONE(0);
    #undef PE_DONE
}
"""
)


@cache
def photoelectric_kernel(cp):
    from ..._backend import exact_rawkernel_options

    return cp.RawKernel(SOURCE, "photon_photoelectric", options=exact_rawkernel_options(cp))
