# Broad-line Chebyshev interpolation bound

Validation: `cxr-line-broad-interpolation`. This page records a fresh-context
verification of the experimental broad-line split. The implementation-context
derivation is in
[`cxr-line-fourier-reduction`](cxr-line-fourier-reduction.md#broad-line-split).
The verifier did not write the implementation. The verifier changed only this
page. The scratch check is `scratch/broad_verify.py` in the issue #1192
worktree at HEAD `93908a08b`; it is not committed.

## Verdict

- **Claim**: `cxr-line-broad-interpolation` — `montecarlo/spectrum/cxr/_bin_broad.py::broad_bin_mean_error_bound`, `::broad_bin_mean_reduce`, `montecarlo/spectrum/cxr/_bin_fourier.py::fourier_bin_mean_reduce`, `montecarlo/spectrum/cxr/_fourier_cuda.py::cuda_fourier_bin_mean_reduce` — $\operatorname{sinc}^2$ triangle transform; Chebyshev root-node remainder; exactness of four-point Gauss–Legendre quadrature through degree seven
- **Filters**: units pass; limits pass ($W_0\to\infty$, $h\to0$); signs and conventions pass (`np.sinc` argument $(E-E_l)/W_l$ with $W_l=\pi/a_l$)
- **Re-derivation**: `matches` — every factor of $\left[2(\pi L/(2W_0))^{64}/64!+127\cdot2(\pi h/W_0)^8/8!\right]\sum_l\lvert w_l\rvert$ is reproduced. The bound is valid but loose (see [Tightness](#tightness))
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/radiation-physics/cxr-line-broad-interpolation.md`
- **Suggested ledger change**: `unverified` → `rederived`. Point the row's derivation link at this page and add it to the radiation-physics toctree. Never `signed-off` from this context

## Independent derivation

### Line profile and its spectrum

With `np.sinc(x)` $=\sin(\pi x)/(\pi x)$, the unit-peak line is

$$
f_l(E)=\operatorname{sinc}^2\!\left(\frac{a_l(E-E_l)}{\pi}\right)
=\frac{\sin^2(a_l x)}{(a_l x)^2},\qquad x=E-E_l .
$$

Its first zero is at $a_l x=\pi$, so $W_l=\pi/a_l$. Write
$\sin(ax)/(ax)=(2a)^{-1}\int_{-a}^{a}e^{itx}\,dt$. Squaring convolves two
boxes, so

$$
f_l(x)=\int_{-2a_l}^{2a_l}\Lambda_l(t)\,e^{itx}\,dt,
\qquad
\Lambda_l(t)=\frac{2a_l-\lvert t\rvert}{4a_l^2}\ge0,
\qquad
\int\Lambda_l\,dt=1=f_l(0).
$$

### Derivative bound

Differentiating under the integral,

$$
\lvert f_l^{(m)}(x)\rvert\le\int\lvert t\rvert^m\Lambda_l(t)\,dt
=\frac{2(2a_l)^m}{(m+1)(m+2)}\le(2a_l)^m=\left(\frac{2\pi}{W_l}\right)^m
\le\left(\frac{2\pi}{W_0}\right)^m .
$$

The bin mean is
$g_l(c)=h^{-1}\int_{c-h/2}^{c+h/2}f_l\,dE=\int\Lambda_l(t)\,\frac{\sin(th/2)}{th/2}\,e^{it(c-E_l)}\,dt$.
The extra factor has modulus at most one, so $g_l^{(m)}$ obeys the same bound.
For the weighted sum $g=\sum_l w_lg_l$,
$\lvert g^{(m)}\rvert\le(2\pi/W_0)^m\sum_l\lvert w_l\rvert$.

### Interpolation remainder

Let the $r$ Chebyshev roots be mapped onto the bin-centre interval of length
$L=(n_{\rm bins}-1)h$. For real smooth $g$, the Cauchy remainder is
$g(c)-I_rg(c)=g^{(r)}(\xi)\,\omega_r(c)/r!$, where $\omega_r$ is the monic node
polynomial. On $[-1,1]$ that polynomial is $2^{1-r}T_r$. Scaling by $L/2$ gives

$$
\sup\lvert\omega_r\rvert=\left(\frac L2\right)^r2^{1-r}=2\left(\frac L4\right)^r .
$$

Therefore

$$
\lvert g-I_rg\rvert\le\frac{2}{r!}\left(\frac{2\pi}{W_0}\cdot\frac L4\right)^r\sum_l\lvert w_l\rvert
=\epsilon_I\sum_l\lvert w_l\rvert,
\qquad
\epsilon_I=\frac{2}{r!}\left(\frac{\pi L}{2W_0}\right)^r .
$$

### Bin-mean quadrature error

At each node, the bin mean is replaced by four-point Gauss–Legendre quadrature
with normalized weights $\tfrac12q_i>0$, $\sum_i\tfrac12q_i=1$. This rule is
exact for polynomials of degree at most seven. Expand $f_l$ about the bin
centre through degree seven. The Taylor remainder on the bin satisfies

$$
\lvert R\rvert\le\frac{(2\pi/W_0)^8(h/2)^8}{8!}=\frac{(\pi h/W_0)^8}{8!}.
$$

Both the exact mean and the quadrature integrate the polynomial part exactly.
Each applies a positive normalized functional to $R$, so the difference is at
most $2\lVert R\rVert_\infty$:

$$
\epsilon_Q=\frac{2}{8!}\left(\frac{\pi h}{W_0}\right)^8
$$

per unit $\sum_l\lvert w_l\rvert$.

### Propagation through the interpolant

The interpolant is linear in its node values. The total error is
$(g-I_rg)+I_re$, where $e_i$ are the node errors with
$\lvert e_i\rvert\le\epsilon_Q\sum_l\lvert w_l\rvert$. At the roots
$x_i=\cos(\pi(i+\tfrac12)/r)$, discrete orthogonality gives
$I_re=\sum_{k<r}c_kT_k$ with

$$
c_0=\frac1r\sum_ie_i,\qquad c_k=\frac2r\sum_ie_iT_k(x_i)\quad(k\ge1).
$$

Thus $\lvert c_0\rvert\le\max_i\lvert e_i\rvert$ and
$\lvert c_k\rvert\le2\max_i\lvert e_i\rvert$. On $[-1,1]$, $\lvert T_k\rvert\le1$.
Hence $\lVert I_re\rVert_\infty\le(2r-1)\max_i\lvert e_i\rvert$, which is $127$ for
$r=64$. This bounds the Lebesgue constant from above. The bound holds because
every target bin centre maps into $[-1,1]$; no extrapolation occurs. The true
Lebesgue constant of 64 Chebyshev roots, computed numerically, is $3.61$.
The factor $127$ is valid but conservative.

Combining,

$$
\boxed{\lvert\Delta\bar\rho_k\rvert\le\left[\frac{2}{64!}\left(\frac{\pi L}{2W_0}\right)^{64}
+127\cdot\frac{2}{8!}\left(\frac{\pi h}{W_0}\right)^8\right]\sum_l\lvert w_l\rvert}
$$

in exact arithmetic. This is identical to the ledger claim.

## Filters

| Check | Result |
|---|---|
| Units | $L/W_0$ and $h/W_0$ are dimensionless (eV/eV). The bound carries the units of $w_l\times$(unit-peak profile), the same as the bin mean |
| $W_0\to\infty$ | Both terms vanish, as $W_0^{-64}$ and $W_0^{-8}$ |
| $h\to0$ | The quadrature term vanishes as $h^8$. The interpolation term depends on $L=(n_{\rm bins}-1)h$ only |
| Signed weights | Linearity bounds the result by $\sum_l\lvert w_l\rvert$; cancellation is not assumed |
| Remote resonances | The derivative bound is uniform in $E_l$, so any resonance position is covered |
| Width selection | The code selects `a <= pi / broad_width_eV`, which is $W_l\ge W_0$. The bound uses the actual minimum broad width, which is at least $W_0$ and gives a bound no larger |

## Source-to-code comparison

Implementation read after the derivation above.

| Derivation | Code (`_bin_broad.py`) | Agreement |
|---|---|---|
| $W_l=\pi/a_l$, argument $(E-E_l)/W_l$ | `width = np.pi / a_width`; `xp.sinc(phase) ** 2` with `phase = (centre + 0.5*h*x - E_res)/width` | yes |
| $r=64$ roots $\cos(\pi(i+\tfrac12)/r)$ mapped to $[c_0,c_0+L]$ | `xn = cos(pi*(arange(64)+0.5)/64)`; `centres = c0 + 0.5*span*(1+xn)` | yes |
| Normalized four-point Gauss–Legendre mean | `leggauss(4)`, weights `0.5*q`, nodes `centre + 0.5*h*x` | yes |
| Degree-63 interpolant at the roots | `chebfit(xn, values, 63)`: a square, well-conditioned system that interpolates | yes |
| Targets in $[-1,1]$ | `targets = 2*arange(n)/(n-1) - 1` | yes, no extrapolation |
| $\epsilon_I+127\,\epsilon_Q$ | `broad_bin_mean_error_bound`: log-space $2(\pi L/(2W))^{64}/64!$ plus `(2*64-1)` times $2(\pi h/W)^8/8!$; overflow maps to `inf` | yes |
| Refusal | `if not bound <= tolerance: raise`, which also refuses `nan` and `inf` | yes |

`residual_tol` is a pointwise error relative to each line's peak density on the
Fourier route. Comparing the per-$\sum_l\lvert w_l\rvert$ bound with it uses
the same normalization. `fourier_bin_mean_reduce` requires $n_{\rm bins}\ge2$
through `uniform_bin_axis`, so $L>0$ and the logarithm is defined. Both the host
call and the CUDA call (`xp=cp`) use the same function and refusal. The CUDA
route computes the 64 node values on the device and fits on the host.

## Independent numerical evidence

The scratch script uses `mpmath` at 40 digits. It forms exact bin means from
the closed-form antiderivative
$\int f\,dx=\operatorname{Si}(2ax)/a-\sin^2(ax)/(a^2x)$, which was checked by
differentiation. It does not use implementation helpers for the reference.

- **Moments:** $\int\lvert t\rvert^m\Lambda/(2a)^m$ equals $2/((m+1)(m+2))$ by
  quadrature for $m=0,1,8,64$. The transform reproduces $f$ to $0$ at 40 digits.
- **Node polynomial:** the sampled $\sup\lvert\omega_{64}\rvert/(2(L/4)^{64})$
  is $1.0000000$.
- **Lebesgue constant:** $3.61$ for 64 roots, below the bound $127$.
- **Bound formula:** `broad_bin_mean_error_bound` equals an independent mpmath
  evaluation to about $10^{-14}$ relative over four $(W_0,h,L)$ triples between
  $10^{-20}$ and $10^{7}$.
- **End to end:** three axes ($h=0.5,2,5$ eV; $n_{\rm bins}=1000,400,300$),
  with six lines each. The lines have signed normal weights and resonances
  spread over $[c_0-2L,c_0+3L]$, inside and outside the band. $W_0$ is set
  just below the $10^{-6}$ refusal threshold. The maximum error against the
  exact mean is $\le1.1\times10^{-12}$, compared with bounds of
  $2.7$–$4.5\times10^{-6}$. All cases pass.
- **Refusal:** a 5 eV line on a 400-bin, 2 eV axis is refused with the
  documented message.
- **Tests:** `tests/montecarlo/spectrum/test_bin_broad.py` has 9 passing tests.

## Tightness

The bound is correct but loose by several orders of magnitude. The derivative
bound overstates the $m$th moment by $(m+1)(m+2)/2$. That is
$2145$ for $m=64$ and $45$ for $m=8$. The node-error amplification $127$
exceeds the Lebesgue constant $3.61$. The Taylor-remainder step omits the
$1/9$ from averaging $\lvert x-c\rvert^8$. The refusal is therefore
conservative: it rejects some splits that would meet `residual_tol`. It never
accepts a split that violates the bound. Tightening any of these factors needs
its own derivation; it is not required for correctness.

## Observations outside the claim

- `cuda_fourier_bin_mean_reduce` defaults to `broad_width_eV=1000.0` and does
  not validate it. A `None` value fails with a `TypeError` instead of
  selecting the unsplit route, which the host reference does. It also trusts
  the caller's `axis` to have $n_{\rm bins}\ge2$. Neither changes the bound.
- In mixed mode, the broad and Fourier routes each meet `residual_tol` on
  their own line subsets. The combined pointwise error is the sum of the two
  bounds, and each is relative to its own subset's weights.
- FP64 rounding in phase formation, the $\operatorname{sinc}$ evaluation,
  block summation and `chebfit`/`chebval` is excluded, as the claim states.
  The observed errors near $10^{-13}$ are consistent with this.

## Threshold scaling (#1173, 2026-11-14)

The device default treats `broad_width_eV=1000.0` as a floor.
`broad_width_for_axis` raises the threshold $W_0$ by factors of $2^{1/8}$
until the bound above holds for the call's bin width $h$ and centre span.
Raising $W_0$ only moves lines from the broad route to the exact Fourier
route. The bound itself is unchanged and is still checked against the
narrowest broad line. A fixed 1 keV threshold refused coarse axes, such as
500 eV bins over 0.5–10 keV (bound about 0.2), and long fine axes, such as a
31 keV span. Wherever the floor already meets the bound, the threshold and
output are unchanged; this includes the 0.5–10 keV axis at 0.09375 eV.
Regression: `test_bin_broad.py::test_coarse_bins_reduce_exactly_with_the_scaled_threshold`.
