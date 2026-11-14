# cxr-line-fourier-reduction

Ledger row: [`cxr-line-fourier-reduction`](../ledger-core-coherent-physics.md#cxr-line-fourier-reduction). Status: anchored. Validation: `cxr-line-fourier-reduction`.

Implementation-context derivation for #1192: the incoherent bin-mean line sum evaluated in the Fourier domain. It computes the same quantity as the exact [`sinc-bin-integration`](sinc-bin-integration.md) mean, without the [`sinc-bin-far-envelope`](sinc-bin-far-envelope.md) approximation. The cost scales with lines times modes, not lines times bins. The host FP64 reference is `montecarlo/spectrum/cxr/_bin_fourier.py`. No production route uses it yet.

## Claim

Take lines $l$ with resonance $E_l$, inverse width $a_l>0$ [eV$^{-1}$], first-zero width $W_l=\pi/a_l$, and weight $w_l$ [photons/eV]. Their density is

```{math}
:label: eq-clfr-density

\rho(E)=\sum_l w_l\,\frac{\sin^2(a_l u_l)}{(a_l u_l)^2},\qquad u_l=E-E_l .
```

The bins are uniform: width $h$, centres $c_k=c_0+kh$ for $k=0,\dots,n-1$. This is the `bin_axis` convention on a uniform node axis whose low edge is not clamped. Let $U$ be the span of the band together with all resonances, $U'=U+h/2$, and let the period be $P=Nh$, with $D=P-U'\ge h$. The reduction

```{math}
:label: eq-clfr-synthesis

\bar\rho_k=\frac1P\Bigl[G_0+2\operatorname{Re}\sum_{j\ge1}G_j\,e^{2\pi i jk/N}\Bigr]-C(c_k),\qquad
G_j=\operatorname{sinc}\!\Bigl(\frac{t_jh}{2\pi}\Bigr)\sum_l w_l\frac{\pi}{a_l}\Bigl(1-\frac{t_j}{2a_l}\Bigr)_{+}e^{-i(E_l-c_0)t_j},
```

with $t_j=2\pi j/P$, modes $j\ge N$ folded onto $j\bmod N$, and the envelope image correction

```{math}
:label: eq-clfr-correction

C(c)=\sum_l\frac{w_l}{2a_l^2}\Bigl(\frac{\pi}{P}\Bigr)^2 g\Bigl(\frac{\pi(c-E_l)}{P}\Bigr),\qquad g(x)=\frac{1}{\sin^2x}-\frac1{x^2},
```

equals the exact bin mean $\sum_l w_l\bar S_{l,k}$ to within, per line,

```{math}
:label: eq-clfr-bounds

|\Delta\bar\rho_k|\le w_l\,\frac{W_l^2}{6D^2}\Bigl(1+\frac{h^2}{D^2}\Bigr),\qquad
\Bigl|\frac{\Delta Y_l}{Y_l}\Bigr|\le\frac{W_l^2}{4\pi D^2}+\frac{\pi^2W_lh^2B}{270D^4},
```

where $Y_l=\pi w_l/a_l$ is the whole-line mass, $\Delta Y_l$ is the change in the band integral, $B=nh$ is the band length, and $D\ge h$. The Chebyshev interpolation of $C$ and FP64 rounding come on top of these bounds; see the error budget below.

## Derivation

**Line transform.** With $\hat f(t)=\int f(E)e^{-iEt}\,\mathrm dE$, the box $\mathbf 1_{|t|<a}$ has inverse transform $\sin(au)/(\pi u)$. So $\sin(au)/(au)$ transforms to $(\pi/a)\mathbf 1_{|t|<a}$. Squaring in $E$ convolves in $t$, with a factor $1/2\pi$:

```{math}
\widehat{\Bigl[\tfrac{\sin^2 au}{(au)^2}\Bigr]}(t)=\frac{1}{2\pi}\Bigl(\frac{\pi}{a}\Bigr)^2(2a-|t|)_+=\frac{\pi}{a}\Bigl(1-\frac{|t|}{2a}\Bigr)_+ .
```

The support is $|t|<2a$, not $|t|<a$ as the #1192 issue text first stated. At $t=0$ the value is the line mass $\pi/a$, which is the total-mass identity. A shift by $E_l$ multiplies by $e^{-iE_lt}$.

**Bin mean.** The mean over $[c-h/2,c+h/2]$ is the convolution with a unit-mass box. The box transforms to $\operatorname{sinc}(th/2\pi)=\sin(th/2)/(th/2)$, so

```{math}
\bar\rho(c)=\frac1{2\pi}\int\hat\rho(t)\operatorname{sinc}\Bigl(\frac{th}{2\pi}\Bigr)e^{ict}\,\mathrm dt .
```

**Poisson summation.** $\bar\rho\in L^1$ is continuous, and it decays as $1/c^2$. Its transform is compactly supported and of bounded variation. Poisson summation therefore gives, exactly,

```{math}
\sum_{n\in\mathbb Z}\bar\rho(c+nP)=\frac1P\sum_{j\in\mathbb Z}\hat\rho(t_j)\operatorname{sinc}\Bigl(\frac{t_jh}{2\pi}\Bigr)e^{ict_j},\qquad t_j=\frac{2\pi j}{P}.
```

The right side is a finite sum, because $\hat\rho(t_j)=0$ for $|t_j|\ge 2\max a_l$. At $c=c_k$, $e^{ic_kt_j}=e^{ic_0t_j}e^{2\pi ijk/N}$, so $j$ enters only modulo $N$. Folding $j\to j\bmod N$ is exact, and lines narrower than a bin, whose transforms extend beyond Nyquist, need no truncation. $\rho$ is real, so $G_{-j}=\overline{G_j}$, which gives the $G_0+2\operatorname{Re}\sum_{j\ge1}$ form in {eq}`eq-clfr-synthesis`. The phase reference $c_0$ only relabels $k$.

**Images.** Subtracting the $n\ne0$ terms leaves the true bin mean. For a band bin and any resonance, $|u_l|=|c_k-E_l|\le U$. Bin points of image $n$ therefore satisfy $|v|\ge|n|P-U'\ge|n|D$, for any $P>U'$. Split $\sin^2(av)/(av)^2=\frac{1}{2a^2v^2}-\frac{\cos 2av}{2a^2v^2}$:

- *Envelope.* The bin mean of $1/v^2$ is $1/(v_c^2-\delta^2)$, with $v_c$ the bin-centre image distance and $\delta=h/2$. Summed over $n\ne0$ at the bin centre ($\delta\to0$), and using $\sum_{n}(x+n\pi)^{-2}=\csc^2x$, the sum is $(\pi/P)^2g(\pi u/P)$. That is {eq}`eq-clfr-correction`. The dropped $\delta$ term is $\sum_{n\ne0}\delta^2/[v_c^2(v_c^2-\delta^2)]\le 2\zeta(4)\delta^2/(D^2(D^2-\delta^2))$. With $v_c\ge D\ge h=2\delta$, $v_c^2-\delta^2\ge\tfrac34v_c^2$. Per unit weight, $q=W^2/(2\pi^2)$, so the term is positive and at most $\pi^2W^2h^2/(270D^4)$ per bin. Pointwise, that is inside the $(1+h^2/D^2)$ factor in {eq}`eq-clfr-bounds`. Summed over a band of length $B$ and divided by $Y_l/w_l=W$, it gives the yield's second term. That term is linear in $W$, so for narrow lines it dominates the first; it was found by the fresh-context verification (finding 1). $g$ is evaluated by its Laurent series through $x^{12}$ ($\tfrac13+\tfrac{x^2}{15}+\tfrac{2x^4}{189}+\tfrac{x^6}{675}+\tfrac{2x^8}{10395}+\tfrac{1382x^{10}}{58046625}+\tfrac{4x^{12}}{1403325}$) for $|x|<0.2$, where the first omitted term $3617x^{14}/10854718875$ is $<6\times10^{-17}$, and directly elsewhere. $|x|\le\pi U/P<\pi/2$, so no pole is approached.
- *Oscillatory remainder* (not corrected). Pointwise, the bin mean of $|\cos(2av)|/(2a^2v^2)$ is at most $1/(2a^2n^2D^2)$. Summed over $n\ne0$ with $2\zeta(2)=\pi^2/3$, it is at most $\pi^2/(6a^2D^2)=W^2/(6D^2)$ of the line peak. For the band integral of image $n$, the bins tile one interval $[X,Y]$ that excludes $0$ and has $|X|,|Y|\ge|n|D$. Integration by parts with $\int\cos 2av=\sin(2av)/2a$ gives $|\int_X^Y\cos(2av)v^{-2}\,\mathrm dv|\le 3/(2a\,n^2D^2)$. Summing over $n\ne0$ and dividing by $Y_l=\pi/a$ (per unit weight) gives $W^2/(4\pi D^2)$.

**Period choice.** `fourier_period` sets $D\ge\max(3U',\,W_{\max}/\sqrt{6\,\tau},\,h)$, with $\tau$ = `residual_tol`. Every line's pointwise bound is then at most $\tau(1+h^2/D^2)\le2\tau$, and at most $\tau$ to $10^{-9}$ relative once $D\ge3\times10^4h$. The bound is driven by the widest line: $N\approx(U'+D)/h$ grows like $W_{\max}/(h\sqrt\tau)$.

## Assumptions

- The incoherent finite-time $\operatorname{sinc}^2$ profile and bin convention of `sinc-bin-integration`. Each line weight is constant across the line.
- The bins are uniform, and the low edge $c_0-h/2>0$ is not clamped. Nonuniform or clamped axes are refused (`uniform_bin_axis`). Production line grids that are nonuniform (resonance-local, windowed) need a separate decision.
- The resonances lie inside the span $U$ that sets $P$. There is no out-of-window treatment: lines far above the band enlarge $U$, and with it $N$.
- FP64 throughout. Phases $|(E_l-c_0)t_j|\le 2a_lU$ are formed in FP64. The host reference sums modes directly ($O(\sum_l 2a_lP/2\pi)$). A NUFFT spreader (Plan item 4) replaces only how $G_j$ is computed.

## Limiting cases

- *Single line, total mass.* $G_0=\sum_l w_l\pi/a_l$. A band that captures a line's window gives the `sincsq_window_mass` captured mass (test: $10^{-9}$).
- *$a\to\infty$ (delta line).* $\hat S\to\pi/a$ for all $|t|<2a$, and folding makes the band sum $(\pi/a)\,\mathbf 1[\text{bin}]/h$ exactly. A $10^{-4}$ eV line on 1 eV bins puts its mass in the resonance bin to $10^{-3}$, and the remainder agrees with the exact mean.
- *Bin narrower than a line.* $\operatorname{sinc}(th/2\pi)\to1$ recovers node sampling of the periodic density.
- *$P\to\infty$.* Both bounds in {eq}`eq-clfr-bounds` vanish as $D^{-2}$, and $C\to\sum_l w_l\pi^2/(6a_l^2P^2)\to0$.

## Error budget

| term | size | control |
| --- | --- | --- |
| oscillatory image remainder, pointwise | $\le w_lW_l^2(1+h^2/D^2)/(6D^2)$ | `residual_tol`, default $10^{-6}$ of each line's peak |
| oscillatory image remainder, yield | $\le Y_lW_l^2/(4\pi D^2)$ | the same $D$; $\le 0.48\,\tau$ relative |
| bin-centre envelope correction, yield | $\le Y_l\pi^2W_lh^2B/(270D^4)$ | $D\ge3U'\ge3B$: $\le 4.6\times10^{-4}\,W_lh^2/U'^3$ |
| Chebyshev interpolation of $C$, 24 nodes | $\lesssim\rho^{-24}\max|C|$ | nearest pole $\ge D\ge3U'$ beyond a band of length $\le U'$, so the ellipse parameter is $\rho\ge 7+\sqrt{48}=13.9$, a ratio $<10^{-27}$ |
| FFT and mode-sum rounding | $O(\varepsilon_{64}\log N)\sum_l w_l\pi/(a_lh)$ | FP64; `residual_tol` $\ge10^{-12}$ is enforced, because below it rounding exceeds the bound (verification finding 3) |

## Numeric checks (host)

`tests/montecarlo/spectrum/test_bin_fourier.py` compares against `_host_bin_mean(..., exact_widths=None)` summed over lines.

| case | `residual_tol` | max pointwise / $\sum w$ | relative yield |
| --- | --- | --- | --- |
| 60 random lines, $W\in[0.2,40]$ eV, 400×1 eV bins, $N\approx1.2\times10^4$ | $10^{-4}$ | $7.1\times10^{-7}$ | $2.1\times10^{-7}$ |
| same | $10^{-6}$ | $8.6\times10^{-9}$ | $7.7\times10^{-9}$ |
| same | $10^{-8}$ | $1.9\times10^{-10}$ | $1.5\times10^{-11}$ |
| lines of 0.01, 0.05, 0.3 eV on 1 eV bins (folded) | $10^{-6}$ | $6.3\times10^{-10}$ of peak | $5.7\times10^{-10}$ |

The pointwise error falls as $D^{-2}$ ($D$ from 100 to 3000 widths: $3.1\times10^{-6}\to8.4\times10^{-9}$), as {eq}`eq-clfr-bounds` predicts.

## Cost (to measure, #1192 Plan items 1 and 5)

Total modes are $\sum_l\lceil 2a_lP/2\pi\rceil=\sum_l P/W_l$. Narrow lines dominate the mode count, and the widest line sets $P$. A NUFFT bucketed by $a_l$ costs $O(\text{lines}\times\text{spread}+\sum_bM_b\log M_b)$, with $M_b\approx P/W_{\min,b}$, against the current $O(\text{lines}\times\text{bins in }\pm64W)$. The production estimate needs the measured $W$ distribution.

Human sign-off pending.

## Exact integer-cutoff transform plan

The experimental planner `_fourier_cutoffs.py` reorganizes the existing
positive-mode sum without approximating widths or triangle support. It is
not connected to the production scorer or to a NUFFT dependency.

Set $J_l=\lfloor2a_l/\Delta t\rfloor$. A zero-amplitude endpoint at
$t=2a_l$ may be included. Sort lines by $J_l$, then partition the active
pairs $\{(l,j):1\le j\le J_l\}$ into disjoint rectangles. Starting with
a power-of-two mode interval $[1,2^k+1)$ covering all active modes, emit
the lines whose cutoffs cover the entire interval. Bisect the interval
for the remaining lines, discarding inactive pairs. At unit-length leaves
every remaining active pair is emitted. Thus each active pair occurs once,
and every emitted rectangle has a power-of-two mode count. Line membership
is a contiguous slice in the cutoff-sorted array; the traversal stores only
a depth-first stack, without duplicating line-index arrays.

Within each rectangle all triangle factors are active. Two transforms of
the same phases $\phi_l=(E_l-c_0)\Delta t$ suffice:

$$
A_j=\sum_l\frac{\pi w_l}{a_l}e^{-ij\phi_l},\qquad
B_j=\sum_l\frac{\pi w_l}{2a_l^2}e^{-ij\phi_l},\qquad
G_j=A_j-t_jB_j.
$$

The sums here are over the rectangle's lines only. Summing its contributions
recovers the original coefficient exactly in real arithmetic. The bin sinc
multiplier, zero mode, folding and image correction remain separate.
`partitioned_positive_modes` accepts a batched transform callback with this
contract; tests use a direct DFT and compare against an independent full
triangle sum, including signed weights and remote resonances. Regrouping
changes floating-point summation and can cause cancellation near a triangle
endpoint; no backend error bound or speedup is inferred from exact indexing.

A prototype callback in `_fourier_transform.py` uses optional FINUFFT type 1
with two source rows. For an interval of $m=j_1-j_0$ modes, the centred
library indices $k=-\lfloor m/2\rfloor,\ldots,\lceil m/2\rceil-1$ are
shifted by $s=j_0+\lfloor m/2\rfloor$: modulating source strengths by
$e^{-is\phi_l}$ and using `isign=-1`, `modeord=0` returns increasing requested
indices $j=k+s$. The prototype requests complex128, `eps=1e-12` and one CPU
thread explicitly. FINUFFT remains an isolated optional runtime import, not
a declared dependency or production route. Its requested tolerance is not a
bound on the final spectrum, especially after subtracting triangle moments.
The callback convention was checked against the current
[FINUFFT Python interface](https://finufft.readthedocs.io/en/latest/python.html)
and [mode ordering options](https://finufft.readthedocs.io/en/latest/opts.html),
then tested with version 2.5.1.

The provisional hybrid uses FINUFFT only for rectangles with at least 64
sources, 32 modes, and 262144 source-mode pairs. Other rectangles use a direct
DFT blocked over both source and mode axes, with at most 262144 complex
exponentials per temporary matrix. This limits that workspace, not total
process memory or output arrays. These thresholds match the frozen cost
audit; they are not measured CPU/GPU crossover rules. Tests check signs,
odd/even shifted intervals, batching, bounded direct blocks, and that small
fallbacks do not import FINUFFT. Optional tests skip when it is unavailable;
the explicit isolated test run includes them.

A line can participate in multiple transforms (at most the number of binary
digits of its last mode). Many small rectangles may make transform setup
expensive. Input duplication, transform counts and direct-fallback work must
be measured on frozen scenes before choosing this as the device design.

## Experimental broad-line split

Validation: `cxr-line-broad-interpolation`. **Status: unverified**; this
implementation-context derivation is separate from the anchored Fourier claim
above. Independent verification is pending.

`fourier_bin_mean_reduce(..., broad_width_eV=W0)` retains every line but
evaluates widths $W_l\ge W_0$ separately, using 64 Chebyshev root nodes over
the bin-centre interval. Only the remaining lines set the Fourier period.
The default `None` preserves the unsplit reference; production does not call
this reference. Line weights must be constant in energy. Signed weights are
allowed, with all error bounds scaled by $\sum_l|w_l|$.

For a unit-peak line, the triangular Fourier transform above is a positive
measure of unit total mass supported on $|t|\le 2\pi/W_l$. Differentiating
its inverse transform therefore bounds the $m$th energy derivative by
$(2\pi/W_0)^m$. Averaging over a fixed-width bin preserves this bound.
On an interval of length $L=(n_{\rm bins}-1)h$, the monic polynomial whose
roots are the $r=64$ Chebyshev nodes has supremum $2(L/4)^r$. The interpolation
remainder is thus at most

$$
\epsilon_I\sum_l|w_l|,
\qquad \epsilon_I=\frac{2}{r!}\left(\frac{\pi L}{2W_0}\right)^r.
$$

The bin averages at interpolation nodes use four-point Gauss--Legendre
quadrature. Its positive normalized weights sum to one and it integrates
degree-seven polynomials exactly. Expanding a line about its bin centre
through degree seven bounds both the integral remainder and quadrature
remainder by $(\pi h/W_0)^8/8!$. Their difference is bounded by

$$
\epsilon_Q=\frac{2}{8!}\left(\frac{\pi h}{W_0}\right)^8.
$$

Each nonconstant Chebyshev coefficient of interpolation-node errors bounded
by $\epsilon_Q$ has magnitude at most $2\epsilon_Q$ by the discrete cosine
formula; the constant coefficient is at most $\epsilon_Q$. Since
$|T_k(x)|\le1$, the interpolated quadrature error is at most
$(2r-1)\epsilon_Q$. The total exact-arithmetic pointwise bound is

$$
|\Delta\bar\rho_k|\le
\left[\epsilon_I+(2r-1)\epsilon_Q\right]\sum_l|w_l|.
$$

`broad_bin_mean_error_bound` computes this coefficient using the minimum
actual broad width; the reducer refuses a split exceeding `residual_tol`.
FP64 evaluation, summation, phase formation and interpolation rounding are
additional errors. This is an **absolute** bound; it does not guarantee
relative band yield for distant resonances or cancellation of signed weights.
Multiplying by $n_{\rm bins}h$ gives an absolute bound on integrated error.

Limits: $W_0\to\infty$ suppresses both error terms; $h\to0$ removes the
quadrature term; an underresolved broad split is refused instead of silently
approximated. An all-broad call avoids any padded FFT. Nonuniform/clamped
bins remain refused, and energy-dependent attenuation/escape cannot be
silently absorbed into constant weights.

Tests in `tests/montecarlo/spectrum/test_bin_broad.py` compare broad-only and
mixed reductions with the exact bin primitive, include negative weights,
out-of-band resonances, threshold equality, a high-precision bin integral,
and refusal of inadequate interpolation or bin quadrature resolution.
These regression checks do not constitute independent validation.

## Independent verification

Fresh-context verifier, 2026-11-14, worktree HEAD `7b216478d`. The verifier derived the transform, the Poisson identity, the envelope image sum, and both remainder bounds from the ledger row and the `sinc-bin-integration` profile and bin convention. That derivation came before the implementation body or the derivation above was read. The verifier changed only this section. Scratch scripts are outside the repository.

- **Claim**: `cxr-line-fourier-reduction` — `montecarlo/spectrum/cxr/_bin_fourier.py::fourier_bin_mean_reduce`, `::fourier_period`, `::image_residual_bounds`, `::uniform_bin_axis` — $\operatorname{sinc}^2$ triangle transform, Poisson summation, $\sum_n(x+n\pi)^{-2}=\csc^2x$, integration by parts
- **Filters**: units pass; limits pass (total mass $G_0$, delta line, $P\to\infty$); signs/conventions pass (phase $e^{-i(E_l-c_0)t_j}$ with the $e^{+2\pi ijk/N}$ synthesis, Hermitian folding)
- **Re-derivation**: `matches` for the triangle transform and its support $\lvert t\rvert<2a$, the Poisson identity, exact folding mod $N$, the envelope image closed form, the Laurent coefficients, and the pointwise bound. It `differs` in one place. The yield bound $W_l^2/(4\pi D^2)$ covers only the oscillatory remainder. It omits the band integral of the bin-centre envelope error, which the pointwise bound does carry through its $h^2/D^2$ factor. That missing term scales as $W_lh^2B/D^4$, not $W_l^2$, so it exceeds $W_l^2/(4\pi D^2)$ for lines with $W_l\lesssim h^2B/D^2$ (finding 1)
- **Verdict**: `discrepancy` (narrow). The reduction itself is rederived. The stated yield bound is unconditional, but a high-precision evaluation exceeds it by up to $4\times$, both under the code's $P\ge2U'$ guard and at the $D=3U'$ that `fourier_period` can choose. The excess is far below any physical significance, around $10^{-9}$ of the line mass
- **Write-up**: `docs/validation/radiation-physics/cxr-line-fourier-reduction.md`
- **Suggested ledger change**: `unverified` → `discrepancy` until one of two fixes lands. (a) The yield bound gains the envelope term $\pi^2W_lh^2B/(360D^4)$, with $B$ the band length, in the ledger row, this page, and `image_residual_bounds`. (b) The correction uses the exact bin-mean envelope (finding 1), which removes the $h^2$ term from both bounds. Then `rederived`. Never `signed-off` from this context.

### Independent derivation

Let $S_a(u)=\sin^2(au)/(au)^2=\operatorname{sinc}^2(au/\pi)$, which has peak $1$ and mass $\pi/a=W$. Let $\hat f(t)=\int f(u)e^{-iut}\,\mathrm du$. Since $\int_{-a}^{a}e^{iut}\,\mathrm dt=2\sin(au)/u$, inversion gives the transform of $\sin(au)/(au)$ as $(\pi/a)\mathbf 1_{\lvert t\rvert<a}$. The transform of the square is the self-convolution divided by $2\pi$:

$$
\widehat{\Bigl[\frac{\sin au}{au}\Bigr]}(t)=\frac{\pi}{a}\mathbf 1_{\lvert t\rvert<a},\qquad
\hat S_a(t)=\frac1{2\pi}\Bigl(\frac{\pi}{a}\Bigr)^2(2a-\lvert t\rvert)_+=\frac{\pi}{a}\Bigl(1-\frac{\lvert t\rvert}{2a}\Bigr)_+ .
$$

So the support is $\lvert t\rvert<2a$. A 40-digit `mpmath` quadrature at $a=0.7$ reproduces this to the quadrature accuracy, and gives zero beyond $t=2a=1.4$. The bin mean over $[c-h/2,c+h/2]$ is the convolution with $h^{-1}\mathbf 1_{\lvert u\rvert<h/2}$, whose transform is $\sin(th/2)/(th/2)=\operatorname{sinc}(th/2\pi)$. Let $m(c)=\sum_lw_l\,\overline{S}_{a_l}(c-E_l)$. Then $\hat m(t)=\operatorname{sinc}(th/2\pi)\sum_lw_l\hat S_{a_l}(t)e^{-iE_lt}$. This is continuous and compactly supported, and $m=O(c^{-2})$, so Poisson summation holds pointwise:

$$
\sum_{n\in\mathbb Z}m(c+nP)=\frac1P\sum_{j\in\mathbb Z}\hat m(t_j)e^{ict_j},\qquad t_j=\frac{2\pi j}{P}.
$$

At $c_k=c_0+kh$ with $P=Nh$, $e^{ic_kt_j}=e^{ic_0t_j}e^{2\pi ijk/N}$. Absorbing $e^{ic_0t_j}$ gives $G_j=\hat m(t_j)e^{ic_0t_j}$, which carries the phase $e^{-i(E_l-c_0)t_j}$ of the claim. The factor $e^{2\pi ijk/N}$ depends on $j$ only mod $N$, so folding is exact, including modes beyond $N$ from lines narrower than a bin. $m$ is real, so $G_{-j}=\overline{G_j}$. Grouping $\pm j$ gives $\frac1P[G_0+2\operatorname{Re}\sum_{j\ge1}G_je^{2\pi ijk/N}]$. When $j\ge1$ folds onto residue $0$, it contributes $2\operatorname{Re}G_j$ to the $k$-independent term, which is still correct.

**Images.** Write $S_a=\frac{1}{2a^2v^2}-\frac{\cos2av}{2a^2v^2}$. Let $U$ bound $\lvert c-E_l\rvert$ over band points and resonances, $\delta=h/2$, $U'=U+\delta$, and $D=P-U'$. Image $n\ne0$ of a bin centre lies at $\lvert v_c\rvert\ge\lvert n\rvert P-U=D+\delta+(\lvert n\rvert-1)P\ge\lvert n\rvert D+\delta$. That needs only $P\ge D$, which always holds, so $P\ge2U'$ is not required by either bound. It is required by the code's guard and by the Chebyshev ellipse margin. The bin mean of the envelope is exactly $1/(v_c^2-\delta^2)$. The centre form $1/v_c^2$, summed with $\sum_n(x+n\pi)^{-2}=\csc^2x$, gives $\frac{1}{2a^2}(\pi/P)^2[\csc^2(\pi u/P)-(\pi u/P)^{-2}]$, which is the claim's correction. The centre form leaves the per-bin envelope residual

$$
E_k=\frac1{2a^2}\sum_{n\ne0}\frac{\delta^2}{v_c^2(v_c^2-\delta^2)}\le\frac1{2a^2}\cdot\frac{2\zeta(4)\,\delta^2}{D^4}=w\,\frac{\pi^2W^2h^2}{360\,D^4}\quad(\text{per unit }w),
$$

using $v_c^2-\delta^2\ge n^2D^2$. This is positive in every bin, because $1/v^2$ is convex.

- *Pointwise.* The oscillatory bin mean is at most $\frac{1}{2a^2}\sum_{n\ne0}(v_c^2-\delta^2)^{-1}\le\frac{1}{2a^2}\frac{\pi^2}{3D^2}=\frac{W^2}{6D^2}$. Adding $E_k$ gives $\frac{W^2}{6D^2}(1+\frac{\pi^2h^2}{60D^2})$, which is inside the claimed $\frac{W^2}{6D^2}(1+h^2/D^2)$. **Matches** (the claim is loose in its $h^2$ term).
- *Yield, oscillatory part.* The band bins tile one interval $[X,Y]$ per image, with $0<\lvert n\rvert D\le X<Y$. One integration by parts, keeping both boundary terms, gives $\lvert\int_X^Y\cos(2av)v^{-2}\mathrm dv\rvert\le\frac1{2a}(X^{-2}+Y^{-2})+\frac1{2a}(X^{-2}-Y^{-2})=\frac{1}{aX^2}$. Summing gives $\frac{1}{2a^2}\frac{\pi^2}{3aD^2}$, which is $W^2/(6\pi D^2)$ relative to the mass $W$. The page's $3/(2a\,n^2D^2)$ is valid but loose by $3/2$.
- *Yield, envelope part.* $h\sum_kE_k\le B\cdot\max_kE_k$, with $B=n_{\rm bins}h$, gives $\pi^2W^2h^2B/(360D^4)$. Relative to the mass $W$, this is $\pi^2Wh^2B/(360D^4)$. **This term is missing from the claimed yield bound.** It is linear in $W$, so for narrow lines it dominates the $W^2$ oscillatory term.

### Filters

| filter | result |
| --- | --- |
| units | $G_j$: photons/eV × eV = photons; $G_j/P$: photons/eV; $C$: $w/a^2\cdot P^{-2}$ = photons/eV. Pass |
| total mass | $G_0=\sum_lw_l\pi/a_l$ (code: `g0`), and $\hat S_a(0)=\pi/a$. Pass |
| delta line $a\to\infty$ | $\hat S\to\pi/a$ on all $\lvert t\rvert<2a$. Folding then sums the full Fourier series of the periodised box, giving $(\pi/a)/h$ in the containing bin. A $W=10^{-4}$ eV line on a 1 eV bin edge matches the exact mean to $6\times10^{-16}$. Pass |
| $P\to\infty$ | $C\to\sum_lw_l\pi^2/(6a_l^2P^2)$, from the Laurent constant $1/3$, and both bounds go to $0$. Pass |
| sign/phase | $\hat m$ uses $e^{-iE_lt}$, and the inverse FFT ($N\cdot$`ifft`) supplies $e^{+2\pi ijk/N}$. A line at $E_l$ peaks at $k=(E_l-c_0)/h$, which was checked numerically. Pass |
| normalisation | `np.sinc(t*h/(2*pi))` $=\sin(th/2)/(th/2)$, and the triangle is $(\pi/a)(1-t/2a)$. Pass |

### Implementation comparison

- `_folded_modes`: $j=1..\lfloor 2a/\Delta t\rfloor$. A mode at exactly $t=2a$ has zero amplitude, so dropping it is exact. The amplitude, the phase $-(E_l-c_0)t$, and `j % n_fft` bincount folding match the derivation. Phases reach $2a_lU\approx2.5\times10^6$ rad in the stress test below, and the FP64 phase error stays under $10^{-12}$ of the line peak there.
- `fourier_bin_mean_reduce`: `(g0 + 2*Re(N*ifft(F))[:n])/P` matches $\frac1P[G_0+2\operatorname{Re}\sum_{j\ge1}\cdots]$, with folded residue $0$ entering as $2\operatorname{Re}$. Pass.
- `_inv_sin_sq_minus_inv_sq`: the coefficients $\frac13,\frac1{15},\frac2{189},\frac1{675},\frac2{10395}$ are the $\csc^2$ Laurent (Bernoulli) coefficients. They match. The first omitted term is $\frac{1382}{49116375}x^{10}=2.9\times10^{-12}$ at $\lvert x\rvert=0.2$, not "$<10^{-17}$" (code comment line 126; page line 71). Measured against 40-digit `mpmath` on $\lvert x\rvert\le0.2$, the relative error is $7.3\times10^{-12}$ (finding 2).
- `_envelope_image_correction`: evaluates the centre form at 24 Chebyshev nodes on $[c_0,c_{n-1}]$, then interpolates. The nearest poles of $\csc^2(\pi(c-E_l)/P)$ are at $c=E_l\pm P$, at least $D$ from the band. Pass.
- `fourier_period`: $U$ is the span of the band edges and the resonances, $U'=U+h/2$, and $N h\ge U'+\max(3U',W_{\max}/\sqrt{6\tau},h)$. So $D\ge W_{\max}/\sqrt{6\tau}$, and the pointwise bound is at most $\tau(1+h^2/D^2)\le2\tau$, with $D\ge3U'$ giving $P\ge4U'$. Pass. It does not control the yield term of finding 1.
- `image_residual_bounds`: transcribes the claimed formulas and refuses $P<2U'$. The yield formula omits the term of finding 1.

### Reproduced numbers

FP64 route against `w[:, None] * _host_bin_mean(..., exact_widths=None)` summed over lines. The ratio columns compare the measured value with the line-weighted bound; the pointwise column is relative to $\sum w$ only where marked.

| case | `residual_tol` | $N$ | pointwise / bound | yield / bound |
| --- | --- | --- | --- | --- |
| 60 random lines, $W\in[0.2,40]$ eV, 400×1 eV | $10^{-6}$ | 16800 | 0.19 | 0.026 |
| same, new draw | $10^{-10}$ | $1.58\times10^6$ | 0.13 | $8.8\times10^{-4}$ |
| one line $W=300$ eV (band 400 eV) | $10^{-6}$ | 122880 | 0.44 | 0.25 |
| one line $W=3000$ eV (wider than band) | $10^{-6}$ | $1.23\times10^6$ | 0.44 | 0.12 |
| 11 lines $W=0.3$ eV exactly on bin edges | $10^{-6}$ | 1617 | 0.0046 | 0.016 |
| $W=10^{-4}$ eV on a bin edge | $10^{-6}$ | 1617 | 0.55 (FP64 floor, $6\times10^{-16}$) | rounding ($8\times10^{-14}$ relative) |
| one line 48.6 keV above band + one in band | $10^{-6}$ | 196020 | 0.38 | $4.5\times10^{-4}$ |
| one line 990 eV below band | $10^{-6}$ | 5600 | 0.38 | 0.0066 |
| negative and positive weights | $10^{-6}$ | 2450 | 0.66 | $6.6\times10^{-5}$ |
| $W=5$ eV | $10^{-14}$ | $2.04\times10^7$ | **1.35** (FP64 floor $1.3\times10^{-14}$) | 0 |
| 299 lines $W=0.01$ eV + one $W=2$ eV, 4000 bins at 20 keV | $10^{-8}$ / $10^{-9}$ | 16038 / 30000 | 0.09 / 0.12 (of $\sum w$) | $2.5\times10^{-12}$ / $2.0\times10^{-13}$ relative |
| $W=h=0.5$ eV, lines on centres and edges | $10^{-6}$ | 810 | $1.8\times10^{-7}$ | $1.5\times10^{-11}$ relative |

FP64 rounding hides the image residual of narrow lines, so the yield bound was tested independently of the FFT. For one line with 12 bins of width $h$, the exact image residual $R_k=\sum_{n\ne0}\bar S(c_k+nP-E)-C(c_k)$ was computed at 50 digits. Images $\lvert n\rvert\le60$ used the exact $\operatorname{Si}$ antiderivative of `sinc-bin-integration`. The bin-averaged envelope tail beyond them used digamma differences.

| $W$ eV | $h$ eV | $D$ eV | $D/U'$ | pointwise / bound | yield rel. | $W^2/(4\pi D^2)$ | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 37.5 | 3 | $7.7\times10^{-4}$ | $4.7\times10^{-9}$ | $5.7\times10^{-5}$ | $8\times10^{-5}$ |
| 0.1 | 1 | 37.5 | 3 | $4.2\times10^{-5}$ | $5.5\times10^{-9}$ | $5.7\times10^{-7}$ | 0.0097 |
| 0.01 | 1 | 37.5 | 3 | $4.2\times10^{-5}$ | $5.5\times10^{-10}$ | $5.7\times10^{-9}$ | 0.097 |
| 0.001 | 1 | 37.5 | 3 | $4.2\times10^{-5}$ | $5.5\times10^{-11}$ | $5.7\times10^{-11}$ | 0.97 |
| 0.001 | 1 | 12.5 | 1 | $1.0\times10^{-4}$ | $1.0\times10^{-9}$ | $5.1\times10^{-10}$ | **1.99** |
| 0.001 | 2 | 25 | 1 | $1.1\times10^{-4}$ | $5.1\times10^{-10}$ | $1.3\times10^{-10}$ | **4.0** |

The yield residual is linear in $W$ below $W\approx0.1$ eV, as finding 1 predicts. The corrected bound $W^2/(4\pi D^2)+\pi^2Wh^2B/(360D^4)$ holds in every row: for example, $1.4\times10^{-8}\ge1.0\times10^{-9}$ in the $D=12.5$ row. At $D/U'=3$, the ratio is 0.97 at $W=10^{-3}$ eV and grows as $1/W$ below that. So the default `fourier_period` choice does not keep the stated yield bound for very narrow lines either.

### Findings

1. **Medium (bound statement): the yield bound omits the bin-centre envelope term.** `image_residual_bounds` line 117, ledger row, and page eq-clfr-bounds and line 72. The correction $C$ uses the bin-centre form of $\sum_{n\ne0}(u+nP)^{-2}$. The band integral of its error, $h\sum_kE_k\le\pi^2W^2h^2B/(360D^4)$ (relative $\pi^2Wh^2B/(360D^4)$), is positive in every bin and not covered by $W^2/(4\pi D^2)$. The pointwise bound does include it, through $h^2/D^2$. The 50-digit evaluation exceeds the stated yield bound by $2\times$ at $P=2U'$, $h=1$, and by $4\times$ at $h=2$, both at $W=10^{-3}$ eV. The excess is around $10^{-9}$ of the line mass, so it has no practical effect on spectra. Fixes: (a) add the term to the yield bound; or (b) correct with the exact bin-mean envelope, $\sum_{n}\bigl[(u+nP)^2-\delta^2\bigr]^{-1}=\frac{\pi}{hP}\bigl[\cot\frac{\pi(u-\delta)}{P}-\cot\frac{\pi(u+\delta)}{P}\bigr]$ minus its $n=0$ term $1/(u^2-\delta^2)$. Option (b) removes the $h^2$ terms from both bounds, but the $n=0$ subtraction needs a cancellation-safe form near $u=\pm\delta$.
2. **Low (documentation): the Laurent truncation is $2.9\times10^{-12}$, not $<10^{-17}$.** `_bin_fourier.py:126` and page line 71. $C\lesssim w W^2/(6P^2)\le\tau w$, so the error in the result is below $10^{-11}\tau w$, which is negligible. Correct the comment, or add the $x^{10}$ term ($1382/49116375$) and lower the switch to $\lvert x\rvert<0.1$ if the stated precision matters.
3. **Low: the FP64 rounding floor exceeds very small `residual_tol`.** At $\tau=10^{-14}$, the measured pointwise error $1.35\times$ the bound is the rounding floor. $N$ also grows as $W_{\max}/(h\sqrt\tau)$, reaching $2\times10^7$ for a single 5 eV line. The page's error budget says rounding comes on top of the bounds. There is no lower clamp on `residual_tol`, and an $\sim10^{-12}$ floor would be prudent.
4. **Info: the claimed constants are valid and loose.** The yield oscillatory constant is $3/2$ loose ($W^2/(6\pi D^2)$ is sufficient), and the pointwise $h^2$ coefficient $1$ could be $\pi^2/60$. The measured pointwise ratios for wide lines reach 0.44–0.66, so the $W^2/(6D^2)$ term is near-sharp there.
5. **Info: $P\ge2U'$ is not needed by either bound.** $\lvert n\rvert P-U'\ge\lvert n\rvert D$ holds for any $P>U'$. The guard is harmless, and the Chebyshev margin needs $D\ge3U'$ anyway.
6. **Info: test gap.** `tests/montecarlo/spectrum/test_bin_fourier.py` never tests the yield bound in the narrow-line regime at small $D$. FP64 rounding would mask it there, so finding 1 needs an extended-precision or analytic residual check rather than an FFT comparison.

### Re-check (2026-11-14)

The same verifier re-checked the uncommitted fix (a) on HEAD `7b216478d`, against the page's "Response to verification" and the diffs to `_bin_fourier.py`, the test file and the ledger row.

- **Yield bound (finding 1): resolved.** The new term $\pi^2W_lh^2B/(270D^4)$ is valid. The re-derivation above already gives the rigorous $\pi^2/360$: $v_c^2\ge n^2D^2$ and $v_c^2-\delta^2\ge\lvert n\rvert D(\lvert n\rvert D+2\delta)\ge n^2D^2$ hold for any $P>U'$, so it is not only leading-order. The response's $\pi^2/270$ is therefore looser by $4/3$, but still correct. At $D\ge h$, the pointwise $h^2$ coefficient $1$ still covers $\pi^2/270$. `image_residual_bounds(a, P, U, h, band)` implements $W^2/(4\pi D^2)+\pi^2Wh^2B/(270D^4)$ as written. It refuses $D<h$ and accepts $D=h$. The test passes $B=nh$.
- **50-digit re-run against the new yield bound.** The method is the same as before (exact $\operatorname{Si}$ images for $\lvert n\rvert\le60$, digamma envelope tail). Every case is now inside the bound:

  | $W$ eV | $h$ eV | bins | $D$ eV | yield rel. | new bound | ratio |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1 | 1 | 12 | 37.5 | $4.7\times10^{-9}$ | $5.7\times10^{-5}$ | $8.3\times10^{-5}$ |
  | 0.1 | 1 | 12 | 37.5 | $5.5\times10^{-9}$ | $5.9\times10^{-7}$ | 0.0093 |
  | 0.01 | 1 | 12 | 37.5 | $5.5\times10^{-10}$ | $7.9\times10^{-9}$ | 0.070 |
  | 0.001 | 1 | 12 | 37.5 | $5.5\times10^{-11}$ | $2.8\times10^{-10}$ | 0.20 |
  | 0.001 | 1 | 12 | 12.5 | $1.0\times10^{-9}$ | $1.8\times10^{-8}$ | 0.055 |
  | 0.001 | 2 | 12 | 25 | $5.1\times10^{-10}$ | $9.1\times10^{-9}$ | 0.056 |
  | $10^{-4}$ | 2 | 12 | 25 | $5.1\times10^{-11}$ | $9.0\times10^{-10}$ | 0.057 |
  | $10^{-6}$ | 2 | 12 | 25 | $5.1\times10^{-13}$ | $9.0\times10^{-12}$ | 0.057 |
  | $10^{-5}$ | 1 | 40 | 2.0 ($\approx h$) | $2.2\times10^{-9}$ | $8.7\times10^{-7}$ | 0.0026 |

  The ratio levels off at about $0.057$ as $W\to0$. The envelope term dominates there, and the bound is loose by its $1/v_c^4\le1/(n^4D^4)$ step.
- **`residual_tol` floor (finding 3): resolved.** `fourier_period` accepts $10^{-12}$ and refuses $9.99\times10^{-13}$.
- **Laurent branch (finding 2): not resolved, and the error came from this verifier's own report.** The original finding cited the $x^{10}$ coefficient as $1382/49116375$, and the fix copied it. Exact Bernoulli coefficients of $\csc^2x-x^{-2}$, $(2n-1)2^{2n}\lvert B_{2n}\rvert x^{2n-2}/(2n)!$, cross-checked against a 40-digit `mpmath` Taylor expansion of $x^2\csc^2x$, are:

  | power | correct | in the code now |
  | --- | --- | --- |
  | $x^{10}$ | $1382/58046625=2.3808\times10^{-5}$ | $1382/49116375=2.8137\times10^{-5}$ |
  | $x^{12}$ | $4/1403325=2.8504\times10^{-6}$ | $4/1576575=2.5371\times10^{-6}$ |
  | $x^{14}$ (first omitted) | $3617/10854718875=3.332\times10^{-7}$; $5.5\times10^{-17}$ at $\lvert x\rvert=0.2$ | $8192/37574026875$; "$<4\times10^{-17}$" |

  The measured maximum relative error on $\lvert x\rvert\le0.2$ is now $1.3\times10^{-12}$, against 40-digit `mpmath`. With the correct coefficients the truncation would be about $1.6\times10^{-16}$ relative. The new continuity test (`rel=1e-13` at $x=0.2$) passes with the wrong coefficients, so it does not catch the error. The numerical effect on the result is still negligible: $C\le\tau w$, so the error is at most about $10^{-12}\tau w$. But the code comment, the page's Derivation paragraph, and its claimed truncation are wrong.
- **Re-check verdict.** The yield-bound fix is rederived. The claim stays at `discrepancy` only for the Laurent coefficients. After the two coefficients and the omitted term are corrected ($1382/58046625$, $4/1403325$, $3617/10854718875<6\times10^{-17}$), the suggested ledger change is `discrepancy`/`unverified` → `rederived`, without another re-derivation. A $10^{-15}$-relative test of the branch against an `mpmath` or `1/sin^2 - 1/x^2` reference at about $\lvert x\rvert=0.1$ would pin it. Never `signed-off` from this context.

## Response to verification

The task owner applied fix (a) on 2026-11-14:

- The yield bound now includes the bin-centre envelope term, in the ledger row, {eq}`eq-clfr-bounds` and `image_residual_bounds`. The coefficient is $\pi^2/270$ instead of the verifier's leading-order $\pi^2/360$; the extra $4/3$ comes from $v_c^2-\delta^2\ge\tfrac34v_c^2$ at $D\ge h$, which makes the bound rigorous rather than asymptotic. `image_residual_bounds` requires $D\ge h$ instead of $P\ge2U'$ (finding 5).
- The Laurent branch now runs through $x^{12}$, with the coefficients from the re-check (checked against `sympy.series`). A test pins it to $10^{-15}$ against 40-digit mpmath (finding 2).
- `fourier_period` refuses `residual_tol` below $10^{-12}$ (finding 3).
- Finding 4 (loose constants) is kept as stated margin. Finding 6 is covered analytically: an FP64 FFT comparison cannot resolve the term.
