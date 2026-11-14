# cxr-line-fourier-reduction

Ledger row: [`cxr-line-fourier-reduction`](../ledger-core-coherent-physics.md#cxr-line-fourier-reduction). Status: unverified. Validation: `cxr-line-fourier-reduction`.

Implementation-context derivation for #1192: the incoherent bin-mean line sum evaluated in the Fourier domain. It computes the same quantity as the exact [`sinc-bin-integration`](sinc-bin-integration.md) mean, without the [`sinc-bin-far-envelope`](sinc-bin-far-envelope.md) approximation. The cost scales with lines times modes, not lines times bins. The host FP64 reference is `montecarlo/spectrum/cxr/_bin_fourier.py`. No production route uses it yet.

## Claim

Take lines $l$ with resonance $E_l$, inverse width $a_l>0$ [eV$^{-1}$], first-zero width $W_l=\pi/a_l$, and weight $w_l$ [photons/eV]. Their density is

```{math}
:label: eq-clfr-density

\rho(E)=\sum_l w_l\,\frac{\sin^2(a_l u_l)}{(a_l u_l)^2},\qquad u_l=E-E_l .
```

The bins are uniform: width $h$, centres $c_k=c_0+kh$ for $k=0,\dots,n-1$. This is the `bin_axis` convention on a uniform node axis whose low edge is not clamped. Let $U$ be the span of the band together with all resonances, $U'=U+h/2$, and let the period be $P=Nh\ge 2U'$, with $D=P-U'$. The reduction

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
\Bigl|\frac{\Delta Y_l}{Y_l}\Bigr|\le\frac{W_l^2}{4\pi D^2},
```

where $Y_l=\pi w_l/a_l$ is the whole-line mass and $\Delta Y_l$ is the change in the band integral. The Chebyshev interpolation of $C$ and FP64 rounding come on top of these bounds; see the error budget below.

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

**Images.** Subtracting the $n\ne0$ terms leaves the true bin mean. For a band bin and any resonance, $|u_l|=|c_k-E_l|\le U$. Bin points of image $n$ therefore satisfy $|v|\ge|n|P-U'\ge|n|D$, using $P\ge2U'$. Split $\sin^2(av)/(av)^2=\frac{1}{2a^2v^2}-\frac{\cos 2av}{2a^2v^2}$:

- *Envelope.* The bin mean of $1/v^2$ is $1/(v_c^2-\delta^2)$, with $v_c$ the bin-centre image distance and $\delta=h/2$. Summed over $n\ne0$ at the bin centre ($\delta\to0$), and using $\sum_{n}(x+n\pi)^{-2}=\csc^2x$, the sum is $(\pi/P)^2g(\pi u/P)$. That is {eq}`eq-clfr-correction`. The dropped $\delta$ term is $\sum_{n\ne0}\delta^2/[v_c^2(v_c^2-\delta^2)]\le 2\zeta(4)\delta^2/(D^2(D^2-\delta^2))$. Relative to $w_l$, this is $\le W_l^2h^2/(6D^4)$, which is the $(1+h^2/D^2)$ factor in {eq}`eq-clfr-bounds`. $g$ is evaluated by its Laurent series $\tfrac13+\tfrac{x^2}{15}+\tfrac{2x^4}{189}+\tfrac{x^6}{675}+\tfrac{2x^8}{10395}$ for $|x|<0.2$, where the first omitted term is $<10^{-17}$, and directly elsewhere. $|x|\le\pi U/P<\pi/2$, so no pole is approached.
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
| Chebyshev interpolation of $C$, 24 nodes | $\lesssim\rho^{-24}\max|C|$ | nearest pole $\ge D\ge3U'$ beyond a band of length $\le U'$, so the ellipse parameter is $\rho\ge 7+\sqrt{48}=13.9$, a ratio $<10^{-27}$ |
| FFT and mode-sum rounding | $O(\varepsilon_{64}\log N)\sum_l w_l\pi/(a_lh)$ | FP64 |

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

Fresh-context validation and human sign-off pending.
