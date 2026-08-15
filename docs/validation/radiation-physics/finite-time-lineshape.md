# Validation: finite-time-lineshape

## Independent derivation

For real $P$, $T>0$, and angular phase with no hidden $2\pi$, define

$$
Q(P,T)=\int_{-T/2}^{T/2}e^{i\,2Pt}\,dt.
$$

Direct integration gives the finite-time amplitude in
{eq}`eq-finite-time-amplitude`.

```{math}
:label: eq-finite-time-amplitude

\begin{aligned}
Q(P,T)
&=\frac{e^{iPT}-e^{-iPT}}{i2P}\\
&=\frac{\sin(PT)}{P}
=T\frac{\sin(PT)}{PT}.
\end{aligned}
```

The value at $P=0$ is defined by continuity as $Q(0,T)=T$. Therefore,
verbatim from the independent result,

$$
\boxed{
|Q(P,T)|^2
=\frac{\sin^2(PT)}{P^2}
=T^2\left(\frac{\sin(PT)}{PT}\right)^2
=T^2\operatorname{sinc}^2\!\left(\frac{PT}{\pi}\right)
}.
$$

Here NumPy's normalized sinc is
$\operatorname{sinc}(u)=\sin(\pi u)/(\pi u)$. Its exact area and
distributional normalization are given by
{eq}`eq-finite-time-normalization`.

```{math}
:label: eq-finite-time-normalization

\int_{-\infty}^{\infty}|Q(P,T)|^2\,dP=\pi T,
\qquad
\frac{|Q(P,T)|^2}{\pi T}
\xrightarrow[T\to\infty]{\mathcal D}\delta(P).
```

Equivalently,

$$
\frac{|Q(P,T)|^2}{T}
\xrightarrow[T\to\infty]{\mathcal D}\pi\delta(P).
$$

For a detuning $\Delta$ defined by $2P=D\Delta$, the independent result is

$$
\boxed{
Q(\Delta,T)=T\frac{\sin(D\Delta T/2)}{D\Delta T/2}
=T\operatorname{sinc}\!\left(\frac{D\Delta T}{2\pi}\right)
},
$$

with unit-area density $|D||Q|^2/(2\pi T)$.

## Units, conventions, and limiting cases

The dimensionless phase requires $[P]=T^{-1}$. Thus $[Q]=T$,
$[|Q|^2]=T^2$, and $[|Q|^2dP]=T$, matching the area $\pi T$.
With $c=1$, time and length share units.

- At $P=0$, $Q=T$ and $|Q|^2=T^2$.
- Zeros occur at $P_n=n\pi/T$, so the central null-to-null width is
  $2\pi/T$.
- The FWHM is approximately $2.783115/T$.
- Peak height scales as $T^2$, width as $T^{-1}$, and total area as $T$.

## Implementation comparison

`src/pyrite/montecarlo/spectrum/lines.py::mc_spectrum` defines

$$
\omega_{\rm res}=\frac{\mathbf v\cdot\mathbf g}
{1-\mathbf v\cdot\hat{\mathbf n}},
\qquad
T=t_L=\frac{L_{\rm seg}}{\beta},
$$

and uses the finite-segment factor

$$
t_L^2\operatorname{sinc}^2\!\left(
\frac{(1-\mathbf v\cdot\hat{\mathbf n})
(\omega-\omega_{\rm res})t_L}{2\pi}
\right).
$$

This maps exactly to the independent convention through

$$
P=\frac{1-\mathbf v\cdot\hat{\mathbf n}}{2}
(\omega-\omega_{\rm res}),
\qquad T=t_L.
$$

In code, photon energy rather than wavenumber is gridded:

```python
a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
x = a_width * (E_grid - E_r) / xp.pi
S = xp.sinc(x) ** 2
```

Since $E-E_{\rm res}=\hbar c(\omega-\omega_{\rm res})$, `x` is exactly
$PT/\pi$. To carry the normalization into the energy coordinate, define

$$
D=1-\mathbf v\cdot\hat{\mathbf n},
\qquad
P=\frac{D(E-E_{\rm res})}{2\hbar c}.
$$

Writing $dP$ for the positive integration measure, the Jacobian is

$$
dP=\frac{|D|}{2\hbar c}\,dE,
\qquad
dE=\frac{2\hbar c}{|D|}\,dP.
$$

(For a subluminal segment, $D>0$, so the absolute values may be omitted.)
The independently derived $P$-area therefore becomes

$$
\int_{-\infty}^{\infty}|Q(P(E),T)|^2\,dE
=\frac{2\hbar c}{|D|}\int_{-\infty}^{\infty}|Q(P,T)|^2\,dP
=\frac{2\pi T\hbar c}{|D|}.
$$

Thus the energy-domain unit-area density and its delta limit are
{eq}`eq-finite-time-energy-density`.

```{math}
:label: eq-finite-time-energy-density

L_T^{(E)}(E)
=\frac{|D|}{2\pi T\hbar c}|Q(P(E),T)|^2
\xrightarrow[T\to\infty]{\mathcal D}\delta(E-E_{\rm res}),
```

or, equivalently,

$$
\frac{|Q(P(E),T)|^2}{T}
\xrightarrow[T\to\infty]{\mathcal D}
\frac{2\pi\hbar c}{|D|}\delta(E-E_{\rm res}).
$$

The leading `t_L**2` is carried once in `pref`. The sinc argument contains the
required $1/(2\hbar c)$, while the separate `1/HBARC_EV_ANG` in `pref`
converts the physical differential spectrum from per wavenumber to per eV.
On energy integration, that prefactor cancels the $\hbar c$ in the
lineshape's energy-domain area above and leaves the expected Jacobian
$2\pi T/|D|$. No factor of two, $\pi$, $D$, or $\hbar c$ is missing.
Both the full-grid and cutoff paths use the same normalized-sinc argument;
truncation only intentionally removes tails outside the requested cutoff.

This comparison concerns the finite-time lineshape. It does not resolve the
separate `line-energy-dispersion` sign adjudication used to define the center
$\omega_{\rm res}$.

## Numerical spot check

The required integration over $P\in[-20/T,20/T]$ produced the values in
{numref}`tbl-finite-time-spot-check`.

```{list-table} Finite-window integration of the squared amplitude.
:name: tbl-finite-time-spot-check
:header-rows: 1

* - $T$
  - Numerical integral
  - $\pi T$
* - 10.0
  - 30.906233356297136
  - 31.41592653589793
* - 100.0
  - 309.0623335629714
  - 314.1592653589793
```

Each finite-window integral is 98.378% of $\pi T$ (1.622% low from omitted
sinc-squared tails), and the result scales by exactly ten when $T$ does.
This agrees with the exact all-real-line area $\pi T$ and the expected finite
integration-window error.

## Adjudication

**match**

The implementation's factor is exactly
$T^2\operatorname{sinc}^2(PT/\pi)$ with
$P=D(\omega-\omega_{\rm res})/2=D(E-E_{\rm res})/(2\hbar c)$,
$D=1-\mathbf v\cdot\hat{\mathbf n}$, and $T=t_L$. Its energy-domain
area is $2\pi T\hbar c/|D|$, and multiplication by the implementation's
per-energy prefactor `1/HBARC_EV_ANG` leaves $2\pi T/|D|$. Signs, the phase's
factor of two, NumPy's $\pi$-normalized sinc, the energy-coordinate
Jacobian, dimensions, normalization, and limiting behavior all agree.
