# Rolled rectangular slit acceptance

Validation: `detector-slit-acceptance`.

## Verification boundary

Fresh-context verification on 2026-11-10 at commit `39778e8e3`. The
derivation below was written from the ledger row, the derivation section
[slit acceptance](../../physics/detectors/detector-solid-angle.md#slit-acceptance),
and the cited source before inspecting the body of
`src/pyrite/detectors/acceptance.py::SlitAcceptance.resolve` or its tests.
The owner returns the full local polar opening $\Delta\theta$ and local
azimuthal opening $\Delta\phi_{\rm local}$ in degrees, and the exact solid
angle $\Omega$ in steradians. Inputs `distance_mm`, `width_mm` and
`length_mm` are positive finite lengths; `roll_deg` is any finite real angle.

Source: the centred-rectangle projected-area integral already verified for
`detector-sensor-acceptance`
([write-up](detector-sensor-acceptance.md);
[PNNL-19616, §2.3.2a, printed p. 9](https://digital.library.unt.edu/ark:/67531/metadc829646/m2/1/high_res_d/1023130.pdf)).
The new content of this claim is rotation invariance about the line of
sight, and the projected extents of a rolled rectangle on the local tangent
axes.

## Independent derivation

Place the point source at the origin and the slit in the plane $z=L>0$,
centred on the foot of the normal. Let $x$ be the local azimuthal tangent
coordinate and $y$ the local polar tangent coordinate. In the slit's own
frame, $u\in[-\ell/2,\ell/2]$ runs along the length axis and
$v\in[-w/2,w/2]$ along the width axis. With the roll $\psi$ measured from the
length axis to the azimuthal tangent,

$$
x=u\cos\psi-v\sin\psi,\qquad y=u\sin\psi+v\cos\psi .
$$

### Solid angle

The projected-area measure is

$$
d\Omega=\frac{L}{(L^2+x^2+y^2)^{3/2}}\,dx\,dy .
$$

The map $(u,v)\mapsto(x,y)$ is a rotation, so $x^2+y^2=u^2+v^2$ and
$dx\,dy=du\,dv$ (unit Jacobian). The integral over the rolled rectangle
therefore equals the integral over the unrolled $\ell\times w$ rectangle. The
four-corner primitive verified in `detector-sensor-acceptance` then gives

$$
\boxed{\Omega=4\arctan\frac{w\ell}{2L\sqrt{4L^2+w^2+\ell^2}}},
$$

independent of $\psi$. Rotation invariance requires the rotation axis to
pass through the foot of the normal. That holds here because the slit is
centred on the line of sight (the point-source-on-normal assumption).

### Projected extents and opening angles

A linear function on a rectangle attains its extremes at corners. The
corners are $(u,v)=(\pm\ell/2,\pm w/2)$, so

$$
\max y-\min y=\ell\lvert\sin\psi\rvert+w\lvert\cos\psi\rvert\equiv h,\qquad
\max x-\min x=\ell\lvert\cos\psi\rvert+w\lvert\sin\psi\rvert\equiv b .
$$

The `SensorAcceptance` convention measures a local polar opening as the
meridional angle $\arctan(y/L)$, that is, the angle of the ray's projection
onto the $y$–$z$ plane; similarly for $x$. With this convention the extreme
polar directions are $\pm\arctan[h/(2L)]$, so

$$
\boxed{\Delta\theta=2\arctan\frac{h}{2L}},\qquad
\boxed{\Delta\phi_{\rm local}=2\arctan\frac{b}{2L}} .
$$

Convention note. For $0<\psi<90^\circ$ the polar-extreme corner is off the
central meridian, at $x_c=\tfrac12(\ell\cos\psi-w\sin\psi)$ (for
$0\le\psi\le90^\circ$). Its true angular elevation from the $x$–$z$ plane is
$\arctan[(h/2)/\sqrt{L^2+x_c^2}]$, smaller than the projected
$\arctan[h/(2L)]$ by a relative amount $\approx x_c^2/(2L^2)$. The claim
explicitly defines the spans as opening angles of the projected extents, and
it is consistent with `SensorAcceptance`. This is therefore a stated
convention, not a discrepancy. It is second order in slit size over distance.

## Cheap filters

| check | result |
|---|---|
| units | every arctangent argument is a ratio of equal powers of length; $\Omega$ in sr, spans in rad converted to deg |
| scale invariance | common rescaling of $L,w,\ell$ leaves all outputs unchanged |
| $\psi=0$ | $h=w$, $b=\ell$: `SensorAcceptance(height=w, width=ℓ)` |
| $\psi=90^\circ$ | $h=\ell$, $b=w$: `SensorAcceptance(height=ℓ, width=w)` |
| $\psi\to180^\circ-\psi$ | $\lvert\cos\rvert$ and $\lvert\sin\rvert$ unchanged, so spans are equal (mirror slits); $\Omega$ is roll-invariant |
| $\psi\to\psi+180^\circ$, $\psi\to-\psi$ | same projected extents; folding to $[0^\circ,180^\circ)$ loses nothing |
| small slit | $\Omega\to w\ell/L^2$, $\Delta\theta\to h/L$, $\Delta\phi_{\rm local}\to b/L$ |
| bounds | $0<\Omega<2\pi$; $\Omega\to2\pi$ as $L\to0^+$; spans in $(0,\pi)$ |
| cube face | $w=\ell=2L$ at any roll gives $\Omega=2\pi/3$ |
| $w\leftrightarrow\ell$ | swapping width and length is equivalent to $\psi\to\psi\pm90^\circ$ for the spans; $\Omega$ is symmetric |

All filters pass.

## Implementation comparison

`resolve` folds the stored roll (already reduced to $[0^\circ,180^\circ)$) to
$\psi'=\min(\psi,180^\circ-\psi)\in[0^\circ,90^\circ]$. On that interval,
$\cos\psi'=\lvert\cos\psi\rvert$ and $\sin\psi'=\lvert\sin\psi\rvert$, so the
code's

```text
polar_extent = width_mm * cosine + length_mm * sine
azimuthal_extent = length_mm * cosine + width_mm * sine
```

are exactly $h$ and $b$. `_cos_sin_deg` returns exact $(1,0)$ and $(0,1)$ at
the endpoints, so the $\psi=0^\circ$ and $90^\circ$ reductions to
`SensorAcceptance` are bitwise. `_opening_deg(extent, distance)` is
$2\,\mathrm{atan2}(e/s, 2L/s)$ in degrees with $s=\max(e,L)$, which equals
$2\arctan[e/(2L)]$. `_rectangle_solid_angle(distance, length, width)` is the
helper already rederived under `detector-sensor-acceptance`. Its formula
is symmetric in its two side arguments, so passing $\ell$ as width and $w$ as
height is immaterial. The roll fold `number % 180.0` maps negative and
over-range inputs into $[0^\circ,180^\circ)$, guarding the $180.0$ round-off
edge. Polar receives $h$ and azimuthal receives $b$, matching the
derivation's assignment. There is no divergent factor, sign, exponent, unit
or axis convention.

### Numeric comparison

An independent scratch script (not a repository test) compared
`SlitAcceptance(L, w, ℓ, ψ).resolve()` against:

- a $400\times400$ Gauss–Legendre integral of $L/(L^2+x^2+y^2)^{3/2}$ over
  the rolled rectangle, in the slit frame mapped by the roll;
- the corner-enumerated projected extents $h,b$ fed to $2\arctan[e/(2L)]$.

The geometries were $(L,w,\ell)$ in mm: $(400,0.1,10)$, $(50,2,30)$,
$(10,20,5)$ (width larger than length), and $(1,3,7)$ (strongly curved
integrand). The rolls were $0, 17, 30, 45, 63, 90, 117, 150, 179, -30$ and
$400$ degrees. The maximum relative error over all outputs was
$3.9\times10^{-14}$. A separate masked uniform grid in the tangent plane
(Cartesian, not slit-frame) agreed to $\le7\times10^{-7}$, consistent with its
edge-discretisation error. At $(L,w,\ell,\psi)=(10^4,0.1,1,37^\circ)$ the
small-slit limits held to $\le1.3\times10^{-9}$. The cube-face case at
$\psi=33^\circ$ gave $2\pi/3$ to $2\times10^{-16}$.

Anchor run:

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/detectors/test_slit_acceptance.py
```

Result: 80 passed in 1.58 s. The repository quadrature anchor
(`test_projected_extents_use_opening_angles`) uses a single nearly flat
geometry ($L=300$, $\ell=10$ mm). There the integrand varies by about
$10^{-4}$, so it constrains roll invariance only weakly. The curved
$(1,3,7)$ case above covers that gap. Adding such a case to the anchor
would strengthen the regression, but this is not required for the verdict.

## Broadening caveat

Downstream scoring multiplies one central-direction intensity by $\Omega$.
It broadens lines with `aperture_fwhm_eV`, the Zhai et al. SI Eq. (14)
Gaussian stand-in for a *uniform* polar span of full width $\Delta\theta$.
The tangent-plane polar coordinate of a uniformly illuminated rolled
rectangle is $y=u\sin\psi+v\cos\psi$, a sum of independent uniforms of full
widths $a=w\lvert\cos\psi\rvert$ and $c=\ell\lvert\sin\psi\rvert$. Its
density is their convolution: a trapezoid with base $a+c=h$ and flat top
$\lvert a-c\rvert$. This is uniform only when $a$ or $c$ vanishes, that is,
at $\psi=0^\circ$ or $90^\circ$. The ledger and derivation statement
("trapezoidal for $0<\psi<90^\circ$, exact only at $\psi=0^\circ$ and
$90^\circ$") is therefore correct, with two qualifications the text could
make explicit:

1. "Exact" means the uniform-span *assumption* holds. The Gaussian
   prefactor of `aperture_fwhm_eV` is itself Zhai's convention for a
   uniform span, and $1/R^3$ weighting plus arctangent nonlinearity make the
   angular density uniform only to second order in size over distance.
2. The error has a definite sign. The uniform model with full width $h$ has
   rms $h/\sqrt{12}$; the trapezoid has rms $\sqrt{a^2+c^2}/\sqrt{12}$. The
   uniform-span broadening therefore **overestimates** the aperture width by
   the factor $(a+c)/\sqrt{a^2+c^2}\in[1,\sqrt2]$. The factor is $\sqrt2$
   for $a=c$, which gives a triangular distribution, for example $w=\ell$ at
   $\psi=45^\circ$. For a narrow slit ($w\ll\ell$) the factor departs from 1
   once $\ell\lvert\sin\psi\rvert$ becomes comparable to $w$, that is, at
   rolls of order $w/\ell$ radians.

Neither qualification changes the certified geometry. They are suggested
wording refinements only.

## Verdict and proposed ledger change

Units: pass. Limits: pass. Signs and conventions: pass; the projected-angle
span convention is consistent with `SensorAcceptance`. Independent
re-derivation: matches the implementation exactly. Verdict: **rederived**.

The task owner may replace "independent fresh-context re-derivation is
pending" with a dated reference to this write-up and advance `filtered` to
`rederived`. The named anchors are green, so `anchored` follows once CI
confirms. Optionally, the Notes could state that the uniform-span broadening
overestimates the aperture rms by at most $\sqrt2$ for intermediate rolls.
Human sign-off remains separate.
