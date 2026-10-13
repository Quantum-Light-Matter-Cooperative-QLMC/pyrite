# Centred rectangular sensor acceptance

Validation: `detector-sensor-acceptance`.

## Verification boundary

Fresh-context verification on 2026-10-13. The derivation below was written
before inspecting the implementation body or its tests. The intended owner is
`src/pyrite/detectors/acceptance.py::SensorAcceptance.resolve`;
`resolve()` returns full local polar and azimuthal opening angles in degrees
and exact solid angle in steradians. Inputs `distance_mm`, `width_mm`, and
`height_mm` are positive finite lengths.

Source: [PNNL-19616, §2.3.2a, printed p. 9](https://digital.library.unt.edu/ark:/67531/metadc829646/m2/1/high_res_d/1023130.pdf),
the four-corner rectangular solid-angle expression. The report uses
half-dimensions; here $w$ and $h$ denote full dimensions. Set its source
coordinates to the rectangle centre and its perpendicular distance to $d$.
The independent derivation starts from projected solid angle, rather than
assuming the report's closed form. Inspection of the rendered source page
confirms all four arctangent terms and its half-dimension convention. Its
preceding displayed integral contains a typographical error: the distance
power is printed as $1/2$, which has incorrect dimensions for solid angle.
The independently derived $3/2$ power below differentiates back from the
report's correct four-corner expression.

## Independent derivation

Place the source at the origin and a source-facing sensor in the plane
$z=d>0$, with local azimuthal tangent coordinate $x\in[-w/2,w/2]$ and polar
tangent coordinate $y\in[-h/2,h/2]$. For a surface element at distance
$R=\sqrt{d^2+x^2+y^2}$, the projected-area definition gives

$$
d\Omega=\frac{\cos\gamma}{R^2}\,dx\,dy
=\frac{d}{(d^2+x^2+y^2)^{3/2}}\,dx\,dy.
$$

The source-facing convention makes this measure positive. Define

$$
F(x,y)=\arctan\left(\frac{xy}{d\sqrt{d^2+x^2+y^2}}\right).
$$

Direct differentiation gives

$$
\frac{\partial F}{\partial x}
=\frac{dy}{(d^2+x^2)\sqrt{d^2+x^2+y^2}},\qquad
\frac{\partial^2 F}{\partial y\,\partial x}
=\frac{d}{(d^2+x^2+y^2)^{3/2}}.
$$

The four-corner evaluation of this primitive is therefore exact. For centred
bounds, the four signs turn into equal positive contributions:

$$
\begin{aligned}
\Omega
&=4\arctan\left(
\frac{(w/2)(h/2)}{d\sqrt{d^2+(w/2)^2+(h/2)^2}}\right)\\
&=\boxed{4\arctan\left(
\frac{wh}{2d\sqrt{4d^2+w^2+h^2}}\right)}.
\end{aligned}
$$

In the central meridional planes, right-triangle geometry gives the full local
opening angles

$$
\boxed{\Delta\theta=2\arctan\left(\frac{h}{2d}\right)},\qquad
\boxed{\Delta\phi_{\mathrm{local}}=2\arctan\left(\frac{w}{2d}\right)}.
$$

These are local directional spans. In particular the azimuthal value is not
a global longitude interval about an external polar axis. The sensor's edges
project to great-circle arcs, so substituting these angles into a
constant-latitude spherical-patch formula changes the accepted region.
No observation-angle Jacobian multiplies this sensor solid angle.

## Cheap filters

Units pass: the arctangent arguments are ratios of equal powers of length.
The result is invariant under a common positive rescaling of $d,w,h$.
Interchanging width and height preserves $\Omega$ and exchanges the two
opening angles. Positive finite dimensions give $0<\Omega<2\pi$ and local
spans between zero and $\pi$.

For $w,h\ll d$,

$$
\Omega\sim\frac{wh}{d^2},\qquad
\Delta\theta\sim\frac{h}{d},\qquad
\Delta\phi_{\mathrm{local}}\sim\frac{w}{d}.
$$

At fixed positive dimensions, $d\to0^+$ gives $\Omega\to2\pi$, the
source-facing hemisphere. Either dimension tending to zero gives zero area.
For the face of a cube viewed from its centre, $w=h=2d$ and

$$
\Omega=4\arctan(1/\sqrt{3})=2\pi/3,
$$

as required by the six identical faces covering the sphere. Both full local
opening angles then equal $\pi/2$.

## Implementation comparison

The final implementation's arctangent ratio is exactly
$wh/(2d\sqrt{4d^2+w^2+h^2})$. It computes the ratio logarithm with a
normalized `math.hypot` radius, then represents the ratio as two bounded
exponentials passed to `math.atan2`. This is algebraically equivalent to
the derived positive-argument arctangent, including its large-ratio limit.
Each local opening angle normalizes its own dimension and distance before
`math.atan2`, preserving the meridional ratios for extreme aspect ratios.
The output ordering correctly assigns height to polar span and width to
local azimuthal span.

A separate 256-point tensor Gauss integration of projected area at five
geometries, including the cube face, two large rectangles, and a
$14.08\times14.08$ millimetre sensor at $400$ millimetres, agrees with the
final implementation to maximum relative error $6.23\times10^{-16}$.

An additional adversarial numerical reference follows directly by taking
$w\to\infty$ in the projected-area integral:

$$
\begin{aligned}
\int_{-\infty}^{\infty}
\frac{d\,dx}{(d^2+x^2+y^2)^{3/2}}
&=\frac{2d}{d^2+y^2},\\
\Omega_{\mathrm{strip}}
&=\int_{-h/2}^{h/2}\frac{2d}{d^2+y^2}\,dy
=4\arctan\left(\frac{h}{2d}\right).
\end{aligned}
$$

For $h=d$, this gives $1.8545904360032244$ steradians, local polar opening
$53.13010235415598^\circ$, and local azimuthal opening tending to
$180^\circ$. The implementation reproduces the strip limit for
`SensorAcceptance(1e-200, 1e200, 1e-200)` and
`SensorAcceptance(1e-300, 1e300, 1e-300)`, and the width/height-swapped
case, to at most $4.76\times10^{-14}$ relative solid-angle error. These
cases would lose a representable opening and solid angle under common
normalization of all three lengths; the final implementation preserves them.

Verification command:

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/pyrite-454-venv uv run pyrite-dev test tests/detectors/test_acceptance.py --durations=0
```

Result: 50 tests passed in 2.50 seconds. These include independent
projected-area quadrature, the cube-face reference, small-sensor limits,
common rescaling by $10^{\pm200}$, and the extreme-aspect strip anchor
`test_sensor_acceptance_extreme_aspect_ratio_strip_limit`.
`tests/dev/test_docs.py` also passed all six checks. The task owner performs
the documentation build and positive rendered-math inspection.

The claim certifies geometry. Multiplying the exact solid angle by one
central-direction intensity still assumes that intensity is sufficiently
uniform across the sensor. No detector response or hardware calibration is
certified here.

## Verdict and proposed ledger change

Units: pass. Limits: pass. Signs and conventions: pass. Independent
re-derivation: matches the final implementation, with no divergent factor,
sign, exponent, unit, or local-span convention. Verdict: **rederived**.

The task owner may replace the pending-verification note with this dated
fresh-context result and advance `filtered` to `rederived`. The named
projected-area, cube-face, and strip regression anchors support `anchored`
once its CI requirement is met. Human sign-off remains separate.
