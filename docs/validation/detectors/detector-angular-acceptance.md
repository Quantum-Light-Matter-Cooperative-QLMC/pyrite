# Centred spherical detector acceptance

Validation: `detector-angular-acceptance`.

## Verification boundary

Fresh-context verification on 2026-10-13. The derivation below was written
before inspecting the implementation body or its tests. The intended owner is
`src/pyrite/detectors/acceptance.py::AngularAcceptance.resolve`;
`resolve(observation_angle_deg)` returns full polar and azimuthal spans in
degrees and solid angle in steradians, resolving exactly two specified values
among `polar_deg`, `azimuthal_deg`, and `solid_angle_sr`.

The source is the unit-sphere area element
$d\Omega=\sin\vartheta\,d\vartheta\,d\phi$. The geometry is a patch centred
at polar observation angle $\theta$, bounded by constant polar angles and
constant azimuths. Its full spans are $p=\Delta\theta$ and $a=\Delta\phi$.
All trigonometric arguments below are in radians.

## Independent derivation

The parameterization of a unit sphere gives orthogonal tangent vectors of
lengths $1$ and $\sin\vartheta$, hence its surface Jacobian is
$\sin\vartheta$. Integrating over the stated bounds gives

$$
\begin{aligned}
\Omega
&=\int_{\phi_0-a/2}^{\phi_0+a/2}
  \int_{\theta-p/2}^{\theta+p/2}\sin\vartheta\,d\vartheta\,d\phi\\
&=a\left[\cos(\theta-p/2)-\cos(\theta+p/2)\right]\\
&=\boxed{2a\sin\theta\sin(p/2)}.
\end{aligned}
$$

A positive centred patch requires

$$
0<\theta<\pi,\qquad
0<p\leq 2\min(\theta,\pi-\theta),\qquad
0<a\leq2\pi.
$$

Positive centred patches at either pole do not exist: any positive polar span
would cross a pole. Longitude periodicity permits a seam crossing, but a full
azimuthal span larger than $2\pi$ repeats surface area and is excluded.

With both spans supplied the displayed integral gives $\Omega$. The other
two cases give

$$
a=\frac{\Omega}{2\sin\theta\sin(p/2)},\qquad
p=2\arcsin\left(\frac{\Omega}{2a\sin\theta}\right).
$$

The inverse sine uses its principal branch. Since the permitted half-span is
at most $\pi/2$, the sine is strictly increasing on the admissible interval;
the inverse is unique. Its argument must also obey the tighter geometric bound

$$
0<\frac{\Omega}{2a\sin\theta}
\leq\sin\left(\min(\theta,\pi-\theta)\right).
$$

Testing only that the argument is at most one would wrongly admit a patch
crossing a pole. Likewise, resolving azimuth requires checking the resulting
$a\leq2\pi$. Input spans and solid angles must be positive and finite. Degree
inputs convert through $\pi/180$ before trigonometry; degree outputs use
$180/\pi$.

## Cheap filters

Units pass: radians and steradians are dimensionless angular measures;
no length scale appears. Signs pass: every factor is positive inside the
admissible domain. Reflection $\theta\mapsto\pi-\theta$ preserves the
solid angle. The azimuthal centre drops out by rotational symmetry.

For a small polar span,

$$
\Omega\sim ap\sin\theta.
$$

At $\theta=\pi/2$, $p=\pi$, and $a=2\pi$, the exact result is $4\pi$,
the whole sphere. At fixed admissible centre, either span tending to zero
makes $\Omega$ vanish. Near a pole, the allowed polar span also vanishes:
with $p=2\theta$ as $\theta\to0$, $\Omega\sim2a\theta^2$.

## Implementation comparison

The implementation's `band` is exactly
$2\sin\theta\sin(p/2)$; multiplying it by `math.radians(azimuthal)`
implements the independently derived area. The azimuth inverse divides by
that same band, and the polar inverse uses the principal inverse sine with
the geometric upper bound derived above. Both pole endpoints are rejected.
Constructor validation enforces exactly two positive finite values and the
full-span limits; resolution supplies the stricter observation-dependent
polar bound.

Boundary comparisons use relative tolerance $10^{-14}$ to absorb floating
point rounding, with clipping of the inverse-sine argument and the full
azimuthal boundary. This introduces only a numerical boundary allowance,
with no changed factor, sign, or convention.

The independent reference $\theta=90^\circ$, $p=60^\circ$,
$a=90^\circ$ gives $\Omega=\pi/2$. The implementation returns
$1.5707963267948963$ steradians. A separate 256-point Gauss integration of
$\sin\vartheta$ at five centres/spans, including $0.1^\circ$,
$179.9^\circ$, and the full sphere, agrees to maximum relative error
$1.97\times10^{-16}$. Both inverse descriptions resolve those quadrature
values successfully. The maintained tests additionally check inversions
near both poles, unattainable spans, the whole sphere, and the small-patch
limit.

Verification command:

```text
UV_CACHE_DIR=/tmp/pyrite-uv-cache UV_PROJECT_ENVIRONMENT=/tmp/pyrite-454-venv uv run pyrite-dev test tests/detectors/test_acceptance.py --durations=0
```

Result: 50 tests passed in 2.50 seconds, including the sensor checks.
`tests/dev/test_docs.py` also passed all six checks. The task owner performs
the documentation build and positive rendered-math inspection.

## Verdict and proposed ledger change

Units: pass. Limits: pass. Signs and conventions: pass. Independent
re-derivation: matches, with no divergent factor, sign, exponent, unit, or
span convention. Verdict: **rederived**.

The task owner may replace the pending-verification note with this dated
fresh-context result and advance `filtered` to `rederived`. The named
quadrature and analytic regression anchors support `anchored` once its CI
requirement is met. Human sign-off remains separate.
