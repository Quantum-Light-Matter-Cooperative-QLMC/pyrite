# Validation: `bremslib-angular-model`

## Scope and independence

Validation: `bremslib-angular-model`. Fresh-context validation on 2026-09-22.
The independent expression below was derived from the ledgered source,
quantity, units, assumptions, and limiting cases before the implementation
bodies were inspected. This validates PyRITE's interpolation and directional
track-length estimator against the BremsLib definitions; it does not establish
experimental accuracy or authorize human `signed-off` status.

## Source quantity

Poškus, *Atomic Data and Nuclear Data Tables* **166** (2025) 101734 and the
BremsLib v2.0.8 manual define the scaled single- and double-differential cross
sections

$$
\chi(T,x)=\frac{k}{Z^2}\frac{d\sigma}{dk},\qquad
D(T,x,\theta)=\frac{k}{Z^2}\frac{d^2\sigma}{dk\,d\Omega},
\qquad x=\frac{k}{T}.
$$

For an unpolarized beam incident on an unoriented target, the distribution is
azimuth-independent. Its solid-angle integral is therefore

$$
I[D]=2\pi\int_0^\pi D(\theta)\sin\theta\,d\theta.
$$

Define the normalized shape $S=D/I[D]$. The physical cross sections are

$$
\frac{d^2\sigma}{dk\,d\Omega}
=\frac{Z^2}{k}\chi(T,x)S(T,x,\theta),\qquad
\frac{d\sigma}{dk}=\frac{Z^2}{k}\chi(T,x),
$$

and hence

$$
\int d\Omega\,\frac{d^2\sigma}{dk\,d\Omega}=\frac{d\sigma}{dk}.
$$

## Piecewise-linear normalization

On one angular interval write $D(\theta)=a+b\theta$. Its exact contribution is

$$
2\pi\left[
-a\cos\theta+b\left(\sin\theta-\theta\cos\theta\right)
\right]_{\theta_0}^{\theta_1}.
$$

Summing this expression over the staged angular grid gives the normalization
used for each node. Division by that integral makes every staged shape have
unit solid-angle integral under the same piecewise-linear representation.

The incident-energy refinement constructs a geometric shape interpolant and
renormalizes it before multiplying by the correspondingly interpolated
$\chi$. Original normalized $(\chi,S)$ nodes are retained unchanged; raw
vendor DDCS values are deliberately rescaled to close against their parent
SDCS. Runtime interpolation is linear in $\theta$, $x=k/T$, and locally in
$\ln T$ on the refined grid. Because DDCS and SDCS use identical linear
weights at runtime, the solid-angle integral commutes with both the $x$ and
$\ln T$ interpolation. Normalization therefore holds at off-grid $(T,k)$, not
only at library nodes.

For the vendor's last ratio $x_{\rm top}<1$, both DDCS and SDCS hold their top
node through $x=1$ and vanish for $x>1$. The shared rule preserves angular
closure at the kinematic tip.

## Segment estimator, conventions, and units

For segment direction $\hat{\mathbf v}_s$ and detector direction
$\hat{\mathbf n}$,

$$
\cos\theta_s=\hat{\mathbf v}_s\cdot\hat{\mathbf n}.
$$

The attenuated track-length estimator is

$$
\frac{d^2N}{dE\,d\Omega}
=\frac{1}{N_e}\sum_{s,Z} n_Z L_s
\frac{Z^2}{k}\chi(T_s,k/T_s)S(T_s,k/T_s,\theta_s)
\exp[-\tau_s(k)].
$$

The stored scaled cross section is in mb or mb/sr, so multiplication by
$10^{-27}$ converts it to cm$^2$ or cm$^2$/sr. Division by photon energy in eV
gives cm$^2$/eV or cm$^2$/(eV sr). With number density in cm$^{-3}$ and segment
length in cm, the estimator returns photons/(eV sr incident electron).

The implementation's directional path temporarily multiplies the DDCS by
$4\pi$ and then applies the reducer's common $1/(4\pi)$ factor. Those factors
cancel exactly. Integrating the directional result over $4\pi$ therefore
recovers the SDCS estimator.

## Source-to-code comparison

`prepare_bremslib_table` applies the exact piecewise-linear solid-angle
integral, normalizes each shape, refines incident energy, and rebuilds DDCS
from the interpolated SDCS and normalized shape. `bremslib_segment_state`
uses $\arccos(\operatorname{clip}(\hat{\mathbf v}\cdot\hat{\mathbf n},-1,1))$
and linear angular interpolation. `evaluate_bremslib` applies the common
$x$ and incident-energy interpolation, the $10^{-27}Z^2/k$ conversion, and
the physical support $0<k\le T$.

`mc_brem_spectrum` evaluates the angle per segment, preserves EEDL as the
warned isotropic fallback for a missing element table or uncovered incident
energy, applies attenuation, and divides by the incident-electron count. The
portable path and the fused CUDA mirror use the same staged rows, interpolation
fractions, tip rule, unit conversion, fallback state, and $4\pi$ cancellation.
The CUDA comparison is hardware-gated; this derivation is a static
source-to-code comparison, not evidence of device execution.

No divergent factor, sign, exponent, unit, interpolation order, angle
convention, limiting case, or fallback scope was found.

## Evidence and verdict

The focused CPU/table anchors cover node and off-grid normalization, angular
averaging, forward peaking, tip support, warned fallback, chunk invariance,
loader resolution, and vendor-oracle comparison. The CUDA-gated anchor compares
the fused directional reducer with the portable CUDA path on the same synthetic
table.

- **Filters:** units pass; limits pass; signs and conventions pass.
- **Re-derivation:** matches.
- **Verdict:** `rederived`.
- **Human sign-off:** pending.
