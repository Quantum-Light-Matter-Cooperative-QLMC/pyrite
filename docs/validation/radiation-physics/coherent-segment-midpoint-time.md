(validation-coherent-segment-midpoint-time)=
# `coherent-segment-midpoint-time`

## Claim and source

A stored flight has midpoint position $\mathbf r_m$, length $L$, constant
speed $\beta$, and segment-start transport age $t_s$ (`t_ang`). With $c=1$,
its duration and midpoint age are

$$
T=L/\beta,\qquad \boxed{t_m=t_s+T/2}.
$$

The phase must pair this age with the same midpoint position:

$$
\Phi_m=\omega(t_m+t_0-\hat{\mathbf n}\cdot\mathbf r_m)
-\mathbf g\cdot\mathbf r_m.
$$

Here $t_0$ is the per-electron absolute bunch offset. Times are time-like
lengths in Å, $\omega=E/(\hbar c)$ and $\mathbf g$ have units Å$^{-1}$,
and every phase is dimensionless. `t_ang` remains the start age.

Source context: the centered constant-amplitude finite-time field attributed
to Feranchuk–Spence (2000), Eqs. (8), (10), (12)–(14), in
[coherent-emission](coherent-emission.md). The 2026-10-20 fresh-context verifier
rederived the result from this integral contract before inspecting source
bodies. Original-paper retrieval failed, so the original equation numbering
was not independently inspected in this pass.

## Independent centered-integral derivation

Write $\mathbf r(s)=\mathbf r_m+\boldsymbol\beta s$ for
$-T/2\le s\le T/2$, with $\boldsymbol\beta=\beta\hat{\mathbf v}$.
For a frozen complex radiation coefficient $A$,

$$
Q=\omega(1-\hat{\mathbf n}\cdot\boldsymbol\beta)
-\mathbf g\cdot\boldsymbol\beta,
\qquad
I_0=Ae^{i\Phi_m}\int_{-T/2}^{T/2}e^{iQs}\,ds
=Ae^{i\Phi_m}T\operatorname{sinc}\!\left(\frac{QT}{2\pi}\right),
$$

where $\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)$. Centering the spatial
coordinate while retaining the start time would introduce the erroneous
phase $e^{-i\omega T/2}$. Its modulus alone hides the error; interference
with another emitter exposes it.

The current scorer also integrates attenuation and the refractive escape
phase within each affine escape piece. Let

$$
\tau(s)=\tau_m+\Delta\tau\,s/T,\qquad
\ell(s)=\ell_m+\Delta\ell\,s/T,
\qquad \delta(E)=1-\operatorname{Re}n(E).
$$

The independently derived field is

$$
\begin{aligned}
I&=A e^{i[\Phi_m-\delta(E)\omega\ell_m]}
\int_{-T/2}^{T/2}
 e^{i[Q-\delta(E)\omega\Delta\ell/T]s}
 e^{-\tau(s)/2}\,ds\\
&=A e^{i[\Phi_m-\delta(E)\omega\ell_m]}
 T e^{-\tau_m/2}\frac{\sinh w}{w},\\
w&=i\left[\frac{QT}{2}-\frac{\delta(E)\omega\Delta\ell}{2}\right]
-\frac{\Delta\tau}{4}.
\end{aligned}
$$

The removable limit $\sinh(w)/w\to1$ is regular. Endpoint ordering follows
travel, so $\Delta\tau=\tau_{\rm end}-\tau_{\rm start}$; absorption uses
amplitude damping $e^{-\tau/2}$, not intensity damping $e^{-\tau}$.
This matches [coherent-formation-absorption](coherent-formation-absorption.md).

## Assumptions and exact subdivision

Velocity, frozen radiation coefficient, attenuation coefficients and material
are constant on the compared physical flight. Every escape-face or other
geometry breakpoint must occur in the partition, making optical depth and
escape length affine within each piece. Piece midpoint positions and ages
must come from one continuous trajectory and clock.

Under those conditions, integrating the same field over adjacent intervals
is additive: one flight equals the sum of its pieces, including absorption
and the refractive escape phase. Energy-dependent $\delta(E)$ alone does
not spoil this identity: at each output energy the same coefficient is used
in every piece. Active piece-dependent `sinc_cutoff` truncation can spoil it,
as can changing frozen amplitudes or energies between pieces. Thus the
midpoint anchor uses an inactive cutoff when testing exact subdivision.

As $L\to0$ the time correction vanishes. With zero absorption and dispersion
the field reduces to the centered vacuum sinc. Uniform attenuation along a
piece factors out as $e^{-\tau_m/2}$.

## Coherent self-term versus incoherent spectrum

A unit-modulus midpoint phase cancels in a self-term, but the full formation
factor does more than change phase. For $\Delta\tau\ne0$, its squared
modulus differs pointwise from the incoherent route's sinc squared times
segment-mean escape probability. A single absorbed segment therefore does
**not** give unconditional pointwise coherent/incoherent equality.

Writing $v=\operatorname{Im}w$ and
$F=e^{-\tau_m/2}\sinh(w)/w$, Parseval gives

$$
\int_{-\infty}^{\infty}|F(v)|^2\,dv
=\pi\langle e^{-\tau}\rangle.
$$

Converting this to equality of energy-integrated yields additionally needs
an affine detuning coordinate with the same constant Jacobian, frozen
prefactors and the complete line window. For full dispersion the Jacobian
contains the derivative term

$$
J=D-E\delta'(E)\boldsymbol\beta\cdot\nabla\ell,\qquad
D=1-\hat{\mathbf n}\cdot\boldsymbol\beta
-\delta(E)\boldsymbol\beta\cdot\nabla\ell.
$$

Replacing $J$ with $D$ is not exact. Finite-window yield checks are numerical
evidence with stated tolerances, not a proof of unconditional equality.

## Current implementation and evidence

- {source}`src/pyrite/montecarlo/spectrum/cxr/_setup.py::_prepare_spectrum` pairs the midpoint
  position with `seg_t_mid = seg_t + 0.5 * t_L_all`.
- {source}`src/pyrite/montecarlo/spectrum/cxr/_formation.py::expand_escape_pieces` shifts each
  piece's midpoint and advances its start age by the corresponding parent
  path offset divided by speed.
- {source}`src/pyrite/montecarlo/spectrum/cxr/_formation.py::formation_argument` and
  {source}`src/pyrite/montecarlo/spectrum/cxr/_formation.py::formation_factor`
  implement the independently derived $w$ and $F$; batched and per-reflection
  consumers are listed in [coherent-formation-absorption](coherent-formation-absorption.md).
- {source}`tests/montecarlo/coherent/test_coherent_emission.py::test_straight_flight_is_invariant_to_two_half_segments`
  compares one flight with two halves and a fixed reference emitter. Both
  routes passed in the 2026-10-20 verifier run (2 passed, 18 deselected).
- {source}`tests/montecarlo/coherent/test_coherent_emission.py::test_single_segment_coherent_equals_incoherent_self_term`
  is a tolerance-based comparison; its name does not establish absorbed
  pointwise equality. The formation tests separately cover independent
  quadrature, Parseval, geometry breakpoints and subdivision.
- {source}`tests/montecarlo/spectrum/test_xray_dispersion.py::test_dispersive_formation_integral_has_the_derivative_jacobian`
  pins the derivative-Jacobian distinction.

## Historical evidence and adjudication

The original vacuum-only derivation correctly established midpoint pairing.
The 2026-08-19 measurements (40/20/10/5 Å flights: residuals
$7.8\times10^{-6}$, $4.0\times10^{-6}$, $2.9\times10^{-6}$,
$1.1\times10^{-6}$ of peak) describe the subsequent **midpoint escape-phase
approximation**, not a limitation of today's formation integral. Issue #181
replaced that approximation; its recorded two-half residual was
$1.8\times10^{-15}$ of peak. These dated measurements retain their original
scope and are not new benchmark runs.

**2026-10-20 fresh-context verdict: `rederived`.** Units, zero-length and
transparent limits, phase sign, midpoint clock, piece clock offsets and the
full formation expression match the current implementation. Exact subdivision
and self-term comparisons require the conditions above. No factor, sign,
unit or convention discrepancy was found in midpoint pairing. This refresh
does not certify every upstream amplitude or the original paper's equation
numbering. Human sign-off remains with #277.
