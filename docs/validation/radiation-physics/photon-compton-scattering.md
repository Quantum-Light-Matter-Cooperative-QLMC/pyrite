# Photon Compton scattering

## Independent source-first derivation

Validation: `photon-compton-scattering`. The verifier read the ledger,
AST-extracted signatures/docstrings, and sources before writing this section;
implementation bodies had not been inspected.

Sources: Geant4 Physics Reference Manual 11.4,
[standard Compton final-state sampling](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html)
and [Livermore Compton final-state sampling](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/livermore_compton.html).
The data convention is pinned EPDL2025 MF=27 MT504 with lin-lin interpolation.

`sample_compton(element, energy_eV, direction, uniforms)` describes one
conditional collision on an already selected atom. Let the incident energy be
$E>0$, incident unit vector $\hat{\mathbf n}$, $\mu=\cos\theta$,
$\delta=1-\mu$, and $\alpha=E/(m_ec^2)$. Energy and momentum conservation for
an initially stationary free electron give

$$
\epsilon=\frac{E'}{E}=\frac{1}{1+\alpha\delta},\qquad
T=\frac{E\alpha\delta}{1+\alpha\delta},\qquad
\mathbf p_ec=E\hat{\mathbf n}-E'\hat{\mathbf n}'.
$$

The unpolarized Klein–Nishina differential cross section is

$$
\frac{d\sigma}{d\Omega}=\frac{r_e^2}{2}\epsilon^2
\left(\epsilon+\epsilon^{-1}-(1-\mu^2)\right)S(x,Z),\qquad
x=\frac{E}{hc}\sqrt{\frac{\delta}{2}}.
$$

Here $x$ is in inverse angstroms when $hc$ is in eV angstroms. It is the
tabulated EPDL variable, not the wave-vector transfer $4\pi x$.
Since $d\mu/d\epsilon=1/(\alpha\epsilon^2)$, the conditional energy density is

$$
p(\epsilon)\propto(\epsilon^{-1}+\epsilon)\,
\left[1-\frac{\epsilon\delta(2-\delta)}{1+\epsilon^2}\right]S(x,Z),
\qquad \epsilon_0=(1+2\alpha)^{-1}\leq\epsilon\leq1.
$$

Integrating the two proposal terms gives mixture masses
$a_1=\ln(1+2\alpha)$ and $a_2=(1-\epsilon_0^2)/2$.
Choose the logarithmic branch with probability $a_1/(a_1+a_2)$.
An independent uniform $u$ gives either $\epsilon=\exp(-ua_1)$ or
$\epsilon=\sqrt{\epsilon_0^2+(1-\epsilon_0^2)u}$.
Rejection probability is the bracket times $S/S_{\max}$.
The bracket lies in $[1/2,1]$, and a piecewise linear function attains its
maximum at knots or accessible endpoints; no monotonicity assumption is needed.
If $S_{\max}=0$, no conditional event exists.

The azimuth is uniform on $[0,2\pi)$. For any orthonormal transverse pair
$\mathbf a,\mathbf b$, put $\mathbf e_\phi=\mathbf a\cos\phi+\mathbf b\sin\phi$;
then

$$
\hat{\mathbf n}'=(1-\delta)\hat{\mathbf n}
+\sqrt{\delta(2-\delta)}\,\mathbf e_\phi.
$$

Units pass: energies are eV; all proposal/rejection factors are dimensionless.
Limits pass: constant $S$ yields free Klein–Nishina; as $\alpha\to0$ this
becomes Thomson; $\delta=0$ gives zero recoil. Signs pass: $T\geq0$,
$0<E'\leq E$, and the recoil momentum is incident minus outgoing momentum.
This model omits binding, Doppler broadening, polarization, and vacancies;
it does not normalize or replace the independently supplied integrated rate.

## Implementation comparison

Body inspection followed the written derivation. `_compton_proposal` uses exactly
the two mixture masses above. Its logarithmic branch evaluates
$\delta=\operatorname{expm1}(ua_1)/\alpha$; the square-root branch evaluates
$\delta=(1-\epsilon_0^2)(1-u)/[(1+\epsilon)\epsilon\alpha]$.
Both equal $(1-\epsilon)/(\alpha\epsilon)$, but avoid subtracting nearby
numbers at low energy. The angular rejection factor and tabulated $S/S_{\max}$
match term by term. `clipped_factor_table` retains the interpolated accessible
endpoint, so its maximum includes every possible piecewise-linear extremum.

The momentum expression uses the independently equivalent stable decomposition

$$
\mathbf p_ec=(T+E'\delta)\hat{\mathbf n}
-E'\sqrt{\delta(2-\delta)}\,\mathbf e_\phi.
$$

Together with $T=E\alpha\delta/(1+\alpha\delta)$ this avoids cancellation
in the forward recoil. `scatter_direction` constructs the proved orthonormal
rotation using the least aligned Cartesian axis. The arbitrary electron
direction at exactly zero recoil is a convention, not a physical prediction.
The code consumes explicit uniforms only, rejects invalid or exhausted tapes,
and raises on rejection-cap exhaustion; it does not force an event.

## Independent numerical evidence

On 2026-10-19, a fresh-context verifier read the pinned archive directly,
without either production factor accessor or proposal/CDF helpers. Its SHA256
was `d36a50d1add5c6f0b3df80a108e83aa8db8a4822445ca192534f6bc0a55bb166`.
Independent trapezoidal quadrature used 100,000 logarithmic $x$ points plus
the original table knots; density in $x$ included its Jacobian, proportional
to $x\epsilon^2[\epsilon+\epsilon^{-1}-\delta(2-\delta)]S(x,Z)$.
For each row, 1,200 events used incident direction $(2,-3,6)/7$ and the
same explicit local stream, initialized once with seed 943411, shared across
alternating Rayleigh and Compton runs.

| Atom | Energy (eV) | Empirical CDF maximum distance |
|---|---:|---:|
| C | 1 | 0.02609 |
| C | 1,000 | 0.02740 |
| Si | 511,000 | 0.02639 |
| Pb | 100,000,000 | 0.01577 |
| Pb | 100,000,000,000 | 0.02790 |

All distances are below $2.2/\sqrt{1200}=0.06351$. This is a distribution
check against the prescribed model, not an independent validation of EPDL data.
Across these events, relative energy closure error was at most
$2.3\times10^{-16}$, relative momentum-vector closure error at most
$6.5\times10^{-16}$, and direction norm error at most $3.4\times10^{-16}$.
The recoil magnitude was reconstructed independently as
$\sqrt{T(T+2m_ec^2)}$.

An independent 75-digit decimal evaluation checked both proposal branches at
all five energies and $u\in\{0,10^{-20},0.25,0.75,\operatorname{nextafter}(1,0)\}$.
Maximum relative error in $\delta$ was $8.7\times10^{-16}$.
Three incident orientations were also checked at
$\delta\in\{0,10^{-30},10^{-15},1,2\}$; direction norm and transverse
orthogonality errors stayed below $5\times10^{-16}$.
For all sampled atoms the tables start at $x=0$ and extend to
$10^9\,\mathrm{\mathring A}^{-1}$, beyond the maximum accessible
$x\simeq8.07\times10^6\,\mathrm{\mathring A}^{-1}$ in the supported energy
domain. No extrapolation was needed, including at 1 eV and 100 GeV.
A separate archive-wide check covered all 200 MF=27 tables: zero origin,
strictly increasing knots, sufficient upper support, and finite nonnegative
factors all passed.

## Verdict

Units, limiting cases, signs and conventions pass. The independent expression
matches the implementation: **rederived**, with no divergent term found.
Suggested ledger change: set status to `rederived`, replace the pending-verifier
note with this source-first derivation and numerical evidence. Human sign-off
and independent validation of the input atomic data remain separate claims.
