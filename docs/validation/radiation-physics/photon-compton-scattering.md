# Photon Compton scattering

Validation: `photon-compton-scattering`. The model samples the angle from
Klein–Nishina times EPDL S(x,Z) and, since #1231, Doppler-broadens the
scattered energy with the impulse approximation (PENELOPE-2024 §2.3); see the
[ledger row](../ledger-transport-background.md#photon-compton-scattering).
Two fresh-context verifications cover it: the 2026-11-20 impulse-approximation
energy sampler (snapshot `3c52ed602`) and the 2026-10-19 angular sampler. The
2026-10-19 free-electron energy and recoil sections are kept as history; the
angular law, proposal and azimuth remain as derived there.

## Impulse-approximation energy sampler (2026-11-20)

Verdict: **rederived** (impulse-approximation energy, shell selection, F
rejection, root selection, binding cutoff, energy balance, recoil direction
and the CUDA port). Units, limits and signs/conventions pass; no divergent
term was found. Sources: PENELOPE-2024 (NEA/MBDAV/R(2024)1) §2.3,
Eqs. 2.30–2.31, 2.38–2.40, 2.43, 2.54–2.56, 2.70–2.72 and the §2.3.2
algorithm; J_i(0) from Biggs, Mendelsohn & Mann, ADNDT 16, 201 (1975) via
xraylib 4.3.0.

### Source transcription (checked against rendered PDF pages 57–66)

Notation: $\mu=\cos\theta$, $\delta=1-\mu$, $\kappa=E/(m_ec^2)$, $\tau=E'/E$, and $\tau_C=E_C/E=1/(1+\kappa\delta)$.

- Eq. 2.30/2.31: $p_z=\dfrac{EE'\delta-m_ec^2(E-E')}{c^2q}$ and $\dfrac{p_z}{m_ec}=\dfrac{E(E'-E_C)}{E_C\,cq}$, with $cq=\sqrt{E^2+E'^2-2EE'\mu}$.
- Eq. 2.38: $F=1+\dfrac{cq_C}{E}\Big[1+\dfrac{E_C(E_C-E\mu)}{(cq_C)^2}\Big]\dfrac{p_z}{m_ec}$. It is used only for $|p_z|<0.2m_ec$, clamped beyond that, and $F\ge0$.
- Eq. 2.40: $d^2\sigma/dE'd\Omega\propto(\ldots)_\theta\,F(p_z)\sum_if_iJ_i(p_z)\Theta(E-E'-U_i)\,dp_z/dE'$.
- Eq. 2.43: $p_{i,\max}=\dfrac{E(E-U_i)\delta-m_ec^2U_i}{c\sqrt{2E(E-U_i)\delta+U_i^2}}$.
- Eq. 2.54 with $n=2$: $J_i^A=J_{i0}\,d_2(d_1+d_2J_{i0}|p_z|)\,e^{d_1^2-(d_1+d_2J_{i0}|p_z|)^2}$, where $d_1=2^{-1/2}$ and $d_2=\sqrt2$. (The general form is $J_{i0}\tfrac{nd_2}{2}(\cdot)^{n-1}\exp[d_1^n-(\cdot)^n]$.)
- Eqs. 2.55/2.56: the CDF and its inverse are given branchwise about $1/2$.
- Eq. 2.70: $E'=E\dfrac{\tau_C}{1-t\tau_C^2}\big[(1-t\tau_C\mu)+\operatorname{sign}(p_z)\sqrt{(1-t\tau_C\mu)^2-(1-t\tau_C^2)(1-t)}\big]$, with $t=(p_z/m_ec)^2$. The τ in that equation is Eq. 2.59's $E_C/E$.
- Algorithm (p. 66): choose shell $i$ with weight $f_i\Theta(E-U_i)n_i^A(p_{i,\max})$. Sample $p_z$ by Eq. 2.56, and reject $p_z<-m_ec$ by redrawing both shell and $p_z$. Accept if $\xi F_{\max}<F(p_z)$, where $F_{\max}$ is at $\pm0.2m_ec$. The electron gets $E_e=E-E'-U_i$ and moves along $\mathbf q=\hbar\mathbf k-\hbar\mathbf k'$ (Eq. 2.72).

### Independent source-first derivation

This derivation was written from the ledger row, the AST-extracted docstrings and signatures, and the PENELOPE text, before the function bodies were read.

**Conditional law in $p_z$.** For a fixed θ, the prefactor of Eq. 2.40 is constant. Changing variables from $E'$ to $p_z$ cancels $dp_z/dE'$, so

$$
p(i,p_z\mid\theta)\propto F(p_z)\,f_i\,J_i(p_z)\,\Theta(E-E'(p_z)-U_i).
$$

From Eq. 2.31, $g(\tau)=p_z/(m_ec)=(\tau-\tau_C)/[\tau_C\sqrt{(1-\tau)^2+2\tau\delta}]$. Its derivative has numerator $1-\tau_C\mu+\tau(\tau_C-\mu)$. That numerator is linear in τ, equals $1-\tau_C\mu>0$ at τ = 0 and $(1+\tau_C)\delta\ge0$ at τ = 1. So $g$ is strictly increasing on $[0,1]$ for δ > 0, and the following hold:

- $E'<E-U_i\iff p_z<p_{i,\max}$.
- $E'=0$ maps to $p_z=-m_ec$, because $cq=E$ there and Eq. 2.30 gives $-m_ec^2E/(cE)$.
- The inverse map is unique on $[0,\tau_{\max}]$, with $\tau_{\max}=1-U_i/E$.

**Binding cutoff in reduced form.** Put $f=U_i/E$ and divide Eq. 2.43 by $m_ec$:

$$
\frac{p_{i,\max}}{m_ec}=\frac{\kappa\delta(1-f)-f}{\sqrt{f^2+2(1-f)\delta}}.
$$

It converts to atomic units as $p_{\rm au}=(p/m_ec)/\alpha_{\rm fs}$, and $-m_ec=-1/\alpha_{\rm fs}$ a.u.

**Shell selection and truncation.** Rejecting $p_z<-m_ec$ and redrawing both shell and $p_z$ makes the accepted pair follow $f_iJ_i$ on $(-m_ec,p_{i,\max}]$. That is the same law as choosing shell $i$ with weight $w_i=f_i[n_i(p_{i,\max})-n_i(-m_ec)]$ and inverse-sampling $p_z$ from the truncated CDF, $n_i^{-1}(n_i(-m_ec)+\xi\,w_i/f_i)$. The shell is accessible only if $E>U_i$, and there is no final state when $\sum w_i=0$. At θ = 0, $p_{i,\max}<0$ and $q\to0$, so this is the degenerate case.

**Profile.** For $n=2$, substitute $u=d_1+d_2J_0|p|$. Then $\int_0^\infty J^A\,dp=\int_{d_1}^\infty u\,e^{d_1^2-u^2}du=1/2$. So the profile is normalized to one electron and $J^A(0)=J_0d_2d_1=J_0$. The CDF is $\tfrac12e^{d_1^2-(d_1-d_2J_0p)^2}$ for $p<0$ and $1-\tfrac12e^{d_1^2-(d_1+d_2J_0p)^2}$ for $p\ge0$.

Inverting it for $A<\tfrac12$ gives $d_1-d_2J_0p=+\sqrt{d_1^2-\ln2A}$, which is the positive branch because $p<0$. So $p=(d_1-\sqrt{d_1^2-\ln 2A})/(d_2J_0)$, and the case $A>\tfrac12$ is symmetric. This reproduces Eqs. 2.55 and 2.56.

**F rejection.** With $cq_C/E=\sqrt{(1-\tau_C)^2+2\tau_C\delta}$ and $1-\tau_C=\kappa\delta\tau_C$, the slope is

$$
s=\frac{cq_C}{E}+\frac{\tau_C(\tau_C-\mu)}{cq_C/E},\qquad \tau_C-\mu=\delta(1-\kappa\tau_C).
$$

$F$ is linear in the clamped $p_z$, so $F_{\max}=1+0.2|s|$. The absolute value matters. At small angles the factor $E^2+2E_C^2-3EE_C\mu\approx E^2\delta(3-\kappa)$, so $s<0$ when κ > 3, i.e. above about 1.5 MeV. PENELOPE's wording "at $+0.2$ or $-0.2$" allows for this.

**Root.** Squaring Eq. 2.31 gives $A\tau^2-2B\tau+C=0$, with

$$
A=1-t\tau_C^2,\qquad B=\tau_C(1-t\tau_C\mu),\qquad C=\tau_C^2(1-t).
$$

The roots are $(B\pm\sqrt{B^2-AC})/A$, which is Eq. 2.70. Because $g$ is monotone, exactly one root in $[0,\tau_{\max}]$ reproduces the signed $p_z$; the other root solves $-p_z$.

**Final state.** The energies are $E'=\tau E$ and $T=E-E'-U_i\ge0$, because $\tau\le\tau_{\max}$. The electron direction is along $E\hat{\mathbf n}-E'\hat{\mathbf n}'$. With $\hat{\mathbf n}'=(1-\delta)\hat{\mathbf n}+\sqrt{\delta(2-\delta)}\,\mathbf e_\phi$, this is $(E-E'(1-\delta))\hat{\mathbf n}-E'\sqrt{\delta(2-\delta)}\,\mathbf e_\phi$. Energy closes as $E=E'+T+U_i$, and the atom takes the momentum imbalance.

**Free-electron limit.** As $J_0\to\infty$, $n_i$ becomes a step at 0, so $p_z\to0$ and $\tau\to\tau_C$, giving $E'=E_C$. As $U_i\to0$, $p_{i,\max}>0$ for every δ > 0, and every shell has weight $f_i$. The result is free Compton with $T=E-E_C$. With constant S, the angular part reduces to Klein–Nishina.

### Implementation comparison (by reading)

All paths below are in `src/pyrite/montecarlo/transport/`.

| Derived term | CPU (`photon_scattering.py`) | CUDA (`_photon_scatter_cuda_kernel.py`) |
|---|---|---|
| $d_1,d_2$ | l.31 | l.73, 80 |
| CDF (Eq. 2.55) | `_profile_cdf` l.124–128 | `compton_cdf` l.72–77 |
| inverse (Eq. 2.56) | `_profile_inverse` l.137–144 | `compton_inverse` l.79–88 |
| $g(\tau)$ (Eq. 2.31) | l.155–156 | l.90–93 |
| $A,B,C$, stable roots, signed-root choice (Eq. 2.70) | l.171–183 | l.96–109 |
| $p_{i,\max}$ (Eq. 2.43; $(\kappa\delta\tau_C-f)/\tau_C=\kappa\delta(1-f)-f$), a.u. via `/ALPHA_FS` | l.204–206 | l.121–123 |
| truncated weight $f_i[n(p_{\max})-n(-m_ec)]$, gated on $f<1$ and $p_{\max}>-m_ec$ | l.208–212 | l.125–127 |
| $cq_C/E$, slope $s$, $F_{\max}=1+0.2\lvert s\rvert$ | l.245–247 | l.140–142 |
| shell draw (strict inverse CDF) | `select_interval` (`photon_sampling.py` l.58–61) | l.144–146 (same threshold; first index with cumulative > threshold) |
| truncated $p_z$, clip, a.u.→$m_ec$ (`*ALPHA_FS`) | l.251–253 | l.148–150 |
| clamped $F\ge0$, acceptance `ξ·F_max < F` | l.254–255 | l.151–154 |
| $\tau_{\max}=1-U/E$, $E'=E\tau$ | l.256–257 | l.155–156 |
| draw order: angle (3), support-rejection loop, azimuth, Doppler trials (3) | l.302–312 | l.204–261 |
| $T=\max(E-E'-U,0)$; electron $\parallel(E-E'(1-\delta))\hat n-E'\sqrt{\delta(2-\delta)}\,\mathbf e_\phi$ | l.313–318 | l.266–273 |

Every row matches the derivation, term for term and in the same evaluation order. `scatter_direction` returns the same $\mathbf e_\phi$ used in $\hat{\mathbf n}'$ (`photon_sampling.py` l.79–81, CUDA l.58–59), so the recoil sign is correct. The CUDA host (`photon_scattering_cuda.py` l.38–53, 108–127) passes the arrays in kernel-signature order: $f_i$, $U_i$, $J_{i0}$, designators, `fine = ALPHA_FS`. The shell order is the same `compton_shells` cascade order, and n ≤ 64 is enforced on the host. Exhausted draws give status 1 and the rejection budget gives status 2, mirroring the CPU exceptions. GPU execution was not run; the ledger already marks it as GPU-gated.

Constants (`materials/crystal.py` l.47–50): $m_ec^2=510998.95$ eV, $\alpha=1/137.035999$, $hc=12398.4198$ eV·Å (CODATA).

### Numerical evidence (independent of the implementation's reference helpers)

The probe script was a disposable verifier artifact and is not committed.

**Conditional E′ law.** The reference density was written directly in E′ space from Eq. 2.40, using:

- $p_z$ from Eq. 2.30;
- a numerical $dp_z/dE'$;
- $J^A$ (Eq. 2.54) with my own unit conversion;
- F with clamp (Eq. 2.38);
- Θ binding cutoffs.

It was integrated by trapezoid on 400k points plus points bracketing each edge. The sample was 4000 draws from `_doppler_energy` at fixed θ. The KS critical value at the 1% level is 0.0258.

| case | KS D |
|---|---:|
| Cu 60 keV, 90° | 0.0169 |
| W 511 keV, 170° | 0.0143 |
| C 2 keV, 120° | 0.0141 |
| Pb 3 MeV, 30° (exercises $s<0$) | 0.0118 |

**Free-electron limit.** With $J_0\times10^7$ and $U\times10^{-9}$ (Cu, 60 keV, 90°), $\max|E'-E_C|/E_C=2.4\times10^{-8}$.

**Root inversion.** Over 20,000 random cases with E from $10^2$ to $10^8$ eV, μ from −1 to 1 and τ from $10^{-6}$ to 1, $p_z$ came from Eq. 2.30 and τ from `_doppler_ratio`. The worst relative error in τ was $1.0\times10^{-10}$, at extreme small τ; this is benign.

**Data.** I checked the table against xraylib 4.3.0 (oracle group):

- 1612 rows; all 1592 rows marked `xraylib` reproduce `ComptonProfile_Partial(Z, shell, 0)` within the 6-significant-figure rounding.
- 20 rows are `partner:`/`neighbor:` substitutions, which are documented.
- The normalization is one electron per shell: $\sum_i f_i^{\rm EADL}J_i(0)$ equals the xraylib total $J(0)$ for Z = 1, 2, 6, 14, 29, 47, 74, 79, 82, 92, to within table rounding (for example H 0.849 = 8/(3π), He 1.07, Cu 5.923 vs 5.93).
- The file SHA-256 equals the `COMPTON_PROFILE_SHA256` pin, `31671e65…5e81`.
- The licence claim holds. xraylib is BSD-3-Clause, `THIRD-PARTY-NOTICES.md` reproduces the clause text and states the modifications (rounding, partner assignment, Bk 6d from Cm), and xraylib is not redistributed.

**Tests.** `pyrite-dev test tests/montecarlo/transport/test_photon_compton_doppler.py`: 10 passed in 10.6 s.

### Observations (non-blocking; none is a discrepancy)

1. **Source of J_i(0).** PENELOPE-2024 itself uses Desclaux MCDF $J_i(0)$ from `pdatconf.p14` (p. 62); Biggs values were used by earlier PENELOPE versions. The code uses Biggs, and the ledger row and docstrings say so explicitly. The ledger Source now names the deviation (applied after verification).
2. **Hybrid model.** The angular marginal is the EPDL $S_{WH}(x,Z)$ rather than the impulse-approximation $S(E,\theta)$ of Eq. 2.47/2.69. Angles with zero impulse-approximation support are rejected. This is a declared assumption, not something the source prescribes.
3. **No 5 MeV switch.** PENELOPE's E > 5 MeV switch to $E'=E_C$ is omitted, as declared. The numerics hold at 3 MeV, and the ledger records higher-energy checks.
4. **Fallback in `_doppler_ratio`.** At l.176 the fallback `best = min(tau_c, tau_max)` applies only if both roots are non-finite, which requires q = 0 and A = 0. The sampled path cannot reach that degenerate case; the code now says so in a comment (applied after verification).
5. **Ledger Equation field.** It writes $|p_z|\le0.2m_ec$ for F. The code clamps $p_z$, as the source prescribes. The field now states $F_{\max}=1+0.2|s|$ (applied after verification).

## Angular sampler and free-electron model (2026-10-19)

### Independent source-first derivation

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

### Implementation comparison

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

### Independent numerical evidence

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

### Verdict

Units, limiting cases, signs and conventions pass. The independent expression
matches the implementation: **rederived**, with no divergent term found.
Suggested ledger change: set status to `rederived`, replace the pending-verifier
note with this source-first derivation and numerical evidence. Human sign-off
and independent validation of the input atomic data remain separate claims.
