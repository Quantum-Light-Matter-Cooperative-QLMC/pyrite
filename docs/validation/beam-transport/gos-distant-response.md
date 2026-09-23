# GOS distant response

Validation: `gos-distant-response`. Independent verdict: rederived;
human sign-off remains pending.

## Source and intended quantity

The [Geant4 PENELOPE ionisation reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html)
gives distant longitudinal and transverse oscillator cross sections in
equations 127–128. The model uses SBETHE optical intervals as oscillator
strengths $f_i$ at resonance energies $W_i$. It assumes a free relativistic
projectile and uses no material finite-$q$ response.

With $m=m_ec^2$ in eV, $r_e$ in cm, and projectile $\beta=v/c$, the common
prefactor is $A=2\pi r_e^2m/\beta^2$, in eV cm$^2$. The minimum recoil is

$$
Q_-=\sqrt{(p_0-p_1)^2+m^2}-m,\quad
p_0=\sqrt{E(E+2m)},\quad
p_1=\sqrt{(E-W)(E-W+2m)}.
$$

The implementation evaluates $p_0-p_1$ by the difference of squared momenta
divided by $p_0+p_1$ and evaluates the outer square-root difference by a
conjugate, avoiding cancellation for small $W$. The longitudinal and
transverse terms are $A f_i/W_i$ times the corresponding dimensionless
logarithms in the cited equations. The raw transverse term omits the Fano
density correction; one common first-moment scale is applied later in the
partition. As $W\to0$, $Q_-\propto W^2$; at $W=E$ the longitudinal logarithm
vanishes. Distant transverse events have no projectile angular recoil in
this model.

Independently, let $p(E)=\sqrt{E(E+2m)}$ and $p_1=p(E-W)$. Since
$p_0^2-p_1^2=W(2E-W+2m)$, the two rationalizations give

$$
\Delta p=\frac{W(2E-W+2m)}{p_0+p_1},\qquad
Q_- =\frac{(\Delta p)^2}{\sqrt{m^2+(\Delta p)^2}+m}.
$$

At small $W$, $\Delta p=W/\beta+O(W^2)$ and
$Q_-=W^2/(2m\beta^2)+O(W^3)$. The source's resonance cross sections are

$$
\sigma_{L,i}=\frac{Af_i}{W_i}
\ln\!\left[\frac{W_i}{Q_-}\frac{Q_-+2m}{W_i+2m}\right],
\qquad
\sigma_{T,i}=\frac{Af_i}{W_i}
\left[\ln\frac{1}{1-\beta^2}-\beta^2-\delta_F\right]
$$

for $W_i\leq E$. Here $A$ has units eV cm$^2$, so each term is
cm$^2$ per molecule. The longitudinal logarithm is nonnegative because
$Q_-\leq W_i$, with equality at $W_i=E$. Omitting $\delta_F$ leaves a
nonnegative transverse bracket: $-\ln(1-x)-x\geq0$ for $x=\beta^2\in[0,1)$.

## Source-to-code comparison

`_qmin_ev` implements both rationalizations exactly. `build_gos_partition`
uses $2\pi r_e^2m/\beta^2$ with $r_e$ in cm and $m$ in eV, multiplies by
$f_i/W_i$, and uses the two source brackets. Its `log1p` argument is
$E(E+2m)/m^2=\gamma^2-1$, hence the result is $\ln(\gamma^2)$.
The code sets $\delta_F=0$ by omission, as declared. The terms agree
symbolically, with the finite-momentum response and density correction
remaining model omissions.
