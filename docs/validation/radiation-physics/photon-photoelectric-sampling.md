# Photoelectric subshell and final-state sampling

Validation: `photon-photoelectric-sampling`.

## Independent derivation before implementation inspection

This derivation was written from the ledger, function signatures/docstrings and
the cited source before reading implementation bodies. The intended event is
conditional on an already selected absorbing atom; elemental number-density
weights do not enter a second time. Cross sections have area units, binding and
kinetic energies have eV units, and all angular quantities below are dimensionless.

The independently verified [EPDL differential-data claim](epdl-photon-differential-data.md)
identifies EPDL2025 MF=23 MT534–599 subshell partial photoionization cross sections.
For the accessible set $\mathcal A=\{i:E\ge B_i,\sigma_i(E)>0\}$, exclusivity of
the primary hole gives

$$
P(i\mid E,Z,\mathrm{photoelectric})=
\frac{\sigma_i(E)}{\sum_{j\in\mathcal A}\sigma_j(E)},\qquad
T=E-B_i\ge0.
$$

The probabilities sum to one when the denominator is positive. A zero denominator
does not define an event. ENDF-102 File 28 shell designators connect this hole to
the corresponding EADL vacancy (see [ENDF-102 File 28, §§28.1–28.3](https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf)). Atomic identity and shell bindings must agree
between the supplied relaxation data and the sampled photoionization data.

The [Geant4 11.4 photoelectric final-state manual](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/photoelectric/photoelec.html),
“Theta Distribution of the Photoelectron”, states the Sauter–Gavrila K-shell
approximation. Write $\mu=\cos\theta$, $\gamma=1+T/(m_ec^2)$,
$\beta=\sqrt{1-\gamma^{-2}}$ and $D=\gamma(\gamma-1)(\gamma-2)/2$.
Its normalized polar density is

$$
p_\mu(\mu)=\frac{f(\mu)}{N},\quad
f(\mu)=\frac{1-\mu^2}{(1-\beta\mu)^4}
\left[1+D(1-\beta\mu)\right],\quad
N=\int_{-1}^{1}f(v)\,dv.
$$

For finite positive $T$, $0<\beta<1$ makes the integral finite. For
$1<\gamma<2$, put $x=\gamma-1\in(0,1)$: then
$|D|=(x-x^3)/2\le1/(3\sqrt3)$, and
$1+D(1-\beta\mu)>1-2/(3\sqrt3)>0$.
For $\gamma\ge2$ the correction is nonnegative. Thus $N>0$ and the
normalized density exists. The azimuth is uniform on $[0,2\pi)$.

For an exact rejection sampler, set $\delta=1-\mu\in[0,2]$,
$A=\beta^{-1}-1>0$ and $C=\beta D$. Dropping only constants independent
of the variate gives

$$
p_\delta(\delta)\propto
\frac{\delta(2-\delta)}{(A+\delta)^4}
\left[1+C(A+\delta)\right].
$$

Now set $w=\delta/(A+\delta)$, $W=2/(A+2)$ and
$B=1+CA=1+D(1-\beta)$. Since
$\delta=Aw/(1-w)$ and $d\delta/dw=A/(1-w)^2$,

$$
p_w(w)\propto w(2-\delta)(1-w+CA),\qquad 0\le w\le W.
$$

Use the normalized proposal $q(w)=2w/W^2$, sampled by $w=W\sqrt u$,
and accept with

$$
r(w)=\frac{2-\delta}{2}\left(1-\frac{w}{B}\right).
$$

Indeed $q(w)r(w)$ is a positive constant times $p_w(w)$. Positivity of
the original correction implies $B-w>0$ on this interval; also $B>0$,
$w\ge0$ and $0\le\delta\le2$, so $0\le r(w)\le1$.
This proves the envelope even below $\gamma=2$, where $C$ is negative.
It also remains valid at large $\gamma$, where $B\sim\gamma/4$.

Cancellation-free evaluation follows from $1-\beta^2=\gamma^{-2}$:

$$
A=\frac{1}{\beta\gamma^2(1+\beta)},\qquad
CA=\frac{(\gamma-1)(\gamma-2)}{2\gamma(1+\beta)}.
$$

Near rest, evaluate $\beta=\sqrt{t(t+2)}/(1+t)$ with
$t=T/(m_ec^2)$ to avoid subtracting nearly equal numbers. As $T\to0^+$,
$p_\mu\to3(1-\mu^2)/4$; its mean cosine is zero. At exactly $T=0$ the
direction of a zero-momentum electron is arbitrary: this API chooses the
incident direction and consumes no angular draws. At high energy the natural
forward angular scale is $\delta\sim A\sim1/(2\gamma^2)$; retaining
$\delta$ avoids forming a rounded polar cosine too early.

For a unit incident vector $\hat{\mathbf n}$ choose orthonormal transverse
vectors $\hat{\mathbf e}_1,\hat{\mathbf e}_2$. With
$\hat{\mathbf e}_\phi=\cos\phi\,\hat{\mathbf e}_1+\sin\phi\,\hat{\mathbf e}_2$,

$$
\hat{\mathbf n}_e=(1-\delta)\hat{\mathbf n}
+\sqrt{\delta(2-\delta)}\,\hat{\mathbf e}_\phi.
$$

The norm is one and its dot product with the incident ray is $1-\delta$.
The azimuthal basis convention therefore cannot affect the unpolarized law.

The separately rederived `eadl-sampled-relaxation` claim supplies a signed
single-vacancy budget. For transition defect
$\delta_t=B_{\rm parent}-\sum B_{\rm daughters}-E_{\rm emitted}$,
telescoping through the vacancy tree gives

$$
B_i=\sum E_\gamma+\sum T_{\rm Auger}+\sum B_{\rm terminal}
+\sum\delta_t,
\qquad
E=T+\sum E_\gamma+\sum T_{\rm Auger}+\sum B_{\rm terminal}
+\sum\delta_t.
$$

The signed defect must not be clipped or counted twice. The infinitely massive
atom takes momentum imbalance with no recoil-energy subtraction. These equations
do not extend the K-shell, zero-order-in-$\alpha Z$ angular approximation to a
full multipole or polarized calculation: applying it to all shells is an explicit
model approximation. Natural linewidths and multiple-vacancy angular corrections
are excluded. Emission transport and production integration are outside scope.

## Implementation comparison

The implementation was inspected after the derivation above was on disk.
`_sauter_delta` uses exactly the derived proposal, inverse transform and acceptance
factor. Its `reduced` is $t$, `a` is $A$, and `c` is $C$. In particular,
`1 + c * a` retains the negative correction for $1<\gamma<2$; no absolute value
or clipping changes that factor. Its stable $\beta$ and $A$ expressions agree
with the derivation. Products remain finite throughout the supported EPDL photon
energy range, including the tested upper energy of $10^{11}$ eV. The intermediate
product $CA$ is safe there; the separately simplified expression above would
avoid unnecessary large intermediates if this domain were ever extended.

`sample_photoelectric` forms a categorical CDF from positive accessible rates,
checks atomic number, all required designators and exact EPDL/EADL binding
agreement, computes $T=E-B_i$, then samples the derived direction and delegates
the selected vacancy to the separately rederived
[sampled EADL relaxation claim](eadl-sampled-relaxation.md).
The threshold branch preserves the incident ray and skips polar and azimuthal
draws. `scatter_direction` uses a least-aligned Cartesian axis to construct the
orthonormal frame and implements the derived rotation. Uniform draws are explicit,
finite and in $[0,1)$; zero total rate is rejected before selecting a shell.

Numerical checks used the checksum-pinned differential artifact regenerated from
the upstream tape (SHA-256
`d36a50d1add5c6f0b3df80a108e83aa8db8a4822445ca192534f6bc0a55bb166`)
and checksum-verified EADL2025, selected through
`PYRITE_HOME=/tmp/pyrite-341-data` and the project runner.

- With seed 302, 18,000 polar samples at each of $T=10^{-6},10^3,10^5,
  511000,10^7$ eV agreed with independently integrated source-law mean cosines
  within 0.52 sample standard errors. Corresponding reference means were
  $1.5826868\times10^{-6}$, $0.05001572$, $0.46749219$, $0.79684355$ and
  $0.98886156$. At $10^{11}$ eV, 18,000 samples remained finite and forward;
  a mean-cosine tolerance is not used there because rare wide-angle tails make
  that finite-sample statistic poorly conditioned.
- With seed 419, 200 complete events each for Si at 2 keV, Cu at 10 keV and
  Au at 100 keV closed the signed energy budget within
  $1.46\times10^{-11}$ eV. Sampled defects included negative values down to
  $-265.99$ eV; no nonnegative-deposit interpretation was imposed.
- Exact first-shell threshold events for all three elements, with relaxation
  suppressed by a high cutoff, returned zero electron energy, the incident
  direction and exactly one consumed draw (shell selection).
- Mismatching atomic number and a 1 eV binding perturbation were rejected before
  consuming uniforms. A below-all-shells event refused its zero accessible rate.
- A 2,000-point stratified-uniform Cu shell check reproduced the normalized
  partial-cross-section categorical weights to within one event per shell.

Verdict: **rederived**. Units, limiting cases, positivity, envelope and
conventions pass; no divergent term or convention was found. Suggested ledger
change: status `filtered` → `rederived`; replace the pending-verification note
with this independent derivation and numerical evidence. Human sign-off is not
asserted. The task owner performs the shared documentation build, documentation
tests and rendered-math inspection after integrating this write-up.
