# PENELOPE shell hard-event primary recoil

Validation: `penelope-shell-hard-recoil`. Status: unverified. This host-side
sampler does not yet create a segment boundary or a laboratory-frame direction.

## Source and quantity

[PENELOPE-2024](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf)
§3.2.5.1, Eqs. 3.126–3.129, gives a distant longitudinal recoil energy $Q$
with conditional density

$$
g(Q)\mathrel{\propto}\frac{1}{Q[1+Q/(2m_ec^2)]},\qquad Q_-<Q<Q'_k.
$$

The CDF is linear in $L(Q)=\ln[Q/(Q+2m_ec^2)]$. Sampling a uniform $\xi$ gives
$L(Q)=(1-\xi)L(Q_-)+\xi L(Q'_k)$, inverted as
$Q=2m_ec^2/[\exp(-L(Q))-1]$. The lower bound uses the shell's modified
resonance $W'_k$; the upper bound is $Q'_k$. The conduction band uses
$W'_k=Q'_k=W_{cb}$. The [shell GOS moments](penelope-shell-gos-moments.md)
compute these same bounds for the longitudinal channel.

The primary polar cosine uses the momenta before and after the *modified
resonance* loss (Eq. 3.129):

$$
\cos\theta=\frac{E(E+2m_ec^2)+(E-W'_k)(E-W'_k+2m_ec^2)
-Q(Q+2m_ec^2)}
{2\sqrt{E(E+2m_ec^2)(E-W'_k)(E-W'_k+2m_ec^2)}}.
$$

This is PENELOPE's angular approximation for a bound-shell triangle: sampled
$W$ sets the primary final energy and secondary accounting, while $W'_k$
sets its angular distribution. The transverse branch neglects primary
deflection. For a close event, $Q=W$ and the primary cosine follows
Eq. 3.134:

$$
\cos^2\theta=\frac{E-W}{E}\frac{E+2m_ec^2}{E-W+2m_ec^2}.
$$

The azimuth is $2\pi\xi_\phi$. The result is in the incoming flight frame;
the caller must rotate it into world coordinates. The secondary angle is
recorded separately in [secondary emission direction](penelope-shell-secondary-direction.md).

## Limits and checks

Longitudinal $\xi=0$ gives $Q_-$ and $\cos\theta=1$ up to floating-point
roundoff. The close cosine approaches 1 as $W/E\to0$ and is finite over the
bound-shell interval $W\le(E+U_k)/2$. The transverse cosine is exactly 1.
In `tests/montecarlo/collisions/test_shell_partition.py`, every nonzero fixture branch
at 10 keV is checked at four recoil quantiles. Longitudinal samples match
direct quadrature of $g(Q)$ within $10^{-9}$; close and longitudinal cosines
match their source equations, and invalid uniforms raise.
Catalog Si, SiO₂ and MoS₂ at 1, 10 and 100 keV exercise every active branch
at each point and keep the primary cosine and transfer in physical bounds.

This angular model has not been independently validated against a measured
recoil distribution or checked for CPU/GPU reproducibility.

## Independent verification

Fresh-context, code-blind re-run (issue #4, batch 3; 2026-11-11). The
derivation below was written before the implementation body, its helpers,
the anchor test, or the text above were opened. Its only inputs were the
ledger row, the extracted signature and docstring, and PENELOPE-2024
(NEA/MBDAV/R(2024)1): Eqs. 3.67, 3.76–3.82, 3.126–3.129 and 3.134, with
Appendix A Eqs. A.29–A.31 and A.40.

### Source definitions

With $(cp(T))^2=T(T+2m_ec^2)$, the recoil energy satisfies
$Q(Q+2m_ec^2)=(cq)^2$ (A.29–A.30). For a loss $W$, the minimum recoil (A.31) is

$$
Q_-(W)=\sqrt{[cp(E)-cp(E-W)]^2+m_e^2c^4}-m_ec^2 .
$$

For a bound shell with $U_k>0$, Eqs. 3.78 and 3.80 give
$W'_k=W_k$, $Q'_k=U_k$ when $E>3W_k-2U_k$. Otherwise they give
$W'_k=(E+2U_k)/3$ and $Q'_k=U_kE/(3W_k-2U_k)$. The conduction band has
$U_{\rm cb}=0$ and cutoff $Q_{\rm cb}=W_{\rm cb}$ (§3.2.2, Fig. 3.8b), with no
modification. The angular law takes $Q_-=Q_-(W'_k)$, independent of the
sampled loss $W$ of Eq. 3.125 (text after Eqs. 3.82 and 3.126).

### Longitudinal recoil and angle

The density of Eq. 3.126 is an exact derivative:

$$
g(Q)=\frac{1}{Q[1+Q/(2m_ec^2)]}=\frac{2m_ec^2}{Q(Q+2m_ec^2)}
=\frac{\mathrm{d}}{\mathrm{d}Q}\ln\frac{Q}{Q+2m_ec^2}\equiv L'(Q).
$$

The CDF on $(Q_-,Q'_k)$ is therefore $[L(Q)-L(Q_-)]/[L(Q'_k)-L(Q_-)]$. Setting it
to $\xi$ and writing $r=e^{L}$ gives

$$
L(Q)=(1-\xi)L(Q_-)+\xi L(Q'_k),\qquad
Q=\frac{2m_ec^2\,r}{1-r}=\frac{2m_ec^2}{e^{-L(Q)}-1}.
$$

This is algebraically identical to Eqs. 3.127–3.128. With
$Q_S=Q_-/[1+Q_-/(2m_ec^2)]$, one finds
$2m_ec^2/Q+1=(2m_ec^2/Q_S)\{(Q_S/Q'_k)[1+Q'_k/(2m_ec^2)]\}^{\xi}$. The polar
angle follows from A.40 with $W\to W'_k$, which is Eq. 3.129:

$$
\cos\theta=\frac{E(E+2m_ec^2)+(E-W'_k)(E-W'_k+2m_ec^2)-Q(Q+2m_ec^2)}
{2\sqrt{E(E+2m_ec^2)(E-W'_k)(E-W'_k+2m_ec^2)}}.
$$

At $Q=Q_-$, $(cq)^2=(cp-cp')^2$ and $\cos\theta=1$. The cosine decreases
monotonically in $Q$. It stays at or above $-1$ because
$Q'_k\le U_k\le W'_k<Q_+$.

### Transverse, close and azimuth

A transverse event has $\cos\theta=1$ (§3.2.5.1). A close event inserts
$Q=W$ into A.40. The numerator reduces to $2(E-W)(E+2m_ec^2)$, so

$$
\cos^2\theta=\frac{(E-W)(E+2m_ec^2)}{E(E-W+2m_ec^2)},
$$

which is Eq. 3.134. It is at most 1 because the numerator minus the
denominator is $-W(E+2m_ec^2)\le0$. It tends to 1 as $W/E\to0$, with
$\cos\theta\ge0$. The azimuth is $\phi=2\pi\xi_\phi$.

### Comparison with the implementation

The implementation was read only after the derivation above was saved.
`sample_shell_hard_collision` matches term by term:

- `_qmin_ev` is the cancellation-free form of A.31. It sets
  $\Delta p=cp-cp'=W[2(E+m_ec^2)-W]/(cp+cp')$ and
  $Q_-=\Delta p^2/[\sqrt{\Delta p^2+m_e^2c^4}+m_ec^2]$.
- A bound shell ($U_k>0$) uses Eqs. 3.78 and 3.80 exactly. A shell with
  $U_k=0$ uses $W'_k=Q'_k=W_k$.
- The inverse CDF `2*_MC2_EV/np.expm1(-log_ratio)` equals
  $2m_ec^2/(e^{-L}-1)$.
- The longitudinal cosine is Eq. 3.129, evaluated at $W'_k$. Its final
  clip to $[-1,1]$ only guards against roundoff.
- The close cosine is the positive root of Eq. 3.134. A transverse event
  returns $\cos\theta=1$ with no recoil. The azimuth is `2*pi*u_azimuth`.

No factor, sign, unit or convention diverges.

An independent mpmath reference (40 digits) checked every active catalog
channel. It covered Si, SiO₂ and MoS₂ at 1, 10 and 100 keV, with
$\xi_Q\in\{0,10^{-6},0.1,0.25,0.5,0.75,0.9,1-10^{-6}\}$, and compared the
sampled $Q$ with a root of the directly integrated CDF of $g(Q)$. The
reference uses $m_ec^2=510998.95069$ eV, the CODATA 2022 value that
`scipy.constants` supplies. Using the rounded $510998.95$ eV leaves an
apparent $3.2\times10^{-10}$ relative offset in $Q_-$, which is caused only by
the constant. Results:

| Quantity | Worst deviation |
|---|---|
| Longitudinal $Q$ (relative) | $8.2\times10^{-13}$ |
| Longitudinal $\cos\theta$ vs Eq. 3.129 | $3.5\times10^{-14}$ |
| Close $\cos\theta$ vs Eq. 3.134 | $3.6\times10^{-14}$ |
| $\xi_Q=0$: $\lvert\cos\theta-1\rvert$ | $<10^{-12}$ |

Modified-resonance channels ($E\le3W_k-2U_k$) were exercised as SiO₂ Si-L1 at
1 keV, SiO₂ Si-K at 10 keV, and MoS₂ M1, M2 and M3 at 1 keV. Transverse
events returned $\cos\theta=1$ with no recoil, and the azimuth equalled
$2\pi\xi_\phi$. The anchor `tests/montecarlo/collisions/test_shell_partition.py`
passed (100 tests).

Verdict: `rederived`. The derivation was code-blind; the implementation
matches PENELOPE-2024 Eqs. 3.126–3.129 and 3.134. This section does not
validate the secondary direction (its own ledger row), segment scheduling,
measured recoil distributions, or CPU/GPU reproducibility.
