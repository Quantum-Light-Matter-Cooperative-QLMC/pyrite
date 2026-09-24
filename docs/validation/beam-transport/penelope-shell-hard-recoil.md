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
In `tests/montecarlo/test_shell_partition.py`, every nonzero fixture branch
at 10 keV is checked at four recoil quantiles. Longitudinal samples match
direct quadrature of $g(Q)$ within $10^{-9}$; close and longitudinal cosines
match their source equations, and invalid uniforms raise.
Catalog Si, SiO₂ and MoS₂ at 1, 10 and 100 keV exercise every active branch
at each point and keep the primary cosine and transfer in physical bounds.

This angular model has not been independently validated against a measured
recoil distribution or checked for CPU/GPU reproducibility.
