# GOS hard-collision recoil

Validation: `gos-hard-recoil`. Independent verdict: rederived for the
host-side polar sampler; human sign-off remains pending.

## Source and intended quantity

The [Geant4 PENELOPE ionisation reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/ionisation/penelope_ionisation.html)
gives the hard-branch sampling and projectile/secondary polar angles after
equation 133. A close collision draws $W\leq E/2$ from the Møller density.
Its primary polar angle obeys

$$
\cos^2\theta=\frac{E-W}{E}\frac{E+2m}{E-W+2m},\qquad m=m_ec^2.
$$

A close event above the secondary production threshold returns an
outer-shell proxy secondary with kinetic energy $W$ and the cited secondary
polar angle for $Q=W$. It carries no shell vacancy or binding correction.
The core-only diagnostic suppresses this proxy because a core secondary's
kinetic energy needs a subshell binding energy absent from the OOS split.
The eventual transport event must also sample a uniform azimuth and place
the secondary opposite the primary azimuth. Neither rotation nor secondary
transport is implemented by this host-side sampler yet.

For a distant longitudinal event, recoil energy $Q$ is sampled from the
reference's $1/[Q(1+Q/(2m))]$ law between $Q_-$ and $W$, and the projectile
polar angle follows its relativistic momentum triangle. Distant transverse
events retain the original direction. No sample exists when the hard rate is
zero. The computed angle cosine is bounded to the physical interval to absorb
roundoff at endpoints; an independent recoil-angle check remains pending.

The longitudinal recoil law integrates to

$$
\int\frac{dQ}{Q(1+Q/(2m))}
=\ln\frac{Q}{Q+2m}+C.
$$

Hence with $R(Q)=Q/(Q+2m)$ and a uniform variate $u\in[0,1]$,
$R=R(Q_-)^{1-u}R(W)^u$ and $Q=2mR/(1-R)$ sample the law on
$[Q_-,W]$. The longitudinal projectile cosine follows the momentum triangle

$$
\cos\theta=\frac{p_0^2+p_1^2-Q(Q+2m)}{2p_0p_1},
\qquad p_0^2=E(E+2m),\quad p_1^2=(E-W)(E-W+2m).
$$

For a close collision, $Q=W$ in the source's secondary-angle formula,

$$
\cos^2\theta_s=\frac{W^2}{\beta^2W(W+2m)}
\left[1+\frac{W(W+2m)-W^2}{2W(E+m)}\right]^2.
$$

These are polar magnitudes. A transported event also needs a uniform azimuth,
with the secondary azimuth opposite the primary.

## Source-to-code comparison

`sample_hard_collision` selects each positive hard component by its
integrated cross section. For a close bin of width $\Delta$ with linear
endpoint densities $h_0,h_1$, it solves
$u\Delta(h_0+h_1)/2=\Delta[h_0t+(h_1-h_0)t^2/2]$ for $t\in[0,1]$ using the
rationalized quadratic root. The primary cosine matches the source equation.
Setting $Q=W$ in the secondary equation above reduces its bracket to
$(E+2m)/(E+m)$, exactly the code's expression. The longitudinal inverse CDF
and momentum-triangle cosine also agree; transverse cosine is one. The code
returns only polar cosines and an outer-shell proxy secondary energy, so
trajectory rotation, azimuth, vacancy and secondary transport remain outside
this claim's implemented scope.
