# PENELOPE shell secondary emission direction

Validation: `penelope-shell-secondary-direction`. Status: unverified. The
host-side collision sampler returns secondary angles only when its secondary
kinetic energy meets the production threshold. It does not yet enqueue a
secondary track or rotate either direction into laboratory coordinates.

## Source and construction

[PENELOPE-2024](https://www.oecd-nea.org/upload/docs/application/pdf/2025-07/nea_mbdav_r_2024_1_penelope-2024_2025-07-10_15-48-34_125.pdf)
§3.2.5.4, Eqs. 3.137–3.138, puts the emitted electron along momentum
transfer $\mathbf q$ under the initially stationary target-electron
approximation. Set $M=m_ec^2$ and $p(T)^2=T(T+2M)$ in energy units. For a
longitudinal event, its polar cosine in the incoming-primary frame is

$$
\cos\theta_s=\frac{p(E)^2+p(Q)^2-p(E-W'_k)^2}
{2p(E)p(Q)}.
$$

This is Eq. 3.137 rearranged using $\mathbf q=\mathbf p_0-\mathbf p_1$.
The angular transfer is the modified resonance $W'_k$ for a broadened bound
shell and the resonance $W_{cb}$ for the conduction band, because the
longitudinal $Q$ was sampled with that value. The sampled loss $W$ still
sets the emitted kinetic energy: $W-U_k$ for a substituted inner shell and
$W$ for an outer shell. For a close event, $Q=W$ and Eq. 3.138 reduces to

$$
\cos\theta_s=\sqrt{\frac{W}{E}\frac{E+2M}{W+2M}}.
$$

The secondary azimuth is $(\phi+\pi)\bmod 2\pi$. The primary's azimuth
$\phi$ is uniform. A distant transverse event has no modeled primary
deflection or sampled $Q$. Its secondary cosine is fixed to $0.5$, following
the [Geant4 Penelope ionisation implementation](https://apc.u-paris.fr/~franco/g4doxy/html/G4PenelopeIonisationModel_8cc-source.html#l00740);
this is a model convention rather than a consequence of Eq. 3.137. The
same opposite-azimuth convention supplies its otherwise arbitrary plane.

## Limits and checks

The close cosine is in $[0,1]$ and approaches zero as $W/E\to0$. At the
minimum longitudinal recoil $Q_-$, the transfer is longitudinal and the
cosine approaches one. All returned cosines are clipped only for roundoff.
No direction is returned when the secondary is not emitted.

## Laboratory-frame handoff

`shell_collision_world_directions` rotates the primary and secondary with the
same orthonormal frame about the incoming unit direction. It reuses the
transport core's `_rotate_direction_scalar` convention. If the frame vectors
are $\mathbf u,\mathbf v,\mathbf d$ with $\mathbf d$ the incoming direction,
each returned vector is

$$
\mathbf d'=\cos\theta\,\mathbf d+\sin\theta
(\cos\phi\,\mathbf u+\sin\phi\,\mathbf v).
$$

Using $\phi_s=\phi+\pi$ in this *same* frame makes the two transverse
components antiparallel. The result is a unit vector with
$\mathbf d'\cdot\mathbf d=\cos\theta$; a forward primary retains the flight
direction, and a suppressed secondary has no world direction. The helper
rejects nonunit or nonfinite input directions. The two directions are not yet
consumed by a transport scheduler, so no flight segment or secondary track is
created here.

`tests/montecarlo/test_shell_partition.py` checks every nonzero fixture
branch at three recoil quantiles, the momentum-triangle expression, close
limit, opposite azimuth, transverse convention, and suppression above the
production threshold. The shell loss sampler's distant-triangle source
conflict remains in `penelope-shell-hard-loss-sampling`; secondary angular
agreement does not resolve it. Independent angle-distribution validation,
transport scheduling, and CPU/GPU comparison remain open. The shared-frame
rotation is checked for on-axis and tilted incoming flights, including unit
norms, polar cosines, opposite transverse components, and a suppressed
secondary.
